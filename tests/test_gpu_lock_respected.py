"""A whole-card job must not be drawn over, and the user must be told why.

The app and the bot must consult gpu_lock before submitting ComfyUI jobs: on
24 GB a render submitted through a live training run is not a queue -- the
diffusion model loads on top of the trainer, both thrash, and the render usually
dies on an OOM after slowing the training for minutes.

Refusing is only half of it. Every renderer returns None on failure and the
tool boundary used to say "the ComfyUI image server encountered an error… tell
the user the image could not be generated", which invites an immediate retry --
for the hours the training run lasts. So the holder is asked FIRST, ahead of the
recorded sub-reason, at each place a failure is turned into words.

Run: venv/Scripts/python.exe tests/test_gpu_lock_respected.py
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name
          + ("" if cond or not detail else "  -- " + str(detail)[:300]))
    if not cond and os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


import comfy_client
import config
import gpu_lock

# The gate ignores the lock under F5_TEST_RUN unless a suite hands in its own
# `gpu_owner` -- otherwise every suite that stubs ComfyUI and drives
# _submit_and_poll would pass or fail depending on whether a training run
# happened to be going on the machine. This suite is testing the mechanism, so
# it hands one in.


class _Holder:
    """Pretend a foreign process holds the card, without starting one."""
    def __init__(self, label="a training run"):
        import types
        self.label = label
        self.ctx = types.SimpleNamespace(deps=types.SimpleNamespace(gpu_owner=lambda: (999999, self.label)))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def test_gpu_holder_reads_the_lock():
    check("a free card reports no holder", comfy_client.gpu_holder() in (None, )
          or isinstance(comfy_client.gpu_holder(), str))
    with _Holder("Ideogram 4 training") as h:
        check("a held card reports the label",
              comfy_client.gpu_holder(h.ctx) == "Ideogram 4 training")
    check("the label does not linger once the lock is gone",
          comfy_client.gpu_holder() is None
          or comfy_client.gpu_holder() != "Ideogram 4 training"
          or gpu_lock.owner() is not None)


def test_the_check_can_be_switched_off():
    """The escape hatch matters: the lock is advisory, and a user who knows the
    holder is finished must not be locked out of their own app."""
    with _Holder() as h:
        h.ctx.deps.COMFY_RESPECT_GPU_LOCK = False
        check("COMFY_RESPECT_GPU_LOCK=0 draws anyway",
              comfy_client.gpu_holder(h.ctx) is None)


def test_a_submit_is_refused_without_touching_the_server():
    """Refused BEFORE the semaphore and before /prompt: the point is not to
    load a model on top of the trainer at all."""
    calls = {"post": 0, "health": 0}
    import types
    server = types.SimpleNamespace(deps=types.SimpleNamespace(
        server_healthy=lambda *a, **k: (calls.__setitem__("health", calls["health"] + 1), True)[1],
        requests=types.SimpleNamespace(post=lambda *a, **k: calls.__setitem__("post", calls["post"] + 1))))
    with _Holder("Ideogram 4 training") as h:
        server.deps.gpu_owner = h.ctx.deps.gpu_owner
        out = comfy_client._submit_and_poll(server, {"1": {}}, timeout=5)
    check("nothing came back", out is None)
    check("and nothing was submitted to ComfyUI", calls["post"] == 0, calls)


def test_a_stale_refusal_is_not_used_to_explain_a_later_failure():
    """The window is what keeps this from becoming a wrong answer: a refusal
    from an hour ago must not explain a fresh content refusal."""
    import time as _t
    comfy_client.LAST_GPU_REFUSAL.update({"label": "old run", "at": _t.time() - 4000})
    check("a stale refusal is ignored", comfy_client.recent_gpu_refusal() is None)
    comfy_client.LAST_GPU_REFUSAL.update({"label": "live run", "at": _t.time()})
    check("a fresh one is reported", comfy_client.recent_gpu_refusal() == "live run")
    comfy_client.LAST_GPU_REFUSAL.update({"label": None, "at": 0.0})
    check("and none at all means none", comfy_client.recent_gpu_refusal() is None)


def test_the_failure_is_explained_rather_than_blamed_on_the_server():
    src = (ROOT / "agent/tool_image_handlers.py").read_text(encoding="utf-8")
    i = src.index("def _gpu_busy_error(")
    branch = src[i:i + 1800]
    # recent_gpu_refusal, NOT gpu_holder: "is the card busy right now" is the
    # wrong question here. Ideogram can refuse a prompt on content grounds
    # during a training run, and blaming the trainer for that is a confident
    # wrong answer. Only a job WE declined for the lock counts.
    check("a refusal we actually made is consulted at the tool boundary",
          "recent_gpu_refusal()" in branch, branch)
    fail = src[src.index('_GENERATE_FAILURE.get("reason"') - 200:]
    check("BEFORE the recorded sub-reason is turned into words",
          fail.index("_gpu_busy_error(") < fail.index('reason == "refused"'), fail[:600])
    check("the video branch explains it the same way",
          '_gpu_busy_error("Video generation")' in src)
    check("and the model is told not to offer a retry",
          "do NOT offer to retry" in branch)

    chars = (ROOT / "bot/tg_characters.py").read_text(encoding="utf-8")
    check("the Telegram character render explains it too",
          "recent_gpu_refusal()" in chars and '"gpu_busy"' in chars)
    strings = (ROOT / "bot/tg_strings.py").read_text(encoding="utf-8")
    check("with a localized string",
          '"gpu_busy"' in strings
          and '"ru":' in strings[strings.index('"gpu_busy"'):
                                 strings.index('"gpu_busy"') + 500])

    gui = (ROOT / "gui/gui.py").read_text(encoding="utf-8")
    check("and the app says it at startup, not only on a failed render",
          "gpu_holder()" in gui)


def _main():
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        try:
            fn()
        except Exception as exc:
            check(fn.__name__ + " raised", False, exc)
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d/%d passed" % (len(RESULTS) - len(bad), len(RESULTS)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_main())
