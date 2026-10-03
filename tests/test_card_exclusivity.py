"""Giving the card to a render, and giving it back.

A 24 GB card holds either the 20 GB chat model or a render, not both. When both
are asked for, nothing errors -- ComfyUI streams the weights it cannot fit and
the job just gets slow (measured: a 30-second song goes 43.5 s -> 273.6 s). So
renders evict the chat model and bring it back afterwards.

Everything that could touch the GPU or LM Studio is stubbed; this asserts the
bookkeeping, which is where the bugs would be: reloading in the middle of a
batch, evicting twice, or losing the model because eviction raised.

Run: venv/Scripts/python.exe tests/test_card_exclusivity.py
"""
import os
import sys
import threading
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import logging; logging.basicConfig(level=logging.CRITICAL)

import comfy_client as CC

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:220])


calls = []


class _FakeLT:
    """Stands in for lora_training: records instead of touching the machine."""
    raise_on_free = False

    @staticmethod
    def loaded_llm_id():
        return "chat-model-26b"

    @staticmethod
    def free_gpu(log=None, min_free_mb=None):
        calls.append("free")
        if _FakeLT.raise_on_free:
            raise RuntimeError("lms is not installed")

    @staticmethod
    def reload_llm(model_id, log=None):
        calls.append("reload:" + model_id)
        return True

    FREE_GPU_WAIT_S = 25.0

    record_wait = False
    free_readings = []          # successive wait_vram_free results; empty = enough

    @staticmethod
    def wait_vram_free(min_free_mb, timeout=25.0, poll=0.5):
        if _FakeLT.record_wait:
            calls.append("wait-vram:%d" % min_free_mb)
        return _FakeLT.free_readings.pop(0) if _FakeLT.free_readings else min_free_mb


sys.modules["lora_training"] = _FakeLT


def reset():
    calls.clear()
    CC._excl_users = 0
    CC._excl_model = ""
    CC._excl_label = ""
    CC._excl_free.set()
    _FakeLT.raise_on_free = False


# ── the plain slot must not touch the chat model ─────────────────────────────
reset()
with CC._gpu_slot():
    pass
check("a normal render does NOT evict anything", calls == [], calls)

# An edit pipeline consults a vision model BETWEEN its submits. Evicting under
# one would break the very turn that asked for the edit, which is why this is
# opt-in per call site rather than global.
import inspect
sp = inspect.getsource(CC._submit_and_poll)
check("_submit_and_poll defaults to sharing the card",
      "exclusive: bool = False" in sp, sp[:200])

# ── the exclusive slot evicts, then restores ─────────────────────────────────
reset()
with CC._gpu_slot(exclusive=True, label="Music3"):
    check("the card reports what it is busy with",
          CC.card_is_exclusive() == "Music3", CC.card_is_exclusive())
    check("and a chat turn can see it must wait",
          CC._excl_free.is_set() is False)
    check("the model is unloaded before the render", calls == ["free"], calls)
check("the model comes back when the render ends",
      calls == ["free", "reload:chat-model-26b"], calls)
check("and the card reads as shared again", CC.card_is_exclusive() == "")
check("waiters are released", CC.wait_for_card(1) is True)

# ── a batch pays the reload ONCE, not per job ────────────────────────────────
# This is the whole point: five pictures in the queue must not reload a 20 GB
# model four times in the gaps between them.
reset()
started = threading.Barrier(3)
done = []


def job(n):
    started.wait(5)
    with CC._gpu_slot(exclusive=True, label="pic %d" % n):
        time.sleep(0.15)
    done.append(n)


threads = [threading.Thread(target=job, args=(i,)) for i in range(3)]
for t in threads: t.start()
for t in threads: t.join(20)

check("three queued renders evicted once", calls.count("free") == 1, calls)
check("and reloaded once, after the last one", calls.count("reload:chat-model-26b") == 1, calls)
check("the reload is the LAST thing that happens", calls[-1].startswith("reload:"), calls)
check("all three renders ran", sorted(done) == [0, 1, 2], done)

# ── nesting must not double-claim ────────────────────────────────────────────
reset()
with CC._gpu_slot(exclusive=True, label="outer"):
    with CC._gpu_slot(exclusive=True, label="inner"):
        pass
    check("an inner slot does not release the card early",
          calls == ["free"], calls)
check("the card is released once the outer slot exits",
      calls == ["free", "reload:chat-model-26b"], calls)

# ── failure must never cost somebody their render ────────────────────────────
reset()
_FakeLT.raise_on_free = True
ran = False
with CC._gpu_slot(exclusive=True, label="Music3"):
    ran = True
check("a failed eviction still lets the render proceed", ran)
check("and does not leave a phantom model to 'restore'",
      "reload:chat-model-26b" not in calls, calls)
check("nor leaves chat turns waiting forever", CC._excl_free.is_set(), calls)

# ── the switch ───────────────────────────────────────────────────────────────
reset()
_saved = CC.COMFY_EVICT_LLM
try:
    CC.COMFY_EVICT_LLM = False
    with CC._gpu_slot(exclusive=True, label="Music3"):
        pass
    check("COMFY_EVICT_LLM=0 leaves the card shared", calls == [], calls)
finally:
    CC.COMFY_EVICT_LLM = _saved

# ── the two call sites that opt in ───────────────────────────────────────────
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(n):
    return open(os.path.join(root, n), encoding="utf-8").read()


check("music renders exclusively", "exclusive=True" in read("media/music.py"))
check("Ideogram renders exclusively", "exclusive=True" in read("imaging/ideogram.py"))
# The refinement loop evaluates each attempt with a vision model, so its own
# submit must stay shared -- the eviction there belongs to ideogram.generate,
# which finishes restoring before it returns.
check("the refinement path does NOT opt in itself",
      "exclusive=True" not in read("imaging/image_generate.py"))

# -- a chat turn waits it out instead of fighting the render ------------------
# Calling LM Studio while the card is rendering would JIT-load 20 GB into what
# the render left over: slow to load, and it steals the card back from the very
# job being waited on. So the turn waits -- and says why.
import llm

reset()
seen_stages = []
ctx = types.SimpleNamespace(set_stage=seen_stages.append)

llm._wait_for_free_card(ctx)
check("a shared card is not waited on", seen_stages == [], seen_stages)

CC._claim_card("Music3")
waited = []


def turn():
    t0 = time.time()
    llm._wait_for_free_card(ctx)
    waited.append(time.time() - t0)


th = threading.Thread(target=turn); th.start()
time.sleep(0.25)
check("the turn is still waiting while the render runs", not waited)
check("and the user is told why, not left staring at nothing",
      seen_stages == ["Waiting for the graphics card"], seen_stages)
CC._release_card()
th.join(10)
check("the turn resumes once the model is back", len(waited) == 1, waited)

# The thread holding the render slot must never wait for itself.
reset()
seen_stages.clear()
with CC._gpu_slot(exclusive=True, label="Music3"):
    _t0 = time.time()
    llm._wait_for_free_card(ctx)
    check("the rendering thread does not wait for itself",
          time.time() - _t0 < 1.0 and seen_stages == [],
          (time.time() - _t0, seen_stages))

import stages as _stages
check("the waiting stage is translated for the user",
      _stages.translate("Waiting for the graphics card", "ru")
      != "Waiting for the graphics card")


# -- the draw pipeline: render, read the picture back, edit the boxes ---------
# The layout critic looks at the render with a MODEL and rewrites the layout
# JSON, so the model has to be back by the time the render call returns. This is
# the sequence that would break if a render ever held the card past its own
# submit.
reset()
order = []


def fake_submit(*a, **kw):
    with CC._gpu_slot(exclusive=kw.get("exclusive", False), label="ideogram4"):
        order.append("render")
    return "pic.png"


with CC.card_session("drawing"):
    fake_submit(exclusive=True)
    fake_submit(exclusive=True)
    fake_submit(exclusive=True)
    check("a repair chain of three redraws evicts once, not three times",
          calls.count("free") == 1, calls)
    check("and does not reload between attempts",
          "reload:chat-model-26b" not in calls, calls)
    check("a thread inside a session never waits for itself",
          CC.this_thread_holds_card() is True)
check("the model is back before the critic can be asked anything",
      calls == ["free", "reload:chat-model-26b"], calls)
check("all three redraws ran", order == ["render"] * 3, order)

# And the critic call itself must go through: after the session the card is
# shared again, so the vision call is not made to wait.
seen2 = []
llm._wait_for_free_card(types.SimpleNamespace(set_stage=seen2.append))
check("the critic's vision call is not blocked afterwards", seen2 == [], seen2)

src_da = read("imaging/draw_agent.py")
check("only the collage repair chain is wrapped, never the critic loop",
      src_da.count("card_session(") == 1, src_da.count("card_session("))


# -- a test run must never unload the operator's live model ------------------
# Suites that drive the render path with a stubbed ComfyUI reached
# `lms unload --all` for real and took the machine's chat model down mid-run.
reset()
_real_lt = sys.modules.pop("lora_training", None)
os.environ["F5_TEST_RUN"] = "1"
try:
    import lora_training as _real_module   # the real one, back in sys.modules
    with CC._gpu_slot(exclusive=True, label="Music3"):
        pass
    check("under F5_TEST_RUN the real model is left alone", calls == [], calls)
finally:
    os.environ.pop("F5_TEST_RUN", None)
    sys.modules["lora_training"] = _FakeLT

reset()
os.environ["F5_TEST_RUN"] = "1"
try:
    with CC._gpu_slot(exclusive=True, label="Music3"):
        pass
    check("but a suite that stubs it still exercises the mechanism",
          calls == ["free", "reload:chat-model-26b"], calls)
finally:
    os.environ.pop("F5_TEST_RUN", None)


# -- the model must come back into a card ComfyUI has VACATED -----------------
# 2026-09-11: after a render the chat model was reloaded while ComfyUI still
# held the render's weights; it came up short of headroom, crashed on the first
# vision call, and every turn for the next hour was "No models loaded".
reset()
freed = []
_orig_free = CC._free_comfy_models
CC._free_comfy_models = lambda: (freed.append("comfy"), calls.append("comfy-free"))
_FakeLT.record_wait = True
try:
    with CC._gpu_slot(exclusive=True, label="Music3"):
        pass
    check("ComfyUI is asked to drop its models BEFORE the chat model reloads",
          calls == ["free", "comfy-free", "wait-vram:%d" % CC.RELOAD_LLM_MIN_FREE_MB,
                    "reload:chat-model-26b"], calls)
    check("...and the driver is asked for the room the model needs before it loads",
          CC.RELOAD_LLM_MIN_FREE_MB >= 19000)
finally:
    # Keep the network out of the rest of the file: the real /free call to a
    # ComfyUI that is not running costs ~2 s of refused connects on Windows and
    # broke the timing check below.
    CC._free_comfy_models = lambda: None
    _FakeLT.record_wait = False

# -- a card ComfyUI has not vacated in time gets a second /free and a longer
#    wait before the model squeezes in (it spilled into shared RAM and the
#    reload crawled at 90 % for minutes, live 2026-10-03)
reset()
CC._free_comfy_models = lambda: calls.append("comfy-free")
_FakeLT.record_wait = True
_FakeLT.free_readings = [6000, CC.RELOAD_LLM_MIN_FREE_MB + 500]
try:
    with CC._gpu_slot(exclusive=True, label="Picture"):
        pass
    w = "wait-vram:%d" % CC.RELOAD_LLM_MIN_FREE_MB
    check("a short card is freed again and waited on before the reload",
          calls == ["free", "comfy-free", w, "comfy-free", w, "reload:chat-model-26b"], calls)
finally:
    CC._free_comfy_models = lambda: None
    _FakeLT.record_wait = False
    _FakeLT.free_readings = []

# -- and a model that LM Studio lost is put back, not mourned every turn -------
revived = []


class _FakeLMS:
    @staticmethod
    def ensure_exclusive(base, model):
        revived.append(model)
        return True, "ok"


_saved_lms = sys.modules.get("lmstudio")
sys.modules["lmstudio"] = _FakeLMS
try:
    llm._revive_last = 0.0
    check("an unrelated 400 does not trigger a reload",
          llm._try_revive_model("n_keep: 9905 >= n_ctx: 8192", {"model": "m"}) is False
          and revived == [], revived)
    check("'No models loaded' brings the model back",
          llm._try_revive_model('{"message": "No models loaded. Please load a model"}',
                                {"model": "chat-model-26b"}) is True
          and revived == ["chat-model-26b"], revived)
    check("a crash report inside the rate limit rides the fresh revive: retry, no reload",
          llm._try_revive_model("The model has crashed without additional information",
                                {"model": "chat-model-26b"}) is True,
          revived)
    check("but not twice within the rate limit -- no reload storm on a model "
          "that crashes on load", revived == ["chat-model-26b"], revived)
    llm._revive_ok = False   # as if that revive had FAILED
    check("a failed revive inside the rate limit means: do not retry",
          llm._try_revive_model("No models loaded", {"model": "chat-model-26b"}) is False
          and revived == ["chat-model-26b"], revived)
    llm._revive_last = 0.0
    check("after the interval it is allowed again",
          llm._try_revive_model("No models loaded", {"model": "chat-model-26b"}) is True
          and revived == ["chat-model-26b"] * 2, revived)
    src_llm = read("agent/llm.py")
    check("both error sites in the stream path call the revive",
          src_llm.count("_try_revive_model(") >= 3, src_llm.count("_try_revive_model("))
    # -- while a render holds the card the model is not "gone", it is lent ----
    reset()
    llm._revive_last = 0.0
    revived.clear()
    CC._excl_model, CC._excl_label = "chat-model-26b", "ideogram4"
    CC._excl_free.clear()
    check("no revive while a render holds the card (the release brings it back)",
          llm._try_revive_model("Model unloaded.", {"model": "chat-model-26b"}) is False
          and revived == [], revived)
    reset()
finally:
    if _saved_lms is not None:
        sys.modules["lmstudio"] = _saved_lms
    else:
        sys.modules.pop("lmstudio", None)

# -- a claim lets the streams that are open finish before unloading ----------
# Live, 2026-09-12: two users drew at once; the first claim unloaded the model
# while the second user's planner was still streaming, the revive loaded it
# back into the render's card, and everything crawled for twenty minutes.
reset()
CC.LLM_INFLIGHT_WAIT_S = 5
order = []
llm._inflight_enter()                     # a stream is open on another thread


def _finish_stream():
    time.sleep(0.4)
    order.append("stream done")
    llm._inflight_exit()


threading.Thread(target=_finish_stream, daemon=True).start()
t0 = time.time()
with CC._gpu_slot(exclusive=True, label="ideogram4"):
    order.append("free at %.1f" % (time.time() - t0))
check("the claim waited for the open stream", order and order[0] == "stream done", order)
check("and then freed the card", calls[:1] == ["free"], calls)
check("the wait was the stream's length, not the timeout", time.time() - t0 < 3, order)
reset()
CC.LLM_INFLIGHT_WAIT_S = 1
llm._inflight_enter()
t0 = time.time()
try:
    with CC._gpu_slot(exclusive=True, label="ideogram4"):
        pass
    check("a stream that never ends is not waited on forever", 0.9 < time.time() - t0 < 3
          and calls[:1] == ["free"], (time.time() - t0, calls))
finally:
    llm._inflight_exit()
check("inflight is back to zero", llm.inflight() == 0)
check("a call made during the wait sees the card as claimed",
      "_excl_model = model" in read("imaging/comfy_client.py").split("wait_no_inflight")[0].split("_claim_card")[-1])
check("every attempt of a call re-checks the card",
      "if attempt:\n                _wait_for_free_card(ctx)" in read("agent/llm.py"))
reset()


print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
