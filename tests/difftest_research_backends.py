"""Differential harness: the SAME research operations through both backends.

WHY THIS EXISTS. Step 3 of the GUI service extraction added a second
implementation of research_api.ResearchClient. Two implementations of one
contract rot apart silently — the in-process one keeps working, the HTTP one
drifts, and every existing suite stays green because every existing suite runs
in-process. This harness runs the identical sequence of boundary operations
against InProcessResearchClient and HttpResearchClient and asserts the results
are equal, including the PROGRESS STREAM and the CANCELLATION behaviour, which
are the two things a naive HTTP boundary silently destroys.

DETERMINISM AND SAFETY. The real pipeline needs a live LM Studio, a GPU and the
open web. None of that belongs in a test whose subject is the BACKEND, so both
sides run the same deterministic stand-in, tests/_fake_deep_research.py: the
in-process side imports it into sys.modules, the service process is pointed at
it through research_service.ENV_DR_MODULE. NO GPU, NO NETWORK beyond localhost,
no real research directory is written (the stand-in returns path=None).

The service runs on an OS-assigned free port, never the 8791 default, and is
shut down through its /shutdown route with terminate/kill as a backstop in a
finally.

WHAT THIS PROVES / DOES NOT PROVE
  proves      the two backends agree on the result dict, on the full ordered
              progress stream, on knob spec/defaults, on estimate_duration and
              lang_of_text, on override application AND restoration, on error
              propagation, and that BOTH stop a run when the caller's ctx is
              cancelled mid-flight.
  not proved  that the real deep_research pipeline behaves identically in a
              service process. It is the same code on both sides, but its
              dependence on config/LM Studio in the service process is not
              exercised here.

MUTATION-CHECKED. This suite is only worth having if it can fail. See
MUTATIONS below for the deliberate breakages it was verified against.

Run:  venv/Scripts/python.exe -u tests/difftest_research_backends.py
Judge by EXIT CODE (0 = the backends agree).
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import logging
logging.disable(logging.WARNING)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
FAKE_DR = "tests._fake_deep_research"

# MUTATIONS this harness was verified to catch (see the branch notes):
#   1. research_service._run_streaming: drop the progress emit  -> progress
#      stream check goes red.
#   2. research_service._run_streaming: ignore the overrides    -> override
#      check goes red.
#   3. research_service._cancel_run: return False without setting the event
#      -> the cancellation check goes red.

_fails = []
_passes = 0


def check(name, cond, detail=""):
    global _passes
    if cond:
        _passes += 1
        print(f"[PASS] {name}")
    else:
        _fails.append(name)
        print(f"[FAIL] {name}  {detail}")


def same(name, a, b):
    check(name, a == b,
          f"\n  inproc: {json.dumps(a, ensure_ascii=False, default=str)[:400]}"
          f"\n  http  : {json.dumps(b, ensure_ascii=False, default=str)[:400]}")


class Ctx:
    """The smallest thing that satisfies what the boundary asks of a ctx."""

    def __init__(self):
        self.cancel_event = threading.Event()

    def is_cancelled(self):
        return self.cancel_event.is_set()

    def set_stage(self, stage):
        pass


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_healthy(url, proc, deadline=30.0):
    import requests
    end = time.time() + deadline
    while time.time() < end:
        if proc.poll() is not None:
            return False
        try:
            if requests.get(f"{url}/health", timeout=2).json().get("ok"):
                return True
        except Exception:
            time.sleep(0.2)
    return False


def collect(client, ctx, topic, **kw):
    """Run once, returning (result, [progress records])."""
    ticks = []
    result = client.run(ctx, topic,
                        progress=lambda p, s, m: ticks.append([p, s, m]), **kw)
    return result, ticks


def main():
    import research_api
    import research_client
    import research_service

    # The in-process side runs the same stand-in the service will be given.
    import importlib
    sys.modules["deep_research"] = importlib.import_module(FAKE_DR)

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    env[research_service.ENV_DR_MODULE] = FAKE_DR
    env["PYTHONPATH"] = os.pathsep.join([REPO, os.path.join(REPO, "core"), env.get("PYTHONPATH", "")])
    proc = subprocess.Popen(
        [PY, "-u", os.path.join(REPO, "research/research_service.py"), "--port", str(port)],
        cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        if not wait_healthy(url, proc):
            out = b""
            try:
                proc.kill()
                out = proc.stdout.read() or b""
            except Exception:
                pass
            check("service started", False, out.decode("utf-8", "replace")[-800:])
            return 1
        check("service started", True)

        ip = research_client.InProcessResearchClient()
        http = research_service.HttpResearchClient(url)

        # -- the contract itself ------------------------------------------- #
        check("both satisfy the Protocol",
              isinstance(ip, research_api.ResearchClient)
              and isinstance(http, research_api.ResearchClient))
        # NOT the same value, and that is the point: each must report honestly.
        check("cancellation modes are declared, and differ",
              ip.cancellation_mode == "shared_event"
              and http.cancellation_mode == "run_id"
              and set(research_api.CANCELLATION_MODES) >= {"shared_event", "run_id"},
              f"{ip.cancellation_mode} / {http.cancellation_mode}")

        # -- pure queries --------------------------------------------------- #
        for depth in research_api.DEPTHS:
            same(f"estimate_duration({depth})",
                 list(ip.estimate_duration(depth)), list(http.estimate_duration(depth)))
        for text in ["quantum computing", "квантовые вычисления", ""]:
            same(f"lang_of_text({text!r})", ip.lang_of_text(text), http.lang_of_text(text))
        same("knob_spec", ip.knob_spec(), http.knob_spec())
        same("knob_defaults", ip.knob_defaults(), http.knob_defaults())

        # -- a full run, result AND progress stream ------------------------- #
        r_ip, t_ip = collect(ip, Ctx(), "reciprocal rank fusion", depth="quick")
        r_http, t_http = collect(http, Ctx(), "reciprocal rank fusion", depth="quick")
        same("run result", r_ip, r_http)
        check("run result has every contract field",
              research_service.result_is_complete(r_ip)
              and research_service.result_is_complete(r_http), repr(r_ip)[:300])
        # The whole reason /op/run streams NDJSON instead of returning one body.
        same("progress stream (ordered, record for record)", t_ip, t_http)
        check("progress actually streamed more than start+end",
              len(t_http) == len(research_api.PHASES), f"{len(t_http)} ticks")

        # -- out_lang derived from the typed topic on BOTH sides ------------ #
        ru_ip, _ = collect(ip, Ctx(), "квантовые вычисления")
        ru_http, _ = collect(http, Ctx(), "квантовые вычисления")
        same("out_lang derived from the topic", ru_ip["report"], ru_http["report"])
        check("...and it really is ru", "lang=ru" in ru_ip["report"], ru_ip["report"][:120])

        # -- overrides: applied for the run, restored after it -------------- #
        ov = {"DR_MAX_QUERIES": 3, "DR_MULTIHOP_ENABLED": False}
        o_ip, _ = collect(ip, Ctx(), "override probe", overrides=dict(ov))
        o_http, _ = collect(http, Ctx(), "override probe", overrides=dict(ov))
        same("overrides reach the run", o_ip["report"], o_http["report"])
        check("...and were really applied",
              "queries=3" in o_ip["report"] and "multihop=False" in o_ip["report"],
              o_ip["report"][:200])
        # A later run must see the defaults again — overrides are per-run.
        same("overrides are restored afterwards",
             ip.knob_defaults(), http.knob_defaults())
        check("...to the ORIGINAL defaults",
              ip.knob_defaults()["DR_MAX_QUERIES"] == 8
              and ip.knob_defaults()["DR_MULTIHOP_ENABLED"] is True,
              repr(ip.knob_defaults()))

        # -- errors propagate the same way ---------------------------------- #
        errs = []
        for c in (ip, http):
            try:
                c.run(Ctx(), "bad knob", overrides={"NOT_A_KNOB": 1})
                errs.append("no exception")
            except Exception as exc:
                errs.append(str(exc))
        check("an unknown knob fails on both backends, naming the knob",
              all("NOT_A_KNOB" in e for e in errs), repr(errs))

        # -- CANCELLATION: the part a naive boundary strands ----------------- #
        for label, client in (("inprocess", ip), ("http", http)):
            ctx = Ctx()
            ticks = []
            box = {}

            def go():
                try:
                    box["r"] = client.run(ctx, "cancel probe",
                                          progress=lambda p, s, m: ticks.append(p))
                except Exception as exc:      # a stranded Stop must not look green
                    box["exc"] = exc

            th = threading.Thread(target=go)
            th.start()
            # Let it get going (first tick in), then press Stop exactly as the
            # GUI does. A blind sleep raced the HTTP backend's start-up on CI.
            end = time.time() + 30
            while not ticks and time.time() < end and th.is_alive():
                time.sleep(0.02)
            ctx.cancel_event.set()
            th.join(timeout=30)
            check(f"{label}: the run ended after Stop", not th.is_alive())
            r = box.get("r") or {}
            check(f"{label}: Stop actually cancelled the run",
                  r.get("cancelled") is True,
                  f"{box.get('exc')!r} {r!r}"[:300])
            check(f"{label}: Stop landed mid-run, not after it",
                  0 < len(ticks) < len(research_api.PHASES), f"{len(ticks)} ticks")

    finally:
        try:
            import requests
            requests.post(f"{url}/shutdown", timeout=5)
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

    print(f"\n{_passes} passed, {len(_fails)} failed")
    if _fails:
        print("FAILED: " + ", ".join(_fails))
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
