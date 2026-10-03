"""Phase 9 end-to-end validation: rerun the EXACT Di Zenzo investigation that
exposed the original weaknesses, at standard depth. Writes the report so it can
be reviewed against the critique (primary paper, citation lineage, derivations,
misconceptions, source hierarchy, bibliography).
"""
import sys, os, io, time, json, threading, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

from models import Context
from deep_research import run_deep_research

# Exact prompt that produced the 6.5-7/10 report (from that run's meta.json).
PRIOR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                     "memory", "research",
                     "1781699035_perform-a-deep-technical-investigation-of-the-di",
                     "meta.json")
TOPIC = json.load(open(PRIOR, encoding="utf-8"))["topic"]

OUT = os.path.join(os.path.dirname(__file__), "_dr_eval", "dizenzo_phase9.md")


def make_ctx():
    ctx = Context.__new__(Context)
    ctx.model_name = ""
    ctx.no_think = True
    ctx.last_api_call_time = 0.0
    ctx.api_min_interval = 0.0
    ctx.api_lock = threading.Lock()
    ctx.cancel_event = threading.Event()
    ctx.last_research_report = ""
    ctx.last_research_path = None
    return ctx


def main():
    print(f">>> Di Zenzo benchmark (standard depth), topic {len(TOPIC)} chars", flush=True)
    t0 = time.time()
    try:
        res = run_deep_research(make_ctx(), TOPIC, depth="standard")
        rep = res.get("report") or ""
        open(OUT, "w", encoding="utf-8").write(rep)
        print(f"<<< done in {time.time()-t0:.0f}s | {res.get('stats')} | {len(rep)} chars",
              flush=True)
        print("written:", OUT, flush=True)
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
