"""Headless real-LLM deep-research runner for iterating on output quality.

Usage: python tests/run_real.py "topic" [depth]

Builds a minimal duck-typed ctx (no F5/Whisper load), runs the full pipeline
against the live LM Studio model, and prints the report path + stats so the
report can be inspected end-to-end."""
import sys, os, time, threading, pathlib
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from deep_research import run_deep_research


class Ctx:
    """Duck-typed context: only the attributes the LLM + pipeline read."""
    def __init__(self, model_name):
        self.model_name = model_name
        self.reasoning_effort = "high"
        self.no_think = True
        self.api_lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.last_api_call_time = 0.0
        self.api_min_interval = 0.0
        self.last_research_report = ""
        self.last_research_path = None

    def is_cancelled(self):
        return self.cancel_event.is_set()

    def remember(self, *a, **k):
        pass

    def set_stage(self, *a, **k):
        pass


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "The Adam optimizer for stochastic gradient descent"
    depth = sys.argv[2] if len(sys.argv) > 2 else "standard"
    model = os.getenv("MODEL_NAME") or "qwen3.6-35b-a3b-uncensored-heretic-i1"
    ctx = Ctx(model)

    t0 = time.time()

    def cb(phase, stats, msg):
        el = int(time.time() - t0)
        print(f"[{el:4d}s] {phase:28s} src={stats.get('sources',0):3d} "
              f"pg={stats.get('pages',0):3d} found={stats.get('findings',0):3d}  {str(msg)[:60]}",
              flush=True)

    print(f"=== RUN topic={topic!r} depth={depth} model={model} ===", flush=True)
    res = run_deep_research(ctx, topic, depth=depth, progress=cb)
    dt = time.time() - t0
    rep = res.get("report") or ""
    print(f"\n=== DONE in {dt:.0f}s · path={res.get('path')} · report_chars={len(rep)} ===")
    st = res.get("stats", {})
    for k in ("sources", "pages", "findings", "math_audit", "stage_timings"):
        if k in st:
            print(f"  {k}: {st[k]}")


if __name__ == "__main__":
    main()
