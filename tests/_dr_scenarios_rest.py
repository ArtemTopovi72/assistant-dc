"""Re-run only the scenarios that didn't complete in the first eval pass
(the run died of VRAM OOM mid-product). Appends to the existing _index.json.
Run: ./venv/Scripts/python.exe tests/_dr_scenarios_rest.py
"""
import sys, os, io, time, json, traceback, gc
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

from models import Context
from deep_research import run_deep_research

OUT = os.path.join(os.path.dirname(__file__), "_dr_eval")
os.makedirs(OUT, exist_ok=True)

SCENARIOS = [
    ("product",      "best budget mechanical keyboard 2026 reviews", "quick"),
    ("news",         "latest AI regulation news 2026", "quick"),
    ("local_poster", "ДК Троицкий Санкт-Петербург афиша расписание мероприятий", "quick"),
]


def make_ctx():
    import threading
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


def load_index():
    p = os.path.join(OUT, "_index.json")
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return []


def main():
    index = load_index()
    done = {r.get("label") for r in index}
    for label, topic, depth in SCENARIOS:
        if label in done:
            print(f"skip {label} (already in index)", flush=True)
            continue
        print(f"\n{'='*70}\n>>> {label}: {topic!r}\n{'='*70}", flush=True)
        ctx = make_ctx()                      # fresh ctx per scenario
        t0 = time.time()
        try:
            res = run_deep_research(ctx, topic, depth=depth)
            elapsed = time.time() - t0
            report = res.get("report") or ""
            with open(os.path.join(OUT, f"{label}.md"), "w", encoding="utf-8") as f:
                f.write(report)
            index.append({"label": label, "topic": topic,
                          "elapsed_sec": round(elapsed, 1),
                          "stats": res.get("stats", {}), "chars": len(report),
                          "queries": res.get("queries", [])})
            print(f"<<< {label} done in {elapsed:.0f}s | {res.get('stats')}", flush=True)
        except Exception:
            traceback.print_exc()
            index.append({"label": label, "topic": topic, "error": traceback.format_exc()})
        with open(os.path.join(OUT, "_index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        del ctx
        gc.collect()
    print("\nREST DONE.", flush=True)


if __name__ == "__main__":
    main()
