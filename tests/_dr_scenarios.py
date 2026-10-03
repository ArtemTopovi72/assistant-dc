"""Drive run_deep_research over a spread of categories to evaluate report quality.
Run: ./venv/Scripts/python.exe tests/_dr_scenarios.py
Writes each report + a summary to tests/_dr_eval/.
"""
import sys, os, io, time, json, traceback
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

from models import Context, Models
from deep_research import run_deep_research

OUT = os.path.join(os.path.dirname(__file__), "_dr_eval")
os.makedirs(OUT, exist_ok=True)

# (label, topic, depth) — spread across the category router.
SCENARIOS = [
    ("science",       "Di Zenzo structure tensor and its modern modifications", "quick"),
    ("entertainment", "Кочегар фильм Балабанов 2010 сюжет", "quick"),
    ("company",       "Wildberries employer reviews is it a good company to work for", "quick"),
    ("product",       "best budget mechanical keyboard 2026 reviews", "quick"),
    ("news",          "latest AI regulation news 2026", "quick"),
    ("local_poster",  "ДК Троицкий Санкт-Петербург афиша расписание мероприятий", "quick"),
]


def make_ctx():
    ctx = Context.__new__(Context)
    # minimal fields the research path touches
    ctx.model_name = ""          # -> falls back to config.MODEL_NAME
    ctx.no_think = True
    ctx.last_api_call_time = 0.0
    ctx.api_min_interval = 0.0
    import threading
    ctx.api_lock = threading.Lock()
    ctx.cancel_event = threading.Event()
    ctx.last_research_report = ""
    ctx.last_research_path = None
    return ctx


def main():
    ctx = make_ctx()
    index = []
    for label, topic, depth in SCENARIOS:
        print(f"\n{'='*70}\n>>> {label}: {topic!r} (depth={depth})\n{'='*70}", flush=True)
        t0 = time.time()
        try:
            res = run_deep_research(ctx, topic, depth=depth)
            elapsed = time.time() - t0
            report = res.get("report") or ""
            fn = os.path.join(OUT, f"{label}.md")
            with open(fn, "w", encoding="utf-8") as f:
                f.write(report)
            stats = res.get("stats", {})
            rec = {"label": label, "topic": topic, "elapsed_sec": round(elapsed, 1),
                   "stats": stats, "chars": len(report),
                   "queries": res.get("queries", []), "file": fn}
            index.append(rec)
            print(f"<<< {label} done in {elapsed:.0f}s | {stats} | {len(report)} chars", flush=True)
        except Exception:
            traceback.print_exc()
            index.append({"label": label, "topic": topic, "error": traceback.format_exc()})
        with open(os.path.join(OUT, "_index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    print("\nALL DONE. Index written to _dr_eval/_index.json", flush=True)


if __name__ == "__main__":
    main()
