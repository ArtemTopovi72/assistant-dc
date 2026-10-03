"""Headless deep-research evaluation harness.

Runs run_deep_research across a spread of categories (quick depth), captures the
planner profile + run stats + the report, and appends a compact summary per
scenario to tests/_dr_eval.jsonl. Also logs the engine's INFO lines (junk drops,
query plan, crawl counts) to tests/_dr_eval.log for inspection.
"""
import sys, os, io, json, time, threading, logging
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from models import Context
import config
from deep_research import run_deep_research

logging.basicConfig(filename="tests/_dr_eval.log", filemode="w", level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    encoding="utf-8")

SCENARIOS = [
    ("science",       "Di Zenzo structure tensor color edge detection modifications"),
    ("entertainment", "Кочегар фильм Балабанов 2010 сюжет"),
    ("news",          "latest NVIDIA AI data center GPU announcement"),
    ("company",       "is Wildberries a good company to work for employee reviews"),
    ("jobs",          "Python backend developer vacancy requirements Yandex"),
    ("health",        "magnesium glycinate benefits and evidence"),
    ("product",       "best mechanical keyboard for programming 2026"),
    ("local",         "ДК Троицкий Санкт-Петербург афиша расписание"),
]

OUT = Path("tests/_dr_eval.jsonl")
OUT.write_text("", encoding="utf-8")


def make_ctx():
    return Context(models=None, transcription_cache={},
                   cache_file=Path("tests/_dr_cache.json"),
                   asr_lock=threading.Lock(), tts_lock=threading.Lock(),
                   model_name=config.MODEL_NAME, no_think=True, api_min_interval=0.2)


def main():
    for i, (expect_cat, topic) in enumerate(SCENARIOS, 1):
        print(f"\n===== [{i}/{len(SCENARIOS)}] {topic} =====", flush=True)
        ctx = make_ctx()
        t0 = time.time()
        try:
            res = run_deep_research(ctx, topic, depth="quick")
        except Exception as exc:
            logging.exception("scenario failed")
            res = {"error": str(exc), "report": "", "stats": {}, "path": None}
        elapsed = round(time.time() - t0, 1)
        report = res.get("report", "") or ""

        # pull the planner profile back from the run's state.json
        profile = {}
        path = res.get("path")
        if path:
            sp = Path(path).parent / "state.json"
            if sp.exists():
                try:
                    profile = json.loads(sp.read_text(encoding="utf-8")).get("profile", {})
                except Exception:
                    pass

        rec = {
            "n": i, "topic": topic, "expect_category": expect_cat,
            "got_category": profile.get("category"), "recency": profile.get("recency"),
            "news": profile.get("news"), "elapsed_sec": elapsed,
            "stats": res.get("stats", {}), "path": path,
            "report_len": len(report), "report_head": report[:2500],
            "error": res.get("error"),
        }
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  done in {elapsed}s · cat={rec['got_category']} "
              f"stats={rec['stats']} len={rec['report_len']}", flush=True)

    print("\nALL DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
