"""Does the extraction cache actually hit on a repeat run? (live web + LM Studio)

Two identical quick runs back to back: the second should serve most pages from
memory/research/_cache (hits) and crawl faster.

Run: venv/Scripts/python.exe bench/dr_cache_live.py ["topic"]
"""
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
import deep_research as D  # noqa: E402
from models import Context  # noqa: E402

topic = sys.argv[1] if len(sys.argv) > 1 else "how does a heat pump work in cold climates"
ctx = Context(models=None, transcription_cache={}, cache_file=Path("tests/_dr_cache.json"),
              asr_lock=threading.Lock(), tts_lock=threading.Lock(),
              model_name=config.MODEL_NAME, no_think=True, api_min_interval=0.2)
rows = []
for i in (1, 2):
    t = time.time()
    r = D.run_deep_research(ctx, topic, depth="quick")
    s = r.get("stats", {})
    rows.append((i, round(time.time() - t), s.get("pages"), s.get("cache")))
    print("run", *rows[-1], flush=True)
(_, t1, _, c1), (_, t2, _, c2) = rows
print(f"RESULT first {t1}s {c1} | repeat {t2}s {c2}")
