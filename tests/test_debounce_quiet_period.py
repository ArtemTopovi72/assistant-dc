"""The per-chat debounce is a QUIET period, not a fixed window.

Fragments of one thought (five, 0.6 s apart) must not be cut by a window measured
from the FIRST message and answered as separate requests: the deadline moves with
every fragment, capped from the first one. Pure: the resolver is stubbed.

Run: venv/Scripts/python.exe tests/test_debounce_quiet_period.py
"""
import os
import queue
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="tg_debounce_"))

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


bot = T.TelegramBot("1:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
batches = []
bot._resolve_and_push = lambda chat_id, batch: batches.append(list(batch))
T._DEBOUNCE_S, T._DEBOUNCE_MAX_S = 0.5, 2.0
T._DEBOUNCE_FWD_S, T._DEBOUNCE_FWD_MAX_S = 1.5, 3.0

q = queue.Queue(); bot._queues[7] = q
th = threading.Thread(target=bot._debounce_loop, args=(7,), daemon=True); th.start()
for i in range(5):
    q.put({"type": "text", "text": f"fragment {i}"}); time.sleep(0.2)
time.sleep(1.2)
check("five fragments 0.2 s apart become ONE batch", len(batches) == 1 and len(batches[0]) == 5, [len(b) for b in batches])

batches.clear()
for i in range(12):
    q.put({"type": "text", "text": f"stream {i}"}); time.sleep(0.2)
time.sleep(1.2)
check("a never-ending stream is cut at the cap, not held forever",
      len(batches) >= 2 and len(batches[0]) < 12, [len(b) for b in batches])

batches.clear()
q.put({"type": "text", "text": "one"}); time.sleep(0.9)
q.put({"type": "text", "text": "two"}); time.sleep(1.2)
check("a pause longer than the quiet period splits", len(batches) == 2, [len(b) for b in batches])

batches.clear()
q.put({"type": "text", "text": "fwd", "forwarded": True}); time.sleep(1.0)
q.put({"type": "text", "text": "объясни прикол"}); time.sleep(1.2)
check("a bare forward waits for the typed instruction: ONE batch", len(batches) == 1 and len(batches[0]) == 2, [len(b) for b in batches])

batches.clear()
q.put({"type": "text", "text": "alone", "forwarded": True}); time.sleep(2.2)
check("a forward with no instruction still goes out", len(batches) == 1, [len(b) for b in batches])

batches.clear()
for i in range(3):   # sent together, delivered late and 2 s apart (a stalled poll): still one batch
    q.put({"type": "text", "text": f"late {i}", "forwarded": True, "_sent_ts": time.time() - 4}); time.sleep(2.0)
time.sleep(6.5)
check("forwards sent together but delivered late stay ONE batch", len(batches) == 1 and len(batches[0]) == 3, [len(b) for b in batches])

# The app was down: a link forwarded at 14:33 and an uncaptioned screenshot sent at 14:57 arrived together
# at start and were answered as one turn -- about the screenshot, the article only «Прочитал» (2026-10-22).
batches.clear()
q.put({"type": "text", "text": "https://www.bbc.com/news/x", "forwarded": True, "_sent_ts": time.time() - 1600})
q.put({"type": "photo", "file_id": "shot", "_sent_ts": time.time() - 160})
time.sleep(2.0)
check("messages sent far apart and delivered together are separate turns",
      len(batches) == 2 and batches[0][0].get("type") == "text" and batches[1][0].get("type") == "photo",
      [[i.get("type") for i in b] for b in batches])

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
