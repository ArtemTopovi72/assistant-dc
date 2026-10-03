import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# the model's read is stubbed; the phrases run live in bench/intent_sweep3_live.py
import intent
from tg_dispatch import _asks_as_file
_FILE = {"пришли эту картинку файлом", "скинь мне эту фотку документом", "пришли файлом", "send it as a file"}
intent.YES_STUB = lambda q, t: "as a FILE" in q and t in _FILE
for t in _FILE:
    assert _asks_as_file(t), t

import inspect, graph_personality
assert "max_chars=16000" in inspect.getsource(graph_personality.personality_node)
print("OK")
from utils import trim_messages
_h = [{"role": "system", "content": "s"}] + [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(40)]
assert len(trim_messages(_h, 60, max_chars=16000)) == 41                 # short turns all kept
_big = [{"role": "system", "content": "s"}] + [{"role": "user", "content": "x" * 3000}] * 20
assert len(trim_messages(_big, 60, max_chars=16000)) == 1 + 8           # budget, floor of 8
print("PASS window by size")
