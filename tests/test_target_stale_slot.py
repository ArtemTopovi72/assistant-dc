"""A photo sent with the message is this turn's picture, whatever pointer a button left in the session
slot minutes ago (01:12: the new upload was answered, and animated, as the old one).

Run: venv/Scripts/python.exe tests/test_target_stale_slot.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_tasks as TT

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


sess = types.SimpleNamespace(target_image="old")
check("a fresh upload ignores the stale slot", TT._target_id_of(types.SimpleNamespace(image_id="", image_path="/new.jpg"), sess) == "")
check("a button's carried id wins over everything",
      TT._target_id_of(types.SimpleNamespace(image_id="btn", image_path="/new.jpg"), sess) == "btn")
check("words with no upload still use the pointer",
      TT._target_id_of(types.SimpleNamespace(image_id="", image_path=""), sess) == "old")

print(f"\n{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
