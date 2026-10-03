"""Live: the picture's LOOK read on the real model, over the fixture table the
suites stub with (tests/_look_stub.py). Run with LM Studio up."""
import os, sys
os.environ["INTENT_LIVE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "agent", "imaging", "core", "tests"):
    sys.path.insert(0, os.path.join(ROOT, d))
from _look_stub import LOOKS
from ideogram_layout import _look

bad = 0
for p, want in LOOKS.items():
    got = _look(p)
    ok = got == want
    bad += not ok
    print("ok  " if ok else "BAD ", repr(p), "->", got, "" if ok else f"(want {want})")
print("ALL OK" if not bad else f"{bad} wrong of {len(LOOKS)}")
