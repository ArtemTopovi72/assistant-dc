"""find_photo picks among the candidates in ONE vision call: the first that passed the
face/colour filters was taken as it came, whoever it showed (2026-10-13).

Run: venv/Scripts/python.exe tests/test_photo_pick.py
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "research"), os.path.join(ROOT, "agent"),
                os.path.join(ROOT, "core")]
import search
import llm
from PIL import Image

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    OK, BAD = (OK + 1, BAD) if cond else (OK, BAD + 1)
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"   {extra}"))


d = tempfile.mkdtemp()
paths = []
for i, c in enumerate(("red", "green", "blue")):
    p = os.path.join(d, f"c{i}.jpg")
    Image.new("RGB", (64, 64), c).save(p)
    paths.append(p)

CALLS = []
llm.analyze_image_with_llm = lambda *a, **k: (CALLS.append(k), "Photo 3")[1]
got = search._pick_photo(object(), "Lucia GTA 6", paths)
check("all candidates go to the vision model in one call, labelled",
      len(CALLS) == 1 and [p for _, p in CALLS[0]["image_paths"]] == paths, CALLS)
check("...and the one it names is taken", got == paths[2], got)

llm.analyze_image_with_llm = lambda *a, **k: "no idea"
check("an answer without a number keeps the first", search._pick_photo(object(), "x", paths) == paths[0])
check("no ctx (no model) keeps the first without a call", search._pick_photo(None, "x", paths) == paths[0])

print(f"\n{OK}/{OK + BAD} passed")
sys.exit(1 if BAD else 0)
