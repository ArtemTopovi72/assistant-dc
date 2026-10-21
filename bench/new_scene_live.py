"""Live: image_style._new_scene -- the people go into a new scene/pose/action (faces become
references, pose free) vs they stay as they stand while something changes (the picture pins pose).
The owner's rule (2026-10-21): add Stalin or Trotsky beside him -> the pose stays; cyber-Lenin with a
new body fighting demons -> the face may even be shown from another angle.

    venv/Scripts/python bench/new_scene_live.py
"""
import os, sys, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [ROOT] + [os.path.join(ROOT, d) for d in ("agent", "core", "bot", "imaging", "media", "knowledge")] \
    + [p for p in sys.path if os.path.basename(p) != "bench"]
sys.stdout.reconfigure(encoding="utf-8")
os.environ.pop("F5_TEST_RUN", None)
import image_style as S

CASES = [
    ("A cinematic scene of Lenin in a golden cybernetic armor suit in hell, punching one demon and "
     "crushing another's neck amid lava.", True),
    ("put him on a beach at sunset, surfing a huge wave", True),
    ("the woman riding a horse across a desert at full gallop", True),
    ("the two men dancing tango in a ballroom", True),
    ("add Stalin standing next to him", False),
    ("add Trotsky to the left of him, looking at the camera", False),
    ("give him a red hat", False),
    ("change the background to a beach", False),
    ("make it night time with a full moon", False),
    ("dress him in a modern business suit", False),
    # not among the question's own examples
    ("the old man fishing from a wooden boat on a lake at dawn", True),
    ("him playing chess against a robot in a dark bar", True),
    ("add a cat sitting on his lap", False),
    ("make the photo black and white and add film grain", False),
]
bad = 0
for words, want in CASES:
    got = S._new_scene(types.SimpleNamespace(), words)
    bad += got is not want
    print(("PASS " if got is want else "FAIL ") + f"{'scene' if want else 'keep '} {words[:70]!r} -> {got}")
print(f"\n{bad} failed")
