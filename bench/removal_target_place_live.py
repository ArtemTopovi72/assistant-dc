"""Live check: when the agent names the removal region as a PLACE («the corner of the table
where the cup is»), the removal target the router reads must be the THING (the cup), not the
place. The place is always visible, so the «is it still there?» check said STILL visible after
a removal that worked (live 2026-10-21 00:57), and the eraser was told to remove the place.

The synth string is the handler's own: f"remove the {region}".

    venv/Scripts/python bench/removal_target_place_live.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [ROOT] + [os.path.join(ROOT, d) for d in ("agent", "core", "bot", "imaging", "media", "knowledge")] \
    + [p for p in sys.path if os.path.basename(p) != "bench"]
sys.stdout.reconfigure(encoding="utf-8")
os.environ.pop("F5_TEST_RUN", None)
import image_router as R

CASES = [  # region the agent writes, a word the target must hold, words it must not be about
    ("the corner of the table where the cup is located", "cup", "corner"),
    ("the left side of the street where the red car is parked", "car", "side"),
    ("the area above the door where the shop sign hangs", "sign", "area"),
    ("the bottom of the image where the shovel is", "shovel", "bottom"),
    ("the man's face where the glasses are", "glass", "face"),
    ("the centre of the wall where the poster is", "poster", "centre"),
    ("the sky where the power lines run", "line", "sky"),
    ("the pavement area where the lamp post stands", "post", "pavement"),
    ("her hand where the coffee cup is held", "cup", "hand"),
    ("the cup on the table", "cup", ""),
    ("the red car", "car", ""),
    ("the lettering on the sign", "letter", ""),
    ("the trash bin next to the bench where the dog sits", "bin", "dog"),
    ("the background behind the woman where the crowd stands", "crowd", "woman"),
]
bad = 0
import image_grounding as G
# A place where the thing still IS is not a former place: read as one, the handler refused the
# removal as «already removed» (live 2026-10-21, «the counter top where the grey cat sleeps»).
for region, _m, _n in CASES[:9] + CASES[-2:]:
    got = G.names_former_place(None, region)
    bad += got is True
    print(("PASS " if got is not True else "FAIL ") + f"not a former place: {region!r} -> {got}")
for region in ("the area where the vase was", "место, где стояла ваза", "the empty spot where the car used to be",
               "the area where the blue vase and flowers were located", "стол там, где лежала книга"):
    got = G.names_former_place(None, region)
    bad += got is not True
    print(("PASS " if got is True else "FAIL ") + f"former place: {region!r} -> {got}")
for region, must, mustnt in CASES:
    t = R.edit_plan(f"remove the {region}")["target"].lower()
    ok = must in t and not (mustnt and mustnt in t)
    bad += not ok
    print(("PASS " if ok else "FAIL ") + f"{region!r} -> {t!r}")
print(f"\n{bad} failed")
