"""Live check of the removal reads (image_router.edit_plan target /
removes_region / about_absence / fill). Phrases from test_removal_targets_region
and test_fixes, which tested the verb regex and the stem matcher.

    venv/Scripts/python bench/removal_intent_live.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [ROOT] + [os.path.join(ROOT, d) for d in ("agent", "core", "bot", "imaging", "media", "knowledge")] \
    + [p for p in sys.path if os.path.basename(p) != "bench"]
sys.stdout.reconfigure(encoding="utf-8")
os.environ.pop("F5_TEST_RUN", None)
import image_grounding as G
import image_router as R
import tool_image_handlers as H

bad = 0
def check(name, ok, got=""):
    global bad
    bad += not ok
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  -> {got}"))

for t, want in [("remove the puddle", True), ("delete the shadow", True), ("erase the logo", True),
                ("get rid of the hat", True), ("wipe out the stain", True), ("take away the cup", True),
                ("hide the watermark", True), ("убери лужу", True), ("удалить тень", True),
                ("сотри пятно", True), ("сними с него очки", True),
                ("clean face without artifacts", False), ("a face without blemishes", False),
                ("make it cleaner", False), ("no more blur", False), ("smooth skin no noise", False),
                ("sharpen details", False), ("почисти кожу", False), ("без артефактов", False)]:
    got = G._is_removal_instruction(t)
    check(("removal: " if want else "not removal: ") + t, got is want, R.edit_plan(t))

for ins, reg in [("remove the puddle", "puddle"), ("remove the puddle", "the puddle on the road"),
                 ("erase the trash can", "trash can"), ("remove her glasses", "glasses"),
                 ("remove the cat", "cat-shaped object"), ("remove the cats", "cat"),
                 ("delete it", "trash can"), ("убери лужу", "лужа"), ("убери шарф", "красный шарфа"),
                 ("bare neck, no scarf", "red scarf"),
                 ("empty space on the wooden table where the vase was, showing only the wood grain",
                  "blue vase with flowers"),
                 ("стол там, где стояла ваза", "ваза"),
                 ("no bag, just the coat and shoulder area", "the dark bag hanging from their shoulder"),
                 ("bare feet on the sand", "sandals"),
                 ("clean background without any text", "lettering")]:
    check(f"removes region: {ins!r} / {reg!r}", G._is_removal_of(ins, reg) is True)
for ins, reg in [("Remove the large black shadow covering the cat-shaped object so that its glossy "
                  "texture and shape are clearly visible.", "cat-shaped object"),
                 ("get rid of the glare on the glasses", "glasses"),
                 ("remove the reflection from the window", "window"), ("убери тень с кота", "кот"),
                 ("make the shirt blue", "shirt"), ("a red vase with tulips", "blue vase with flowers"),
                 ("wooden table where the cat sat, now with a book", "vase")]:
    check(f"edit, not removal: {ins!r} / {reg!r}", G._is_removal_of(ins, reg) is False)

for t, want in [("is the puddle still there?", True), ("лужа всё ещё видна?", True),
                ("the logo is gone", True), ("is the sky blue?", False), ("the hat is red", False)]:
    check(("absence check: " if want else "not an absence check: ") + t, H._is_absence_check(t) is want)

check("target named", "puddle" in R.edit_plan("убери лужу")["target"].lower(), R.edit_plan("убери лужу"))
check("fill white", R.edit_plan("remove the background, make it white")["fill"] == "white")
check("fill transparent", R.edit_plan("вырежи фон")["fill"] == "transparent")
for t, want in [("the same, in landscape orientation for YouTube", True), ("сделай горизонтально", True),
                ("make it 16:9", True),
                ("winter scene, snowy landscape visible through the window, a portrait of a cat", False)]:
    check(("format: " if want else "not a format: ") + t, R.edit_plan(t)["reformats"] is want, R.edit_plan(t))
for t, want in [("fix the head proportions", True), ("голова слишком большая", True), ("увеличь кота", True),
                ("make the shirt blue", False)]:
    check(("resize: " if want else "not a resize: ") + t, R.edit_plan(t)["resizes"] is want, R.edit_plan(t))
import intent
for t, want in [("нет, верни как было", True), ("откати картинку", True), ("undo", True),
                ("верни собаку в человека", False), ("сделай ярче", False)]:
    check(("undo: " if want else "not undo: ") + t, intent.read(None, t)["undo"] is want)
for ph, facial, head in [("lips", True, True), ("губы", True, True), ("hair", False, True),
                         ("red shirt", False, False), ("beard", True, True)]:
    a = G._item_attributes(None, ph)
    check(f"face/head: {ph}", (a.get("facial"), a.get("head")) == (facial, head), a)
for ph, want in [("the area where the blue vase and flowers were located", True),
                 ("место, где стояла ваза", True), ("the blue vase", False), ("wooden table", False)]:
    check(f"former place: {ph}", G._item_attributes(None, ph).get("former_place") is want,
          G._item_attributes(None, ph))
print("ALL OK" if not bad else f"{bad} failed")
sys.exit(1 if bad else 0)
