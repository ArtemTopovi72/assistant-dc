"""Live check of intent.about_picture / is_question / image wants: is a message
about the picture or video on screen, an order to change it, or neither?
Phrases from the word lists it replaced (graph._ASKS_ABOUT_CURRENT_IMAGE_RE,
_PICTURE_EDIT_RE, _VIDEO_FOLLOWUP_RE) plus the live misses.

    venv/Scripts/python bench/intent_picture_live.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "agent"), os.path.join(ROOT, "core")]
sys.stdout.reconfigure(encoding="utf-8")
os.environ.pop("F5_TEST_RUN", None)
import intent
from graph_fastpath import _IMAGE_ACTION_TOOLS

LOOK, EDIT, NO = "look", "edit", "no"
CASES = [
    ("Что там написано?", "a photo", LOOK), ("what does it say", "a photo", LOOK),
    ("опиши картинку", "a photo", LOOK), ("что на фото", "a photo", LOOK),
    ("Что нарисовано?", "a photo", LOOK), ("какой товар самый дорогой?", "a photo", LOOK),
    ("кто это?", "a photo", LOOK), ("обсудим цитату", "a photo", LOOK),
    ("дай экспертное мнение", "a video", LOOK), ("заметил косяки?", "a video", LOOK),
    ("Что ответить?", "a video", LOOK), ("any flaws in the footage?", "a video", LOOK),
    ("убери лужу", "a photo", EDIT), ("сделай её ночной", "a photo", EDIT),
    ("можешь сделать ярче?", "a photo", EDIT), ("Расширь сцену", "a photo", EDIT),
    ("нарисуй кота", "a photo", EDIT), ("добавь ей очки", "a photo", EDIT),
    ("привет", "a photo", NO), ("какая погода в Москве", "a photo", NO),
    ("Скажи а", "a photo", NO), ("спасибо", "a photo", NO),
]
bad = 0
for text, att, want in CASES:
    r = intent.read(None, text, "", att)
    edit = bool(set(r["wants"]) & _IMAGE_ACTION_TOOLS)
    got = EDIT if edit else LOOK if r["about_picture"] else NO
    ok = got == want
    bad += not ok
    print(("PASS " if ok else "FAIL ") + f"{text!r} [{att}] -> {got} (want {want}) {r}" * 1)
print(f"{len(CASES) - bad}/{len(CASES)}")
sys.exit(1 if bad else 0)
