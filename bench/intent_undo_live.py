"""Live check of intent.undo on the house model: the previous FILE comes back only when
nothing new is asked. «верни обратно как фото, но оставь вечер и дождь» read as undo and
handed back the first picture, without the rain (debug chat 2026-10-25).

    venv/Scripts/python bench/intent_undo_live.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "agent"), os.path.join(ROOT, "core")]
sys.stdout.reconfigure(encoding="utf-8")
os.environ.pop("F5_TEST_RUN", None)
import intent

CASES = [
    ("верни как было", True),
    ("отмени последнее изменение", True),
    ("нет, верни предыдущую картинку", True),
    ("undo", True),
    ("put it back the way it was", True),
    ("не, прошлый вариант был лучше, верни его", True),
    ("верни обратно как фото, но оставь вечер и дождь", False),
    ("сделай обратно фото, только пусть будет ночь", False),
    ("верни коту рыжий цвет", False),
    ("верни ему очки", False),
    ("make it a photo again but keep the rain", False),
    ("верни как было, а потом добавь шляпу", False),
    ("сделай реалистично", False),
    ("ещё раз", False),
]
bad = 0
for text, want in CASES:
    got = intent.read(None, text)["undo"]
    bad += got != want
    print(("PASS " if got == want else "FAIL ") + f"{text!r} -> undo={got}")
print(f"{len(CASES) - bad}/{len(CASES)}")
sys.exit(1 if bad else 0)
