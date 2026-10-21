"""Live: graph_personality._offers_instead -- a reply that asks leave to do what was asked
(an offer) vs one that asks for a detail the request lacks. An offer ended a picture turn with
nothing made: «Хочешь, чтобы я попробовал нарисовать эту битву заново?» (live 2026-10-21).

    venv/Scripts/python bench/offers_instead_live.py
"""
import os, sys, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [ROOT] + [os.path.join(ROOT, d) for d in ("agent", "core", "bot", "imaging", "media", "knowledge")] \
    + [p for p in sys.path if os.path.basename(p) != "bench"]
sys.stdout.reconfigure(encoding="utf-8")
os.environ.pop("F5_TEST_RUN", None)
import graph_personality as gp

CASES = [
    ("edit the image: Ленин в кибернетическом костюме попадает в ад и бьётся с чертями в лаве",
     "Слушай, задача эпическая! Но inpaint_image слишком слаб для такой сцены. Мне нужно создать новую "
     "картинку с нуля через generate_image. Хочешь, чтобы я попробовал нарисовать эту битву заново?", True),
    ("нарисуй кота-космонавта на луне",
     "Отличная идея! Могу нарисовать его в реалистичном стиле. Нарисовать?", True),
    ("сделай фон закатным", "Понял. Сделать фон закатным прямо сейчас?", True),
    ("оживи фото: пусть он помашет рукой",
     "Это можно сделать через генерацию видео, займёт пару минут. Запускать?", True),
    ("убери с фото человека", "На фото три человека. Кого из них убрать: слева, по центру или справа?", False),
    ("нарисуй мой логотип", "Как называется компания и какие цвета использовать в логотипе?", False),
    ("сними видео", "Какой длины сделать ролик?", False),
    ("сделай открытку на день рождения", "Для кого открытка и как его зовут, чтобы подписать?", False),
]
bad = 0
ctx = types.SimpleNamespace()
for req, reply, want in CASES:
    got = gp._offers_instead(ctx, req, reply)
    bad += got is not want
    print(("PASS " if got is want else "FAIL ") + f"{'offer' if want else 'question'}: {reply[-60:]!r} -> {got}")
print(f"\n{bad} failed")
