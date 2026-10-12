"""A deck's web pictures are checked against their slides in ONE vision call and the
ones that do not fit are dropped (2026-10-13): searched one at a time, a picture was
never looked at, and a stranger's photo could illustrate a slide.

Run: venv/Scripts/python.exe tests/test_deck_picture_check.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "services"), os.path.join(ROOT, "agent"),
                os.path.join(ROOT, "core")]
import slides
import llm

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    OK, BAD = (OK + 1, BAD) if cond else (OK, BAD + 1)
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"   {extra}"))


deck = {"title": "Чай", "slides": [{"title": "История"}, {"title": "Сорта"}, {"title": "Цены"}]}
CALLS = []
llm.analyze_image_with_llm = lambda *a, **k: (CALLS.append(k), "2")[1]
images = {"cover": "c.jpg", 0: "a.jpg", 1: "b.jpg", 2: "d.jpg"}
slides._drop_mismatched_images(object(), deck, images)
check("the slide pictures go in one call, each labelled with its slide",
      len(CALLS) == 1 and [lab for lab, _ in CALLS[0]["image_paths"]][1].startswith("Slide 2 -- Сорта"), CALLS)
check("...the one named as not fitting is dropped, the rest and the cover kept",
      images == {"cover": "c.jpg", 0: "a.jpg", 2: "d.jpg"}, images)

llm.analyze_image_with_llm = lambda *a, **k: "NONE"
images = {0: "a.jpg", 1: "b.jpg"}
slides._drop_mismatched_images(object(), deck, images)
check("NONE keeps every picture", images == {0: "a.jpg", 1: "b.jpg"}, images)

print(f"\n{OK}/{OK + BAD} passed")
sys.exit(1 if BAD else 0)
