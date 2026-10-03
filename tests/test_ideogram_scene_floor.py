"""A layout with no high_level_description must still say it is ONE picture.

Measured at one seed on the worst render of a 14-scene run: a black cat on the
floor, a ginger cat in an armchair, a wooden table, over the one-word background
"комната". It came back as a 2x2 grid of four separate stock photographs, two of
which contained invented people.

The thin background was not the cause -- the same layout rendered with a rich
background produced the SAME 2x2 collage. The missing high_level_description
was: every element names its own surface, nothing says they share a frame, and
the model resolves three settings as three pictures. Supplying one synthesized
sentence, changing nothing else, produced a single coherent room with all three.

So this suite pins the floor: the sentence exists, it names every element, it is
written in the layout's own language, and it never overwrites one the planner
actually wrote.

Offline: no renderer, no LLM, no GPU.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ideogram_layout as L  # noqa: E402

OK = BAD = 0


def check(name, cond, detail=""):
    global OK, BAD
    if cond:
        OK += 1
        print("PASS  " + name)
    else:
        BAD += 1
        print("FAIL  " + name + ((": " + str(detail)) if detail else ""))
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise AssertionError(str(name) + ((": " + str(detail)) if detail else ""))


RU = {"background": "комната", "elements": [
    {"desc": "чёрный кот сидит на полу", "x": .10, "y": .60, "w": .22, "h": .28},
    {"desc": "рыжий кот спит на кресле", "x": .60, "y": .55, "w": .26, "h": .30},
    {"desc": "деревянный стол", "x": .30, "y": .20, "w": .34, "h": .28}]}

EN = {"background": "a room", "elements": [
    {"desc": "a black cat on the floor", "x": .10, "y": .60, "w": .22, "h": .28},
    {"desc": "a ginger cat asleep in an armchair", "x": .60, "y": .55, "w": .26,
     "h": .30}]}

print("=" * 66)
print("THE COLLAGE LAYOUT GETS A SCENE SENTENCE")
print("=" * 66)

cap = L.layout_to_caption(dict(RU))
hl = cap.get("high_level_description", "")
check("a layout with no high_level gets one", bool(hl.strip()), hl)
check("it says the elements share ONE frame", "одном кадре" in hl, hl)
# Measured: "Одно изображение" left the 2x2 collage standing AND added a
# transparency checkerboard under it. The noun has to be ФОТОГРАФИЯ.
check("it says PHOTOGRAPH, the word that actually held the frame together",
      "фотография" in hl.lower(), hl)
for el in RU["elements"]:
    check("it names %r" % el["desc"][:22], el["desc"] in hl, hl)
check("it names the background too", "комната" in hl, hl)
check("it is written in the layout's language",
      not any("a" <= ch <= "z" for ch in hl.lower() if ch.isascii() and ch.isalpha()),
      hl)

en_hl = L.layout_to_caption(dict(EN)).get("high_level_description", "")
check("an English layout gets an English sentence",
      "SAME" in en_hl and "black cat" in en_hl, en_hl)
check("and calls itself a photograph too", "photograph" in en_hl.lower(), en_hl)

art = L.layout_to_caption(dict(RU, art_style="watercolour"))["high_level_description"]
check("a drawing is not called a photograph", "фотогра" not in art.lower(), art)
check("but it is still ONE picture", "одном кадре" in art, art)

print()
print("=" * 66)
print("THE FLOOR IS A FLOOR, NOT AN OVERRIDE")
print("=" * 66)

mine = dict(RU, high_level_description="Моя своя фраза.")
check("a high_level the planner wrote is never replaced",
      L.layout_to_caption(mine)["high_level_description"] == "Моя своя фраза.",
      L.layout_to_caption(mine)["high_level_description"])

blank = dict(RU, high_level_description="   ")
check("whitespace does not count as a description",
      L.layout_to_caption(blank)["high_level_description"].strip() != "",
      L.layout_to_caption(blank)["high_level_description"])

check("a layout with no elements gets no invented sentence",
      L.synth_high_level({"background": "комната", "elements": []}) == "")
check("an element with no background still yields a sentence",
      L.synth_high_level({"background": "", "elements": [{"desc": "кот"}]}) != "")

# The caption must stay the shape the renderer's verifier expects.
check("high_level_description stays the FIRST key",
      list(cap)[0] == "high_level_description", list(cap))
check("the scene itself is untouched",
      len(cap["compositional_deconstruction"]["elements"]) == 3,
      cap["compositional_deconstruction"])

print()
print("%d passed, %d failed" % (OK, BAD))
sys.exit(1 if BAD else 0)
