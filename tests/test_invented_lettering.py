"""Lettering nobody asked for is not drawn: «Дедушка какает в кулачок» got a sign «ОЙ-ОЙ!» from the
planner, drawn «ОМРОM!» and «OЙ-O!», and the picture went out with an apology (live 2026-10-21)."""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "core", "imaging"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("F5_TEST_RUN", "1")

import ideogram  # noqa: E402

PROMPT = "A humorous cartoon of a funny old man pooping into his clenched fist."


def _layout():
    return {"elements": [{"desc": "an old man on a stool", "x": 0.2, "y": 0.2, "w": 0.6, "h": 0.7},
                         {"desc": "a speech bubble with bold letters", "text": "ОЙ-ОЙ!",
                          "x": 0.6, "y": 0.05, "w": 0.3, "h": 0.15}]}


def test_a_made_up_sign_goes_when_no_words_were_asked_for():
    ctx = types.SimpleNamespace(yes_no=lambda q, text, **k: False)
    out = ideogram.drop_invented_text(_layout(), PROMPT, ctx)
    assert [el["desc"] for el in out["elements"]] == ["an old man on a stool"]


def test_made_up_lettering_stays_when_the_user_asked_for_a_caption():
    ctx = types.SimpleNamespace(yes_no=lambda q, text, **k: True)
    out = ideogram.drop_invented_text(_layout(), PROMPT + " Add a funny caption.", ctx)
    assert any(el.get("text") == "ОЙ-ОЙ!" for el in out["elements"])


def test_lettering_the_user_wrote_is_never_questioned():
    asked = []
    ctx = types.SimpleNamespace(yes_no=lambda q, text, **k: asked.append(q) or False)
    out = ideogram.drop_invented_text(_layout(), PROMPT + ' A sign reads "ОЙ-ОЙ!".', ctx)
    assert len(out["elements"]) == 2 and not asked
