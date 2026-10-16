"""Decisions about what the user asked for read the USER's words only.

Example: a forwarded video's storyboard says «…текст сообщает о предыдущих
инцидентах…»; «предыдущих» must not read as «the previous picture», or
«Что думаешь по этому поводу?» targets an old quote photo. Quoted material is
framed by prompt_guard.wrap_quoted and removed by prompt_guard.user_words -- one
pair, used by the bot and the agent alike.
"""
import os, sys, inspect
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, d) for d in ("bot", "agent", "core")]
import types
import tg_bot, tg_resolve, graph, graph_finalize, graph_personality
sys.path.insert(0, os.path.join(ROOT, "tests"))
from fakes import reads
# The model's read (agent/intent.py) is scripted: «предыдущ…» anywhere names the
# previous picture -- so the fixture shows what the user_words cut is for.
_R = types.SimpleNamespace(**reads(read=lambda t: {"names_picture": -2} if "предыдущ" in t else None))
from prompt_guard import user_words, wrap_quoted, wrap_document

said = ("👁 Раскадровка видео:\n0:02 — На изображении мужчина по прозвищу «Фикаланджело».\n"
        "0:14 — Тот же кадр, текст сообщает о предыдущих инцидентах в других магазинах.")
turn = tg_bot._FWD_TEXT_FRAME.format(text=said) + "\nЧто думаешь по этому поводу?"
live = [{"id": i} for i in ("a1", "b2", "c3")]

assert tg_bot._ordinal_target(turn, live, ctx=_R), "fixture no longer reproduces the live miss"
own = user_words(turn)
assert own == "Что думаешь по этому поводу?", own
assert tg_bot._ordinal_target(own, live, ctx=_R) == ""
# the user's own words still name a picture
assert tg_bot._ordinal_target(user_words(tg_bot._FWD_TEXT_FRAME.format(text=said)
                                         + "\nчто на предыдущей картинке?"), live, ctx=_R) == "b2"
# a reply quote and an attached file are not the user's words either
assert user_words(wrap_quoted("the user is replying to this message", "первая картинка")
                  + "\nну как?") == "ну как?"
assert user_words("глянь " + wrap_document("a.txt", "вторая картинка")) == "глянь"
# nor what OCR read on the photo: replayed by ▶ Render, its «picture 1» once named the
# chat's first picture and the clip animated an old photo instead of the one just sent
from prompt_guard import OCR_NOTE_HEAD
_ocr = ("оживи фото: кот зевает\n" + OCR_NOTE_HEAD + " each «…» is a separate label: «Lenovo]». "
        "The lettering is part of what the picture says, so name it.]")
assert user_words(_ocr) == "оживи фото: кот зевает", user_words(_ocr)
assert OCR_NOTE_HEAD in inspect.getsource(tg_resolve) or "OCR_NOTE_HEAD" in inspect.getsource(tg_resolve)
# one implementation: every reader goes through it
assert "_ordinal_target(own_words" in inspect.getsource(tg_resolve)
assert "user_words(merged)" in inspect.getsource(tg_resolve)
for mod in (graph, graph_finalize, graph_personality):
    assert "FWD_QUOTE_RE" not in inspect.getsource(mod) and "_FWD_FRAME_RE" not in inspect.getsource(mod)
print("ok")
