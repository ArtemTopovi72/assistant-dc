"""The first-round forced tool comes from the model's read (intent.must_call).

The phrases run against the real model in bench/intent_force_live.py. Here: what
the loop does with a read, and what the read is given.
"""
import os, sys
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import inspect
import intent
import graph_personality as G
from types import SimpleNamespace as NS
from fakes import reads

ctx = NS(last_image_path=None)
TOOLS = [{"function": {"name": n}} for n in ("calculate", "search", "create_presentation")]
seen = []


def stub(t):
    seen.append(t)
    return {"needs_tool": True, "must_call": {"сделай презентацию": "create_presentation",
                                              "раздели чек на троих": "calculate",
                                              "найди фото": "find_photo"}.get(t, "")}


ctx.intent_read = reads(read=stub)["intent_read"]
f = lambda t, called=(), tools=TOOLS: G._forced_intent(ctx, {"messages": []}, t, False, tools, called)
assert f("сделай презентацию") == "create_presentation"
assert f("раздели чек на троих") == "calculate"
assert f("привет") == ""
assert f("сделай презентацию", {"create_presentation"}) == ""          # already ran this turn
assert f("найди фото") == ""                                            # not on offer: never force an absent schema
assert f("") == ""
print("ok must_call is forced only when offered and not yet called")

# The read gets what came with the message: a photo, a pasted table.
got = {}
told = lambda c, t, prev="", att="": got.setdefault("att", att) and intent.FALLBACK
G._turn_intent(NS(last_image_path="x.png", intent_read=told), {"messages": []}, "сколько я потратил?")
assert got["att"] == "a photo", got
got.clear()
csv = [{"role": "user", "content": "Jan;125400;98300\nFeb;131200;101500\nMar;118900;97800\nApr;142300;110200"}]
ctx.intent_read = told
G._turn_intent(ctx, {"messages": csv}, "total profit?")
assert "numbers" in got["att"], got
print("ok the read is told about photos and data")

# «Сколько стоит чай» under a forwarded кружок with a receipt went to ozon_search and the
# answer guessed 12 €: the read saw the words and nothing of what they were about (2026-10-24).
from prompt_guard import wrap_quoted
board = ("Итог: чаепитие в кафе.\n" + "0:05 — кадр\n" * 40 +
         "\nТекст документа в видео (0:13), дословно:\nCAJ MINT 2,000 6,50 13,00\nUKUPNO 21,00")
got.clear()
G._turn_intent(ctx, {"messages": []}, wrap_quoted("the user forwarded this", "Александр: " + board) + "\nСколько стоит чай")
assert "forwarded" in got["att"] and "чаепитие" in got["att"] and "CAJ MINT 2,000 6,50" in got["att"], got
assert len(got["att"]) < 900, len(got["att"])
print("ok the read is told what the forwarded material holds")

assert "or _forced_intent(" in inspect.getsource(G)
print("ok the loop consults it")

# The forced tool must be among the tools sent: «прочитай буфер и нарисуй» read
# wants=[generate_image], must_call=read_clipboard; retrieval dropped read_clipboard,
# the force had nothing to name and the model drew «the clipboard is empty» (2026-10-24).
import tool_retrieval as R
ALL = [{"function": {"name": n}} for n in ("search", "calculate", "remember_fact", "generate_image", "read_clipboard")]
src = inspect.getsource(G)
assert '*filter(None, [_read["must_call"]])' in src, "select_tools gets must_call with wants"
got = {s["function"]["name"] for s in R.select_tools("x", ALL, wants=["generate_image", "read_clipboard"])}
assert "read_clipboard" in got, got
print("ok the forced tool is always offered")

# A field the model got wrong takes its default; the rest of the read stands.
r = intent._validate({"song": "rap", "names_picture": 7, "is_question": True,
                      "weather": {"city": "Сочи", "when": "tomorrow"}})
assert r.song == "" and r.names_picture == 0 and r.is_question and r.weather.city == "Сочи", r
assert intent._validate(None) == intent.Intent()
print("ok a wrong field falls back alone")
