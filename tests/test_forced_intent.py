"""The first-round forced tool comes from the model's read (intent.must_call).

Replaces the keyword forcers (_forced_deck, _forced_photo_math,
_forced_place_search, _forced_aggregate, _forced_code_execution): their phrases
now run against the real model in bench/intent_force_live.py. Here: what the
loop does with a read, and what the read is given.
"""
import os, sys
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import inspect
import intent
import graph_personality as G
from types import SimpleNamespace as NS

ctx = NS(last_image_path=None)
TOOLS = [{"function": {"name": n}} for n in ("calculate", "search", "create_presentation")]
seen = []


def stub(t):
    seen.append(t)
    return {"needs_tool": True, "must_call": {"сделай презентацию": "create_presentation",
                                              "раздели чек на троих": "calculate",
                                              "найди фото": "find_photo"}.get(t, "")}


intent.STUB = stub
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
intent.STUB = None
orig = intent.read
intent.read = lambda c, t, prev="", att="": got.setdefault("att", att) and intent.FALLBACK
G._turn_intent(NS(last_image_path="x.png"), {"messages": []}, "сколько я потратил?")
assert got["att"] == "a photo", got
got.clear()
csv = [{"role": "user", "content": "Jan;125400;98300\nFeb;131200;101500\nMar;118900;97800\nApr;142300;110200"}]
G._turn_intent(ctx, {"messages": csv}, "total profit?")
assert "numbers" in got["att"], got
intent.read = orig
print("ok the read is told about photos and data")

assert "or _forced_intent(" in inspect.getsource(G)
print("ok the loop consults it")

# A field the model got wrong takes its default; the rest of the read stands.
r = intent._validate({"song": "rap", "names_picture": 7, "is_question": True,
                      "weather": {"city": "Сочи", "when": "tomorrow"}})
assert r.song == "" and r.names_picture == 0 and r.is_question and r.weather.city == "Сочи", r
assert intent._validate(None) == intent.Intent()
print("ok a wrong field falls back alone")
