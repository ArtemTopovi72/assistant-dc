import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import slides
import types as _types
_T = [None]   # the text model the context hands out
_G = [lambda ctx, q: ("", [])]   # the web search it hands out
_S = [None]
# What a request is (own figures / reorder-delete / content) is the model's read, scored live
# in bench/deck_read_eval.py; scripted here, the decisions made on it are checked.
_READS = {"поменяй местами 2 и 3 слайд": {"kind": "structural", "drop": []},
          "удали последний слайд": {"kind": "structural", "drop": [-1]},
          "удали 2-й слайд": {"kind": "structural", "drop": [2]}}
TXT = _types.SimpleNamespace(text_model=lambda *a, **k: _T[0](*a, **k),
                             deps=_types.SimpleNamespace(gather_facts=lambda *a, **k: _G[0](*a, **k),
                                                         support=lambda *a, **k: _S[0](*a, **k),
                                                         deck_read=lambda ctx, t: _READS.get(
                                                             t, {"kind": "content", "drop": []})))


assert slides._clean_chart({"title": "Даты гибели князя", "unit": "год", "labels": ["a", "b"], "values": [972, 973]}) == {}
assert slides._clean_chart({"title": "Население", "unit": "тыс.", "labels": ["a", "b"], "values": [1200, 1500]})
assert slides._clean_chart({"title": "Хронология первых укреплений", "labels": ["1156", "XII век"], "values": [1156, 1100]}) == {}
assert slides._clean_chart({"title": "Население по годам", "unit": "тыс.", "labels": ["1900", "2000"], "values": [1200, 1500]})
print("PASS years are not a chart")

# «поменяй местами 2 и 3 слайд»: a web search for that phrase added sources,
# the closing slide was re-counted into the first plan's "5", a content slide was cut.
_G[0] = lambda ctx, q: (_ for _ in ()).throw(AssertionError("searched for a reorder"))
_prev = {"title": "T", "slides": [{"heading": f"S{i}", "bullets": ["b"]} for i in range(4)], "sources": []}
import json, llm
_swapped = dict(_prev, slides=[_prev["slides"][i] for i in (0, 2, 1, 3)],
                sources=[{"title": "x", "url": "https://x.org"}])
_T[0] = lambda *a, **k: json.dumps(_swapped)
d = slides.plan_deck(TXT, "поменяй местами 2 и 3 слайд", 5, "ru", previous=_prev)
assert [s["heading"] for s in d["slides"]] == ["S0", "S2", "S1", "S3"], d["slides"]
print("PASS a reorder keeps every slide and searches nothing")

_deck = {"slides": [
    {"heading": "a", "chart": {"title": "охват", "values": [100.0, 100.0, 100.0]}},
    {"heading": "b", "chart": {"title": "торговля", "values": [40.0, 30.0, 30.0]}},
    {"heading": "c", "chart": {"title": "население", "labels": ["1900 год", "2025 год"], "values": [1200.0, 1329825.0]}},
    {"heading": "d", "chart": {"title": "охват", "labels": ["Ирландия", "Русь"], "values": [10.0, 9.0]}}]}
slides._drop_ungrounded_charts(_deck, "В 2025 году — 1 329 825 жителей, в 1900 году было 1200. "
                                      "Викинги жили в X веке, 9 кораблей.")
assert [bool(s["chart"]) for s in _deck["slides"]] == [False, False, True, False], _deck
print("PASS charts without data behind them are dropped")
_t = "итоги года нашей кофейни: выручка по кварталам Q1 1,2 млн, Q2 1,5 млн, Q3 0,9 млн, Q4 2,1 млн"
assert slides._dropped_slide(_prev, {"drop": [1, 2, 3, 4]}) is None, "a delete never empties the deck"
_d = {"slides": [{"chart": {"labels": ["Q1", "Q2", "Q3", "Q4"], "values": [1.2, 1.5, 0.9, 2.1]}}]}
slides._drop_ungrounded_charts(_d, "" + " " + _t)
assert _d["slides"][0]["chart"], _d
print("PASS the user's own figures skip the web and ground their chart")
_G[0] = lambda ctx, q: ("", [])
_T[0] = lambda *a, **k: json.dumps(dict(_prev, theme="dark_green_accent"))
_th = slides.plan_deck(TXT, "сделай тёмную тему с зелёным акцентом", 0, "ru", previous=dict(_prev, theme="nature"))
assert _th["theme"] == "nature", _th["theme"]
assert "nature (dark green, green accent)" in slides._EDIT_PROMPT
print("PASS an invented theme name does not replace a real one")

_T[0] = lambda *a, **k: json.dumps(_prev)
_four = dict(_prev, slides=[{"heading": h, "bullets": ["x"]} for h in "ABCD"])
assert [s["heading"] for s in slides.plan_deck(TXT, "удали последний слайд", 0, "ru", previous=_four)["slides"]] == list("ABC")
assert [s["heading"] for s in slides.plan_deck(TXT, "удали 2-й слайд", 0, "ru", previous=_four)["slides"]] == list("ACD")
print("PASS a bare delete by position removes that slide")

import tools
import types
from fakes import reads   # the model's read of the words (agent/intent.py); phrases: bench/intent_lang_live.py
_R = types.SimpleNamespace(**reads(read={"сделай презентацию на английском про Марс": {"reply_language": "en"}}.get))
assert tools._lang_of_original({"user_input_original": "сделай презентацию на английском про Марс"}, "", ctx=_R) == "en"
assert tools._lang_of_original({"user_input_original": "сделай презентацию про Марс"}, "", ctx=_R) == "ru"
print("PASS a deck asked for in English is written in English")
