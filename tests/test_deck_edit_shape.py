import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import slides

assert slides._clean_chart({"title": "Даты гибели князя", "unit": "год", "labels": ["a", "b"], "values": [972, 973]}) == {}
assert slides._clean_chart({"title": "Население", "unit": "тыс.", "labels": ["a", "b"], "values": [1200, 1500]})
assert slides._clean_chart({"title": "Хронология первых укреплений", "labels": ["1156", "XII век"], "values": [1156, 1100]}) == {}
assert slides._clean_chart({"title": "Население по годам", "unit": "тыс.", "labels": ["1900", "2000"], "values": [1200, 1500]})
print("PASS years are not a chart")

# «поменяй местами 2 и 3 слайд»: a web search for that phrase added sources,
# the closing slide was re-counted into the first plan's "5", a content slide was cut.
slides.gather_facts = lambda ctx, q: (_ for _ in ()).throw(AssertionError("searched for a reorder"))
_prev = {"title": "T", "slides": [{"heading": f"S{i}", "bullets": ["b"]} for i in range(4)], "sources": []}
import json, llm
_swapped = dict(_prev, slides=[_prev["slides"][i] for i in (0, 2, 1, 3)],
                sources=[{"title": "x", "url": "https://x.org"}])
llm.call_llm_simple = lambda *a, **k: json.dumps(_swapped)
d = slides.plan_deck(None, "поменяй местами 2 и 3 слайд", 5, "ru", previous=_prev)
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
assert slides._own_data(_t) and not slides._own_data("история Кремля с 1156 по 1495 год, 3 этапа")
_d = {"slides": [{"chart": {"labels": ["Q1", "Q2", "Q3", "Q4"], "values": [1.2, 1.5, 0.9, 2.1]}}]}
slides._drop_ungrounded_charts(_d, "" + " " + _t)
assert _d["slides"][0]["chart"], _d
print("PASS the user's own figures skip the web and ground their chart")
slides.gather_facts = lambda ctx, q: ("", [])
llm.call_llm_simple = lambda *a, **k: json.dumps(dict(_prev, theme="dark_green_accent"))
_th = slides.plan_deck(None, "сделай тёмную тему с зелёным акцентом", 0, "ru", previous=dict(_prev, theme="nature"))
assert _th["theme"] == "nature", _th["theme"]
assert "nature (dark green, green accent)" in slides._EDIT_PROMPT
print("PASS an invented theme name does not replace a real one")

llm.call_llm_simple = lambda *a, **k: json.dumps(_prev)
_four = dict(_prev, slides=[{"heading": h, "bullets": ["x"]} for h in "ABCD"])
assert [s["heading"] for s in slides.plan_deck(None, "удали последний слайд", 0, "ru", previous=_four)["slides"]] == list("ABC")
assert [s["heading"] for s in slides.plan_deck(None, "удали 2-й слайд", 0, "ru", previous=_four)["slides"]] == list("ACD")
print("PASS a bare delete by position removes that slide")

import tools
import intent   # the model's read of the words (agent/intent.py); phrases: bench/intent_lang_live.py
intent.STUB = {"сделай презентацию на английском про Марс": {"reply_language": "en"}}.get
assert tools._lang_of_original({"user_input_original": "сделай презентацию на английском про Марс"}, "") == "en"
assert tools._lang_of_original({"user_input_original": "сделай презентацию про Марс"}, "") == "ru"
print("PASS a deck asked for in English is written in English")
