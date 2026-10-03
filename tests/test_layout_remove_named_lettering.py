import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The model's reads are stubbed here; the phrases run live in bench/intent_rest_live.py.
import intent
intent.YES_STUB = lambda q, t: "centre" in q and any(w in t.lower() for w in ("центр", "middle", "center", "посередин"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sweep_stub  # noqa: F401  chains the centre stub above
import draw_agent as D

# «убери дату» on a poster was refused as "the critic's say-so" (live 2026-09-29)
assert D._instruction_removes("убери дату", "12 октября")
assert D._instruction_removes("удали заголовок", "РОК НОЧЬ")
assert D._instruction_removes("remove the date", "12 October")
assert not D._instruction_removes("убери собаку", "12 октября")
print("PASS removal names the kind of lettering")

import ideogram as G
_poster = G.normalize_layout({"background": "stage", "elements": [
    {"desc": "guitarist", "x": .55, "y": .2, "w": .4, "h": .7},
    {"desc": "drummer", "x": .05, "y": .2, "w": .4, "h": .7},
    {"desc": "title", "text": "РОК НОЧЬ", "x": 0, "y": 0, "w": 1, "h": .3}]})
# 13% on each head: clear pair by pair, 27% in sum (live 2026-09-29)
assert any(p["kind"] == "text_collision" for p in D.geometry_report(_poster))
_fixed, _ = D.auto_fix_geometry(_poster)
assert not D.geometry_report(_fixed), D.geometry_report(_fixed)
print("PASS title over two heads is shrunk clear")

import inspect
# 700 tokens: Gemma thought 8k chars, wrote no ops, "the model proposed no change"
assert "max_tokens=6000" in inspect.getsource(D.edit_layout)
print("PASS layout edit has room to think")
# a cat moved onto the dog kept "a small tabby cat": two overlapping boxes, no relation
assert "reword its description to say so" in D._EDIT_PROMPT
print("PASS on/in edits reword the description")

assert D._instruction_removes("Вику убери", "Вика")
assert D._instruction_removes("телефон сделай крупнее, а адрес убери", "ул. Ленина 5")
_bal = G.normalize_layout({"background": "sky", "elements": [
    {"desc": "a yellow balloon", "text": "Гоша", "x": .2, "y": .5, "w": .2, "h": .3}]})
_bal, _ = D.apply_ops(_bal, [{"op": "move", "target": 0, "x": .2, "y": .05},
                             {"op": "resize", "target": 0, "scale": 3}], instruction="в центр")
_e = _bal["elements"][0]
assert abs(_e["x"] + _e["w"] / 2 - .5) < .05, _e       # centred, not in the corner
print("PASS inflected name; move-then-grow keeps the move's corner")
_bk = G.normalize_layout({"background": "meadow", "elements": [
    {"desc": "a black cat", "x": .65, "y": .45, "w": .25, "h": .35}]})
_bk, _ = D.apply_ops(_bk, [{"op": "move", "target": 1, "x": .38, "y": .3},
                           {"op": "resize", "target": 1, "w": .38, "h": .52}], instruction="чёрный кот в центре и больше")
_e = _bk["elements"][0]
assert abs(_e["x"] + _e["w"] / 2 - .5) < .02, _e
print("PASS 'in the centre and bigger' stays centred whatever the move's x meant")
_jars = G.normalize_layout({"background": "shelf", "elements": [
    {"desc": "a glass jar of strawberry jam", "text": "STRAWBERRY", "x": .1, "y": .3, "w": .2, "h": .4},
    {"desc": "a glass jar of cherry jam", "text": "CHERRY", "x": .4, "y": .3, "w": .2, "h": .4}]})
assert len(D.apply_ops(_jars, [{"op": "delete", "target": 2}], instruction="убери среднюю банку")[0]["elements"]) == 1
assert len(D.apply_ops(_jars, [{"op": "delete", "target": 2}])[0]["elements"]) == 2
print("PASS the user removes a labelled jar; the critic still cannot")
_ln = G.normalize_layout({"background": "bags", "elements": [
    {"desc": "big headline", "text": "СКИДКИ ДО 70%", "x": .1, "y": .2, "w": .8, "h": .3},
    {"desc": "small line", "text": "ТОЛЬКО ДО ВОСКРЕСЕНЬЯ", "x": .15, "y": .55, "w": .7, "h": .12}]})
_ln, _ = D.apply_ops(_ln, [{"op": "swap_places", "a": 1, "b": 2}], instruction="поменяй строки местами")
_h, _s = _ln["elements"]
assert _s["y"] + _s["h"] <= _h["y"] + 1e-6 and abs(_h["y"] + _h["h"] - .67) < .01, _ln["elements"]
print("PASS swapped lines of different heights do not land on each other")
_win = G.normalize_layout({"background": "room", "elements": [
    {"desc": "a window with a wide windowsill", "x": .6, "y": .1, "w": .35, "h": .7},
    {"desc": "a cat on the floor", "x": .4, "y": .75, "w": .2, "h": .2}]})
_win, _ = D.apply_ops(_win, [{"op": "replace", "target": 2, "desc": "a cat on the windowsill"},
                             {"op": "move", "target": 2, "x": .65, "y": .2},
                             {"op": "delete", "target": 2}], instruction="кота посади на подоконник")
assert any("cat" in e["desc"] for e in _win["elements"]), _win["elements"]
print("PASS a moved cat is not deleted in the same answer")
_ts = G.normalize_layout({"background": "bridge", "elements": [
    {"desc": "the text on the t-shirt", "text": "Я ЛЮБЛЮ ПИТЕР", "x": .35, "y": .4, "w": .3, "h": .1}]})
_, _n = D.apply_ops(_ts, [{"op": "replace", "target": 1, "desc": "the text on the t-shirt", "text": "ПИТЕР ЛУЧШЕ"}],
                    instruction="надпись замени на ПИТЕР ЛУЧШЕ")
assert _n == ["lettering on “the text on the t-shirt” -> “ПИТЕР ЛУЧШЕ”"], _n
print("PASS a new string reads as a lettering change, not 'replaced X with X'")
_dup = G.normalize_layout({"background": "card", "elements": [
    {"desc": "a festive sign", "text": "С НОВЫМ ГОДОМ", "x": .2, "y": .05, "w": .3, "h": .15}]})
_dup, _ = D.apply_ops(_dup, [{"op": "resize", "target": 1, "scale": 2}] * 2, instruction="надпись в два раза больше")
assert abs(_dup["elements"][0]["h"] - .3) < .01, _dup["elements"]
print("PASS the same op twice in one answer applies once")
assert D._instruction_removes("убери второй шаг", "засыпать заварку", "Text for step two: 'засыпать заварку'")
assert not D._instruction_removes("убери второй шаг", "нагреть воду", "Text for step one: 'нагреть воду'")
assert D._instruction_removes("убери второй шаг", "2. Засыпать заварку", "a sign board for the second step")
assert D._instruction_removes("убери второй шаг", "2. Засыпать заварку", "a sign board")
print("PASS 'убери второй шаг' takes the step's label too")
assert "written in their language" in D._EDIT_PROMPT
print("PASS made-up lettering follows the instruction language")

_five = G.normalize_layout({"background": "sky", "elements": [
    {"desc": "a green balloon", "text": "Вика", "x": .7, "y": .1, "w": .2, "h": .3},
    {"desc": "a red balloon", "text": "Аня", "x": .1, "y": .1, "w": .2, "h": .3}]})
_five, _n = D.apply_ops(_five, [{"op": "delete", "target": 0}], instruction="Вику убери")
assert [e["text"] for e in _five["elements"]] == ["Аня"], _n
print("PASS a balloon named by its lettering is deleted")

_st = G.normalize_layout({"background": "street", "elements": [
    {"desc": "a woman", "x": .35, "y": .2, "w": .3, "h": .7},
    {"desc": "a bicycle", "x": .05, "y": .6, "w": .2, "h": .3},
    {"desc": "a lamppost", "x": .8, "y": .1, "w": .1, "h": .8}]})
_st, _n = D.apply_ops(_st, [{"op": "delete", "target": 2}, {"op": "delete", "target": 3}],
                      instruction="убери всё кроме девушки")
assert [e["desc"] for e in _st["elements"]] == ["a woman"], _n
print("PASS numbers mean the numbering the model saw")

_an = G.normalize_layout({"background": "street", "photo": "standard lens, wide aperture",
                          "elements": [{"desc": "a woman", "x": .3, "y": .2, "w": .3, "h": .7}]})
_an, _ = D.apply_ops(_an, [{"op": "style", "medium": "digital illustration",
                            "aesthetics": "anime"}], instruction="в стиле аниме")
assert not _an.get("photo"), _an.get("photo")
print("PASS a drawn medium drops the lens")

_card = G.normalize_layout({"background": "night", "elements": [
    {"desc": "tree", "x": .15, "y": .2, "w": .35, "h": .7},
    {"desc": "title", "text": "Happy New Year!", "x": .1, "y": .05, "w": .8, "h": .15},
    {"desc": "sign", "text": "from the Smiths", "x": .2, "y": .83, "w": .6, "h": .12},
    {"desc": "big year", "text": "2027", "x": .15, "y": .05, "w": .36, "h": .15}]})
_card, _ = D.auto_fix_geometry(_card)
_t = [e for e in _card["elements"] if e.get("text")]
from draw_geometry import _inter
assert all(_inter(a, b) < 1e-6 for k, a in enumerate(_t) for b in _t[k + 1:]), _t
print("PASS lettering never lands on lettering")

_sw = G.normalize_layout({"background": "kitchen", "elements": [
    {"desc": "a glass vase with apples", "x": .25, "y": .35, "w": .25, "h": .3},
    {"desc": "a ginger cat sitting to the right of the vase", "x": .55, "y": .3, "w": .3, "h": .35}]})
_sw, _ = D.apply_ops(_sw, [{"op": "move", "target": 1, "x": .55, "y": .3},
                           {"op": "move", "target": 2, "x": .2, "y": .3}], instruction="поменяй местами")
assert "to the left of the vase" in _sw["elements"][1]["desc"], _sw["elements"][1]
print("PASS side words follow the boxes")
_tb = G.normalize_layout({"background": "kitchen", "elements": [
    {"desc": "a wooden table", "x": .1, "y": .5, "w": .8, "h": .4},
    {"desc": "a dog sitting on the right side of the table", "x": .7, "y": .6, "w": .25, "h": .3}]})
_tb, _ = D.apply_ops(_tb, [{"op": "move", "target": 2, "x": .05, "y": .6}], instruction="собаку влево")
assert "on the left side of the table" in _tb["elements"][1]["desc"], _tb["elements"][1]
print("PASS 'on the right side of' follows the box too")

_cats = G.normalize_layout({"background": "garden", "elements": [
    {"desc": "a grey cat", "x": .05, "y": .5, "w": .2, "h": .3},
    {"desc": "a grey cat", "x": .4, "y": .5, "w": .2, "h": .3},
    {"desc": "a grey cat", "x": .75, "y": .5, "w": .2, "h": .3}]})
_c1, _ = D.apply_ops(G.normalize_layout(dict(_cats)), [{"op": "replace", "target": "a grey cat", "desc": "a ginger cat"}],
                     instruction="кота справа сделай рыжим")
assert [e["desc"] for e in _c1["elements"]] == ["a grey cat", "a grey cat", "a ginger cat"], _c1
_c2, _ = D.apply_ops(G.normalize_layout(dict(_cats)), [{"op": "delete", "target": "a grey cat"}],
                     instruction="среднего кота убери")
assert [e["x"] for e in _c2["elements"]] == [.05, .75], _c2
print("PASS identical twins are told apart by side words")
_c3, _ = D.apply_ops(G.normalize_layout(dict(_cats)), [{"op": "move", "target": "a grey cat", "x": .6}],
                     instruction="move the left cat to the right")
assert sorted(e["x"] for e in _c3["elements"]) == [.4, .6, .75], _c3
print("PASS the earliest side word names the element")
_c4, _ = D.apply_ops(G.normalize_layout(dict(_cats)), [{"op": "delete", "target": "the middle cat"}],
                     instruction="среднего кота убери")
assert [e["x"] for e in _c4["elements"]] == [.05, .75], _c4
print("PASS 'the middle cat' resolves")
_poster = {"elements": [{"desc": "elegant lettering on the top of the frame", "text": "Утро",
                         "x": .2, "y": .1, "w": .6, "h": .15}], "background": "cafe"}
_p2, _ = D.apply_ops(G.normalize_layout(_poster), [{"op": "move", "target": 0, "x": .2, "y": .75}],
                     instruction="надпись перенеси вниз")
assert "bottom of the frame" in _p2["elements"][0]["desc"], _p2
print("PASS a moved element's frame-position words follow its box")
_bub = {"elements": [{"desc": "a dog", "x": .1, "y": .3, "w": .4, "h": .6},
                     {"desc": "a parrot", "x": .5, "y": .3, "w": .4, "h": .6},
                     {"desc": "a white speech bubble", "x": .1, "y": .05, "w": .4, "h": .25},
                     {"desc": "words in the bubble", "text": "Мой!", "x": .15, "y": .1, "w": .3, "h": .1}],
        "background": "room"}
_b2, _ = D.apply_ops(G.normalize_layout(_bub), [{"op": "move", "target": "the speech bubble", "x": .5, "y": .05}],
                     instruction="облачко перенеси к попугаю")
assert abs(_b2["elements"][3]["x"] - .55) < 1e-6, _b2
assert _b2["elements"][0]["x"] == .1, _b2
print("PASS what sits inside a moved box travels with it")
_pair = {"elements": [{"desc": "an angry dog looking towards the cat", "x": .1, "y": .3, "w": .4, "h": .6},
                      {"desc": "a grumpy tabby cat with arched back facing the dog", "x": .5, "y": .3, "w": .4, "h": .6},
                      {"desc": "a ball next to the red car", "x": .4, "y": .8, "w": .1, "h": .1},
                      {"desc": "a red car", "x": .0, "y": .8, "w": .3, "h": .2}], "background": "room"}
_q, _ = D.apply_ops(G.normalize_layout(_pair), [{"op": "replace", "target": 2, "desc": "a colorful parrot facing the dog"},
                                                {"op": "replace", "target": 4, "desc": "a blue car"}],
                    instruction="замени кота на попугая, машину сделай синей")
assert _q["elements"][0]["desc"] == "an angry dog looking towards the parrot", _q
assert _q["elements"][2]["desc"] == "a ball next to the red car", _q
print("PASS other boxes follow a replaced subject's new name")
_row = {"elements": [{"desc": "an ice floe", "x": .1, "y": .6, "w": .8, "h": .4},
                     {"desc": "a tiny penguin", "x": .2, "y": .55, "w": .1, "h": .25},
                     {"desc": "an emperor penguin", "x": .75, "y": .4, "w": .2, "h": .4}], "background": "sea"}
_r, _ = D.apply_ops(G.normalize_layout(_row), [{"op": "swap_places", "a": 1, "b": 3}], instruction="поменяй местами крайних")
_t, _e = _r["elements"][1], _r["elements"][2]
assert (_t["desc"], _e["desc"]) == ("a tiny penguin", "an emperor penguin"), _r
assert abs(_t["x"] + _t["w"] / 2 - .85) < 1e-6 and abs(_e["x"] + _e["w"] / 2 - .25) < 1e-6, _r
assert abs(_t["y"] + _t["h"] - .8) < 1e-6 and abs(_e["y"] + _e["h"] - .8) < 1e-6, _r
assert _r["elements"][0]["x"] == .1
print("PASS swap_places exchanges centres and feet, keeps sizes and words")
_wc = G.normalize_layout({"elements": [{"desc": "a girl", "x": .3, "y": .3, "w": .3, "h": .5}], "background": "Paris",
                          "aesthetics": "watercolor painting, soft edges, bleeding colors, moody", "medium": "watercolor"})
_w2, _ = D.apply_ops(_wc, [{"op": "style", "medium": "photography", "lighting": "night"}], instruction="обратно как фото, ночью")
assert _w2["aesthetics"] == "soft edges, moody", _w2["aesthetics"]
print("PASS a new medium drops the old medium's aesthetics")
_man = {"elements": [{"desc": "a young man in a white t-shirt", "x": .2, "y": .15, "w": .6, "h": .85},
                     {"desc": "lettering on the t-shirt", "text": "КОД НЕ ТРОГАТЬ", "x": .3, "y": .4, "w": .4, "h": .12},
                     {"desc": "a chalkboard", "x": 0, "y": 0, "w": 1, "h": 1}], "background": "class"}
_m2, _ = D.apply_ops(G.normalize_layout(_man), [{"op": "move", "target": 1, "x": .4}], instruction="парня передвинь вправо")
assert abs(_m2["elements"][1]["x"] - .5) < 1e-6 and _m2["elements"][2]["x"] == 0, _m2
print("PASS a big subject carries its lettering; the backdrop does not move")
_m3, _ = D.apply_ops(G.normalize_layout(_man), [{"op": "resize", "target": 1, "scale": 0.5}], instruction="парня сделай в два раза меньше")
_g, _l = _m3["elements"][0], _m3["elements"][1]
assert abs(_g["y"] + _g["h"] - 1.0) < 1e-6, _g
assert abs(_l["w"] - .2) < 1e-6 and _g["x"] <= _l["x"] and _l["x"] + _l["w"] <= _g["x"] + _g["w"], _m3
print("PASS a shrunk subject keeps its feet and its lettering shrinks with it")
assert "one text element PER LINE" in G._PLANNER_PROMPT
print("PASS the planner is told unquoted dictated lines are lettering")
