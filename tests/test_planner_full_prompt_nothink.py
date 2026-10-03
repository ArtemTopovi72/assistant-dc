import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sweep_stub  # noqa: F401  the model's narrow reads, stubbed
import llm, ideogram as G

asked = []
llm.send_to_lm_studio = lambda ctx, msgs, **k: asked.append(msgs[0]["content"]) or {"content": ""}
class Ctx: model_name = "gemma4-26b-a4b"; no_think = True
G._ask_planner(Ctx(), "ваза на столе")
# no-think Gemma answers the full prompt in seconds; the minimal one dropped the table
assert asked[0] == G._PLANNER_PROMPT, asked[0][:60]
asked.clear(); Ctx.no_think = False
G._ask_planner(Ctx(), "ваза на столе")
assert asked[0] == G._PLANNER_MINIMAL
assert "is its own element, boxed under what it carries" in G._PLANNER_PROMPT
print("PASS the full planner prompt unless the model is thinking")

_l = G.ensure_text_elements({"background": "white", "elements": [
    {"desc": "logo lettering", "text": "ЗЕРНО", "x": .3, "y": .6, "w": .4, "h": .12}]},
    "логотип кофейни с надписью 'Зерно'")
assert _l["elements"][0]["text"] == "Зерно", _l["elements"][0]
print("PASS requested lettering keeps the user's casing")
assert G.requested_strings("логотип кофейни 'Зерно' в минималистичном стиле") == ["Зерно"]
print("PASS 'логотип' is a lettering cue")
# «логотип кофейни» was planned as macro photography with an 85mm lens
assert 'medium="flat vector graphic design"' in G._PLANNER_PROMPT
print("PASS a logo is graphic design")
users = []
llm.send_to_lm_studio = lambda ctx, msgs, **k: users.append(msgs[1]["content"]) or {"content": ""}
Ctx.no_think, Ctx.reply_lang = True, "ru"
G._ask_planner(Ctx(), "A wooden sign for a small bakery")
assert "lettering you make up yourself is in Russian" in users[0], users[0]
print("PASS a translated Russian request letters in Russian")
