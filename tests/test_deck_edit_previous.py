"""'добавь слайд про спутники Марса' edits the deck just made.

Live 2026-09-12 (journey 19): the follow-up re-planned a whole new deck from a
merged topic — different theme (tech → energy), different slides, a different
title. Now the previous plan travels on ctx.last_deck (persisted per Telegram
session), the planner receives it with the change, and the untouched slides,
title and theme survive.
"""
import os, sys, json, tempfile
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sweep_stub  # noqa: F401  the model's narrow reads, stubbed
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import slides
import tools
import tool_args
import tg_sessions

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

PREV = slides.normalize_deck({
    "title": "Марс — красная планета", "subtitle": "для школьников", "theme": "tech",
    "slides": [{"heading": "Где Марс", "bullets": ["четвёртая планета"], "stats": [{"value": "228 млн км", "label": "до Солнца"}]},
               {"heading": "Вода на Марсе", "bullets": ["полярные шапки"]},
               {"heading": "Марс и Земля", "bullets": ["сутки 24 ч 37 мин"]}],
    "sources": [{"title": "NASA", "url": "https://mars.nasa.gov/"}]}, "Mars")

calls = []
def fake_llm(ctx, system, user, **kw):
    calls.append((system, user))
    if "You edit an existing presentation deck" not in system:
        return ""
    deck = json.loads(user.split("Current deck:\n", 1)[1].split("\n\nChange request:")[0])
    deck["slides"].append({"heading": "Спутники Марса", "bullets": ["Фобос и Деймос"],
                           "stats": [{"value": "2", "label": "спутника"}]})
    return json.dumps(deck, ensure_ascii=False)

import llm as _llm
_llm.call_llm_simple = fake_llm
slides.gather_facts = lambda ctx, topic: ("", [])

# 1. the planner path: an edit keeps what it was not asked to touch
got = slides.plan_deck(None, "add a slide about the moons of Mars", previous=PREV)
check("the edit prompt was used", any("You edit an existing" in s for s, _ in calls))
check("the change request and the current deck reach the model",
      any("Change request: add a slide" in u and '"Где Марс"' in u for _, u in calls))
check("theme survives", got.get("theme") == "tech", got.get("theme"))
check("title survives", got.get("title") == PREV["title"], got.get("title"))
check("the old slides survive in order", [s["heading"] for s in got["slides"]][:3] == ["Где Марс", "Вода на Марсе", "Марс и Земля"], [s["heading"] for s in got["slides"]])
check("the new slide is there", any(s["heading"] == "Спутники Марса" for s in got["slides"]))
check("sources survive", any("nasa" in s["url"] for s in got.get("sources", [])), got.get("sources"))
check("edited flag set", got.get("edited") is True)

# 2. a planner that answers nothing leaves the previous deck untouched, flagged
_llm.call_llm_simple = lambda *a, **k: ""
got2 = slides.plan_deck(None, "remove the second slide", previous=PREV)
check("no answer → previous deck kept and edit_failed", got2.get("edit_failed") and len(got2["slides"]) == 3)
r = slides.make_presentation(None, "remove the second slide", tempfile.mkdtemp(), previous=PREV)
check("make_presentation refuses to rewrite the file on a failed edit", r["path"] == "" and "could not be applied" in r["error"], r)

# 3. the tool: edit_previous / the phrasing picks up ctx.last_deck
# the model's reads are stubbed; the phrases run live in bench/intent_sweep_live.py
import intent
intent.CHOICE_STUB = lambda q, t: ("edit" if t.lower().startswith(("добавь", "add")) else "new") if "IN that deck" in q else None
check("edit phrasing recognised (ru)", tools._edits_deck("добавь слайд про спутники Марса"))
check("edit phrasing recognised (en)", tools._edits_deck("add a slide about Mars' moons"))
check("a fresh topic is not an edit", not tools._edits_deck("Mars for schoolchildren"))
check("edit_previous is a tool argument", "edit_previous" in tool_args.CreatePresentationArgs.model_fields)

seen = {}
def fake_make(ctx, topic, out_dir, **kw):
    seen.update(kw); seen["topic"] = topic
    return {"path": os.path.join(out_dir, "x.pptx"), "deck": dict(PREV, edited=True), "images": {}, "error": ""}
slides.make_presentation = fake_make
class Ctx:
    last_deck = PREV; web_search_enabled = False
    def set_stage(self, s): pass
    def remember(self, *a, **k): pass
    def memory_text(self): return ""
    def is_cancelled(self): return False
import image as _image
_image.OUTPUT_DIR = tempfile.mkdtemp()
tools.image_mod.OUTPUT_DIR = _image.OUTPUT_DIR
st = {}
out = tools._handle_create_presentation(Ctx(), st, {"topic": "add a slide about the moons", "edit_previous": True, "illustrate": False})
check("the handler passes the previous deck to the planner", seen.get("previous") is PREV, list(seen))
check("the file is delivered", st.get("document_path", "").endswith("x.pptx"), out[:120])
ctx2 = Ctx(); ctx2.last_deck = None; seen.clear()
tools._handle_create_presentation(ctx2, {}, {"topic": "Mars for schoolchildren", "illustrate": False})
check("a fresh request plans anew", seen.get("previous") is None)
check("an unasked deck keeps the 4-picture cap", seen.get("max_images") == 4, seen.get("max_images"))
seen.clear()
tools._handle_create_presentation(ctx2, {"user_input_original": "про динозавров, с картинкой на каждом слайде"},
                                  {"topic": "dinosaurs", "illustrate": False})
check("'на каждом слайде' lifts the cap", seen.get("max_images") == slides.MAX_SLIDES, seen.get("max_images"))
check("the new deck is remembered on ctx", isinstance(ctx2.last_deck, dict) and ctx2.last_deck.get("title") == PREV["title"])

# 4. the Telegram session persists the plan
s = tg_sessions._Session(1)
s.last_deck = PREV
d = s.to_dict()
check("session serialises last_deck", d.get("last_deck", {}).get("title") == PREV["title"])
s.clear_context()
check("clear_context drops it", not s.last_deck)

# Live 2026-09-28: a content slide named "Титульный лист" and an edit that
# duplicated a slide instead of removing one.
_d = slides.normalize_deck({"title": "История чая", "slides": [
    {"heading": "Титульный лист", "bullets": ["a"]}, {"heading": "Древние истоки", "bullets": ["b"]},
    {"heading": "Япония", "bullets": ["c"]}, {"heading": "Древние истоки", "bullets": ["b"]}]}, "t", 0)
check("title-slide and duplicate slides dropped",
      [s_["heading"] for s_ in _d["slides"]] == ["Древние истоки", "Япония"])

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
