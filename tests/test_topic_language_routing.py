"""The topic's OWN language must decide the output language of research
reports and presentations — not the session's RU/EN UI toggle, and not the
tool's `topic` argument (which has usually already been through the
English-first entry translation, see graph.translate_node).

Two bugs fixed here:

  1. tg_bot.py's "do a deep research on: X" bypass (typed directly, skips the
     graph/tool-calling entirely) passed `out_lang=lang` — the session's UI
     setting — so a Russian-UI user researching an English topic got a
     Russian-language paper about it, and vice versa. Fixed to
     `out_lang=deep_research.lang_of_text(topic, default=lang)`, matching
     what the in-graph tool-calling path (tools.py's deep_research handler)
     already did correctly via `lang_of_text(state["user_input_original"])`.

  2. slides.py's plan_deck/make_presentation had NO language parameter at
     all — they relied purely on a soft planner-prompt instruction ("written
     in the language of the topic"), and `topic` is the tool arg, already
     translated to English by the time _handle_create_presentation runs. A
     Russian request for a presentation produced an English deck. Fixed by
     detecting the language from state["user_input_original"] in
     tools.py's _handle_create_presentation and threading it through
     make_presentation -> plan_deck as an explicit `out_lang` directive
     (deep_research._lang_directive), same mechanism deep research uses.

Offline only (no live LM Studio) — everything that would call the LLM is
monkeypatched.

Run: venv/Scripts/python.exe tests/test_topic_language_routing.py
"""
import os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T
import deep_research as D
import slides as S

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_topiclang_")
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


print("=" * 70)
print("1. lang_of_text: the script-level heuristic itself")
print("=" * 70)

check("a Russian (Cyrillic) topic detects as ru",
      D.lang_of_text("квантовые компьютеры") == "ru")
check("an English topic detects as en (the default)",
      D.lang_of_text("quantum computers") == "en")
check("an ambiguous/empty topic falls back to the given default",
      D.lang_of_text("", default="ru") == "ru")
check("a topic with only proper nouns/numbers falls back to the default too",
      D.lang_of_text("Tesla 3000", default="ru") == "ru")


print()
print("=" * 70)
print("2. Deep-research BYPASS path: out_lang follows the TOPIC, not the UI toggle")
print("=" * 70)

def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    sent = []
    def rec(method, payload=None, **kw):
        sent.append((method, payload))
        return {"ok": True, "result": {"message_id": len(sent)}}
    bot._api_post = rec
    return bot, sent


def run_bypass(bot, chat_id, ui_lang, topic_text):
    """Drive exactly the code path tg_bot.py uses for a typed
    "do a deep research on: X" message, capturing what out_lang reaches
    run_deep_research — without actually running research (fully stubbed)."""
    u = T._User(chat_id=chat_id, name="U", tg_username="u", password_hash="x",
               status="approved")
    bot._user_store.put(u)
    sess = bot._get_session(chat_id)
    sess.lang = ui_lang
    bot._store.put(sess)

    captured = {}
    _orig = D.run_deep_research
    def _fake(ctx, topic, out_lang="en", **kw):
        captured["out_lang"] = out_lang
        return {"report": "stub", "reply": "stub", "stats": {"findings": 0}}
    D.run_deep_research = _fake
    try:
        task = T._Task(task_id=f"t-{chat_id}", chat_id=chat_id,
                       user_text=f"do a deep research on: {topic_text}")
        bot._run_task_inner(task, sess, u, u.name, ui_lang, _FakeCtx(), None)
    finally:
        D.run_deep_research = _orig
    return captured.get("out_lang")


class _FakeCtx:
    """A minimal stand-in — the bypass path only needs set_stage/is_cancelled/
    cancel_event off of ctx before it reaches run_deep_research (stubbed above)."""
    def __init__(self):
        self.stage_callback = None
        import threading
        self.cancel_event = threading.Event()
    def set_stage(self, s):
        if self.stage_callback: self.stage_callback(s)
    def is_cancelled(self): return False


bot, sent = make_bot()
got = run_bypass(bot, 7001, ui_lang="ru", topic_text="quantum computing breakthroughs")
check("RU-UI session + ENGLISH topic -> out_lang='en' (follows the topic, not the UI)",
      got == "en", got)

bot2, sent2 = make_bot()
got2 = run_bypass(bot2, 7002, ui_lang="en", topic_text="прорывы в квантовых вычислениях")
check("EN-UI session + RUSSIAN topic -> out_lang='ru' (follows the topic, not the UI)",
      got2 == "ru", got2)

bot3, sent3 = make_bot()
got3 = run_bypass(bot3, 7003, ui_lang="ru", topic_text="Tesla 3000")
check("an ambiguous topic (no script signal) falls back to English, "
      "matching lang_of_text's own default (NOT the UI setting — the function "
      "cannot tell 'confidently English' from 'no signal', so trusting the UI "
      "setting there would silently defeat the fix for a real English topic)",
      got3 == "en", got3)


print()
print("=" * 70)
print("3. plan_deck appends an explicit language directive for a non-English out_lang")
print("=" * 70)

_seen_prompts = []
def _fake_llm(ctx, system_prompt, user_msg, **kw):
    _seen_prompts.append(user_msg)
    return '{"title": "T", "slides": [{"heading":"H","bullets":["b"]}]}'

import llm as _llm_mod
_orig_call = _llm_mod.call_llm_simple
S_call = None
try:
    import slides
    # plan_deck imports call_llm_simple locally from `llm` — patch the module.
    _llm_mod.call_llm_simple = _fake_llm
    _seen_prompts.clear()
    S.plan_deck(None, "История Транссибирской магистрали", out_lang="ru")
    check("a Russian-language deck request carries the OUTPUT LANGUAGE directive",
          _seen_prompts and "OUTPUT LANGUAGE" in _seen_prompts[0]
          and "Russian" in _seen_prompts[0],
          _seen_prompts[0] if _seen_prompts else None)

    _seen_prompts.clear()
    S.plan_deck(None, "History of the Trans-Siberian Railway", out_lang="en")
    check("out_lang='en' adds NO directive (English is the silent default)",
          _seen_prompts and "OUTPUT LANGUAGE" not in _seen_prompts[0],
          _seen_prompts[0] if _seen_prompts else None)

    _seen_prompts.clear()
    S.plan_deck(None, "История Транссибирской магистрали")
    check("out_lang omitted entirely -> no directive (backward compatible)",
          _seen_prompts and "OUTPUT LANGUAGE" not in _seen_prompts[0],
          _seen_prompts[0] if _seen_prompts else None)
finally:
    _llm_mod.call_llm_simple = _orig_call


print()
print("=" * 70)
print("4. _handle_create_presentation detects language from the ORIGINAL user text,")
print("   not the (already English-translated) topic tool argument")
print("=" * 70)

import tools as T_mod

_captured_out_lang = {}
def _fake_make_presentation(ctx, topic, out_dir, *, n_slides=0, image_fetcher=None,
                            max_images=4, out_lang="", previous=None):
    _captured_out_lang["out_lang"] = out_lang
    return {"path": "/tmp/fake.pptx",
            "deck": {"title": topic, "slides": [{"heading": "a"}, {"heading": "b"}]}}

class _FakeImageMod:
    class OUTPUT_DIR:
        @staticmethod
        def mkdir(*a, **k): pass
    OUTPUT_DIR = type("P", (), {"mkdir": staticmethod(lambda *a, **k: None),
                                "__str__": lambda self: "/tmp"})()

class _FakeCtx2:
    web_search_enabled = False
    def set_stage(self, s): pass
    def remember(self, *a, **k): pass
    def memory_text(self): return ""

_orig_slides_mp = S.make_presentation
S.make_presentation = _fake_make_presentation
_orig_image_mod = getattr(T_mod, "image_mod", None)
T_mod.image_mod = _FakeImageMod
try:
    state = {"user_input_original": "сделай презентацию про историю железных дорог"}
    ctx = _FakeCtx2()
    args = {"topic": "the history of railways", "illustrate": False}  # already-translated topic
    T_mod._handle_create_presentation(ctx, state, args)
    check("the deck language is detected from user_input_original (ru), not the English topic arg",
          _captured_out_lang.get("out_lang") == "ru", _captured_out_lang)
finally:
    S.make_presentation = _orig_slides_mp
    if _orig_image_mod is not None:
        T_mod.image_mod = _orig_image_mod


print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
