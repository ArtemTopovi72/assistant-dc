"""Regression: the need-tool gate and first-failure recovery.

All three defects here were found by bench/tc_run.py (the live tool-calling
bench) and are reproduced offline with no LM Studio, no GPU and no network --
every assertion is a pure regex/AST check.

1. _TOOL_TRIGGER_RE is an ALLOWLIST that decides whether the agent is given
   tools at all. Whole capability families were missing from it, so the fast
   path answered them with no tools and the model denied capabilities it has:
   "я текстовый помощник и не могу создать видеофайл, используйте Runway"
   (video), "я не вижу содержимое вашего буфера" (clipboard), and the deck
   request came back as slide text in the chat instead of a .pptx file.

2. _FAST_PATH_CAPABILITY_DENIAL_RE is the safety net for whatever the
   allowlist still misses -- it throws a tool-less denial away and falls
   through to the real loop. It only matched IMAGE denials, so both live
   denials above sailed past it as final answers.

3. Nothing spoke to the FIRST tool failure; the re-plan nudge only fires after
   REPLAN_FAILURE_THRESHOLD failures in a row and tells the model to give up.
   A single injected error made the agent apologise and stop on 4/4 bench runs.

Run: .\venv\Scripts\python.exe tests\test_need_tool_gate.py
"""
import sys
sys.path.insert(0, ".")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import graph
import graph_fastpath as F
import graph_personality as P

checks = []


def check(name, ok, detail=""):
    checks.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if not ok else ""))


# --- 1. the fast path follows the model's read (agent/intent.py) ----------
# What a message needs is read by the model, not a keyword allowlist (phrases
# live in bench/intent_live.py against the real model). Here: the gate obeys it.
import intent
from types import SimpleNamespace as _NS
_fctx = _NS(last_image_path=None, is_cancelled=lambda: False, sandbox=None)
_READ = {}
intent.STUB = lambda t: _READ.get(t)
_READ["сделай рилс про кофе"] = {"needs_tool": True, "wants": ["generate_video"]}
_READ["как дела"] = {"needs_tool": False}
_READ["забудь всё, что я про себя рассказывал"] = {"needs_tool": True, "wants": ["forget_facts"]}
check("needs_tool_skips_the_fast_path",
      not F._fast_path_allowed(_fctx, {"messages": []}, "make a reel about coffee", "сделай рилс про кофе"))
check("forget_request_skips_the_fast_path",
      not F._fast_path_allowed(_fctx, {"messages": []}, "forget everything", "забудь всё, что я про себя рассказывал"))
check("small_talk_takes_it", F._fast_path_allowed(_fctx, {"messages": []}, "how are you", "как дела"))
check("an_unread_message_takes_the_full_loop",
      not F._fast_path_allowed(_fctx, {"messages": []}, "hi", "никем не прочитано"))


# --- 2. the denial net must catch non-image denials too ---------------------
DENIALS = [
    ("video_live",     "Я текстовый помощник и не могу напрямую создать видеофайл, "
                       "но вы можете использовать Runway или Pika для этой задачи."),
    ("clipboard_live", "Чтобы я мог это сделать, пожалуйста, вставьте текст из буфера "
                       "обмена в чат. Сейчас я не вижу содержимое вашего буфера."),
    ("no_access",      "У меня нет доступа к вашим файлам."),
    ("image_legacy",   "Извините, я не могу нарисовать это."),
    ("english",        "I cannot access your clipboard."),
]
for name, text in DENIALS:
    check(f"denial_catches_{name}",
          bool(F._FAST_PATH_CAPABILITY_DENIAL_RE.search(text)), repr(text[:60]))

NOT_DENIALS = [
    ("physics", "Небо голубое из-за рэлеевского рассеяния света в атмосфере."),
    ("deck",    "Вот план презентации из 8 слайдов: 1. Основание Рима и легенда."),
    ("chat",    "Привет! У меня всё хорошо, чем могу помочь?"),
    ("refusal_about_topic",
                "Я не могу говорить об этом, давай сменим тему."),
]
for name, text in NOT_DENIALS:
    check(f"denial_ignores_{name}",
          not F._FAST_PATH_CAPABILITY_DENIAL_RE.search(text), repr(text[:60]))


import inspect
# --- 2b. a tool-less DELIVERY CLAIM is thrown away too --------------------------
# "а теперь просто картинку этого же кота" had no draw verb, took the fast
# path, and came back as "Вот ваша картинка с рыжим котом" with nothing
# attached (journey 28).
CLAIMS = [
    ("ru_picture", "Вот ваша картинка с рыжим котом, бегущим по пляжу."),
    ("ru_deck",    "Готово! Презентация сохранена и отправлена."),
    ("en_picture", "Here is your picture of the cat on the beach."),
    ("en_made",    "I've created the video you asked for."),
    ("ru_looks",   "Вот так выглядит Эйфелева башня ночью."),
]
for name, text in CLAIMS:
    check(f"claim_caught_{name}", bool(F._FAST_PATH_DELIVERY_CLAIM_RE.search(text)), repr(text[:60]))
NOT_CLAIMS = [
    ("fact",   "Столица Австралии — Канберра."),
    ("code",   "Вот три теста на pytest для функции."),
    ("plan",   "Вот план: сначала посчитаем, потом нарисуем."),
    ("chat",   "Привет! Чем могу помочь?"),
    ("described", "Вот как выглядит сибирский кот: крупный, пушистый. У него густая шерсть."),
]
for name, text in NOT_CLAIMS:
    check(f"claim_ignores_{name}", not F._FAST_PATH_DELIVERY_CLAIM_RE.search(text), repr(text[:60]))
check("claim_falls_through", "_FAST_PATH_DELIVERY_CLAIM_RE.search(fast_content)" in inspect.getsource(F))


# --- 3. first-failure retry -------------------------------------------------
retryable = P._RETRY_ON_FIRST_FAILURE
check("retry_set_covers_cheap_tools",
      {"search", "generate_image", "find_photo", "calculate"} <= retryable,
      sorted(retryable))
# The minutes-long tools must stay OUT: a blind retry there doubles the cost of
# a real failure instead of routing around it.
check("retry_set_excludes_expensive",
      not ({"deep_research", "generate_video", "create_presentation"} & retryable),
      sorted(retryable))

src = inspect.getsource(P)
check("first_failure_nudge_exists", "FIRST failure of" in src)
check("nudge_is_once_per_tool", "retry_nudged" in src and "retry_nudged.add" in src)
check("nudge_fires_before_giveup",
      src.index("_RETRY_ON_FIRST_FAILURE\n") if False else
      src.index("FIRST failure of") < src.index("REPLAN_FAILURE_THRESHOLD):"),
      "the retry nudge must be reachable before the give-up nudge")

check("ok_i_will_do_it_is_a_promise",
      bool(F._FAST_PATH_ACTION_CLAIM_RE.search("Хорошо, я сделаю фон светлым и белым для всех слайдов."))
      and not F._FAST_PATH_ACTION_CLAIM_RE.search("Я сделаю вывод: удалёнка подходит не всем."))
_hist = {"messages": [{"role": "user", "content": "сделай видео с лисой"},
                     {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "generate_video"}}]},
                     {"role": "tool", "content": "[done]"}, {"role": "assistant", "content": "Вот видео"}]}
_READ["а теперь то же самое, но ночью"] = {"needs_tool": True, "wants": ["generate_video"]}
check("the_same_again_continues_a_tool_turn",
      F._followup_of_tool_turn(_hist, "а теперь то же самое, но ночью", _fctx))
check("a_leading_sure_is_still_a_claim",
      bool(F._FAST_PATH_DELIVERY_CLAIM_RE.search("Конечно! Вот то же самое видео с лисой")))
_dctx = _NS(last_image_path=None, is_cancelled=lambda: False, sandbox=None, last_deck={"slides": [{}]})
check("a_deck_keeps_the_fast_path_closed",
      not F._fast_path_allowed(_dctx, {"messages": []}, "make it shorter", "сделай её короче"))
check("a_deck_pins_its_tool", "create_presentation" in P._kit_always(_dctx, set()))
check("a_bare_past_tense_is_a_claim", bool(F._FAST_PATH_ACTION_CLAIM_RE.search("Сделал еще короче, теперь только главное.")))
_vctx = _NS(last_image_path=None, is_cancelled=lambda: False, sandbox=None, last_video_path="v.mp4")
check("a_video_keeps_the_fast_path_closed", not F._fast_path_allowed(_vctx, {"messages": []}, "make it longer", "сделай его подлиннее"))
check("a_video_pins_its_tool", "generate_video" in P._kit_always(_vctx, set()))
