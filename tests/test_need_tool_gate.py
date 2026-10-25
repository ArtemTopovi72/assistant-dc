"""Regression: the need-tool gate and first-failure recovery.

Offline: no LM Studio, no GPU and no network -- every assertion is a pure
regex/AST check (the live tool-calling bench is bench/tc_run.py).

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

Run: venv/Scripts/python.exe tests/test_need_tool_gate.py
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
from types import SimpleNamespace as _NS
from fakes import reads
_READ = {}
_fctx = _NS(last_image_path=None, is_cancelled=lambda: False, sandbox=None, **reads(read=lambda t: _READ.get(t)))
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


# --- 2. denials, delivery and action claims leave the fast path ------------------
# Which drafts are which is the model's reading (agent/reply_read.py); the phrases that
# used to sit here are scored live in bench/reply_read_set.
import inspect
_fp = inspect.getsource(F._fast_path_reply)
check("claim_falls_through", 'said.get("delivers")' in _fp and 'said.get("denies")' in _fp
      and '("did", "promises", "tool_as_text", "does_tool_job")' in _fp)


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
      src.index("FIRST failure of") < src.index("REPLAN_FAILURE_THRESHOLD):"),
      "the retry nudge must be reachable before the give-up nudge")

_hist = {"messages": [{"role": "user", "content": "сделай видео с лисой"},
                     {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "generate_video"}}]},
                     {"role": "tool", "content": "[done]"}, {"role": "assistant", "content": "Вот видео"}]}
_READ["а теперь то же самое, но ночью"] = {"needs_tool": True, "wants": ["generate_video"]}
check("the_same_again_continues_a_tool_turn",
      F._followup_of_tool_turn(_hist, "а теперь то же самое, но ночью", _fctx))
_dctx = _NS(last_image_path=None, is_cancelled=lambda: False, sandbox=None, last_deck={"slides": [{}]})
check("a_deck_keeps_the_fast_path_closed",
      not F._fast_path_allowed(_dctx, {"messages": []}, "make it shorter", "сделай её короче"))
check("a_deck_pins_its_tool", "create_presentation" in P._kit_always(_dctx, set()))
_vctx = _NS(last_image_path=None, is_cancelled=lambda: False, sandbox=None, last_video_path="v.mp4")
check("a_video_keeps_the_fast_path_closed", not F._fast_path_allowed(_vctx, {"messages": []}, "make it longer", "сделай его подлиннее"))
check("a_video_pins_its_tool", "generate_video" in P._kit_always(_vctx, set()))
