"""Regression: an all-reasoning round must not end the tool phase.

BUG (found live on google/gemma-4-26b-a4b-qat): Gemma regularly spends a whole
turn in its reasoning channel and returns neither content nor a tool call. The
loop treated "no tool_calls" as "the model is done and this is the final answer"
and broke out — after ONE silent round. The next call therefore carried zero tool
schemas and the system line "the tool phase is over — tool calls are IMPOSSIBLE
now", at which point the model emitted generate_image where it could no longer be
run, and signed off with "I cannot draw, the drawing tools are unavailable".

Reproduced end to end: a plain "нарисуй лису" got 11 tools on round 1, returned
nothing, and the user got a refusal for a feature that works.

The fix retries the round WITH the tools still offered, bounded by
_MAX_EMPTY_ROUNDS and paid for out of an extra round each time so a genuine tool
phase is not consumed by the silence.

Run: venv/Scripts/python.exe tests/test_empty_round_retry.py
"""
import os, sys, json, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.disable(logging.CRITICAL)

import test_graph_full as H          # reuse the scripted harness
import graph as G

ok = bad = 0


def check(label, cond, detail=""):
    global ok, bad
    if cond:
        ok += 1
        print(f"PASS {label}")
    else:
        bad += 1
        print(f"FAIL {label}  {detail}")


def _state(text="нарисуй рыжую лису"):
    return {"messages": [{"role": "system", "content": "sys"}],
            "user_input": text, "image_data": None, "final_answer": "",
            "session_memory_text": "", "vision_summary": "", "image_path": "",
            "image_score": 0, "image_attempt": 0, "image_status": ""}


def _tools_offered(call):
    return sorted(t.get("function", {}).get("name")
                  for t in (call["kw"].get("tools") or []))


# ── 1. one silent round, then a tool call: the tool must still run ───────────
llm = H.LLM([
    H._asst(""),                                   # all-reasoning: nothing at all
    H._asst("", [H._tc("generate_image", {"description": "a red fox"})]),
    H._asst("Готово, нарисовал лису."),
])
tools = H.Tools({"generate_image": "OK image at out/fox.png"})
out = H._run(H._ctx(), llm, tools, _state(), fastpath=False)

check("the silent round did NOT end the turn", len(llm.calls) >= 3,
      f"only {len(llm.calls)} model call(s)")
check("the retry still carried the tool schemas",
      len(llm.calls) > 1 and _tools_offered(llm.calls[1]), "round 2 had no tools")
check("the tool actually ran", any(c[0] == "generate_image" for c in tools.calls),
      str(tools.calls))
check("the user gets the real answer, not a refusal",
      "лис" in (out.get("final_answer") or "").lower(),
      repr(out.get("final_answer")))
check("the empty turn was kept out of the history",
      not any(m.get("role") == "assistant" and not m.get("content")
              and not m.get("tool_calls")
              for m in llm.calls[-1]["messages"]),
      "an empty assistant turn survived in the context")

# ── 2. the nudge tells the model the tools are still there ───────────────────
nudge = "".join(str(m.get("content", "")) for m in llm.calls[1]["messages"]
                if m.get("role") == "system")
check("the retry nudge names the silence and the tools",
      "no output at all" in nudge and "tools ARE available" in nudge, nudge[-200:])

# ── 3. the retry is BOUNDED — a permanently mute model still terminates ──────
mute = H.LLM([H._asst("") for _ in range(12)])
mtools = H.Tools({})
out3 = H._run(H._ctx(), mute, mtools, _state(), fastpath=False)
check("a permanently silent model terminates", len(mute.calls) <= 8,
      f"{len(mute.calls)} calls — the retry is not bounded")
check("and it does not run tools it never asked for", not mtools.calls,
      str(mtools.calls))

# ── 4. a NON-empty answer with no tool call still ends the turn immediately ──
plain = H.LLM([H._asst("Столица Франции — Париж."), H._asst("should not be reached")])
ptools = H.Tools({})
out4 = H._run(H._ctx(), plain, ptools, _state("какая столица Франции"), fastpath=False)
check("a real answer still closes the turn in one round", len(plain.calls) == 1,
      f"{len(plain.calls)} calls — the retry fired on a non-empty answer")
check("and it is returned unchanged", "Париж" in (out4.get("final_answer") or ""),
      repr(out4.get("final_answer")))

# ── 5. whitespace-only content counts as empty, not as an answer ─────────────
ws = H.LLM([
    H._asst("   \n  "),
    H._asst("", [H._tc("generate_image", {"description": "a fox"})]),
    H._asst("Нарисовал."),
])
wtools = H.Tools({"generate_image": "OK image"})
H._run(H._ctx(), ws, wtools, _state(), fastpath=False)
check("whitespace-only output is treated as an empty round",
      any(c[0] == "generate_image" for c in wtools.calls), str(wtools.calls))

print(f"\n{ok}/{ok + bad} checks passed")
sys.exit(1 if bad else 0)
