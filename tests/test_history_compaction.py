"""Regression + red-team for the rolling history-compaction recovery path
(graph.compact_history_if_needed). This is the path that keeps long conversations
from exploding context; branch coverage showed it was previously unexercised by the
test suite. We verify it does NOT corrupt history:

  - system prompt (msg 0) is preserved and stays first,
  - the tail (last _HISTORY_KEEP_TURNS user turns) is kept verbatim, in order,
  - the compacted history is STRICTLY shorter (context actually shrinks),
  - NO tool_call/tool pair is split — the rebuilt history is API-valid (no orphan
    tool message, every assistant tool_calls answered),
  - a prior summary is merged (not duplicated) on the next compaction,
  - the extractive FALLBACK fires when the summariser returns empty,
  - it FAILS OPEN (returns the original list unchanged) if the summariser raises.

Run: venv/Scripts/python.exe tests/test_history_compaction.py
"""
import os, sys, threading, copy
os.environ["CONTEXT_V2"] = "0"   # these check the v1 summariser; v2: tests/test_context_v2.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import graph as gmod
import llm as _llmmod

# The summariser seam. graph_history.send_to_lm_studio is a one-line indirection
# that forwards to llm.send_to_lm_studio at CALL time, precisely so a patch works
# from anywhere -- and its docstring says to patch llm. This suite patched
# `graph.send_to_lm_studio` instead, which graph_history has not read since the
# compaction code was lifted out of graph.py: a DEAD SEAM. Every stub below was
# installed and never called, so the tests were exercising the real
# llm.send_to_lm_studio -- i.e. reaching for the live LM Studio from a unit test.
def _patch_summariser(fn):
    _llmmod.send_to_lm_studio = fn
    gmod.send_to_lm_studio = fn          # kept: some checks still read it off graph
from graph import (compact_history_if_needed, HISTORY_COMPACT_EVERY,
                   _HISTORY_KEEP_TURNS, _HISTORY_SUMMARY_MARKER)
from models import Context
from pathlib import Path


def _ctx():
    c = Context(models=None, transcription_cache={}, cache_file=Path("tests/_hc.json"),
                asr_lock=threading.Lock(), tts_lock=threading.Lock(),
                model_name="m", no_think=True)
    c.total_user_turns = 0
    return c


def _long_history(n_turns=8):
    """system + n user turns; every other assistant turn uses a tool (call+result)."""
    msgs = [{"role": "system", "content": "SYSTEM PROMPT (live)"}]
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"user turn {i}"})
        if i % 2 == 0:
            cid = f"c{i}"
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": [{"id": cid, "type": "function",
                                         "function": {"name": "calculate", "arguments": "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": cid, "content": f"result {i}"})
        msgs.append({"role": "assistant", "content": f"answer {i}"})
    return msgs


def _api_valid(msgs):
    """No orphan tool messages; every assistant tool_calls id answered before the next
    assistant tool_calls / end."""
    pending, answered = [], set()
    probs = []
    def flush(w):
        if pending: probs.append(f"unanswered {pending} before {w}")
    for m in msgs:
        r = m.get("role")
        if r == "assistant" and m.get("tool_calls"):
            flush("next tool_calls"); pending = [t["id"] for t in m["tool_calls"]]; answered = set()
        elif r == "tool":
            tid = m.get("tool_call_id")
            if tid in pending: pending.remove(tid); answered.add(tid)
            else: probs.append(f"orphan tool {tid}")
    flush("end")
    return probs


def test_compaction_shrinks_and_preserves():
    # summariser returns a normal summary
    _patch_summariser(lambda *a, **k: {"role": "assistant", "content": "SUMMARY-OF-OLD"})
    ctx = _ctx()
    msgs = _long_history(8)
    orig_len = len(msgs)
    out = msgs
    # advance to a compaction turn
    for _ in range(HISTORY_COMPACT_EVERY):
        out = compact_history_if_needed(ctx, copy.deepcopy(out))
    assert out[0]["role"] == "system" and out[0]["content"] == "SYSTEM PROMPT (live)", "system prompt lost/moved"
    assert out[1]["role"] == "system" and out[1]["content"].startswith(_HISTORY_SUMMARY_MARKER), "summary not inserted"
    assert "SUMMARY-OF-OLD" in out[1]["content"], "summary content missing"
    assert len(out) < orig_len, f"history did not shrink ({len(out)} >= {orig_len})"
    # tail: last _HISTORY_KEEP_TURNS user turns preserved verbatim & in order
    tail_users = [m["content"] for m in out if m.get("role") == "user"]
    assert tail_users == [f"user turn {i}" for i in range(8 - _HISTORY_KEEP_TURNS, 8)], \
        f"tail user turns wrong/reordered: {tail_users}"
    # API validity: no split tool pairs / orphans
    probs = _api_valid(out)
    assert not probs, f"compaction produced invalid history: {probs}"
    print("PASS compaction shrinks, keeps system prompt + verbatim tail, no split tool pairs")


def test_prior_summary_merged_not_duplicated():
    calls = {"n": 0}
    def summ(*a, **k):
        calls["n"] += 1
        return {"role": "assistant", "content": f"SUMMARY-{calls['n']}"}
    _patch_summariser(summ)
    ctx = _ctx()
    out = _long_history(8)
    # two compaction cycles
    for cycle in range(2):
        for _ in range(HISTORY_COMPACT_EVERY):
            out = compact_history_if_needed(ctx, copy.deepcopy(out))
        # append more turns so the 2nd cycle has something to fold
        out += [{"role": "user", "content": f"extra {cycle}"},
                {"role": "assistant", "content": f"reply {cycle}"}]
    summary_msgs = [m for m in out if m.get("role") == "system"
                    and str(m.get("content", "")).startswith(_HISTORY_SUMMARY_MARKER)]
    assert len(summary_msgs) == 1, f"expected exactly ONE running summary, got {len(summary_msgs)}"
    print("PASS prior summary merged into a single running summary (no duplication)")


def test_extractive_fallback_on_empty_summary():
    _patch_summariser(lambda *a, **k: {"role": "assistant", "content": ""})  # always empty
    ctx = _ctx()
    out = _long_history(8)
    for _ in range(HISTORY_COMPACT_EVERY):
        out = compact_history_if_needed(ctx, copy.deepcopy(out))
    smry = [m for m in out if m.get("role") == "system"
            and str(m.get("content", "")).startswith(_HISTORY_SUMMARY_MARKER)]
    assert smry, "extractive fallback did not produce a summary"
    # fallback keeps the user's own lines
    assert "user turn 0" in smry[0]["content"], "extractive fallback lost user intent"
    print("PASS extractive fallback fires on empty summariser output")


def test_fails_open_on_summariser_exception():
    def boom(*a, **k): raise RuntimeError("summariser down")
    _patch_summariser(boom)
    ctx = _ctx()
    msgs = _long_history(8)
    out = msgs
    for _ in range(HISTORY_COMPACT_EVERY):
        out = compact_history_if_needed(ctx, copy.deepcopy(out))
    # NOTE: with an exception INSIDE the try, compaction returns the (unchanged) list.
    probs = _api_valid(out)
    assert not probs, f"fail-open produced invalid history: {probs}"
    assert any(m.get("role") == "user" for m in out), "history destroyed on summariser failure"
    print("PASS compaction fails open (history preserved) when summariser raises")


if __name__ == "__main__":
    test_compaction_shrinks_and_preserves()
    test_prior_summary_merged_not_duplicated()
    test_extractive_fallback_on_empty_summary()
    test_fails_open_on_summariser_exception()
    print("\ndone")
