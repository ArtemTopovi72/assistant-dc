"""An explicit length in the request raises the answer budget (live: '1200 слов' cut at 700 tokens)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _ask_stub  # noqa: F401  the model's read, stubbed
import graph_compose as g
assert g._asked_tokens("инструкцию на 1200 слов") >= 3600
assert g._asked_tokens("расскажи подробно") >= 2600
assert g._asked_tokens("привет, 2025 год") == 0
print("ok")

# "N messages ago" gets a numbered list of the user's earlier messages
from types import SimpleNamespace
ctx = SimpleNamespace(facts_text=lambda *_: "", memory_text=lambda: "", custom_personality_text="")
st = {"user_input": "what did i ask three messages ago?", "messages": [
    {"role": "user", "content": "capital of France?"}, {"role": "assistant", "content": "Paris"},
    {"role": "user", "content": "and Germany?"}, {"role": "user", "content": "2+2"},
    {"role": "user", "content": "what did i ask three messages ago?"}]}
_, eff, _ = g._compose_user_message(ctx, st)
assert "1. (3 ago) capital of France?\n2. (2 ago) and Germany?\n3. (1 ago) 2+2]" in eff, eff
print("ok2")

# an elliptical follow-up after a tool turn keeps the tools
import graph_fastpath as F, intent
intent.STUB = lambda t: {"needs_tool": t != "спасибо"}   # the model's read
_h = [{"role": "user", "content": "time in Tokyo?"}, {"role": "assistant", "content": ""},
      {"role": "tool", "content": "00:06"}, {"role": "assistant", "content": "00:06"}]
assert F._followup_of_tool_turn({"messages": _h + [{"role": "user", "content": "x"}]}, "а в Нью-Йорке?")
assert F._followup_of_tool_turn({"messages": _h}, "а в Нью-Йорке?")
assert not F._followup_of_tool_turn({"messages": _h}, "спасибо")
assert not F._followup_of_tool_turn({"messages": [_h[0], _h[3]]}, "а в Нью-Йорке?")
print("ok3")
