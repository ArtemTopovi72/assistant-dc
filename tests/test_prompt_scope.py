"""prompt_scope: a tool's rules travel only with that tool; policy and safety
lines always travel."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prompt_scope as PS
import tools as T
from prompts import SYSTEM_PROMPT_PERSONALITY as S

KNOWN = [t["function"]["name"] for t in T.TOOL_SCHEMAS]


def test_a_greeting_drops_the_video_and_image_rules():
    out = PS.scope(S, ["search", "calculate", "remember_fact"], KNOWN)
    assert "generate_video" not in out and "inpaint_image" not in out
    assert len(out) < len(S) * 0.8


def test_the_rules_come_back_with_their_tool():
    out = PS.scope(S, ["generate_video"], KNOWN)
    assert "generate_video" in out


def test_safety_and_policy_lines_always_stay():
    out = PS.scope(S, [], KNOWN)
    for must in ("untrusted", "TOOL ERROR", "Actions happen ONLY", "Default to NO tool"):
        assert must in out, must
    for section in ("[Language]", "[About you]", "[Formatting Rules]", "[Anti-leakage]"):
        assert section in out


def test_it_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("PROMPT_SCOPE", "0")
    assert PS.scope(S, [], KNOWN) == S
