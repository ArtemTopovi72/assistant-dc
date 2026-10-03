"""Before/after proof that the thinking toggle is now a REAL direct-execution mode.

Runs the SAME task through a faithful copy of the agent's tool loop in both modes
(thinking ON = ctx.no_think False; DIRECT = ctx.no_think True) against the live
LM Studio model, and reports the behavioural delta: tool rounds, tool-call count,
final-answer length, and reasoning-leak. Proves the toggle changes BEHAVIOUR, not
just the visibility of the <think> channel.

Usage:  venv/Scripts/python.exe tests/validate_concise.py
"""
import sys, threading, time
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import json
from config import MODEL_NAME, MAX_TOOL_ROUNDS
from prompts import build_system_prompt
from llm import send_to_lm_studio
from tools import TOOL_SCHEMAS
from utils import strip_textual_tool_calls, strip_reasoning_leak


class MiniCtx:
    """Minimal stand-in exposing only what send_to_lm_studio / throttle read."""
    def __init__(self, no_think):
        self.model_name = MODEL_NAME
        self.no_think = no_think
        self.api_lock = threading.Lock()
        self.last_api_call_time = 0.0
        self.api_min_interval = 0.2


# Reasoning-leak heuristics (English + Russian planning prose the direct mode bans).
_LEAK = ("let me", "first i", "i should", "i need to", "i will", "let's",
         "давай", "сначала", "мне нужно", "я должен", "хорошо,", "okay, so",
         "step 1", "<think")


def run(no_think, task):
    ctx = MiniCtx(no_think)
    concise = no_think
    temp = 0.2 if concise else 0.5
    maxtok = 700 if concise else 1500
    sysp = build_system_prompt("", concise=concise)
    messages = [{"role": "system", "content": sysp},
                {"role": "user", "content": task}]
    rounds = 0
    tool_calls = 0
    final = ""
    t0 = time.time()
    for r in range(MAX_TOOL_ROUNDS):
        rounds += 1
        msg = send_to_lm_studio(ctx, messages, tools=TOOL_SCHEMAS,
                                tool_choice="auto", temperature=temp, max_tokens=maxtok)
        if not msg:
            break
        messages.append(msg)
        tcs = msg.get("tool_calls") or []
        if not tcs:
            final = strip_textual_tool_calls(strip_reasoning_leak(msg.get("content", "") or ""))
            break
        tool_calls += len(tcs)
        # We don't actually execute (no GPU side effects); feed a neutral stub result
        # so the loop can terminate, mirroring shape only.
        for tc in tcs:
            messages.append({"role": "tool", "tool_call_id": tc.get("id", "x"),
                             "content": "[stub] done."})
    dt = time.time() - t0
    raw = strip_reasoning_leak(messages[-1].get("content", "") if messages else "")
    leak = [w for w in _LEAK if w in (final or raw or "").lower()]
    return {"rounds": rounds, "tool_calls": tool_calls, "chars": len(final or ""),
            "secs": round(dt, 1), "leak": leak, "answer": (final or "")[:280]}


if __name__ == "__main__":
    task = ("Посмотри, сколько сейчас времени в Токио, и скажи, "
            "стоит ли мне сейчас звонить коллеге туда.")
    print(f"MODEL: {MODEL_NAME}\nTASK: {task}\n" + "=" * 70)
    for label, nt in [("THINKING ON  (no_think=False)", False),
                      ("DIRECT MODE  (no_think=True) ", True)]:
        r = run(nt, task)
        print(f"\n### {label}")
        print(f"  tool rounds : {r['rounds']}")
        print(f"  tool calls  : {r['tool_calls']}")
        print(f"  answer chars: {r['chars']}")
        print(f"  wall secs   : {r['secs']}")
        print(f"  reasoning-leak markers: {r['leak'] or 'none'}")
        print(f"  answer: {r['answer']!r}")
