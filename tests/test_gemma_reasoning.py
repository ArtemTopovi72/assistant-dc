"""Gemma 4 is a third reasoning contract, and we were sending it the other two.

Measured live 2026-07-29 on google/gemma-4-12b-qat:
  * `reasoning_effort`, `reasoning:"off"` and `enable_thinking` are ALL no-ops —
    five configurations returned byte-identical output. There is no knob.
  * It thinks on every turn already, 90-800 tokens, scaling with difficulty, and
    that thinking is billed against max_tokens. Our chat budgets were 700-1500,
    so a hard question plus a real answer ran out mid-thought.
  * Through our real system prompt and transport: tool routing 5/5, tag leaks 0.

Run: venv/Scripts/python.exe tests/test_gemma_reasoning.py
"""
import os, sys, json, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import llm as L

_REAL_STREAM = L._stream_chat
_n = _bad = 0


def check(label, cond, detail=""):
    global _n, _bad
    _n += 1
    if cond:
        print(f"  ok   {label}")
    else:
        _bad += 1
        print(f"  FAIL {label}  {detail}")


class Ctx:
    def __init__(self, model_name, no_think=False):
        self.model_name = model_name
        self.no_think = no_think
        self.reasoning_effort = "high"
    def is_cancelled(self): return False


def capture(model, **kw):
    """Send one request and return the payload that would hit the server."""
    seen = {}
    def fake(ctx, payload):
        seen.update(payload)
        return {"role": "assistant", "content": "ok"}
    L._stream_chat = fake
    L.time.sleep = lambda *a, **k: None
    L.throttle_external_calls = lambda *a, **k: None
    try:
        L.send_to_lm_studio(Ctx(model, kw.pop("no_think", True)),
                            [{"role": "user", "content": "x"}], **kw)
    finally:
        L._stream_chat = _REAL_STREAM
    return seen


print("\nRECOGNISING THE FAMILY")
for name in ("google/gemma-4-12b-qat", "google/gemma-4-31b-qat", "gemma-4-26b-a4b-qat"):
    check(f"{name} is Gemma 4", L._is_gemma4(name))
for name in ("qwen3.5-9b-uncensored", "openai/gpt-oss-120b", ""):
    check(f"{name!r} is not", not L._is_gemma4(name))

print("\nNO OTHER FAMILY'S SWITCHES ARE SENT AT IT")
p = capture("google/gemma-4-12b-qat")
check("no reasoning:'off' — it cannot honour it and it measured as a no-op",
      "reasoning" not in p, p.get("reasoning"))
check("no reasoning_effort (that is a gpt-oss level)", "reasoning_effort" not in p)
check("no enable_thinking (that is a Qwen switch)", "chat_template_kwargs" not in p, p)

p = capture("google/gemma-4-12b-qat", no_think=True)
check("the thinking toggle does not try to silence it either",
      "chat_template_kwargs" not in p and "reasoning" not in p, p)
check("and no /no_think marker is injected into its prompt",
      all("/no_think" not in (m.get("content") or "") for m in p["messages"]),
      p["messages"])

p = capture("google/gemma-4-12b-qat", prefill="<think></think>")
check("the Qwen <think></think> prefill is not prepended to a Gemma turn",
      all("<think>" not in (m.get("content") or "") for m in p["messages"]),
      p["messages"])

print("\nOTHER MODELS ARE UNTOUCHED")
p = capture("qwen3.5-9b-uncensored")
check("Qwen still gets reasoning:'off'", p.get("reasoning") == "off")
check("Qwen gets no top_k — that profile is Gemma's", "top_k" not in p)
p = capture("openai/gpt-oss-120b")
check("gpt-oss gets the lowest effort level (it cannot switch reasoning off)", p.get("reasoning_effort") == "low")
check("gpt-oss keeps its own sampling", "top_p" not in p)
p = capture("qwen3.5-9b-uncensored", prefill="<think></think>")
check("the Qwen prefill still works for Qwen",
      any("<think>" in (m.get("content") or "") for m in p["messages"]))

print("\nROOM TO FINISH THINKING AND STILL ANSWER")
print("\nNO THINKING BY DEFAULT (2026-09-23: 'всегда и везде')")
p = capture("google/gemma-4-12b-qat", max_tokens=700)
check("every Gemma call is prefilled with the empty thought block",
      (p["messages"][-1].get("role") == "assistant"
       and p["messages"][-1].get("content") == L.GEMMA_NO_THINK_PREFILL), p["messages"][-1])
check("with no thinking to fund, the caller's budget is kept",
      p["max_tokens"] == 700, p["max_tokens"])

def _tool(name):
    return {"type": "function", "function": {"name": name, "parameters": {}}}


p = capture("google/gemma-4-12b-qat", max_tokens=700, tools=[_tool("edit_file"), _tool("web_search")])
check("code-editing turns do NOT think by default either (user, 2026-09-23)",
      p["messages"][-1].get("content") == L.GEMMA_NO_THINK_PREFILL, p["messages"][-1])
p = capture("google/gemma-4-12b-qat", max_tokens=700, no_think=False)
check("the thinking switch ON does not bring reasoning back (live 16:51: 2813 "
      "tokens on six storyboard lines)",
      p["messages"][-1].get("content") == L.GEMMA_NO_THINK_PREFILL
      and p["max_tokens"] == 700, (p["messages"][-1], p["max_tokens"]))

p = capture("google/gemma-4-12b-qat", max_tokens=700, tools=[_tool("generate_video")],
            tool_choice="required")
check("a FORCED tool call gets no prefill (the grammar rejects it: 400, live 6/6)",
      all(m.get("content") != L.GEMMA_NO_THINK_PREFILL for m in p["messages"]), p["messages"][-1])
p = capture("google/gemma-4-12b-qat", max_tokens=700, tools=[_tool("generate_video")],
            tool_choice="required", prefill=L.GEMMA_NO_THINK_PREFILL)
check("not even when the caller passes it explicitly",
      all(m.get("content") != L.GEMMA_NO_THINK_PREFILL for m in p["messages"]), p["messages"][-1])

p = capture("google/gemma-4-12b-qat", max_tokens=700, tools=[_tool("calculate")],
            tool_choice="required", prefill="<think></think>")
check("nor the Qwen prefill (400 'after accepting piece: think', tool bench)",
      all("<think>" not in str(m.get("content")) for m in p["messages"]), p["messages"][-1])

print("\nNOTHING TURNS REASONING BACK ON")
p = capture("google/gemma-4-12b-qat", max_tokens=700, force_think=True)
check("force_think does not think either",
      p["messages"][-1].get("content") == L.GEMMA_NO_THINK_PREFILL and p["max_tokens"] == 700,
      (p["messages"][-1], p["max_tokens"]))
check("there is no environment switch for it", not hasattr(L, "GEMMA_NO_THINK_DEFAULT"))
p = capture("qwen3.5-9b-uncensored", max_tokens=700)
check("and the floor does not apply to other models", p["max_tokens"] == 700)

print("\nSAMPLING IS LEFT TO THE CALLER (measured: the card profile changes nothing)")
# 45 questions x 2 repeats, sampling the only variable: 89/90 both ways, +0.0.
# The set saturates at 98.9%, so this says "no difference detectable here", not
# "no difference exists" — either way it is not grounds to change behaviour.
p = capture("google/gemma-4-12b-qat", temperature=0.5)
check("the caller's temperature is respected", p["temperature"] == 0.5, p["temperature"])
check("no top_p is forced on", "top_p" not in p, p.get("top_p"))
check("no top_k is forced on", "top_k" not in p, p.get("top_k"))
p = capture("google/gemma-4-12b-qat", temperature=0.1)
check("a cold structured-output caller stays cold", p["temperature"] == 0.1, p["temperature"])
check("the card profile is still recorded for whoever tries again",
      L.GEMMA_SAMPLING == {"temperature": 1.0, "top_p": 0.95, "top_k": 64})

print("\nTHINKING IS KEPT ON TOOL-CALL TURNS (card requires it)")
hist = [
    {"role": "user", "content": "погода в Берлине?"},
    {"role": "assistant", "content": "<|channel>thought\nI should search<channel|>",
     "tool_calls": [{"id": "1", "type": "function",
                     "function": {"name": "search", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "1", "content": "18C"},
    {"role": "assistant", "content": "<|channel>thought\nnow answer<channel|>В Берлине 18."},
]
out = L._strip_model_artifacts(hist)
check("the tool-call turn keeps its reasoning",
      "thought" in (out[1].get("content") or ""), out[1])
check("its tool_calls survive intact", out[1].get("tool_calls") == hist[1]["tool_calls"])
check("an ordinary assistant turn is still stripped",
      "<|" not in (out[3].get("content") or ""), out[3])
check("and the plain answer text survives that strip",
      "Берлине" in (out[3].get("content") or ""), out[3])
check("the user turn is untouched", out[0] == hist[0])

print("\nFORMAT D: THE CALL SHAPE THAT REACHED THE USER AS TEXT")
# Live on gemma-4-12b-qat: `call:calculate{expression: "17 * 23 - 91"}<tool_call|>`
# — no `<|tool_call>` opener, so Format A missed it; plain quotes instead of
# <|"|>, so Format C missed it. Neither executed NOR stripped: the tool never ran
# and the raw markup was the answer the user got.
tc, rest = L.extract_gemma4_tool_calls(
    'Посчитаю. call:calculate{expression: "17 * 23 - 91"}<tool_call|>')
check("the call is recognised", len(tc) == 1, tc)
check("with the right tool", tc and tc[0]["function"]["name"] == "calculate")
check("and the expression intact",
      tc and json.loads(tc[0]["function"]["arguments"]) == {"expression": "17 * 23 - 91"},
      tc and tc[0]["function"]["arguments"])
check("the markup is gone from what the user sees", "call:" not in rest and "<tool_call|>" not in rest, rest)
check("the model's own prose survives", "Посчитаю" in rest, rest)

tc, _ = L.extract_gemma4_tool_calls('call:search{query: "погода", max_results: 3}<tool_call|>')
args = json.loads(tc[0]["function"]["arguments"]) if tc else {}
check("a spaced numeric argument is not silently dropped",
      args.get("max_results") == 3, args)
check("alongside the string argument", args.get("query") == "погода", args)
tc, _ = L.extract_gemma4_tool_calls('call:x{a: true, b: -2.5}<tool_call|>')
args = json.loads(tc[0]["function"]["arguments"]) if tc else {}
check("booleans and negative floats parse", args == {"a": True, "b": -2.5}, args)

for prose in ("Я могу call: посчитать это для вас.",
              "Он сказал: call:me{maybe}",
              "Сначала call: и потом."):
    tc, rest = L.extract_gemma4_tool_calls(prose)
    check(f"prose is not mistaken for a call: {prose[:28]!r}", tc == [] and rest == prose, tc)

# The older formats must still work — this is an addition, not a replacement.
tc, _ = L.extract_gemma4_tool_calls('<|tool_call>call:search{query:<|"|>x<|"|>}<tool_call|>')
check("Format A still parses", len(tc) == 1 and tc[0]["function"]["name"] == "search", tc)
tc, _ = L.extract_gemma4_tool_calls(
    '[TOOL_REQUEST]\n{"name":"calculate","arguments":{"expression":"2+2"}}\n[END_TOOL_REQUEST]')
check("Format B still parses", len(tc) == 1 and tc[0]["function"]["name"] == "calculate", tc)
tc, _ = L.extract_gemma4_tool_calls('call:search{query:<|"|>x<|"|>}')
check("Format C still parses", len(tc) == 1 and tc[0]["function"]["name"] == "search", tc)

p = capture("google/gemma-4-12b-qat")
check("and the detection gate knows the new shape",
      "<tool_call|>" in __import__("inspect").getsource(L.send_to_lm_studio))

print(f"\n{_n - _bad}/{_n} checks passed")
sys.exit(1 if _bad else 0)
