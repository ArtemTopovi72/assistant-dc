"""Backend-truth capture of the reasoning toggle, straight from LM Studio.

Builds the EXACT payload send_to_lm_studio() constructs for each mode, POSTs it
raw, and records usage + unstripped content + tool_calls. No app-side stripping.

  OFF  = reasoning shown : no prefill, temp 0.5, max_tokens 1500, no_think False
  ON   = direct mode     : prefill '<think></think>', temp 0.2, max_tokens 700,
                           no_think True (enable_thinking False, /no_think appended)

Writes docs/reasoning_runtime_evidence.json and prints a summary table.
"""
import json, sys, copy
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import requests
from config import LM_STUDIO_URL, MODEL_NAME
from llm import _apply_no_think

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]

SYSTEM = "You are a helpful assistant with tools. Use a tool when it fits."
# Two prompts: a simple tool prompt, and a reasoning-heavy planning prompt that
# makes the model think a lot in OFF mode (so the token delta is visible).
USER = "What's the weather in Tokyo right now? Use the tool."
USER_HEAVY = (
    "I'm planning a 12-day trip across Tokyo, Paris and New York with a tight "
    "budget. Work out a sensible visiting order to minimise backtracking given "
    "typical flight routes, decide how many days in each, and explain your "
    "reasoning step by step. Also check the weather in the first city."
)


def build_payload(*, no_think: bool, prefill, temperature, max_tokens, user=USER):
    """Replicates send_to_lm_studio()'s payload construction exactly."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
    outbound = _apply_no_think(messages) if no_think else list(messages)
    if prefill is not None:
        outbound = list(outbound) + [{"role": "assistant", "content": prefill}]
    payload = {
        "model": MODEL_NAME,
        "messages": outbound,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "reasoning": "off",
    }
    if no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    payload["tools"] = TOOLS
    payload["tool_choice"] = "auto"
    return payload


def run(label, **kw):
    payload = build_payload(**kw)
    r = requests.post(LM_STUDIO_URL, json=payload, timeout=600)
    data = r.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    rc = msg.get("reasoning_content")
    tcs = msg.get("tool_calls") or []
    usage = data.get("usage", {})
    rec = {
        "label": label,
        "request_payload": payload,
        "raw_response": data,
        "model_id_returned": data.get("model"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_content_present": rc is not None and str(rc).strip() != "",
        "think_in_content": "<think>" in content,
        "content": content,
        "tool_calls": [{"name": tc.get("function", {}).get("name"),
                        "arguments": tc.get("function", {}).get("arguments")} for tc in tcs],
    }
    return rec


def report(off, on, title):
    hdr = f"{'metric':<28}{'OFF':<16}{'ON (prefill)':<16}"
    print(f"\n=== {title} ===")
    print(hdr); print("-" * len(hdr))
    rows = [
        ("model id (response)", "model_id_returned"),
        ("prompt_tokens", "prompt_tokens"),
        ("completion_tokens", "completion_tokens"),
        ("total_tokens", "total_tokens"),
        ("reasoning_content present", "reasoning_content_present"),
        ("<think> in content", "think_in_content"),
        ("tool_calls", None),
    ]
    for label, key in rows:
        if key is None:
            a = ", ".join(t["name"] for t in off["tool_calls"]) or "none"
            b = ", ".join(t["name"] for t in on["tool_calls"]) or "none"
        else:
            a, b = off[key], on[key]
        print(f"{label:<28}{str(a):<16}{str(b):<16}")
    ct_off, ct_on = off["completion_tokens"], on["completion_tokens"]
    if ct_off and ct_on:
        print(f"completion-token drop:      {ct_off} -> {ct_on}  ({(1-ct_on/ct_off)*100:.0f}% fewer)")


if __name__ == "__main__":
    print(f"LM_STUDIO_URL = {LM_STUDIO_URL}")
    print(f"config MODEL_NAME = {MODEL_NAME}")

    # Scenario A: simple tool prompt
    a_off = run("A/OFF", no_think=False, prefill=None, temperature=0.5, max_tokens=1500)
    a_on = run("A/ON", no_think=True, prefill="<think></think>", temperature=0.2, max_tokens=700)
    # Scenario B: reasoning-heavy planning prompt (still tool-enabled)
    b_off = run("B/OFF", no_think=False, prefill=None, temperature=0.5, max_tokens=1500, user=USER_HEAVY)
    b_on = run("B/ON", no_think=True, prefill="<think></think>", temperature=0.2, max_tokens=700, user=USER_HEAVY)

    out = ROOT / "docs" / "reasoning_runtime_evidence.json"
    out.write_text(json.dumps(
        {"scenario_A_simple_tool": {"off": a_off, "on": a_on},
         "scenario_B_reasoning_heavy": {"off": b_off, "on": b_on}},
        indent=2, ensure_ascii=False), encoding="utf-8")

    report(a_off, a_on, "Scenario A: simple tool prompt (Tokyo weather)")
    report(b_off, b_on, "Scenario B: reasoning-heavy planning prompt (+tool)")
    print(f"\nfull raw payloads+responses -> {out}")
    print(f"\nB/OFF content head: {b_off['content'][:200]!r}")
    print(f"B/ON  content head: {b_on['content'][:200]!r}")
    print(f"B/OFF tool_calls: {[t['name'] for t in b_off['tool_calls']]}  "
          f"B/ON tool_calls: {[t['name'] for t in b_on['tool_calls']]}")
