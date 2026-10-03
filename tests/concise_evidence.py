"""Hard evidence for the direct-mode toggle.

Intercepts the REAL HTTP request sent to LM Studio (no reconstruction) and captures,
for BOTH modes on the SAME model:
  - exact model id (from /v1/models AND from the payload)
  - full request payload (temperature, max_tokens, reasoning, chat_template_kwargs)
  - full raw response message (content AND reasoning_content)
  - token usage (prompt/completion) — the real "did it reason less" signal
  - timing

Single-round, NO tool loop, so the comparison isolates the MODEL's behaviour from
the agent's tool-orchestration loop. A second pass repeats WITH tools to show the
tool-planning effect separately.

Usage: venv/Scripts/python.exe tests/concise_evidence.py
"""
import sys, json, time, threading
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import requests
import llm as llm_mod
from config import MODEL_NAME, LM_STUDIO_URL, LM_STUDIO_BASE
from prompts import build_system_prompt
from tools import TOOL_SCHEMAS

# ---- capture the exact outbound payload + inbound json by wrapping requests.post ----
_CAP = {}
_real_post = requests.post
def _spy_post(url, *a, **kw):
    resp = _real_post(url, *a, **kw)
    if "json" in kw and isinstance(kw["json"], dict) and "messages" in kw["json"]:
        _CAP["payload"] = kw["json"]
        try: _CAP["resp"] = resp.json()
        except Exception: _CAP["resp"] = None
    return resp
llm_mod.requests.post = _spy_post


class MiniCtx:
    def __init__(self, no_think):
        self.model_name = MODEL_NAME
        self.no_think = no_think
        self.api_lock = threading.Lock()
        self.last_api_call_time = 0.0
        self.api_min_interval = 0.2


def server_model_ids():
    try:
        r = requests.get(f"{LM_STUDIO_BASE}/models", timeout=10).json()
        return [m.get("id") for m in r.get("data", [])]
    except Exception as e:
        return [f"<error {e}>"]


def run_once(no_think, task, with_tools):
    _CAP.clear()
    ctx = MiniCtx(no_think)
    temp = 0.2 if no_think else 0.5
    maxtok = 700 if no_think else 1500
    sysp = build_system_prompt("", concise=no_think)
    messages = [{"role": "system", "content": sysp},
                {"role": "user", "content": task}]
    t0 = time.time()
    if with_tools:
        msg = llm_mod.send_to_lm_studio(ctx, messages, tools=TOOL_SCHEMAS,
                                        tool_choice="auto", temperature=temp, max_tokens=maxtok)
    else:
        msg = llm_mod.send_to_lm_studio(ctx, messages, temperature=temp, max_tokens=maxtok)
    dt = time.time() - t0
    return msg, dt, dict(_CAP)


def redact_payload(p):
    """Shrink message bodies so the structural fields are readable."""
    p = json.loads(json.dumps(p))  # deep copy
    for m in p.get("messages", []):
        c = m.get("content")
        if isinstance(c, str) and len(c) > 160:
            m["content"] = c[:160] + f"... <+{len(c)-160} chars>"
    if "tools" in p:
        p["tools"] = f"<{len(p['tools'])} tool schemas>"
    return p


def show(label, no_think, task, with_tools):
    msg, dt, cap = run_once(no_think, task, with_tools)
    payload = cap.get("payload", {})
    resp = cap.get("resp", {}) or {}
    rmsg = (resp.get("choices") or [{}])[0].get("message", {}) if resp else {}
    usage = resp.get("usage", {}) if resp else {}
    print("\n" + "=" * 74)
    print(f"### {label}  (with_tools={with_tools})")
    print(f"  payload.model          : {payload.get('model')!r}")
    print(f"  payload.temperature    : {payload.get('temperature')}")
    print(f"  payload.max_tokens     : {payload.get('max_tokens')}")
    print(f"  payload.reasoning      : {payload.get('reasoning')!r}")
    print(f"  payload.chat_template_kwargs : {payload.get('chat_template_kwargs')!r}")
    print(f"  payload.reasoning_effort     : {payload.get('reasoning_effort', '<absent>')!r}")
    # system prompt tail (the concise directive presence)
    sysmsg = next((m['content'] for m in payload.get('messages', []) if m.get('role') == 'system'), '')
    print(f"  system prompt chars    : {len(sysmsg)}")
    print(f"  '[DIRECT MODE' present  : {'[DIRECT MODE' in sysmsg}")
    print(f"  '/no_think' present     : {'/no_think' in sysmsg}")
    print(f"  usage.prompt_tokens    : {usage.get('prompt_tokens')}")
    print(f"  usage.completion_tokens: {usage.get('completion_tokens')}   <-- model's generated tokens")
    rc = str(rmsg.get('reasoning_content') or '')
    ct = str(rmsg.get('content') or '')
    print(f"  raw reasoning_content len: {len(rc)}   <-- internal CoT the model actually produced")
    print(f"  raw content len          : {len(ct)}")
    print(f"  wall secs              : {dt:.1f}")
    print(f"  --- redacted payload ---")
    print("  " + json.dumps(redact_payload(payload), ensure_ascii=False)[:900])
    print(f"  --- raw reasoning_content (first 400) ---\n  {rc[:400]!r}")
    print(f"  --- raw content (first 400) ---\n  {ct[:400]!r}")
    return {"completion_tokens": usage.get("completion_tokens"),
            "reasoning_len": len(rc), "secs": dt, "model": payload.get("model")}


if __name__ == "__main__":
    print("config.MODEL_NAME :", repr(MODEL_NAME))
    print("config.LM_STUDIO_URL :", LM_STUDIO_URL)
    print("server /v1/models loaded ids:", server_model_ids())
    task = ("Объясни, стоит ли сегодня брать зонт, если за окном пасмурно "
            "и я живу в Москве.")
    print("\nTASK:", task)
    a = show("THINKING ON  (no_think=False)", False, task, with_tools=False)
    b = show("DIRECT MODE  (no_think=True)",  True,  task, with_tools=False)
    print("\n" + "#" * 74)
    print("SAME MODEL?:", a["model"] == b["model"], "->", a["model"])
    print(f"completion_tokens  ON={a['completion_tokens']}  OFF={b['completion_tokens']}")
    print(f"reasoning_content  ON={a['reasoning_len']}  OFF={b['reasoning_len']}")
    print(f"wall secs          ON={a['secs']:.1f}  OFF={b['secs']:.1f}")
