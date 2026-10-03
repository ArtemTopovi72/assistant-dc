"""Find a lever that ACTUALLY stops this finetune from generating <think>.

Tests four configs on the same model/task (no tools), capturing whether the raw
content still contains a <think> block and how many completion tokens were spent.
"""
import sys, json, time, threading
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import requests
import llm as llm_mod
from config import MODEL_NAME
from prompts import build_system_prompt

_CAP = {}
_real = requests.post
def _spy(url, *a, **kw):
    r = _real(url, *a, **kw)
    if "json" in kw and isinstance(kw["json"], dict) and "messages" in kw["json"]:
        _CAP["p"] = kw["json"]
        try: _CAP["r"] = r.json()
        except Exception: _CAP["r"] = None
    return r
llm_mod.requests.post = _spy

class C:
    def __init__(s, nt): s.model_name=MODEL_NAME; s.no_think=nt; s.api_lock=threading.Lock(); s.last_api_call_time=0.0; s.api_min_interval=0.2

TASK = "Сколько будет 18 умножить на 24? Дай только число."

def go(label, no_think, prefill):
    _CAP.clear()
    ctx=C(no_think)
    sysp=build_system_prompt("", concise=no_think)
    msgs=[{"role":"system","content":sysp},{"role":"user","content":TASK}]
    t0=time.time()
    msg=llm_mod.send_to_lm_studio(ctx,msgs,temperature=0.2,max_tokens=700,prefill=prefill)
    dt=time.time()-t0
    r=_CAP.get("r") or {}
    rm=(r.get("choices") or [{}])[0].get("message",{})
    usage=r.get("usage",{})
    raw=str(rm.get("content") or "")
    has_think="<think>" in raw or "</think>" in raw
    print(f"\n### {label}")
    print(f"  prefill={prefill!r}  no_think={no_think}")
    print(f"  completion_tokens={usage.get('completion_tokens')}  secs={dt:.1f}")
    print(f"  raw content has <think>: {has_think}")
    print(f"  raw content (first 220): {raw[:220]!r}")
    print(f"  final (stripped) content: {(msg.get('content') if msg else '')[:120]!r}")
    return usage.get("completion_tokens"), has_think

if __name__=="__main__":
    print("MODEL:",MODEL_NAME,"\nTASK:",TASK)
    go("A no_think=False, no prefill", False, None)
    go("B no_think=True,  no prefill", True, None)
    go("C no_think=True,  prefill=<think></think>", True, "<think></think>")
    go("D no_think=False, prefill=<think></think>", False, "<think></think>")
