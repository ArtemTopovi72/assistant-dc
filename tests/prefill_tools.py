"""Does prefill=<think></think> survive WITH tools? (the only way to kill the
think block inside the agent's tool loop). Capture whether tool_calls still return."""
import sys, json, threading
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import requests, llm as llm_mod
from config import MODEL_NAME
from prompts import build_system_prompt
from tools import TOOL_SCHEMAS

_CAP={}
_real=requests.post
def _spy(u,*a,**k):
    r=_real(u,*a,**k)
    if "json" in k and isinstance(k["json"],dict) and "messages" in k["json"]:
        _CAP["p"]=k["json"]
        try:_CAP["r"]=r.json()
        except Exception:_CAP["r"]=None
    return r
llm_mod.requests.post=_spy

class C:
    def __init__(s):s.model_name=MODEL_NAME;s.no_think=True;s.api_lock=threading.Lock();s.last_api_call_time=0.0;s.api_min_interval=0.2

TASK="Найди в интернете текущую погоду в Москве."  # clearly needs the search tool

def go(label, prefill):
    _CAP.clear()
    msgs=[{"role":"system","content":build_system_prompt("",concise=True)},
          {"role":"user","content":TASK}]
    msg=llm_mod.send_to_lm_studio(C(),msgs,tools=TOOL_SCHEMAS,tool_choice="auto",
                                  temperature=0.2,max_tokens=700,prefill=prefill)
    r=_CAP.get("r") or {}
    rm=(r.get("choices") or [{}])[0].get("message",{})
    usage=r.get("usage",{})
    raw=str(rm.get("content") or "")
    tcs=msg.get("tool_calls") if msg else None
    print(f"\n### {label}  prefill={prefill!r}")
    print(f"  HTTP ok: {bool(r)}  completion_tokens={usage.get('completion_tokens')}")
    print(f"  tool_calls returned: {len(tcs) if tcs else 0} -> {[t['function']['name'] for t in (tcs or [])]}")
    print(f"  raw content has <think>: {'<think>' in raw}")
    print(f"  raw content (first 200): {raw[:200]!r}")

if __name__=="__main__":
    print("MODEL:",MODEL_NAME,"\nTASK:",TASK)
    go("no prefill (current loop behaviour)", None)
    go("prefill=<think></think> + tools", "<think></think>")
