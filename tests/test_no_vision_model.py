"""A text-only model is not a blind one.

Live log 2026-09-24 00:04: muse-glimmer-30b (no vision) got the start-up vision
self-test; LM Studio answered 400 "does not support image inputs", the call was
retried 3x, the self-test logged "broken projector" and reloaded the model.
"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm
import vision_selftest as v

ok = 0
checks = []


def check(name, cond):
    global ok
    checks.append(name)
    ok += bool(cond)
    print(("PASS  " if cond else "FAIL  ") + name)


body = '{"error":{"message":"The provided messages contain images, but muse-glimmer-30b does not support image inputs."}}'
check("400 body recognised as no-vision", bool(llm._NO_VISION_RE.search(body)))
check("llama.cpp wording recognised", bool(llm._NO_VISION_RE.search("image input is not supported - hint: ...")))
check("ordinary 400 not matched", not llm._NO_VISION_RE.search('{"error":"n_keep: 9905 >= n_ctx: 8192"}'))

ctx = types.SimpleNamespace(model_name="muse-glimmer-30b")
reloads = []
orig_ask, orig_reload, orig_q = v._ask, v._reload_house_model, v.quarantine_f16_projectors
v.quarantine_f16_projectors = lambda *a, **k: []   # never touch the real models dir


def fake_ask(c):
    c.last_llm_error = "no_vision"
    return ""


v._ask = fake_ask
v._reload_house_model = lambda c: reloads.append(1) or True
os.environ.pop("F5_TEST_RUN", None)
try:
    r = v._run_once(ctx)
    check("self-test returns None (no verdict), not False", r is None)
    v._HEALED["done"] = False
    v.run(ctx)
    check("no reload for a text-only model", not reloads)
finally:
    v._ask, v._reload_house_model, v.quarantine_f16_projectors = orig_ask, orig_reload, orig_q

# --- forced tool call rejected by the grammar on Gemma's <|channel> token ---
import json, requests


class _Resp:
    def __init__(self, lines): self._l, self.status_code = lines, 200
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def iter_lines(self, decode_unicode=False): yield from self._l
    def close(self): pass


_err = ("data: " + json.dumps({"error": {"message": "Engine protocol predict request returned 400: "
        "Failed to initialize samplers: Unexpected empty grammar stack after accepting piece: <|channel> (100)"}})).encode()
_orig_post = requests.post
requests.post = lambda url, **kw: _Resp([_err])
try:
    pl = {"messages": [], "tool_choice": "required",
          "tools": [{"type": "function", "function": {"name": "generate_video"}}]}
    llm._stream_chat_raw(types.SimpleNamespace(), pl)
    check("grammar reject drops tool_choice=required to auto", pl["tool_choice"] == "auto")
    check("the single forced schema is kept", len(pl["tools"]) == 1)
    pl2 = {"messages": [], "tool_choice": "auto"}
    llm._stream_chat_raw(types.SimpleNamespace(), pl2)
    check("non-forced payload untouched", pl2["tool_choice"] == "auto")
finally:
    requests.post = _orig_post

print(f"\n{ok}/{len(checks)} checks passed")
sys.exit(0 if ok == len(checks) else 1)
