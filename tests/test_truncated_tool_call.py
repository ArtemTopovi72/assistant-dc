"""A tool call cut off at the token ceiling must not poison the turn.

Live 2026-09-15 19:11: a 300-line matplotlib script did not fit the 1500-token
loop ceiling, the call arrived with half a JSON string ("Failed to parse tool
call"), the half-written call stayed in history and LM Studio answered 500 to
the next nine requests. Now: rounds that may write code get a bigger ceiling,
a cut call is answered with how to write long code in parts, its arguments are
replaced in history, and a bare 500 mends the payload before retrying.
"""
import os, sys, json
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm
import graph_personality as GP
import test_agent_loop_hardening as H

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

# --- the payload mender ----------------------------------------------------------
good = {"role": "assistant", "content": "", "tool_calls": [
    {"id": "a", "type": "function", "function": {"name": "calculate", "arguments": '{"expression": "1+1"}'}}]}
bad = {"role": "assistant", "content": "", "tool_calls": [
    {"id": "b", "type": "function", "function": {"name": "run_code", "arguments": '{"code": "import matplotlib\\nfig = plt.sub'}}]}
payload = {"messages": [{"role": "system", "content": "s"}, good, {"role": "tool", "tool_call_id": "a", "content": "2"},
                        bad, {"role": "tool", "tool_call_id": "b", "content": "[TOOL ERROR] cut"}]}
check("half a JSON string is mended", llm._sanitize_tool_calls(payload)
      and json.loads(bad["tool_calls"][0]["function"]["arguments"]) == {"_truncated": True})
check("a valid call is left alone", good["tool_calls"][0]["function"]["arguments"] == '{"expression": "1+1"}')
check("nothing to mend -> False", not llm._sanitize_tool_calls(payload))
check("dict arguments count as valid",
      not llm._sanitize_tool_calls({"messages": [{"role": "assistant", "tool_calls": [
          {"function": {"name": "x", "arguments": {"a": 1}}}]}]}))

src = open(llm.__file__, encoding="utf-8").read()
check("a bare 500 is reported as a server error, not swallowed",
      "return _SERVER_ERROR if resp.status_code >= 500 else None" in src)
check("...and the retry loop mends the payload first",
      "if _sanitize_tool_calls(payload) or _shrink_payload(payload):" in src)

# --- through the loop: the cut call -----------------------------------------------
seen_tokens = []
_orig_send = H._send
def _send(ctx, messages, tools=None, **k):
    seen_tokens.append((k.get("max_tokens"), sorted((t.get("function") or {}).get("name") for t in (tools or []))))
    return _orig_send(ctx, messages, tools=tools, **k)
import graph as graph_mod
graph_mod.send_to_lm_studio = _send
llm.send_to_lm_studio = _send
half = '{"code": "import matplotlib.pyplot as plt\\n' + "x = 1\\n" * 400
def fn(i):
    if i == 1:
        m = H._msg(tcs=[{"id": "c1", "type": "function", "function": {"name": "run_code", "arguments": half}}])
        m["finish_reason"] = "length"
        return m
    return H._msg("Скрипт слишком длинный, запишу его в файл частями.")
import types, tempfile
from pathlib import Path
import sandbox_access as _sa
_orig_may = _sa.may_run_code
_sa.may_run_code = lambda u: True
import intent   # the model reads the request as code to run (agent/intent.py)
intent.STUB = lambda t: {"needs_tool": True, "wants": ["run_code"]} if "python script" in t else None
def _run_with_sandbox(fn, user):
    """H._run with a sandbox on the ctx so the code tools are offered."""
    H._script["fn"] = fn; H._script["n"] = 0
    c = H._ctx()
    c.sandbox = types.SimpleNamespace(root=Path(tempfile.mkdtemp()), user_id=1)
    c.sandbox_user = None
    g = graph_mod.build_graph(c)
    st = H._base_state(); st["user_input"] = user
    return c, g.invoke(st)
try:
    _, final = _run_with_sandbox(fn, user="make a proper flowchart with a python script and send it")
finally:
    _sa.may_run_code = _orig_may
msgs = final["messages"]
tool_msgs = [m for m in msgs if m.get("role") == "tool"]
check("the cut call is answered with how to write long code in parts",
      tool_msgs and "cut off" in tool_msgs[0]["content"] and "write_file" in tool_msgs[0]["content"]
      and "edit_file" in tool_msgs[0]["content"], tool_msgs[:1])
stored = [m for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")]
check("the half-written arguments do not stay in history",
      stored and stored[0]["tool_calls"][0]["function"]["arguments"] == GP._TRUNCATED_ARGS,
      stored and stored[0]["tool_calls"][0]["function"]["arguments"][:60])
check("the turn still ends with an answer", "частями" in (final.get("final_answer") or ""), final.get("final_answer"))

# --- the ceiling ---------------------------------------------------------------------
builder_rounds = [t for t, names in seen_tokens if "run_code" in names or "write_file" in names]
plain_rounds = [t for t, names in seen_tokens if names and "run_code" not in names and "write_file" not in names]
check("a round that may write code gets room for the code",
      builder_rounds and all(t >= GP.BUILDER_LOOP_TOKENS for t in builder_rounds), seen_tokens)

# a plain malformed call (not cut) keeps the old short refusal
seen_tokens.clear()
def fn2(i):
    if i == 1:
        return H._msg(tcs=[{"id": "m1", "type": "function", "function": {"name": "calculate", "arguments": "not json"}}])
    return H._msg("ok")
_, final2 = H._run(fn2, user="2+2")
tm = [m for m in final2["messages"] if m.get("role") == "tool"]
check("a short malformed call keeps the plain refusal", tm and "Malformed" in tm[0]["content"], tm[:1])

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
