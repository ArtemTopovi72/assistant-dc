"""Past write_file bodies are stubbed in the OUTBOUND copy only (context room)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")
import graph_personality as G

ok = []


def check(c, m):
    ok.append(bool(c)); print(("ok   " if c else "FAIL ") + m)


big = "def f():\n    pass\n" * 200
msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "write_file",
             "arguments": json.dumps({"path": "rx.py", "content": big})}},
            {"id": "2", "type": "function", "function": {"name": "run_code",
             "arguments": json.dumps({"code": big})}}]},
        {"role": "tool", "tool_call_id": "1", "content": "Wrote rx.py."}]
out = G._stub_written_files(msgs)
a = json.loads(out[2]["tool_calls"][0]["function"]["arguments"])
check(a["path"] == "rx.py" and "401 lines written to rx.py" in a["content"], a["content"])
check(len(out[2]["tool_calls"][0]["function"]["arguments"]) < 200, "the write_file call shrinks to a stub")
check(json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"])["content"] == big, "persisted history untouched")
check(json.loads(out[2]["tool_calls"][1]["function"]["arguments"])["code"] == big, "only write_file is stubbed")
small = [{"role": "assistant", "tool_calls": [{"function": {"name": "write_file",
          "arguments": json.dumps({"path": "a.py", "content": "x=1"})}}]}]
check(G._stub_written_files(small) is small, "short bodies kept as they are")
bad = [{"role": "assistant", "tool_calls": [{"function": {"name": "write_file", "arguments": '{"_truncated": true}'}}]}]
check(G._stub_written_files(bad) is bad, "truncation marker passes through")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
