"""Live check of agent/intent.read on the house model: does the message need a
tool? Same labelled set as the old keyword gate + embedding probe
(bench/probe/need_tool_probe.py), so the three can be compared.

    venv/Scripts/python bench/intent_live.py
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "agent")]
sys.stdout.reconfigure(encoding="utf-8")
import re
src = open(os.path.join(ROOT, "bench", "probe", "need_tool_probe.py"), encoding="utf-8").read()
EXTRA = re.search(r'EXTRA_NO_TOOL = """(.*?)"""', src, re.S).group(1).strip().splitlines()
from bench.tc_cases import cases_for
from bench.tc_para import PARA
rows = [(c["text"], 0 if c["cat"] == "gate" else 1, c["cat"]) for c in cases_for(
    ["gate", "route", "args", "chain", "recover", "safety"])]
rows += [(p["text"], 1, "para") for p in PARA] + [(t, 0, "chat") for t in EXTRA]
import intent
bad, ts = [], []
for text, want, cat in rows:
    t = time.time(); got = intent.read(None, text); ts.append(time.time() - t)
    if int(got["needs_tool"]) != want or not got["ok"]:
        bad.append((cat, want, got, text))
for b in bad:
    print("FAIL", b)
ts.sort()
print(f"{len(rows) - len(bad)}/{len(rows)}  median {ts[len(ts)//2]:.2f}s  p90 {ts[int(len(ts)*.9)]:.2f}s")
