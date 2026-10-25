"""Score the picture-complaint reader (imaging/complaint_read.py) on bench/complaint_set/cases.json:
live critic complaints + the darkness/taste cases from the suites. `regex` scores the word lists
it replaced (exposure.DARK_RE / BLIND_RE / _SUBSTANTIVE_RE, draw_agent._LAYOUT_CONCERN_RE).
A real complaint dropped (content read as taste or dark) is the costly error: the flaw ships.
    venv/Scripts/python.exe bench/complaint_eval.py [regex]"""
import json, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("F5_TEST_RUN", "1")
os.environ["INTENT_LIVE"] = "1"
sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", d) for d in ("imaging", "agent", "core", ".")]

cases = json.load(open(os.path.join(os.path.dirname(__file__), "complaint_set", "cases.json"), encoding="utf-8"))
t = time.time()
if sys.argv[1:] == ["regex"]:
    import exposure as E, draw_agent as D
    got = ["blind" if E.BLIND_RE.search(c["text"]) else "dark" if E.is_dark_complaint(c["text"])
           else "content" if D._LAYOUT_CONCERN_RE.search(c["text"]) else "taste" for c in cases]
else:
    import complaint_read
    got = []
    for i in range(0, len(cases), 20):
        got += complaint_read.kinds(None, [c["text"] for c in cases[i:i + 20]])
ok = dropped = 0
for c, g in zip(cases, got):
    ok += g in c["ok"]
    dropped += "content" in c["ok"] and c["ok"][0] == "content" and g != "content"
    if g not in c["ok"]:
        print(f"MISS want {c['ok']} got {g}: {c['text'][:110]}")
print(f"{ok}/{len(cases)} right, {dropped} real complaints dropped, {time.time() - t:.0f} s")
