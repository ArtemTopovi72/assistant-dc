"""Score draw_agent._mismatches on bench/mismatch_read_set/cases.json: which critic complaint
says a layout element came out with the wrong look, and the look it should have. `regex`
scores the «X is Y instead of Z» pattern it replaced (2026-10-25). Needs LM Studio up.
    venv/Scripts/python.exe bench/mismatch_read_eval.py [regex]"""
import json, os, re, sys, time
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("F5_TEST_RUN", "1")
os.environ["INTENT_LIVE"] = "1"
for d in ("", "agent", "imaging", "core", "media"):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", d))
import draw_agent as D

_OLD = re.compile(
    r"(?:^|\b)(?:the |a |an )?(?P<what>[\w][\w' -]{0,40}?)\s+(?:is|are|was|were|came out|appears?|looks?|rendered)\s+"
    r"(?P<got>[\w][\w /-]{0,30}?)\s+(?:instead of|rather than|not)\s+(?P<want>[\w][\w /-]{0,40}?)(?:[.;,]|$)", re.I)


def old(els, p):
    m = _OLD.search(p)
    if not m:
        return 0, ""
    want, head = m.group("want").strip().rstrip("."), m.group("what").strip().lower().split()[-1]
    for i, d in enumerate(els, 1):
        if re.search(r"\b" + re.escape(head) + r"\b", d.lower()) and want.lower() in d.lower():
            return i, want
    for i, d in enumerate(els, 1):
        if want.lower() in d.lower():
            return i, want
    return 0, ""


cases = json.load(open(os.path.join(os.path.dirname(__file__), "mismatch_read_set", "cases.json"), encoding="utf-8"))
t = time.time()
ok = 0
for c in cases:
    if sys.argv[1:] == ["regex"]:
        n, want = old(c["els"], c["p"])
    else:
        k, want, _got = D._mismatches(None, [{"desc": d} for d in c["els"]], [c["p"]])[0]
        n = k + 1
    good = n == c["el"] and (n == 0 or c["want"].lower() in want.lower())
    ok += good
    if not good:
        print(f"MISS want {c['el']} {c.get('want', '')!r} got {n} {want!r}: {c['p']}")
print(f"{ok}/{len(cases)} right, {time.time() - t:.0f} s")
