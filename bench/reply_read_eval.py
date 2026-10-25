"""Score the reply-claims read on labelled replies (bench/reply_read_set/cases.json): 120 live
replies from runtime/failed_turns + 30 documented failures. `regex` scores the word lists it
replaced, mapped onto the same fields.
    venv/Scripts/python.exe bench/reply_read_eval.py [regex]"""
import json, os, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("F5_TEST_RUN", "1")
os.environ["INTENT_LIVE"] = "1"
sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", d) for d in ("agent", "core", "services", ".")]

cases = json.load(open(os.path.join(os.path.dirname(__file__), "reply_read_set", "cases.json"), encoding="utf-8"))
BOOLS = ("delivers", "promises", "already", "computed", "admits_failure", "denies", "tool_as_text", "does_tool_job")


def by_regex(reply, user):
    import graph_finalize as F, graph_fastpath as P, turn_audit as A
    ids = {c for c, rx, *_ in A._CLAIMS if rx.search(reply)}
    m = {"cart": "cart", "remember": "remember", "reminder": "reminder", "reminder_cancel": "reminder_cancel",
         "wrote_down": "reminder", "forget": "forget", "location": "pickup_point", "ran_code": "ran_code",
         "edited_file": "edited_file"}
    did = {m[i] for i in ids}
    if F._FILE_READY_RE.search(reply):
        did.add("file")
    if (F._ACTION_CLAIM_RE.search(reply) or P._FAST_PATH_ACTION_CLAIM_RE.search(reply)) and not did:
        did.add("other")
    return {"did": sorted(did), "delivers": bool(P._FAST_PATH_DELIVERY_CLAIM_RE.search(reply)),
            "promises": bool(F._FILE_PROMISE_RE.search(reply)), "already": bool(F._ALREADY_DONE_RE.search(reply)),
            "computed": bool(F._STATED_RESULT_RE.search(reply)), "admits_failure": bool(F._ADMISSION_RE.search(reply)),
            "denies": bool(P._FAST_PATH_CAPABILITY_DENIAL_RE.search(reply)),
            "tool_as_text": '"action' in reply or "<tool_call>" in reply, "does_tool_job": False}


def score(want, got):
    """{field: hit or None (not scored)}"""
    out = {}
    w = want.get("did", [])
    if w != "?":
        g = set(got["did"])
        out["did"] = bool(g) if w == "*" else g == set(w)
    for f in BOOLS:
        w = want.get(f, False)
        if w != "?":
            out[f] = got[f] == w
    if "claims_result" in want:          # made it or hands it over -- either reading will do
        out["claims_result"] = bool(got["did"] or got["delivers"]) == want["claims_result"]
    if "falls_through" in want:          # graph_fastpath: the draft leaves the tool-less path
        out["falls_through"] = bool(got["delivers"] or got["did"] or got["promises"] or got["tool_as_text"]
                                    or got["does_tool_job"]) == want["falls_through"]
    return out


if sys.argv[1:] == ["regex"]:
    reader = by_regex
else:
    import reply_read
    reader = lambda reply, user: reply_read.read(None, reply, user)
t = time.time()
with ThreadPoolExecutor(4) as ex:
    got = list(ex.map(lambda c: reader(c["reply"], c["user"]), cases))
hits, n = Counter(), Counter()
whole = 0
for c, g in zip(cases, got):
    s = score(c["want"], g)
    whole += all(s.values())
    for f, ok in s.items():
        n[f] += 1
        hits[f] += ok
        if not ok:
            print(f"MISS {f}: want {c['want'].get(f, [] if f == 'did' else False)} "
                  f"got {g.get(f, g['did'] or g['delivers'])} | {c['reply'][:90]!r}")
print("\n".join(f"  {f:15} {hits[f]}/{n[f]}" for f in n))
print(f"{whole}/{len(cases)} replies read right on every field, {time.time() - t:.0f} s")
