"""Score video.read_sentences on bench/script_read_set/cases.json: each shot-script sentence's
kind (action / look / once), its beats (a [low, high] range), a held action, and whether a
camera move is cut off an action of its own. `regex` scores the word rules it replaced, now
kept only as the suites' scripted answer (tests/fakes.script_model). Needs LM Studio up.
    venv/Scripts/python.exe bench/script_read_eval.py [regex]"""
import json, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("F5_TEST_RUN", "1")
os.environ["INTENT_LIVE"] = "1"
for d in ("", "agent", "imaging", "core", "media", "tests"):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", d))
import video as V

cases = json.load(open(os.path.join(os.path.dirname(__file__), "script_read_set", "cases.json"), encoding="utf-8"))
if sys.argv[1:] == ["regex"]:
    import fakes
    V._script_model = fakes.script_model
t = time.time()
reads = V.read_sentences([c["s"] for c in cases])
ok = 0
for c, r in zip(cases, reads):
    split = bool(r["camera"] and r["rest"])
    good = split == c.get("split", False) and (split or r["kind"] == c["kind"])
    if good and c["kind"] == "action" and not split:
        lo, hi = c.get("beats", [1, 12])
        good = lo <= r["beats"] <= hi and r["hold"] == c.get("hold", False)
    ok += good
    if not good:
        print(f"MISS want {c['kind']} {c.get('beats')} hold={c.get('hold', False)} split={c.get('split', False)} "
              f"got {r['kind']} {r['beats']} hold={r['hold']} split={split}: {c['s']}")
print(f"{ok}/{len(cases)} right, {time.time() - t:.0f} s")
