"""Score image_sizing.frame_read on bench/frame_read_set/cases.json: is a picture FORMAT asked
for (wide / tall / square) or not (none: the scene's lean is not scored). `regex` scores the
word lists it replaced (tool_args, 2026-10-25). Needs LM Studio up.
    venv/Scripts/python.exe bench/frame_read_eval.py [regex]"""
import json, os, re, sys, time
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("F5_TEST_RUN", "1")
os.environ["INTENT_LIVE"] = "1"
sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", d) for d in ("imaging", "agent", "core", ".")]

cases = json.load(open(os.path.join(os.path.dirname(__file__), "frame_read_set", "cases.json"), encoding="utf-8"))
t = time.time()
if sys.argv[1:] == ["regex"]:
    def read(d):
        wide = re.search(r"\b(?:horizontal|landscape (?:format|orientation|image|shot)|widescreen|"
                         r"wide (?:format|image|banner|cover)|youtube|banner|16:9)\b", d, re.I)
        tall = re.search(r"\b(?:vertical|stor(?:y|ies)|reels?|tiktok|9:16)\b", d, re.I)
        return "wide" if wide and not tall else "tall" if tall and not wide else "none"
else:
    import image_sizing
    def read(d):
        got = image_sizing.frame_read(d, None)
        return got if got in ("wide", "tall", "square") else "none"
ok = 0
for c in cases:
    g = read(c["text"])
    ok += g == c["ok"]
    if g != c["ok"]:
        print(f"MISS want {c['ok']} got {g}: {c['text']}")
print(f"{ok}/{len(cases)} right, {time.time() - t:.0f} s")
