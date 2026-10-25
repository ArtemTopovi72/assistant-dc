"""Score tool_image_handlers.asks_for_video on bench/asks_video_set/cases.json: do the user's
words ask for a clip. `regex` puts back the word list that answered yes before the model read
(removed 2026-10-25). Needs LM Studio up.
    venv/Scripts/python.exe bench/asks_video_eval.py [regex]"""
import json, os, re, sys, time
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("F5_TEST_RUN", "1")
os.environ["INTENT_LIVE"] = "1"
for d in ("", "agent", "imaging", "media", "bot", "core", "voice"):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", d))
import tool_image_handlers as H

_OLD = re.compile(r"видео|видос|ролик|клип|аними|анимац|ожив|кружок|кружоч|мультфильм|мультик\b|"
                  r"\bгиф|\bgif|video|clip|animat|movie|footage|\breel|shorts|фильм", re.I)


class _Ctx:
    last_video_path = None


cases = json.load(open(os.path.join(os.path.dirname(__file__), "asks_video_set", "cases.json"), encoding="utf-8"))
t = time.time()
ok = 0
for c in cases:
    st = {"user_input_original": c["text"], "user_input": c["text"]}
    g = bool(sys.argv[1:] == ["regex"] and _OLD.search(c["text"])) or H.asks_for_video(_Ctx(), st)
    ok += g == c["ok"]
    if g != c["ok"]:
        print(f"MISS want {c['ok']} got {g}: {c['text']}")
print(f"{ok}/{len(cases)} right, {time.time() - t:.0f} s")
