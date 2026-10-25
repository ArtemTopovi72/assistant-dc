"""Score slides._request_read on bench/deck_read_set/cases.json: the user's own figures, a pure
reorder/delete (with the positions a bare delete names), or new content. `regex` scores the
word lists it replaced (2026-10-25). Needs LM Studio up.
    venv/Scripts/python.exe bench/deck_read_eval.py [regex]"""
import json, os, re, sys, time
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("F5_TEST_RUN", "1")
os.environ["INTENT_LIVE"] = "1"
for d in ("", "agent", "services", "core", "imaging"):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", d))
import slides

_OWN = re.compile(r"\b(?:наш\w*|мо[йяеи]\w*|свое\w*|our|my)\b", re.I)
_STRUCT = re.compile(r"поменя\w* местами|переставь|перенеси|удали|убери|swap|reorder|move|delete|remove", re.I)
_DROP = re.compile(
    r"^\W*(?:удали|убери|delete|remove)\s+(?:the\s+)?(последн\w*|перв\w*|last|first|(\d+))(?:-?\w{0,3})?\s+(?:слайд|slide)\w*\W*$"
    r"|^\W*(?:удали|убери|delete|remove)\s+(?:слайд|slide)\s+(?:№\s*)?(\d+)\W*$", re.I)


def old(t):
    if _OWN.search(t) and len(re.findall(r"\d+(?:[.,]\d+)?", t)) >= 3:
        return {"kind": "own_figures", "drop": []}
    m = _DROP.match(t.strip())
    drop = []
    if m:
        num = m.group(2) or m.group(3)
        drop = [int(num)] if num else [1 if m.group(1).lower().startswith(("перв", "first")) else -1]
    return {"kind": "structural" if _STRUCT.search(t) else "content", "drop": drop}


cases = json.load(open(os.path.join(os.path.dirname(__file__), "deck_read_set", "cases.json"), encoding="utf-8"))
t = time.time()
ok = 0
for c in cases:
    g = old(c["text"]) if sys.argv[1:] == ["regex"] else slides._request_read(None, c["text"])
    good = g["kind"] == c["kind"] and sorted(g["drop"]) == sorted(c.get("drop", []))
    ok += good
    if not good:
        print(f"MISS want {c['kind']} {c.get('drop', [])} got {g['kind']} {g['drop']}: {c['text']}")
print(f"{ok}/{len(cases)} right, {time.time() - t:.0f} s")
