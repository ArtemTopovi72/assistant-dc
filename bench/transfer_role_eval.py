"""Score image_transfer.infer_reference_roles on bench/transfer_role_set/cases.json: what a
reference picture gives an edit. `regex` scores the keyword lists it replaced (2026-10-25).
Needs LM Studio up.
    venv/Scripts/python.exe bench/transfer_role_eval.py [regex]"""
import json, os, re, sys, time
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("F5_TEST_RUN", "1")
os.environ["INTENT_LIVE"] = "1"
for d in ("", "agent", "imaging", "core", "media"):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", d))
import image_transfer as T

_OLD = [("clothing", r"\b(dress|shirt|jacket|coat|outfit|clothes|garment|suit|hoodie|sweater|uniform|costume|pants|skirt|jeans|blouse)\b"),
        ("hair", r"\b(hair|hairstyle|haircut|hairdo)\b"),
        ("object", r"\b(hat|cap|glasses|sunglasses|watch|bag|necklace|earrings?|product|bottle|logo|prop|accessory|scarf|tie|crown|helmet|mask|furniture|chair|sofa|lamp|table)\b"),
        ("face", r"\b(face|facial features|likeness|look like (?:him|her|them))\b"),
        ("pose", r"\b(pose|posture|stance|position|gesture)\b"),
        ("style", r"\b(style|material|texture|pattern|palette|colou?r scheme|aesthetic|vibe)\b"),
        ("scene", r"\b(scene|background|environment|setting|room|place (?:it|them) in)\b"),
        ("lighting", r"\b(lighting|light|mood|shadows?|illumination)\b"),
        ("identity", r"\b(identity|character|same person)\b")]
_BY_ROLE = {v[0]: k for k, v in T._ROLE_READS.items()}

cases = json.load(open(os.path.join(os.path.dirname(__file__), "transfer_role_set", "cases.json"), encoding="utf-8"))
t = time.time()
ok = 0
for c in cases:
    if sys.argv[1:] == ["regex"]:
        g = next((k for k, rx in _OLD if re.search(rx, c["text"], re.I)), "object")
    else:
        ref = T.ReferenceImage("x.png", role=None)
        T.infer_reference_roles(c["text"], [ref])
        g = _BY_ROLE.get(ref.role, "?")
    ok += g == c["ok"]
    if g != c["ok"]:
        print(f"MISS want {c['ok']} got {g}: {c['text']}")
print(f"{ok}/{len(cases)} right, {time.time() - t:.0f} s")
