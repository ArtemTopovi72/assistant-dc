"""emopack: reference lookup, idempotent build (Seed-VC stubbed), speak(emotion=) swaps the reference."""
import json, os, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "voice", "agent"):
    sys.path.insert(0, str(ROOT / sub))
os.environ["F5_TEST_RUN"] = "1"
import emopack
import voice_clone

fails = []
def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond: fails.append(name)

tmp = Path(tempfile.mkdtemp()); ref = tmp / "v.wav"; ref.write_bytes(b"x")
check("no pack -> None", emopack.ref_for(str(ref), "joy") is None)
d = emopack.pack_dir(str(ref)); d.mkdir()
(d / "pack.json").write_text(json.dumps({"joy": [{"wav": "a.wav", "text": "ta"}, {"wav": "b.wav", "text": "tb"}]}), encoding="utf-8")
check("has_pack", emopack.has_pack(str(ref)))
check("joy ref", emopack.ref_for(str(ref), "joy") == ("a.wav", "ta"))
check("variant wraps", emopack.ref_for(str(ref), "joy", 3) == ("b.wav", "tb"))
check("neutral = plain", emopack.ref_for(str(ref), "neutral") is None)
check("unknown emotion = plain", emopack.ref_for(str(ref), "anger") is None)
check("build is idempotent when a pack exists", emopack.build(str(ref)) is True)

# delivery.pick: the reference follows the line's emotion
import audio, delivery, intent
class C: delivery_emotion = None; delivery_ik3 = None
c = C()
c.delivery_emotion = "joy"
check("forced joy picks the pack ref", delivery.pick(c, str(ref), "plain", "Какой замечательный день!") == ("a.wav", "ta"))
c.delivery_emotion = ""
check("forced plain keeps the ref", delivery.pick(c, str(ref), "plain", "x") == (str(ref), "plain"))
c.delivery_emotion = None
intent.CHOICE_STUB = lambda q, t: "joy" if "ура" in t.lower() else "neutral"
check("auto: model says joy", delivery.pick(c, str(ref), "plain", "Ура, мы победили всех!") == ("a.wav", "ta"))
check("auto: model says neutral", delivery.pick(c, str(ref), "plain", "Поезд отправляется в девять.") == (str(ref), "plain"))
check("short line costs no call", delivery.pick(c, str(ref), "plain", "Ура!") == (str(ref), "plain"))
os.environ["EMOTION_DELIVERY"] = "0"
check("off switch", delivery.pick(c, str(ref), "plain", "Ура, мы победили всех!") == (str(ref), "plain"))
del os.environ["EMOTION_DELIVERY"]

# speak() hands explicit controls to the synthesiser through the ctx
seen = {}
audio.synth_single_segment = lambda cc, i, a, t, out_stem=None: seen.update(e=cc.delivery_emotion, k=cc.delivery_ik3, r=cc.custom_ref_wav) or "x.wav"
voice_clone.speak(object(), str(ref), "plain", "привет", str(tmp), emotion="anger", ik3=False)
check("speak passes emotion/ik3", seen == {"e": "anger", "k": False, "r": str(ref)})
voice_clone.speak(object(), str(ref), "plain", "привет", str(tmp))
check("speak default = decide per line", seen["e"] is None and seen["k"] is None)

import prosody
check("one_question yes", prosody.one_question("Ты придёшь?"))
check("not a question", not prosody.one_question("Ты придёшь."))
check("two sentences are not one question", not prosody.one_question("Привет. Ты придёшь?"))
calls = []
prosody.question_shape = lambda cx, w, t, d="": calls.append(t) or w
check("finish: one question -> IK-3", delivery.finish(c, "w.wav", "Ты придёшь?") == "w.wav" and calls == ["Ты придёшь?"])
delivery.finish(c, "w.wav", "Ты придёшь.")
check("finish: statement untouched", calls == ["Ты придёшь?"])
c.delivery_ik3 = False
delivery.finish(c, "w.wav", "Ты придёшь?")
check("finish: ik3=False", len(calls) == 1)
sys.exit(1 if fails else 0)
