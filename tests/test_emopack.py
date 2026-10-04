"""emopack: reference lookup, idempotent build (Seed-VC stubbed), speak(emotion=) swaps the reference."""
import json, os, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "voice"):
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

# speak(): emotional reference reaches the synthesiser
import audio
seen = {}
audio.synth_single_segment = lambda c, i, a, t, out_stem=None: seen.update(ref=c.custom_ref_wav, txt=c.custom_ref_text) or "x.wav"
voice_clone.speak(object(), str(ref), "plain", "привет", str(tmp), emotion="joy")
check("speak joy uses pack ref", seen == {"ref": "a.wav", "txt": "ta"})
voice_clone.speak(object(), str(ref), "plain", "привет", str(tmp))
check("speak without emotion uses plain ref", seen == {"ref": str(ref), "txt": "plain"})
sys.exit(1 if fails else 0)
