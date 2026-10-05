"""Inputs for a LeVo run: a 10 s chorus clip of the reference (loudest vocal window, taken from the full mix) + the jsonl.
    venv/Scripts/python.exe bench/levo_prep.py ref.wav vocals.wav lyrics.txt out_dir"""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf
ref, vox, lyr, out = sys.argv[1:5]
TL = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя", ["a","b","v","g","d","ye","yo","zh","z","i","y","k","l","m","n","o","p","r","s","t","u","f","j","ts","ch","sh","sch","","i","","e","yu","ya"]))
def translit(t):
    out, prev = [], " "
    for ch in t:
        lo = ch.lower()
        if lo in TL:
            r = TL[lo]
            if lo in "еёюя" and prev not in " -aeiouаеёиоуыэюяъь":
                r = {"ye": "e", "yo": "o", "yu": "u", "ya": "a"}[r]       # after a consonant: soft vowel -> plain
            out.append(r.capitalize() if ch.isupper() and r else r)
        else:
            out.append(ch)
        prev = lo
    return "".join(out)
out = Path(out); out.mkdir(parents=True, exist_ok=True)
v, sr = sf.read(vox, dtype="float32"); v = v.mean(1) if v.ndim > 1 else v
hop = sr * 10
best = max(range(0, max(1, len(v) - hop), sr), key=lambda a: float(np.abs(v[a:a + hop]).mean()))
m, msr = sf.read(ref, dtype="float32")
a = int(best / sr * msr); sf.write(out / "prompt.wav", m[a:a + msr * 10], msr)
print("chorus window", best // sr, "s")
lines = [l.strip() for l in Path(lyr).read_text(encoding="utf-8").splitlines() if l.strip()]
if "--translit" in sys.argv:
    lines = [translit(l) for l in lines]
def sec(tag, ls): return f"[{tag}] " + ". ".join(x.rstrip(".,!?…") for x in ls) + "."
verse1, chorus, verse2 = lines[0:6], lines[15:19], lines[23:29]
text = " ; ".join(["[intro-short]", sec("verse", verse1), sec("chorus", chorus), "[inst-short]", sec("verse", verse2), sec("chorus", chorus), "[outro-short]"])
(out / "in.jsonl").write_text(json.dumps({"idx": "govnovoz_tl" if "--translit" in sys.argv else "govnovoz", "gt_lyric": text, "prompt_audio_path": str((out / "prompt.wav").resolve())}, ensure_ascii=False) + "\n", encoding="utf-8")
print(text[:600])
