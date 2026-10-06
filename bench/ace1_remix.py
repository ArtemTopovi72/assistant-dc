"""Vocal of the ACE-edited song (Demucs) over the ORIGINAL backing (the ACE codec smears the backing).
    venv/Scripts/python.exe bench/ace1_remix.py ace1_full.wav out.mp3"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "media", "core"):
    sys.path.insert(0, str(ROOT / sub))
os.chdir(ROOT)
import numpy as np, soundfile as sf, cover, mashup_stems, remix
SR = remix.SR
back, _ = sf.read(ROOT / "outputs/svc_lyrics/backing.wav", dtype="float32")
vox0, _ = sf.read(ROOT / "outputs/svc_lyrics/vocals.wav", dtype="float32")
w = cover.to_wav(sys.argv[1], str(ROOT / "outputs/ace1_full_44.wav"))
v = mashup_stems.separate(w)["vocals"]
v = v.mean(1) if v.ndim > 1 else v
v = v * float(np.sqrt((vox0 ** 2).mean()) / max(1e-6, np.sqrt((v ** 2).mean())))   # same level as the original vocal
sf.write(ROOT / "outputs/ace1_vox.wav", v, SR)
print(remix._mix(str(ROOT / "outputs/ace1_vox.wav"), back, os.path.abspath(sys.argv[2])))
