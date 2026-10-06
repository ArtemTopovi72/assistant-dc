"""Demucs vocal + backing of a rendered song.  venv/Scripts/python.exe bench/yue_vox.py song.wav out_dir"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "media", "core"):
    sys.path.insert(0, str(ROOT / sub))
import soundfile as sf, cover, mashup_stems, remix
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
st = mashup_stems.separate(cover.to_wav(sys.argv[1], str(out / "in44.wav")))
for k, v in st.items():
    sf.write(out / f"{k}.wav", v, remix.SR)
print(list(st))
