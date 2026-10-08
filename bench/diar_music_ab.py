"""Does Demucs vocal separation help diarization when music plays under the talk?
Builds two copies of bench/diar_set: <set>_music (speech + an instrumental bed at 0 dB SNR) and <set>_music_sep
(the same after Demucs `vocals`); score both with DIAR_SET=<dir> bench/diar_eval.py current.

    venv/Scripts/python.exe bench/diar_music_ab.py
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SET = os.path.join(ROOT, "bench", "diar_set")
MIX = SET + "_music"
SEP = SET + "_music_sep"
PY = sys.executable


def demucs(src: str, out: str) -> str:
    subprocess.run([PY, "-m", "demucs", "--two-stems=vocals", "-d", "cuda", "-o", out, src], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.join(out, "htdemucs", os.path.splitext(os.path.basename(src))[0])


def mono16(path):
    x, sr = sf.read(path, dtype="float32")
    x = x.mean(axis=1) if x.ndim > 1 else x
    if sr != 24000:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=24000)
    return x


def main():
    for d in (MIX, SEP):
        os.makedirs(d, exist_ok=True)
    bed_dir = demucs(os.path.join(ROOT, "cover", os.environ.get("TG_CHAT_ID", "user"), "song.ogg"), os.path.join(MIX, "_song"))
    bed = mono16(os.path.join(bed_dir, "no_vocals.wav"))
    for f in sorted(os.listdir(SET)):
        if not (f.startswith("clip") and f.endswith(".wav")):
            continue
        c = f[:-4]
        x = mono16(os.path.join(SET, f))
        music = np.resize(bed[24000 * 5:], len(x))
        gain = np.sqrt(np.mean(x ** 2)) / max(1e-6, np.sqrt(np.mean(music ** 2)))      # 0 dB SNR
        mixed = np.clip(x + music * gain, -1, 1)
        sf.write(os.path.join(MIX, f), mixed, 24000)
        shutil.copy(os.path.join(SET, c + ".json"), os.path.join(MIX, c + ".json"))
        out = demucs(os.path.join(MIX, f), os.path.join(MIX, "_sep"))
        shutil.copy(os.path.join(out, "vocals.wav"), os.path.join(SEP, f))
        shutil.copy(os.path.join(SET, c + ".json"), os.path.join(SEP, c + ".json"))
        print("built", c)


if __name__ == "__main__":
    main()
