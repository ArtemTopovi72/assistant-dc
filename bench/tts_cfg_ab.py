"""F5 delivery A/B at a FIXED 10 steps: classifier-free-guidance strength (lower = livelier, less stable) against
intelligibility (GigaAM WER) and liveliness (F0 spread, loudness swing). WAVs kept in outputs/cfg_ab/<cfg>/ for a listen.

    unset F5_TEST_RUN; venv/Scripts/python.exe -u bench/tts_cfg_ab.py [--cfg 2.0,1.6,1.3,1.0] [--voice DC_short_ref12.wav]
"""
import argparse
import os
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "bench", "voice", "agent", "core", "media", "bot"):
    sys.path.insert(0, str(ROOT / sub))

PHRASES = {
    "narration": "Старый дом стоял на краю деревни. Окна его давно потемнели, но по вечерам в них мелькал свет, и никто не знал, кто там живёт.",
    "dialogue": "— Ты опять опоздал! — крикнула она. — Я ждала тебя целый час, а ты даже не позвонил!",
    "verse": "Три девицы под окном, пряли поздно вечерком. Кабы я была царица, говорит одна девица, то на весь крещёный мир приготовила б я пир.",
    "question": "Неужели ты правда думаешь, что всё так просто? Скажи честно, ты хоть раз пытался это сделать?",
}


def words(s):
    return re.findall(r"[а-яёa-z0-9]+", s.lower().replace("ё", "е"))


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def liveliness(path):
    """(F0 standard deviation in semitones, loudness swing in dB): flat reading scores low on both."""
    import librosa
    import numpy as np
    y, sr = librosa.load(path, sr=16000)
    f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=400, sr=sr)
    f = f0[voiced & ~np.isnan(f0)]
    st = float(np.std(12 * np.log2(f / np.median(f)))) if len(f) > 20 else 0.0
    rms = librosa.feature.rms(y=y)[0]
    db = 20 * np.log10(rms[rms > 0.02 * rms.max()] + 1e-6)
    return st, float(np.percentile(db, 90) - np.percentile(db, 10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="2.0,1.6,1.3,1.0")
    ap.add_argument("--voice", default="DC_short_ref12.wav")
    a = ap.parse_args()
    import live_tg_drive as D
    import voice_clone
    from audio import transcribe_audio_file
    _bot, ctx = D.build()
    out = ROOT / "outputs" / "cfg_ab"
    ref, ref_text = voice_clone.prepare_reference(ctx, str(ROOT / a.voice), str(out / "_ref"), lang="ru")
    voice_clone.speak(ctx, ref, ref_text, "Прогрев.", str(out / "_tmp"))
    print(f"{'cfg':>5} {'kind':10} {'WER':>6} {'F0sd(st)':>9} {'dB swing':>9}")
    totals = {}
    for cfg in [float(x) for x in a.cfg.split(",")]:
        d = out / f"{cfg:g}"
        d.mkdir(parents=True, exist_ok=True)
        for kind, text in PHRASES.items():
            for rep in range(2):
                wav = voice_clone.speak(ctx, ref, ref_text, text, str(d), cfg=cfg)
                heard = transcribe_audio_file(ctx, wav, lang_hint="ru") or ""
                w = wer(text, heard)
                st, sw = liveliness(wav)
                os.replace(wav, d / f"{kind}_{rep}.wav")
                t = totals.setdefault(cfg, [0, 0, 0, 0])
                t[0] += w; t[1] += st; t[2] += sw; t[3] += 1
                print(f"{cfg:5.1f} {kind:10} {w:6.2f} {st:9.2f} {sw:9.1f}", flush=True)
    print("== mean per cfg")
    for cfg, t in totals.items():
        print(f"cfg {cfg:4.1f}: WER {t[0] / t[3]:.3f}  F0 sd {t[1] / t[3]:.2f} st  dB swing {t[2] / t[3]:.1f}")


if __name__ == "__main__":
    main()
