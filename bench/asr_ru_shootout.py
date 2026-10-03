"""GigaAM against Whisper on REAL Russian speech, with real ground truth.

The published comparison that started this (habr.com/ru/articles/1002260) is
five files synthesised by a TTS. Synthetic speech has no room noise, no
disfluency and no accent, which is exactly where an ASR earns its WER, and the
author says so himself. So this measures on two real corpora instead:
read audiobook speech and crowd recordings made on real devices.

One caveat that cannot be measured away: GigaAM was trained on 700k hours of
Russian and both corpora are public, so some of this may be in its training
data. That biases the comparison IN GIGAAM'S FAVOUR, so a loss here is
conclusive and a win should be read as an upper bound.

    venv/Scripts/python.exe bench/asr_ru_shootout.py [n] [rudevices|rulibrispeech]
"""
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

OUT = ROOT / "runtime" / "asr_shootout"
N = 100


def normalise(text: str) -> str:
    """Fair comparison needs one spelling of the same words.

    Whisper writes "1500" where the reference says "тысяча пятьсот", and both
    engines differ on ё and on punctuation. Scoring those as errors would
    measure formatting, not recognition.
    """
    t = (text or "").lower().replace("ё", "е")
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def wer(ref: str, hyp: str) -> float:
    r, h = normalise(ref).split(), normalise(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    # Levenshtein over words, iterative so a long line cannot blow the stack.
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


# Two sets, deliberately different in kind. rulibrispeech is read audiobook
# speech: clean, fluent, the easy case. sova_rudevices is crowd-recorded on
# real devices: room noise, clipping, half-swallowed words -- much closer to a
# Telegram voice note or a headset in Skyrim, which is what this decides.
# Common Voice was the first choice and is not usable: it ships as a loading
# script, which current `datasets` refuses to execute.
DATASETS = {
    "rulibrispeech": ("bond005/rulibrispeech", "transcription"),
    "rudevices": ("bond005/sova_rudevices", "transcription"),
}


def corpus_wer(rows, key: str) -> float:
    """Total edits over total reference words, not the mean of per-clip WER."""
    edits = words = 0
    for r in rows:
        ref = normalise(r["ref"]).split()
        edits += wer(r["ref"], r.get(key, "")) * len(ref)
        words += len(ref)
    return edits / words if words else 0.0


def load_samples(n: int, which: str = "rudevices"):
    """Streamed, so nothing is downloaded in full.

    Decoding is done here rather than by `datasets`: its Audio feature routes
    through torchaudio -> torchcodec, whose DLLs are version-locked against
    torch and FFmpeg and do not load in this venv. soundfile reads the same
    bytes with no such coupling -- the same reason the TTS path stopped asking
    f5_tts to transcribe its own reference.
    """
    import io as _io
    import soundfile as sf
    from datasets import load_dataset, Audio

    repo, field = DATASETS[which]
    ds = load_dataset(repo, split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    out = []
    for row in ds:
        a = row["audio"]
        text = row.get(field) or row.get("sentence") or row.get("text") or ""
        raw = a.get("bytes")
        try:
            if raw:
                x, sr = sf.read(_io.BytesIO(raw), dtype="float32", always_2d=False)
            else:
                x, sr = sf.read(a["path"], dtype="float32", always_2d=False)
        except Exception as exc:
            print("  skipped a clip (%s)" % exc)
            continue
        if getattr(x, "ndim", 1) > 1:
            x = x.mean(axis=1)
        x = np.asarray(x, dtype=np.float32)
        if sr != 16000:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=16000).astype(np.float32)
        if not len(x):
            continue
        out.append({"audio": x, "text": text})
        if len(out) >= n:
            break
    return out


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N
    which = sys.argv[2] if len(sys.argv) > 2 else "rudevices"
    OUT.mkdir(parents=True, exist_ok=True)

    print("fetching %d %s test clips..." % (n, which), flush=True)
    samples = load_samples(n, which)
    total_audio = sum(len(s["audio"]) / 16000 for s in samples)
    print("%d clips, %.1f s of audio\n" % (len(samples), total_audio), flush=True)

    rows = []

    # ---- GigaAM, CPU. Running it on the CPU is the entire point: it is what
    # would free the card for the LLM and the image model.
    import onnx_asr
    t0 = time.time()
    giga = onnx_asr.load_model("gigaam-v3-e2e-rnnt")
    print("GigaAM loaded in %.1fs (CPU)" % (time.time() - t0), flush=True)
    g_time = 0.0
    for i, s in enumerate(samples):
        t = time.time()
        try:
            hyp = giga.recognize(s["audio"], sample_rate=16000)
        except Exception as exc:
            hyp = ""
            print("  giga failed on %d: %s" % (i, exc))
        g_time += time.time() - t
        rows.append({"ref": s["text"], "giga": hyp})
    print("GigaAM done: %.1fs total\n" % g_time, flush=True)

    # ---- Whisper large-v3-turbo, GPU, exactly as the app loads it.
    from faster_whisper import WhisperModel
    import config as C
    t0 = time.time()
    wm = WhisperModel("large-v3-turbo", device=C.WHISPER_DEVICE,
                      compute_type=C.WHISPER_COMPUTE_TYPE)
    print("Whisper loaded in %.1fs (%s)" % (time.time() - t0, C.WHISPER_DEVICE),
          flush=True)
    w_time = 0.0
    for i, s in enumerate(samples):
        t = time.time()
        try:
            segs, _ = wm.transcribe(s["audio"], language="ru", beam_size=5,
                                    vad_filter=True)
            hyp = " ".join(x.text for x in segs).strip()
        except Exception as exc:
            hyp = ""
            print("  whisper failed on %d: %s" % (i, exc))
        w_time += time.time() - t
        rows[i]["whisper"] = hyp
    print("Whisper done: %.1fs total\n" % w_time, flush=True)

    # CORPUS WER: total edits over total reference words. Averaging per-clip
    # WER instead lets one bad label decide the result -- this corpus contains
    # a two-letter reference ("аа") against a whole sung line, which scores
    # 2100% on its own and moved the mean by three points. Corpus WER is also
    # the number everyone else reports.
    g_wer = corpus_wer(rows, "giga")
    w_wer = corpus_wer(rows, "whisper")
    g_win = sum(1 for r in rows
                if wer(r["ref"], r["giga"]) < wer(r["ref"], r["whisper"]))
    long_rows = [r for r in rows if len(normalise(r["ref"]).split()) >= 3]
    (OUT / ("rows_%s.json" % which)).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                   encoding="utf-8")

    print("=== %d clips, %.1f s of speech ===" % (len(rows), total_audio))
    print("WER          GigaAM %.1f%%   Whisper %.1f%%" % (g_wer * 100, w_wer * 100))
    print("total time   GigaAM %.1fs (CPU)   Whisper %.1fs (%s)"
          % (g_time, w_time, C.WHISPER_DEVICE))
    print("RTF          GigaAM %.2f          Whisper %.2f"
          % (g_time / total_audio, w_time / total_audio))
    if long_rows:
        print("WER, refs of 3+ words   GigaAM %.1f%%   Whisper %.1f%%   (n=%d)"
              % (corpus_wer(long_rows, "giga") * 100,
                 corpus_wer(long_rows, "whisper") * 100, len(long_rows)))
    print("GigaAM closer on %d of %d clips" % (g_win, len(rows)))
    print("exact matches           GigaAM %d   Whisper %d"
          % (sum(1 for r in rows if wer(r["ref"], r["giga"]) == 0),
             sum(1 for r in rows if wer(r["ref"], r["whisper"]) == 0)))
    print("rows:", OUT / ("rows_%s.json" % which))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
