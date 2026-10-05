"""Which syllable of a sung word got the stress? (2026-09-28: YuE2 sings гнИлое.)

Vocal stem (demucs) -> Whisper word timestamps -> inside the target word, the
voiced runs (pyin) = syllable nuclei -> the stressed one is the longest (the
note the melody lands on), louder breaking ties. A heuristic: calibrate it
against the user's ears before trusting a score.
usage: stress_judge.py WORD_PREFIX file.mp3 ...   (e.g. гнил)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def vocals(path):
    import mashup_stems as S
    from mashup_stems import SAMPLE_RATE
    return S.separate(path)["vocals"], SAMPLE_RATE


_W = []
PROMPT = "Никто ни хера не делает, лишь хамят в лицо, собрали в поликлинике сплошное гнилое яйцо!"


def words(vox, sr):
    import librosa
    from faster_whisper import WhisperModel
    if not _W:
        _W.append(WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16"))
    y16 = librosa.resample(vox, orig_sr=sr, target_sr=16000)
    segs, _ = _W[0].transcribe(y16, language="ru", word_timestamps=True, vad_filter=True,
                               condition_on_previous_text=False, initial_prompt=PROMPT)
    return [(w.word.strip().lower(), w.start, w.end) for s in segs for w in s.words]


def nuclei(vox, sr, t0, t1):
    """Voiced runs inside [t0, t1] as (start, dur, rms)."""
    import librosa
    seg = vox[int(t0 * sr):int(t1 * sr)]
    if len(seg) < sr // 10:
        return []
    hop = 256
    f0, voiced, _ = librosa.pyin(seg, fmin=80, fmax=1000, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=seg, hop_length=hop)[0][:len(voiced)]
    runs, cur = [], None
    for i, v in enumerate(voiced):
        if v and cur is None:
            cur = i
        if (not v or i == len(voiced) - 1) and cur is not None:
            end = i if not v else i + 1
            if end - cur >= 2:
                runs.append((t0 + cur * hop / sr, (end - cur) * hop / sr, float(rms[cur:end].mean())))
            cur = None
    return runs


def judge(path, prefix, n_syll=3):
    vox, sr = vocals(path)
    hits = [(w, a, b) for w, a, b in words(vox, sr) if w.strip(".,!?«»").startswith(prefix)]
    out = []
    for w, a, b in hits:
        runs = nuclei(vox, sr, max(0, a - 0.05), b + 0.15)
        if len(runs) < 2:
            out.append((w, None, runs))
            continue
        runs = sorted(sorted(runs, key=lambda r: -r[1])[:n_syll])   # keep the n longest, in time order
        best = max(range(len(runs)), key=lambda i: (round(runs[i][1], 2), runs[i][2]))
        out.append((w, best + 1, runs))
    return out


if __name__ == "__main__":
    prefix, files = sys.argv[1], sys.argv[2:]
    for f in files:
        for w, syl, runs in judge(f, prefix):
            d = " ".join(f"{r[1]:.2f}s" for r in runs)
            print(f"{os.path.basename(f):22} {w:12} stressed syllable: {syl}   durations: {d}")
        else:
            pass
