"""Score diarization pipelines on bench/diar_set (made by diar_make_set.py): current = Nemotron turns + GigaAM
per turn; moss = MOSS-Transcribe-Diarize one pass (bench/diar_set/moss.jsonl from scripts/moss_worker.py).

Metrics: DER-lite (20 ms frames, speakers mapped optimally: missed + false alarm + confusion over true speech)
and cpWER (words, best speaker permutation, the whole conversation as one concatenated stream per speaker).

    unset F5_TEST_RUN; venv/Scripts/python.exe -u bench/diar_eval.py [current] [moss]
"""
import itertools
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "bench", "voice", "agent", "core", "media", "bot"):
    sys.path.insert(0, os.path.join(ROOT, sub))
SET = os.environ.get("DIAR_SET") or os.path.join(ROOT, "bench", "diar_set")
FRAME = 0.02


def words(t):
    return re.findall(r"\w+", (t or "").lower().replace("ё", "е"))


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cpwer(ref: dict, hyp: dict) -> float:
    """ref/hyp: {speaker: text}. Best assignment of hyp speakers to ref speakers (missing = empty)."""
    rs, hs = list(ref), list(hyp)
    n = max(len(rs), len(hs))
    rs += [None] * (n - len(rs)); hs += [None] * (n - len(hs))
    total = sum(len(words(ref[r])) for r in ref if r) or 1
    best = min(sum(edit(words(ref.get(r, "")) if r else [], words(hyp.get(h, "")) if h else []) for r, h in zip(rs, perm))
               for perm in itertools.permutations(hs))
    return best / total


def der(truth: list, pred: list, dur: float) -> float:
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    n = int(dur / FRAME) + 1
    names = sorted({t["speaker"] for t in truth}); pn = sorted({p["speaker"] for p in pred})
    T = np.full(n, -1); P = np.full(n, -1)
    for t in truth:
        T[int(t["start"] / FRAME):int(t["end"] / FRAME)] = names.index(t["speaker"])
    for p in pred:
        P[int(p["start"] / FRAME):int(p["end"] / FRAME)] = pn.index(p["speaker"])
    M = np.zeros((len(names), max(1, len(pn))))
    for i in range(len(names)):
        for j in range(len(pn)):
            M[i, j] = np.sum((T == i) & (P == j))
    r, c = linear_sum_assignment(-M)
    mapping = {j: i for i, j in zip(r, c)}
    speech = T >= 0
    miss = np.sum(speech & (P < 0)); fa = np.sum(~speech & (P >= 0))
    conf = sum(1 for k in np.where(speech & (P >= 0))[0] if mapping.get(P[k], -2) != T[k])
    return float(miss + fa + conf) / max(1, speech.sum())


def per_speaker(segs: list) -> dict:
    out = {}
    for s in segs:
        out[s["speaker"]] = (out.get(s["speaker"], "") + " " + s.get("text", "")).strip()
    return out


def run_current(clips, ctx):
    import diarize
    from audio import transcribe_audio_file
    import soundfile as sf
    res = {}
    for c in clips:
        wav = os.path.join(SET, c + ".wav")
        ts = diarize.turns(diarize.segments(wav) or [])
        order = diarize.label_order(ts)
        sr_x, sr = sf.read(wav, dtype="float32")[0], sf.info(wav).samplerate
        segs = []
        for t in ts:
            a, b = int(max(0, t["start"] - 0.15) * sr), int((t["end"] + 0.15) * sr)
            tmp = os.path.join(ROOT, "bench", "diar_set", "_tmp", "turn.wav")
            sf.write(tmp, sr_x[a:b], sr)
            segs.append({"speaker": f"S{order[t['speaker']]}", "start": t["start"], "end": t["end"],
                         "text": transcribe_audio_file(ctx, tmp, lang_hint="ru") or ""})
        res[c] = segs
    return res


def run_moss(clips):
    out = {}
    p = os.path.join(SET, "moss.jsonl")
    for line in open(p, encoding="utf-8-sig", errors="replace"):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        out[os.path.splitext(os.path.basename(row["wav"]))[0]] = row["segments"]
    return {c: out[c] for c in clips if c in out}


def main(which):
    import soundfile as sf
    clips = sorted(f[:-4] for f in os.listdir(SET) if f.startswith("clip") and f.endswith(".wav"))
    ctx = None
    if "current" in which:
        import live_tg_drive as D
        _bot, ctx = D.build()
    results = {}
    if "current" in which:
        results["current"] = run_current(clips, ctx)
    if "moss" in which:
        results["moss"] = run_moss(clips)
    for name, res in results.items():
        ders, wers = [], []
        for c in clips:
            if c not in res:
                continue
            truth = json.load(open(os.path.join(SET, c + ".json"), encoding="utf-8"))["turns"]
            dur = sf.info(os.path.join(SET, c + ".wav")).duration
            d = der(truth, res[c], dur)
            w = cpwer(per_speaker(truth), per_speaker(res[c]))
            ders.append(d); wers.append(w)
            print(f"  {name:8s} {c}: speakers true {len({t['speaker'] for t in truth})} found {len({s['speaker'] for s in res[c]})}"
                  f"  DER {d:.3f}  cpWER {w:.3f}")
        if ders:
            print(f"== {name}: mean DER {sum(ders) / len(ders):.3f}  mean cpWER {sum(wers) / len(wers):.3f}  ({len(ders)} clips)")
    json.dump({k: v for k, v in results.items()}, open(os.path.join(SET, "results.json"), "w", encoding="utf-8"),
              ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1:] or ["current", "moss"])
