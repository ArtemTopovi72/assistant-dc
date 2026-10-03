"""F5 against XTTS on EVERY voice in the Mantella speaker library.

The single-phrase comparison answered "which engine sounds better on Stepan",
which is not the question that decides a migration. Mantella needs 90-odd
different NPCs to stay recognisably themselves, and that is the one thing XTTS
was chosen for. So this measures, per voice:

  * cloning fidelity — cosine similarity between a speaker embedding of the
    REFERENCE and of the synthesised line (resemblyzer). This is the number
    that decides it.
  * intelligibility — Whisper transcribes the result and it is compared with
    the sentence that was requested. A voice can be a perfect clone and still
    mumble.
  * wall time per line.

Both engines are given the SAME reference and the SAME sentence, so what
varies is the engine.

    venv/Scripts/python.exe bench/tts_engine_voice_sweep.py [limit]
"""
import json
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C

SPEAKERS = Path(r"C:\llamacpp\xtts_speakers")
XTTS_URL = "http://127.0.0.1:8020/tts_to_audio/"
OUT = ROOT / "runtime" / "tts_sweep"
PHRASE = "Слушай, я вчера полночи чинил этот проектор, а он всё равно жуёт плёнку."


def voices():
    """One canonical reference per voice.

    A top-level <name>.wav wins over <name>/clip0.wav: the loose file is the
    curated sample, the folder holds raw clips of varying quality, and mixing
    the two would compare engines on different audio.
    """
    found = {}
    for d in sorted(SPEAKERS.iterdir()):
        if d.is_dir():
            clips = sorted(d.glob("*.wav"))
            if clips:
                found.setdefault(d.name, clips[0])
    for f in sorted(SPEAKERS.glob("*.wav")):
        found[f.stem] = f
    return found


def _ctx():
    import models as M
    return types.SimpleNamespace(
        models=M.Models.load(), model_name=C.MODEL_NAME, no_think=True,
        reasoning_effort="high", asr_lock=threading.Lock(),
        tts_lock=threading.Lock(), api_lock=threading.Lock(),
        last_api_call_time=0.0, api_min_interval=1.0, session_memory=[],
        pinned_facts=[], cancel_event=threading.Event(),
        transcription_cache={}, save_cache=lambda: None, custom_ref_wav=None)


def xtts(voice_id: str, dest: Path):
    body = json.dumps({"text": PHRASE, "speaker_wav": voice_id,
                       "language": "ru"}).encode()
    req = urllib.request.Request(XTTS_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    data = urllib.request.urlopen(req, timeout=300).read()
    dest.write_bytes(data)
    return time.time() - t0


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 6
    OUT.mkdir(parents=True, exist_ok=True)
    import audio as A
    import difflib
    from resemblyzer import VoiceEncoder, preprocess_wav

    enc = VoiceEncoder()
    ctx = _ctx()

    def emb(path):
        try:
            return enc.embed_utterance(preprocess_wav(Path(path)))
        except Exception:
            return None

    def sim(a, b):
        import numpy as np
        ea, eb = emb(a), emb(b)
        if ea is None or eb is None:
            return None
        return float(np.dot(ea, eb))

    def heard(path):
        return A.transcribe_audio_file(ctx, str(path)) or ""

    def acc(text):
        return difflib.SequenceMatcher(None, PHRASE.lower(), text.lower()).ratio()

    rows = []
    vs = list(voices().items())[:limit]
    print("voices: %d\n" % len(vs), flush=True)
    for i, (name, ref) in enumerate(vs):
        # F5 — its own reference handling (trim + transcript) is the point,
        # so it is driven through the production entry point, not around it.
        ctx.custom_ref_wav = str(ref)
        t0 = time.time()
        try:
            w5 = A.synth_single_segment(ctx, i, "DC", PHRASE, apply_stress=True,
                                        out_stem=str(OUT / ("f5_" + name)))
        except Exception as exc:
            print("[%s] F5 raised: %s" % (name, exc), flush=True)
            w5 = None
        t_f5 = time.time() - t0

        wx = OUT / ("xtts_%s.wav" % name)
        try:
            t_x = xtts(name, wx)
        except Exception as exc:
            print("[%s] XTTS failed: %s" % (name, exc), flush=True)
            wx, t_x = None, None

        s5 = sim(ref, w5) if w5 else None
        sx = sim(ref, wx) if wx else None
        a5 = acc(heard(w5)) if w5 else None
        ax = acc(heard(wx)) if wx else None
        rows.append({"voice": name, "f5_sim": s5, "xtts_sim": sx,
                     "f5_acc": a5, "xtts_acc": ax,
                     "f5_s": t_f5, "xtts_s": t_x})
        print("[%3d/%d] %-22s F5 sim %s acc %s %.1fs | XTTS sim %s acc %s %s"
              % (i + 1, len(vs), name[:22],
                 "%.3f" % s5 if s5 is not None else "  -  ",
                 "%.2f" % a5 if a5 is not None else " - ", t_f5,
                 "%.3f" % sx if sx is not None else "  -  ",
                 "%.2f" % ax if ax is not None else " - ",
                 "%.1fs" % t_x if t_x else "-"), flush=True)

    (OUT / "sweep.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                    encoding="utf-8")

    def avg(key):
        got = [r[key] for r in rows if r[key] is not None]
        return sum(got) / len(got) if got else 0.0

    wins = sum(1 for r in rows if r["f5_sim"] and r["xtts_sim"]
               and r["f5_sim"] > r["xtts_sim"])
    both = sum(1 for r in rows if r["f5_sim"] and r["xtts_sim"])
    print("\n=== %d voices ===" % len(rows))
    print("cloning similarity   F5 %.3f   XTTS %.3f" % (avg("f5_sim"), avg("xtts_sim")))
    print("intelligibility      F5 %.2f    XTTS %.2f" % (avg("f5_acc"), avg("xtts_acc")))
    print("seconds per line     F5 %.2f    XTTS %.2f" % (avg("f5_s"), avg("xtts_s")))
    print("F5 closer to the reference on %d of %d voices" % (wins, both))
    print("rows:", OUT / "sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
