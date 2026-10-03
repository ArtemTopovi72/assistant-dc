"""Speed, measured the way the shim will actually run.

The voice sweep timed F5 at 2.62 s against XTTS's 1.48 s, but that included
preparing a reference F5 had never seen — trimming it and transcribing it —
which a server does ONCE per voice and then caches. Timing that on every line
answers a question nobody is asking.

Here each voice is warmed first, then timed over several lines of differing
length, so what is compared is steady-state synthesis.

    venv/Scripts/python.exe bench/tts_speed_fair.py
"""
import json
import statistics
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
OUT = ROOT / "runtime" / "tts_speed"
VOICES = ["adril", "mercerfrey", "femaledarkseducer"]
LINES = [
    "Да.",
    "Слушай, я вчера полночи чинил этот проектор.",
    "Слушай, я вчера полночи чинил этот проектор, а он всё равно жуёт плёнку, "
    "и теперь придётся идти к кузнецу за новой шестерёнкой.",
]
REPEATS = 3


def _ctx():
    import models as M
    return types.SimpleNamespace(
        models=M.Models.load(), model_name=C.MODEL_NAME, no_think=True,
        reasoning_effort="high", asr_lock=threading.Lock(),
        tts_lock=threading.Lock(), api_lock=threading.Lock(),
        last_api_call_time=0.0, api_min_interval=1.0, session_memory=[],
        pinned_facts=[], cancel_event=threading.Event(),
        transcription_cache={}, save_cache=lambda: None, custom_ref_wav=None)


def xtts(voice, text, dest):
    body = json.dumps({"text": text, "speaker_wav": voice,
                       "language": "ru"}).encode()
    req = urllib.request.Request(XTTS_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    dest.write_bytes(urllib.request.urlopen(req, timeout=300).read())
    return time.time() - t0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    import audio as A
    import soundfile as sf

    ctx = _ctx()
    rows = []
    for voice in VOICES:
        ref = SPEAKERS / (voice + ".wav")
        if not ref.is_file():
            cand = sorted((SPEAKERS / voice).glob("*.wav"))
            if not cand:
                print("no reference for", voice); continue
            ref = cand[0]
        ctx.custom_ref_wav = str(ref)

        # Warm BOTH: F5 caches the trimmed reference and its transcript, XTTS
        # caches its speaker latents. Neither should pay that on the clock.
        A.synth_single_segment(ctx, 0, "DC", "Раз.", apply_stress=True,
                               out_stem=str(OUT / ("warm_" + voice)))
        xtts(voice, "Раз.", OUT / ("warm_x_%s.wav" % voice))

        for li, line in enumerate(LINES):
            f5, xt, dur_f5, dur_x = [], [], 0.0, 0.0
            for r in range(REPEATS):
                t0 = time.time()
                w = A.synth_single_segment(
                    ctx, li * 10 + r, "DC", line, apply_stress=True,
                    out_stem=str(OUT / ("f5_%s_%d" % (voice, li))))
                f5.append(time.time() - t0)
                if w:
                    dur_f5 = sf.info(w).duration
                p = OUT / ("xtts_%s_%d.wav" % (voice, li))
                xt.append(xtts(voice, line, p))
                dur_x = sf.info(str(p)).duration
            rows.append({"voice": voice, "line": li, "chars": len(line),
                         "f5": statistics.median(f5), "xtts": statistics.median(xt),
                         "f5_audio": dur_f5, "xtts_audio": dur_x})
            print("%-18s line %d (%3d chars)  F5 %.2fs (audio %.2fs, RTF %.2f) | "
                  "XTTS %.2fs (audio %.2fs, RTF %.2f)"
                  % (voice, li, len(line), rows[-1]["f5"], dur_f5,
                     rows[-1]["f5"] / max(dur_f5, .01), rows[-1]["xtts"], dur_x,
                     rows[-1]["xtts"] / max(dur_x, .01)), flush=True)

    (OUT / "speed.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    f5 = statistics.median([r["f5"] for r in rows])
    xt = statistics.median([r["xtts"] for r in rows])
    rf5 = statistics.median([r["f5"] / max(r["f5_audio"], .01) for r in rows])
    rxt = statistics.median([r["xtts"] / max(r["xtts_audio"], .01) for r in rows])
    print("\n=== warm, %d voices x %d lines x %d runs ==="
          % (len(VOICES), len(LINES), REPEATS))
    print("median seconds per line   F5 %.2f   XTTS %.2f" % (f5, xt))
    print("median RTF                F5 %.2f   XTTS %.2f" % (rf5, rxt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
