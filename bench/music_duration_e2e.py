"""Does a rendered song come back at the length that was asked for?

bench/music_duration.py measures the PLANNER (how many seconds it decides on).
This measures the DELIVERABLE: it renders a real song at a chosen duration and
reads the length of the audio file. That is the contract the user sees -- the
picker offers 30/60/120/180/240 and the song should last that long.

Measured so far, one render each:

     30s, budget's own lyric  (9 lines) ->  30.0s  (100%, 3.33 s/line)
     60s, budget's own lyric (18 lines) ->  60.0s  (100%, 3.33 s/line)
    120s, budget's own lyric (36 lines) -> 120.0s  (100%, 3.33 s/line)
    180s, budget's own lyric (55 lines) -> 179.9s  (100%, 3.27 s/line)
    240s, budget's own lyric (73 lines) -> 149.5s   (62%, 2.05 s/line)
    240s, 110 lines                     -> 175.6s   (73%, 1.60 s/line)

So the rate holds at ~3.3 s per sung line all the way to 180 and then collapses.
Half again as many words bought 17% more song. The planner COMPRESSES as the
lyric grows -- 2.05 s per sung line at 73 lines, 1.60 s at 110 -- so the slot
cannot be filled by writing more, and the linear rate model (lines = seconds /
3.3, calibrated at 30s and 180s) does not extend to the top of the range.

Acted on: 240 was removed from music.DURATIONS and MUSIC_MAX_SECONDS lowered to
180. Raise either only with a measurement from this bench.

Every duration the picker offers is now measured, not inferred.

Run: venv/Scripts/python.exe bench/music_duration_e2e.py --seconds 240 [--lines N]
"""
import argparse
import contextlib
import os
import sys
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import music  # noqa: E402

_VERSE = [
    "Город просыпается под серым небом",
    "Первый поезд тянет за собой рассвет",
    "Мы идём по улицам ещё пустым",
    "И считаем окна, где горит ответ",
    "Ветер обещает нам ещё один",
    "День, который мы не отдадим назад",
]


def audio_seconds(path: str) -> float:
    """Length of the rendered file, whatever container it came back in."""
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.frames / float(info.samplerate)
    except Exception:
        pass
    try:
        with contextlib.closing(wave.open(path)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return -1.0


def build_lyric(n_lines: int) -> str:
    lines = (_VERSE * (n_lines // len(_VERSE) + 1))[:n_lines]
    half = len(lines) // 2
    return ("[verse]\n" + "\n".join(lines[:half])
            + "\n\n[chorus]\n" + "\n".join(lines[half:]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=240)
    ap.add_argument("--lines", type=int, default=0,
                    help="override the budget's own line count")
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    target, floor, ceiling = music.line_budget(args.seconds)
    n = args.lines or target
    print("budget for %ds: target=%d floor=%d ceiling=%d  -> using %d lines"
          % (args.seconds, target, floor, ceiling, n))

    lyric = build_lyric(n)
    print("sung lines in the lyric:", music.count_sung_lines(lyric))

    t0 = time.perf_counter()
    try:
        path = music.generate_music(
            None, lyric, "мелодичный русский поп, женский вокал, среднее темпо",
            duration_s=args.seconds, seed=args.seed,
            on_progress=lambda *a, **k: None)
    except Exception as exc:
        print("render failed: %s: %s" % (type(exc).__name__, exc))
        return 1
    took = time.perf_counter() - t0

    got = audio_seconds(path)
    if got <= 0:
        print("rendered %s in %.0fs but its length could not be read" % (path, took))
        return 1
    pct = 100.0 * got / args.seconds
    print("rendered in %.0fs -> %s" % (took, path))
    print("asked %ds, got %.1fs  (%.0f%% of the ask, %.2f s per sung line)"
          % (args.seconds, got, pct, got / max(1, n)))
    # 95% is the bar the shorter durations already clear.
    print("VERDICT:", "ok" if pct >= 95 else "SHORT — the ask was not met")
    return 0 if pct >= 95 else 2


if __name__ == "__main__":
    sys.exit(main())
