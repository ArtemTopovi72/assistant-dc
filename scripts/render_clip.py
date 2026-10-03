"""Render one MiniMax H3 clip to a chosen folder, with per-phase timing.

Separate from smoke_video.py: that one answers "is the pipeline wired correctly",
this one answers "how long does a clip actually take on this machine, and where
does the time go". The phase split matters because the two halves behave very
differently on a 24GB card — sampling streams the DiT, and the VAE decode is a
second big model that has to fit alongside whatever is already resident.

Run:
  venv/Scripts/python.exe scripts/render_clip.py --seconds 5 --out runtime/video
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import config as C
import video as V
import models

DEFAULT_PROMPT = ("a red fox steps out of tall grass and turns toward the camera, "
                  "slow push-in, late afternoon light; wind in the grass and one "
                  "distant bird, no music")


def human(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0,
                    help="clip length; 0 -> this many seconds (default 5)")
    ap.add_argument("--aspect", default="16:9")
    ap.add_argument("--out", default=os.path.join("runtime", "video"),
                    help="target folder for the finished clip")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    # video.generate_video copies its result out of ComfyUI's tree into
    # video.OUTPUT_DIR; point that at the folder the caller asked for.
    V.OUTPUT_DIR = out_dir

    ok, why = V.engine_available(None)
    if not ok:
        print(f"Cannot render: {why}")
        return 2

    frames = V.seconds_to_frames(args.seconds)
    w, h = V.resolve_size(args.aspect)
    print("=" * 68)
    print(f"target folder : {out_dir}")
    print(f"clip          : 0.00s -> {V.frames_to_seconds(frames):.2f}s "
          f"({frames} frames @ {C.VIDEO_FPS}fps)")
    print(f"canvas        : {w}x{h}   steps: {C.VIDEO_STEPS}   cfg: {C.VIDEO_CFG}")
    print(f"prompt        : {args.prompt[:90]}")
    print("=" * 68, flush=True)

    ctx = models.Context(models=None, transcription_cache={}, cache_file=None,
                         asr_lock=threading.Lock(), tts_lock=threading.Lock())

    marks: list[tuple[str, float]] = []
    t0 = time.time()
    state = {"last": 0, "sampling_done": None, "first_step": None}

    def on_stage(s):
        marks.append((s, time.time() - t0))
        print(f"  [{human(time.time() - t0)}] {s}", flush=True)

    def on_progress(i, total):
        now = time.time() - t0
        if state["first_step"] is None:
            state["first_step"] = now
            print(f"  [{human(now)}] sampling started ({total} steps)", flush=True)
        if i != state["last"]:
            state["last"] = i
            print(f"\r  [{human(now)}] step {i}/{total}", end="", flush=True)
            if i >= total:
                state["sampling_done"] = now
                print()

    ctx.set_stage = on_stage
    res = V.generate_video(ctx, args.prompt, seconds=args.seconds,
                           aspect=args.aspect,
                           seed=args.seed or None, on_progress=on_progress)
    total = time.time() - t0

    print("=" * 68)
    if res.get("status") != "success" or not res.get("path"):
        print(f"FAILED after {human(total)}: {res.get('reason')}")
        print(f"failure detail: {V._VIDEO_FAILURE}")
        return 1

    path = res["path"]
    info = V.probe(path)
    load_s = state["first_step"] or 0.0
    sample_s = (state["sampling_done"] or total) - load_s
    decode_s = total - (state["sampling_done"] or total)

    print(f"SUCCESS in {human(total)}")
    print()
    print(f"  file        : {path}")
    print(f"  folder      : {os.path.dirname(path)}")
    print(f"  size        : {info.get('bytes', 0) / 1e6:.1f} MB")
    print(f"  duration    : {info.get('seconds')}s   ({frames} frames @ {C.VIDEO_FPS}fps)")
    print(f"  resolution  : {info.get('width')}x{info.get('height')}")
    print(f"  audio track : {'yes' if info.get('has_audio') else 'NO — expected one'}")
    print(f"  seed        : {res.get('seed')}")
    print()
    print("  time breakdown")
    print(f"    model load / init : {human(load_s)}")
    print(f"    sampling          : {human(sample_s)}")
    print(f"    VAE decode + mux  : {human(decode_s)}")
    print(f"    TOTAL             : {human(total)}")
    if not info.get("has_audio"):
        print("\n  WARNING: no audio stream — H3 should always emit one.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
