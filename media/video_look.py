"""Look at a video, not only listen to it.

Live 2026-09-14 21:23: a forwarded round video (33 s, a grow lamp on a stand
by a window) was handled as «Пересланное голосовое» -- the audio track was
transcribed and the picture never seen. The user's ask: frames every 5 s or
some other way, but the bot must SEE what is in a video.

One vision call per video: frames are sampled every `every` seconds (the
interval grows so a long clip still fits `max_frames`), tiled into a numbered
contact sheet with the timestamp drawn on each cell, and the house vision
model describes what happens across the sheet in order. The sheet stays on
disk so a follow-up («что там на 20-й секунде?») has a picture to point at.
"""
from __future__ import annotations

import logging
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("assistant.video_look")

EVERY_SECONDS = 5
MAX_FRAMES = 12
GRID_COLS, CELL = 4, 320

_SHEET_SYSTEM = (
    "You are looking at frames sampled from ONE video, in time order, each "
    "labelled with its timestamp. Write a STORYBOARD: "
    "one line per frame, strictly in this form and nothing else on the line:\n"
    "  0:05 — what this frame shows (setting, people/objects, action)\n"
    "Text written on a frame (captions, signs, meme text) is often the whole "
    "point: QUOTE it word for word in «», in its own language, the first time "
    "it appears -- never just say that there is text (live 2026-10-02: a meme "
    "came back as «текст на русском языке» twice and the joke was lost). "
    "The overall sentence says what the text and pictures mean together.\n"
    "Use every timestamp given, in order, and no others. Be "
    "concrete: name things, not impressions. When a frame repeats the previous "
    "one, say so in a few words. After the lines add ONE sentence starting with "
    "the overall label given below that says what the whole video is about. Name "
    "things for what they are, not by shape or colour. Do not invent sound or speech -- you only see the picture.")

def duration_seconds(path: str) -> float:
    """Length of the clip via ffprobe; 0.0 when unknown."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=20)
        return float((out.stdout or "0").strip() or 0)
    except Exception:
        return 0.0


def sample_times(duration: float, every: float = EVERY_SECONDS,
                 max_frames: int = MAX_FRAMES) -> list[float]:
    """Timestamps to grab: every `every` s from 0, thinned so at most
    `max_frames` remain; a clip shorter than `every` still yields one frame."""
    if duration <= 0:
        return [0.0]
    n = int(duration // every) + 1
    step = every
    if n > max_frames:
        step = duration / max_frames
        n = max_frames
    times = [round(i * step, 2) for i in range(n)]
    # never ask for a frame past the end (ffmpeg returns nothing there)
    return [t for t in times if t < max(duration - 0.05, 0.01)] or [0.0]


def extract_frames(path: str, times: list[float], out_dir: Path) -> list[tuple[float, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in times:
        dst = out_dir / f"frame_{int(t * 100):07d}.jpg"
        try:
            subprocess.run(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", path,
                            "-frames:v", "1", "-q:v", "3", str(dst)],
                           capture_output=True, check=True, timeout=30)
            if dst.exists() and dst.stat().st_size > 0:
                frames.append((t, dst))
        except Exception:
            logger.debug("frame at %.2fs failed", t, exc_info=True)
    return frames


# ── key moments by frame difference ─────────────────────────────────────────
# «Every 5 s» misses what happens between the ticks and lands on the blur of
# a camera swing. Instead: a dense low-res pass (DENSE_FPS frames per second),
# a change score between neighbours (mean absolute difference of small
# grayscale frames) and a sharpness score (variance of the Laplacian). Where
# the change score spikes a new SHOT begins; inside each shot the sharpest
# frame is the one worth looking at -- the face, then the step with «67» on
# it, never the smear in between.
DENSE_FPS = 2
SHOT_CHANGE = 18.0      # mean |Δ| on 0..255 gray at 96 px wide; a pan/cut is far above
MIN_SHOT_SECONDS = 1.0  # shorter spikes are shake, not a new shot


def dense_pass(path: str, out_dir: Path, fps: float = DENSE_FPS, width: int = 96) -> list[tuple[float, "object"]]:
    """Tiny gray frames of the whole clip in one ffmpeg run: [(t, array)]."""
    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("d_*.png"):
        f.unlink()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                    "-vf", f"fps={fps},scale={width}:-2,format=gray",
                    str(out_dir / "d_%05d.png")],
                   capture_output=True, check=True, timeout=300)
    from PIL import Image
    frames = []
    for i, f in enumerate(sorted(out_dir.glob("d_*.png"))):
        frames.append((i / fps, np.asarray(Image.open(f), dtype=np.float32)))
    return frames


def _sharpness(a) -> float:
    import numpy as np
    # variance of a 4-neighbour Laplacian; blur/motion smear collapses it
    lap = (-4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:])
    return float(np.var(lap))


def key_times(frames: list, max_frames: int = MAX_FRAMES, change: float = SHOT_CHANGE,
              min_shot: float = MIN_SHOT_SECONDS) -> list[float]:
    """Timestamps of the sharpest frame of every shot, at most `max_frames`.

    Shot boundaries are the strongest jumps of the change score, taken
    greedily with a minimum spacing between them (`min_shot`, or the clip
    length over `max_frames` when that is longer) and only where the jump
    stands out against the clip's own motion level (hand-held footage keeps
    a floor of noise everywhere; a fixed threshold cut it every frame). In
    each shot the sharpest frame is picked, never the boundary frame itself
    (that is the swing). A clip that never changes yields its one sharpest
    frame plus evenly spaced anchors when it is long."""
    import numpy as np
    if not frames:
        return [0.0]
    n = len(frames)
    if n == 1:
        return [frames[0][0]]
    diffs = [0.0] + [float(np.mean(np.abs(frames[i][1] - frames[i - 1][1]))) for i in range(1, n)]
    dt = frames[1][0] - frames[0][0]
    total = frames[-1][0]
    gap = max(min_shot, total / max_frames)
    gap_i = max(1, int(round(gap / dt)))
    floor = max(change, 1.5 * float(np.median(diffs[1:])))
    cuts = [0]
    for i in sorted(range(1, n), key=lambda k: -diffs[k]):
        if diffs[i] < floor or len(cuts) >= max_frames:
            break
        if all(abs(i - c) >= gap_i for c in cuts):
            cuts.append(i)
    cuts.sort()
    bounds = list(zip(cuts, cuts[1:] + [n]))          # [start, end)
    # a long quiet stretch still moves (a slow pan, a hand entering): split
    # it into `gap`-sized windows so every window shows its sharpest frame
    split = []
    for a, b in bounds:
        k = max(1, (b - a) // gap_i)
        edges = [a + (b - a) * m // k for m in range(k)] + [b]
        split += list(zip(edges, edges[1:]))
    bounds = split[:max_frames]
    picks, sharp = [], []
    for a, b in bounds:
        lo = a + 1 if b - a > 2 else a
        best = max(range(lo, b), key=lambda i: _sharpness(frames[i][1]))
        picks.append(frames[best][0]); sharp.append(_sharpness(frames[best][1]))
    # a window that is blur through and through is the swing itself: drop
    # it when it is far softer than the rest (never below two frames)
    ref = float(np.median(sharp)) if sharp else 0.0
    if len(picks) > 2 and ref >= 20:      # a flat clip has no blur to drop
        keep = [(t, sh) for t, sh in zip(picks, sharp) if sh >= 0.35 * ref]
        if len(keep) >= 2:
            picks = [t for t, _ in keep]
    if len(picks) == 1 and total >= 3 * EVERY_SECONDS:
        anchors = sample_times(total, max(EVERY_SECONDS, total / max(2, max_frames - 1)), max_frames)
        picks = sorted(set(picks) | set(anchors))[:max_frames]
    return sorted(round(t, 2) for t in picks)


def _stamp(t: float) -> str:
    t = int(t); h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def make_sheet(frames: list[tuple[float, Path]], out_path: Path, offset: float = 0.0) -> Path:
    """`offset`: seconds to add to every label -- a portion of a long video
    is labelled with its ABSOLUTE position in the whole video."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    n = max(1, len(frames))
    cols = min(GRID_COLS, n)
    rows = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * CELL, rows * CELL), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except Exception:
        font = ImageFont.load_default()
    for i, (t, p) in enumerate(frames):
        try:
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            im.thumbnail((CELL - 8, CELL - 8))
        except Exception:
            continue
        x = (i % cols) * CELL + 4
        y = (i // cols) * CELL + 4
        sheet.paste(im, (x, y))
        label = _stamp(t + offset)
        draw.rectangle((x, y, x + 24 + 20 * len(label), y + 44), fill=(255, 210, 0))
        draw.text((x + 6, y + 2), label, fill=(0, 0, 0), font=font)
    sheet.save(out_path, "JPEG", quality=88)
    return out_path


_LANG_LINE = {"ru": " Answer in Russian; the overall line starts with 'Итог:'.",
              "en": " Answer in English; the overall line starts with 'Overall:'."}


def combine_sheets(parts: list, out_path: Path, width: int = 960) -> Path:
    """Several clips' sheets as ONE picture, each under a bar naming its clip.

    A forwarded batch of video notes used to leave only the LAST clip's sheet
    as the chat's picture, so a follow-up question "looked again" at one clip
    of nine. `parts` is [(label, sheet_path)] in order.
    """
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 34)
    except Exception:
        font = ImageFont.load_default()
    bar = 52
    tiles = []
    for label, p in parts:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        im = im.resize((width, max(1, round(im.height * width / im.width))))
        tiles.append((label, im))
    out = Image.new("RGB", (width, sum(bar + im.height for _, im in tiles) or 1), (24, 24, 24))
    draw = ImageDraw.Draw(out)
    y = 0
    for label, im in tiles:
        draw.rectangle((0, y, width, y + bar), fill=(40, 90, 200))
        draw.text((12, y + 8), label, fill=(255, 255, 255), font=font)
        out.paste(im, (0, y + bar))
        y += bar + im.height
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, "JPEG", quality=88)
    return Path(out_path)


def look_at_video(ctx, path: str, *, every: float = EVERY_SECONDS,
                  max_frames: int = MAX_FRAMES, work_dir: Optional[Path] = None,
                  vision: Optional[Callable] = None, lang: str = "ru",
                  transcript: str = "", offset: float = 0.0) -> dict:
    """Return {"duration", "times", "sheet", "description"}; description is ""
    when nothing could be seen (no frames, vision down).

    `transcript` -- what is SAID in the clip, if already known. The ears help
    the eyes: live 23:01 a grow lamp on a stand was called «камера на
    штативе» in every frame while the speaker talked about turning the lamp
    towards the window. Named objects in the speech settle what a small
    frame cannot; the model is still told to trust the picture for what is
    not in the words."""
    if vision is None:
        import llm as _llm
        vision = lambda frames, question, system: _llm.analyze_image_with_llm(  # noqa: E731
            ctx, image_paths=frames, user_text=question, system_prompt=system,
            temperature=0.1, max_tokens=900)
    dur = duration_seconds(path)
    work_dir = Path(work_dir or tempfile.mkdtemp(prefix="video_look_"))
    try:
        times = key_times(dense_pass(path, work_dir / "dense"), max_frames)
        how = "key moments"
    except Exception:
        logger.warning("video_look: dense pass failed, falling back to every %ss", every, exc_info=True)
        times = sample_times(dur, every, max_frames)
        how = f"every {every:.0f} s"
    frames = extract_frames(path, times, work_dir / "frames")
    if not frames:
        logger.warning("video_look: no frames from %s (%.1fs)", os.path.basename(path), dur)
        return {"duration": dur, "times": [], "sheet": "", "description": ""}
    sheet = make_sheet(frames, work_dir / "sheet.jpg", offset=offset)
    # Every frame goes to the model as its OWN picture, in one call, each
    # labelled with its time. The sheet gave a frame ~CELL px and was shrunk
    # again (live 2026-10-01: a projector became «красный цилиндрический
    # объект»); frames read one by one lost how they follow each other. One
    # call with all frames at full size has both (and is 1 call, not 7).
    # The sheet is still made: it is the picture a follow-up question sees.
    q = ((f"These are {len(frames)} frames from the portion {_stamp(offset)}-{_stamp(offset + dur)} "
          f"of a longer video -- the " if offset else
          f"These are {len(frames)} frames from a {dur:.0f}-second video -- the ")
         + f"sharpest frame of each moment where the picture changed ({how}), each "
         f"shown above with its timestamp: "
         + ", ".join(_stamp(t + offset) for t, _ in frames)
         + ". Give the storyboard, one line per timestamp, then the overall sentence.")
    if (transcript or "").strip():
        q += ("\n\nWhat the speaker SAYS in this video. Use it to NAME the things "
              "you see (a thing on a stand the speaker calls a lamp is a lamp), never "
              "to add things that are not in the frames -- if the frames show none "
              "of what is said, describe only what they show:\n«"
              + transcript.strip()[:1500] + "»")
    try:
        desc = (vision([(f"Frame {_stamp(t + offset)}:", str(p)) for t, p in frames], q, _SHEET_SYSTEM + _LANG_LINE.get(lang, _LANG_LINE["en"])) or "").strip()
    except Exception:
        logger.exception("video_look: vision failed")
        desc = ""
    logger.info("video_look: %s %.0fs -> %d frames (%s) at %s; %s", os.path.basename(path), dur,
                len(frames), how, [_stamp(t) for t, _ in frames], desc[:120].replace("\n", " "))
    return {"duration": dur, "times": [t + offset for t, _ in frames], "sheet": str(sheet),
            "description": desc}
