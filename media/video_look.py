"""Look at a video, not only listen to it.

A forwarded round video (33 s, a grow lamp on a stand
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
from seams import vision_model

import logging
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("assistant.video_look")
# The forwarded clip, kept beside its frame sheet so a typed «продолжи» can continue it.
CONTINUE_CLIP = "clip.mp4"

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
    "concrete: name things, not impressions. Say what the people or animals DO -- "
    "the action, not only the pose: chewing its paw, licking, jumping at the camera, "
    "waving. A home video is about that moment, and a frame-by-frame list of poses "
    "missed it (live 2026-10-16: a dog gnawing its own paw came back as «сидит и "
    "смотрит в сторону»). A smeared frame is fast movement, never «a blurry spot»: say "
    "what moves and how, from the frames around it and the speech. When a frame repeats the previous "
    "one, say so in a few words. After the lines add ONE sentence starting with "
    "the overall label given below that says what the whole video is about -- what "
    "happens in it, and what is funny or striking when it is a moment. Name "
    "things for what they are, not by shape or colour. Do not invent sound or speech -- you only see the picture.\n"
    "When a frame shows a document with small print -- a receipt, a bill, a menu, a price list, a "
    "screen, a letter -- add one last line «TEXT AT: 0:13, 0:14» with those timestamps; the "
    "document is then read from sharper frames, so never guess its figures yourself.")

# A receipt panned past a кружок's camera: the storyboard frame showed «RACUN - NARUDZBA» and
# nothing else, the names on one frame, the prices on the next (live 2026-10-24, a guessed
# «12 €» for tea that cost 6,50). The stretch is cut again at full size and READ_FRAMES sharp,
# spread-out frames go to one call that merges them into one copy.
_TEXT_AT = re.compile(r"^\W*TEXT AT:\s*(.*)$", re.I | re.M)
READ_FPS, READ_FRAMES, READ_PAD = 6, 6, 2.0
_READ_SYSTEM = (
    "These frames show ONE document (a receipt, menu, screen, letter...) filmed by a moving "
    "hand-held camera; each frame may show a different part of it. Transcribe its text exactly as "
    "printed, line by line, merging what the frames show into one copy: every item line with its "
    "quantity, unit price and total, names, dates, totals. Keep each figure in its own column: "
    "an item's quantity times its unit price is its total, and a line where it is not is misread -- "
    "look at it again (a tax-rate column of 15,00 was copied as a dessert's total). Write a line you cannot read as "
    "[неразборчиво]; never guess a figure.\n"
    "Money with no currency sign: when the document shows where it comes from (a city, a country, a "
    "country code in a tax number), end with one line «Currency: <ISO code> -- <the place and how the "
    "document shows it>». A receipt from Cetinje answered «валюта не указана, я взял евро» (2026-10-24).")

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
# A short clip is one moment: more candidates per window, so the pick is a sharp frame
# of the action and not the smear of it (a 7 s кружок at 2 fps had two frames a second,
# both blurred while the dog jumped).
SHORT_CLIP_FPS = ((20, 6), (60, 4))


def dense_fps(duration: float) -> float:
    return next((fps for limit, fps in SHORT_CLIP_FPS if 0 < duration <= limit), DENSE_FPS)
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


def _stamp(t: float, tenths: bool = False) -> str:
    """«0:07»; with `tenths` «0:02.3», so two picks inside one second of a short clip
    are not both labelled «0:02»."""
    whole = int(t); h, r = divmod(whole, 3600); m, s = divmod(r, 60)
    out = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    return out + (f".{min(9, int(round((t - whole) * 10)))}" if tenths else "")


_LINE_STAMP = re.compile(r"^(\W*)(?:(\d+):)?(\d{1,2}):(\d{2})(?!\d)")


def absolute_stamps(desc: str, times: list[float], offset: float) -> str:
    """Put the real timestamp at the head of every storyboard line. With a transcript in
    the prompt the model sometimes counts from the portion's start (0:05 for 6:05); the
    frame the line means is the one nearest to what it wrote, read either way."""
    if not offset or not times:
        return desc
    out = []
    for line in (desc or "").splitlines():
        m = _LINE_STAMP.match(line)
        if m:
            h, mi, sec = int(m.group(2) or 0), int(m.group(3)), int(m.group(4))
            said = h * 3600 + mi * 60 + sec
            best = min(((abs(t - said), t) for t in times), default=None)
            best_rel = min(((abs(t - (said + offset)), t) for t in times), default=None)
            t = (best_rel if best_rel[0] < best[0] else best)[1]
            line = m.group(1) + _stamp(t) + line[m.end():]
        out.append(line)
    return "\n".join(out)


def make_sheet(frames: list[tuple[float, Path]], out_path: Path, offset: float = 0.0,
               tenths: bool = False) -> Path:
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
        label = _stamp(t + offset, tenths)
        draw.rectangle((x, y, x + 24 + 20 * len(label), y + 44), fill=(255, 210, 0))
        draw.text((x + 6, y + 2), label, fill=(0, 0, 0), font=font)
    sheet.save(out_path, "JPEG", quality=88)
    return out_path


def _seconds(stamp: str) -> Optional[float]:
    parts = [p for p in re.findall(r"\d+(?:\.\d+)?", stamp)]
    if len(parts) < 2:
        return None
    *hm, s = parts
    return sum(float(x) * 60 ** (len(hm) - i) for i, x in enumerate(hm)) + float(s)


def read_document(path: str, stamps: list[float], work_dir: Path, vision: Callable,
                  duration: float) -> str:
    """The text of the document seen at `stamps`, from the sharpest frames around them."""
    import numpy as np
    from PIL import Image
    lo = max(0.0, min(stamps) - READ_PAD)
    hi = min(duration or max(stamps) + READ_PAD, max(stamps) + READ_PAD)
    out = work_dir / "read"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("r_*.png"):
        f.unlink()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{lo:.2f}", "-t", f"{hi - lo:.2f}",
                    "-i", path, "-vf", f"fps={READ_FPS}", str(out / "r_%04d.png")],
                   capture_output=True, check=True, timeout=120)
    scored = []
    for i, f in enumerate(sorted(out.glob("r_*.png"))):
        gray = Image.open(f).convert("L")
        gray = gray.resize((256, max(1, round(256 * gray.height / gray.width))))
        scored.append((_sharpness(np.asarray(gray, dtype=np.float32)), lo + i / READ_FPS, f))
    # the sharpest frame of each equal slice: the four sharpest overall were all of the price
    # column and the item names, shown half a second later, never reached the reader
    k = min(READ_FRAMES, len(scored))
    picks = [max(scored[len(scored) * j // k: len(scored) * (j + 1) // k]) for j in range(k)]
    frames = []
    for _, t, f in sorted(picks, key=lambda p: p[1]):
        im = Image.open(f).convert("RGB")
        if im.width < 768:            # a кружок is 384 px; the print is a few pixels tall
            im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
            im.save(f)
        frames.append((f"Frame {t:.2f} s:", str(f)))
    if not frames:
        return ""
    return (vision(frames, "Transcribe the document.", _READ_SYSTEM) or "").strip()


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
                  transcript: str = "", offset: float = 0.0, caption: str = "") -> dict:
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
        vision = lambda frames, question, system: vision_model(ctx)(  # noqa: E731
            ctx, image_paths=frames, user_text=question, system_prompt=system,
            temperature=0.1, max_tokens=900)
    dur = duration_seconds(path)
    work_dir = Path(work_dir or tempfile.mkdtemp(prefix="video_look_"))
    try:
        times = key_times(dense_pass(path, work_dir / "dense", fps=dense_fps(dur)), max_frames)
        how = "key moments"
    except Exception:
        logger.warning("video_look: dense pass failed, falling back to every %ss", every, exc_info=True)
        times = sample_times(dur, every, max_frames)
        how = f"every {every:.0f} s"
    frames = extract_frames(path, times, work_dir / "frames")
    if not frames:
        logger.warning("video_look: no frames from %s (%.1fs)", os.path.basename(path), dur)
        return {"duration": dur, "times": [], "sheet": "", "description": ""}
    tenths = len({_stamp(t + offset) for t, _ in frames}) < len(frames)
    sheet = make_sheet(frames, work_dir / "sheet.jpg", offset=offset, tenths=tenths)
    # Every frame goes to the model as its OWN picture, in one call, each
    # labelled with its time. The sheet gave a frame ~CELL px and was shrunk
    # again (a projector became «красный цилиндрический
    # объект»); frames read one by one lost how they follow each other. One
    # call with all frames at full size has both (and is 1 call, not 7).
    # The sheet is still made: it is the picture a follow-up question sees.
    q = ((f"These are {len(frames)} frames from the portion {_stamp(offset)}-{_stamp(offset + dur)} "
          f"of a longer video -- the " if offset else
          f"These are {len(frames)} frames from a {dur:.0f}-second video -- the ")
         + f"sharpest frame of each moment where the picture changed ({how}), each "
         f"shown above with its timestamp: "
         + ", ".join(_stamp(t + offset, tenths) for t, _ in frames)
         + ". Give the storyboard, one line per timestamp, then the overall sentence.")
    if (transcript or "").strip():
        q += ("\n\nWhat the speaker SAYS in this video. Use it to NAME the things "
              "you see (a thing on a stand the speaker calls a lamp is a lamp) and to READ "
              "what is happening: when the speaker says what someone does («он чуть лапу "
              "свою не съел»: the dog gnaws its paw), find it in the frames and say it on "
              "the lines where it shows. Never add things that are not in the frames -- if "
              "the frames show none of what is said, describe only what they show:\n«"
              + transcript.strip()[:1500] + "»")
    if (caption or "").strip():
        q += ("\n\nThe sender's caption over this video: «" + caption.strip()[:300] + "». It is the "
              "setup of the joke when the picture does not match it: say what the frames show, "
              "quote any text on them, and do not describe the scene as if it were serious.")
    try:
        desc = (vision([(f"Frame {_stamp(t + offset, tenths)}:", str(p)) for t, p in frames], q, _SHEET_SYSTEM + _LANG_LINE.get(lang, _LANG_LINE["en"])) or "").strip()
    except Exception:
        logger.exception("video_look: vision failed")
        desc = ""
    m = _TEXT_AT.search(desc)
    if m:
        desc = _TEXT_AT.sub("", desc).strip()
        at = [s - offset for s in (_seconds(x) for x in m.group(1).split(",")) if s is not None]
        try:
            text = read_document(path, at, work_dir, vision, dur) if at else ""
        except Exception:
            logger.exception("video_look: reading the document failed")
            text = ""
        logger.info("video_look: document at %s read: %s", m.group(1).strip(), text[:160].replace("\n", " | "))
        if text:
            label = "Текст документа в видео ({}), дословно" if lang == "ru" else "Text of the document in the video ({}), verbatim"
            desc += "\n\n" + label.format(m.group(1).strip()) + ":\n" + text
    logger.info("video_look: %s %.0fs -> %d frames (%s) at %s; %s", os.path.basename(path), dur,
                len(frames), how, [_stamp(t) for t, _ in frames], desc[:120].replace("\n", " "))
    times = [t + offset for t, _ in frames]
    desc = absolute_stamps(desc, times, offset)
    return {"duration": dur, "times": times, "sheet": str(sheet), "description": desc, "gist": gist(desc)}


def gist(desc: str) -> str:
    """The overall line and the start of a document read out of the clip: what one chat line
    can say about the video."""
    lines = [l.strip() for l in (desc or "").splitlines() if l.strip()]
    overall = next((l for l in lines if re.match(r"(Итог|Overall)\s*:", l)), "")
    parts = re.split(r"(?:дословно|verbatim):\n", desc or "", maxsplit=1)
    doc = parts[1] if len(parts) == 2 else ""
    return " ".join(filter(None, [overall, "Документ: " + " ".join(doc.split())[:200] if doc else ""]))
