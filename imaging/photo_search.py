"""Find photos matching a description inside a folder of pictures.

Live 2026-09-14: «найди мост» over a 173-photo DCIM.zip. The agent had
open_image + inspect_image -- one picture per two tool calls -- so 173 photos
were 346 calls, far past the round limit. It could not do the task, so it
narrated a plan instead ("Сначала я распакую архив ... Начинаю поиск.") and
ended the turn, twice. No prompt fixes a task the tools cannot carry.

Two stages, both on the house vision model:
  1. contact sheets -- thumbnails tiled in a numbered grid, one vision call
     per sheet asks which cells show the thing (173 photos -> ~15 calls);
  2. verification -- every candidate is shown alone and asked yes/no, so a
     grid misread never reaches the user as a "found" photo.

Numbers over the whole DCIM.zip on the 3090: ~12 photos per sheet, ~6 s per
sheet, ~3 s per verification -- a 173-photo folder in under two minutes.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("assistant.photo_search")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
# 4x3 cells of 320 px: the sheet stays under the 1536-px vision proxy while a
# bridge-sized subject still covers a few hundred pixels.
GRID_COLS, GRID_ROWS, CELL = 4, 3, 320
PER_SHEET = GRID_COLS * GRID_ROWS
MAX_PHOTOS = 600

_SHEET_SYSTEM = (
    "You are looking at a contact sheet: a grid of numbered photo thumbnails. "
    "Each cell has its number drawn in the top-left corner. Answer ONLY with a "
    "JSON list of the cell numbers whose photo clearly shows the requested "
    "thing, e.g. [3, 7]. An empty list [] if none. No prose.")
_VERIFY_SYSTEM = (
    "Answer with one word, YES or NO: does this photo clearly show the "
    "requested thing? Be strict: NO when you are guessing.")


def list_photos(folder: Path) -> list[Path]:
    out = []
    for p in sorted(folder.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
            continue
        # Hidden folders hold our own contact sheets and the run_code
        # scripts' home; live 2026-09-14 a sheet came back as a "bridge".
        if any(part.startswith(".") for part in p.relative_to(folder).parts):
            continue
        out.append(p)
    return out[:MAX_PHOTOS]


def make_sheet(paths: list[Path], out_path: Path) -> Path:
    """Tile up to PER_SHEET photos into a numbered grid, numbers 1-based."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    sheet = Image.new("RGB", (GRID_COLS * CELL, GRID_ROWS * CELL), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 44)
    except Exception:
        font = ImageFont.load_default()
    for i, p in enumerate(paths[:PER_SHEET]):
        try:
            im = Image.open(p)
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((CELL - 8, CELL - 8))
        except Exception:
            continue
        x = (i % GRID_COLS) * CELL + 4
        y = (i // GRID_COLS) * CELL + 4
        sheet.paste(im, (x, y))
        label = str(i + 1)
        draw.rectangle((x, y, x + 34 + 22 * len(label), y + 52), fill=(255, 210, 0))
        draw.text((x + 8, y + 2), label, fill=(0, 0, 0), font=font)
    sheet.save(out_path, "JPEG", quality=88)
    return out_path


def _parse_cells(text: str, n: int) -> list[int]:
    text = text or ""
    m = re.search(r"\[[^\]]*\]", text)
    raw = m.group(0) if m else text
    try:
        vals = json.loads(raw)
        if not isinstance(vals, list):
            vals = []
    except Exception:
        vals = [int(v) for v in re.findall(r"\d+", raw)]
    return sorted({int(v) for v in vals if isinstance(v, (int, float)) and 1 <= int(v) <= n})


def find_in_photos(ctx, folder: Path, query: str, *, work_dir: Optional[Path] = None,
                   vision: Optional[Callable] = None, progress: Optional[Callable] = None,
                   verify: bool = True, max_results: int = 8) -> dict:
    """Return {"matches": [paths], "scanned": n, "sheets": k, "candidates": m}."""
    if vision is None:
        import llm as _llm
        vision = lambda path, question, system: _llm.analyze_image_with_llm(   # noqa: E731
            ctx, image_path=str(path), user_text=question, system_prompt=system,
            temperature=0.0, max_tokens=120)
    photos = list_photos(Path(folder))
    work_dir = Path(work_dir or (Path(folder) / ".contact_sheets"))
    work_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[Path] = []
    sheets = 0
    for start in range(0, len(photos), PER_SHEET):
        batch = photos[start:start + PER_SHEET]
        sheet = make_sheet(batch, work_dir / f"sheet_{sheets:03d}.jpg")
        sheets += 1
        if progress:
            progress(f"sheet {sheets}/{(len(photos) + PER_SHEET - 1) // PER_SHEET}")
        ans = vision(sheet, f"Which numbered cells show: {query}?", _SHEET_SYSTEM)
        cells = _parse_cells(ans or "", len(batch))
        logger.info("photo_search sheet %d: %s -> %s", sheets, (ans or "")[:80], cells)
        candidates.extend(batch[c - 1] for c in cells)
    matches: list[Path] = []
    for p in candidates:
        if not verify:
            matches.append(p)
        else:
            ans = (vision(p, f"Does this photo clearly show: {query}?", _VERIFY_SYSTEM) or "")
            ok = bool(re.match(r"\s*yes", ans, re.IGNORECASE))
            logger.info("photo_search verify %s: %s", p.name, (ans or "")[:40])
            if ok:
                matches.append(p)
        if len(matches) >= max_results:
            break
    return {"matches": matches, "scanned": len(photos), "sheets": sheets,
            "candidates": len(candidates)}
