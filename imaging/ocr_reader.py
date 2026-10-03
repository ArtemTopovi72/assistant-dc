"""A deterministic reader for the lettering in a render (EasyOCR, ru+en, CPU).

The vision model reads Cyrillic badly and deterministically: a crisp,
hand-painted "У ОЛЬГИ" came back "УАЗЬЛ" four times out of four (live,
2026-09-12), so a perfect café sign was condemned, redrawn twice, and the
agent spent nine more minutes "fixing" it. EasyOCR on the same crop reads
"У ОАЬГИ" -- one letter off in a stylised font -- and the three-line
champagne label character for character.

It is the FIRST reader in draw_text.verify_text: when it confirms the words
the vision call is skipped altogether (no model swap, no hallucination);
when it does not, the vision model still gets its say and the better of the
two readings counts. CPU on purpose -- the chat model has the card.

Install (never let pip touch the CUDA torch pin -- see memory torch-cuda-pin-hazard):
    venv/Scripts/pip install --no-deps easyocr
    venv/Scripts/pip install opencv-python-headless python-bidi Shapely pyclipper ninja
The ru+en recognition weights (~100 MB) download on first use into ~/.EasyOCR.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger("assistant.ocr")

_lock = threading.Lock()
_reader = None
_failed = False


def available() -> bool:
    """True when EasyOCR can be imported (it need not be loaded yet)."""
    if os.getenv("F5_TEST_RUN"):
        return False
    try:
        import easyocr  # noqa: F401
        return True
    except Exception:
        return False


class _WorkerReader:
    """`readtext` of an EasyOCR reader living in ocr_worker.py's process. Same
    call shape as easyocr.Reader.readtext (path or RGB array), so callers are
    unchanged. Started once, model kept loaded; restarted if it dies."""

    def __init__(self):
        import subprocess
        import sys
        import json
        self._json = json
        exe = sys.executable.replace("pythonw.exe", "python.exe")
        self._p = subprocess.Popen(
            [exe, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "imaging/ocr_worker.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        first = self._p.stdout.readline()
        if '"ready"' not in first:
            raise RuntimeError("ocr_worker did not start: %r" % first[:200])
        self._io = threading.Lock()

    def alive(self) -> bool:
        return self._p.poll() is None

    def readtext(self, image, detail=1, paragraph=False):
        import tempfile
        tmp = None
        if not isinstance(image, str):
            from PIL import Image
            fd, tmp = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            Image.fromarray(image).save(tmp)
            image = tmp
        try:
            with self._io:
                self._p.stdin.write(self._json.dumps({"path": image}) + "\n")
                self._p.stdin.flush()
                line = self._p.stdout.readline()
            res = self._json.loads(line or "{}")
            if "ok" not in res:
                raise RuntimeError("ocr_worker: %s" % (res.get("error") or "no answer"))
            return [(box, text, conf) for box, text, conf in res["ok"]]
        finally:
            if tmp:
                try:
                    os.remove(tmp)
                except OSError:
                    pass


def _get():
    global _reader, _failed
    if _reader is not None and not getattr(_reader, "alive", lambda: True)():
        _reader = None                       # the worker died: start a new one
    if _reader is not None or _failed:
        return _reader
    with _lock:
        if _reader is not None or _failed:
            return _reader
        try:
            import easyocr  # noqa: F401 -- installed? the worker imports it for real
            _reader = _WorkerReader()
            logger.info("EasyOCR reader ready (ru+en, CPU, own process)")
        except Exception:
            _failed = True
            logger.warning("EasyOCR unavailable -- lettering read-back falls back to "
                           "the vision model alone", exc_info=True)
    return _reader


def read(image_path: str) -> Optional[list]:
    """Every piece of lettering EasyOCR finds, largest first, plus the pieces
    joined left-to-right as one more candidate. None when it cannot run."""
    if os.getenv("F5_TEST_RUN") or not image_path or not os.path.exists(image_path):
        return None
    r = _get()
    if r is None:
        return None
    # Two passes, at the crop's own size and doubled: a stylised letter the
    # recogniser misses at one scale it often gets at the other ("ОАЬГИ" at
    # 1x, "УОХЬГИ" at 2x for "У ОЛЬГИ"), and the caller keeps the best match.
    variants = [image_path]
    try:
        from PIL import Image
        import numpy as np
        im = Image.open(image_path).convert("RGB")
        if max(im.size) < 1600:
            variants.append(np.asarray(im.resize((im.width * 2, im.height * 2), Image.LANCZOS)))
    except Exception:
        pass
    out = []
    for v in variants:
        try:
            out.append(r.readtext(v, detail=1, paragraph=False))
        except Exception:
            logger.warning("EasyOCR failed on %s", image_path, exc_info=True)
    if not out:
        return None
    pieces, _passes = [], []
    for chunk in out:
        pass_pieces = []
        for box, text, _conf in chunk:
            t = str(text or "").strip()
            if not t:
                continue
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            pass_pieces.append((min(ys), min(xs), area, t))
        pieces.extend(pass_pieces)
        _passes.append(pass_pieces)
    if not pieces:
        return []
    seen, by_size = set(), []
    for *_, t in sorted(pieces, key=lambda p: -p[2]):
        if t not in seen:
            seen.add(t); by_size.append(t)
    # reading order per pass: top-to-bottom rows, left-to-right within a row;
    # the pieces of each pass joined are one more candidate each.
    for chunk in _passes:
        ordered = [t for *_, t in sorted(chunk, key=lambda p: (round(p[0] / 20), p[1]))]
        joined = " ".join(ordered)
        if len(chunk) > 1 and joined not in seen:
            seen.add(joined); by_size.append(joined)
    return by_size
