"""Model-based prompt-injection scan for text the user did NOT write (web pages, attached files,
video descriptions). prompt_guard.py is the cheap regex layer for facts/summaries; this is the
multilingual classifier layer (Horizon-Labs/prompt-injection-guard-base, mmBERT 308M, Apache-2.0, CPU ~30 ms/chunk).

scrub() cuts out ONLY the chunks the model flags -- the rest of the page is still usable -- and
fails open (returns the text) when the model is missing: a defence must not take the feature down."""
import logging
import os
import re
import threading

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.getenv("INJECTION_GUARD_MODEL", os.path.join(ROOT, "models_ext", "prompt-guard"))
THRESHOLD = float(os.getenv("INJECTION_GUARD_THRESHOLD", "0.5"))
CHUNK = 900                         # chars; the model reads up to 8k tokens but a smaller window localises the cut
REMOVED = "[removed: text aimed at the AI, not at the reader]"

_lock = threading.Lock()
_tok = _model = None
_failed = False


def _load() -> bool:
    global _tok, _model, _failed
    if _model is not None:
        return True
    if _failed:
        return False
    with _lock:
        if _model is not None:
            return True
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            _tok = AutoTokenizer.from_pretrained(MODEL)
            _model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
            return True
        except Exception:
            _failed = True
            logger.warning("[injection_scan] guard model unavailable at %s; scan is off", MODEL, exc_info=True)
            return False


def scores(chunks: list) -> list:
    """Injection probability per chunk; [] when the model is unavailable."""
    if not chunks or not _load():
        return []
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(chunks), 16):
            enc = _tok(chunks[i:i + 16], truncation=True, max_length=512, padding=True, return_tensors="pt")
            enc.pop("token_type_ids", None)
            out += torch.softmax(_model(**enc).logits, -1)[:, 1].tolist()
    return out


def _chunks(text: str) -> list:
    """Paragraph-aligned pieces of about CHUNK characters."""
    pieces, cur = [], ""
    for para in re.split(r"(?<=\n)\n+", text):
        while len(para) > CHUNK * 2:                      # one endless line: cut it
            pieces.append(para[:CHUNK]); para = para[CHUNK:]
        if cur and len(cur) + len(para) > CHUNK:
            pieces.append(cur); cur = ""
        cur += para
    if cur:
        pieces.append(cur)
    return pieces


def scrub(text: str, source: str = "") -> str:
    """`text` with every flagged chunk replaced by a marker."""
    if not text or len(text.strip()) < 20:
        return text
    try:
        ch = _chunks(text)
        sc = scores(ch)
        if not sc:
            return text
        bad = [i for i, s in enumerate(sc) if s >= THRESHOLD]
        if not bad:
            return text
        logger.warning("[injection_scan] %s: %d of %d chunks flagged (max %.2f): %r", source or "text", len(bad),
                       len(ch), max(sc), ch[bad[0]][:120])
        return "".join(REMOVED + "\n\n" if i in bad else c for i, c in enumerate(ch))
    except Exception:
        logger.warning("[injection_scan] scan failed; text passed through", exc_info=True)
        return text
