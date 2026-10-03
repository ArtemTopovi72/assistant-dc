"""The character registry: who exists, which LoRA is theirs, who is in Telegram.

One store, read by two surfaces. The desktop tab creates and trains characters;
the Telegram bot offers them under Творчество -> Персонажи. The "available in
Telegram" checkbox is the reason this module exists at all: keep that flag in
the GUI and the bot would each hold their own copy of it, and the two would
disagree the first time either was edited while the other ran.

Storage is a single JSON file under runtime/, which .gitignore excludes -- a
character record names a folder of someone's private photos, so it must never
reach git.

Status is a fact about the artifact, not a wish:

    dataset   images are prepared, no adapter exists yet
    training  a run is in progress
    ready     a .safetensors exists on disk and can be loaded
    failed    the last run ended without producing one

`ready` is not taken on trust. list_characters() re-checks the file each call
and demotes a character whose adapter has been moved or deleted, because the
alternative is a bot that offers a persona it cannot render.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from pathlib import Path

_LOCK = threading.RLock()
_STORE = Path(__file__).resolve().parents[1].joinpath("runtime") / "characters.json"

VALID_STATUS = ("dataset", "training", "ready", "failed")
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def redirect_store(path) -> None:
    """Point the registry at `path`. Call this before touching it in tests.

    Without it a test writes into the operator's live character list -- the
    same class of leak that once registered test chat ids in the real user DB.
    """
    global _STORE
    _STORE = Path(path)


# Names in this house are Russian, and the slug becomes a filename, a LoRA
# trigger word and a Telegram callback token -- three places where Cyrillic
# either breaks or silently mangles. Transliterate rather than drop: without
# this, slugify("НейроСтепан") returns "character" and every Russian
# name collides on that one id.
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def slugify(name: str) -> str:
    low = (name or "").strip().lower()
    s = "".join(_TRANSLIT.get(ch, ch) for ch in low)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:64] or "character"


def _read() -> dict:
    try:
        raw = json.loads(_STORE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _write(data: dict) -> None:
    _STORE.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-replace: a half-written registry would lose every character,
    # and this file is edited from the GUI thread while the bot polls it.
    fd, tmp = tempfile.mkstemp(dir=str(_STORE.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, _STORE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _settle(rec: dict) -> dict:
    """Reconcile a stored record with what is actually on disk."""
    rec = dict(rec)
    lora = rec.get("lora_file") or ""
    have = bool(lora) and Path(lora).is_file()
    if rec.get("status") == "ready" and not have:
        rec["status"] = "failed"
        rec["note"] = "adapter file is missing: " + str(lora)
    rec["lora_exists"] = have
    if not have:
        # A character whose adapter is gone cannot be offered anywhere, no
        # matter what the checkbox says.
        rec["tg_available"] = False
    else:
        rec["tg_available"] = bool(rec.get("tg_enabled"))
    return rec


def list_characters() -> list[dict]:
    """Every character, ordered as created. Records are settled against disk."""
    with _LOCK:
        data = _read()
    out = [_settle(v) for k, v in data.items() if isinstance(v, dict)]
    out.sort(key=lambda r: (r.get("created") or 0, r.get("slug") or ""))
    return out


def get(slug: str) -> dict | None:
    with _LOCK:
        rec = _read().get(slug)
    return _settle(rec) if isinstance(rec, dict) else None


def telegram_characters() -> list[dict]:
    """What the bot may offer: enabled AND backed by an adapter that exists."""
    return [c for c in list_characters() if c.get("tg_available")]


def upsert(slug: str, **fields) -> dict:
    """Create or update one character. Unknown fields are kept as given."""
    if not _SLUG_RE.match(slug or ""):
        raise ValueError("bad character id: %r" % (slug,))
    status = fields.get("status")
    if status is not None and status not in VALID_STATUS:
        raise ValueError("bad status: %r" % (status,))
    with _LOCK:
        data = _read()
        rec = dict(data.get(slug) or {})
        rec.update({k: v for k, v in fields.items() if v is not None})
        rec["slug"] = slug
        rec.setdefault("name", slug)
        rec.setdefault("trigger", slug)
        rec.setdefault("status", "dataset")
        rec.setdefault("tg_enabled", False)
        # 1.5, not 0.9: this field is read ONLY by the Telegram render
        # path (tg_characters._render_character), and the desktop tab has
        # its own spin box. The bot asked for a firmer identity than the
        # default gave.
        rec.setdefault("strength", 1.5)
        rec.setdefault("created", int(__import__("time").time()))
        data[slug] = rec
        _write(data)
    return _settle(rec)


def set_tg_enabled(slug: str, enabled: bool) -> dict:
    return upsert(slug, tg_enabled=bool(enabled))


def delete(slug: str) -> bool:
    """Forget a character. The dataset and the adapter file are left alone."""
    with _LOCK:
        data = _read()
        if slug not in data:
            return False
        del data[slug]
        _write(data)
    return True
