"""Reminders for the Telegram bot: "напомни через 2 минуты выпить воды".

Live 2026-09-28 the bot answered "Договорились! Напомню" with no way to do it.
Pending reminders live in a JSON file next to the bot's data, so a restart
re-arms them (an overdue one fires at once, marked late); a threading.Timer
per reminder does the sending through the sender the bot registers at start.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger("assistant.reminders")

_LOCK = threading.Lock()
_PATH: Path | None = None
_SENDER = None                      # callable(chat_id: int, text: str) -> None
_ITEMS: dict[str, dict] = {}        # id -> {"owner", "due", "text"}
_TIMERS: dict[str, threading.Timer] = {}
MAX_PER_OWNER = 20
# Longest single Timer sleep. A reminder further out than this wakes up early
# and re-arms. On Windows a lock wait cannot exceed threading.TIMEOUT_MAX
# (~49.7 days): a Timer for "через 2 года" raised OverflowError on its own
# thread and the reminder silently never fired.
_MAX_TIMER_S = 86400.0


def _save() -> None:
    if _PATH is None:
        return
    try:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(_ITEMS, ensure_ascii=False), encoding="utf-8")
        tmp.replace(_PATH)
    except Exception:
        logger.exception("could not save reminders")


def _fire(rid: str) -> None:
    with _LOCK:
        cur = _ITEMS.get(rid)
        if cur is not None and cur["due"] - time.time() > 1.0:
            _arm(rid)                   # a capped wait ended early: sleep on
            return
        item = _ITEMS.pop(rid, None)
        _TIMERS.pop(rid, None)
        # «каждый день в 9» was stored as a one-off while the bot promised
        # daily (live 2026-09-28): a repeating one re-arms for its next slot.
        if item and item.get("every"):
            nxt = dict(item)
            while nxt["due"] <= time.time():
                nxt["due"] += nxt["every"]
            _ITEMS[rid] = nxt
            _arm(rid)
        _save()
    if not item or _SENDER is None:
        return
    late = time.time() - item["due"] > 120
    try:
        # "remind me in 1 minute" fired as «⏰ Напоминание: check the oven» (live).
        pre = (("⏰ Reminder (late): " if late else "⏰ Reminder: ") if item.get("lang") == "en"
               else ("⏰ Напоминание (с опозданием): " if late else "⏰ Напоминание: "))
        _SENDER(int(item["owner"]), pre + item["text"])
    except Exception:
        logger.exception("reminder %s could not be sent", rid)


def _arm(rid: str) -> None:
    wait = min(_MAX_TIMER_S, max(0.0, _ITEMS[rid]["due"] - time.time()))
    t = threading.Timer(wait, _fire, (rid,))
    t.daemon = True
    _TIMERS[rid] = t
    t.start()


def register(sender, path) -> int:
    """Called once by the bot at start. Returns how many reminders were re-armed."""
    global _SENDER, _PATH
    with _LOCK:
        _SENDER, _PATH = sender, Path(path)
        try:
            loaded = json.loads(_PATH.read_text(encoding="utf-8")) if _PATH.exists() else {}
        except Exception:
            logger.exception("reminders file unreadable -- starting empty")
            loaded = {}
        for rid, item in loaded.items():
            if rid not in _ITEMS:
                _ITEMS[rid] = item
                _arm(rid)
        return len(loaded)


def add(owner, delay_s: float, text: str, every_s: float = 0, lang: str = "ru") -> str:
    """Schedule `text` for `owner` in `delay_s` seconds. Raises ValueError."""
    if _SENDER is None:
        raise ValueError("reminders are not running here")
    if not (0 < delay_s <= 5 * 366 * 86400):   # "продлить паспорт через 2 года" is real
        raise ValueError("the time must be in the future and within five years")
    with _LOCK:
        if sum(1 for i in _ITEMS.values() if str(i["owner"]) == str(owner)) >= MAX_PER_OWNER:
            raise ValueError(f"already {MAX_PER_OWNER} reminders pending")
        rid = uuid.uuid4().hex[:10]
        _ITEMS[rid] = {"owner": str(owner), "due": time.time() + delay_s, "text": text.strip()[:500]}
        if every_s:
            _ITEMS[rid]["every"] = float(every_s)
        if lang == "en":
            _ITEMS[rid]["lang"] = "en"
        _save()
        _arm(rid)
    return rid


def pending(owner) -> list[dict]:
    with _LOCK:
        return sorted((dict(i, id=k) for k, i in _ITEMS.items() if str(i["owner"]) == str(owner)),
                      key=lambda i: i["due"])


def cancel(owner, words: str) -> list[str]:
    """Drop the owner's reminders whose text contains any word of `words`
    ('all' or empty = every one). Returns the cancelled texts."""
    ws = [w for w in words.lower().split() if len(w) > 2 and w not in ("про", "все", "всё", "мне", "напоминание", "напоминания")]
    everything = not ws or words.strip().lower() in ("all", "все", "всё")
    gone = []
    with _LOCK:
        for rid, item in list(_ITEMS.items()):
            if str(item["owner"]) != str(owner):
                continue
            low = item["text"].lower()
            if everything or any(w[:max(3, len(w) - 2)] in low for w in ws):
                _ITEMS.pop(rid)
                t = _TIMERS.pop(rid, None)
                if t:
                    t.cancel()
                gone.append(item["text"])
        if gone:
            _save()
    return gone


def stop() -> None:
    """Cancel the timers (the file keeps the reminders for the next start)."""
    global _SENDER
    with _LOCK:
        for t in _TIMERS.values():
            t.cancel()
        _TIMERS.clear()
        _ITEMS.clear()
        _SENDER = None
