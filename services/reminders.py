"""Reminders for the Telegram bot: "напомни через 2 минуты выпить воды".

Pending reminders live in a JSON file next to the bot's data, so a restart
re-arms them (an overdue one fires at once, marked late); a threading.Timer
per reminder does the sending through the sender the bot registers at start.

kind "task" is a request the bot carries out at its time («каждое утро присылай
погоду»): the runner the bot registers feeds it in as the user's message. kind
"price" is an Ozon price watch (price_watch.py): checked at every slot, it speaks
only of a drop, and ends at its "until".
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger("assistant.reminders")

_LOCK = threading.Lock()
_PATH: Path | None = None
_SENDER = None                      # callable(chat_id: int, text: str) -> None
_RUNNER = None                      # callable(chat_id: int, request: str) -> None
_PRICE = None                       # callable(item) -> (message | None, fields to keep)
_ITEMS: dict[str, dict] = {}        # id -> {"owner", "due", "text"}
_TIMERS: dict[str, threading.Timer] = {}
MAX_PER_OWNER = 20
# Longest single Timer sleep. A reminder further out than this wakes up early
# and re-arms. On Windows a lock wait cannot exceed threading.TIMEOUT_MAX
# (~49.7 days): a Timer for "через 2 года" raised OverflowError on its own
# thread and the reminder silently never fired.
_MAX_TIMER_S = 86400.0
_TIMER = threading.Timer           # register(timer=...) swaps it for a suite


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
        ended = bool(item and item.get("until") and time.time() >= item["until"])
        # A repeating reminder re-arms for its next slot.
        if item and item.get("every") and not ended:
            nxt = dict(item)
            while nxt["due"] <= time.time():
                nxt["due"] += nxt["every"]
            _ITEMS[rid] = nxt
            _arm(rid)
        _save()
    if not item or _SENDER is None:
        return
    en = item.get("lang") == "en"
    owner = int(item["owner"])
    try:
        if item.get("kind") == "price":
            msg, keep = _PRICE(item) if _PRICE else (None, {})
            if keep:
                with _LOCK:
                    if rid in _ITEMS:
                        _ITEMS[rid].update(keep)
                        _save()
            if msg:
                _SENDER(owner, msg)
            if ended:
                _SENDER(owner, (f"Price watch ended: {item.get('name') or item['text']}" if en
                                else f"Слежение за ценой закончилось: {item.get('name') or item['text']}"))
            return
        if item.get("kind") == "task" and _RUNNER is not None:
            _SENDER(owner, ("⏰ Scheduled: " if en else "⏰ По расписанию: ") + item["text"])
            _RUNNER(owner, item["text"])
            return
        late = time.time() - item["due"] > 120
        pre = (("⏰ Reminder (late): " if late else "⏰ Reminder: ") if en
               else ("⏰ Напоминание (с опозданием): " if late else "⏰ Напоминание: "))
        _SENDER(owner, pre + item["text"])
    except Exception:
        logger.exception("reminder %s could not be sent", rid)


def _arm(rid: str) -> None:
    wait = min(_MAX_TIMER_S, max(0.0, _ITEMS[rid]["due"] - time.time()))
    t = _TIMER(wait, _fire, (rid,))
    t.daemon = True
    _TIMERS[rid] = t
    t.start()


def register(sender, path, max_timer_s: float = 86400.0, timer=None, runner=None,
             price_check=None) -> int:
    """Called once by the bot at start. Returns how many reminders were re-armed."""
    global _SENDER, _PATH, _MAX_TIMER_S, _TIMER, _RUNNER, _PRICE
    with _LOCK:
        _SENDER, _PATH, _RUNNER, _PRICE = sender, Path(path), runner, price_check
        _MAX_TIMER_S, _TIMER = max_timer_s, timer or threading.Timer
        try:
            loaded = json.loads(_PATH.read_text(encoding="utf-8")) if _PATH.exists() else {}
        except Exception:
            # Keep the damaged file: the next save would overwrite it and every
            # reminder in it would be gone for good.
            keep = _PATH.with_name(f"{_PATH.name}.unreadable-{int(time.time())}")
            try:
                _PATH.replace(keep)
            except OSError:
                keep = _PATH
            logger.exception("reminders file unreadable -- starting empty (kept as %s)", keep)
            loaded = {}
        if not isinstance(loaded, dict):
            logger.error("reminders file holds %s, not a dict -- ignored", type(loaded).__name__)
            loaded = {}
        armed = 0
        for rid, item in loaded.items():
            # One bad entry (a hand edit, a half-migrated file) raised out of
            # _arm mid-loop: the rest were never armed and the bad one stayed
            # in _ITEMS, where pending() and cancel() then failed on it.
            if not _valid(item):
                logger.error("reminder %s is malformed -- dropped: %r", rid, item)
                continue
            if rid not in _ITEMS:
                _ITEMS[rid] = item
                _arm(rid)
                armed += 1
        return armed


def _valid(item) -> bool:
    num = (int, float)
    try:
        every = item.get("every") or 0
        kind = item.get("kind") or "text"
        return (str(item["owner"]).lstrip("-").isdigit()
                and isinstance(item["due"], num) and item["due"] == item["due"]   # not NaN
                and isinstance(item.get("text"), str)
                and isinstance(every, num) and (every == 0 or every >= 60)
                and kind in ("text", "task", "price")
                and (kind != "price" or (isinstance(item.get("product"), str)
                                         and isinstance(item.get("price"), num))))
    except (AttributeError, KeyError, TypeError):
        return False


def add(owner, delay_s: float, text: str, every_s: float = 0, lang: str = "ru",
        kind: str = "text", extra: dict | None = None) -> str:
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
        if kind != "text":
            _ITEMS[rid].update(extra or {}, kind=kind)
        if not _valid(_ITEMS[rid]):
            _ITEMS.pop(rid)
            raise ValueError(f"a {kind} needs its details")
        _save()
        _arm(rid)
    return rid


def pending(owner) -> list[dict]:
    with _LOCK:
        return sorted((dict(i, id=k) for k, i in _ITEMS.items() if str(i["owner"]) == str(owner)),
                      key=lambda i: i["due"])


def get(owner, rid: str) -> dict | None:
    with _LOCK:
        item = _ITEMS.get(rid)
        return dict(item, id=rid) if item and str(item["owner"]) == str(owner) else None


def remove(owner, rid: str) -> bool:
    with _LOCK:
        item = _ITEMS.get(rid)
        if not item or str(item["owner"]) != str(owner):
            return False
        _ITEMS.pop(rid)
        t = _TIMERS.pop(rid, None)
        if t:
            t.cancel()
        _save()
        return True


def update(owner, rid: str, **fields) -> dict | None:
    """Change one of the owner's items (the 📋 Tasks buttons); a new `due` re-arms it.
    None when it is gone or not theirs; ValueError when the result would be malformed."""
    with _LOCK:
        item = _ITEMS.get(rid)
        if not item or str(item["owner"]) != str(owner):
            return None
        new = dict(item, **fields)
        for k in [k for k, v in fields.items() if v is None]:
            new.pop(k, None)
        if not _valid(new):
            raise ValueError("that change would break the item")
        _ITEMS[rid] = new
        if new["due"] != item["due"]:
            t = _TIMERS.pop(rid, None)
            if t:
                t.cancel()
            _arm(rid)
        _save()
        return dict(new, id=rid)


_ALL_WORDS = frozenset({"all", "every", "все", "всё"})
_FILLER = frozenset({"про", "мне", "мои", "напоминание", "напоминания", "напоминалку",
                     "напоминалки", "reminder", "reminders", "my", "the", "about", "me"})


def cancel(owner, words: str) -> list[str]:
    """Drop the owner's reminders whose text contains any word of `words`
    ('all' or empty = every one). Returns the cancelled texts."""
    tokens = re.findall(r"\w+", (words or "").lower())
    content = [w for w in tokens if w not in _ALL_WORDS and w not in _FILLER]
    # Only an empty request or one made of "all"/filler words cancels every
    # reminder. Short words used to be dropped first, so "cancel ТО" (the car
    # service) or "в 9" left nothing and wiped the user's whole list.
    everything = not content
    ws = [w for w in content if len(w) > 2]
    short = [w for w in content if len(w) <= 2]
    gone = []
    with _LOCK:
        for rid, item in list(_ITEMS.items()):
            if str(item["owner"]) != str(owner):
                continue
            low = item["text"].lower()
            words_in = set(re.findall(r"\w+", low))
            if (everything or any(w[:max(3, len(w) - 2)] in low for w in ws)
                    or (not ws and any(w in words_in for w in short))):
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
