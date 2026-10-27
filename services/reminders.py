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
import math
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
_FIRING: set[str] = set()
MAX_PER_OWNER = 20
# Longest single Timer sleep. A reminder further out than this wakes up early
# and re-arms. On Windows a lock wait cannot exceed threading.TIMEOUT_MAX
# (~49.7 days): a Timer for "через 2 года" raised OverflowError on its own
# thread and the reminder silently never fired.
_MAX_TIMER_S = 86400.0
_TIMER = threading.Timer           # register(timer=...) swaps it for a suite


def _save(items: dict) -> None:
    if _PATH is None:
        return
    try:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        tmp.replace(_PATH)
    except Exception as exc:
        logger.exception("could not save reminders")
        raise ValueError("could not save reminders") from exc
    _ITEMS.clear()
    _ITEMS.update(items)


def _fire(rid: str) -> None:
    with _LOCK:
        item = _ITEMS.get(rid)
        if item is None:
            _TIMERS.pop(rid, None)
            return
        if rid in _FIRING or _SENDER is None:
            _arm(rid, min_wait=60)
            return
        if item["due"] - time.time() > 1.0:
            _arm(rid)                   # a capped wait ended early: sleep on
            return
        _FIRING.add(rid)
    try:
        delivery = item.get("_delivery")
        if delivery is None:
            delivery = _prepare_delivery(item)
            with _LOCK:
                if _ITEMS.get(rid) is not item:
                    return
                prepared = dict(item, _delivery=delivery)
                items = dict(_ITEMS)
                items[rid] = prepared
                _save(items)             # no output before the pending delivery is durable
                item = prepared

        def checkpoint(**fields):
            with _LOCK:
                if _ITEMS.get(rid) is not item:
                    return False
                # Keep accepted progress in memory even if its disk checkpoint fails.
                # shortcut: crash between remote acceptance and save can duplicate; transport idempotency closes this gap.
                delivery.update(fields)
                _save(dict(_ITEMS))
                return True

        if not checkpoint():
            return
        owner = int(item["owner"])
        for index in range(delivery["sent"], len(delivery["messages"])):
            _SENDER(owner, delivery["messages"][index])
            if not checkpoint(sent=index + 1):
                return
        if delivery["task"] and not delivery["ran"]:
            if _RUNNER is None:
                raise RuntimeError("scheduled task runner is unavailable")
            _RUNNER(owner, item["text"])
            if not checkpoint(ran=True):
                return
        with _LOCK:
            if _ITEMS.get(rid) is not item:
                return
            items = dict(_ITEMS)
            items.pop(rid)
            if item.get("every") and not delivery["ended"]:
                nxt = dict(item, **delivery["keep"])
                nxt.pop("_delivery", None)
                nxt["due"] += max(1, math.floor((time.time() - nxt["due"]) / nxt["every"]) + 1) * nxt["every"]
                items[rid] = nxt
            _save(items)
            old_timer = _TIMERS.pop(rid, None)
            if old_timer:
                old_timer.cancel()
            if rid in _ITEMS:
                _arm(rid)
    except Exception:
        logger.exception("reminder %s could not be delivered; retaining it for retry", rid)
        with _LOCK:
            if _ITEMS.get(rid) is item:
                _arm(rid, min_wait=60)
    finally:
        with _LOCK:
            _FIRING.discard(rid)


def _prepare_delivery(item: dict) -> dict:
    en = item.get("lang") == "en"
    late = time.time() - item["due"] > 120
    ended = bool(item.get("until") and time.time() >= item["until"])
    keep = {}
    task = item.get("kind") == "task"
    if item.get("kind") == "price":
        if _PRICE is None:
            raise RuntimeError("price checker is unavailable")
        msg, keep = _PRICE(item)
        messages = [msg] if msg else []
        if ended:
            messages.append(f"Price watch ended: {item.get('name') or item['text']}" if en
                            else f"Слежение за ценой закончилось: {item.get('name') or item['text']}")
    elif task:
        head = (("⏰ Scheduled (late -- I was off, sorry): " if late else "⏰ Scheduled: ")
                if en else
                ("⏰ По расписанию (я был выключен — извини, вот что накопилось): " if late
                 else "⏰ По расписанию: "))
        messages = [head + item["text"]]
    else:
        pre = (("⏰ Reminder (late): " if late else "⏰ Reminder: ") if en
               else ("⏰ Напоминание (я был выключен — извини, вот что накопилось): " if late
                     else "⏰ Напоминание: "))
        messages = [pre + item["text"]]
    return {"messages": messages, "sent": 0, "task": task, "ran": False,
            "keep": keep, "ended": ended}


def _arm(rid: str, min_wait: float = 0.0) -> None:
    wait = max(min_wait, min(_MAX_TIMER_S, max(0.0, _ITEMS[rid]["due"] - time.time())))
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
        until = item.get("until") or 0
        kind = item.get("kind") or "text"
        return (str(item["owner"]).lstrip("-").isdigit()
                and isinstance(item["due"], num) and math.isfinite(item["due"])
                and isinstance(item.get("text"), str)
                and isinstance(every, num) and math.isfinite(every) and (every == 0 or every >= 60)
                and isinstance(until, num) and math.isfinite(until)
                and kind in ("text", "task", "price")
                and (kind != "price" or (isinstance(item.get("product"), str)
                                         and isinstance(item.get("price"), num))))
    except (AttributeError, KeyError, TypeError, OverflowError):
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
        item = {"owner": str(owner), "due": time.time() + delay_s, "text": text.strip()[:500]}
        if every_s:
            item["every"] = float(every_s)
        if lang == "en":
            item["lang"] = "en"
        if kind != "text":
            item.update(extra or {}, kind=kind)
        if not _valid(item):
            raise ValueError(f"a {kind} needs its details")
        items = dict(_ITEMS)
        items[rid] = item
        _save(items)
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
        items = dict(_ITEMS)
        items.pop(rid)
        _save(items)
        t = _TIMERS.pop(rid, None)
        if t:
            t.cancel()
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
        new.pop("_delivery", None)
        items = dict(_ITEMS)
        items[rid] = new
        _save(items)
        if new["due"] != item["due"] or rid in _FIRING:
            t = _TIMERS.pop(rid, None)
            if t:
                t.cancel()
            _arm(rid)
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
        removed = []
        for rid, item in list(_ITEMS.items()):
            if str(item["owner"]) != str(owner):
                continue
            low = item["text"].lower()
            words_in = set(re.findall(r"\w+", low))
            if (everything or any(w[:max(3, len(w) - 2)] in low for w in ws)
                    or (not ws and any(w in words_in for w in short))):
                removed.append(rid)
                gone.append(item["text"])
        if gone:
            _save({rid: item for rid, item in _ITEMS.items() if rid not in removed})
            for rid in removed:
                t = _TIMERS.pop(rid, None)
                if t:
                    t.cancel()
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
