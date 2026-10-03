"""A trace of EVERY turn, for debugging: what happened, in what order, how long
it took and what it cost.

turn_audit keeps the turns that visibly failed. The ones that hurt most did not
look failed from the inside: a guard swapped a good answer for «готового файла
у меня нет», a forwarded number forced the calculator, a voice retelling was
rejected and the list was read 1-2-3. Each was one log line among thousands,
and the answer the guard threw away was not kept anywhere. Here every turn
leaves one JSON line in runtime/turns/<date>.jsonl:

  meta      chat, task, user text (head), outcome, seconds
  phases    each stage with its start offset and duration
  llm       each model call: purpose, prompt/completion tokens, seconds, finish
  events    every assistant.* log line of INFO and above during the turn
            (guards, forced tools, rounds, tool results), with its offset
  answers   the draft and the final answer when finalization changed it
  totals    calls, tokens, tool count

Read them with scripts/turns.py. Bookkeeping never breaks a turn: every entry
point swallows its own errors.
"""
from __future__ import annotations

import base64
import contextvars
import hashlib
import json
import logging
import os
import config as _cfg_env   # env_int/env_float: a bad .env value falls back, never crashes the import
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TURNS_DIR = Path(os.getenv("TURNS_DIR", Path(__file__).resolve().parents[1] / "runtime" / "turns"))
KEEP_DAYS = _cfg_env.env_int("TURNS_KEEP_DAYS", 14)

_cur: contextvars.ContextVar = contextvars.ContextVar("turn_trace", default=None)
_write_lock = threading.Lock()


def current():
    return _cur.get()


def _now(t) -> float:
    return round(time.monotonic() - t["_t0"], 2)


class _Handler(logging.Handler):
    """Copies the turn's own log lines into its trace (same thread/context only)."""

    def emit(self, record):
        t = _cur.get()
        if t is None or record.levelno < logging.INFO:
            return
        try:
            t["events"].append({"t": _now(t), "lvl": record.levelname[0],
                                "src": record.name.rsplit(".", 1)[-1],
                                "msg": record.getMessage()[:600]})
        except Exception:
            pass


_handler = _Handler()
try:
    import log_redact                   # the trace is a file too: no bot token in it
    log_redact.install(_handler)
except Exception:
    pass
_root = logging.getLogger("assistant")
if not any(isinstance(h, _Handler) for h in _root.handlers):
    _root.addHandler(_handler)


def start(**meta):
    """Open a trace for the turn running in this context; returns the reset token."""
    t = {"_t0": time.monotonic(), "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
         "meta": dict(meta), "phases": [], "llm": [], "events": [], "answers": {},
         "_live": 0, "_closed": False}
    return _cur.set(t)


def spawn(target, *args, name: str = "", daemon: bool = True) -> threading.Thread:
    """Start a thread that carries on the current turn: it runs in a copy of
    this context (a new thread otherwise starts EMPTY, and whatever it did --
    a forwarded video's retelling, a song, a render -- was missing from the
    turn's trace and the chat's transcript), and the turn is written only once
    its last such thread has ended. Returns the started thread."""
    t = _cur.get()
    if t is not None:
        with _write_lock:
            t["_live"] += 1

    def run():
        try:
            target(*args)
        finally:
            if t is not None:
                with _write_lock:
                    t["_live"] -= 1
                    done = t["_closed"] and t["_live"] == 0
                if done:
                    _write(t)

    th = threading.Thread(target=contextvars.copy_context().run, args=(run,),
                          name=name or None, daemon=daemon)
    th.start()
    return th


class Pool(ThreadPoolExecutor):
    """A ThreadPoolExecutor whose jobs run in the submitter's context: a turn's
    parallel searches, page reads and model calls belong to that turn's trace
    and chat transcript (plain pool workers start with an empty context)."""

    def submit(self, fn, /, *args, **kwargs):
        return super().submit(contextvars.copy_context().run, fn, *args, **kwargs)


def stage(name: str) -> None:
    t = _cur.get()
    if t is None or not (name or "").strip():
        return
    now = _now(t)
    if t["phases"] and t["phases"][-1].get("s") is None:
        t["phases"][-1]["s"] = round(now - t["phases"][-1]["t"], 2)
    t["phases"].append({"t": now, "name": name, "s": None})


def llm(purpose: str, usage: dict, seconds: float, finish: str = "", chars: int = 0,
        messages=None, reply=None) -> None:
    """One model call, whole: every message sent (pictures saved to files
    under turns/media and named by path), and what came back -- text,
    reasoning and tool calls. A trace that keeps only a prompt's first words
    could not say why a projector was called a red cylinder."""
    t = _cur.get()
    if t is None:
        return
    usage = usage or {}
    rec = {"t": _now(t), "purpose": (purpose or "")[:80],
           "in": usage.get("prompt_tokens"), "out": usage.get("completion_tokens"),
           "s": round(seconds, 2), "finish": finish, "chars": chars}
    if messages:
        rec["messages"] = [{"role": m.get("role"), "content": _content(m.get("content"))}
                           for m in messages if isinstance(m, dict)]
    if reply:
        rec["reply"] = {k: reply.get(k) for k in ("content", "reasoning_content", "tool_calls")
                        if reply.get(k)}
    t["llm"].append(rec)


def _content(c):
    """Message content with every inline picture written to a file."""
    if not isinstance(c, list):
        return c
    out = []
    for part in c:
        url = ((part or {}).get("image_url") or {}).get("url", "") if isinstance(part, dict) else ""
        if url.startswith("data:") and "," in url:
            out.append({"image": _save_media(base64.b64decode(url.split(",", 1)[1]), ".jpg")})
        else:
            out.append(part)
    return out


def _save_media(data: bytes, ext: str) -> str:
    try:
        d = TURNS_DIR / "media"
        d.mkdir(parents=True, exist_ok=True)
        p = d / (hashlib.sha1(data).hexdigest()[:16] + ext)
        if not p.exists():
            p.write_bytes(data)
        return str(p)
    except Exception:
        return ""


def media_bytes(data: bytes, ext: str = "", kind: str = "") -> None:
    """Keep a copy of what the user sent (a video, a photo, a voice note) with
    the turn: the temp file is gone by the time anyone asks what went wrong."""
    t = _cur.get()
    if t is None or not data:
        return
    saved = _save_media(data, ext or ".bin")
    if saved:
        t.setdefault("media", []).append({"t": _now(t), "kind": kind, "path": saved})


def answer(kind: str, text: str) -> None:
    t = _cur.get()
    if t is not None:
        t["answers"][kind] = (text or "")[:6000]


def finish(token, *, keep_idle: bool = True, **result) -> str:
    """Close the trace. It is written now, or -- when threads spawned for this
    turn still run -- by the last of them. Returns the file path ('' if not
    written now). keep_idle=False drops a turn that never reached the model
    and logged nothing above INFO (a menu press, a message that only joined
    the queue: the queued task has its own trace)."""
    t = _cur.get()
    try:
        _cur.reset(token)
    except Exception:
        pass
    if t is None:
        return ""
    t["meta"].update(result)
    t["_keep_idle"] = keep_idle
    with _write_lock:
        t["_closed"] = True
        if t["_live"]:
            return ""
    return _write(t)


def _write(t) -> str:
    try:
        if t["phases"] and t["phases"][-1].get("s") is None:
            t["phases"][-1]["s"] = round(_now(t) - t["phases"][-1]["t"], 2)
        a = t["answers"]
        # Formatting (83.5588 -> 83,5588, markup, spacing) is not a replaced answer.
        if "draft" in a and _words(a["draft"]) == _words(a.get("final", "")):
            del a["draft"]
        t["meta"]["seconds"] = _now(t)
        t["totals"] = {
            "llm_calls": len(t["llm"]),
            "tokens_in": sum(c.get("in") or 0 for c in t["llm"]),
            "tokens_out": sum(c.get("out") or 0 for c in t["llm"]),
            "llm_seconds": round(sum(c.get("s") or 0 for c in t["llm"]), 1),
            "tools": sum(1 for e in t["events"] if e["msg"].startswith("Tool: ")),
            "warnings": sum(1 for e in t["events"] if e["lvl"] in "WEC"),
        }
        idle = not t["llm"] and not a and not t["totals"]["warnings"]
        out = {k: v for k, v in t.items() if not k.startswith("_")}
        if (idle and not t.get("_keep_idle", True)) or not _enabled():
            return ""
        TURNS_DIR.mkdir(parents=True, exist_ok=True)
        path = TURNS_DIR / f"{time.strftime('%Y-%m-%d')}.jsonl"
        with _write_lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            for old in sorted(TURNS_DIR.glob("*.jsonl"))[:-KEEP_DAYS]:
                try: old.unlink()
                except OSError: pass
            cutoff = time.time() - KEEP_DAYS * 86400
            for old in (TURNS_DIR / "media").glob("*"):
                try:
                    if old.stat().st_mtime < cutoff: old.unlink()
                except OSError: pass
        return str(path)
    except Exception:
        return ""


def _words(text: str) -> str:
    import re
    return re.sub(r"[\W_]+", "", (text or "").lower())


def _enabled() -> bool:
    """The live app only, like turn_audit: suites drive scripted turns."""
    if os.getenv("TURNS_DIR"):
        return True
    import sys
    if "pytest" in sys.modules or os.getenv("F5_TEST_RUN"):
        return False
    main = str(getattr(sys.modules.get("__main__"), "__file__", "") or "").replace("\\", "/")
    return not any(s in main for s in ("/tests/", "/bench/"))
