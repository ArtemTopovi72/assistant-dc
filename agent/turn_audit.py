"""End-of-turn audit: claims checked against what actually ran, and a trace of
every turn that went wrong, kept for triage.

Why this exists (2026-09-23): every "the model is dumb" failure traced that day
turned out to be the harness -- a guard that never fired, a round budget spent
before delivery, a tool result that did not say what to do next. None of it
was visible from the final answer; all of it was obvious in the full trace.
So the trace of a bad turn is kept automatically, from the live chat, not only
from the bench, and `bench/triage_failures.py` groups them by reason.

The claim check generalises the guards that already exist one task at a time
(file ready, calculator, image edits): an answer that says it DID something
with side effects must have a successful call of a tool that does that thing.
"""
from __future__ import annotations
from seams import intent_read, reply_read

import json
import logging
import os
import config as _cfg_env   # env_int/env_float: a bad .env value falls back, never crashes the import
import threading
import time
from pathlib import Path

logger = logging.getLogger("assistant." + __name__)

TRACE_DIR = Path(os.getenv("FAILED_TURNS_DIR",
                           Path(__file__).resolve().parents[1] / "runtime" / "failed_turns"))


def _trace_dir() -> Path:
    return Path(os.getenv("FAILED_TURNS_DIR") or TRACE_DIR)


# Keep the folder bounded: oldest traces go first.
MAX_TRACES = _cfg_env.env_int("FAILED_TURNS_MAX", 500)

# -- claims ----------------------------------------------------------------
# What the reply says it did (reply_read.read "did") -> (tools of which ONE must have
# succeeded, needs_sandbox, the honest line appended when none did). The reading is the
# model's: «Хорошо, я записал. Одна минута до воды» and «Он записал альбом в 1999» differ
# by meaning, and a verb list could not tell them apart.
_CLAIMS = {
    "cart": ({"ozon_cart", "ozon_shop"}, False,
             "Оговорка: в корзину я ничего не добавлял — инструмент корзины в этом ходе не вызывался."),
    "remember": ({"remember_fact"}, False,
                 "Оговорка: в долговременную память я это не записал — сохранение не выполнялось."),
    "reminder": ({"set_reminder"}, False,
                 "Оговорка: напоминание не поставлено — инструмент напоминаний не вызывался."),
    "reminder_cancel": ({"set_reminder"}, False,
                        "Оговорка: напоминание не отменено — инструмент напоминаний не вызывался."),
    "forget": ({"forget_facts"}, False,
               "Оговорка: из памяти я ничего не удалял — удаление не выполнялось."),
    "pickup_point": ({"ozon_set_location"}, False,
                     "Оговорка: пункт выдачи я не менял — это действие не выполнялось."),
    "ran_code": ({"run_code", "run_tests"}, False,
                 "Оговорка: код я в этом ходе не запускал, так что проверенным его считать нельзя."),
    "edited_file": ({"write_file", "edit_file", "run_code", "undo_edit", "delete_path"}, True,
                    "Оговорка: файлы я в этом ходе не менял — изменений на диске нет."),
}


def unmet_claims(answer: str, succeeded: set, has_sandbox: bool, ctx=None, user: str = "") -> list:
    """Kinds of side effect `answer` claims that no successful tool call backs."""
    if not (answer or "").strip():
        return []
    did = reply_read(ctx)(ctx, answer, user)["did"]
    return [k for k in did if k in _CLAIMS and not (succeeded & _CLAIMS[k][0])
            and (has_sandbox or not _CLAIMS[k][1])]


def correct_claims(answer: str, succeeded: set, has_sandbox: bool, ctx=None, user: str = "") -> tuple:
    """(answer with one honest line per unmet claim appended, [claim kinds])."""
    ids = unmet_claims(answer, succeeded, has_sandbox, ctx, user)
    if not ids:
        return answer, []
    logger.warning("answer claims %s without the tool call that does it", ids)
    return (answer or "").rstrip() + "\n\n" + "\n".join(_CLAIMS[k][2] for k in ids), ids


# -- complaints about the previous turn --------------------------------------
def is_complaint(text: str, ctx=None) -> bool:
    return bool((text or "").strip()) and intent_read(ctx)(None, text[:300])["complaint"]


# The previous turn of each conversation, so a complaint can file it.
_LAST: dict = {}
_LOCK = threading.Lock()


def _owner(ctx) -> str:
    return str(getattr(ctx, "ozon_owner", "") or "desktop")


def remember_turn(ctx, snapshot: dict) -> None:
    with _LOCK:
        _LAST[_owner(ctx)] = snapshot


def file_complaint(ctx, user_text: str) -> str:
    """If `user_text` complains, save the PREVIOUS turn as failed. Returns the path."""
    if not is_complaint(user_text, ctx=ctx):
        return ""
    with _LOCK:
        prev = _LAST.pop(_owner(ctx), None)
    if not prev:
        return ""
    prev = dict(prev)
    prev["reasons"] = sorted(set(prev.get("reasons") or []) | {"user_complaint"})
    prev["complaint"] = (user_text or "")[:500]
    return _write(prev)


# -- traces -------------------------------------------------------------------
def _slim_messages(messages: list, keep: int = 60) -> list:
    out = []
    for m in (messages or [])[-keep:]:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, list):  # multimodal: keep the text, drop the pixels
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        e = {"role": m.get("role"), "content": str(c or "")[:4000]}
        if m.get("tool_calls"):
            e["tool_calls"] = [
                {"name": (t.get("function") or {}).get("name"),
                 "args": str((t.get("function") or {}).get("arguments"))[:1500]}
                for t in m["tool_calls"] if isinstance(t, dict)]
        out.append(e)
    return out


def snapshot(ctx, state, messages, user_input, final_answer, reasons, calls) -> dict:
    return {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "owner": _owner(ctx),
        "model": getattr(ctx, "model_name", ""),
        "thinking": not getattr(ctx, "no_think", True),
        "user_input": (user_input or "")[:2000],
        "final_answer": (final_answer or "")[:4000],
        "reasons": sorted(set(reasons or [])),
        "calls": list(calls or []),
        "document_path": str((state or {}).get("document_path") or ""),
        "messages": _slim_messages(messages),
    }


def _enabled() -> bool:
    """Only the live app files traces. Suites and benches drive the same loop
    with scripted models -- their "failures" are the point of the test and
    would bury the real ones (the live tg_users.db was polluted exactly so).
    FAILED_TURNS_DIR set explicitly always wins."""
    if os.getenv("FAILED_TURNS_DIR"):
        return True
    import sys
    if "pytest" in sys.modules:
        return False
    main = str(getattr(sys.modules.get("__main__"), "__file__", "") or "").replace("\\", "/")
    return not any(s in main for s in ("/tests/", "/bench/", "/scripts/test"))


def _write(snap: dict) -> str:
    if not _enabled():
        return ""
    try:
        _trace_dir().mkdir(parents=True, exist_ok=True)
        tag = "-".join(snap.get("reasons") or ["unknown"])[:60]
        path = _trace_dir() / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{tag}.json"
        n = 1
        while path.exists():
            path = path.with_name(f"{path.stem}_{n}.json"); n += 1
        path.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
        old = sorted(_trace_dir().glob("*.json"))
        for f in old[:max(0, len(old) - MAX_TRACES)]:
            try: f.unlink()
            except OSError: pass
        logger.info("failed turn saved for triage: %s", path.name)
        return str(path)
    except Exception as exc:  # never let bookkeeping break a turn
        logger.warning("could not save failed-turn trace: %s", exc)
        return ""


def finish_turn(ctx, state, messages, user_input, final_answer, reasons, calls) -> str:
    """Remember the turn for a later complaint; save it now if it already failed."""
    snap = snapshot(ctx, state, messages, user_input, final_answer, reasons, calls)
    remember_turn(ctx, snap)
    return _write(snap) if snap["reasons"] else ""
