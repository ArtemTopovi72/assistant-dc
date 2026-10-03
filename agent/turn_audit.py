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

import json
import logging
import os
import config as _cfg_env   # env_int/env_float: a bad .env value falls back, never crashes the import
import re
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

TRACE_DIR = Path(os.getenv("FAILED_TURNS_DIR",
                           Path(__file__).resolve().parents[1] / "runtime" / "failed_turns"))
# Keep the folder bounded: oldest traces go first.
MAX_TRACES = _cfg_env.env_int("FAILED_TURNS_MAX", 500)

# -- claims ----------------------------------------------------------------
# (id, pattern over the answer, tools of which ONE must have succeeded,
#  needs_sandbox, the honest line appended when none did)
_CLAIMS = (
    ("cart",
     re.compile(r"(?:добавил\w*|полож\w+|внес\w*)[^.!?\n]{0,50}(?:корзин|список покупок)"
                r"|added[^.!?\n]{0,40}(?:cart|basket)", re.I),
     {"ozon_cart", "ozon_shop"}, False,
     "Оговорка: в корзину я ничего не добавлял — инструмент корзины в этом ходе не вызывался."),
    ("remember",
     re.compile(r"\b(?:запомнил\w*|сохранил\w*\s+(?:в\s+памят|себе|это))"
                r"|\bI(?:'ll| will)? remember\b|\bsaved (?:it|that) to memory\b", re.I),
     {"remember_fact"}, False,
     "Оговорка: в долговременную память я это не записал — сохранение не выполнялось."),
    # Live 2026-09-28: "Хорошо, я записал. Одна минута до того, как вы
    # выпьете воды!" -- no tool ran, nothing was scheduled.
    ("reminder",
     re.compile(r"\bнапомню\b|напоминани\w*[^.!?\n]{0,20}(?:установлен|поставлен|создан|записан)"
                r"|(?:поставил|установил|создал)\w*[^.!?\n]{0,20}напоминани|\bI(?:'ll| will) remind\b", re.I),
     {"set_reminder"}, False,
     "Оговорка: напоминание не поставлено — инструмент напоминаний не вызывался."),
    ("reminder_cancel",
     re.compile(r"(?:напоминани|будильник)\w*[^.!?\n]{0,30}\b(?:отмен[её]н|удал[её]н|снят)\w*"
                r"|(?:отменил|удалил|снял)\w*[^.!?\n]{0,30}(?:напоминани|будильник)", re.I),
     {"set_reminder"}, False,
     "Оговорка: напоминание не отменено — инструмент напоминаний не вызывался."),
    ("wrote_down",
     re.compile(r"(?:^|[.!?]\s*|\bя\s+)(?:хорошо,?\s+|ок,?\s+)?(?:я\s+)?(?:это\s+)?записал\b(?!\s+в\s+файл)", re.I),
     {"remember_fact", "set_reminder", "write_file", "edit_file"}, False,
     "Оговорка: я ничего не записывал — ни в память, ни в напоминания."),
    ("forget",
     re.compile(r"(?:удалил\w*|стёр\w*|стер\w*)[^.!?\n]{0,30}памят", re.I),
     {"forget_facts"}, False,
     "Оговорка: из памяти я ничего не удалял — удаление не выполнялось."),
    ("location",
     re.compile(r"(?:установил\w*|сменил\w*|поставил\w*|выбрал\w*)[^.!?\n]{0,40}"
                r"пункт\w* выдачи", re.I),
     {"ozon_set_location"}, False,
     "Оговорка: пункт выдачи я не менял — это действие не выполнялось."),
    ("ran_code",
     re.compile(r"\b(?:запустил\w*|прогнал\w*|протестировал\w*)\s+(?:её|ее|его|их|код|скрипт|функци|программ|тест)"
                r"|тесты\s+(?:прошли|проходят|зел[её]н)"
                # live 2026-09-28, no sandbox at all: «и результат её работы для
                # слова «шалаш»» -- nothing ran.
                r"|результат\w*\s+(?:её|ее|его|их)\s+(?:работы|выполнения|запуска)", re.I),
     {"run_code", "run_tests"}, False,
     "Оговорка: код я в этом ходе не запускал, так что проверенным его считать нельзя."),
    ("edited_file",
     re.compile(r"(?:исправил\w*|изменил\w*|отредактировал\w*|дописал\w*|переписал\w*)"
                r"[^.!?\n]{0,60}(?:файл|\.json|\.py|\.txt|\.toml|\.ya?ml|конфиг|тег)", re.I),
     {"write_file", "edit_file", "run_code", "undo_edit", "delete_path"}, True,
     "Оговорка: файлы я в этом ходе не менял — изменений на диске нет."),
)

# Already admits it did not happen: never add a second, contradicting line.
_NEGATED = re.compile(r"не\s+(?:смог|удалось|получилось|стал|буду|добавил|запомнил|"
                      r"запускал|менял|сохранил|напомн|записывал)|\bcould(?:n't| not)\b|\bdid not\b", re.I)


def unmet_claims(answer: str, succeeded: set, has_sandbox: bool) -> list:
    """Ids of claims in `answer` that no successful tool call backs."""
    out = []
    for cid, rx, tools, needs_box, _ in _CLAIMS:
        if needs_box and not has_sandbox:
            continue
        m = rx.search(answer or "")
        if not m or (succeeded & tools):
            continue
        # The sentence around the claim must not already deny it.
        start = max(0, (answer or "").rfind(".", 0, m.start()) + 1)
        sentence = (answer or "")[start:m.end() + 40]
        if _NEGATED.search(sentence):
            continue
        out.append(cid)
    return out


def correct_claims(answer: str, succeeded: set, has_sandbox: bool) -> tuple:
    """(answer with one honest line per unmet claim appended, [claim ids])."""
    ids = unmet_claims(answer, succeeded, has_sandbox)
    if not ids:
        return answer, []
    lines = [line for cid, _, _, _, line in _CLAIMS if cid in ids]
    logger.warning("answer claims %s without the tool call that does it", ids)
    return (answer or "").rstrip() + "\n\n" + "\n".join(lines), ids


# -- complaints about the previous turn --------------------------------------
def is_complaint(text: str) -> bool:
    import intent   # the model's read; a word list took «нет, спасибо» for one
    return bool((text or "").strip()) and intent.read(None, text[:300])["complaint"]


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
    if not is_complaint(user_text):
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
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        tag = "-".join(snap.get("reasons") or ["unknown"])[:60]
        path = TRACE_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{tag}.json"
        n = 1
        while path.exists():
            path = path.with_name(f"{path.stem}_{n}.json"); n += 1
        path.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
        old = sorted(TRACE_DIR.glob("*.json"))
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
