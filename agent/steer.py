"""Steering: what the user says WHILE the bot works changes the work.

Live 2026-09-18: «я прошу график, вижу, что он пишет код, и вкидываю "а фон
пусть будет красный" -- и он просто на ближайшем свободном моменте: ага,
учту». Before this a message sent during a task waited in the queue and was
answered as a separate request once the work was already done (or, in a
slow phase, ran as a parallel task that knew nothing of the first).

Now a plain text that arrives while THIS chat's task is still running goes
into the running task's inbox: the user gets a one-line «👌 Учту: …» at once,
and the note is folded into the agent loop at its next round -- appended to
the last tool result (template-safe: no user turn is inserted between an
assistant tool call and its result), so the model applies it to the rest of
the work. A note the task never got to read (it ended first) is re-queued as
an ordinary request, so nothing is lost.
"""
from __future__ import annotations
import logging
import re
import threading
from typing import List, Optional

logger = logging.getLogger("assistant.steer")

MAX_NOTE = 400


class Inbox:
    """Per-task, thread-safe: the bot thread appends, the agent loop drains."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._notes: List[str] = []
        self.consumed: List[str] = []

    def put(self, text: str) -> None:
        with self._lock:
            self._notes.append(text)

    def drain(self) -> List[str]:
        with self._lock:
            out, self._notes = self._notes, []
        self.consumed.extend(out)
        return out

    def unread(self) -> List[str]:
        with self._lock:
            return list(self._notes)


# A question during a slow, backgroundable phase is still answered on the
# side (the old parallel-task path) -- «а сколько времени?» during a ten-
# minute research must not wait for the research. Everything else steers.


def eligible(text: str, has_file: bool, interruptible: bool) -> bool:
    """Should this arrival steer the running task instead of queueing?"""
    t = (text or "").strip()
    if not t or has_file or len(t) > MAX_NOTE:
        return False
    if t.startswith("/"):
        return False
    if interruptible and __import__("intent").read(None, t)["is_question"]:
        return False
    return True


def ack(text: str, lang: str) -> str:
    # Only the user's own words: menu prefixes («find on ozon: ») and machine
    # frames («[the user forwarded this message…]») were echoed as is (live 16:36).
    from prompt_guard import user_words
    t = re.sub(r"\[[^\]]*\]", " ", user_words(text), flags=re.S)
    t = re.sub(r"^\s*[a-z][a-z ]*(?:\([^)]*\))?:\s*", "", t)
    t = " ".join(t.split()) or "…"
    if len(t) > 80:
        t = t[:77].rstrip() + "…"
    return (f"👌 Учту: «{t}»" if (lang or "ru").startswith("ru") else f"👌 Noted: “{t}”")


def fold_into(messages: list, notes: List[str]) -> bool:
    """Put the notes where the model reads them next: appended to the last
    tool result, or to the last user message when no tool has run yet."""
    if not notes or not messages:
        return False
    block = ("\n\n[UPDATE FROM THE USER while you were working -- apply it to "
             "the rest of this task; if it is unrelated, answer it briefly in "
             "your reply]:\n" + "\n".join(f"- {n}" for n in notes))
    last = messages[-1]
    if last.get("role") in ("tool", "user"):
        last["content"] = (last.get("content") or "") + block
        return True
    # An assistant message last (no tools this round): add a user turn.
    messages.append({"role": "user", "content": block.strip()})
    return True


def take(ctx) -> List[str]:
    """For a pipeline that is not the agent loop (a song): its own checkpoint
    before the expensive, irreversible step. Notes read here are consumed;
    ones that arrive after the last checkpoint are re-queued by the bot."""
    inbox: Optional[Inbox] = getattr(ctx, "steer_inbox", None)
    return inbox.drain() if inbox is not None else []


def with_notes(request: str, notes: List[str]) -> str:
    """The request as the user now means it: later wishes win over earlier."""
    if not notes:
        return request
    return (request + "\n\nThe user added while it was being made (these win "
            "where they conflict with the above): " + "; ".join(notes))


def drain_into(ctx, messages: list) -> int:
    """Called by the agent loop before each model call."""
    inbox: Optional[Inbox] = getattr(ctx, "steer_inbox", None)
    if inbox is None:
        return 0
    notes = inbox.drain()
    if notes:
        fold_into(messages, notes)
        logger.info("steer: %d note(s) folded into the round: %r", len(notes), notes[0][:80])
    return len(notes)
