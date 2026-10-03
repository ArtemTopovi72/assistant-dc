"""A word to the user BEFORE a long tool runs.

A drawing, a song, a research run or a deck takes minutes, and until now the
chat showed only the status line while the model's own «Сейчас нарисую кота в
шляпе» (written on the same round as the tool call) was thrown away. The user
asked for a reaction in real time, not only at the end (2026-09-18: «было бы
круто чтобы он как-то мог реагировать в реал тайме даже до того как завершит
задачу»).

The graph offers the model's pre-tool sentence when it is clean and short
(nothing otherwise: the status line already names the work); the bot sends
it as an ordinary message. One per turn, and only for the tools that actually take a while.
"""
from __future__ import annotations
import logging
import re
from typing import Optional

logger = logging.getLogger("assistant.interim")

# tool -> icon. Only tools that take a while get a note.
# A canned «Понял, перерисовываю — это пара минут» used to go out when the
# model said nothing of its own; it repeated the status line right under it
# («🎨 Перерисовываю картинку…» with its Cancel button) word for word (live
# 10-03: «лишне дублирует»). Only the model's own sentence is sent now.
_LONG_TOOLS = {
    "generate_image": "🎨", "redraw_image": "🎨", "inpaint_image": "🖌",
    "transfer_image": "🖌", "upscale_image": "🔍", "fix_hands": "🖌",
    "find_photo": "🖼", "find_content": "🖼", "generate_video": "🎬",
    "generate_music": "🎵", "deep_research": "🔬", "create_presentation": "📊",
    "run_code": "💻", "dedupe_photos": "🖼",
}
MAX_INTERIMS_PER_TURN = 1

# What the model wrote is only worth sending when it reads as a sentence to
# the user: no control tokens, no tool-call leaks, no half-written JSON.
_UNCLEAN_RE = re.compile(r"[<>{}\[\]]|```|tool_call|function|<\|", re.I)


def long_tool_in(tool_calls) -> Optional[str]:
    """The first long tool named by a round's calls, or None."""
    for tc in tool_calls or []:
        try:
            name = tc["function"]["name"]
        except (KeyError, TypeError):
            continue
        if name in _LONG_TOOLS:
            return name
    return None


def clean_note(content: str) -> str:
    """The model's own pre-tool sentence, or '' when it is not fit to send."""
    t = " ".join((content or "").split())
    if not t or len(t) > 220 or _UNCLEAN_RE.search(t):
        return ""
    # a first sentence only -- a plan of five steps is not a reaction
    parts = re.split(r"(?<=[.!…])\s+", t)
    t = parts[0].strip()
    if len(t) < 6 or t.endswith(":"):
        return ""
    return t


def compose(tool_name: str, content: str) -> str:
    """The message to send before `tool_name` runs; '' when the model wrote
    nothing worth sending (the status line already names the work)."""
    own = clean_note(content)
    return f"{_LONG_TOOLS[tool_name]} {own}" if own else ""


def offer(ctx, content: str, tool_calls, sent_so_far: int) -> bool:
    """Send one interim note through ctx.interim_callback. True when sent."""
    cb = getattr(ctx, "interim_callback", None)
    if cb is None or sent_so_far >= MAX_INTERIMS_PER_TURN:
        return False
    name = long_tool_in(tool_calls)
    if not name:
        return False
    text = compose(name, content)
    if not text:
        return False
    try:
        cb(text)
    except Exception:
        logger.warning("interim note failed", exc_info=True)
        return False
    logger.info("Interim note before %s: %s", name, text[:120])
    return True
