"""A word to the user BEFORE a long tool runs.

A drawing, a song, a research run or a deck takes minutes, and until now the
chat showed only the status line while the model's own «Сейчас нарисую кота в
шляпе» (written on the same round as the tool call) was thrown away. The user
asked for a reaction in real time, not only at the end (2026-09-18: «было бы
круто чтобы он как-то мог реагировать в реал тайме даже до того как завершит
задачу»).

The graph offers the model's pre-tool sentence when it is clean and short,
otherwise a one-line note naming the work; the bot sends it as an ordinary
message. One per turn, and only for the tools that actually take a while.
"""
from __future__ import annotations
import logging
import re
from typing import Optional

logger = logging.getLogger("assistant.interim")

# tool -> (icon, ru, en). The wording is a promise of WORK, not of a result.
_LONG_TOOLS = {
    # generate_image/redraw_image can silently loop: inspect_image checks the
    # result against the instructions and, on a miss, the graph retries with
    # another render before ever answering (tool_image_handlers.py's "Next
    # step: call inspect_image..." contract). Live, 2026-09-19: the user
    # called out "about a minute" as dishonest after a redraw that needed two
    # corrective rounds actually took ~4 minutes end to end -- "here it should
    # honestly say about a few minutes".
    "generate_image":      ("🎨", "рисую — это займёт пару минут", "drawing — a couple of minutes"),
    "redraw_image":        ("🎨", "перерисовываю — это пара минут, иногда дольше", "redrawing — a couple of minutes, sometimes longer"),
    "inpaint_image":       ("🖌", "правлю картинку — около минуты", "editing the picture — about a minute"),
    "transfer_image":      ("🖌", "переношу на картинку — около минуты", "transferring onto the picture — about a minute"),
    "upscale_image":       ("🔍", "увеличиваю картинку — с полминуты", "upscaling — half a minute or so"),
    "fix_hands":           ("🖌", "правлю руки — около минуты", "fixing the hands — about a minute"),
    "find_photo":          ("🖼", "ищу фото — минуту", "looking for the photo — a minute"),
    "find_content":        ("🖼", "просматриваю снимки — несколько минут", "going through the photos — a few minutes"),
    "generate_video":      ("🎬", "делаю видео — это несколько минут", "making the video — a few minutes"),
    "generate_music":      ("🎵", "сочиняю и пою — несколько минут", "writing and singing — a few minutes"),
    "deep_research":       ("🔬", "изучаю вопрос — это займёт несколько минут", "researching — a few minutes"),
    "create_presentation": ("📊", "собираю презентацию — пару минут", "building the deck — a couple of minutes"),
    "run_code":            ("💻", "запускаю код", "running the code"),
    "dedupe_photos":       ("🖼", "убираю дубли — минуту", "removing duplicates — a minute"),
}
_ACKS = {"ru": "Понял,", "en": "Got it,"}
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


def compose(tool_name: str, content: str, lang: str) -> str:
    """The message to send before `tool_name` runs."""
    icon, ru, en = _LONG_TOOLS[tool_name]
    own = clean_note(content)
    if own:
        return f"{icon} {own}"
    lang = "ru" if (lang or "ru").lower().startswith("ru") else "en"
    return f"{icon} {_ACKS[lang]} {ru if lang == 'ru' else en}."


def offer(ctx, content: str, tool_calls, sent_so_far: int) -> bool:
    """Send one interim note through ctx.interim_callback. True when sent."""
    cb = getattr(ctx, "interim_callback", None)
    if cb is None or sent_so_far >= MAX_INTERIMS_PER_TURN:
        return False
    name = long_tool_in(tool_calls)
    if not name:
        return False
    text = compose(name, content, getattr(ctx, "reply_lang", "ru"))
    try:
        cb(text)
    except Exception:
        logger.warning("interim note failed", exc_info=True)
        return False
    logger.info("Interim note before %s: %s", name, text[:120])
    return True
