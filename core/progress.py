"""How far a long job is, for every front end: one call in, one line out.

A stage says WHAT runs («Generating a video»); progress says how far it is (step 14 of 20,
the model loading). Every generation reaches it through two shared places, not per tool:
ComfyUI's own step counter (comfy_client, all pictures, videos and Music3) and the progress
lines a worker in its own venv prints (music.run_gpu_worker: YuE2, Seed-VC, SoulX...). The
bot draws it under the status line, the admin panel and the desktop window beside the stage.

    progress.report(ctx, 14, 20)                    # a step counter
    progress.report(ctx, 0, 0, "Loading the model") # a phase with no count
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("assistant.progress")

STEP = "Step"

# tqdm and its kin: «45%|████▌     | 9/20 [00:12<00:15, 1.3it/s]»; the last one in a log wins
_COUNTER = re.compile(r"(\d+)/(\d+) \[")


def report(ctx, done: int, total: int, what: str = STEP) -> None:
    """Tell the context's front end how far the job is. Never raises: a progress line is
    never a reason for a render to fail."""
    fn = getattr(ctx, "set_progress", None)
    if fn is None:
        return
    try:
        fn(int(done), int(total), what or STEP)
    except Exception:
        logger.debug("progress report failed", exc_info=True)


def from_log(text: str) -> Optional[tuple]:
    """(done, total) of the last progress counter in a worker's output, or None."""
    last = None
    for m in _COUNTER.finditer(text or ""):
        done, total = int(m.group(1)), int(m.group(2))
        if 0 < total and done <= total:
            last = (done, total)
    return last


def bar(done: int, total: int, width: int = 10) -> str:
    """«▰▰▰▱▱▱▱▱▱▱ 30%», or "" when there is no count to draw."""
    if total <= 0:
        return ""
    k = max(0, min(width, round(width * done / total)))
    return "▰" * k + "▱" * (width - k) + f" {round(100 * done / total)}%"


def line(done: int, total: int, what: str = STEP, lang: str = "en") -> str:
    """The progress line under a status: «▰▰▰▱▱▱▱▱▱▱ 30% · шаг 6/20», «⏳ загружаю модель»."""
    import stages
    name = stages.translate(what or STEP, lang)
    if total <= 0:
        return f"⏳ {name}"
    return f"{bar(done, total)} · {name.lower()} {done}/{total}"
