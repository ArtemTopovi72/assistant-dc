"""What a tool handler writes back onto ctx/state after it succeeds.

Thirteen lines in their own module, and the reason is structural rather than
aesthetic. Both halves of the tool table call these — tools.py keeps six uses
of _remember and three of _set_current_image, tool_image_handlers has nine
between them. Leaving them in tools.py would have forced the handler module to
import tools, which imports the handler module: a cycle. A leaf both sides can
import is the way out of that.
"""
from typing import Optional


def _remember(ctx, state, kind: str, text: str, meta: Optional[dict] = None) -> None:
    """Log a memory entry and refresh the session-memory snapshot the GUI/graph
    read — the two calls every successful handler below makes together."""
    ctx.remember(kind, text, meta or {})
    state["session_memory_text"] = ctx.memory_text()


def _set_current_image(ctx, state, path: str, status: str = "ok") -> None:
    """Point ctx/state at the new current image after a successful edit — the
    three assignments every image handler below makes on success."""
    ctx.last_image_path = path
    state["image_path"] = path
    state["image_status"] = status
