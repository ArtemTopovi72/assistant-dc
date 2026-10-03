"""Eyes for run_code: a picture a script drew is LOOKED AT before it is sent.

Live 2026-09-18: asked for a flowchart, the model lost graphviz (no `dot`
in the sandbox then), hand-placed boxes in matplotlib and sent a column of
blocks with one arrow drawn straight through every box and its text. The
script ran without error, so nothing in the loop could tell that the result
was garbage -- the only check was "did it write a file".

Now the newest picture a script wrote is shown to the vision model together
with the request; concrete layout defects (lines through boxes or text,
overlapping or unreadable labels, an empty canvas, cut-off content) come
back into the tool result as a fix request, once per turn. The judge is told
to answer OK unless a defect is plainly visible: the draw critic's habit of
inventing problems is known, so only the literal, geometric kind counts.
"""
from __future__ import annotations
import logging

logger = logging.getLogger("assistant.tools")

SYSTEM = (
    "You check a picture that a Python script just drew for the user's request. "
    "Look ONLY for plainly visible technical defects: an arrow or line drawn "
    "THROUGH a box or through text; labels overlapping each other; text cut off "
    "at the edge or unreadable (too small, garbled, boxes/tofu instead of "
    "letters); an empty or nearly empty canvas; elements stacked on top of each "
    "other; a diagram whose arrows do not connect the shapes they should. Do NOT "
    "judge style, colours, or whether you would have designed it differently. "
    "If no such defect is visible, answer exactly: OK. Otherwise list the "
    "defects, one short line each, at most four lines, plain text.")


def critique(ctx, image_path: str, request: str) -> str:
    """'' when the picture passes (or the check cannot run), else the defects."""
    try:
        import llm as _llm
        got = _llm.analyze_image_with_llm(
            ctx, image_path=image_path,
            user_text=f"The user's request was: {(request or '').strip()[:400] or '(unknown)'}",
            system_prompt=SYSTEM, temperature=0.1, max_tokens=250) or ""
    except Exception:
        logger.warning("visual check: could not look at %s", image_path, exc_info=True)
        return ""
    got = got.strip()
    if not got or got.upper().startswith("OK") or got.upper().rstrip(".") == "OK":
        return ""
    lines = [ln.strip(" -•*") for ln in got.splitlines() if ln.strip(" -•*")]
    return "\n".join(lines[:4])


def note_for(defects: str) -> str:
    return ("\n[VISUAL CHECK of that picture found defects:\n" + defects +
            "\nFix the script with edit_file and run it again before answering; "
            "for a flowchart or any graph use graphviz (the dot binary is "
            "installed), never hand-placed boxes.]")
