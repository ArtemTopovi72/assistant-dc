"""One line after a tool result saying what to do next.

Without thinking, the model does what the LAST tool result suggests. Measured
2026-09-23 on the delivery bench: a bare "Wrote X." after each identical write
made it write again, five times, until the budget ran out; "Nothing changed --
pack it now" ended the loop at once. So results that leave the next step open
get it spelled out here, in one place, instead of in every handler.

A result that already carries its own guidance ([NOTE], [TOOL ERROR], "Next:",
"NOW") is left alone -- two instructions that disagree are worse than none.
"""
from __future__ import annotations

import difflib
import re

_HAS_GUIDANCE = re.compile(r"\[NOTE\]|\[TOOL ERROR\]|\bNext:|\bNOW\b|Nothing changed", re.I)
_UNPACKED = re.compile(r"^(.*?[^/]*_unpacked)(?:/|$)")


def _unpacked_root(path: str) -> str:
    m = _UNPACKED.match(str(path or "").replace("\\", "/").lstrip("./"))
    return m.group(1) if m else ""


def next_step(name: str, args: dict, result: str) -> str:
    """The hint to append to a successful `result`, or ""."""
    if not isinstance(result, str) or _HAS_GUIDANCE.search(result):
        return ""
    args = args or {}
    path = str(args.get("path") or "")
    if name in ("write_file", "edit_file", "delete_path", "undo_edit"):
        root = _unpacked_root(path)
        if root:
            return (f"\nNext: once every change is made, pack_archive '{root}' -- until "
                    f"then the user has nothing to install. Do not repeat this edit.")
        if path.lower().endswith(".py") and name != "delete_path":
            return "\nNext: run it (run_code or run_tests) before saying it works."
    if name in ("rag_search", "search_files", "find_content") and re.search(
            r"^(?:No |Nothing )", result.strip()):
        return ("\nNext: try once with different words; if still nothing, tell the user "
                "plainly it was not found -- do not answer from memory as if it was.")
    if name == "ozon_reviews" and "no reviews" in result.lower():
        return "\nNext: tell the user plainly this product has no reviews; do not invent any."
    return ""


def unknown_tool(name: str, known) -> str:
    close = difflib.get_close_matches(str(name or ""), list(known), n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    return f"[TOOL ERROR] Unknown tool: {name}.{hint} Use only the tools you were given."
