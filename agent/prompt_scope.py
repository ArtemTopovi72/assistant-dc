"""Send the rules for a tool only on the turns that carry that tool.

payload_audit 2026-09-23: the system prompt is ~4800 tokens on every turn, and
~11.5k of its 14.5k characters are the [Tools] section -- most of it rules for
one tool each (the inspect-then-fix loop, video reference tags, find_photo,
transfer_image, deep_research...). On "привет" all of it went out, for tools
that were not even in the payload. A rule about a tool the model cannot call
is pure distraction.

So each "- " line of [Tools] that names specific tools is kept only when at
least one of them is in this call's tool list. Lines naming no tool (general
policy) and the safety lines (untrusted data, [TOOL ERROR], "actions happen
only through tool calls") are always kept.

PROMPT_SCOPE=0 sends the full prompt (for A/B runs).
"""
from __future__ import annotations

import os
import re

_ALWAYS_KEEP = re.compile(
    r"untrusted|DATA from the outside|^- When a tool result starts with \[TOOL ERROR\]"
    r"|Actions happen ONLY|rules come only"
    r"|Default to NO tool|Do NOT call a tool for greetings|base your answer strictly",
    re.I)


def enabled() -> bool:
    return os.getenv("PROMPT_SCOPE", "1").strip() not in ("0", "false", "no")


def _mentions(line: str, known: set) -> set:
    found = {n for n in known if "_" in n and re.search(rf"\b{re.escape(n)}\b", line)}
    if re.search(r"\bcalculate tool\b", line):
        found.add("calculate")
    if re.search(r"\bsearch tool\b|\buse search\b", line, re.I):
        found.add("search")
    return found


def scope(system: str, tool_names, known) -> str:
    """`system` with the [Tools] lines for absent tools removed."""
    if not enabled() or not system or "[Tools]" not in system:
        return system
    present, known = set(tool_names or ()), set(known or ())
    start = system.index("[Tools]")
    end = system.find("\n[", start + 1)
    end = len(system) if end < 0 else end
    kept = []
    for line in system[start:end].split("\n"):
        if line.startswith("- ") and not _ALWAYS_KEEP.search(line):
            named = _mentions(line, known)
            if named and not (named & present):
                continue
        kept.append(line)
    return system[:start] + "\n".join(kept) + system[end:]
