"""Who may use a sandbox, and how far.

Four levels, because "can use the sandbox" is not one decision. Handing someone
a place to keep files is nothing like handing them the ability to execute code
on your machine, and the gap between "executes in a container" and "executes on
the host" is larger still — so each is a separate, revocable grant rather than
one switch an administrator flips while thinking about the first one.

    off    nothing. The default for everybody, including new users.
    files  keep, list, read, edit, unpack and pack files. No execution.
    code   the above, plus running code INSIDE the container.
    host   the above, but execution falls back to the host when the container
           engine is down. Unisolated. Only ever a deliberate choice about a
           person the machine's owner trusts with the machine.

Stored in the user row's `prefs`, which exists precisely so a new per-user
option needs no schema migration.

Default-deny is the whole design: an unknown level, a missing user, a corrupted
value and a store that raises all resolve to `off`. There is no path through
this module that turns "I could not tell" into permission.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("assistant.sandbox_access")

OFF = "off"
FILES = "files"
CODE = "code"
HOST = "host"

# Ordered weakest to strongest; membership and comparison both read off this.
LEVELS = (OFF, FILES, CODE, HOST)
_RANK = {name: i for i, name in enumerate(LEVELS)}

PREFS_KEY = "sandbox"

LABELS = {
    OFF:   "off — no sandbox",
    FILES: "files — keep and edit files, no execution",
    CODE:  "code — plus running code in the container",
    HOST:  "host — plus running code UNISOLATED if the container is unavailable",
}


def normalize(value) -> str:
    """Any stored value -> a level this module recognises, or `off`.

    Anything unrecognised is a denial, never a default-to-something-useful: a
    typo in a hand-edited row must not silently grant execution.
    """
    v = str(value or "").strip().lower()
    return v if v in _RANK else OFF


def level_for(user) -> str:
    """The level of a user row (or anything with `.prefs` / a plain dict)."""
    if user is None:
        return OFF
    try:
        prefs = user.get("prefs") if isinstance(user, dict) else getattr(user, "prefs", None)
        if isinstance(prefs, dict):
            return normalize(prefs.get(PREFS_KEY))
    except Exception:                      # a broken row is not an authorisation
        logger.debug("unreadable prefs; denying sandbox", exc_info=True)
    return OFF


def at_least(user, wanted: str) -> bool:
    return _RANK[level_for(user)] >= _RANK[normalize(wanted)]


def may_use_files(user) -> bool:
    return at_least(user, FILES)


def may_run_code(user) -> bool:
    return at_least(user, CODE)


def allow_host_execution(user) -> bool:
    """Only the top level, and only ever because someone chose it explicitly."""
    return level_for(user) == HOST


def set_level(store, chat_id: int, level: str) -> str:
    """Grant or revoke. Returns the level actually stored.

    Writes through the store so the change is durable and visible to the bot
    immediately; there is no in-process cache to go stale.
    """
    level = normalize(level)
    user = store.get(chat_id)
    if user is None:
        raise KeyError(f"no such user: {chat_id}")
    prefs = dict(getattr(user, "prefs", None) or {})
    prefs[PREFS_KEY] = level
    user.prefs = prefs
    store.put(user)
    logger.info("sandbox level for %s set to %s", chat_id, level)
    return level


def describe(user) -> str:
    return LABELS.get(level_for(user), LABELS[OFF])
