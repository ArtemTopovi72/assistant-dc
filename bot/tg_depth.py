"""Task-kind quota accounting and deep-research depth selection.

Split out of tg_bot.py. Every caller outside this module already reaches
these through `tg_bot.<name>` (tg_accounts, tg_library, tg_resolve, tg_tasks,
tg_commands, tg_dispatch, tg_queue all do) — that convention is what makes
this move safe: tg_bot.py re-exports every name here by import, so those call
sites need no change.

The one seam that matters: test_depth_eta_fixes.py patches
`tg_bot._depth_eta_seconds` directly to make `_fmt_eta`'s queue estimate
deterministic. `_fmt_eta` reaches it through `tg_bot._depth_eta_seconds()` at
call time (not a bare name) so that patch still lands here, the same
"cycle by design; attrs read at call time" pattern tg_accounts.py already
uses for everything it reads off tg_bot.
"""
import html as _html_mod
import logging
import os
import re

import config as _config
from tg_strings import _DEFAULT_LANG, _nav_back_row, _t

logger = logging.getLogger("assistant.tg_bot")


# ── quota accounting ──────────────────────────────────────────────────────────
# Task kinds tracked against a per-user daily budget. "task" counts everything;
# the other two are the operations that actually monopolise the GPU.
KIND_RESEARCH = "deep_research"
KIND_IMAGE    = "image"
KIND_TASK     = "task"

_IMAGE_INTENT_RE = re.compile(
    r"^\s*(?:generate an image|edit the image|"
    r"regenerate the image|outpaint|describe and analyze this image)",
    re.IGNORECASE)

# Which requests MUST end with a picture on screen. Deliberately narrower than
# _IMAGE_INTENT_RE: that one also matches "describe and analyze this image",
# which is answered in words and correctly produces no new image.
_IMAGE_PRODUCING_RE = re.compile(
    r"^\s*(?:generate an image|edit the image|"
    r"regenerate the image|change the person's outfit)",
    re.IGNORECASE)


def _classify_task(text: str) -> str:
    """Which quota bucket a resolved task falls into (before it reaches the agent)."""
    low = (text or "").lstrip().lower()
    if low.startswith("do a deep research on:"):
        return KIND_RESEARCH
    if _IMAGE_INTENT_RE.match(text or ""):
        return KIND_IMAGE
    return KIND_TASK


def _kind_label(kind: str, lang: str = _DEFAULT_LANG) -> str:
    """Human name for a quota bucket. The status page used to print the raw key
    ("deep_research" with the underscore swapped for a space), untranslated."""
    return _t({KIND_TASK: "kind_task", KIND_IMAGE: "kind_image",
               KIND_RESEARCH: "kind_research"}.get(kind, ""), lang) \
        if kind in (KIND_TASK, KIND_IMAGE, KIND_RESEARCH) else kind.replace("_", " ")


def _quota_limit(kind: str) -> int:
    try:
        import config as _cfg_mod
        return int({
            KIND_RESEARCH: getattr(_cfg_mod, "TG_QUOTA_DEEP_RESEARCH", 5),
            KIND_IMAGE:    getattr(_cfg_mod, "TG_QUOTA_IMAGE", 40),
            KIND_TASK:     getattr(_cfg_mod, "TG_QUOTA_TASKS", 200),
        }.get(kind, 0))
    except Exception:
        return 0


def _cfg_int(name: str, default: int) -> int:
    try:
        import config as _cfg_mod
        return int(getattr(_cfg_mod, name, default))
    except Exception:
        return default


def _fmt_eta(tasks: list, lang: str = _DEFAULT_LANG) -> str:
    """Turn the tasks ahead of you into a human wait estimate.

    A deep research is two orders of magnitude slower than an ordinary turn, so a
    plain task count ("2 ahead") tells the user nothing about whether that means
    a minute or half an hour.
    """
    import tg_bot as _tb  # the queue-eta seam: _depth_eta_seconds is patched on tg_bot
    per_task = max(1, _cfg_int("TG_ETA_TASK_SEC", 45))
    # A research task's real cost is measured (deep_research records every completed
    # run); an env var stays an explicit override. The old `_cfg_int(..., 0) or
    # ...` never actually reached the measured branch: config.py bakes
    # TG_ETA_RESEARCH_SEC to a nonzero default (900) whether or not the user set
    # it, so the "0 means unset" sentinel could never fire — this quoted the
    # same flat 900s forever, off by more than half on a machine where a
    # standard run takes ~35 min. Read the raw environment variable instead, so
    # "unset" and "explicitly 900" are actually distinguishable.
    _eta_override = os.environ.get("TG_ETA_RESEARCH_SEC", "").strip()
    per_res = int(_eta_override) if _eta_override.isdigit() else int(_tb._depth_eta_seconds())
    per_res = max(1, per_res)
    secs = sum(per_res if _classify_task(getattr(t, "user_text", "")) == KIND_RESEARCH
               else per_task for t in tasks)
    return _fmt_secs(secs, lang)


def _fmt_secs(secs: float, lang: str = _DEFAULT_LANG) -> str:
    """A duration a person can read: seconds under a minute and a half, then
    minutes, then hours."""
    secs = max(0, int(round(secs)))
    if secs < 90:
        return f"{secs} s" if lang != "ru" else f"{secs} с"
    mins = int(round(secs / 60.0))
    if mins < 60:
        return f"{mins} min" if lang != "ru" else f"{mins} мин"
    hours = mins / 60.0
    return f"{hours:.1f} h" if lang != "ru" else f"{hours:.1f} ч"


# ── deep-research depth ───────────────────────────────────────────────────────
# The estimate comes from deep_research, which measures every completed run. This
# module never invents a number: if the import fails the UI says "unknown" rather
# than promising a wait it cannot back up.
_DEPTHS = ("quick", "standard", "deep")


def _resolve_depth(sess) -> str:
    # `dr_depth` is user-controlled persisted state (loaded straight from a
    # JSON/DB row), so a non-string value already sitting there — corrupted
    # data, a manual edit, a future schema change — must not crash `.lower()`.
    d = getattr(sess, "dr_depth", "") or ""
    d = d.lower() if isinstance(d, str) else ""
    return d if d in _DEPTHS else _config_default_depth()


def _config_default_depth() -> str:
    d = str(getattr(_config, "DR_DEFAULT_DEPTH", "standard") or "standard").lower()
    return d if d in _DEPTHS else "standard"


def _depth_eta(depth: str = "standard") -> tuple:
    """(seconds, n_samples). n_samples 0 = a seed, not a measurement."""
    try:
        import deep_research as _dr
        return _dr.estimate_duration(depth)
    except Exception:
        logger.debug("no research timing available", exc_info=True)
        return 0.0, 0


def _depth_eta_seconds(depth: str = "standard") -> float:
    secs, _ = _depth_eta(depth)
    return secs or 900.0        # last-resort floor for the QUEUE estimate only


def _depth_name(depth: str, lang: str) -> str:
    return _t("d_" + depth, lang)


def _depth_eta_text(depth: str, lang: str) -> str:
    secs, _n = _depth_eta(depth)
    return _fmt_secs(secs, lang) if secs else "?"


def _depth_menu_kb(sess, lang: str = _DEFAULT_LANG) -> dict:
    """One row per depth, each carrying its own estimate, with the ✅ on the
    active one — the keyboard states the current setting and its cost together,
    which is the whole point of asking before a 40-minute wait."""
    cur = _resolve_depth(sess)
    rows = []
    for d in _DEPTHS:
        rows.append([{
            "text": ("✅ " if cur == d else "") + f"{_depth_name(d, lang)} · "
                    f"~{_depth_eta_text(d, lang)}",
            "callback_data": "depth:" + d,
                    }])
    rows.append(_nav_back_row(lang))
    return {"inline_keyboard": rows}


def _depth_menu_text(sess, lang: str = _DEFAULT_LANG) -> str:
    cur = _resolve_depth(sess)
    _secs, n = _depth_eta(cur)
    provenance = (_t("depth_eta_measured", lang, n=n) if n
                  else _t("depth_eta_guess", lang))
    return (_t("depth_title", lang, name=_depth_name(cur, lang),
               eta=_depth_eta_text(cur, lang))
            + "\n\n" + _html_mod.escape(_t("depth_hint", lang))
            + "\n" + _html_mod.escape(provenance))
