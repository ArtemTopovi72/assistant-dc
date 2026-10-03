"""Client abstraction for the deep-research pipeline (see research_api.py).

Every consumer that asks for a research run goes through `open_research()` here
instead of importing `deep_research` and orchestrating the run itself. The
backend is chosen by the env var RESEARCH_BACKEND and defaults to the
in-process one, so nothing about the running product changes unless it is
explicitly switched.

    RESEARCH_BACKEND=inprocess   (default) -> InProcessResearchClient
    RESEARCH_BACKEND=http                  -> HttpResearchClient
    RESEARCH_SERVICE_URL=http://127.0.0.1:8791   (http backend only)

WHY THE LAZY `import deep_research` INSIDE EVERY METHOD
-------------------------------------------------------
It is not cosmetic and it is not an import-cycle dodge. Two things depend on it:

  * `tests/test_gui_workers.py` swaps `sys.modules['deep_research']` for a stub
    defining exactly apply_overrides / restore_overrides / lang_of_text /
    run_deep_research, then calls DeepResearchWorker.run(). Resolving the module
    at CALL time keeps that seam alive now that the orchestration has moved out
    of the worker and into here.
  * `deep_research` is a heavy import (the whole crawl/extract/synthesis stack).
    A module imported at GUI startup should not drag it in.

The call SHAPES matter for the same reason and are deliberately minimal:
`lang_of_text(topic)` with one positional argument, and `run_deep_research(ctx,
topic, depth=, out_lang=, progress=)` with no other keywords — that is exactly
what the stubs accept. Do not "tidy" them.

WHAT THIS CLIENT OWNS THAT THE CALLERS USED TO
-----------------------------------------------
The apply/run/restore dance. Every caller previously open-coded it, and the GUI
worker was the only one that got the `finally:` right. Here it is one
try/finally in one place, so overrides can never leak into a later run — which
is also the only way the operation is safe to expose to a backend that might one
day run two of them.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

from research_api import (RESEARCH_API_MARKER, ResearchBackendError,  # noqa: F401
                          ResearchClient)

__all__ = ["open_research", "InProcessResearchClient", "backend_name",
           "ResearchClient", "ResearchBackendError", "RESEARCH_API_MARKER"]

logger = logging.getLogger("assistant.research")

ENV_BACKEND = "RESEARCH_BACKEND"
ENV_URL = "RESEARCH_SERVICE_URL"
DEFAULT_URL = "http://127.0.0.1:8791"


def backend_name() -> str:
    """Which backend open_research() would pick right now. Read at call time so
    a test (or the differential suite) can flip the env var between calls."""
    v = (os.environ.get(ENV_BACKEND) or "inprocess").strip().lower()
    return "http" if v in ("http", "https", "remote", "service") else "inprocess"


def open_research() -> "ResearchClient":
    """Open a research client for one caller.

    Cheap and stateless — construct one per run rather than caching it, so a
    backend switch takes effect immediately and no run shares a token with
    another.
    """
    if backend_name() == "http":
        from research_service import HttpResearchClient   # local: stdlib-only module
        return HttpResearchClient(os.environ.get(ENV_URL) or DEFAULT_URL)
    return InProcessResearchClient()


# --------------------------------------------------------------------------- #
# In-process backend
# --------------------------------------------------------------------------- #
class InProcessResearchClient:
    """The default backend: a thin, behaviour-preserving delegator to deep_research.

    It adds no policy of its own beyond owning the overrides save/restore that
    every caller used to open-code. Every other method forwards one-to-one, so
    switching a consumer from `import deep_research` to `open_research()` cannot
    change what that consumer observes — including which exceptions it sees,
    which is why nothing here is wrapped in try/except.
    """

    backend = "inprocess"
    #: The pipeline polls the CALLER's own ctx.cancel_event, so the GUI's Stop
    #: button reaches it with no round-trip and no loss. See research_api.
    cancellation_mode = "shared_event"

    # -- the run ------------------------------------------------------------ #
    def run(self, ctx, topic: str, *, depth: str = "standard",
            out_lang: Optional[str] = None,
            overrides: Optional[Dict[str, Any]] = None,
            progress: Optional[Callable[[str, dict, str], None]] = None) -> Dict[str, Any]:
        import deep_research as dr          # call-time resolution: see module docstring
        # research-service-boundary-v1 / out-lang-derived-from-typed-topic:
        # the report is a document the USER reads, so it is written in the
        # language the topic was typed in — not the pipeline's internal English
        # and not any UI toggle. Callers pass out_lang=None to get this.
        if out_lang is None:
            out_lang = dr.lang_of_text(topic)
        saved: Dict[str, Any] = {}
        try:
            # Inside the try: an invalid knob must surface to the caller as a
            # failure of the RUN (the GUI un-busies and shows it), never as an
            # exception that escapes before the finally is armed.
            saved = dr.apply_overrides(overrides) if overrides else {}
            return dr.run_deep_research(ctx, topic, depth=depth, out_lang=out_lang,
                                        progress=progress)
        finally:
            dr.restore_overrides(saved)     # per-run, never sticky

    # -- estimation / language ---------------------------------------------- #
    def estimate_duration(self, depth: str = "standard") -> Tuple[float, int]:
        import deep_research as dr
        return dr.estimate_duration(depth)

    def lang_of_text(self, text: str, default: str = "en") -> str:
        import deep_research as dr
        return dr.lang_of_text(text, default)

    # -- knobs -------------------------------------------------------------- #
    def knob_spec(self) -> Dict[str, Any]:
        import deep_research as dr
        return dict(dr.MANUAL_OVERRIDE_SPEC)

    def knob_defaults(self) -> Dict[str, Any]:
        import deep_research as dr
        g = vars(dr)
        return {name: g.get(name) for name in dr.MANUAL_OVERRIDE_SPEC}

    # -- cancellation -------------------------------------------------------- #
    def cancel(self, ctx=None) -> bool:
        """No-op that reports success: the pipeline is already polling the
        caller's own cancel_event, which the caller set before calling this.
        Kept so the two backends have one call shape."""
        return True
