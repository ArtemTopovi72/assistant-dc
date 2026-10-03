"""The deep-research service boundary.

This module defines — and ONLY defines — the contract between the front ends
that ask for a research run and the pipeline that performs one. It holds no
state, imports nothing from `deep_research` (or PyQt5, or the model stack), and
so is safe to import from anywhere, including a standalone service process.

It is the same shape as `knowledge_api.py`: a Protocol derived from real call
sites, an in-process client as the DEFAULT (`research_client.py`), and an
out-of-process backend only where the analysis says cancellation and progress
genuinely survive the hop (see docs/gui_service_boundaries.md).

WHY A BOUNDARY HERE
-------------------
Deep research is the one heavy GUI operation whose inputs are all scalars, whose
output is already a JSON-shaped dict persisted to disk, and whose progress
callback carries only JSON-serialisable values. It does not need the
GPU-resident Whisper/F5 models; it talks to LM Studio, which is already a
separate process. Four independent front ends consume it:

    gui_workers.DeepResearchWorker.run   the desktop Research tab
    tools._handle_deep_research          the agent's deep_research tool
    tg_tasks                             the Telegram bot's direct bypass
    tg_bot._depth_eta                    the ETA shown before a run starts

THE CALLS THAT ACTUALLY CROSS THE BOUNDARY
------------------------------------------
Derived by grepping the real call sites, not from what looked tidy:

  run one research pass
    gui_workers.DeepResearchWorker.run
        dr.apply_overrides(overrides)
        dr.run_deep_research(ctx, topic, depth=, out_lang=dr.lang_of_text(topic),
                             progress=lambda phase, stats, msg: ...)
        dr.restore_overrides(saved)          # in a finally: never sticky
    tools._handle_deep_research
        run_deep_research(ctx, topic, depth=, out_lang=, progress=_cb)
    tg_tasks
        _dr.run_deep_research(ctx, topic, out_lang=, depth=, progress=)

  estimate how long a run will take
    tg_bot._depth_eta                    _dr.estimate_duration(depth)

  language of the document
    every caller                         lang_of_text(topic)

  the manual-control knobs
    gui_manual_panel._build_manual_control_panel   deep_research.MANUAL_OVERRIDE_SPEC
    gui_manual_panel._reset_manual_control         the current DR_* module globals

NOTE THAT apply_overrides / restore_overrides ARE **NOT** ON THE BOUNDARY
------------------------------------------------------------------------
They are not a service operation; they are the pipeline mutating its OWN module
globals in place for the duration of one run (deep_research.py:228-250, and see
dr_timing.py:52 / dr_outline.py:12, which read those globals). Exposing them
across a boundary would export a race: two concurrent runs in one server process
would stamp on each other's knobs. The contract therefore takes `overrides` as
an ARGUMENT TO run(), and the backend owns the save/apply/restore as an
indivisible part of executing that one run. In-process this is exactly what the
GUI worker did by hand; out-of-process it is what makes concurrency safe at all.

CANCELLATION — READ THIS BEFORE ADDING A REMOTE BACKEND
--------------------------------------------------------
`deep_research` polls `ctx.is_cancelled()` — i.e. `ctx.cancel_event`, a
`threading.Event` — at six points (deep_research.py:749, 765, 1025, 1524, 1811,
2377). A `threading.Event` is PROCESS-LOCAL. The GUI's Stop button
(gui.AssistantWindow._cancel_current, gui.py:2255) sets that event; the
Madhouse/transfer/storyboard tabs hold a private one via
`gui_common._ScopedCtx`.

So: an in-process backend preserves cancellation exactly, because it passes the
same ctx straight through. A REMOTE backend does not get it for free. It needs a
run id and an explicit cancel route, with the server holding its own Event that
the pipeline polls, and the client polling the local ctx and firing that route.
`CANCELLATION_MODES` below records which mode a backend implements, and
`ResearchClient.cancellation_mode` must report it honestly. A backend that
cannot cancel must say so rather than leave a user on a Stop button that does
nothing.

PROGRESS
--------
`progress(phase: str, stats: dict, msg: str)` — the only progress callback in
this codebase whose every argument is already JSON-serialisable, which is why
this boundary is viable at all. `phase` is one of PHASES. A callable cannot
cross a process boundary, so a remote backend must stream one record per tick
(NDJSON) and the CLIENT re-invokes the local callable; it never calls back over
the wire. Anything less collapses the GUI's phase-by-phase progress strip
(gui._on_research_progress) into "started" and "done".

RETURN SHAPE
------------
run() returns the dict `deep_research.run_deep_research` returns, unchanged:
RESULT_FIELDS below. `report` is markdown; `path` is a path ON THE MACHINE THAT
RAN IT — a remote backend on another host returns a path the client cannot open.
Same-host only; this is a constraint to write down, not to discover in
production.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol, Tuple, runtime_checkable

# Marker grepped by the contract suites INSTEAD of a function name, so that
# moving any single method (or the derivation site) does not break the anchor.
RESEARCH_API_MARKER = "research-service-boundary-v1"

#: Placed in a comment at the one site that derives the document language from
#: the topic the user actually typed. test_dr_output_language anchors on this
#: rather than on a literal call expression, which moves when code moves.
OUT_LANG_FROM_TOPIC_MARKER = "out-lang-derived-from-typed-topic"

#: Phase labels the pipeline emits, in order. Mirrors deep_research.PHASES; the
#: GUI's progress strip indexes its stepper by position in this tuple.
PHASES = ("Expanding queries", "Searching", "Crawling", "Extracting",
          "Deduplicating", "Verifying", "Building report", "Complete")

#: Depths a caller may ask for. Mirrors dr_timing.DEPTHS.
DEPTHS = ("quick", "standard", "deep")

#: Keys of the dict run() returns.
RESULT_FIELDS = ("report", "path", "stats", "cancelled")

#: How faithfully a backend can stop a run in flight.
#:   "shared_event"  the backend polls the caller's own ctx.cancel_event —
#:                   Stop is instantaneous and exact (in-process).
#:   "run_id"        the backend holds its own token, reached by a cancel route
#:                   keyed on a run id; Stop costs one round-trip.
#:   "none"          the run cannot be stopped. A client reporting this MUST be
#:                   refused by any UI that offers a Stop button.
CANCELLATION_MODES = ("shared_event", "run_id", "none")

#: Operation names a remote backend exposes. One route per boundary method.
OPERATIONS = ("run", "estimate_duration", "lang_of_text", "knob_spec",
              "knob_defaults", "cancel")


@runtime_checkable
class ResearchClient(Protocol):
    """What a deep-research backend must provide, in-process or over the wire."""

    #: One of CANCELLATION_MODES. Reported honestly; see the module docstring.
    cancellation_mode: str

    def run(self, ctx, topic: str, *, depth: str = "standard",
            out_lang: Optional[str] = None,
            overrides: Optional[Dict[str, Any]] = None,
            progress: Optional[Callable[[str, dict, str], None]] = None) -> Dict[str, Any]:
        """Run one research pass and return RESULT_FIELDS.

        ctx        the assistant context. In-process it carries the cancel event
                   the pipeline polls; a remote backend uses it ONLY to poll
                   cancellation locally, never sends it.
        out_lang   None means "derive it from `topic`" — the document is written
                   in the language the user typed in, which is what every caller
                   wants and what every caller previously open-coded.
        overrides  DR_* knobs for THIS RUN ONLY. Applied and restored by the
                   backend as an indivisible part of the run, including on
                   failure. An unknown key or an out-of-range value raises, and
                   nothing is left half-applied.
        progress   called as progress(phase, stats, msg) with phase in PHASES.
                   Always invoked on the CALLER's side.
        """

    def estimate_duration(self, depth: str = "standard") -> Tuple[float, int]:
        """(seconds, n_samples) for one run at this depth. n_samples == 0 means
        the number is a seed, not a measurement, and the UI should say so."""

    def lang_of_text(self, text: str, default: str = "en") -> str:
        """Language code for a topic or user turn. Script heuristic, no I/O."""

    def knob_spec(self) -> Dict[str, Any]:
        """The manual-control knob spec: {name: {type, min, max, choices}}.
        Feeds the Research tab's Manual Control panel."""

    def knob_defaults(self) -> Dict[str, Any]:
        """Current value of every knob in knob_spec(), so a UI can seed itself
        from live behaviour — 'untouched' must mean 'current default'."""

    def cancel(self, ctx=None) -> bool:
        """Stop the run this client is executing. Returns True if a stop was
        actually delivered. For cancellation_mode == 'shared_event' the caller's
        own Stop already did this and it is a no-op returning True."""


class ResearchBackendError(RuntimeError):
    """Raised by a remote backend when the service reports a failure.

    In-process failures propagate as whatever deep_research raised, so existing
    except-clauses in the consumers keep behaving identically.
    """
