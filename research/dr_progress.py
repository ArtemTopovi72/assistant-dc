"""Progress plumbing for a deep-research run.

`_Progress` accumulates the run's stats, times each pipeline stage at its
transitions, and forwards everything to an optional UI callback (the GUI
Research tab's progress panel, the Telegram status message). The callback is
untrusted: an exception from the UI side is swallowed rather than aborting a
research run that is otherwise going fine.
"""
import time
from typing import Callable, Optional


# Phase labels surfaced to the GUI progress panel (the spec's example statuses).
PHASES = ("Expanding queries", "Searching", "Crawling", "Extracting",
          "Deduplicating", "Verifying", "Building report", "Complete")


# --------------------------------------------------------------------------- #
# Progress plumbing
# --------------------------------------------------------------------------- #
class _Progress:
    """Accumulates run stats and forwards them to an optional UI callback.

    The callback receives (phase: str, stats: dict, message: str). It must never
    raise into the engine — any exception from the UI side is swallowed.
    """

    def __init__(self, cb: Optional[Callable[[str, dict, str], None]]):
        self._cb = cb
        self.stats = {"queries": 0, "sources": 0, "pages": 0, "findings": 0,
                      "stage_timings": {}}
        self.phase = ""
        # Per-stage wall-clock timing, accumulated at every phase TRANSITION so it
        # needs no call-site changes — answers "is the bottleneck network or the
        # 9B?". stage_timings[phase] = total seconds spent reporting that phase.
        self._phase_started = time.time()
        # Ordered (stage_name, text) snapshots for the per-stage formula trace
        # (only populated on scientific/math runs — see _snap_formula).
        self.formula_snapshots = []

    def _accumulate_phase_time(self, new_phase: str) -> None:
        if new_phase == self.phase:
            return
        now = time.time()
        if self.phase:
            t = self.stats["stage_timings"]
            t[self.phase] = round(t.get(self.phase, 0.0) + (now - self._phase_started), 2)
        self._phase_started = now

    def finalize_timings(self) -> None:
        """Flush the in-flight phase's elapsed time (call once at run end)."""
        now = time.time()
        if self.phase:
            t = self.stats["stage_timings"]
            t[self.phase] = round(t.get(self.phase, 0.0) + (now - self._phase_started), 2)
        self._phase_started = now

    def snap_formula(self, stage: str, text: str) -> None:
        """Record a stage text snapshot for the formula trace (best-effort)."""
        try:
            self.formula_snapshots.append((stage, text or ""))
        except Exception:
            pass

    def bump(self, phase: Optional[str] = None, message: str = "", **delta):
        """ADD to the counters instead of replacing them.

        The crawler runs in several waves -- the first search, the expansion
        pass, the contradiction hunt -- and each wave counts its OWN pages from
        zero. Reported through update(), which assigns, the run header then
        claimed "4 pages read" for a run that read twenty: the last wave
        overwrote the total. Anything genuinely cumulative belongs here.
        """
        merged = {k: int(self.stats.get(k, 0) or 0) + int(v) for k, v in delta.items()}
        return self.update(phase, message, **merged)

    def update(self, phase: Optional[str] = None, message: str = "", **delta):
        if phase is not None:
            self._accumulate_phase_time(phase)
            self.phase = phase
        for k, v in delta.items():
            self.stats[k] = v
        if self._cb is not None:
            try:
                self._cb(self.phase, dict(self.stats), message)
            except Exception:
                pass
