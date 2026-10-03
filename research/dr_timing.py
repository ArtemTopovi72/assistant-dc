"""Research depth profiles and the MEASURED duration estimate behind the ETA.

The three depths the product advertises (`quick` / `standard` / `deep`) and the
only honest way to say how long one takes: record what completed runs actually
cost and quote the median of the recent ones.

`record_run_duration` appends a completed run, `estimate_duration` returns
`(seconds, n_samples)` — n_samples == 0 meaning the number is still the seed
rather than a measurement. `_profile_signature` fingerprints what each depth
DOES, so changing the caps discards history instead of averaging across two
different pipelines.

Split out of deep_research.py. The caps themselves stay in deep_research: they
are computed from DR_* module globals that `apply_overrides` mutates in place
for a single run, so they have exactly one home and are read through it (see
`_profile_signature`'s deferred import).
"""
import json
import logging
import os
import threading

from config import DR_DIR

logger = logging.getLogger("assistant.research")

DEPTHS = ("quick", "standard", "deep")
_TIMING_FILE = DR_DIR / "_timings.json"
# The three options the product advertises: ~15 min / ~30 min / ~1 hour. These are
# only the STARTING estimate — once a depth has completed runs, estimate_duration
# returns the median of the real ones instead, so the shown ETA converges on this
# machine's actual speed. Derived from one real observed standard run (2074s on
# this machine) scaled by the caps ratio.
_TIMING_SEED = {"quick": 900, "standard": 1800, "deep": 3600}
_TIMING_KEEP = 12          # median over the last N runs per depth
# Below this a "run" did not really happen (stub, abort, cache hit).
# 300s, not 60. Measured 2026-08-29 from the live history: quick held
# [5492, 120, 120, 93, 117, 90] and standard [119, 244, 120, 126]. Exactly one
# of those is a research run -- the 91-minute one; the rest are stubs and
# aborted runs from test and bench passes that reached the "completed" call.
# The median was therefore ~2 minutes, and the bot quoted "2 min" to a user
# about to wait half an hour. The FASTEST depth is seeded at 900s, so nothing
# real finishes in five minutes and the floor can say so.
_MIN_PLAUSIBLE_RUN_S = float(os.getenv("DR_MIN_PLAUSIBLE_RUN_S", "300"))
_timing_lock = threading.Lock()


def _profile_signature() -> str:
    """Fingerprint of what each depth actually DOES.

    A measured duration only describes the profile it was measured under. When
    the caps change — e.g. `quick` stopped doing a second reflection pass — every
    stored timing becomes a measurement of a pipeline that no longer exists, and
    the UI kept quoting "43 min" for a run that had just been halved. Changing the
    caps must therefore discard the history rather than average across two
    different products.

    `_resolve_caps` is imported HERE rather than at module scope on purpose: it
    reads deep_research's DR_* globals, which `apply_overrides` swaps out for the
    duration of a run. Binding it once at import would fingerprint the caps as
    they were at startup and miss an override entirely.
    """
    import hashlib
    from deep_research import _resolve_caps
    keys = ("max_queries", "max_pages", "depth", "links_per_page",
            "report_tokens", "reflect_iters", "reflect_budget")
    raw = ";".join(f"{d}:" + ",".join(str(_resolve_caps(d).get(k)) for k in keys)
                   for d in DEPTHS)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def redirect_timings(directory) -> None:
    """Point the timing history at `directory` — for tests.

    The history is persistent shared state: without this a suite recorded its
    stub runs into the operator's real file and then read them back, which is
    how the two-minute estimate above got there in the first place.
    """
    global _TIMING_FILE
    from pathlib import Path as _Path
    d = _Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    _TIMING_FILE = d / "_timings.json"


def _load_timings() -> dict:
    try:
        with open(_TIMING_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("_profile") != _profile_signature():
            logger.info("research depth profile changed — discarding stale timings")
            return {d: [] for d in DEPTHS}
        return {d: [float(x) for x in data.get(d, []) if x] for d in DEPTHS}
    except Exception:
        return {d: [] for d in DEPTHS}


def record_run_duration(depth: str, seconds: float) -> None:
    """Remember how long a COMPLETED run actually took, so the next estimate is
    better. Cancelled and failed runs are not recorded — they would drag the
    estimate down and quote a wait nobody will actually get."""
    depth = (depth or "standard").lower()
    if depth not in DEPTHS or not seconds or seconds <= 0:
        return
    # A real run of ANY depth takes minutes. Anything shorter is a stubbed test,
    # an aborted run, or a cache hit — recording it poisons the median and the
    # bot then promises the user "0 min". Observed live: the file held
    # [0.1, 0.1, 0.0] and the quick estimate was 0 minutes.
    if float(seconds) < _MIN_PLAUSIBLE_RUN_S:
        logger.debug("ignoring an implausible %.1fs run at depth=%s", seconds, depth)
        return
    try:
        with _timing_lock:
            data = _load_timings()
            data[depth] = (data[depth] + [round(float(seconds), 1)])[-_TIMING_KEEP:]
            # Stamp the profile these numbers describe, so a later cap change
            # discards them instead of blending two different pipelines.
            data["_profile"] = _profile_signature()
            _TIMING_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = _TIMING_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(_TIMING_FILE)
    except Exception:
        logger.debug("could not record run duration", exc_info=True)


def _median(samples: list) -> float:
    ordered = sorted(samples)
    mid = len(ordered) // 2
    return (ordered[mid] if len(ordered) % 2
            else (ordered[mid - 1] + ordered[mid]) / 2)


def estimate_duration(depth: str = "standard") -> tuple:
    """(seconds, n_samples) for one run at this depth.

    n_samples == 0 means the number is the seed, not a measurement — callers
    show it as an approximation either way, but it lets the UI say so.
    """
    depth = (depth or "standard").lower()
    if depth not in DEPTHS:
        depth = "standard"
    timings = _load_timings()
    # Discard junk that predates the floor above (the file is persisted, so a
    # previously poisoned history would otherwise keep quoting 0 minutes).
    samples = [x for x in (timings.get(depth) or []) if x >= _MIN_PLAUSIBLE_RUN_S]
    if samples:
        return float(_median(samples)), len(samples)
    # No measurement of its own: infer it from whichever sibling depths DO
    # have one, scaled by the declared seed ratio between them, rather than
    # quoting the flat unrelated seed. Observed live: `standard` had nothing
    # left but old stub runs below the floor, so it quoted the raw 1800s seed
    # while `deep`'s 2 real runs measured a faster ~1350s -- the menu showed
    # Standard (~30 min) taking longer than Deep (~23 min).
    scaled = []
    for other in DEPTHS:
        if other == depth:
            continue
        other_samples = [x for x in (timings.get(other) or [])
                         if x >= _MIN_PLAUSIBLE_RUN_S]
        if other_samples:
            scaled.append(_median(other_samples)
                          * (_TIMING_SEED[depth] / _TIMING_SEED[other]))
    if scaled:
        return float(sum(scaled) / len(scaled)), 0
    return float(_TIMING_SEED[depth]), 0
