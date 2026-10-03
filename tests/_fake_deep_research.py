"""A deterministic stand-in for deep_research, used by the differential harness.

WHY IT EXISTS. The real pipeline needs a live LM Studio, a GPU and the open web,
and one run takes minutes. None of that can be part of a differential test whose
job is to prove the two BACKENDS agree — the pipeline is the same code on both
sides, so making it real would only add nondeterminism to the one thing under
test. This module reproduces exactly the surface research_client and
research_service touch, and nothing else:

    MANUAL_OVERRIDE_SPEC / the DR_* globals it names
    apply_overrides / restore_overrides   (same fail-loud validation)
    lang_of_text                          (same Cyrillic-vs-Latin rule)
    estimate_duration
    run_deep_research                     (emits PHASES ticks, polls is_cancelled)

It is imported directly by the in-process side and installed into the SERVICE
process through research_service.ENV_DR_MODULE, so both backends run the same
stand-in and any difference the harness reports is a difference in the BACKEND.
"""
from __future__ import annotations

import time

PHASES = ("Expanding queries", "Searching", "Crawling", "Extracting",
          "Deduplicating", "Verifying", "Building report", "Complete")

DR_MAX_QUERIES = 8
DR_MAX_PAGES = 40
DR_MULTIHOP_ENABLED = True
DR_DOMAIN_WHITELIST = ""

MANUAL_OVERRIDE_SPEC = {
    "DR_MAX_QUERIES":      {"type": "int",  "min": 1, "max": 40},
    "DR_MAX_PAGES":        {"type": "int",  "min": 1, "max": 200},
    "DR_MULTIHOP_ENABLED": {"type": "bool"},
    "DR_DOMAIN_WHITELIST": {"type": "str"},
}

#: Seeded, not measured — same distinction estimate_duration makes upstream.
_SEED = {"quick": 300.0, "standard": 900.0, "deep": 2400.0}

#: Seconds between progress ticks. Long enough that the cancel round-trip has
#: somewhere to land, short enough that the harness stays fast.
TICK_S = 0.15
#: The cancel probe's ticks: its Stop crosses a poll interval (0.25 s) and an
#: HTTP round-trip, which a loaded CI runner stretched past the whole 1.2 s run.
CANCEL_PROBE_TICK_S = 1.0


def _coerce(name, value):
    spec = MANUAL_OVERRIDE_SPEC[name]
    t = spec["type"]
    if t == "int":
        v = int(value)
        if not (spec["min"] <= v <= spec["max"]):
            raise ValueError(f"{name} out of range: {v}")
        return v
    if t == "bool":
        return bool(value)
    return str(value)


def apply_overrides(overrides: dict) -> dict:
    """Validate everything first, mutate nothing until all pass — so a bad knob
    cannot leave the module half-applied. Same contract as the real one."""
    coerced = {}
    for name, value in (overrides or {}).items():
        if name not in MANUAL_OVERRIDE_SPEC:
            raise KeyError(f"not a manual-control knob: {name}")
        coerced[name] = _coerce(name, value)
    saved = {}
    for name, new_value in coerced.items():
        saved[name] = globals()[name]
        globals()[name] = new_value
    return saved


def restore_overrides(saved: dict) -> None:
    for name, value in (saved or {}).items():
        globals()[name] = value


def lang_of_text(text: str, default: str = "en") -> str:
    if any("Ѐ" <= ch <= "ӿ" for ch in (text or "")):
        return "ru"
    return default


def estimate_duration(depth: str = "standard"):
    depth = (depth or "standard").lower()
    if depth not in _SEED:
        depth = "standard"
    return float(_SEED[depth]), 0


def run_deep_research(ctx, topic: str, *, depth: str = "standard",
                      out_lang: str = "en", progress=None) -> dict:
    """Walk PHASES, tick progress, honour ctx.is_cancelled() at every phase.

    The stats dict is a pure function of the phase index so the two backends'
    progress streams are comparable record for record.
    """
    topic = (topic or "").strip()
    if not topic:
        return {"report": "", "path": None, "stats": {}, "cancelled": False,
                "error": "empty topic"}
    for i, phase in enumerate(PHASES):
        if ctx is not None and ctx.is_cancelled():
            return {"report": "", "path": None,
                    "stats": {"sources": i, "pages": i * 2, "findings": i * 3},
                    "cancelled": True}
        if progress is not None:
            progress(phase, {"sources": i, "pages": i * 2, "findings": i * 3}, f"{phase}…")
        time.sleep(CANCEL_PROBE_TICK_S if topic == "cancel probe" else TICK_S)
    return {
        "report": f"# {topic}\n\nlang={out_lang} depth={depth} "
                  f"queries={DR_MAX_QUERIES} pages={DR_MAX_PAGES} "
                  f"multihop={DR_MULTIHOP_ENABLED} whitelist={DR_DOMAIN_WHITELIST!r}\n",
        "path": None,          # deliberately None: a real path is host-local and
                               # would legitimately differ between backends.
        "stats": {"sources": len(PHASES), "pages": len(PHASES) * 2,
                  "findings": len(PHASES) * 3},
        "cancelled": False,
    }
