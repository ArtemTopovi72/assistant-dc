"""The one home for the deep-research runtime knobs.

Every DR_* value the Ultra Search pipeline reads lives HERE, in exactly one
place, and every pipeline module reads it as an ATTRIBUTE (``S.DR_MAX_PAGES``)
rather than binding a copy with ``from dr_settings import DR_MAX_PAGES``.

That is not style — it is the whole point. These knobs are MUTABLE at runtime:

* the GUI's "Manual Control" panel calls `apply_overrides` to retune a single
  run and `restore_overrides` afterwards, and
* the suites patch them directly (``deep_research.DR_MAX_SOURCES = 3``).

A ``from`` import would snapshot the value at import time, so a module holding
its own copy would quietly keep running on the pre-override number while the
panel showed the new one. One home, attribute access, no split state.

`deep_research` installs a module proxy so the historical patch surface
(`deep_research.DR_X`, which the GUI panel reads and the suites write) reads and
writes THIS module — see the `_SettingsView` note there.
"""
from config import (
    LM_STUDIO_BASE,
    DR_MAX_QUERIES, DR_RESULTS_PER_QUERY, DR_MAX_PAGES, DR_MAX_DEPTH,
    DR_LINKS_PER_PAGE, DR_PAGE_TIMEOUT, DR_PAGE_CHARS, DR_BRIEF_TOKENS,
    DR_FETCH_DELAY, DR_USER_AGENT, DR_DIR,
    DR_SEARCH_CONCURRENCY, DR_FETCH_CONCURRENCY, DR_FETCH_JITTER,
    SOCIAL_OUTLINKS_PER_PAGE,
    DR_CACHE_ENABLED, DR_CACHE_DIR, DR_ENABLE_ADAPTERS, DR_REPLAN_ENABLED,
    DR_PDF_ENABLED, DR_PDF_MAX_PAGES, DR_PDF_MAX_BYTES, DR_PDF_TEXT_CHARS,
    DR_CITATIONS_ENABLED, DR_CITATION_MAILTO,
    DR_REPORT_TOKENS_QUICK, DR_REPORT_TOKENS_STANDARD, DR_REPORT_TOKENS_DEEP,
    DR_REPORT_TOKENS_OVERRIDE, DR_RERANK_ENABLED, DR_RERANK_BACKEND, DR_RERANK_TOP_K, DR_RERANK_AUTHORITY_WEIGHT,
    DR_CONTRADICTION_ENABLED, DR_CONTRADICTION_QUERIES, DR_MULTIHOP_ENABLED, DR_MULTIHOP_DEPTH, DR_MULTIHOP_ENTITIES, DR_REFLECTION_ENABLED, DR_DECISION_GATE_ENABLED, DR_GATE_MIN_STRONG, DR_GATE_MIN_CLUSTERS,
    DR_HIERARCHICAL_ENABLED, DR_CLUSTER_THRESHOLD, DR_CLUSTER_MAJOR_MIN,
    DR_GRAPH_EXPANSION_ENABLED, DR_GRAPH_EXPANSION_QUERIES,
    DR_CLAIM_MERGE_ENABLED, DR_CLAIM_MERGE_THRESHOLD, DR_CLUSTER_DIGEST_MAX,
    DR_HIERARCHY_MAX_DEPTH, DR_HIERARCHY_MIN_CLUSTER_SIZE, DR_HIERARCHY_SPLIT_THRESHOLD,
    DR_COMMUNITIES_ENABLED, DR_REFLECTION_MAX_ITERATIONS, DR_REFLECTION_RESEARCH_BUDGET,
    DR_PHASE_HISTORY_ENABLED,
    DR_MIN_SOURCES, DR_TARGET_SOURCES, DR_MAX_SOURCES, DR_MIN_UNIQUE_DOMAINS,
    DR_MAX_PAGES_PER_DOMAIN, DR_MAX_COLLECTION_ROUNDS, DR_FORCE_EXHAUSTIVE,
    DR_DOMAIN_WHITELIST, DR_DOMAIN_BLACKLIST, DR_SOURCE_TYPES_EXCLUDE,
    DR_QUERY_MUTATION_ATTEMPTS,
    DR_SURVEY_MODE, DR_SURVEY_SECTION_TOKENS, DR_SURVEY_MAX_SECTIONS,
    DR_SECTION_TOKEN_CEILING, DR_MODEL_CONTEXT, DR_PLAN_TOKENS,
    DR_REASONING_HEADROOM, DR_REASONING_EFFORT,
    DR_BRIEF_TOKEN_CEILING, DR_BRIEF_INPUT_CHARS, DR_BRIEF_MODEL,
)

# The exact set of names this module owns — computed from what was just
# imported, so adding a knob to the import list above is the only step needed.
# `deep_research`'s proxy consults this to decide which attribute reads/writes
# to forward here.
SETTING_NAMES = frozenset(
    n for n, v in list(globals().items())
    if n.isupper() and not n.startswith("_")
)


# --------------------------------------------------------------------------- #
# Manual control — whitelist of knobs a caller (the GUI's "Manual Control"
# panel) may override for a single run, without touching the process-wide
# config defaults. Every entry mirrors a real DR_* global above (set at import
# time from config.py, so config.DR_X after import has no effect — these helpers
# patch THIS module's globals, the only thing the pipeline actually consults).
# --------------------------------------------------------------------------- #
MANUAL_OVERRIDE_SPEC = {
    "DR_MAX_QUERIES":              {"type": "int",   "min": 1,    "max": 40},
    "DR_MAX_PAGES":                {"type": "int",   "min": 1,    "max": 200},
    "DR_RESULTS_PER_QUERY":        {"type": "int",   "min": 1,    "max": 30},
    "DR_MULTIHOP_ENABLED":         {"type": "bool"},
    "DR_MULTIHOP_DEPTH":           {"type": "int",   "min": 0,    "max": 5},
    "DR_CONTRADICTION_ENABLED":    {"type": "bool"},
    "DR_CONTRADICTION_QUERIES":    {"type": "int",   "min": 0,    "max": 20},
    "DR_REFLECTION_ENABLED":       {"type": "bool"},
    "DR_REFLECTION_MAX_ITERATIONS": {"type": "int",  "min": 0,    "max": 10},
    "DR_REFLECTION_RESEARCH_BUDGET": {"type": "int", "min": 0,    "max": 50},
    "DR_HIERARCHICAL_ENABLED":     {"type": "bool"},
    "DR_COMMUNITIES_ENABLED":      {"type": "bool"},
    "DR_GRAPH_EXPANSION_ENABLED":  {"type": "bool"},
    "DR_CLAIM_MERGE_ENABLED":      {"type": "bool"},
    "DR_CITATIONS_ENABLED":        {"type": "bool"},
    "DR_PDF_ENABLED":              {"type": "bool"},
    "DR_RERANK_ENABLED":           {"type": "bool"},
    "DR_RERANK_BACKEND":           {"type": "choice", "choices": ("auto", "cross", "dense", "lexical")},
    "DR_RERANK_TOP_K":             {"type": "int",   "min": 0,    "max": 60},  # 0 = keep all
    # Report output budget + reasoning depth (see synthesize_report). Big budgets are
    # intentional: gpt-oss splits reasoning+output from one pool, so headroom is added.
    "DR_REPORT_TOKENS_OVERRIDE":   {"type": "int",   "min": 0,    "max": 60000},
    "DR_REASONING_HEADROOM":       {"type": "int",   "min": 0,    "max": 60000},
    "DR_REASONING_EFFORT":         {"type": "choice", "choices": ("auto", "low", "medium", "high")},
    # Survey mode: dynamic-outline, section-by-section long-form document synthesis.
    "DR_SURVEY_MODE":              {"type": "bool"},
    "DR_SURVEY_SECTION_TOKENS":    {"type": "int",   "min": 500,  "max": 16000},
    "DR_SURVEY_MAX_SECTIONS":      {"type": "int",   "min": 3,    "max": 24},
    "DR_DECISION_GATE_ENABLED":    {"type": "bool"},
    "DR_REPLAN_ENABLED":           {"type": "bool"},
    "DR_GATE_MIN_STRONG":          {"type": "int",   "min": 0, "max": 10},
    "DR_GATE_MIN_CLUSTERS":        {"type": "int",   "min": 0, "max": 10},
    "DR_CLAIM_MERGE_THRESHOLD":    {"type": "float", "min": 0.0, "max": 1.0},
    # Sources section
    "DR_MIN_SOURCES":              {"type": "int",  "min": 0, "max": 500},
    "DR_TARGET_SOURCES":           {"type": "int",  "min": 0, "max": 500},
    "DR_MAX_SOURCES":              {"type": "int",  "min": 0, "max": 1000},
    "DR_MIN_UNIQUE_DOMAINS":       {"type": "int",  "min": 0, "max": 200},
    "DR_MAX_PAGES_PER_DOMAIN":     {"type": "int",  "min": 0, "max": 100},
    "DR_DOMAIN_WHITELIST":         {"type": "str"},
    "DR_DOMAIN_BLACKLIST":         {"type": "str"},
    "DR_SOURCE_TYPES_EXCLUDE":     {"type": "str"},
    # Research depth / exhaustiveness section
    "DR_MAX_COLLECTION_ROUNDS":    {"type": "int",  "min": 1, "max": 20},
    "DR_FORCE_EXHAUSTIVE":         {"type": "bool"},
    # Retry / resilience section
    "DR_QUERY_MUTATION_ATTEMPTS":  {"type": "int",  "min": 0, "max": 3},
    # I/O parallelism (1 = serial). Bounded to keep ddgs/site rate-limits happy.
    "DR_SEARCH_CONCURRENCY":       {"type": "int",  "min": 1, "max": 8},
    "DR_FETCH_CONCURRENCY":        {"type": "int",  "min": 1, "max": 8},
}


def _coerce_override(name: str, value):
    spec = MANUAL_OVERRIDE_SPEC[name]
    if spec["type"] == "bool":
        return bool(value)
    if spec["type"] == "choice":
        if value not in spec["choices"]:
            raise ValueError(f"{name}: {value!r} not in {spec['choices']}")
        return value
    if spec["type"] == "str":
        return str(value or "")
    n = float(value) if spec["type"] == "float" else int(value)
    if n < spec["min"] or n > spec["max"]:
        raise ValueError(f"{name}: {n} outside [{spec['min']}, {spec['max']}]")
    return n


def apply_overrides(overrides: dict) -> dict:
    """Patch this module's globals with caller-supplied values for one run.
    Unknown keys raise (fail loud rather than silently ignoring a typo'd knob).
    Returns the previous values so the caller can restore them afterward —
    overrides are per-run, never a permanent config change."""
    # Validate + coerce EVERYTHING first, mutate nothing until all pass — so an
    # invalid knob can't leave the module in a half-applied state (which, since
    # the caller never receives `saved`, would leak permanently into later runs).
    coerced = {}
    for name, value in (overrides or {}).items():
        if name not in MANUAL_OVERRIDE_SPEC:
            raise KeyError(f"not a manual-control knob: {name}")
        coerced[name] = _coerce_override(name, value)
    saved = {}
    for name, new_value in coerced.items():
        saved[name] = globals()[name]
        globals()[name] = new_value
    return saved


def restore_overrides(saved: dict) -> None:
    for name, value in (saved or {}).items():
        globals()[name] = value
