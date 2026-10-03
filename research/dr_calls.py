"""How this engine talks to the model.

One place for the two things every LLM-backed research stage needs and neither
briefing nor synthesis should re-invent:

* `_think_call` — a "let it think, but make it ANSWER" call. gpt-oss-class models
  spend reasoning and output from one token pool, so a long deliberation can eat
  the whole budget and return an empty final answer. This retries with a shrunk
  budget and an explicit "write the final answer NOW, keeping the Markdown
  structure" nudge, and clamps every budget to the model's real window.
* `_model_context` — that real window, read LIVE from LM Studio's /api/v0 rather
  than guessed, and cached per model id.

This module is also the ONE home of `call_llm_simple` for the research engine:
every stage that talks to a model does so through here (dr_brief calls
`dr_calls.call_llm_simple` rather than importing its own), so a suite that fakes
the model patches exactly one name and the fake holds for the whole pipeline.
"""
import logging

import requests

import dr_settings as S
from dr_outline import _est_tokens
from llm import call_llm_simple

logger = logging.getLogger("assistant.research")


_EFFORT_FROM_CONFIG = object()  # sentinel: derive effort from DR_REASONING_EFFORT


_VALID_EFFORT = ("low", "medium", "high")


def _resolve_effort(effort):
    """Map the `effort` argument onto llm.call_llm_simple's `reasoning_effort`.

    `_EFFORT_FROM_CONFIG` (the default) reads DR_REASONING_EFFORT, where the
    string "auto" means "leave it to ctx / the model default" -> None. An explicit
    low/medium/high is passed straight through; anything else degrades to None
    rather than sending garbage to the backend."""
    if effort is _EFFORT_FROM_CONFIG:
        effort = S.DR_REASONING_EFFORT
    if isinstance(effort, str) and effort.lower() in _VALID_EFFORT:
        return effort.lower()
    return None


def _think_call(ctx, system: str, user: str, max_tokens: int,
                temperature: float = 0.3, retries: int = 2,
                effort=_EFFORT_FROM_CONFIG) -> str:
    """A synthesis LLM call that LETS THE MODEL THINK and keeps only the written
    answer. Reasoning fine-tunes (Qwen <think>, gpt-oss harmony) write far better
    long-form prose WITH chain-of-thought; suppressing it (the old prefill+/no_think
    combo) backfired into empty sections. `max_tokens` must cover think + answer.
    Returns the cleaned content (think block already stripped by the llm layer).

    Defensively clamps max_tokens so prompt + output stays inside DR_MODEL_CONTEXT —
    otherwise LM Studio rejects the request ('n_keep >= n_ctx') and returns empty."""
    prompt_tok = _est_tokens(system) + _est_tokens(user)
    safe = max(256, _model_context(ctx) - prompt_tok - 80)
    budget = min(max_tokens, safe)
    _effort = _resolve_effort(effort)
    out = call_llm_simple(ctx, system, user, temperature=temperature,
                          max_tokens=budget, force_think=False,   # 2026-09-25 user: no reasoning;
                          reasoning_effort=_effort)             # the thinking retry below stays the fallback
    if (out or "").strip():
        return out.strip()

    # Empty is not "the model had nothing to say" — measured on
    # google/gemma-4-26b-a4b-qat, 4 of 10 identical calls came back as a
    # `<|channel>thought` block with NO final answer after it, so stripping the
    # reasoning left nothing. It is not a budget cliff either: it happened at 1500
    # tokens and at 6767, and MORE budget made it more likely, because a bigger
    # allowance is a bigger invitation to keep deliberating.
    #
    # So: ask again, smaller and more bluntly. A shorter allowance and an explicit
    # "write it now" is the only lever a model with no reasoning-effort knob has.
    #
    # `retries` is capped here because a wrong PROMPT TEMPLATE (not a budget cliff)
    # produces the exact same empty-reasoning-only symptom, and no amount of
    # shrink+nudge fixes a template mismatch. Retrying 3x against the wrong
    # template burned ~600s of wall-clock EACH on a mis-routed "practical"
    # synthesis (tractor-comparison topic sent to the local-services template),
    # ~30 minutes wasted before giving up. Callers that suspect their template may
    # be wrong for the topic should pass a smaller `retries` so they fail fast and
    # fall back to a different template instead of hammering the same one.
    for attempt, shrink in enumerate((0.5, 0.3)[:retries], start=1):
        # The nudge must not cost the FORMAT. An earlier version said only "output
        # the finished text", and the retry came back as a wall of unstructured
        # prose — the model dropped the Markdown headings along with the
        # deliberation, which is a worse failure than the empty it was fixing.
        nudge = ("\n\nWrite the final answer NOW, in full, in the Markdown "
                 "structure the instructions above require — keep every `#` "
                 "heading and every list. Do not deliberate further and do not "
                 "explain your approach; output only the finished document.")
        retry_budget = max(512, int(budget * shrink))
        logger.warning("synthesis came back empty (all reasoning, no answer) — "
                       "retry %d at %d tokens", attempt, retry_budget)
        out = call_llm_simple(ctx, system + nudge, user + nudge,
                              temperature=min(0.9, temperature + 0.1 * attempt),
                              max_tokens=retry_budget, force_think=False,
                              reasoning_effort=_effort)
        if (out or "").strip():
            return out.strip()
    logger.error("synthesis produced no answer after %d attempt(s)", retries + 1)
    return ""


_CONTEXT_CACHE: dict = {}


def _model_context(ctx) -> int:
    """The ACTUAL context window the model is loaded with, read live from LM Studio's
    native API (`/api/v0/models/<id>` → loaded_context_length). This is what makes
    "no limits" real: budgets derive from the true window (e.g. 262144) instead of a
    guessed constant. Cached per model; falls back to DR_MODEL_CONTEXT if the API is
    unavailable or returns nothing usable."""
    name = getattr(ctx, "model_name", "") or ""
    if name in _CONTEXT_CACHE:
        return _CONTEXT_CACHE[name]
    detected = 0
    try:
        r = requests.get(f"{S.LM_STUDIO_BASE}/api/v0/models/{name}", timeout=8)
        if r.status_code == 200:
            d = r.json()
            detected = int(d.get("loaded_context_length")
                           or d.get("max_context_length") or 0)
    except Exception as exc:
        logger.debug("context auto-detect failed (%s); using DR_MODEL_CONTEXT", exc)
    val = detected if detected >= 2048 else S.DR_MODEL_CONTEXT
    _CONTEXT_CACHE[name] = val
    logger.info("model context window: %d tokens (%s)",
                val, "live-detected" if detected else "config fallback")
    return val
