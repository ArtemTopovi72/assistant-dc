# Vision pipeline forensics — is the empty-vision bug the model or the app?

**Date:** 2026-06-27. Investigation demanded raw-evidence proof before any architecture
decision. Conclusion: **neither the parser nor the model is at fault. The empties came
from LM Studio failing to serve the requested model** (compounded by a test-harness
confound where unloaded/phantom model ids were requested).

## Method
Standalone scripts talked to LM Studio directly (`/v1/chat/completions`), capturing the
**raw** response with zero processing: HTTP status, headers, full JSON body, every
message field (`content`, `reasoning_content`, `tool_calls`, …), and — for streaming —
every SSE delta. The **served** `model` id was asserted against the requested id. The
same image+prompt was run through the 9B and a 35B. `strip_think_tags` was audited with
before/after diffs.

## Key evidence

### The earlier "35B is blind" finding was a CONFOUND
- `qwen3.6-35b-a3b-uncensored-heretic-i1` is **not in LM Studio's catalog**. Requesting it
  returned a body whose `"model"` field was the loaded **9B** — LM Studio silently
  substitutes the resident model for an unknown id.
- The real catalog 35B (`...hauhaucs@q4_k_m`, 22 GB) returned `{"error":"Model unloaded."}`
  when JIT-requested — too big to stay resident on the 12 GB GPU.
- Concurrent test runs corrupted LM Studio's global model state → spurious empties.

So no 35B was ever actually exercised until it was **explicitly loaded via `lms load`**
and its served id confirmed.

### 9B (loaded), confirmed served
`content` = `<think>\n\n</think>\n\n{json}` (empty closed think block), `reasoning_content`
empty. `strip_think_tags` removed exactly the 19-char think block; the JSON `success/9`
survived. Deterministic across 3 runs, with **and** without the `<think></think>` prefill
(prefill is neither cause nor cure).

### 35B (explicitly loaded, served id asserted == request)
`content` = a **closed** `<think>…1121 chars of real reasoning…</think>\n\n{json}`. The 35B
**sees the image** (describes the white cat + chocolate chips accurately) and returns
`success/10`. `reasoning_content` empty (reasoning is inline in `content`, properly
tagged). `strip_think_tags` removed the 1121-char think block and **kept** the 243-char
JSON answer. Full app `evaluate_image` on this model → `success/9`, real verdict.

### strip_think_tags audit
Removes only `<think>…</think>` (paired, unclosed-through-EOF, or orphan-close). On every
real sample the JSON answer was preserved. It is **not** discarding valid content.

## Conclusion (decision tree)
- Bug in parser / `strip_think_tags` / extraction? **No** — answer is in `content` and is
  preserved for both models.
- Model returns only-reasoning with no usable content after correct parsing? **No** — both
  models put a real answer after a closed think block.
- Therefore **vision routing to a dedicated model is NOT justified** and was reverted (it
  also had a false-positive: one transient empty permanently blacklisted even the working
  9B). Real empties are an **infra/model-serving** issue, handled by a same-model retry +
  the caller's fail-safe accept.

## What was kept vs reverted
- **Reverted:** `config.VISION_MODEL`, `Context._vision_blind_models`, the
  `model_override` plumbing, and the reroute-on-empty logic in `analyze_image_with_llm`
  (now: resident model + one retry).
- **Kept (independently justified):** `evaluate_image` fail-safe accept on a persistent
  empty (graceful for "Model unloaded") + retry; the fair `VISION_EVAL_PROMPT`; the
  honest `inspect_image` fallback; the `generate_image` per-turn cap in `graph.py`.

## Operational note
Requesting a model id that LM Studio has not loaded is unreliable: it may substitute the
resident model (unknown id) or error (`Model unloaded`, oversized). The app should drive
vision with whatever model the user has loaded — which, for the default 9B, is fully
vision-capable.

## The testing confound (this is what actually misled the investigation)
The first conclusion ("the 35B is blind / emits reasoning-only") was **false, and it was
the *test methodology* that was broken**, not the app. Four distinct confounds stacked up:

1. **Phantom model id.** `qwen3.6-35b-a3b-uncensored-heretic-i1` is not in LM Studio's
   catalog. It *looked* like a real model in casual `/v1/models` output but could never be
   served.
2. **Silent substitution.** Requesting an unknown id does not error — LM Studio answers
   with the **currently loaded** model. The response's `model` field said `9B` while we
   believed we were testing the 35B. Every "35B is blind" sample was actually the 9B (or,
   in the reasoning-leak probe, both requests hit the same fallback model and returned
   identical text — the tell we missed).
3. **Model unloaded.** A real but oversized 35B (22–44 GB) returned `{"error":"Model
   unloaded."}` instead of serving, which the harness then mistook for an "empty model
   response."
4. **Global model state.** LM Studio serves one model at a time. Running tests
   concurrently (a background 35B load while a foreground 9B request ran) caused the 9B
   request to return `served=None`/empty mid-swap — which produced the bogus "no-prefill
   empties the 9B" result.

Only after (a) **explicitly `lms load`-ing** a real 35B and (b) **asserting the served
`model` id == requested** did a clean 35B response appear — and it was a perfectly good
`success/10`. The lesson: *never trust the requested model name; verify the served one.*

### Hardened harness — `tests/lmstudio_harness.py`
All future LM Studio integration tests must go through it. It makes every confound above
impossible by construction:
- **Served-id assertion** on every `chat()` → `ModelSubstitution` on mismatch (stops immediately).
- **Preflight availability** via `/api/v0/models` → `ModelNotFound` for a phantom id,
  `ModelUnavailable` if it can't be loaded.
- **Cross-process serial lock** (`serial_lock()`) → two model tests can never run at once.
- **Error bodies are failures** → `{"error": "Model unloaded"}` raises `ModelUnavailable`,
  never a silent empty.

`bench/lmstudio_harness_selftest.py` proves all five guards (incl. catching the phantom id and
the silent substitution that originally misled us): `5/5 guards verified`.

## Routing decision (final)
Routing was **removed**, not merely simplified — the evidence shows neither the model nor
the parser is at fault, so there is nothing for routing to fix. `analyze_image_with_llm`
is now resident-model + one retry (for a transient mid-swap empty); a persistent empty
falls to the caller's fail-safe accept. Critically, **no code treats a single empty
response as "this model is blind" anymore.** Should routing ever be reconsidered, it must
react only to a *verified* infrastructure failure (an explicit `Model unloaded` error or
repeated consecutive failures with the served id confirmed), never a lone empty.
