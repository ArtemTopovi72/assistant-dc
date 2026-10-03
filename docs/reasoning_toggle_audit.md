# "Disable reasoning" toggle — audit, hard evidence, real fix

## Test rig (exact, reproducible)

- **Model id:** `qwen3.5-9b-uncensored-hauhaucs-aggressive@q8_0` (confirmed in BOTH
  request payloads — see below, identical string in ON and OFF runs).
- **Provider / backend:** LM Studio, local, llama.cpp GGUF (`@q8_0` quant).
- **Endpoint:** `http://localhost:1234/v1/chat/completions` (`config.LM_STUDIO_URL`).
- Capture method: `requests.post` wrapped inside `llm.py` to record the **real**
  outbound payload and inbound JSON — not a reconstruction.
  (`tests/concise_evidence.py`, `tests/reasoning_lever.py`, `tests/prefill_tools.py`.)
- `usage.completion_tokens` is the decisive metric: it counts tokens the model
  **actually generated** (including any `<think>` block), so it distinguishes
  "reasoned less" from "hid the reasoning".

## What the toggle did originally — PRESENTATION ONLY

`ctx.no_think` only affected `llm.py`: append `/no_think`, set
`chat_template_kwargs.enable_thinking=False`, set `reasoning:"off"` (the last one was
set unconditionally in both modes). It then **stripped** `<think>` from the output.
Everything governing behaviour — system prompt, tools, `temperature=0.5`,
`max_tokens=1500`, the whole tool loop — was identical. Verdict: it hid the
reasoning channel; it did not change behaviour.

## Hard evidence — the user was RIGHT

Same model, same task, no tools (`tests/concise_evidence.py`):

| | THINKING ON | "DIRECT" (no_think only) |
|---|---|---|
| payload.model | `qwen3.5-9b…@q8_0` | `qwen3.5-9b…@q8_0` (same) |
| temperature | 0.5 | 0.2 |
| max_tokens | 1500 | 700 |
| reasoning | `"off"` | `"off"` |
| chat_template_kwargs | `None` | `{enable_thinking: False}` |
| reasoning_effort | absent | absent |
| **completion_tokens** | **110** | **109** |
| raw content | `<think>…full plan…</think>…` | `<think>…full plan…</think>…` |

`enable_thinking:False` + `/no_think` were **NOT honored** for this prompt: the model
emitted a **full `<think>` block in both modes** and generated essentially the same
number of tokens (110 vs 109). The shorter final answer seen earlier was just the
post-`</think>` text being shorter — **chain-of-thought suppression was hiding output,
not reducing reasoning.** Exactly the reported symptom.

(`/no_think` IS honored for *trivial* prompts — e.g. "18×24": no_think gave an empty
`<think></think>`, 8 tokens vs 62 — but it is unreliable and fails precisely on the
agent/tool prompts that matter. `reasoning_content` was always empty; the model
inlines CoT into `content`.)

## The lever that actually works — prefilled closed `<think></think>`

`tests/reasoning_lever.py` and `tests/prefill_tools.py`, same model:

| Config | `<think>` generated? | completion_tokens | tool_calls |
|---|---|---|---|
| no_think, no tools | yes | 62 | — |
| no_think + `/no_think`, no tools | empty block | 8 | — |
| **prefill `<think></think>`, no tools** | **NO** | **5** | — |
| no prefill, **with tools** | **yes** | **97** | ✅ search |
| **prefill `<think></think>`, with tools** | **NO** | **40** | ✅ search |

Prefilling a **closed** `<think></think>` as the assistant turn forces generation to
begin *after* the reasoning block, so the model **cannot** generate one. This:
- genuinely cuts generated tokens (97 → 40 with tools; 62 → 5 without) — the model
  **reasons less**, it is not merely stripped afterward;
- **still returns correct `tool_calls`** (the `llm.py` docstring's "do not combine
  with tools" warning is overly cautious for this LM Studio backend — empirically
  verified compatible).

## The fix (graph.py)

Direct mode (`ctx.no_think`) now sets, for **every** agent LLM call (the tool-loop
call and the forced closing reply):
- `prefill="<think></think>"` — **the load-bearing change**: the model stops
  generating the planning block;
- the concise system directive (`prompts._CONCISE_DIRECTIVE`): fewest tools, short
  answers, act-don't-narrate;
- `temperature 0.2`, `max_tokens 700`.

`/no_think` + `enable_thinking:false` are kept (harmless; help models that DO honor
them).

## Honest limits

- **Agent tool-orchestration LOOP is the same code in both modes.** `MAX_TOOL_ROUNDS`,
  retry/repeat-fail guards, fabrication correctives all still run. Direct mode makes
  the model *choose* fewer tool calls (concise directive) and stops its `<think>`
  planning (prefill), but the loop machinery itself is not bypassed — nor should it
  be (it is the safety net). So "tool planning still executes" is partly true at the
  *orchestration* layer; what changed is the *model's* per-call reasoning.
- **No provider-injected reasoning / reasoning_effort.** `reasoning_effort` is absent
  from payloads; `reasoning_content` is always empty; there is no hidden server-side
  CoT — the reasoning was the model's own inline `<think>`, now suppressed at the
  source by the prefill.
- Prefill compatibility is **backend/model specific**; verified here for this LM
  Studio GGUF. A different model/backend should be re-checked with
  `tests/prefill_tools.py`.

## Files
`prompts.py` (`_CONCISE_DIRECTIVE`, `build_system_prompt(concise=)`) ·
`graph.py` (`gen_prefill`/`gen_temperature`/`gen_max_tokens` from `concise`) ·
`gui.py` (label + hint) · `tests/concise_evidence.py`, `tests/reasoning_lever.py`,
`tests/prefill_tools.py`.
