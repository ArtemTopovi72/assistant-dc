# Where the desktop GUI can — and cannot — be split into services

The goal is to make the GUI a thin client over services. The honest version of
that goal is narrower than the slogan: **a PyQt window cannot become a
microservice.** Widgets, the `QApplication` event loop, the `QThread` workers
and every `pyqtSignal` live in one process by construction. What *can* move
behind a boundary is the **work** the GUI performs, not the window that requests
it.

This document names, from real call sites found by grep, what genuinely crosses
a process boundary, what cannot, and what each candidate costs. The pattern
followed is the one the knowledge layer set: `knowledge/knowledge_api.py`
(Protocol) + `knowledge/knowledge_client.py` (in-process default, HTTP behind an
env var) + `tests/difftest_knowledge_backends.py` (differential,
mutation-checked).

Call sites are named by module and function rather than line number, because
the line numbers drift with every edit.

---

## 0. The two things that make this hard

Everything below is dominated by two mechanisms, so they come first.

### Cancellation

`models.Context.cancel_event` is a `threading.Event` — a **process-local**
object. Three layers depend on it:

* `gui.AssistantWindow._cancel_current` (`gui/gui.py`) sets
  `ctx.cancel_event`, sets `self._turn_cancelled`, calls `self._scan_worker.cancel()`
  (the scan loop polls its own bool, not the event), and then calls `cancel()` on
  the tabs that hold their own token (`transfer_tab`, `storyboard_tab`,
  `code_tab`).
* `gui_common._ScopedCtx` (`gui/gui_common.py`) is a delegating view of `ctx`
  whose **only** override is `cancel_event`. It exists because a single global
  flag collided across surfaces: one Stop press muted the Madhouse permanently,
  and any surface that *cleared* the global flag un-cancelled whatever else was
  in flight.
* The research pipeline polls `ctx.is_cancelled()` at several points across
  `deep_research.py`, `dr_brief.py`, `dr_collect.py`, `dr_crawl.py` and
  `dr_synthesis.py`.

A naive HTTP boundary breaks all three: a `threading.Event` in the GUI process
is invisible to a server process. Cancellation over HTTP is only possible if the
service exposes a **run id** and a separate `POST /cancel/<run_id>` route, and
the server-side handler owns its own `Event` that the pipeline polls. That is a
real design, not a wrapper — and if a candidate cannot support it, the correct
outcome is to say so, not to ship a Stop button that does nothing.

### Progress

Every long GUI operation reports progress through a **Python callable passed by
reference** into the work:

| worker | signal | callback shape |
|---|---|---|
| `LibraryBuildWorker` | `progress(str,int,int)` | `progress=lambda s,d,t: ...` |
| `DeepResearchWorker` | `progress(str,dict,str)` | `progress=lambda phase,stats,msg: ...` |
| `ScanWorker` | `progress(int,int)` | `progress=lambda i,n: ...` |
| `ModelLoader` | `progress(str)` | `status_cb=self.progress.emit` |

A callable cannot cross a process boundary. Preserving live progress requires a
streaming response (chunked/SSE/NDJSON) that the client reads in its worker
thread and re-emits as the same Qt signal. Anything less turns a phase-by-phase
progress strip (`_on_research_progress` in `gui/gui_research_progress.py`) into two ticks —
"started" and "done" — which is a visible product regression, not an
implementation detail.

---

## 1. What genuinely can cross a process boundary

### 1.1 Deep research — **the strongest candidate**

Call sites (grep `run_deep_research`):

* `gui_workers.DeepResearchWorker.run` — `gui/gui_workers.py`
  (apply overrides / `run_deep_research(ctx, topic, depth=, out_lang=, progress=)` /
  restore overrides; now one call through `research_client`, see §4)
* `tools._handle_deep_research` — `agent/tools.py` (the agent's tool call)
* `tg_tasks` — `bot/tg_tasks.py` (the Telegram bot's direct call)
* `tg_depth._depth_eta` — `bot/tg_depth.py` (`_dr.estimate_duration(depth)`)
* `_build_manual_control_panel` / `_reset_manual_control` —
  `gui/gui_manual_panel.py` (reads `DR_*` module globals to seed the Manual
  Control widgets)

Why it can move:

* **Four independent front ends** already consume it (GUI, agent tool, TG bot,
  ETA display) — the same multi-consumer shape that justified the knowledge
  boundary.
* Its inputs are scalars: `topic`, `depth`, `out_lang`, an overrides dict of
  `DR_*` knobs. Its output is a dict (`report`, `path`, stats, `cancelled`) —
  already JSON-shaped because it is persisted to disk.
* It does **not** need the GPU-resident models. It talks to LM Studio, which is
  *already* a separate process over HTTP. `ctx` is used by deep research for LLM
  configuration and cancellation, not for the Whisper/F5 model objects.
* Its progress callback is `(phase: str, stats: dict, msg: str)` — every
  argument is JSON-serialisable. This is the one progress callback in the
  codebase that survives a wire format unchanged.

Cost:

* **Latency**: negligible relative to the work. A standard run is minutes
  (`tg_depth._depth_eta_seconds` floors the queue estimate at 900 s). One HTTP
  round-trip is noise.
* **Serialization**: the report is a multi-hundred-KB markdown string. Fine once
  per run; it is already written to disk, so the wire could carry the path.
* **Progress**: needs a streaming response. The `(phase, stats, msg)` tuple maps
  to one NDJSON line per tick with no loss.
* **Cancellation**: needs a run id + explicit cancel route. Achievable, but it
  is the part that must be built rather than wrapped, and until it is built the
  HTTP backend must not be the default.
* **Sharp edge**: `apply_overrides` mutates `deep_research` **module globals**
  in place and `restore_overrides` puts them back (`dr_settings.py`). In-process that is a per-run save/restore under the GUI's
  single-run busy gate. In a shared server process it is global mutable state
  across concurrent runs — a correctness hazard the boundary must serialise or
  scope per run.

### 1.2 The knowledge / document layer — **done**

`knowledge/knowledge_api.py` + `knowledge_client.py` + `knowledge_service.py`.
The GUI's `LibraryBuildWorker`, `RequestWorker` (RAG augmentation) and
`ScanWorker` all go through `knowledge_client.open_library()` in
`gui/gui_workers.py`. Nothing to redo. Note what that boundary *did not* move, and why the same
reasoning applies here: `map_reduce_scan` stays client-side because `map_fn` /
`reduce_fn` are **LLM callbacks**, and `research_cache.ExtractionCache` stays
in-process because a per-URL network hop defeats a cache whose purpose is
avoiding per-URL latency.

### 1.3 The memory / profile store — **partially, and low value**

`memory_center.MemoryStore` (`agent/memory_center.py`) is a file-backed store
over `MEMORY_DIR`, consumed by `gui/gui_memory_tab.py` (the Memory console);
`gui/gui_system_info_tab.py` reads the same directory. Those *browse/administer*
calls are request/response with small payloads and would cross cleanly.

But the hot path does not. `ctx.remember(...)` and `ctx.memory_text()` are
called **inside the agent turn**, from `agent/graph.py`,
`agent/graph_personality.py`, `agent/graph_compose.py`, `agent/tool_context.py`,
`agent/tools.py` and from `gui_workers.RequestWorker` (`gui/gui_workers.py`).
Putting the store behind a process boundary puts an RPC in the middle of every
graph node. `ctx.session_memory` is also an in-memory list that graph nodes read
and write directly; the file is only the persistence end of it. **Verdict: the
admin surface could move, the turn-local memory cannot, and splitting them buys
little for real risk.** Not migrated.

---

## 2. What cannot cross, and why

Each of these fails on a concrete mechanism, not on taste.

### 2.1 The window, the tabs, the workers — QObject and the event loop

`AssistantWindow` (`gui/gui.py`) is a `QMainWindow` mixing in `LayoutMixin`,
`DatabaseTabMixin`, `VoiceMixin` and others. The nine workers in `gui_workers.py` are
`QThread` subclasses communicating by `pyqtSignal`. A `QObject` has thread
affinity and no meaningful serialisation; a signal is a C++ connection, not a
message. These stay in one process, full stop. About thirty test files reference
`AssistantWindow` directly, which is a second, practical reason not to move the
*caller* of anything (see the seam rule in `gui_workers.py`'s docstring: moving
a callee is safe, moving the caller kills every `gui.X` monkeypatch).

### 2.2 The agent turn (`graph.build_graph` / `app_runtime.build_runtime`)

`build_runtime` (`core/app_runtime.py`) returns `(ctx, base_state, graph)` where:

* `ctx.models` is `Models.load()` — **Whisper and F5-TTS resident on the GPU**;
* `ctx.asr_lock` / `ctx.tts_lock` are `threading.Lock` objects shared with the
  audio path;
* `ctx.cancel_event` is a `threading.Event`;
* `graph` is a compiled LangGraph object closed over `ctx`;
* it monkeypatches `f5_tts.infer.utils_infer.transcribe` to a **closure** over
  `ctx` (`lambda path: transcribe_audio_file(ctx, path)` in `build_runtime`).

`RequestWorker` (`gui/gui_workers.py`) then calls `self.graph.invoke(state)`,
mutates `self.ctx.web_search_enabled` around the call, deep-copies
`self.base_state["messages"]`, and feeds them back through
`compact_history_if_needed(self.ctx, msgs)` (defined in `agent/graph_history.py`).

None of that survives serialisation. `ModelLoader.ready` literally emits
`(ctx, base_state, graph)` — three live Python objects — into a Qt signal that
the window stores as attributes used by every subsequent action. A boundary here
would have to reproduce the entire mutable session across the wire on every
turn. **Not movable without an architectural change that is not obviously safe.
Not attempted.**

### 2.3 TTS / ASR

`audio.transcribe_audio_array(ctx, audio)` (`voice/audio.py`) is called from
`gui_workers.RequestWorker` and `gui_common.TranscribeWorker.run`.
It takes a **numpy array** and needs `ctx.models` (GPU-resident) and
`ctx.asr_lock`. The models share one 24 GB card with the LLM and the image
stack; a second process means a second CUDA context and a second copy of the
weights, on a card that is already the binding constraint. Playback is worse
still: it is a device handle, not data.

**Verdict: cannot move.** Not because the interface is ugly — because the
resource is singular and the payload is bulk binary on the same machine.

### 2.4 Anything holding a Qt paint callback or a live signal

`gui_audio_visualizer.py` paints from the audio stream inside a Qt paint event.
`gui_log_bridge.py` funnels log records into widgets. `_ScopedCtx` consumers
(the transfer and storyboard tabs, reached by `_cancel_current`'s
`getattr(tab, "cancel")` loop) hold a token that is checked synchronously by
in-process code. All local by definition.

### 2.5 Anything identified by a local filesystem path

`OUTPUT_DIR` images, `MEMORY_DIR` profiles, the research report `path` returned
by `run_deep_research`, `db_path` for the library, mask PNGs handed to the image
workers. The GUI displays these by opening the file. A service on another host
returns a path that does not exist on the client. Same-host is fine; that is a
constraint to write down, not to discover in production.


---

## 3. Consequences

Ranked by value, highest first. The deep-research boundary follows this order;
§4 records where each step stands.

1. **Define the deep-research boundary** (`research/research_api.py`): a
   Protocol derived from the real call sites above, plus explicit notes on the
   `apply_overrides` global-state hazard and the cancellation requirement.
2. **Route consumers through an in-process client**
   (`research/research_client.py`). This is the step that actually
   de-monolithises: after it, the code behind the boundary is not constructed
   by the caller, and behaviour is unchanged. **It has value even if no HTTP
   backend ever runs.**
3. **An out-of-process backend only where step 1 proved it safe** — selected by
   env var, defaulting to in-process, with a run id so cancellation survives.
4. **A differential test** proving both backends agree, mutation-checked.

And, written down so it is not rediscovered: the agent turn, TTS/ASR, the
window itself and the turn-local memory are **deliberately not migrated**. A
clear statement of why the GUI cannot be split at those seams is worth more than
a migration that strands the Stop button.

### Test-seam hazards the boundary respects

* `tests/test_dr_output_language.py` checks that the language derivation is
  reachable from the GUI path by looking for
  `research_api.OUT_LANG_FROM_TOPIC_MARKER` in the source of `gui`,
  `gui_workers` and `research_client`, and anchors the Telegram path on the
  literal `_dr.run_deep_research(` in the **tg_bot + tg_tasks** source. Any
  re-routing must repoint those anchors at the module family keyed on a marker,
  never on a function name.
* `tests/test_gui_workers.py` installs a fake `deep_research` module in
  `sys.modules` exposing exactly `apply_overrides`, `restore_overrides`,
  `lang_of_text`, `run_deep_research`. A client must therefore resolve
  `deep_research` at **call time**, exactly as `knowledge_client` does for
  `library`, or the stub dies.
* Never import a boundary name by value on both sides of the split: that
  split-brain runs the real worker while the suite prints PASS (it caused a
  native 0xC0000409 abort once).

---

## 4. Current state of the deep-research boundary

| step | artefact | state |
|---|---|---|
| 1 | `research/research_api.py` — the Protocol | in place |
| 2 | `research/research_client.py` — in-process client; `DeepResearchWorker` goes through it | in place, default |
| 3 | `research/research_service.py` — HTTP backend + client, `RESEARCH_BACKEND=http` | in place, opt-in |
| 4 | `tests/difftest_research_backends.py` — differential, mutation-checked | in place |

Two things are worth carrying forward:

**The overrides dance lives in the boundary.** Callers do not open-code
`apply_overrides` / run / `restore_overrides`. It is one try/finally in
`research_client`, which is also the only reason the operation is safe to
expose to a backend that might run two of them — and why `research_service`
serialises runs behind `_RUN_LOCK` rather than pretending concurrent runs with
different module globals are fine.

**Cancellation was earned, not assumed.** The in-process client reports
`cancellation_mode == "shared_event"` because it hands the caller's own `ctx`
straight to the pipeline. The HTTP client reports `"run_id"`, because it has to:
the service issues a run id as the first record of the stream, holds its own
`Event`, and the client runs a watchdog thread polling the caller's real `ctx`
and firing `POST /op/cancel`. The differential suite presses Stop mid-run on
*both* backends and requires `cancelled=True` from each — and mutation 3
(making `_cancel_run` a no-op) turns that red, which is the evidence that the
check is real.

### Not routed through the boundary, with the reason

* **The agent turn, TTS/ASR, the window itself, turn-local memory** — §2 above.
* **`tools._handle_deep_research` (`agent/tools.py`), `tg_tasks`
  (`bot/tg_tasks.py`), `tg_depth._depth_eta` (`bot/tg_depth.py`).** These still
  call `deep_research` directly. They are real consumers of this boundary but
  not GUI consumers, and two suites (`tests/test_dr_output_language.py`,
  `tests/test_tg_research_depth.py`) anchor on the literal
  `_dr.run_deep_research(` in the **tg_bot + tg_tasks** source. Routing them
  means repointing those anchors at the module family on a marker, as was done
  for the GUI anchor, and doing that as its own change rather than folding it
  into this one.
* **The Research tab's Manual Control panel.** `gui/gui_manual_panel.py` reads
  the `deep_research.DR_*` globals directly to seed its widgets. `knob_spec()`
  and `knob_defaults()` exist on the contract so this *can* migrate, but doing
  it is mechanical churn across two long functions with source-grep risk and no
  boundary value: the knob **values** already cross the boundary, as the
  `overrides` argument to `run()`. Left alone on purpose.
