# Explicit upscale modes & the task queue

## Upscale vs Restore — two explicit operations, never auto-switched

These are deliberately kept distinct (the user's expectation differs):

| mode | what it does | face behaviour |
|---|---|---|
| **Upscale** (face-safe) | RealESRGAN on body/clothing/background, then the **real face pasted back** from a faithful Lanczos enlargement, seam-blended | identity preserved exactly |
| **Restore** (Upscale + Face Restore) | face-detailer **reconstructs** the face, then ESRGAN upscale | face intentionally enhanced (may change slightly) |

The system never silently converts one into the other when a face is detected.

### How each is reached
- **Routing** (`image.classify_edit_intent`): plain "upscale / up-res / 4x" → `upscale`
  (face-safe). An explicit combined phrasing — "upscale and restore the face",
  "restore the face", "enhance the face", "восстанови лицо" — matches `_FACE_RESTORE_RE`
  and routes to `restore`. This rule has **precedence over multi-op splitting**, so
  "upscale and restore the face" is one combined op, not two. A genuine two-op request
  ("upscale and remove the background") still splits to `multi_op`. (13/13 routing cases.)
- **Tool** (`redraw_image`): explicit `mode` of `upscale` or `restore` (in addition to
  `enhance` / `redraw`). `upscale` → `upscale_image_with_comfy(protect_face=True)`;
  `restore` → `restore_image_with_comfy`. A `4x`/`2x` token in the instructions sets the
  scale.

### Face-safe upscale internals (`_preserve_face_after_upscale`)
1. detect the face on the **original** (`identity_metrics.face_box`, pad 0.30);
2. paste the face back from a **Lanczos** enlargement of the original (invents no detail,
   identity-exact) through a feathered ellipse;
3. seam-suppress with `_blend_region` (continuation path → Poisson).
Falls through to the plain ESRGAN result when there is no face (landscapes/objects) or on
any error. `restore` calls the upscaler with `protect_face=False` so it does not overwrite
the detailer's enhanced face. Validated: face pixel 7× closer to the faithful original
than to a deliberately "ruined" ESRGAN stand-in; real-photo end-to-end produces a 2× image
with the face composited.

## Task queue (GUI)

The single-slot pre-fill was upgraded to a real queue (`AssistantWindow`):

- **Add tasks** any time — type and press **＋⏳**, or press Enter/Send while busy (busy
  submissions enqueue instead of dropping).
- **Reorder** (▲/▼), **remove** (✕), **clear**, and **pause/resume** (⏸/▶) from the queue
  panel above the input.
- **Auto-drain**: when the app is idle and not paused, the next task (FIFO) runs; the next
  is popped after every worker finishes (hooked into `_set_busy(False)`, so it chains
  across request/redraw/research/model-switch workers).
- **Execution-state tracking**: each task carries a `status` — `pending` → `running` →
  `done`/`failed`. The running task stays visible (▶), is protected from remove/reorder,
  and is finalized (✓/✗) when its worker finishes; `_completed_count`/`_failed_count`
  summarise history in the panel tooltip. Failure is detected via `_on_failed`
  (`_turn_had_error`).
- State: `self._task_queue` (list of `{"text", "status"}`), `self._queue_paused`,
  `self._running_task`.

Validated headless (offscreen Qt, real widgets) — `tests/test_task_queue.py`, **12/12**:
enqueue-while-busy, pending status, dispatch+mark-running, running-not-removable,
pending-only reorder, done/✓ counting, failed/✗ counting, empty-after-drain, pause holds,
resume dispatches, clear keeps the running task. The compositor invariants have their own
suite — `tests/test_seam_compositor.py` (**21/21**).

### Example
While an image renders, stage: `upscale 4x` → `caption it` → `make the shirt blue`. They
run one after another as the assistant frees up; pause to hold them, reorder to change
priority, ✕ to drop one.
