# Video generation — MiniMax H3

Local text/image/video → **video with native stereo audio**, through ComfyUI.

H3 is omni-modal: one packed latent carries the video *and* its soundtrack, so a
clip comes back with sound already muxed. There is no separate audio pass, and
"describe what it sounds like" is a real part of the prompt.

## What you need

| Piece | Version / file | Notes |
|---|---|---|
| ComfyUI | **≥ 0.30.0** | the `MiniMaxH3*` nodes landed here |
| ComfyUI-GGUF | installed | provides `UnetLoaderGGUF` / `CLIPLoaderGGUF` |
| `models/unet/` | `MiniMax-H3-FL2VA-Q4_K_M.gguf` (19.9 GB) | text / 1 image / 2 images |
| `models/unet/` | `MiniMax-H3-Ref2VA-Q4_K_M.gguf` (19.9 GB) | many images, video, audio |
| `models/text_encoders/` | `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` (14.6 GB) | H3-Encoder |
| `models/vae/` | `minimax_h3_video_vae_fp16.safetensors` (5.2 GB) | H3-VisualVAE |
| `models/vae/` | `minimax_h3_audio_vae_fp32.safetensors` (0.6 GB) | H3-AudioVAE |

```bash
venv/Scripts/python.exe scripts/fetch_h3_weights.py
```

Resumable — re-running skips whatever is already complete. `--check` reports
without downloading.

### Why GGUF and not the official packaging

MiniMax/Comfy-Org ship `fp8_scaled` and `nvfp4_awq` builds. Neither suits this
box: fp8 needs sm_89+ (Ada) for native support and gets upcast on Ampere, and
nvfp4 is Blackwell-only. GGUF Q4_K_M runs natively on a 3090 through
ComfyUI-GGUF. If Q4 will not fit, `Q3_K_M` (15.6 GB) is the next rung down.

## The four modes

Chosen automatically from what is attached — callers never pick a checkpoint.

| attached | mode | what the inputs MEAN |
|---|---|---|
| nothing | `t2va` | pure text→video |
| 1 image | `i2va` | that image is **frame 0** |
| 2 images | `flf2va` | first frame and last frame |
| 3–9 images, or any video/audio | `ref2va` | references, not literal frames |

The distinction matters: a keyframe *is* a frame of the output, a reference is
something the model is reminded of. One picture as a keyframe animates that exact
picture; the same picture as a reference produces a new scene that resembles it.

## Hard constraints (from the node, not preference)

* **24 fps**, frame count must satisfy `n % 17 == 5`. `video.snap_frames()`
  rounds onto that grid. 124 frames ≈ 5.2 s; MiniMax trained ~124–362 (5–15 s).
* **768 short edge**, area capped at `768*1344`, every axis a multiple of 32
  (`video.fit_canvas()`, a port of the node's own `adapt_canvas`).
* **cfg = 1.0.** The released weights are CFG-distilled, so guidance is baked in
  and a negative prompt is mathematically inert. Raising cfg does not strengthen
  the prompt, it doubles the cost of every step.

## The two numbering systems — read this before touching references

This is the easiest thing to get wrong, and it is silent when you do.

* **On the wire**, reference slots are **0-based AND dotted**:
  `ref_images.ref_image_0` … `ref_images.ref_image_8`. ComfyUI's
  `COMFY_AUTOGROW_V3` advertises a *container* (`ref_images`) in `/object_info`;
  `comfy_api/latest/_io.py` builds the slot names as
  `names = [f"{prefix}{i}" for i in range(max)]`, and `parse_class_inputs()`
  then prefixes each one with the input's own id —
  `finalize_prefix(["ref_images"], "ref_image_0")`. At execution
  `build_nested_inputs()` splits on `.` to rebuild the dict that
  `execute(ref_images={...})` takes.

  A 1-based graph drops the first reference and asks for a slot 9 that does not
  exist. A **bare** `ref_image_0` is worse, because it looks like it works: the
  name is not in the schema at all, so validation ignores it as an unknown extra
  and `POST /prompt` returns 200 — then execution passes it through as a stray
  kwarg and dies with `execute() got an unexpected keyword argument
  'ref_image_0'`, *after* the encoder and both VAEs have loaded. **Schema
  acceptance proves nothing about this node; only a completed reference render
  does.** (Same failure class as `SaveVideo.codec` taking a dict, not a string.)
* **In the prompt**, the same references are **1-based per type**: `<Picture 1>`,
  `<Video 1>`, `<Audio 1>` (`video.reference_tags()`).

Both are correct; they are different namespaces. The prompt must use the tags, or
the model cannot tell which reference a sentence refers to.

Order is fixed: images, then videos (each soundtrack's `<Audio j>` label
immediately before its `<Video k>`), then standalone audio.

## Using it

**Chat, either surface** — just ask. The agent calls `generate_video`, which uses
the pictures already in the conversation unless told otherwise:

> "animate this" · "make a video from these three photos" ·
> "нарисуй видео с лисой в траве" · "use the camera move from that clip"

**Directly:**

```python
import video
res = video.generate_video(ctx, "a red fox steps out of tall grass, slow push-in; "
                                "wind and one distant bird, no music",
                           seconds=5, aspect="16:9")
# {'path': ..., 'mode': 't2va', 'seconds': 5.17, 'width': 1344, 'height': 768, ...}
```

## Prompting

Write a **shot**, not an image caption: what is in frame, what *moves*, what the
camera does over those seconds, then the soundscape.

> a red fox steps out of tall grass and turns toward the camera, slow push-in,
> late afternoon light; wind in the grass, one distant bird, no music

"a fox" gets you a near-still frame and arbitrary audio.

## Cost and limits

A 33B DiT plus a 32B encoder against 24 GB of VRAM means constant offload. The
components load **sequentially** — encoder first, then evicted, then the DiT —
which is what makes ~40 GB of weights fit at all. Expect **minutes per clip**,
scaling with duration.

Guards that exist because of it:

* `VIDEO_JOB_TIMEOUT` (default 5400 s) — the job is abandoned and reported
  honestly rather than hanging a chat turn forever.
* A clip costs the **whole turn's render budget**, so the agent cannot fire two.
* Telegram: clips over 50 MB are re-encoded, then fall back to a document. A
  delivery failure is surfaced — the bot never narrates a video it did not send.

## Configuration

All in [config.py](../config.py) under "Video generation":
`VIDEO_ENGINE`, `VIDEO_DEFAULT_FRAMES`, `VIDEO_MAX_FRAMES`, `VIDEO_SHORT_EDGE`,
`VIDEO_MAX_PIXELS`, `VIDEO_STEPS`, `VIDEO_CFG`, `VIDEO_SHIFT_VIDEO`,
`VIDEO_SHIFT_AUDIO`, `VIDEO_JOB_TIMEOUT`, `VIDEO_TG_MAX_BYTES`.

## Checking it works

```bash
venv/Scripts/python.exe scripts/smoke_video.py            # schemas only, fast
venv/Scripts/python.exe scripts/smoke_video.py --render   # a real clip, slow
```

Staged deliberately, so a failure says *which* assumption broke: server version →
weights visible to ComfyUI → graphs validate against live schemas → real render.

Offline suites (no GPU, no ComfyUI, no weights):

```bash
venv/Scripts/python.exe tests/test_video_generation.py    # 96 checks
venv/Scripts/python.exe tests/test_tg_video_delivery.py   # 24 checks
```

## ComfyUI version note

0.30.0 lives at `ComfyUI-0.30.0` beside the old `ComfyUI-0.28.0`, sharing one
venv, `models/` and `custom_nodes/` via `--base-directory`.
`scripts/launch_all.py` picks the **highest** version present, so rolling back is
just removing the newer directory or setting `COMFY_SRC`. 0.30.0 keeps Ideogram 4
(`CLIPLoader` type `ideogram4`), so the image pipeline is unaffected.

Upgrading bumped five pinned packages: `comfyui-frontend-package` 1.47.11,
`comfyui-workflow-templates` 0.11.27, `comfyui-embedded-docs` 0.5.9,
`comfy-kitchen` 0.2.26, `comfy-aimdo` 0.4.11. **torch stays 2.10.0+cu130** — check
it after any ComfyUI dependency change, since a pip resolution can quietly swap
the CUDA build for a CPU one.
