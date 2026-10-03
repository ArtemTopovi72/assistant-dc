# Resolution-Drift Audit & Fix

Goal: an edit on a 3243×4864 image must return 3243×4864 — same person, same
framing — with no hidden downscale and **no blurry low-res regeneration**.

## 1. Resolution flow per pipeline (image.py)

Every generation pipeline has a `_enforce_output_size(wf, sw, sh)` guard that
inserts a lanczos `ImageScale` before every `SaveImage`, so the **saved dimensions**
always equal the source. The remaining problem was never the *number* — it was that
some pipelines reach the right number by **upscaling a low-res generation** (soft
output), and the localized-edit path **OOM'd at high res and fell back** to exactly
that soft path.

| Pipeline | fn | Internal working size | Final dims | Sharpness risk |
|---|---|---|---|---|
| FireRed whole-frame edit | `edit_image_with_firered` | node 191 `ImageScaleToTotalPixels` → **1.0 MP** | ✅ guard restores src | ⚠ **upscaled from 1 MP = soft** |
| Redraw | `redraw_image_with_comfy` | 1024 long-side | ✅ guard | ⚠ upscaled from 1024 |
| BG replace | `replace_background_with_comfy` | cap **1536**, ÷16 | ✅ guard | subject re-composited then rescaled |
| BG remove | `remove_background_with_comfy` | native matte | ✅ native | none |
| Object/person removal | `remove_object_with_comfy` | native inpaint + composite-back | ✅ native | none (masked-region only) |
| Relight | `relight_image_with_comfy` | IC-Light SD1.5 band | ✅ guard | ⚠ upscaled |
| Restore / upscale | ESRGAN 4× + `ImageScaleBy` | intentional | ✅ by design | n/a |
| **Localized edit (in-graph)** | `edit_region_contained_with_comfy` | **native** VAE-encode on full `L` | ✅ native by construction | none — **but OOMs >~4 MP** |
| **Localized edit (crop)** | `edit_region_contained_cropped` *(new)* | only the **mask-bbox tile**, ≤1280 | ✅ native canvas | none — tile near-native |
| Multi-op chain | `edit_image_*` chain | per-step | ✅ end-of-chain PIL restore | per-step |

## 2. Unintended-resolution-change table (root causes)

| Symptom | Root cause | Location | Status |
|---|---|---|---|
| output saved at 1 MP ("tiny/blurry") | `ImageScaleToTotalPixels` in FireRed template, no restore | workflow node 191 | **fixed** — `_enforce_output_size` (image.py:821) |
| bg-replace silently 1536-capped | `cap = 1536` working size | image.py:884 | **fixed** — guard restores src (image.py:914) |
| redraw 1024 long-side leaked to output | internal resize | image.py:588 | **fixed** — guard (image.py:592) |
| **hat edit on 15.77 MP came back blurry** | in-graph contained inpaint **OOMs**, falls back to whole-frame FireRed (1 MP → upscale) | image.py contained path + tools.py routing | **fixed** — crop-based path (below) |
| aspect drift from ÷16 latent rounding | latent snapping | bg/contained graphs | **fixed** — guard rescales to exact src |

## 3. The hard rule, enforced

For localized edits the pipeline is now strictly:

```
original (native HxW)
  → Florence-2 mask of the named region (native res)
  → [≤4 MP]  inpaint full frame, composite back over original     (native, no downscale)
  → [>4 MP]  crop mask-bbox tile → inpaint tile ≤1280 → paste back into full-res original
  → SaveImage  (dims == original, guard is a no-op safety net)
```

The `>4 MP` branch (`CONTAINED_FULLRES_MP`, image.py) is the key addition: it
**never downscales the whole canvas**. It bounds only the *tile* for VRAM, inpaints
it, and pastes it back through a feathered mask, so:

- output dimensions == source (native),
- every pixel outside the region is byte-for-byte the original (identity + framing),
- the edited region is sharp (the tile is a small fraction at/near native res, not
  the whole 15.77 MP image squeezed to 1 MP),
- no OOM → **no fallback to the blurry whole-frame path**.

## 4. Validation (real images, measured)

| Image | MP | Pipeline | Output dims | Drift | Identity cosine | Face changed | Overall changed | Time |
|---|---|---|---|---|---|---|---|---|
| pozner_large 1800×1200 | 2.16 | FireRed/bg/relight | = src | **0.00 %** | — | — | — | — |
| portrait_odd 667×1153 (non-÷16) | 0.77 | contained in-graph | = src | **0.00 %** | — | — | — | — |
| pozner jacket | 2.16 | contained in-graph | = src | 0 % | **0.982** | byte-exact | — | — |
| **sydney_native 3243×4864** | **15.77** | **crop-based** | **3243×4864** | **0.00 %** | **0.999** | **0.29 %** | **1.03 %** | **63 s** |

The 15.77 MP case is the headline: the user's exact failure class (a >10 MP
portrait, localized edit). Before the fix the full-res path **timed out at 1900 s
and the FireRed fallback timed out again** (3801 s total, no output — `becvdfkz8`).
After: native resolution, same person (0.999), face untouched (0.29 %), edit
localized (1.03 % of the frame), in 63 s.

`tests/validate_hires_contained.py <image> <region> <prompt>` reports all of the
above for any image.

## 5. Identity protection (added)

Resolution alone is not enough: Florence masks for face-adjacent nouns (esp.
"hair") bleed onto the forehead/eyes, which silently regenerates the face (first
15.77 MP run: face became undetectable). The crop path now **subtracts the detected
face box** (`identity_metrics.face_box`) from the edit mask for any non-facial
region (`_FACIAL_FEATURE_RE` gates this; "hair"/"hat"/"head" are NOT facial
features, so they keep the face protected). Identity is preserved **by
construction**, taking the 15.77 MP hair edit from face-undetectable → cosine 0.999.

## Files
`image.py` (`edit_region_contained_cropped`, `_florence_mask_file`,
`_inpaint_crop_graph`, `CONTAINED_FULLRES_MP`, `_FACIAL_FEATURE_RE`, Florence-proxy
+ face protection, routing in `edit_region_contained_with_comfy`) ·
`identity_metrics.py` (`face_box`) · `tests/validate_hires_contained.py`.
