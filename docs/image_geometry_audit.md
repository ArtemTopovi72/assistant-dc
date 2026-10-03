# Image Geometry-Preservation Forensic Audit

**Scope:** every editing pipeline in `image.py` + the workflow JSON templates.
**Method:** static dimension-flow trace of each ComfyUI graph (node ids + source
lines). No fixes applied — this is the evidence pass requested ("audit first").

Runtime subject-bounding-box measurement requires a live ComfyUI+GPU run; the
resize transforms below are **deterministic from code**, so the pixel-dimension
drift is computed exactly. A worked portrait example is given at the end.

---

## TL;DR — where geometry changes

| Pipeline | Output resolution rule | Aspect | Subject scale vs source | Verdict |
|---|---|---|---|---|
| background_remove | = source | exact | exact | ✅ SAFE |
| object_remove (LaMa) | = source (`destination=L`) | exact | exact outside hole | ✅ SAFE |
| object_remove (the old model) | = source (`destination=L`) | exact | exact outside hole | ✅ SAFE |
| person_remove | = source | exact | exact | ✅ SAFE |
| object_insert (region/anchor) | = source (`destination=L`) | exact | exact outside mask | ✅ SAFE |
| outpaint | = padded canvas; original center untouched | exact | exact (original unscaled) | ✅ SAFE |
| upscale | source × factor | exact | enlarged **by request** | ✅ SAFE (by design) |
| restore | face-detail (≈source) → upscale ×2 | exact | ×2 **by request** | ✅ SAFE (by design) |
| **background_replace** | ≤1536 long side, each axis floored to ÷16 | minor drift | **downscaled if src>1536** | ⚠️ RESOLUTION LOSS |
| **face/clothing/product/subject/style edit** | **1.0 MP** (FireRed) | preserved | **downscaled to ~1 MP + full re-render** | ❌ DOWNSCALE+DRIFT |
| **object_insert (no region/anchor)** | **1.0 MP** (FireRed fallback) | preserved | downscaled to ~1 MP | ❌ DOWNSCALE |
| **redraw (GUI button)** | **1024 long side** | preserved | downscaled to 1024 | ❌ DOWNSCALE |
| **relight** | SD1.5 working res, full re-synth, no restore | possible letterbox | **likely large downscale** | ❌ DOWNSCALE (confirm at runtime) |
| **multi_op chain** | min of every step's rule | compounds | **cumulative shrink** | ❌ COMPOUNDING |

---

## Exact root causes (node / line)

### RC-1 — FireRed forces 1.0 megapixel  ★ highest impact
- **File/stage:** `workflow_firered_edit.json`, node **`191 ImageScaleToTotalPixels` `megapixels=1.0`**, sitting between `143 LoadImage` and `185 VAEEncode`.
- **Reached by:** `edit_image_with_firered` ([image.py:687](../image.py)) → used by `route_edit_request` for `face_edit`, `clothing_edit`, `product_edit`, `subject_edit`, `style_transfer` ([image.py:1467](../image.py)), anchorless `object_insert` ([image.py:1433](../image.py)), and `inpaint_region_with_comfy` ([image.py:1543](../image.py)).
- **What changes:** input is rescaled so total pixels ≈ 1.0 MP **before** encoding; the whole frame is then re-rendered (Qwen-Image-Edit, full denoise) and saved at that ~1 MP size. No restore to source resolution.
- **Necessary?** Qwen-Image-Edit has a preferred ~1 MP operating band, so *some* normalization is reasonable — but hard-capping at 1.0 MP with **no upscale-back to the source size** is the cause of "blurry / small when viewed at original size." Aspect ratio is preserved (uniform scale), so this is resolution loss, not stretch.

### RC-2 — relight has no source-resolution anchor  ★ prime "tiny" suspect
- **File/stage:** `relight_image_with_comfy` ([image.py:1168](../image.py)). Graph: `L → easy icLightApply (remove_bg, SD1.5) → VAEEncode → KSampler denoise=1.0 → VAEDecode → Save`.
- **What changes:** output dimensions = whatever `easy icLightApply` (node `IC`) emits as its lighting/foreground image. IC-Light is an **SD1.5** model (`realisticVisionV51…`, [image.py:1159](../image.py)); SD1.5 works at ~512–768 px. There is **no** `destination=L` composite and **no** scale-back, so the result is delivered at SD1.5 working resolution. If `easy icLightApply` square-pads/letterboxes to a 1:1 working tile, the subject also shrinks **relative to the frame** (the classic "tiny subject"). Full denoise=1.0 also re-synthesizes the subject.
- **Confidence:** node-level resize inside `easy icLightApply` is in third-party code; the *absence* of any source-size anchor in our graph is certain from the lines above. Exact output dims should be confirmed with one live run.

### RC-3 — background_replace caps at 1536 and floors to ÷16
- **File/stage:** [image.py:785-788](../image.py):
  ```
  cap = 1536
  scale = min(1.0, cap / max(sw, sh))
  w = max(64, (int(sw*scale)//16)*16)
  h = max(64, (int(sh*scale)//16)*16)
  ```
  Background is generated at `w×h` ([image.py:740](../image.py) `EmptySD3LatentImage`); the original subject is composited as `source=L` with **`resize_source=True`** onto `destination=57:8` (the generated bg) ([image.py:806-807](../image.py)).
- **What changes:** for any source with a long side > 1536, the entire output (bg **and** the resize_source'd subject) is downscaled. The independent `//16` floor on each axis perturbs the aspect ratio slightly (typically < 2%; zero when both axes are already ÷16). Subject keeps its in-frame fraction (bg aspect ≈ source aspect), so this is mainly resolution loss.

### RC-4 — redraw forces 1024 long side
- **File/stage:** redraw workflow, node **`652:78 ResizeImageMaskNode` `resize_type='scale longer dimension', longer_size=1024`**; reached by `redraw_image_with_comfy` ([image.py:449](../image.py)) (GUI "redraw / enhance whole frame" button).
- **What changes:** output long side = 1024, full-frame denoise=1 regeneration. Aspect preserved; resolution lost for any image > 1024 long side.

### RC-5 — multi_op compounds every downscale
- **File/stage:** `_orchestrate_multi_op` ([image.py:1471](../image.py)) threads each step's output into the next ([image.py:1501-1506](../image.py)).
- **What changes:** geometry of the chain = the **minimum-resolution rule among its steps applied in sequence**. A chain that includes a FireRed edit, a background_replace, or (worst) a relight ratchets the resolution **down at each such step and never back up** (except a trailing `upscale`, which only interpolates the already-lost detail). This is the strongest driver of the "tiny blurry dwarf" end-state.

---

## Dimension-flow per pipeline (traced)

Notation: `S=source(sw×sh)`, `→` = node output.

- **background_remove:** `S → easy imageRemBg(matte) → Save`. All at `S`. ✅
- **object_remove (LaMa):** `S → Florence2(mask@S) → GrowMask@S → LamaRemover@S → CompositeMasked(dest=S, src=LaMa, resize_source, mask@S) → Save`. Output `S`; outside-mask pixels byte-exact. ✅
- **object_remove (the old model):** `S → mask@S → VAEEncodeForInpaint@S → KSampler → VAEDecode(~S) → CompositeMasked(dest=S, resize_source) → Save`. Output `S`. ✅
- **object_insert (region/anchor):** identical composite-back shape, `dest=S`. Output `S`. ✅
- **outpaint:** `S → ImagePadForOutpaint(+L/T/R/B) = P(padded) → VAEEncodeForInpaint@P → KSampler → VAEDecode(~P) → CompositeMasked(dest=P, mask=pad) → Save`. Output `P`; original center kept at exact scale. ✅
- **upscale:** `S → UpscaleModel ×4 → [ImageScaleBy factor/4] → Save`. Output `S×factor`. Uniform. ✅ (intentional)
- **restore:** `enhance_faces (DetailerForEach, crops re-rendered, canvas ≈ S) → upscale ×2`. Output `≈S×2`. ✅ (intentional)
- **background_replace:** `S → bg gen @ w×h (RC-3) ; matte @ S → CompositeMasked(dest=bg w×h, src=S, resize_source) → Save`. Output `w×h ≤ 1536`. ⚠️
- **FireRed family:** `S → ImageScaleToTotalPixels 1.0MP (RC-1) → VAEEncode → KSampler(full) → VAEDecode → Save`. Output `~1.0 MP`. ❌
- **redraw:** `S → ResizeImageMaskNode long=1024 (RC-4) → controlnet redraw → Save`. Output `long=1024`. ❌
- **relight:** `S → easy icLightApply (SD1.5, RC-2) → VAEEncode → KSampler(denoise 1.0) → VAEDecode → Save`. Output ≈ SD1.5 res. ❌

---

## Worked subject-scale drift — portrait 2048×3072 (6.29 MP)

Uniform scales preserve the subject's **fraction of the frame**; the loss is in
absolute pixels (→ blur at original size). Relight is the exception if it letterboxes.

| Pipeline | Output px | Long-side Δ | Area Δ | Subject-in-frame |
|---|---|---|---|---|
| object_remove / insert / outpaint(pad only) / bg_remove | 2048×3072 | 0% | 0% | unchanged ✅ |
| background_replace | 1024×1536 | −50% | −75% | ~unchanged |
| FireRed edit (face/clothing/style/subject/product) | 816×1225 | −60% | −84% | ~unchanged, blurry |
| redraw | 1024×1536 (long 1024 on portrait → 683×1024) | −67% | −89% | ~unchanged, blurry |
| relight (SD1.5 ≈768 long) | 512×768 | −75% | −94% | **shrinks if square-padded** |
| multi_op: remove→bg_replace→relight→upscale×2 | 1024×1536 *(detail ≈0.39 MP)* | net −50% px, detail −94% | — | tiny + blurry |

---

## Ranked fix plan (proposed — not yet applied)

1. **RC-1 FireRed restore-to-source (highest impact, most categories).**
   After node 190 `VAEDecode`, add an `ImageScale`/`ResizeImageMaskNode` back to
   the *original* `sw×sh` (lanczos), or raise `megapixels` to match source when
   source ≤ a VRAM-safe cap and upscale-back otherwise. Keeps Qwen's working band
   but returns the user's resolution.
2. **RC-2 relight composite-back at source resolution.** Capture `sw×sh`, and
   after `D VAEDecode` resize to source; better, matte the relit subject and
   `ImageCompositeMasked` over the *original* full-res frame so only illumination
   changes and pixel geometry is exact. Confirm `easy icLightApply` output dims
   with one instrumented run first.
3. **RC-5 multi_op resolution guard.** Record the input `sw×sh` once at the start
   and resize the final result back to it (or carry a "target size" through the
   chain), so chained downscales can't compound.
4. **RC-3 background_replace cap.** Raise/parameterize the 1536 cap and replace the
   independent `//16` floors with an aspect-locked snap (scale, then pad/crop the
   bg to the matte's exact `S`), so output = source resolution and aspect is exact.
5. **RC-4 redraw long-side.** Raise `longer_size` to the source long side (VRAM
   permitting) or upscale the redraw back to source.

**Preservation strategy (target invariant):** every pipeline records `S=(sw,sh)`
at entry and guarantees `output==S` unless the user explicitly asked to resize
(`upscale`, `outpaint`, `restore`). Composite-based edits must use
`destination = original-full-res` (as remove/insert already do) so untouched
pixels stay byte-exact; generation-based edits must end with an explicit
resize/upscale-back to `S`.
