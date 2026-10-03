# Seam-suppressing compositor

When a region is re-rendered (clothing change, face guard, contained edit) and pasted
back over the original, the boundary can show a visible **seam**: a low-frequency colour
/ illumination step, or a high-frequency texture mismatch. A simple alpha feather blends
*opacity*, not *tone or texture*, so it cannot hide either. This module
(`image.py`) layers three techniques, each routed to the regime where it actually helps,
proven by `tests/bench_seam.py`.

## Functions

| function | role |
|---|---|
| `_band_masks(m, w)` | inside/outside boundary rings (cv2 morphology) |
| `_is_continuation(bg, patch, m)` | LAB ΔE between patch boundary and surroundings → is the patch a *continuation* of its surroundings (skin/wall) or *distinct content* (a recoloured garment)? |
| `_color_harmonize(bg, patch, m)` | Reinhard LAB mean/std transfer toward the original's outside ring, **boundary-weighted** (falls off to ~0 by 25 % inward) and clipped, so a real recolour's core is never washed |
| `_seamless_clone_arr(bg, patch, m)` | array Poisson (`cv2.seamlessClone`, NORMAL_CLONE) — membrane-levels low frequencies, preserves gradients |
| `_laplacian_blend(bg, patch, α)` | multi-band (Laplacian-pyramid) blend: low freqs cross the boundary gradually, high freqs sharply |
| `_blend_region(bg, patch, m)` | **unified entry point** — routes by `_is_continuation` |
| `_seam_blend_tile(...)` | back-compat Poisson-only wrapper |

## Routing (the key design decision)

The benchmark showed Poisson is the single most effective stage, and that stacking
harmonize/multi-band *on top of* Poisson slightly **hurts** a continuation seam (they
perturb the gradients Poisson preserves). So `_blend_region` picks one path:

- **Continuation** (skin/wall; ΔE ≤ 22): clean **Poisson** + region feather. No harmonize,
  no multi-band.
- **Distinct content** (a recoloured garment): Poisson is *skipped* so the edit isn't
  pulled toward its surroundings; instead **boundary-weighted harmonize + Laplacian
  multi-band** smooth the seam while leaving the interior intact.

This is the evidence-based refinement of the original "harmonize → Poisson → multi-band"
stack: all three are implemented, but each is applied only where it measurably wins.

## Benchmark (16 real images, `tests/bench_seam.py`)

Seam quality is measured in a thin band straddling the boundary, against the original
photo as the seamless reference.

**Continuation seam** (the user's reported neck-crease):

| method | grad_excess ↓ | color_disc ↓ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| naive paste | 1.94 | 12.28 | 0.809 | 0.261 |
| feather only | −5.44 | 3.15 | 0.789 | 0.256 |
| **Poisson (= full)** | **−0.96** | **3.31** | **0.861** | **0.180** |
| multiband only | −5.33 | 3.21 | 0.795 | 0.246 |

Poisson lifts SSIM 0.809 → 0.861 and cuts LPIPS 0.261 → 0.180 and colour discontinuity
12.3 → 3.3. Feather/multiband alone *over-smooth* (negative grad_excess, SSIM below naive).

**Distinct recolour** (harmonize + multi-band path), no seamless reference:

| metric | naive paste | full pipeline |
|---|---|---|
| boundary gradient ridge ↓ | 52.78 | **37.32** (−29 %) |
| interior colour drift ΔE ↓ | 0.00 | **9.05** |

The seam is smoothed 29 % while the recoloured interior is preserved (boundary-weighted
harmonize dropped interior drift from **16.9 → ~9 ΔE** vs the naive uniform shift).

## Robustness: multi-band geometry cap

A Laplacian blend mixes background into the patch over a band ~`2**levels` px wide. If
that exceeds the region's interior depth, the coarsest pyramid level pulls the **deep
interior** toward the background. The regression suite caught this: a recoloured core was
washed ~36 % at `levels=5` on a small region. Fix: `_laplacian_blend` **caps pyramid depth
by geometry** — `levels ≤ floor(log2(reach/3))` where `reach` is the max interior distance
transform — so the blend stays within the outer third of the region. Large regions still
get deep pyramids (smooth low-freq); small ones don't bleed inward. After the cap the
synthetic core wash dropped to ~0 (interior ΔE 15.4 → 0.5) for a +5 ridge cost on the
distinct path (still −21 % vs naive). Verified by `tests/test_seam_compositor.py` (21/21)
and the benchmark before/after delta.

## Regression tests & reproducibility

- `tests/test_seam_compositor.py` — 21 deterministic invariant checks (band masks,
  continuation routing, harmonize interior/outside, Poisson guards/locality, Laplacian
  alpha extremes, blend routing, determinism, face-guard passthrough). No GPU/ComfyUI.
- `tests/bench_seam.py --save-baseline` stores `results_baseline.json`; every run prints
  a delta vs the baseline so regressions are visible. Seed-fixed → reproducible.

## Where it is used

- `edit_region_contained_via_firered` whole-region composite-back (clothing/hair edits).
- `_preserve_face_after_upscale` — blends the faithful face back over the ESRGAN result.

## Limitations / future work

- The benchmark's patches are derived from the original photo, so Poisson (gradient-
  preserving) is somewhat favoured; a held-out set of *true* independent re-renders paired
  with hand-cleaned seamless targets would measure the texture regime more faithfully.
- `_is_continuation` uses a fixed LAB ΔE threshold (22); a learned or per-image adaptive
  threshold could route borderline cases better.
- LPIPS uses AlexNet; a higher-capacity backbone would be more discriminative but slower.
