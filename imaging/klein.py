"""FLUX.3 Image Klein: the one picture engine (drawing and editing).

Klein reads a caption paragraph that cites element ids in angle brackets, followed by
a JSON table of those elements with [y_min, x_min, y_max, x_max] boxes on a 0-1000
grid -- the same grid and box order as our layout planner (ideogram_layout), so a
planned layout converts mechanically (`caption_to_prompt`). Russian lettering is exact
only with that table; a bare prose prompt writes gibberish (bench/klein_vs_ideogram.py).

Graphs are built here, not loaded from JSON: the distilled model samples 16 steps at
cfg 1, and an edit adds a reference latent plus the Consistency LoRA.
"""
import contextlib
import contextvars
import json
import re
from typing import Optional

from config import (KLEIN_CLIP, KLEIN_EDIT_LORA, KLEIN_EDIT_LORA_STRENGTH, KLEIN_STEPS,
                    KLEIN_UNET, KLEIN_VAE)

# Node ids, so callers patch a graph by role.
N_UNET, N_CLIP, N_VAE, N_POS, N_NEG = "1", "2", "3", "4", "5"
N_LATENT, N_SHIFT, N_SAMPLER, N_DECODE, N_SAVE = "6", "7", "8", "9", "10"
N_LOAD, N_SCALE, N_SIZE, N_REF, N_LORA = "11", "12", "13", "14", "20"

_MU = 2.77259        # log(16): the template derives max_shift/base_shift from alpha=16

_POS_V = ("upper", "middle", "lower")
_POS_H = ("left", "centre", "right")


def _position(box) -> str:
    y = (box[0] + box[2]) / 2000.0
    x = (box[1] + box[3]) / 2000.0
    return f"{_POS_V[0 if y < .34 else 2 if y > .66 else 1]} {_POS_H[0 if x < .34 else 2 if x > .66 else 1]}"


def _noun(desc: str) -> str:
    """Class noun for an element id: the head word of 'a X of Y' / the first word."""
    words = re.findall(r"[a-z]+", desc.lower())
    head = desc.lower().split(" of ")[0] if " of " in desc.lower() else desc.lower()
    cand = re.findall(r"[a-z]+", head)
    return (cand[-1] if cand else (words[0] if words else "")) or "object"


_MIN_SPAN = 60       # a box thinner than this on the 0-1000 grid often draws nothing (BFL tests)
_TEXT_PAD = 8        # BFL annotator: 5-15 units of margin around a text block


def _fit_box(box, is_text: bool):
    """Clamp to the grid, pad text boxes, widen tiny ones around their centre -> (box, widened)."""
    y0, x0, y1, x1 = (max(0, min(1000, int(round(float(v))))) for v in box)
    if is_text:
        y0, x0, y1, x1 = max(0, y0 - _TEXT_PAD), max(0, x0 - _TEXT_PAD), min(1000, y1 + _TEXT_PAD), min(1000, x1 + _TEXT_PAD)
    widened = False
    if y1 - y0 < _MIN_SPAN:
        c = (y0 + y1) // 2
        y0, y1, widened = max(0, min(c - _MIN_SPAN // 2, 1000 - _MIN_SPAN)), 0, True
        y1 = y0 + _MIN_SPAN
    if x1 - x0 < _MIN_SPAN:
        c = (x0 + x1) // 2
        x0, x1, widened = max(0, min(c - _MIN_SPAN // 2, 1000 - _MIN_SPAN)), 0, True
        x1 = x0 + _MIN_SPAN
    return [y0, x0, y1, x1], widened


def caption_to_prompt(caption: dict) -> str:
    """Planner caption (high_level / style / background + boxed elements) -> Klein prompt."""
    comp = (caption or {}).get("compositional_deconstruction") or {}
    st = (caption or {}).get("style_description")
    st = st if isinstance(st, dict) else {}
    table, cites, count, texts = [], [], {}, 0
    for el in comp.get("elements") or []:
        if not isinstance(el, dict):
            continue
        desc = str(el.get("desc") or "").strip()
        text = str(el.get("text") or "").strip()
        if el.get("type") == "text" or text:
            texts += 1
            ident = f"Ru_Text_{texts}"
        else:
            noun = _noun(desc)
            count[noun] = count.get(noun, 0) + 1
            ident = f"{noun}_{count[noun]}"
        row = {"id": ident}
        box = el.get("bbox")
        small = False
        if isinstance(box, (list, tuple)) and len(box) == 4:
            row["bbox"], small = _fit_box(box, bool(text))
        caps = " in capital letters" if text.isupper() and "capital" not in desc.lower() else ""
        row["desc"] = ("Small " if small and not text else "") + desc + caps + (f'. Text: "{text}"' if text else "")
        table.append(row)
        where = f"{_position(row['bbox'])}: " if "bbox" in row else ""
        cites.append(f"{where}<{ident}>")
    art = str(st.get("art_style") or "").strip()
    photo = str(st.get("photo") or "").strip()
    medium = str(st.get("medium") or "").strip()
    lead = (f"{art} {medium}".strip() if art else
            "Full-color photograph" if photo or medium in ("", "photograph", "photo") else medium.capitalize())
    parts = [f"{lead}, {str(caption.get('high_level_description') or '').strip()}".rstrip(", ")]
    if comp.get("background"):
        parts.append(f"Setting: {comp['background']}.")
    mood = "; ".join(x for x in (st.get("lighting"), photo, st.get("aesthetics")) if x)
    if mood:
        parts.append(f"Lighting and look: {mood}.")
    if texts:
        parts.append("Place all lettering fully inside the frame, with clear margins.")
    if cites:
        parts.append(". ".join(c[0].upper() + c[1:] for c in cites) + ".")
    prompt = " ".join(p for p in parts if p)
    if table:
        prompt += " " + json.dumps(table, ensure_ascii=False, separators=(",", ":"))
    return prompt


_KEEP = "Everything else stays identical to the source image."
_KEEP_WORDS = ("unchanged", "identical", "stays the same", "stay the same", "keep the rest", "preserve")


def with_keep_clause(instruction: str) -> str:
    """BFL editing format closes an instruction with one clause naming what stays; add it if missing."""
    t = (instruction or "").strip()
    if not t or any(w in t.lower() for w in _KEEP_WORDS):
        return t
    return t + (" " if t.endswith((".", "!", "?", '"')) else ". ") + _KEEP


def _base(prompt: str, lora: Optional[tuple]) -> dict:
    wf = {
        N_UNET: {"class_type": "UNETLoader", "inputs": {"unet_name": KLEIN_UNET, "weight_dtype": "default"}},
        N_CLIP: {"class_type": "CLIPLoader", "inputs": {"clip_name": KLEIN_CLIP, "type": "flux3", "device": "default"}},
        N_VAE: {"class_type": "VAELoader", "inputs": {"vae_name": KLEIN_VAE}},
        N_POS: {"class_type": "CLIPTextEncode", "inputs": {"clip": [N_CLIP, 0], "text": prompt}},
        N_NEG: {"class_type": "CLIPTextEncode", "inputs": {"clip": [N_CLIP, 0], "text": ""}},
        N_DECODE: {"class_type": "VAEDecode", "inputs": {"samples": [N_SAMPLER, 0], "vae": [N_VAE, 0]}},
        N_SAVE: {"class_type": "SaveImage", "inputs": {"filename_prefix": "klein", "images": [N_DECODE, 0]}},
    }
    model = [N_UNET, 0]
    if lora:
        wf[N_LORA] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"model": model, "lora_name": lora[0], "strength_model": float(lora[1])}}
        model = [N_LORA, 0]
    wf[N_SHIFT] = {"class_type": "ModelSamplingFlux", "inputs": {
        "model": model, "max_shift": _MU, "base_shift": _MU, "width": 1024, "height": 1024}}
    return wf


_STEPS = contextvars.ContextVar("klein_steps", default=None)


@contextlib.contextmanager
def steps_for_task(n: Optional[int]):
    """Every Klein render inside the block samples `n` steps (None: KLEIN_STEPS)."""
    tok = _STEPS.set(n)
    try:
        yield
    finally:
        _STEPS.reset(tok)


def _steps(steps: Optional[int]) -> int:
    return steps or _STEPS.get() or KLEIN_STEPS


def _sampler(positive, negative, latent, seed: int, steps: int) -> dict:
    return {"class_type": "KSampler", "inputs": {
        "model": [N_SHIFT, 0], "positive": positive, "negative": negative, "latent_image": latent,
        "seed": int(seed), "steps": int(steps), "cfg": 1.0, "sampler_name": "euler",
        "scheduler": "simple", "denoise": 1.0}}


def t2i_workflow(prompt: str, width: int, height: int, seed: int, *, steps: Optional[int] = None,
                 lora: Optional[tuple] = None, refs: Optional[list] = None) -> dict:
    """A new picture; `refs` (uploaded names) go in as reference latents only -- what a named
    object really looks like -- the frame is the empty latent's, not theirs."""
    wf = _base(prompt, lora)
    wf[N_LATENT] = {"class_type": "EmptyFlux2LatentImage",
                    "inputs": {"width": int(width), "height": int(height), "batch_size": 1}}
    wf[N_SHIFT]["inputs"].update(width=int(width), height=int(height))
    cond = [N_POS, 0], [N_NEG, 0]
    if refs:
        ref_in = {"positive": [N_POS, 0], "negative": [N_NEG, 0], "vae": [N_VAE, 0],
                  "resolution": "resize", "resolution.megapixels": 1.0}
        for i, name in enumerate(refs[:4]):
            wf[f"ref_load_{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
            ref_in[f"images.image{i}"] = [f"ref_load_{i}", 0]
        wf[N_REF] = {"class_type": "ReferenceMultiLatent", "inputs": ref_in}
        cond = [N_REF, 0], [N_REF, 1]
    wf[N_SAMPLER] = _sampler(cond[0], cond[1], [N_LATENT, 0], seed, _steps(steps))
    return wf


def edit_workflow(image_name: str, instruction: str, seed: int, *, megapixels: float = 1.0,
                  refs: Optional[list] = None, steps: Optional[int] = None,
                  extra_lora: Optional[tuple] = None, consistency: bool = True,
                  ref_mp: float = 1.0) -> dict:
    """Instruction edit: the picture (and up to a few extra reference images) go in as reference
    latents; the output is sized like the picture scaled to `megapixels`. `consistency` loads the
    Consistency LoRA that keeps the edit on the original's frame; `extra_lora` stacks a task LoRA."""
    wf = _base(instruction, (KLEIN_EDIT_LORA, KLEIN_EDIT_LORA_STRENGTH) if consistency else None)
    if extra_lora:
        wf["21"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": wf[N_SHIFT]["inputs"]["model"], "lora_name": extra_lora[0], "strength_model": float(extra_lora[1])}}
        wf[N_SHIFT]["inputs"]["model"] = ["21", 0]
    wf[N_LOAD] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    wf[N_SCALE] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": [N_LOAD, 0], "upscale_method": "lanczos", "megapixels": float(megapixels), "resolution_steps": 16}}
    wf[N_SIZE] = {"class_type": "GetImageSize", "inputs": {"image": [N_SCALE, 0]}}
    ref_in = {"positive": [N_POS, 0], "negative": [N_NEG, 0], "vae": [N_VAE, 0],
              "resolution": "resize", "resolution.megapixels": float(ref_mp), "images.image0": [N_SCALE, 0]}
    if consistency:     # the bench pairing (klein_lora): reference latent at the picture's own size
        ref_in["resolution"] = "keep resolution"
        ref_in.pop("resolution.megapixels")
    for i, name in enumerate((refs or [])[:4], 1):
        wf[f"ref_load_{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        ref_in[f"images.image{i}"] = [f"ref_load_{i}", 0]
    wf[N_REF] = {"class_type": "ReferenceMultiLatent", "inputs": ref_in}
    wf[N_LATENT] = {"class_type": "EmptyFlux2LatentImage",
                    "inputs": {"width": [N_SIZE, 0], "height": [N_SIZE, 1], "batch_size": 1}}
    wf[N_SHIFT]["inputs"].update(width=[N_SIZE, 0], height=[N_SIZE, 1])
    wf[N_SAMPLER] = _sampler([N_REF, 0], [N_REF, 1], [N_LATENT, 0], seed, _steps(steps))
    return wf
