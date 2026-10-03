"""Image geometry: session size resolution, [generate] tag parsing, orientation
inference, resolution normalisation and the output-size guard.

Extracted from image.py. Everything here is pure/deterministic apart from the
config lookups, so it is the cheapest layer to test and the one the size-choice
suites drive directly.
"""
import logging
import random
import re
from typing import Optional, Tuple

from config import DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_STEPS, DEFAULT_CFG
import config as _config          # read live so the GUI can switch engines/sizes

logger = logging.getLogger("assistant.image")


def _snap_to_8(n: int) -> int:
    """Round n down to the nearest multiple of 8 (min 64)."""
    return max(64, (n // 8) * 8)


def session_image_size(ctx) -> Tuple[int, int]:
    """The size this caller wants, from ctx.image_aspect / ctx.image_quality.

    The Telegram bot swaps a per-chat choice onto ctx around graph.invoke (same
    pattern as the working image and pinned facts), so generation size follows
    the user who asked instead of being a global.
    """
    return _config.resolve_image_size(getattr(ctx, "image_aspect", "") or "",
                                      getattr(ctx, "image_quality", "") or "")


def session_size_pinned(ctx) -> bool:
    """True when the caller explicitly chose an aspect (so don't re-orient it)."""
    return bool(getattr(ctx, "image_aspect", ""))


def parse_generation_params(tag_content: str, *,
                            defaults: Optional[Tuple[int, int]] = None,
                            pinned: bool = False
                            ) -> Tuple[str, int, float, int, int, int]:
    """Parse [generate]PROMPT | steps=N | cfg=N | orientation=… | seed=N[/generate].

    `defaults` is the size to use when the tag names none — the caller's
    (per-session) choice. Without it the module default applies.

    `pinned` means the user CHOSE an aspect in the 📐 picker, so the model's
    `orientation=` is ignored: an explicit choice outranks the model's reading of
    the scene, exactly as it already outranks the text heuristic in
    `fix_image_params`. Without this, picking 9:16 and asking for "a cat on a
    windowsill" came back landscape — observed live.
    """
    parts = [p.strip() for p in tag_content.split("|")]
    prompt = parts[0].strip()

    steps = DEFAULT_STEPS
    cfg = DEFAULT_CFG
    seed = random.randint(1, 999_999_999)
    width, height = defaults or (DEFAULT_WIDTH, DEFAULT_HEIGHT)

    # Resolution keywords in prompt. A PINNED aspect must beat these — the same
    # rule the `orientation=` branch below already follows. Without this guard,
    # a model-written "...in stunning 4k detail" or a literal "1920x1080" landed
    # in the PROMPT TEXT (not a structured tag) and silently overrode a 9:16 pin
    # with a 3840x2160 landscape render — observed live, the same "prompt
    # carries a value the resolver owns" bug class as the original 960x544 fix.
    if not pinned:
        prompt_lower = prompt.lower()
        resolution_keywords = {
            "8k": (7680, 4320), "7680x4320": (7680, 4320),
            "4k": (3840, 2160), "3840x2160": (3840, 2160), "2160p": (3840, 2160),
            "ultra hd": (3840, 2160), "2k": (2560, 1440), "2560x1440": (2560, 1440),
            "qhd": (2560, 1440), "1440p": (2560, 1440),
            "full hd": (1920, 1080), "1920x1080": (1920, 1080), "1080p": (1920, 1080),
            "fhd": (1920, 1080), "hd": (1280, 720), "1280x720": (1280, 720), "720p": (1280, 720),
        }
        for keyword, (w, h) in resolution_keywords.items():
            if keyword in prompt_lower:
                width, height = w, h
                break
        else:
            match = re.search(r'(\d+)\s*[x*×]\s*(\d+)', prompt_lower)
            if match:
                w, h = int(match.group(1)), int(match.group(2))
                if 64 <= w <= 8192 and 64 <= h <= 8192:
                    width, height = w, h

    for part in parts[1:]:
        key_part = part.lstrip("/").strip()
        if "=" not in key_part:
            continue
        key, val = key_part.split("=", 1)
        key = key.strip().lower()
        val = val.strip()
        if not val:
            continue
        try:
            if key == "steps":
                v = int(val)
                if 1 <= v <= 100:
                    steps = v
            elif key == "cfg":
                v = float(val)
                if 0.5 <= v <= 30.0:
                    cfg = v
            elif key == "seed":
                v = int(val)
                if v >= 1:
                    seed = v
            elif key == "width":
                if not pinned:
                    v = int(val)
                    if 64 <= v <= 8192:
                        width = v
            elif key == "height":
                if not pinned:
                    v = int(val)
                    if 64 <= v <= 8192:
                        height = v
            elif key == "orientation":
                # SHAPE from the model, PIXELS from the user's quality setting.
                # The generator prompt used to dictate literal sizes (960x544 /
                # 544x960 / 768x768), and an explicit width= always beat the
                # caller's default — so every picture came out at 0.5 MP no
                # matter what 📐 Size said, which is the "blurry" complaint. The
                # model is the right judge of the shape and the wrong judge of
                # the resolution.
                shape = val.strip().lower()
                _aspect = {"landscape": "16:9", "portrait": "9:16",
                           "square": "1:1"}.get(shape)
                if _aspect and not pinned:
                    long_side = max(width, height)
                    short_side = min(width, height)
                    if shape == "square":
                        side = _snap_to_8(int((long_side * short_side) ** 0.5))
                        width = height = side
                    elif shape == "landscape":
                        width, height = long_side, short_side
                    else:
                        width, height = short_side, long_side
        except (ValueError, TypeError):
            continue

    if seed < 1:
        seed = random.randint(1, 999_999_999)

    width = _snap_to_8(width)
    height = _snap_to_8(height)

    return prompt, steps, cfg, seed, width, height


def detect_orientation_from_text(text: str) -> str:
    """landscape | portrait | unknown for a picture request. Read by the model:
    an explicit ask wins ("landscape photo of a man" is landscape although a
    man leans portrait), else the scene decides (one person standing ->
    portrait, a street or a crowd -> landscape). The word lists this replaced
    scored «пейзаж» and «man» in one pool and fell through to the default."""
    import intent
    return intent.ask_choice(
        "A user asked for this picture: {text}. Which frame fits it? An explicit ask "
        "(landscape photo, horizontal, wide, panorama, 16:9 / portrait orientation, "
        "vertical, phone wallpaper, 9:16) wins even over the subject -- \"landscape photo "
        "of a man\" is landscape; "
        "otherwise portrait for one person or a tall subject, landscape for a scene, a "
        "place or a group; unknown when nothing suggests either.",
        text or "", ("landscape", "portrait", "unknown"), "unknown")


def normalize_resolution(width: int, height: int) -> Tuple[int, int]:
    """Clamp a size onto the supported rails, KEEPING its aspect and pixel budget.

    This used to `return 960, 544` (or its transpose) for every input, which
    threw away any size the user or the prompt-planner had chosen — asking for
    1920x1080 got you 0.52 MP and a blurry picture. Now it only enforces the
    min/max side and the multiple-of-16 snap.
    """
    try:
        w, h = int(width), int(height)
    except (TypeError, ValueError):
        return _config.DEFAULT_WIDTH, _config.DEFAULT_HEIGHT
    if w <= 0 or h <= 0:
        return _config.DEFAULT_WIDTH, _config.DEFAULT_HEIGHT

    max_side = getattr(_config, "IMAGE_MAX_SIDE", 2048)
    min_side = getattr(_config, "IMAGE_MIN_SIDE", 256)
    over = max(w, h) / float(max_side)
    if over > 1.0:
        w, h = w / over, h / over
    under = min_side / float(min(w, h))
    if under > 1.0:
        w, h = w * under, h * under

    snap = lambda n: max(min_side, int(round(n / 16.0)) * 16)
    return snap(w), snap(h)


def fix_image_params(goal: str, prompt: str, width: int, height: int, *,
                     pinned: bool = False) -> Tuple[int, int]:
    """Orient a size from the request text, then put it on the rails.

    `pinned` means the user picked the aspect explicitly (Draw ▸ Size). Their
    choice outranks the text heuristic: someone who selects 9:16 and then asks
    for a "wide city panorama" gets a tall panorama, not a silent override.
    """
    if pinned:
        return normalize_resolution(width, height)
    orientation = detect_orientation_from_text(f"{goal} {prompt}")
    if orientation == "portrait" and width > height:
        width, height = height, width
    elif orientation == "landscape" and height > width:
        width, height = height, width
    return normalize_resolution(width, height)


def _source_dims(image_path: str) -> Optional[tuple]:
    """Return (width, height) of the source image, or None if unreadable."""
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            return im.size
    except Exception as exc:
        logger.warning("geometry: cannot read source dims for %s: %s", image_path, exc)
        return None


def _enforce_output_size(workflow: dict, sw: int, sh: int) -> int:
    """Geometry guard: resize every SaveImage's input back to (sw, sh) with a
    lanczos ``ImageScale`` so the saved image matches the ORIGINAL canvas exactly.

    This undoes the internal working-resolution downscale that several pipelines
    apply for VRAM/model-band reasons (FireRed's 1 MP `ImageScaleToTotalPixels`,
    IC-Light's SD1.5 res, the bg-gen 1536 cap, redraw's 1024 long-side) — the root
    cause of the "tiny / blurry person" output documented in
    docs/image_geometry_audit.md. The rescale is uniform (no crop), so when the
    pipeline preserved aspect (all of them do) the subject's in-frame fraction and
    perspective are unchanged; only absolute resolution is restored.

    Returns the number of SaveImage nodes patched (0 = no-op, e.g. bad dims).
    """
    if not sw or not sh or sw < 1 or sh < 1:
        return 0
    patched = 0
    for nid, node in list(workflow.items()):
        if node.get("class_type") != "SaveImage":
            continue
        src = node.get("inputs", {}).get("images")
        if not isinstance(src, list):
            continue
        gid = f"GEOFIX_{nid}"
        workflow[gid] = {
            "inputs": {"image": src, "width": int(sw), "height": int(sh),
                       "upscale_method": "lanczos", "crop": "disabled"},
            "class_type": "ImageScale",
        }
        node["inputs"]["images"] = [gid, 0]
        patched += 1
    if patched:
        logger.info("geometry: enforced output size %dx%d on %d SaveImage node(s)",
                    sw, sh, patched)
    return patched
