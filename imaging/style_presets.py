"""Named art-style presets for the single-image style change (redraw_image,
mode='redraw'). No second reference photo needed -- the old controlnet
workflow's controlnet keeps composition/pose from the original, so the prompt
only has to describe the STYLE change (image.redraw_image_with_comfy folds
ctx.last_image_prompt -- the subject -- back in automatically).

Every prompt ends with an explicit "keep the pose/outfit/setting" clause.
Measured live (2026-09-18) against a real red-carpet photo: dropping that
clause let the controlnet's own style-transfer strength drift the framing on
a couple of presets (the subject stayed recognisable but cropped differently
between runs); keeping it made the four re-tested presets consistently
preserve the exact composition run to run.
"""

_KEEP = ("keep the exact same pose, camera framing, outfit and background "
         "setting -- change only the art style, not the content")

STYLE_PRESETS: dict[str, dict] = {
    "anime": {
        "label_ru": "🎌 Аниме", "label_en": "🎌 Anime",
        "prompt": ("convert to anime art style, cel-shaded anime illustration, "
                   "clean bold line art, vibrant flat anime colors, " + _KEEP),
    },
    "watercolor": {
        "label_ru": "🎨 Акварель", "label_en": "🎨 Watercolor",
        "prompt": ("convert to a watercolor painting, soft translucent washes of "
                   "color, visible paper texture and pigment bleed at the edges, "
                   "loose painterly brushwork, " + _KEEP),
    },
    "oil_painting": {
        "label_ru": "🖼️ Масло", "label_en": "🖼️ Oil painting",
        "prompt": ("convert to a classical oil painting, visible canvas texture "
                   "and confident brush strokes, rich layered color, dramatic "
                   "painterly lighting like a fine-art portrait, " + _KEEP),
    },
    "comic": {
        "label_ru": "💥 Комикс", "label_en": "💥 Comic book",
        "prompt": ("convert to a comic book / pop-art illustration, bold black "
                   "ink outlines, flat halftone-dot shading, punchy saturated "
                   "colors, graphic-novel cover style, " + _KEEP),
    },
    "cyberpunk": {
        "label_ru": "🌆 Киберпанк", "label_en": "🌆 Cyberpunk neon",
        "prompt": ("convert to a cyberpunk neon illustration, glowing magenta "
                   "and cyan rim lighting, futuristic neon signage reflections, "
                   "moody high-contrast night atmosphere, " + _KEEP),
    },
    "pencil": {
        "label_ru": "✏️ Карандаш", "label_en": "✏️ Pencil sketch",
        "prompt": ("convert to a detailed graphite pencil sketch, fine cross-"
                   "hatching and shading, monochrome, visible paper grain, "
                   "sketchbook portrait style, " + _KEEP),
    },
}

# Display order for the keyboard -- dict insertion order is already this, but
# a keyboard-builder should not depend on that being remembered.
STYLE_PRESET_ORDER = ("anime", "watercolor", "oil_painting", "comic",
                      "cyberpunk", "pencil")


def preset_prompt(key: str) -> str:
    """The redraw instructions for `key`, or "" if it is not a known preset."""
    p = STYLE_PRESETS.get(key)
    return p["prompt"] if p else ""


def preset_label(key: str, lang: str = "ru") -> str:
    p = STYLE_PRESETS.get(key)
    if not p:
        return key
    return p["label_ru"] if lang == "ru" else p["label_en"]
