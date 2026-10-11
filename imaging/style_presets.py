"""Named art-style presets for the single-image style change (redraw_image,
mode='redraw', rendered by image_style.restyle).

Each style is spelled out as what makes it THAT style -- medium, line, shading,
faces, palette. One-word styles all came back alike: «anime» and «comic» were
the same outlined comic on the photo (owner, 2026-10-11), «oil painting» a photo
with a warm filter. bench/style_ab.py compared the wordings and the sampler on a
street photo and two cartoons; the detail only shows once the sampler has real
guidance (image_style.STYLE_STEPS / STYLE_CFG).

`target` is what the style turns the picture INTO: "drawn" gets a negative
against the photo look, "photo" (realism) the realism LoRA and a negative
against the drawn look.
"""

STYLE_PRESETS: dict[str, dict] = {
    "anime": {
        "label_ru": "🎌 Аниме", "label_en": "🎌 Anime", "target": "drawn",
        "prompt": ("Turn picture 1 into a still frame from a Japanese 2D anime film. Every person "
                   "gets an anime face: large expressive eyes with bright highlights, a tiny nose "
                   "and mouth, smooth simplified skin, hair drawn as clean pointed clumps with "
                   "sharp highlights. Flat cel shading with two or three tones and hard-edged "
                   "shadow shapes, thin clean dark lineart around every form, a painted anime "
                   "background with soft gradients and vivid saturated colors."),
    },
    "watercolor": {
        "label_ru": "🎨 Акварель", "label_en": "🎨 Watercolor", "target": "drawn",
        "prompt": ("Turn picture 1 into a traditional watercolor painting on textured cold-press "
                   "paper: transparent pigment washes, soft wet-in-wet edges bleeding into each "
                   "other, pigment granulation, white paper left unpainted in the highlights, "
                   "faint pencil underdrawing lines, loose simplified details and no hard outlines."),
    },
    "oil_painting": {
        "label_ru": "🖼️ Масло", "label_en": "🖼️ Oil painting", "target": "drawn",
        "prompt": ("Turn picture 1 into a classical oil painting on canvas: thick impasto "
                   "brushstrokes visible on every surface, canvas weave texture, rich layered "
                   "glazes, warm Rembrandt-style chiaroscuro light, painterly simplified details "
                   "like a museum fine-art painting."),
    },
    "comic": {
        "label_ru": "💥 Комикс", "label_en": "💥 Comic book", "target": "drawn",
        "prompt": ("Turn picture 1 into a panel from an American comic book printed on paper. "
                   "Heavy black ink outlines of varying weight, solid black spot shadows and ink "
                   "hatching, flat bold printed colors, Ben-Day halftone dot patterns in the mid "
                   "tones, gritty inked texture. Faces keep realistic proportions drawn in a "
                   "classic comic-book way (not anime, no big eyes)."),
    },
    "cyberpunk": {
        "label_ru": "🌆 Киберпанк", "label_en": "🌆 Cyberpunk neon", "target": "drawn",
        # Cyberpunk is a world, not a filter: the same hoodie under neon light read as
        # "a photo with a pink tint" (owner: «надо адаптировать и текущую одежду под
        # киберпанк»). The clothes keep their type and cut and become their cyberpunk
        # version; pose, face and framing stay.
        "keeps_clothes": False,
        "prompt": ("Turn picture 1 into a cyberpunk neon illustration: a rainy neon-lit "
                   "futuristic city at night, glowing magenta and cyan rim light, neon sign "
                   "reflections on wet surfaces, high-contrast moody atmosphere. Re-imagine the "
                   "clothes each person wears as their cyberpunk version -- the same kind of "
                   "garment and cut in glossy techwear fabric with glowing LED seams, chrome "
                   "hardware and small cybernetic details. Keep each face and identity."),
    },
    "pencil": {
        "label_ru": "✏️ Карандаш", "label_en": "✏️ Pencil sketch", "target": "drawn",
        "prompt": ("Turn picture 1 into a hand-drawn graphite pencil sketch on white sketchbook "
                   "paper: grey graphite only with no color, confident contour lines, "
                   "cross-hatching in the shadows, smudged soft shading, sketchy unfinished edges "
                   "fading into blank paper."),
    },
    "photo": {
        "label_ru": "📷 Реализм", "label_en": "📷 Real photo", "target": "photo",
        "prompt": ("Change picture 1 into a real photograph taken with a full-frame DSLR camera, "
                   "as if the scene were shot in real life: real human beings with natural adult "
                   "human proportions, realistic faces and eyes of normal size, natural skin "
                   "texture with pores and fine hairs, real hair strands, real fabric weave, real "
                   "materials, natural photographic lighting and shallow depth of field."),
    },
}

# Display order for the keyboard -- dict insertion order is already this, but
# a keyboard-builder should not depend on that being remembered.
STYLE_PRESET_ORDER = ("anime", "watercolor", "oil_painting", "comic",
                      "cyberpunk", "pencil", "photo")


def preset_prompt(key: str) -> str:
    """The redraw instructions for `key`, or "" if it is not a known preset."""
    p = STYLE_PRESETS.get(key)
    return p["prompt"] if p else ""


def preset_for(instructions: str):
    """The preset whose prompt `instructions` carries (the button's task passes it
    through the agent, which may add words around it), or None."""
    text = " ".join((instructions or "").split())
    for key, p in STYLE_PRESETS.items():
        if p["prompt"][:80] in text:
            return key
    return None


def preset_label(key: str, lang: str = "ru") -> str:
    p = STYLE_PRESETS.get(key)
    if not p:
        return key
    return p["label_ru"] if lang == "ru" else p["label_en"]
