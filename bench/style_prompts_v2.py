"""Candidate style prompts for bench/style_ab.py: the style first and concrete
(medium, line, shading, faces), a short keep-clause last, and a negative that
names what the style is NOT (the photo for a drawn style, the drawing for realism)."""

_KEEP = (" Keep the same composition, framing, people, their poses and positions, "
         "clothing and setting; restyle the whole image including the background.")

STYLES = {
    "anime": ("Turn picture 1 into a still frame from a Japanese 2D anime film. Every person "
              "gets an anime face: large expressive eyes with bright highlights, a tiny nose and "
              "mouth, smooth simplified skin, hair drawn as clean pointed clumps with sharp "
              "highlights. Flat cel shading with two or three tones and hard-edged shadow "
              "shapes, thin clean dark lineart around every form, a painted anime background "
              "with soft gradients and vivid saturated colors."),
    "comic": ("Turn picture 1 into a panel from an American comic book printed on paper. "
              "Heavy black ink outlines of varying weight, solid black spot shadows and ink "
              "hatching, flat bold printed colors, Ben-Day halftone dot patterns in the mid "
              "tones, gritty inked texture. Faces keep realistic proportions drawn in a "
              "classic comic-book way (not anime, no big eyes)."),
    "watercolor": ("Turn picture 1 into a traditional watercolor painting on textured "
                   "cold-press paper: transparent pigment washes, soft wet-in-wet edges bleeding "
                   "into each other, pigment granulation, white paper left unpainted in the "
                   "highlights, faint pencil underdrawing lines, loose simplified details and "
                   "no hard outlines."),
    "oil_painting": ("Turn picture 1 into a classical oil painting on canvas: thick impasto "
                     "brushstrokes visible on every surface, canvas weave texture, rich layered "
                     "glazes, warm Rembrandt-style chiaroscuro light, painterly simplified "
                     "details like a museum fine-art painting."),
    "pencil": ("Turn picture 1 into a hand-drawn graphite pencil sketch on white sketchbook "
               "paper: grey graphite only with no color, confident contour lines, "
               "cross-hatching in the shadows, smudged soft shading, sketchy unfinished edges "
               "fading into blank paper."),
    "cyberpunk": ("Turn picture 1 into a cyberpunk neon illustration: a rainy neon-lit "
                  "futuristic city at night, glowing magenta and cyan rim light, neon sign "
                  "reflections on wet surfaces, high-contrast moody atmosphere. Re-imagine "
                  "the clothes each person wears as their cyberpunk version -- the same kind "
                  "of garment and cut in glossy techwear fabric with glowing LED seams, "
                  "chrome hardware and small cybernetic details. Keep each face and identity."),
    "realism": ("Change picture 1 into a real photograph taken with a full-frame DSLR camera, "
                "as if the scene were shot in real life: real human beings with natural adult "
                "human proportions, realistic faces and eyes of normal size, natural skin "
                "texture with pores and fine hairs, real hair strands, real fabric weave, real "
                "materials, natural photographic lighting and shallow depth of field."),
}

NEG = {
    "drawn": ("photo, photograph, photorealistic, realistic skin texture, skin pores, 3d render, "
              "blurry, low quality, watermark, text, signature"),
    "realism": ("cartoon, anime, illustration, drawing, painting, 3d render, cgi, pixar, "
                "plastic skin, doll, big eyes, cel shading, lineart, blurry, low quality, watermark"),
}


def prompt(style: str, who: str = "") -> str:
    key = "realism" if style.startswith("realism") else style
    keep = _KEEP.replace("clothing and setting; ", "clothing colors and setting; ") if key == "realism" else _KEEP
    if key == "cyberpunk":
        keep = keep.replace("clothing and setting; ", "")
    if who:
        keep = (f" The picture shows {who}; the photo shows exactly this real person with "
                "the same gender, age, hair and facial hair, at real-life human body "
                "proportions (a normal-sized head on a normal adult body)." + keep)
    return STYLES[key] + keep


def negative(style: str) -> str:
    return NEG["realism"] if style.startswith("realism") else NEG["drawn"]
