"""A/B of FireRed style transfer: prompts × sampler settings, one fixed seed.

    venv/Scripts/python.exe bench/style_ab.py <variant>[,<variant>] [style,...]

Variants: new_l8 (the style prompts on the template's Lightning 8 steps cfg 1;
the old one-word prompts there were the 2026-10-11 baseline), new_f28 (new prompts, no Lightning, 28 steps,
cfg 4, a per-style negative). Writes runtime/style_ab/<variant>/<style>.png
plus a sheet per variant.
"""
import json
import os
import re
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, d) for d in
                ("imaging", "media", "agent", "core", "services", "research", "knowledge",
                 "voice", "sandbox", "")]
os.environ.setdefault("F5_TEST_RUN", "0")

import image as image_mod  # noqa: E402
import image_engines  # noqa: E402
import style_presets  # noqa: E402

OUT = os.path.join(ROOT, "runtime", "style_ab")
PHOTO = os.path.join(ROOT, "tests", "fixtures", "faces3.png")
TOON = {"realism_pixar": os.path.join(ROOT, "tmp", "style", "cartoon_man.png"),
        "realism_anime": os.path.join(ROOT, "tmp", "style", "anime_girl.png")}
SEED = 424242
# What a vision pass would say about who is in the picture (the app asks the VLM).
DESC = {"realism_pixar": "a chubby middle-aged man with a thick black handlebar mustache, short dark "
                         "hair under a white baker's cap, a white apron, holding a baguette loaf",
        "realism_anime": "a teenage Japanese schoolgirl with a short blue bob haircut and blue eyes, "
                         "a white sailor school uniform with a navy collar and a red neckerchief"}


class _Ctx:
    def __getattr__(self, k):
        return None

    def set_stage(self, *a):
        pass

    def is_cancelled(self):
        return False


def _workflow(steps: int, cfg: float, lightning: bool, negative: str) -> str:
    _, path = image_engines._edit_engine_workflow(None)
    wf = json.load(open(path, encoding="utf-8"))
    wf["189"]["inputs"]["steps"], wf["189"]["inputs"]["cfg"] = steps, cfg
    if not lightning:
        wf["183"]["inputs"]["strength_model"] = 0.0
    wf["188"]["inputs"]["prompt"] = negative
    fd, tmp = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(wf, f)
    return tmp


def run(variant: str, styles: list) -> None:
    import style_prompts_v2 as v2
    os.makedirs(os.path.join(OUT, variant), exist_ok=True)
    for style in styles:
        src = TOON.get(style, PHOTO)
        instr, neg = v2.prompt(style), v2.negative(style)
        if variant.startswith("d") and style in DESC:
            instr = v2.prompt(style, DESC[style])
            variant_sampler = variant[1:]
        else:
            variant_sampler = variant
        if variant == "new_l8":
            wf = _workflow(8, 1.0, True, "")
        elif variant == "new_f28":
            wf = _workflow(28, 4.0, False, neg)
        else:                                   # f16c3 / l8c2: steps, cfg, Lightning on/off
            m = re.fullmatch(r"([fl])(\d+)c([\d.]+)", variant_sampler)
            wf = _workflow(int(m[2]), float(m[3]), m[1] == "l", neg)
        image_mod._edit_engine_workflow = lambda e, _p=wf: ("firered", _p)
        t = time.time()
        out = image_mod._edit_image_with_firered_once(_Ctx(), src, instr, seed=SEED)
        dst = os.path.join(OUT, variant, f"{style}.png")
        if out:
            shutil.copy(out, dst)
        print(f"{variant} {style} {time.time() - t:.0f}s -> {dst if out else 'FAILED'}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    variants = sys.argv[1].split(",")
    styles = (sys.argv[2].split(",") if len(sys.argv) > 2 else
              list(style_presets.STYLE_PRESET_ORDER) + list(TOON))
    for v in variants:
        run(v, styles)
