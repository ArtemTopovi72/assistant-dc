"""Edit-engine registry: which ComfyUI workflow backs each named edit engine.

Tiny by design — it is imported by every edit pipeline (including as a default
argument value), so it must sit at the bottom of the dependency graph with no
imports back into image.py.
"""
from typing import Optional

from config import WORKFLOW_FIRERED_EDIT_PATH


# Instruction-edit engines for edit_image_with_firered / the contained-edit
# pipeline. FireRed is the only one: the base Qwen-Image-Edit 2509 alternative
# ("qwenimage") was removed from the product along with its checkpoint.
EDIT_ENGINE_WORKFLOWS = {
    "firered": WORKFLOW_FIRERED_EDIT_PATH,
}


DEFAULT_EDIT_ENGINE = "firered"


def _edit_engine_workflow(engine: Optional[str]):
    """Resolve an engine name to its workflow path, defaulting to FireRed."""
    key = (engine or DEFAULT_EDIT_ENGINE).strip().lower()
    return key if key in EDIT_ENGINE_WORKFLOWS else DEFAULT_EDIT_ENGINE, \
        EDIT_ENGINE_WORKFLOWS.get(key, WORKFLOW_FIRERED_EDIT_PATH)


# FireRed (Qwen-Image-Edit) working-resolution cap. Node 191 in the workflow
# downscales the input to this many megapixels before editing; the old flat 1.0 MP
# crushed a large garment tile and produced the "compressed cartoon" look. We size
# the working MP to the actual tile, capped here for VRAM on the 12 GB 3060 (shared
# with the LM Studio LLM). Measured A/B on a 3.42 MP garment tile (lanczos-restored
# to tile size, laplacian-variance sharpness): 1.0 MP -> 9, 2.0 MP -> 42 (4.7x),
# 3.0 MP -> 54. 3.0 ran without OOM (~154 s) but the decisive gain is 1->2 MP; 2.0
# keeps that win with the most VRAM headroom. Raise this single constant to trade
# sharpness for time/VRAM.
FIRERED_MAX_MP = 2.0
