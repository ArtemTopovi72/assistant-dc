"""A/B: does the background signage guard (commit 4595f07) help or hurt?

The guard appends an instruction to every caption background ("signs ... must be
abstract, blurred or angled away"). An earlier measurement found that an
instruction in `background` turns the scene into cut-outs on a checkerboard and
the invented lettering shows up anyway. This renders the same layouts on the
same seeds with and without the clause, straight through ideogram.generate
(no collage re-roll, which would hide exactly the failure being measured).

    venv/Scripts/python.exe bench/signage_guard_ab.py [out_dir]
"""
import json
import shutil
import sys
import threading
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import comfy_client
import draw_agent
import ideogram

SCENES = [
    ("night_street", "a rainy night street in Tokyo lined with glowing neon shop signs, "
                     "photograph", [{"desc": "a woman with a transparent umbrella walking",
                                     "x": 0.38, "y": 0.35, "w": 0.24, "h": 0.6}]),
    ("storefronts", "a sunny old-town street with small cafes and shop storefronts with "
                    "signboards, photograph", [{"desc": "a red bicycle leaning on a lamp post",
                                                "x": 0.1, "y": 0.5, "w": 0.35, "h": 0.45}]),
    ("station", "a busy train station hall with departure boards and advertising "
                "posters, photograph", [{"desc": "a man in a grey coat with a suitcase",
                                         "x": 0.55, "y": 0.3, "w": 0.22, "h": 0.65}]),
]
SEEDS = {"night_street": 1111, "storefronts": 2222, "station": 3333}


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runtime" / "signage_ab"
    out.mkdir(parents=True, exist_ok=True)
    ctx = types.SimpleNamespace(cancel_event=threading.Event(),
                                set_stage=lambda *a, **k: None, image_quality="")
    # The guard was removed after this A/B (it changed nothing). Re-create it
    # here so the bench still reproduces the comparison.
    clause = (" Any signs, posters, screens or other lettering-shaped objects visible "
              "in the background that are NOT one of the text elements above must be "
              "abstract, blurred or angled away so no legible word is attempted on them.")
    report = []
    with comfy_client.card_session("signage A/B"):
        for name, bg, els in SCENES:
            layout = ideogram.normalize_layout({"background": bg, "medium": "photograph",
                                                "elements": els})
            for arm in ("guard", "plain"):
                cap = ideogram.layout_to_caption(layout)
                if arm == "guard":
                    cd = cap["compositional_deconstruction"]
                    cd["background"] = str(cd["background"]).strip() + clause
                path = ideogram.generate(ctx, "", width=1280, height=720,
                                         seed=SEEDS[name], caption=cap)
                row = {"scene": name, "arm": arm,
                       "background": cap["compositional_deconstruction"]["background"]}
                if path:
                    dst = out / f"{name}_{arm}.png"
                    shutil.copy(path, dst)
                    row.update(file=str(dst), collage=bool(draw_agent.looks_like_collage(str(dst))))
                else:
                    row.update(file=None)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                report.append(row)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
