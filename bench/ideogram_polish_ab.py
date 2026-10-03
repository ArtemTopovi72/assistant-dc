"""A/B the flat cfg=7 schedule against Ideogram's own polish-step schedule.

Ideogram 4's published sampler presets never run at a constant guidance weight:
V4_QUALITY_48 is 45 steps @ gw=7 then 3 @ gw=3, V4_DEFAULT_20 is 18 + 2,
V4_TURBO_12 is 11 + 1. Our workflow ran a flat 7 for the whole schedule -- the
one shape upstream does not use. config.IDEOGRAM_POLISH_STEPS wires the tail in,
but it defaults to 0 until this measurement says it should not.

Captions are built by HAND rather than planned. Two reasons: the planner needs
the chat model resident, which would take VRAM from the very renders being
timed, and an LLM in the loop is a second variable in a test meant to have one.
The first version of this script did use the planner, found no model loaded,
silently fell back to a flat styleless caption, and Ideogram answered the
lettering prompt with its safety card.

Same seed, same caption, one variable. Takes the GPU lock, so it will not run
beside a trainer.

    venv/Scripts/python.exe bench/ideogram_polish_ab.py
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import gpu_lock
import ideogram
import ideogram_layout as IL

OUT = Path.home() / "Desktop" / "Ideogram polish A-B"
SEED = 20260908
STEPS = 20
VARIANTS = [("flat", 0), ("polish2", 2), ("polish3", 3)]


def _cafe():
    """Lettering: the polish tail should show up first on letter edges."""
    return IL.build_caption(
        "a quiet city street at dusk, wet cobblestones reflecting warm light",
        [IL.element("a small corner cafe with large windows and a striped awning",
                    IL.bbox(0.15, 0.25, 0.70, 0.60)),
         IL.element("a painted sign board above the cafe door, bold white block "
                    "capitals", IL.bbox(0.30, 0.12, 0.40, 0.10), text="CAFE ROSA"),
         IL.element("a bicycle leaning against the kerb",
                    IL.bbox(0.05, 0.62, 0.20, 0.25))],
        high_level="A corner cafe at dusk with a lit sign reading CAFE ROSA.",
        aesthetics="photorealistic, cinematic, richly detailed",
        lighting="warm shop light against cool blue dusk",
        photo="35mm lens, shallow depth of field",
        medium="photography")


def _portrait():
    """Skin and fabric: the other place a low-guidance cleanup pass shows."""
    return IL.build_caption(
        "a birch forest on an overcast afternoon, pale trunks receding",
        [IL.element("a man in a grey uniform jacket, standing, waist up, "
                    "looking at the camera", IL.bbox(0.30, 0.20, 0.40, 0.70))],
        high_level="A man in a grey uniform standing among birch trees.",
        aesthetics="photorealistic, natural skin texture",
        lighting="soft overcast daylight",
        photo="85mm portrait lens",
        medium="photography")


CAPTIONS = [("cafe_sign", _cafe), ("portrait", _portrait)]


def main() -> int:
    # Six full renders is a whole-card job. Run beside a trainer and both the
    # timings and the trainer's step time are wrong -- which is exactly how
    # tonight's 5.93 -> 8.98 s/it degradation happened.
    with gpu_lock.hold("ideogram polish A/B", wait=7200):
        return _run()


def _run() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for i, (label, make) in enumerate(CAPTIONS, 1):
        # ONE caption per scene, reused across variants: rebuilding it per
        # variant would change the picture for reasons unrelated to the
        # sampler, and the comparison would mean nothing.
        caption = make()
        for name, polish in VARIANTS:
            config.IDEOGRAM_POLISH_STEPS = polish
            ideogram.IDEOGRAM_POLISH_STEPS = polish
            t0 = time.time()
            try:
                path = ideogram.generate(None, "", caption=caption, seed=SEED,
                                         steps=STEPS, width=1024, height=1024)
            except ideogram.ContentRefused:
                # One refused render must not end the comparison -- the other
                # five still answer the question being asked.
                print("  %s/%-8s REFUSED by the model" % (label, name), flush=True)
                continue
            dt = time.time() - t0
            if not path:
                print("  %s/%-8s FAILED" % (label, name), flush=True)
                continue
            shutil.copy(path, OUT / ("%02d_%s_%s.png" % (i, label, name)))
            print("  %s/%-8s %5.1fs" % (label, name, dt), flush=True)

    print("\nout:", OUT)
    print("Look at lettering edges and fine texture -- the polish tail is a "
          "cleanup pass, so if it does anything it shows there first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
