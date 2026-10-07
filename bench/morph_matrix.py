"""Part 2 of the chef chain from the SAME part-1 latent, one knob at a time:
steps 3 vs 6, two-stage vs single, 22 vs 39 pinned frames. Writes a contact sheet
per variant to judge object morphing (knife -> spoon, extra spoons) by eye.

    VIDEO_MOTION_CONTEXT=1 venv/Scripts/python bench/morph_matrix.py <part1.mp4> <part1_latent>
"""
import os, sys, time, subprocess, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
for d in ("", "core", "media", "agent", "bot", "imaging"):
    sys.path.insert(0, os.path.join(ROOT, d))
import video as V
import tg_continue

OUT = os.path.join(ROOT, "outputs", "morph_matrix")
os.makedirs(OUT, exist_ok=True)
PART = ("He sweeps the chopped carrots into a steaming pot with the flat of the knife, stirs it with a "
        "wooden spoon, tastes from the spoon, frowns, then shakes salt in and says «Соли маловато, "
        "сейчас исправим», and keeps stirring for a few seconds.")
VARIANTS = [  # name, steps, two_stage, pinned frames
    ("base", 3, "1", 22), ("s6", 6, "1", 22), ("single", 3, "0", 22),
    ("s6_single", 6, "0", 22), ("ctx39", 3, "1", 39),
    ("s8", 8, "1", 22), ("full20", 20, "1", 22),
]
OVERRIDES = {"full20": '{"13.strength_model": 0.0}'}  # base model, no turbo LoRA


class Ctx:
    def set_stage(self, *_): pass
    def is_cancelled(self): return False


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main(p1, lat):
    frame, tail = os.path.join(OUT, "seed.jpg"), os.path.join(OUT, "tail.mp4")
    assert tg_continue.seed_frame(p1, frame) and tg_continue.cut_tail(p1, tail)
    part = V.bridge_part(None, frame, PART)
    log("part:", part)
    only = sys.argv[3:]
    for name, steps, two, ctxn in VARIANTS:
        if only and name not in only:
            continue
        os.environ["VIDEO_TWO_STAGE"] = two
        V.MOTION_CONTEXT_FRAMES = ctxn
        os.environ["VIDEO_WF_OVERRIDES"] = OVERRIDES.get(name, "{}")
        t = time.time()
        r = V.generate_video(Ctx(), V.CONTINUE_CTX_PREFIX + part, aspect="3:4", steps=steps,
                             context_video=tail, context_latent=lat, seed=779)
        dt = time.time() - t
        path = r.get("path")
        log(f"{name}: {r.get('status')} {dt:.0f}s {path}")
        if not path:
            continue
        dst = os.path.join(OUT, f"{name}.mp4")
        os.replace(path, dst)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", dst, "-vf",
                        "select='not(mod(n,10))',scale=150:-1,tile=9x3", "-frames:v", "1",
                        os.path.join(OUT, f"{name}.jpg")])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
