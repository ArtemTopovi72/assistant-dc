"""A/B: continuing a clip with the tail as a <Video 1> reference (old) vs pinned
Motion Context frames (new). Same part 1, same seeds, same parts 2/3; prints render
time per part and writes both joined clips. Every line is timestamped and flushed.

    venv/Scripts/python bench/motion_context_ab.py
"""
import os, sys, time, json, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
for d in ("", "core", "media", "agent", "bot", "imaging"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import video as V
import tg_continue

OUT = os.path.join(ROOT, "outputs", "motion_ctx_ab")
os.makedirs(OUT, exist_ok=True)
PARTS = [
    "A grey-haired man in a blue apron stands at a wooden kitchen counter, picks up a large knife, "
    "and starts chopping carrots fast while saying «Ну что, начнём готовить!»",
    "He sweeps the chopped carrots into a steaming pot with the flat of the knife, stirs it with a "
    "wooden spoon, tastes from the spoon, frowns, then shakes salt in and says «Соли маловато, "
    "сейчас исправим», and keeps stirring for a few seconds.",
    "He smiles, turns to the camera and says «Вот теперь отлично!»",
]
SEED1, SEED = 424242, 777


class Ctx:
    def set_stage(self, *_): pass
    def is_cancelled(self): return False


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def render(desc, **kw):
    t = time.time()
    r = V.generate_video(Ctx(), desc, aspect="3:4", **kw)
    dt = time.time() - t
    log(f"  -> {r.get('status')} {dt:.0f}s {r.get('frames')} frames {r.get('path')}")
    return r.get("path"), dt


def chain(first, pinned):
    path, times = first, []
    for i, part in enumerate(PARTS[1:], start=2):
        frame = os.path.join(OUT, f"{'new' if pinned else 'old'}_seed_{i}.jpg")
        tail = os.path.join(OUT, f"{'new' if pinned else 'old'}_tail_{i}.mp4")
        assert tg_continue.seed_frame(path, frame) and tg_continue.cut_tail(path, tail)
        log(f"{'NEW' if pinned else 'OLD'} part {i}")
        if pinned:
            new, dt = render(V.CONTINUE_CTX_PREFIX + part, context_video=tail, seed=SEED + i)
            j = V.join_pinned(path, new)
        else:
            new, dt = render(V.CONTINUE_PREFIX + part, images=[frame], videos=[tail], seed=SEED + i)
            j = V.join_continuation(path, new)
        times.append(dt)
        path = os.path.join(OUT, f"{'new' if pinned else 'old'}_upto{i}.mp4")
        shutil.move(j, path)
    return path, times


if __name__ == "__main__":
    log("motion context on:", V.motion_context_on())
    p1 = os.path.join(OUT, "part1.mp4")
    if not os.path.exists(p1):
        log("part 1")
        src, _ = render(PARTS[0], seed=SEED1)
        shutil.copy(src, p1)
    which = sys.argv[1:] or ["new", "old"]
    res = {}
    for w in which:
        final, times = chain(p1, w == "new")
        res[w] = {"final": final, "times": [round(t) for t in times], "seconds": V.probe(final)["seconds"]}
        log(w, json.dumps(res[w]))
    json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1)
    log("done", json.dumps(res))
