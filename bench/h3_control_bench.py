"""H3 Fun ControlNet-Union v2v: restyle base clips over their own motion (canny).
restyle() starts ComfyUI-qi21 itself (video_control.on_control_server).
    venv/Scripts/python bench/h3_control_bench.py [t1_anime,t2_winter]"""
import os, sys, time, json, shutil, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import video_control as VC

OUT = os.path.join(ROOT, "runtime", "night_video", "ctrl")
JOBS = [
    ("t2_winter", "runtime/night_video/base/t2.mp4",
     "A red vintage tractor drives through a snowy field in winter, snowflakes falling, grey overcast sky; "
     "a farmer in a fur hat waves from the seat."),
    ("t1_anime", "runtime/night_video/base/t1.mp4",
     "Anime style, cel-shaded 2D animation: a bearded man in a knitted sweater sits at a kitchen table, "
     "lifts a white mug and drinks, warm evening light."),
]

class Ctx:
    def is_cancelled(self): return False
    def set_stage(self, *a, **k): pass

def main():
    os.makedirs(OUT, exist_ok=True)
    want = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
    tf = os.path.join(OUT, "timings.json")
    res = json.load(open(tf)) if os.path.exists(tf) else {}
    for tag, src, prompt in JOBS:
        if want and tag not in want:
            continue
        t0 = time.time()
        try:
            out = VC.restyle(Ctx(), os.path.join(ROOT, src), prompt, kind="canny", seed=42)
            dt = time.time() - t0
            if out:
                dst = os.path.join(OUT, tag + ".mp4"); shutil.copy(out, dst)
                for t in (0.2, 1.5, 3.0, 4.5):
                    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t), "-i", dst,
                                    "-frames:v", "1", os.path.join(OUT, f"{tag}_t{int(t*10):02d}.png")])
            res[tag] = {"seconds": round(dt, 1), "file": out}
        except Exception as exc:
            res[tag] = {"error": repr(exc)[:1500]}
        print(tag, res[tag], flush=True)
        json.dump(res, open(os.path.join(OUT, "timings.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
