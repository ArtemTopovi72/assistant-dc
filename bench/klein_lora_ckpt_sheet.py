"""Character LoRA checkpoints on the distilled Klein: step x strength contact sheet.

    venv/Scripts/python.exe bench/klein_lora_ckpt_sheet.py neurostepan 1500,1750,2000 0.75,1.0,1.25

Checkpoints are copied into ComfyUI's loras folder as zz_eval_* (no "klein" in the name, so the app
never loads them) and removed afterwards. Same seed and prompts for every cell; sheets go to
runtime/klein_lora_ckpt/.
"""
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:] = [x for x in sys.path if os.path.abspath(x) != os.path.dirname(os.path.abspath(__file__))]
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")
import tg_bot  # noqa: E402,F401
import image  # noqa: E402
import klein  # noqa: E402
import lora_training as LT  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

slug = sys.argv[1]
steps = [int(x) for x in sys.argv[2].split(",")]
strengths = [float(x) for x in sys.argv[3].split(",")]
OUT = os.path.join(ROOT, "runtime", "klein_lora_ckpt")
os.makedirs(OUT, exist_ok=True)
RUN = LT.TRAIN_ROOT / (slug + "_klein") / (slug + "_klein")
PROMPTS = {
    "portrait": f"Photograph of {slug}, a man, close-up portrait, plain light wall, natural window light, sharp focus on the eyes and lips, visible skin texture.",
    "street": f"Photograph of {slug}, a man in a dark coat walking down a city street in daylight, half body, looking at the camera, sharp detail.",
}


class Ctx:
    pass


cells = {}
for st in steps:
    src = RUN / f"{slug}_klein_{st:09d}.safetensors"
    if not src.exists():          # the last save carries no step number
        src = RUN / f"{slug}_klein.safetensors"
    name = f"zz_eval_{slug}_{st}.safetensors"
    try:
        shutil.copy2(str(src), str(LT.LORA_OUT_DIR / name))
    except OSError:     # ComfyUI still holds the copy from an earlier run of this script
        pass
    for s in strengths:
        for pk, prompt in PROMPTS.items():
            t = time.time()
            wf = klein.t2i_workflow(prompt, 1024, 1024, 7, lora=(name, s))
            p = image._submit_and_poll(Ctx(), wf, timeout=900, label=f"{st}_{s}_{pk}", exclusive=True)
            if p:
                dst = os.path.join(OUT, f"{slug}_{pk}_{st}_{s}.png")
                shutil.copy2(p, dst)
                cells[(pk, st, s)] = dst
            print("OK " if p else "FAIL", pk, st, s, "%.0fs" % (time.time() - t), flush=True)

for pk in PROMPTS:
    W = Image.new("RGB", (512 * len(strengths), 540 * len(steps)), "white")
    d = ImageDraw.Draw(W)
    for r, st in enumerate(steps):
        for c, s in enumerate(strengths):
            if (pk, st, s) in cells:
                W.paste(Image.open(cells[(pk, st, s)]).convert("RGB").resize((512, 512)), (512 * c, 540 * r + 28))
            d.text((512 * c + 6, 540 * r + 8), f"step {st}  strength {s}", fill="black")
    W.save(os.path.join(OUT, f"{slug}_{pk}_sheet.jpg"), quality=88)
print("sheets:", OUT)
print("ComfyUI keeps the zz_eval_* copies open: delete them from", LT.LORA_OUT_DIR, "after it restarts")
