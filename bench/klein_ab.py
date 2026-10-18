"""Klein A/B grid from docs/design/klein_findings.md: alpha, Consistency LoRA strength, cfg, TE precision.

    venv/Scripts/python.exe bench/klein_ab.py alpha|cons|cfg|cyr|all

Same seeds and prompts inside a group; results and contact sheets go to runtime/klein_ab/.
"""
import math
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:] = [x for x in sys.path if os.path.abspath(x) != os.path.dirname(os.path.abspath(__file__))]  # bench/knowledge.py shadows the package
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")
import tg_bot  # noqa: E402,F401  (sets up the flat import path)
import image  # noqa: E402
import klein  # noqa: E402
from PIL import Image, ImageChops, ImageDraw  # noqa: E402

OUT = os.path.join(ROOT, "runtime", "klein_ab")
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(ROOT, "runtime", "klein_vs_ideogram", "ig_full", "s1_fisherman.png")


class Ctx:
    pass


def run(wf, name):
    t = time.time()
    p = image._submit_and_poll(Ctx(), wf, timeout=900, label=name, exclusive=True)
    if not p:
        print("FAIL", name)
        return None
    dst = os.path.join(OUT, name + ".png")
    shutil.copy2(p, dst)
    print("OK  %-28s %.1fs" % (name, time.time() - t), flush=True)
    return dst


def sheet(paths, labels, name):
    ims = [Image.open(p).convert("RGB").resize((512, 512)) for p in paths if p]
    W = Image.new("RGB", (512 * len(ims), 540), "white")
    d = ImageDraw.Draw(W)
    for i, (im, lb) in enumerate(zip(ims, labels)):
        W.paste(im, (512 * i, 28))
        d.text((512 * i + 6, 8), lb, fill="black")
    W.save(os.path.join(OUT, name + ".jpg"), quality=88)


BAKERY = ('Full-color photograph, a cozy bakery storefront at dusk with a hand-painted sign. '
          'Upper centre: gold serif sign lettering above the door <Ru_Text_1>. Lower centre: bakery window full of bread loaves <window_1>. '
          '[{"id":"Ru_Text_1","bbox":[80,200,220,800],"desc":"Gold serif sign lettering. Text: \\"ХЛЕБ И СОЛЬ\\""},'
          '{"id":"window_1","bbox":[300,100,950,900],"desc":"bakery window full of bread loaves"}]')


def alpha():
    ps, lb = [], []
    for a in (6.93, 16, 32):
        for seed in (7, 8):
            wf = klein.t2i_workflow(BAKERY, 1024, 1024, seed)
            mu = math.log(a)
            wf[klein.N_SHIFT]["inputs"].update(max_shift=mu, base_shift=mu)
            ps.append(run(wf, "alpha%g_s%d" % (a, seed)))
            lb.append("alpha %g seed %d" % (a, seed))
    sheet(ps, lb, "sheet_alpha")


def cons():
    src = Image.open(SRC).convert("RGB")
    up = image._upload_image_to_comfy(SRC, image.COMFY_URL)
    ps, lb = [], []
    for st in (0.6, 0.8, 1.0):
        klein_lora = klein.KLEIN_EDIT_LORA
        wf = klein.edit_workflow(up, klein.with_keep_clause("Dress the man in a black suit with a white shirt and a dark red tie."), 43)
        wf[klein.N_LORA]["inputs"]["strength_model"] = st
        p = run(wf, "cons%g" % st)
        if p:
            im = Image.open(p).convert("RGB").resize(src.size)
            diff = ImageChops.difference(src, im).convert("L")
            top = diff.crop((0, 0, src.size[0], src.size[1] // 5))     # background strip above the head
            mean = sum(top.getdata()) / (top.size[0] * top.size[1])
            print("   strength %.1f  mean |diff| in background strip %.2f" % (st, mean))
        ps.append(p)
        lb.append("cons %.1f" % st)
    sheet(ps, lb, "sheet_cons")
    del klein_lora


def cfg():
    p1 = "Full-color photograph, a woman holds a cup of tea with both hands, close view of her hands and the cup, soft window light."
    ps, lb = [], []
    for c in (1.0, 1.3, 1.6):
        for seed in (11, 12):
            wf = klein.t2i_workflow(p1, 1024, 1024, seed)
            wf[klein.N_SAMPLER]["inputs"]["cfg"] = c
            if c > 1.0:
                wf[klein.N_NEG]["inputs"]["text"] = "deformed hands, extra fingers, blurry"
            ps.append(run(wf, "cfg%g_s%d" % (c, seed)))
            lb.append("cfg %g seed %d" % (c, seed))
    sheet(ps, lb, "sheet_cfg")


def cyr():
    ps, lb = [], []
    for (w, h) in ((1024, 1024), (832, 1216), (1216, 832)):
        wf = klein.t2i_workflow(BAKERY, w, h, 5)
        ps.append(run(wf, "cyr_%dx%d" % (w, h)))
        lb.append("%dx%d" % (w, h))
    sheet(ps, lb, "sheet_cyr")


def caps():
    ps, lb = [], []
    for seed in (5, 7, 9):
        for tag, desc in (("plain", "Gold serif sign lettering. Text: \\\"ХЛЕБ И СОЛЬ\\\""),
                          ("caps", "Gold serif sign lettering in capital letters. Text: \\\"ХЛЕБ И СОЛЬ\\\" (all uppercase)")):
            txt = BAKERY.replace('Gold serif sign lettering. Text: \\"ХЛЕБ И СОЛЬ\\"', desc)
            wf = klein.t2i_workflow(txt, 1024, 1024, seed)
            ps.append(run(wf, "caps_%s_s%d" % (tag, seed)))
            lb.append("%s seed %d" % (tag, seed))
    sheet(ps, lb, "sheet_caps")


def te():
    ps, lb = [], []
    for clip in ("qwen3vl_4b_int8_convrot.safetensors", "qwen3vl_4b_bf16.safetensors"):
        for seed in (5, 7):
            wf = klein.t2i_workflow(BAKERY, 1024, 1024, seed)
            wf[klein.N_CLIP]["inputs"]["clip_name"] = clip
            ps.append(run(wf, "te_%s_s%d" % (clip[:14], seed)))
            lb.append("%s s%d" % (clip[:16], seed))
    sheet(ps, lb, "sheet_te")


if __name__ == "__main__":
    for g in (sys.argv[1:] or ["all"]):
        for fn in ((alpha, cons, cfg, cyr, caps, te) if g == "all" else (globals()[g],)):
            fn()
