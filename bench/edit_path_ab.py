"""A change to a part of the picture (recolour, re-dress, reshape): the contained mask path
against Klein on the whole frame with the annotator's box lock.

    venv/Scripts/python.exe bench/edit_path_ab.py [case ...]
Writes tmp/edit_path_ab/<case>_{mask,whole}.png and a side-by-side <case>.jpg to judge by eye.
The mask path pasted a light halo around a blue shirt and left a dark rectangle of brown hair
at the face box (2026-10-22); Klein on the whole frame drew both clean.
"""
import json, os, shutil, sys, threading
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.join(ROOT, "bench")]
os.chdir(ROOT); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from models import Context
import image as IMG
from PIL import Image, ImageDraw

CASES = {   # name: (source, region, instructions) -- as the agent sent them, from the logs
    "shirt": ("tg_images/tg_1792609373217_fc8628.jpg", "shirt", "a solid blue button-down shirt without patterns"),
    "hair": ("tg_images/tg_1792661776087_85d1a5.jpg", "hair", "ginger hair"),
    "tie": ("tg_images/tg_1792621743559_49eedb.jpg", "tie", "a solid red tie"),
    "nose": ("tg_images/tg_1792619681044_0d9baa.jpg", "nose", "a larger, more prominent nose"),
    "helmet": ("runtime/comfy/ideogram_00184_.png", "space helmet", "a shiny, polished gold space helmet"),
    "armor_girl": ("runtime/_working_input_1792194794750.png", "the armor of the girl on the right", "red armor"),
    "armor_man": ("runtime/upscaled_faceguard_1792207590192.png", "man with a shovel",
                  "the man wearing red and black plate armor"),
    "hand_lenin": ("runtime/comfy/klein_edit_00100_.png", "the left hand of Lenin",
                   "a clean five-fingered hand punching a demon in the face"),
    "hand_baker": ("runtime/comfy/transfer_00003_.png", "the right hand of the baker",
                   "a clean five-fingered human hand holding bread correctly without any fusion"),
}
OUT = os.path.join("tmp", "edit_path_ab")


def _ctx():
    c = Context(models=None, transcription_cache={}, cache_file=None, asr_lock=threading.Lock(),
                tts_lock=threading.Lock())
    c.set_stage = lambda *a, **k: None
    return c


def run(name):
    src, region, instr = CASES[name]
    ctx = _ctx()
    mask = IMG.inpaint_region_with_comfy(ctx, src, region, instr)
    whole = IMG._edit_image_with_firered_once(ctx, src, f"change the {region} to {instr}", seed=1234567)
    paths = {}
    for k, p in (("mask", mask), ("whole", whole)):
        if p:
            paths[k] = os.path.join(OUT, f"{name}_{k}.png")
            shutil.copy(p, paths[k])
    ims = [("source", src)] + [(k, paths[k]) for k in ("mask", "whole") if k in paths]
    H = 420
    tiles = [(lab, Image.open(p).convert("RGB")) for lab, p in ims]
    tiles = [(lab, im.resize((int(im.width * H / im.height), H))) for lab, im in tiles]
    sheet = Image.new("RGB", (sum(t.width for _, t in tiles) + 10 * len(tiles), H + 24), "white")
    x = 0
    for lab, t in tiles:
        sheet.paste(t, (x, 24)); ImageDraw.Draw(sheet).text((x + 4, 4), f"{name}: {lab}", fill="black")
        x += t.width + 10
    sheet.save(os.path.join(OUT, f"{name}.jpg"), quality=88)
    print(name, {k: bool(v) for k, v in (("mask", mask), ("whole", whole))}, flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for n in sys.argv[1:] or CASES:
        run(n)
