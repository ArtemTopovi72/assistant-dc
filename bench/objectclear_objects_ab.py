"""ObjectClear vs FireRed on OBJECT removal (lettering was settled 2026-09-25).

Same SAM3 mask for both arms, so only the fill engine differs. Output:
runtime/oc_obj_ab/<obj>_{mask,oc,fr}.png + a side-by-side sheet.
    venv/Scripts/python bench/objectclear_objects_ab.py IMG "blue vase with flowers" "cat"
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import image as I
import image_contained_firered as CF
import image_lettering_remove as L

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime", "oc_obj_ab")
os.makedirs(OUT, exist_ok=True)


class Ctx:
    def set_stage(self, *a, **k): pass
    def is_cancelled(self): return False


def main(img, objs):
    ctx = Ctx()
    for obj in objs:
        tag = obj.split()[-1]
        r = CF._contained_region_mask(ctx, img, obj, grow=12, seed=42, timeout=600)
        if not r:
            print(obj, "NO MASK"); continue
        mpath = os.path.join(OUT, f"{tag}_mask.png"); r[1].save(mpath)
        t = time.time(); oc = L._fill_objectclear(ctx, img, mpath, 12); t_oc = time.time() - t
        t = time.time()
        with I.firered_extra_lora(I.REMOVAL_LORA, I.REMOVAL_LORA_STRENGTH):
            fr = CF.edit_region_contained_via_firered(
                ctx, img, obj, f"Remove the {obj} completely; fill the space with the background "
                "that would be behind it, matching texture, light and perspective.",
                grow=12, mask_override=mpath, vlm_qa=False)
        t_fr = time.time() - t
        src = Image.open(img).convert("RGB"); w, h = src.size
        sheet = Image.new("RGB", (w * 3, h), "black")
        for i, p in enumerate((img, oc, fr)):
            if p: sheet.paste(Image.open(p).convert("RGB").resize((w, h)), (w * i, 0))
        sheet.save(os.path.join(OUT, f"{tag}_sheet.png"))
        print(f"{obj}: ObjectClear {t_oc:.0f}s -> {oc}\n{' ' * len(obj)}  FireRed {t_fr:.0f}s -> {fr}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
