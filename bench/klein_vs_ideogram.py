"""FLUX.3 Image Klein vs our Ideogram 4 on the same scenes, seeds and boxes (3090).

One scene table feeds both models: Ideogram gets the structured caption our
pipeline builds (imaging/ideogram_layout), Klein gets BFL's native input -- a
caption paragraph citing <ids> plus a JSON element table with [y_min, x_min,
y_max, x_max] boxes on the same 0-1000 grid. The prompt upsampler is replaced by
this hand-written table (the upper bound of what it can give).

    venv/Scripts/python bench/klein_vs_ideogram.py --variants ig_app,ig_full,kl_dist_int8
Outputs: runtime/klein_vs_ideogram/<variant>/s<N>.png + timings.json, sheet.png.
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
sys.path.insert(0, ROOT)
import night_firered  # noqa: E402
from night_firered import run  # noqa: E402
from imaging.ideogram_layout import build_caption, element  # noqa: E402

OUT = os.path.join(ROOT, "runtime", "klein_vs_ideogram")

# kind: txt (lettering) | obj
SCENES = [
    {"name": "bakery", "bg": "a cobblestone street at dusk",
     "high": "A cozy bakery storefront at dusk with a hand-painted sign.",
     "look": ("warm, inviting", "golden window light", "35mm street photo"),
     "klein": "Full-color outdoor photograph, a cozy bakery storefront on a cobblestone street at dusk, warm golden window light, 35mm street photo. Above the door hangs a hand-painted sign <Ru_Text_1>; below it a bakery window <window_1> is full of bread loaves.",
     "els": [("Ru_Text_1", "txt", [80, 200, 220, 800], "ХЛЕБ И СОЛЬ", "gold serif sign lettering above the door"),
             ("window_1", "obj", [300, 100, 950, 900], "", "bakery window full of bread loaves")]},
    {"name": "fisherman", "bg": "a grey sea and a wooden pier",
     "high": "Portrait of an old fisherman on a pier.",
     "look": ("documentary", "overcast soft light", "85mm portrait, shallow depth of field"),
     "klein": "Full-color outdoor photograph, documentary portrait of an old fisherman <person_1> standing on a wooden pier above a grey sea, overcast soft light, 85mm portrait lens with shallow depth of field.",
     "els": [("person_1", "obj", [100, 250, 1000, 750], "", "weathered old fisherman in a knitted sweater, grey beard")]},
    {"name": "street3", "bg": "a narrow old town street, afternoon",
     "high": "A narrow old-town street with a pharmacy, a cafe and a cat by the entrance.",
     "look": ("realistic", "soft afternoon daylight", "35mm street photo"),
     "klein": "Full-color outdoor photograph, a narrow old-town street in soft afternoon daylight. In the upper left a green pharmacy sign <Ru_Text_1>; in the upper right a cafe sign <Ru_Text_2>; in the centre a wooden door <door_1> with a small plate <Ru_Text_3>; at the lower right a ginger cat <cat_1> sits on the cobblestones.",
     "els": [("Ru_Text_1", "txt", [60, 60, 200, 420], "АПТЕКА", "white block letters on a green pharmacy signboard"),
             ("Ru_Text_2", "txt", [60, 560, 200, 940], "КАФЕ «УЮТ»", "cream cursive lettering on a dark brown cafe signboard"),
             ("door_1", "obj", [330, 380, 800, 620], "", "heavy wooden door with iron handle"),
             ("Ru_Text_3", "txt", [260, 430, 320, 570], "ВХОД", "small brass plate with engraved lettering above the door"),
             ("cat_1", "obj", [700, 700, 960, 900], "", "ginger cat sitting upright")]},
    {"name": "poster", "bg": "a dark teal vintage film poster",
     "high": "A vintage cinema poster with a steam train in the middle and a title on top.",
     "look": ("retro", "dramatic backlight", ""),
     "klein": "Vintage cinema poster illustration on a dark teal background. At the top a large title <Ru_Text_1>; in the middle a black steam locomotive <train_1> rushes towards the viewer through fog; at the bottom a credit line <Ru_Text_2>.",
     "els": [("Ru_Text_1", "txt", [40, 120, 220, 880], "ПОСЛЕДНИЙ ПОЕЗД", "huge cream condensed capital letters, slightly worn print"),
             ("train_1", "obj", [260, 180, 800, 820], "", "black steam locomotive with glowing headlight, smoke and fog"),
             ("Ru_Text_2", "txt", [860, 200, 960, 800], "Фильм Алексея Громова", "small cream serif lettering")]},
]


def ig_caption(s):
    a, l, p = s["look"]
    els = [element(d, box, text=t) if k == "txt" else element(d, box) for _id, k, box, t, d in s["els"]]
    return build_caption(s["bg"], els, high_level=s["high"], aesthetics=a, lighting=l, photo=p,
                         medium="illustration" if s["name"] == "poster" else "photograph")


def klein_prompt(s, layout):
    if not layout:
        return s["klein"]
    table = []
    for _id, k, box, t, d in s["els"]:
        desc = d if not t else f'{d}. Text: "{t}"'
        table.append({"id": _id, "bbox": box, "desc": desc})
    return s["klein"] + " " + json.dumps(table, ensure_ascii=False, separators=(",", ":"))


KLEIN = {  # variant -> (unet, clip, steps, cfg)
    "kl_dist_int8": ("klein\\flux3-image-klein-int8-convrot.safetensors", "klein\\qwen3vl_4b_int8_convrot.safetensors", 16, 1.0),
    "kl_base_int8": ("klein\\flux3-image-klein-base-int8-convrot.safetensors", "klein\\qwen3vl_4b_int8_convrot.safetensors", 50, 3.5),
    "kl_dist_bf16": ("klein\\flux3-image-klein-bf16.safetensors", "klein\\qwen3vl_4b_bf16.safetensors", 16, 1.0),
}


def klein_wf(prompt, seed, unet, clip, steps, cfg):
    """API form of the template's 'Diffusion (FLUX 3 Klein)' subgraph; mu = log(alpha=16)."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "flux3", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "klein\\flux2-vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "7": {"class_type": "ModelSamplingFlux", "inputs": {"model": ["1", 0], "max_shift": 2.77259, "base_shift": 2.77259,
                                                           "width": 1024, "height": 1024}},
        "8": {"class_type": "KSampler", "inputs": {"model": ["7", 0], "positive": ["4", 0], "negative": ["5", 0],
                                                    "latent_image": ["6", 0], "seed": seed, "steps": steps, "cfg": cfg,
                                                    "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"filename_prefix": "klein_ab", "images": ["9", 0]}},
    }


def ig_wf(base, cap, seed, mode):
    wf = json.loads(json.dumps(base))
    wf["167"]["inputs"]["text"] = json.dumps(cap, separators=(",", ":"), ensure_ascii=False)
    wf["165"]["inputs"]["noise_seed"] = seed
    if mode == "ig_app":     # what imaging/ideogram.py renders: turbo LoRA, 4 steps, cfg 1.5
        wf.pop("184")
        wf["300"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "lora_name": "ideogram_4_turbotime_v1.safetensors", "strength_model": 1.0, "model": ["166", 0]}}
        wf["198"]["inputs"]["model"] = ["300", 0]
        wf["182"]["inputs"]["model"] = ["198", 0]
        wf["182"]["inputs"]["cfg"] = 1.5
        wf["190"]["inputs"]["steps"] = 4
    else:                    # ig_full: the stock 20-step cfg 7 path
        wf["190"]["inputs"]["steps"] = 20
    return wf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="ig_app,ig_full,kl_dist_int8")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--no-layout", action="store_true", help="Klein without the JSON box table (words only)")
    ap.add_argument("--comfy", default="")
    a = ap.parse_args()
    if a.comfy:
        night_firered.COMFY = a.comfy
    import llama_backend
    try:
        llama_backend.unload_lmstudio_chat()
    except Exception as exc:
        print("unload LLM failed:", exc, flush=True)
    base = json.load(open(os.path.join(ROOT, "workflows", "image", "workflow_ideogram4.json"), encoding="utf-8"))
    idx = [int(i) for i in a.scenes.split(",")] if a.scenes else range(len(SCENES))
    for v in a.variants.split(","):
        d = os.path.join(OUT, v + ("_nolayout" if a.no_layout and v.startswith("kl") else ""))
        os.makedirs(d, exist_ok=True)
        res = []
        for i in idx:
            s = SCENES[i]
            seed = 1234 + i
            try:
                if v.startswith("ig"):
                    wf = ig_wf(base, ig_caption(s), seed, v)
                else:
                    wf = klein_wf(klein_prompt(s, not a.no_layout), seed, *KLEIN[v])
                img, dt = run(wf)
                p = os.path.join(d, f"s{i}_{s['name']}.png")
                open(p, "wb").write(img)
                res.append({"scene": s["name"], "seconds": round(dt, 1), "file": p})
                print(f"{v} {s['name']} {dt:.1f}s", flush=True)
            except Exception as exc:
                res.append({"scene": s["name"], "error": str(exc)[:600]})
                print(f"{v} {s['name']} ERROR {exc}", flush=True)
        json.dump(res, open(os.path.join(d, "timings.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
