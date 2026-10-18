"""Editing: FLUX.3 Klein int8 (image-to-image, reference latent) vs our FireRed 1.1, same picture, same instruction.

Source is Ideogram's fisherman from bench/klein_vs_ideogram.py. Klein runs on a second ComfyUI (:8010,
master, has the flux3 CLIP type); FireRed on the live one (:8000) through night_firered's workflow.

    venv/Scripts/python bench/klein_edit_vs_firered.py --variants firered,klein
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
import night_firered as nf  # noqa: E402

OUT = os.path.join(ROOT, "runtime", "klein_edit_hard")
V1 = os.path.join(ROOT, "runtime", "klein_vs_ideogram", "ig_full")
V2 = os.path.join(ROOT, "runtime", "klein_hard", "ig_full")
BS = chr(92)
# (source picture, instruction); the hard set: outfit swaps, lettering removal/replacement, several objects at once
EDITS = [
    (V1 + "/s1_fisherman.png", "replace the sweater with a bright yellow fisherman's raincoat and put a yellow sou'wester hat on his head, keep his face, beard and the background unchanged"),
    (V1 + "/s1_fisherman.png", "dress him in a black suit with a white shirt and a dark red tie, keep his face, pose and the background unchanged"),
    (V1 + "/s2_street3.png", "remove all lettering from the three signs: the green pharmacy board, the brown cafe board and the small plate above the door must stay as blank boards, keep everything else unchanged"),
    (V1 + "/s2_street3.png", "change the lettering on the brown cafe sign from 'КАФЕ «УЮТ»' to 'БАР «ЛУНА»', same style and colour, keep everything else unchanged"),
    (V2 + "/s0_market.png", "change the boy's yellow raincoat to a blue one and the vendor woman's red apron to a green one, keep all people, faces and the rest unchanged"),
    (V2 + "/s0_market.png", "remove the lettering from the arch sign and from the price tags, leave the boards blank, keep everything else unchanged"),
    (V2 + "/s1_bookshop.png", "remove the sleeping cat from the counter and the stack of books from the floor, fill the spots naturally, keep everything else unchanged"),
    (V2 + "/s3_menu.png", "change the price of the borscht line from 280 to 310 rubles, keep the handwriting and everything else unchanged"),
]


def klein_wf(img, prompt, seed, lora=False):
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "klein" + BS + "flux3-image-klein-int8-convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "klein" + BS + "qwen3vl_4b_int8_convrot.safetensors", "type": "flux3", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "klein" + BS + "flux2-vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "11": {"class_type": "LoadImage", "inputs": {"image": img}},
        "12": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["11", 0], "upscale_method": "lanczos",
                                                                    "megapixels": 1.0, "resolution_steps": 16}},
        "13": {"class_type": "GetImageSize", "inputs": {"image": ["12", 0]}},
        "14": {"class_type": "ReferenceMultiLatent", "inputs": {"positive": ["4", 0], "negative": ["5", 0], "vae": ["3", 0],
                                                               "resolution": "resize", "resolution.megapixels": 1.0,
                                                               "images.image0": ["12", 0]}},
        "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": ["13", 0], "height": ["13", 1], "batch_size": 1}},
        "7": {"class_type": "ModelSamplingFlux", "inputs": {"model": ["1", 0], "max_shift": 2.77259, "base_shift": 2.77259,
                                                           "width": ["13", 0], "height": ["13", 1]}},
        "8": {"class_type": "KSampler", "inputs": {"model": ["7", 0], "positive": ["14", 0], "negative": ["14", 1],
                                                    "latent_image": ["6", 0], "seed": seed, "steps": 16, "cfg": 1.0,
                                                    "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"filename_prefix": "klein_edit", "images": ["9", 0]}},
    }
    if lora:   # ausboss Consistency LoRA: on the distilled model at 1.0, reference at keep resolution
        wf["20"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["1", 0], "lora_name": "klein" + BS + "flux-3-image-klein-consistency.safetensors", "strength_model": 1.0}}
        wf["7"]["inputs"]["model"] = ["20", 0]
        wf["14"]["inputs"]["resolution"] = "keep resolution"
        wf["14"]["inputs"].pop("resolution.megapixels")
    return wf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="firered,klein")
    a = ap.parse_args()
    base = json.load(open(os.path.join(ROOT, "workflows", "image", "workflow_firered_edit.json"), encoding="utf-8"))
    for v in a.variants.split(","):
        d = os.path.join(OUT, v)
        os.makedirs(d, exist_ok=True)
        res = []
        nf.COMFY = "http://127.0.0.1:8010" if v.startswith("klein") else "http://127.0.0.1:8000"
        for i, (src, instr) in enumerate(EDITS):
            try:
                name = nf.upload(src)
                if v.startswith("klein"):
                    wf = klein_wf(name, instr, 43, lora=v == "klein_lora")
                else:
                    wf = nf.variant(base, "bf16")
                    wf["143"]["inputs"]["image"] = name
                    wf["187"]["inputs"]["prompt"] = instr
                    wf["189"]["inputs"]["seed"] = 43
                img, dt = nf.run(wf)
                p = os.path.join(d, f"e{i}.png")
                open(p, "wb").write(img)
                res.append({"scene": f"e{i}", "seconds": round(dt, 1), "file": p})
                print(f"{v} e{i} {dt:.1f}s", flush=True)
            except Exception as exc:
                res.append({"scene": f"e{i}", "error": str(exc)[:600]})
                print(f"{v} e{i} ERROR {exc}", flush=True)
        json.dump(res, open(os.path.join(d, "timings.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
