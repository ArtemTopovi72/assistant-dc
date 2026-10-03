"""A/B Ideogram 4 transformer formats on the 3090: fp8_scaled (ours) vs int8.

Ampere has no FP8 cores, so fp8 weights are upcast every layer; ComfyUI's
comfy-kitchen runs int8 on the integer tensor cores. Same seed, same captions;
outputs + timings go to runtime/ideogram_int8_ab/<variant>/.

    venv/Scripts/python bench/ideogram_int8_ab.py --variants fp8,int8mr
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
from night_firered import run  # noqa: E402

OUT = os.path.join(ROOT, "runtime", "ideogram_int8_ab")
UNETS = {
    "fp8": ("ideogram4_fp8_scaled.safetensors", "ideogram4_unconditional_fp8_scaled.safetensors"),
    "int8mr": ("ig4-int8mixedrow_simple.safetensors", "ig4_uncond-int8mixedrow_simple.safetensors"),
}
sys.path.insert(0, ROOT)
from imaging.ideogram_layout import build_caption, element  # noqa: E402

CAPTIONS = [
    build_caption("a cobblestone street at dusk", [
        element("gold serif sign lettering above the door", [80, 200, 220, 800], text="ХЛЕБ И СОЛЬ"),
        element("bakery window full of bread loaves", [300, 100, 950, 900])],
        high_level="A cozy bakery storefront at dusk with a hand-painted sign.",
        aesthetics="warm, inviting", lighting="golden window light", photo="35mm street photo"),
    build_caption("a grey sea and a wooden pier", [
        element("weathered old fisherman in a knitted sweater, grey beard", [100, 250, 1000, 750])],
        high_level="Portrait of an old fisherman on a pier.",
        aesthetics="documentary", lighting="overcast soft light", photo="85mm portrait, shallow depth of field"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="fp8,int8mr")
    a = ap.parse_args()
    base = json.load(open(os.path.join(ROOT, "workflows", "image", "workflow_ideogram4.json"), encoding="utf-8"))
    for v in a.variants.split(","):
        d = os.path.join(OUT, v)
        os.makedirs(d, exist_ok=True)
        res = []
        for i, cap in enumerate(CAPTIONS):
            wf = json.loads(json.dumps(base))
            if v in UNETS:
                wf["166"]["inputs"]["unet_name"], wf["181"]["inputs"]["unet_name"] = UNETS[v]
            wf["167"]["inputs"]["text"] = json.dumps(cap, separators=(",", ":"), ensure_ascii=False)
            wf["165"]["inputs"]["noise_seed"] = 1234 + i
            if v.startswith("app"):
                # what imaging/ideogram.py renders: turbo LoRA, 4 steps, cfg 1.5
                # through the dual-model guider (the uncond model still runs)
                wf["166"]["inputs"]["unet_name"], wf["181"]["inputs"]["unet_name"] = UNETS["int8mr"]
                wf.pop("184")
                wf["300"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
                    "lora_name": "ideogram_4_turbotime_v1.safetensors", "strength_model": 1.0,
                    "model": ["166", 0]}}
                wf["198"]["inputs"]["model"] = ["300", 0]
                wf["182"]["inputs"]["model"] = ["198", 0]
                wf["182"]["inputs"]["cfg"] = 1.5
                wf["190"]["inputs"]["steps"] = 4
            elif v == "flash":
                # flash-attn 2 for head_dim 256 (SDPA 65 ms vs FA2 35 ms per call, 3090)
                wf["210"] = {"class_type": "PatchFlashAttentionKJ", "inputs": {"model": ["166", 0]}}
                wf["211"] = {"class_type": "PatchFlashAttentionKJ", "inputs": {"model": ["181", 0]}}
                wf["198"]["inputs"]["model"] = ["210", 0]
                wf["182"]["inputs"]["model_negative"] = ["211", 0]
            elif v in ("zjourney", "lenovo"):
                # skin realism LoRAs (civitai 2686999 / 1662740), conditional branch only
                f = {"zjourney": "zjourneyv2.safetensors", "lenovo": "lenovo_ideogram4.safetensors"}[v]
                wf["300"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
                    "lora_name": f, "strength_model": 0.8, "model": ["166", 0]}}
                wf["198"]["inputs"]["model"] = ["300", 0]
            elif v.startswith("turbo"):
                # ostris TurboTime LoRA: few steps, cfg 1, no unconditional model
                wf["166"]["inputs"]["unet_name"] = UNETS["int8mr"][0]
                for k in ("181", "182", "184", "159"):
                    wf.pop(k)
                wf["300"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
                    "lora_name": "ideogram_4_turbotime_v1.safetensors", "strength_model": 1.0,
                    "model": ["166", 0]}}
                wf["198"]["inputs"]["model"] = ["300", 0]
                wf["301"] = {"class_type": "CFGGuider", "inputs": {
                    "cfg": 1.0, "model": ["198", 0], "positive": ["167", 0], "negative": ["167", 0]}}
                wf["161"]["inputs"]["guider"] = ["301", 0]
                wf["190"]["inputs"]["steps"] = int(v[5:])
            try:
                img, dt = run(wf)
                p = os.path.join(d, f"c{i}.png")
                open(p, "wb").write(img)
                res.append({"caption": i, "seconds": round(dt, 1), "file": p})
                print(f"{v} c{i} {dt:.1f}s", flush=True)
            except Exception as exc:
                res.append({"caption": i, "error": str(exc)[:500]})
                print(f"{v} c{i} ERROR {exc}", flush=True)
        json.dump(res, open(os.path.join(d, "timings.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
