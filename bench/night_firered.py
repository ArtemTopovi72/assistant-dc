"""Night A/B for FireRed Image Edit: transformer quant, extra LoRAs, recaption.

Runs workflow_firered_edit.json straight against ComfyUI with one knob changed
per variant, same seed and same edits, and writes every output plus timings to
runtime/night_firered/<variant>/. Judge the pictures afterwards.

    venv/Scripts/python bench/night_firered.py --variants bf16,fp8 [--reps 1]
"""
import argparse, copy, json, os, sys, time, uuid
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
COMFY = "http://127.0.0.1:8000"
OUT = os.path.join(ROOT, "runtime", "night_firered")

EDITS = [
    ("runtime/_working_input_1790934481665.png", "change the red-haired man's shirt to a bright blue tunic, keep both faces unchanged"),
    ("runtime/_working_input_1790934481665.png", "make it a snowy winter day, keep the two men and their faces unchanged"),
    ("runtime/_working_input_1790934481665.png", "remove the small figure in the background and its shadow, keep everything else unchanged"),
    ("runtime/_working_input_1790934481665.png", "add a large bold title 'ДЕДЫ ВОЕВАЛИ' in the sky in white poster lettering"),
]


def variant(wf, name):
    """Apply one variant to a copy of the workflow."""
    wf = copy.deepcopy(wf)
    if name == "bf16":
        pass
    elif name == "fp8":
        wf["184"]["inputs"]["unet_name"] = "night\\firered_image_edit_1.1_fp8_scaled_e4m3fn.safetensors"
        wf["184"]["class_type"] = "UNETLoader"
    elif name in ("q8", "q6"):
        f = {"q8": "FireRed-Image-Edit-1.1-Q8_0.gguf", "q6": "FireRed-Image-Edit-1.1-Q6_K.gguf"}[name]
        wf["184"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "night\\" + f}}
    elif name in ("makeup", "covercraft"):
        f = {"makeup": "FireRed-Image-Edit-Makeup.safetensors",
             "covercraft": "FireRed-Image-Edit-Covercraft.safetensors"}[name]
        wf["300"] = {"class_type": "LoraLoaderModelOnly",
                     "inputs": {"model": ["183", 0], "lora_name": "night\\" + f, "strength_model": 1.0}}
        wf["172"]["inputs"]["model"] = ["300", 0]
    elif name in ("makeup_full", "covercraft_full", "bf16_full"):
        # the Zoo LoRAs were trained on the base model: no Lightning, 40 steps, cfg 4
        wf["172"]["inputs"]["model"] = ["184", 0]
        if name != "bf16_full":
            f = {"makeup_full": "FireRed-Image-Edit-Makeup.safetensors",
                 "covercraft_full": "FireRed-Image-Edit-Covercraft.safetensors"}[name]
            wf["300"] = {"class_type": "LoraLoaderModelOnly",
                         "inputs": {"model": ["184", 0], "lora_name": "night\\" + f, "strength_model": 1.0}}
            wf["172"]["inputs"]["model"] = ["300", 0]
        wf["189"]["inputs"].update(steps=40, cfg=4.0)
    elif name in ("l11v11", "l11v12"):
        # the 1.1-native Lightning LoRAs (we ran 1.0's on the 1.1 model)
        wf["183"]["inputs"]["lora_name"] = "FireRed-Image-Edit-1.1-Lightning-8steps-v1.%s.safetensors" % name[-1]
    elif name == "te_bf16":
        wf["182"]["inputs"]["clip_name"] = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    elif name == "fp8mixed":
        wf["184"]["inputs"]["unet_name"] = "FireRed-Image-Edit-1.1_fp8mixed_comfy.safetensors"
    elif name == "zoomfix":
        # lrzjason QwenEditUtils: the encoder pads instead of squeezing to 1 MP,
        # the result is cropped back by pad_info -> no shift/zoom (Qwen-Edit #229)
        for nid, src in (("300", "187"), ("301", "188")):
            wf[nid] = {"class_type": "TextEncodeQwenImageEditPlusAdvance_lrzjason", "inputs": {
                "clip": ["182", 0], "vae": ["181", 0], "prompt": wf[src]["inputs"]["prompt"],
                "vl_resize_image1": ["143", 0], "target_size": 1024, "target_vl_size": 384,
                "upscale_method": "lanczos", "crop_method": "pad"}}
        wf["189"]["inputs"].update(positive=["300", 0], negative=["301", 0], latent_image=["300", 1])
        wf["302"] = {"class_type": "CropWithPadInfo", "inputs": {"image": ["190", 0], "pad_info": ["300", 9]}}
        wf["9"]["inputs"]["images"] = ["302", 0]
    elif name == "recaption":
        pass  # prompt rewritten by the caller
    else:
        raise SystemExit("unknown variant " + name)
    return wf


def upload(path):
    with open(path, "rb") as f:
        r = requests.post(COMFY + "/upload/image", files={"image": (os.path.basename(path), f)},
                          data={"overwrite": "true"}, timeout=60)
    r.raise_for_status()
    return r.json()["name"]


def _idle(max_wait=4 * 3600):
    """Wait for an empty ComfyUI queue: time spent behind another job is not render time."""
    t0 = time.time()
    while time.time() - t0 < max_wait:
        q = requests.get(COMFY + "/queue", timeout=30).json()
        if not q.get("queue_running") and not q.get("queue_pending"):
            return
        time.sleep(5)


def run(wf, timeout=1800):
    _idle()
    pid = requests.post(COMFY + "/prompt", json={"prompt": wf, "client_id": uuid.uuid4().hex},
                        timeout=60).json()
    if "prompt_id" not in pid:
        raise RuntimeError(json.dumps(pid)[:800])
    pid = pid["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = requests.get(f"{COMFY}/history/{pid}", timeout=30).json()
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                raise RuntimeError(json.dumps(st)[:1500])
            for node in h[pid]["outputs"].values():
                for im in node.get("images", []):
                    return requests.get(COMFY + "/view", params=im, timeout=60).content, time.time() - t0
        time.sleep(1.0)
    raise TimeoutError(pid)


def recaption(src, instr):
    """FireRed agent style: expand a short instruction into a detailed one."""
    import llm
    q = ("You write instructions for an image-EDITING model. Rewrite the short edit request below "
         "into ONE detailed English edit instruction (80-150 words): say exactly what changes "
         "(object, colour, material, placement, lettering exactly as quoted), and explicitly list what "
         "must stay unchanged (identity, face, pose, background, lighting, framing). No preamble.\n\n"
         "Request: " + instr)
    return (llm.analyze_image_with_llm(None, image_path=src, user_text=q, max_tokens=1500) or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="bf16")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--edits", default="")
    a = ap.parse_args()
    base = json.load(open(os.path.join(ROOT, "workflows", "image", "workflow_firered_edit.json"), encoding="utf-8"))
    idx = [int(i) for i in a.edits.split(",")] if a.edits else range(len(EDITS))
    for v in a.variants.split(","):
        d = os.path.join(OUT, v)
        os.makedirs(d, exist_ok=True)
        res = []
        for i in idx:
            src, instr = EDITS[i]
            prompt = instr
            if v == "recaption":
                cache = os.path.join(d, f"prompt_{i}.txt")
                if os.path.exists(cache):
                    prompt = open(cache, encoding="utf-8").read()
                else:
                    prompt = recaption(os.path.join(ROOT, src), instr)
                    open(cache, "w", encoding="utf-8").write(prompt)
            wf = variant(base, v)
            wf["143"]["inputs"]["image"] = upload(os.path.join(ROOT, src))
            wf["187"]["inputs"]["prompt"] = prompt
            if "300" in wf:
                wf["300"]["inputs"]["prompt"] = prompt
            for r in range(a.reps):
                # a repeat with the same seed is served from ComfyUI's cache in 1 s
                wf["189"]["inputs"]["seed"] = 43 + r
                try:
                    img, dt = run(wf)
                    p = os.path.join(d, f"e{i}_r{r}.png")
                    open(p, "wb").write(img)
                    res.append({"edit": i, "rep": r, "seconds": round(dt, 1), "file": p})
                    print(f"{v} e{i} r{r} {dt:.1f}s", flush=True)
                except Exception as exc:
                    res.append({"edit": i, "rep": r, "error": str(exc)[:500]})
                    print(f"{v} e{i} r{r} ERROR {exc}", flush=True)
        json.dump(res, open(os.path.join(d, "timings.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
