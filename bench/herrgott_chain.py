"""The chef chain through Herrgott's H3 Infinite Continuation Suite v1.4 (native masked
AV continuation: the previous latent is copied into the new one and protected by the
per-stream denoise mask), on our models. Clip 1 = Start, clips 2..3 = Continue, then the
suite's own StitchSavedChain joins the saved latents.

    venv/Scripts/python bench/herrgott_chain.py [tag] [steps] [lora 0/1] [sampler]
"""
import json, os, sys, time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "http://127.0.0.1:8000"
TAG = sys.argv[1] if len(sys.argv) > 1 else "hg"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
LORA = (sys.argv[3] if len(sys.argv) > 3 else "1") == "1"
SAMPLER = sys.argv[4] if len(sys.argv) > 4 else "euler"
W, H, SEC = 768, 1024, 6.0
PREFIX = f"h3_continuous/{TAG}"
PARTS = [
    "A grey-haired man in a blue apron stands at a wooden kitchen counter, picks up a large knife, "
    "and starts chopping carrots fast while saying «Ну что, начнём готовить!»",
    "He reaches to the stove at the left to grab the pot, then picks up a wooden spoon from the counter "
    "and a salt shaker from off-frame. Then he sweeps the chopped carrots into a steaming pot with the flat of the knife, stirs it with a "
    "wooden spoon, tastes from the spoon, frowns, then shakes salt in and says «Соли маловато, "
    "сейчас исправим», and keeps stirring for a few seconds.",
    "He smiles, turns to the camera and says «Вот теперь отлично!»",
]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def base() -> dict:
    wf = {
        "unet": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-convrot.safetensors",
            "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoaderGGUF", "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3-Q4_K_M.gguf", "type": "minimax"}},
        "vv": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "va": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["unet", 0], "strength_model": 1.0 if LORA else 0.0,
            "lora_name": "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors"}},
        "shift": {"class_type": "MiniMaxH3SigmaShift", "inputs": {
            "model": ["lora", 0], "shift_video": 12.0, "shift_audio": 3.0}},
    }
    return wf


def tail(wf: dict, idx: int, cond: str, head=None, mode="Stitch Ready") -> dict:
    wf.update({
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": [cond, 0]}},
        "ks": {"class_type": "KSampler", "inputs": {
            "model": ["shift", 0], "positive": [cond, 0], "negative": ["neg", 0], "latent_image": [cond, 1],
            "seed": 424242 + idx, "steps": STEPS, "cfg": 1.0, "sampler_name": SAMPLER,
            "scheduler": "simple", "denoise": 1.0}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vv", 0]}},
        "deca": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["ks", 0], "vae": ["va", 0]}},
        "an": {"class_type": "H3ContinuousAnalyzeHandoverV14", "inputs": {
            "images": ["dec", 0], "preset": "Balanced", "analysis_window": 72, "freeze_hold": 8,
            "safety_margin": 3, "context_frames": "39", "analysis_size": 192,
            "final_mean_diff_threshold": 0.012, "final_active_pixel_threshold": 0.025,
            "max_final_active_area_percent": 3.0, "transition_mean_diff_threshold": 0.002,
            "transition_active_pixel_threshold": 0.01, "max_transition_active_area_percent": 1.0,
            "min_static_transition_percent": 70.0, "max_consecutive_motion_outliers": 2,
            "final_reference_frames": 15, "min_final_match_percent": 75.0,
            "max_consecutive_final_outliers": 3, "safety_mode": "fixed"}},
        "save": {"class_type": "H3ContinuousSaveLatent", "inputs": {
            "latent": ["ks", 0], "filename_prefix": PREFIX, "clip_index": idx, "handover": ["an", 0]}},
        "full": {"class_type": "CreateVideo", "inputs": {"images": ["dec", 0], "audio": ["deca", 0], "fps": 24.0}},
        "savev": {"class_type": "SaveVideo", "inputs": {
            "video": ["full", 0], "filename_prefix": f"video/{TAG}_full{idx}", "format": "auto", "codec": "auto"}},
    })
    if head:
        wf["save"]["inputs"]["head_context_frames"] = head
    return wf


def run(wf: dict) -> dict:
    r = requests.post(URL + "/prompt", json={"prompt": wf}, timeout=30)
    if r.status_code != 200:
        raise SystemExit(f"rejected: {r.text[:2000]}")
    pid = r.json()["prompt_id"]
    while True:
        time.sleep(3)
        h = requests.get(f"{URL}/history/{pid}", timeout=30).json()
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
                raise SystemExit(f"failed: {json.dumps(msgs)[:3000]}")
            return h[pid]["outputs"]


def files(out: dict) -> list:
    got = []
    for v in out.values():
        for k in ("images", "videos", "gifs"):
            for f in v.get(k, []):
                got.append(os.path.join(f.get("subfolder", ""), f["filename"]))
        for k in ("text", "string"):
            got += [str(x)[:300] for x in v.get(k, [])]
    return got


def main():
    # the app frees the card before a render; a bench posting straight to ComfyUI must too
    # (10-07: Gemma's 18 GB stayed loaded beside H3 for a whole run)
    sys.path.insert(0, os.path.join(ROOT, "agent"))
    import llama_backend
    llama_backend.unload_lmstudio_chat()
    for i, part in enumerate(PARTS, start=1):
        wf = base()
        if i == 1:
            wf["cond"] = {"class_type": "H3ContinuousStartV14", "inputs": {
                "clip": ["clip", 0], "vae": ["vv", 0], "prompt": part, "width": W, "height": H,
                "duration": SEC, "ref_image_size": "match"}}
            tail(wf, i, "cond")
        else:
            wf["load"] = {"class_type": "H3ContinuousLoadLatent", "inputs": {"latent_path": PREFIX.rsplit("/", 1)[0],
                                                                              "clip_index": i - 1}}
            wf["cond"] = {"class_type": "H3ContinuousContinueV14", "inputs": {
                "clip": ["clip", 0], "vae": ["vv", 0], "previous_latent": ["load", 0], "handover": ["load", 3],
                "prompt": part, "width": W, "height": H, "duration": SEC, "masked_context_frames": "39",
                "audio_feather_ticks": 0, "ref_image_size": "match", "duration_mode": "Net New Content",
                "audio_tail_carryover": "Full Previous Tail"}}
            tail(wf, i, "cond", head=["cond", 2])
        t = time.time()
        out = run(wf)
        log(f"clip {i}: {time.time() - t:.0f}s", files(out))
    st = {"st": {"class_type": "H3ContinuousStitchSavedChainV14", "inputs": {
        "video_vae": ["vv", 0], "audio_vae": ["va", 0], "latent_prefix": PREFIX, "first_clip": 1, "last_clip": 3,
        "filename_prefix": f"video/{TAG}_stitched", "video_crossfade_frames": 4, "audio_crossfade_ms": 15.0,
        "luminance_match": False, "luminance_fade_frames": 16, "max_luminance_correction_percent": 10.0,
        "crf": 18, "max_safe_tail_bridge_frames": 0}},
          "vv": base()["vv"], "va": base()["va"]}
    t = time.time()
    out = run(st)
    log(f"stitched: {time.time() - t:.0f}s", files(out), json.dumps(out)[:600])


if __name__ == "__main__":
    main()
