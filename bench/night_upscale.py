"""Item 7: LBH latent upscaler two-stage vs single-stage at full size.

Stage 1 renders at half width/height with the app's own 4-step graph; the video
latent is upscaled 2x by MinimaxH3LatentUpscaler3D, re-joined with the audio
latent and refined with sigmas 0.6316 -> 0.3158 -> 0 (the upstream workflow3
tail). Same seed and prompt as night_video t1/t2.

    venv/Scripts/python bench/night_upscale.py [--cases t1,t2]
"""
import argparse, json, os, sys, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import video as V  # noqa: E402
from bench import night_video as NV  # noqa: E402

UPS = "night\\minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"


def build(case, prompt):
    mode, _, _ = NV.CASES[case]
    w, h = V.resolve_size("", None)
    wf = V.build_workflow(prompt, mode=mode, width=w // 2 // 32 * 32, height=h // 2 // 32 * 32,
                          frames=V.snap_frames(124), seed=NV.SEED, steps=V.VIDEO_STEPS)
    wf.update({
        "20": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["8", 0]}},
        "21": {"class_type": "MinimaxH3LatentUpscaler3D", "inputs": {
            "latent": ["20", 0], "model_name": UPS, "mode": "scale by multiplier", "mode.scale": 2.0,
            "align": 32, "enable_temporal_chunking": True, "force_unload": True,
            "device": "cuda", "precision": "fp16"}},
        "22": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["21", 0], "audio_latent": ["20", 1]}},
        "23": {"class_type": "ManualSigmas", "inputs": {"sigmas": "0.6316, 0.3158, 0"}},
        "24": {"class_type": "RandomNoise", "inputs": {"noise_seed": NV.SEED + 1}},
        "25": {"class_type": "CFGGuider", "inputs": {"model": ["2", 0], "positive": ["6", 0],
                                                    "negative": ["7", 0], "cfg": 1.0}},
        "26": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "27": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["24", 0], "guider": ["25", 0], "sampler": ["26", 0],
            "sigmas": ["23", 0], "latent_image": ["22", 0]}},
    })
    wf["9"]["inputs"]["samples"] = ["27", 0]
    wf["10"]["inputs"]["samples"] = ["27", 0]
    tag = f"night_ups_{case}_{uuid.uuid4().hex[:6]}"
    wf["12"]["inputs"]["filename_prefix"] = "video/" + tag
    return wf, tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="t1,t2")
    a = ap.parse_args()
    d = os.path.join(NV.OUT, "ups")
    os.makedirs(d, exist_ok=True)
    res = {}
    for case in a.cases.split(","):
        cache = os.path.join(NV.OUT, "cir", f"{case}_prompt.txt")
        prompt = open(cache, encoding="utf-8").read() if os.path.exists(cache) else NV.CASES[case][2]
        try:
            wf, tag = build(case, prompt)
            src, dt = NV.run(wf, tag)
            dst = os.path.join(d, case + ".mp4")
            open(dst, "wb").write(open(src, "rb").read())
            NV.frames(dst, os.path.join(d, case))
            res[case] = {"seconds": round(dt, 1)}
            print(f"ups {case} {dt:.0f}s", flush=True)
        except Exception as exc:
            res[case] = {"error": str(exc)[:1500]}
            print(f"ups {case} ERROR {exc}", flush=True)
        json.dump(res, open(os.path.join(d, "timings.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
