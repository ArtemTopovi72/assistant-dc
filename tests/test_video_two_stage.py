"""Two-stage text-to-video: half-size render, LBH latent upscale 2x, 2-step refine.
Off inside suites unless VIDEO_TWO_STAGE=1; never on a keyframe/reference graph."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import video as V

ok = []


def check(c, m):
    ok.append(bool(c)); print(("ok   " if c else "FAIL ") + m)


os.environ.pop("VIDEO_TWO_STAGE", None)
wf = V.build_workflow("x", mode="t2va", width=1344, height=768, frames=124, seed=5)
check(not any(n["class_type"] == "MinimaxH3LatentUpscaler3D" for n in wf.values()), "off in suites by default")

os.environ["VIDEO_TWO_STAGE"] = "1"
wf = V.build_workflow("x", mode="t2va", width=1344, height=768, frames=124, seed=5)
c = wf[V.N_COND]["inputs"]
check((c["width"], c["height"]) == (672, 384), f"stage 1 at half size ({c['width']}x{c['height']})")
kinds = {n["class_type"]: k for k, n in wf.items()}
for t in ("LTXVSeparateAVLatent", "MinimaxH3LatentUpscaler3D", "LTXVConcatAVLatent", "SamplerCustomAdvanced"):
    check(t in kinds, f"{t} present")
# decoders read the final pack: refined video + stage 1's clean audio (19c210a)
last_cat = max((k for k, n in wf.items() if n["class_type"] == "LTXVConcatAVLatent"), key=int)
check(wf["9"]["inputs"]["samples"] == [last_cat, 0] and wf["10"]["inputs"]["samples"] == [last_cat, 0],
      "both decoders read the refined-video / stage-1-audio pack")
first_sep = min((k for k, n in wf.items() if n["class_type"] == "LTXVSeparateAVLatent"), key=int)
check(wf[first_sep]["inputs"]["av_latent"] == [V.N_SAMPLER, 0], "upscale reads stage 1")
check(wf[kinds["MinimaxH3LatentUpscaler3D"]]["inputs"]["mode.scale"] == V.UPSCALE_FACTOR == 1.5, "1.5x (user pick 2026-09-27), dotted dynamic-combo key")
check(wf[kinds["RandomNoise"]]["inputs"]["noise_seed"] == 6, "refine noise seeded from the clip seed")

V._upload = lambda p, kind: "k.png"
wf = V.build_workflow("x", mode="t2va", width=1344, height=768, frames=124, seed=5, images=["a.png"])
check("MinimaxH3LatentUpscaler3D" not in {n["class_type"] for n in wf.values()}, "keyframe graph untouched")
check(wf[V.N_COND]["inputs"]["width"] == 1344, "keyframe graph keeps full size")
os.environ.pop("VIDEO_TWO_STAGE")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
