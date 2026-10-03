"""A clip with voice samples keeps the turbo LoRA on (8 steps zeroed it: a fist turned to mush)."""
import os, sys
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(R, d) for d in ("media", "agent", "services", "core", "voice", "")]
import video as V
V._upload = lambda p, kind="image": os.path.basename(p)   # no ComfyUI in tests
from config import VIDEO_STEPS_REF2VA_VOICES
wf = V.build_workflow("x", mode="ref2va", width=1344, height=768, frames=121, seed=1,
                      images=["a.png"], audios=["v1.wav"], steps=VIDEO_STEPS_REF2VA_VOICES)
lora = wf[V.N_LORA]["inputs"]
assert lora["strength_model"] > 0, lora
assert "ref2v_turbo_8step" in lora["lora_name"], lora   # v0.1 4-step smeared hands; 8-step v1.0 is native
# rank21 @0.4 rides on top of the turbo LoRA (night A/B 10-02: sharpest clip, no slow-mo)
r21 = [n["inputs"] for n in wf.values() if n["class_type"] == "LoraLoaderModelOnly"
       and "rank_21" in n["inputs"]["lora_name"]]
assert r21 and r21[0]["strength_model"] == 0.4 and r21[0]["model"] == [V.N_LORA, 0], r21
assert wf["2"]["inputs"]["model"][0] != V.N_LORA, "the shift node skips rank21"
launch = open(os.path.join(R, "scripts", "launch_all.py"), encoding="utf-8").read()
assert "--use-ck-attention" in launch and "--use-sage-attention" not in launch
smp = [n["inputs"] for n in wf.values() if "sampler_name" in n.get("inputs", {})]
assert smp and smp[0]["sampler_name"] == "euler" and smp[0]["scheduler"] == "beta", smp
print("ok")

# two-stage: half size, the latent upscaled 2x back to the asked size
wf2 = V.build_workflow("x", mode="ref2va", width=1344, height=768, frames=121, seed=1,
                       images=["a.png"], audios=["v1.wav"], steps=8, two_stage=True)
ups = [n["inputs"] for n in wf2.values() if n["class_type"] == "MinimaxH3LatentUpscaler3D"]
assert ups and ups[0]["mode.scale"] == 2.0, ups
assert wf2[V.N_COND]["inputs"]["width"] == 672, wf2[V.N_COND]["inputs"]
print("two-stage ok")

V.REFINE_STEPS = 4      # VIDEO_REFINE_STEPS: the same sigma tail in 4 even steps
wf4 = V.build_workflow("x", mode="ref2va", width=1344, height=768, frames=121, seed=1,
                       images=["a.png"], audios=["v1.wav"], steps=12, two_stage=True)
sig = [n["inputs"]["sigmas"] for n in wf4.values() if n["class_type"] == "ManualSigmas"]
assert sig == ["0.6316, 0.4737, 0.3158, 0.1579, 0"], sig
print("refine steps ok")

# the decoded sound is stage 1's: the refine's audio (re-noised, 2 steps) is dropped
cats = [k for k, n in wf2.items() if n["class_type"] == "LTXVConcatAVLatent"]
last_cat = max(cats, key=int)
sep1 = min((k for k, n in wf2.items() if n["class_type"] == "LTXVSeparateAVLatent"), key=int)
assert wf2[last_cat]["inputs"]["audio_latent"] == [sep1, 1], wf2[last_cat]
# every consumer of the sampled pack reads the stage-1-audio pack, nobody the raw refine
users = [(k, f) for k, n in wf2.items() for f, v in n["inputs"].items() if v == [last_cat, 0]]
assert users, "nothing decodes the final pack"
print("stage-1 audio ok")
