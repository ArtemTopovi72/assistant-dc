from huggingface_hub import hf_hub_download
import shutil, os
repo, f, dst = ("xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI",
                "MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-convrot.safetensors",
                "Z:/night_models/diffusion_models")
p = hf_hub_download(repo, f, local_dir="Z:/night_models/_dl/int8")
os.makedirs(dst, exist_ok=True); shutil.move(p, os.path.join(dst, f)); print("ok", flush=True)
