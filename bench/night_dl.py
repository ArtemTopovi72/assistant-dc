"""Night-list downloads to Z:/night_models/<comfy subdir>. Sequential, resumable."""
from huggingface_hub import hf_hub_download
import shutil, os, time
R = "Z:/night_models"
J = [
 ("FireRedTeam/FireRed-Image-Edit-LoRA-Zoo", "FireRed-Image-Edit-Makeup.safetensors", "loras"),
 ("FireRedTeam/FireRed-Image-Edit-LoRA-Zoo", "FireRed-Image-Edit-Covercraft.safetensors", "loras"),
 ("lightx2v/Minimax-h3-Turbo", "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors", "loras"),
 ("lightx2v/Minimax-h3-Turbo", "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors", "loras"),
 ("lightx2v/Minimax-h3-Turbo", "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors", "loras"),
 ("larryvrh/MiniMax-H3-Turbo-Lora", "minimax_h3_turbo_v4_step600_ema.safetensors", "loras"),
 ("LBH-123-AI/Minimax_h3_latent_Upscaler", "minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors", "latent_upscale_models"),
 ("drbaph/FireRed-Image-Edit-1.1_ComfyUI_Quants", "firered_image_edit_1.1_fp8_scaled_e4m3fn.safetensors", "diffusion_models"),
 ("vantagewithai/FireRed-Image-Edit-1.1-GGUF", "FireRed-Image-Edit-1.1-Q8_0.gguf", "unet"),
 ("vantagewithai/FireRed-Image-Edit-1.1-GGUF", "FireRed-Image-Edit-1.1-Q6_K.gguf", "unet"),
 ("xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI", "MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-convrot.safetensors", "diffusion_models"),
]
for repo, f, sub in J:
    dst = os.path.join(R, sub, os.path.basename(f))
    if os.path.exists(dst):
        print("have", dst, flush=True); continue
    t = time.time()
    for a in range(3):
        try:
            p = hf_hub_download(repo, f, local_dir=os.path.join(R, "_dl", repo.replace("/", "__")))
            os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.move(p, dst)
            print(f"ok {dst} {time.time()-t:.0f}s", flush=True); break
        except Exception as e:
            print("retry", f, e, flush=True); time.sleep(30)
print("ALL DONE", flush=True)
