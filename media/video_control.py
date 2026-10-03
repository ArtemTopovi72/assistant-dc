"""Video-to-video with MiniMax H3 Fun ControlNet-Union 2.0.

"Redraw this video as anime / in winter / as a clay animation": the user's
clip supplies the MOTION and the composition (as a control video), the prompt
supplies the new look, and an optional first frame pins the identity.

The graph is the ordinary t2va graph (video.build_workflow) with one model
patch in front of the sampler:

    LoadVideo -> GetVideoComponents -> ImageFromBatch(frames) -> ImageScale
      -> [Canny]  -> MiniMaxH3FunControlNetApply(model <- ModelPatchLoader)

Needs a ComfyUI with the H3 Fun control nodes (upstream, 2026-09) and the
checkpoint in models/model_patches. Canny is a core node; pose/depth need a
preprocessor pack and are passed in as an already-processed control clip
(kind="raw").
"""
import contextlib
import logging
import os
import random
import threading
from typing import Optional

import video as V

logger = logging.getLogger(__name__)

CONTROL_PATCH = os.getenv("H3_CONTROL_PATCH", "minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors")  # Comfy-Org, curve-adaln to match our pruned int8 base
KINDS = ("canny", "raw")

# The control nodes exist only in the newer ComfyUI clone (qi21, own venv). The
# live ComfyUI stays as it is; a restyle starts qi21 for the job and stops it
# after, so its models never sit beside the live server's in 24 GB.
QI21_URL = os.getenv("H3_CONTROL_URL", "http://127.0.0.1:8010")
_COMFY_HOME = os.getenv("COMFY_HOME", os.path.expanduser(r"~\Documents\ComfyUI"))
QI21_CMD = [os.path.join(_COMFY_HOME, "ComfyUI-qi21", ".venv", "Scripts", "python.exe"),
            os.path.join(_COMFY_HOME, "ComfyUI-qi21", "main.py"), "--base-directory", _COMFY_HOME,
            "--port", QI21_URL.rsplit(":", 1)[-1], "--reserve-vram", "2",
            "--use-sage-attention"]   # sageattention installed in the qi21 venv 2026-09-27 (same wheel as main)
from config import OUTPUT_DIR_COMFY as _OUT, INPUT_DIR_COMFY as _IN
QI21_CMD += ["--output-directory", str(_OUT), "--input-directory", str(_IN)]
_swap_lock = threading.Lock()


def _up(url: str) -> bool:
    try:
        import requests
        return requests.get(url + "/system_stats", timeout=3).ok
    except Exception:
        return False


@contextlib.contextmanager
def on_control_server(ctx=None, start_timeout: int = 300):
    """Run the body against qi21, starting it if needed and stopping what we started.

    ponytail: swaps the module-level COMFY_URL of comfy_client and video under a lock.
    Safe because renders take the card exclusively (one at a time); a per-call URL
    through comfy_client is the upgrade if two servers ever run jobs at once.
    """
    import comfy_client
    import subprocess
    import time
    with _swap_lock:
        proc = None
        if not _up(QI21_URL):
            if ctx is not None and hasattr(ctx, "set_stage"):
                ctx.set_stage("Starting the video engine")
            try:   # the live server's models out of VRAM first
                import requests
                requests.post(comfy_client.COMFY_URL.rstrip("/") + "/free",
                              json={"unload_models": True, "free_memory": True}, timeout=10)
            except Exception:
                pass
            rt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime")
            os.makedirs(rt, exist_ok=True)      # a fresh install has no runtime/ yet
            log = open(os.path.join(rt, "qi21.log"), "ab")
            proc = subprocess.Popen(QI21_CMD, cwd=_COMFY_HOME, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, PYTHONUTF8="1"),
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            t0 = time.time()
            while not _up(QI21_URL):
                cancelled = bool(ctx is not None and getattr(ctx, "is_cancelled", None) and ctx.is_cancelled())
                if cancelled or proc.poll() is not None or time.time() - t0 > start_timeout:
                    proc.kill()
                    try:        # reaped: a half-started server still holds VRAM
                        proc.wait(30)
                    except Exception:
                        pass
                    log.close()
                    raise RuntimeError("cancelled" if cancelled else "qi21 did not start (see runtime/qi21.log)")
                time.sleep(2)
            logger.info("qi21 up on %s in %.0fs", QI21_URL, time.time() - t0)
        old = comfy_client.COMFY_URL, V.COMFY_URL
        comfy_client.COMFY_URL = V.COMFY_URL = QI21_URL
        try:
            yield
        finally:
            comfy_client.COMFY_URL, V.COMFY_URL = old
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(30)
                except Exception:
                    proc.kill()
                    try:
                        proc.wait(30)
                    except Exception:
                        pass
                log.close()


N_APPLY = "36"


def build_control_workflow(prompt: str, control_video: str, *, kind: str = "canny",
                           width: int = 1344, height: int = 768, frames: int = 124,
                           seed: Optional[int] = None, strength: float = 1.0,
                           first_frame: Optional[str] = None, steps: Optional[int] = None) -> dict:
    """`control_video` / `first_frame` are ComfyUI-side names (already uploaded)."""
    if kind not in KINDS:
        raise ValueError(f"unknown control kind {kind!r}; one of {KINDS}")
    seed = seed if seed and seed > 0 else random.randint(1, 2**31 - 1)
    wf = V.build_workflow(prompt, mode="t2va", width=width, height=height, frames=frames,
                          seed=seed, images=(), steps=steps, two_stage=False)
    # two_stage=False: at the half-size first stage the control could not hold the
    # layout, and the refine pass redrew it (bench 2026-09-25); full size holds it.
    if first_frame:
        wf["20"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
        wf[V.N_COND]["inputs"]["first_frame"] = ["20", 0]
    wf["30"] = {"class_type": "LoadVideo", "inputs": {"file": control_video}}
    wf["31"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["30", 0]}}
    wf["32"] = {"class_type": "ImageFromBatch",
                "inputs": {"image": ["31", 0], "batch_index": 0, "length": int(frames)}}
    wf["33"] = {"class_type": "ImageScale",
                "inputs": {"image": ["32", 0], "upscale_method": "lanczos",
                           "width": int(width), "height": int(height), "crop": "center"}}
    ctrl = ["33", 0]
    if kind == "canny":
        wf["34"] = {"class_type": "Canny",
                    "inputs": {"image": ["33", 0], "low_threshold": 0.1, "high_threshold": 0.3}}
        ctrl = ["34", 0]
    wf["35"] = {"class_type": "ModelPatchLoader", "inputs": {"name": CONTROL_PATCH}}
    sampler = wf[V.N_SAMPLER]["inputs"]
    wf[N_APPLY] = {"class_type": "MiniMaxH3FunControlNetApply",
                   "inputs": {"model": sampler["model"], "model_patch": ["35", 0],
                              "vae": wf[V.N_COND]["inputs"]["vae"], "strength": float(strength),
                              "start_percent": 0.0, "end_percent": 1.0, "control_video": ctrl}}
    sampler["model"] = [N_APPLY, 0]
    return wf


def restyle(ctx, video_path: str, prompt: str, *, kind: str = "canny",
            first_frame_path: Optional[str] = None, seconds: float = 5.17,
            seed: Optional[int] = None, timeout: int = 3600) -> Optional[str]:
    """Render `prompt` over the motion of `video_path`. Returns the clip path or None.
    Runs on the control server (qi21), started for the job if it is not up."""
    with on_control_server(ctx):
        return _restyle(ctx, video_path, prompt, kind=kind, first_frame_path=first_frame_path,
                        seconds=seconds, seed=seed, timeout=timeout)


def _restyle(ctx, video_path, prompt, *, kind, first_frame_path, seconds, seed, timeout):
    import comfy_client
    info = V.probe(video_path) or {}
    src = (info.get("width"), info.get("height")) if info.get("width") else None
    w, h = V.resolve_size("", src)
    frames = V.snap_frames(V.seconds_to_frames(min(seconds, float(info.get("seconds") or seconds))))
    up = V._upload(video_path, kind="video")
    if not up:
        return None
    ff = V._upload(first_frame_path, kind="image") if first_frame_path else None
    wf = build_control_workflow(prompt, up, kind=kind, width=w, height=h, frames=frames,
                                seed=seed, first_frame=ff)
    wf = V._normalize_save_node(wf)
    return comfy_client._submit_and_poll(ctx, wf, timeout=timeout, label="h3_control",
                                         exclusive=True, validate=V._valid_video_file,
                                         job_timeout=timeout)
