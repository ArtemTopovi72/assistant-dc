"""Video generation through MiniMax H3 on a local ComfyUI.

H3 is omni-modal: one packed latent carries BOTH the video and a native 32kHz
stereo track, so there is no separate audio pass — a clip arrives with its own
sound already muxed. It ships as two task-specific checkpoints, and between them
they cover every mode this project exposes:

    FL2VA   text            -> video          (no reference images)
            one image       -> video          (that image is the first frame)
            two images      -> video          (first frame and last frame)
    Ref2VA  many images     -> video          (up to 9, as <Picture i> references)
            video + image   -> video          (up to 3 clips + images, and audio)

`generate_video(ctx, description, images=[...], videos=[...])` picks the mode from
what it is handed; callers do not choose a checkpoint.

Hard model constraints, taken from the node implementation rather than guessed —
these are not quality knobs, they are shapes H3 was trained on and anything else
either errors in the node or produces something the model has never seen:

  * 24 fps, and the frame count must satisfy ``n % 17 == 5``. `snap_frames`
    rounds to that grid. 124 frames = ~5.2s; MiniMax trained ~124-362 (5-15s).
  * a 768-short-edge canvas with a 768*1344 area cap, each axis a multiple of 32
    (`fit_canvas`, a port of the node's own adapt_canvas).
  * the weights are CFG-distilled: cfg stays 1.0 and a negative prompt is inert.

References are addressed positionally in the prompt as <Picture i>, <Video k> and
<Audio j>, 1-based per type, in the order they are attached. `reference_tags`
builds that vocabulary so the caller can describe the relationship between the
inputs and the intended output, which is how H3 is meant to be driven.
"""
from __future__ import annotations

import json
import logging
import math
import os
import config as _cfg_env   # env_int/env_float: a bad .env value falls back, never crashes the import
import random
import re
import shutil
import subprocess
import tempfile
from typing import Callable, Optional, Sequence

import config as _config
# ComfyUI transport. Was a pair of function-local `from image import ...` calls
# that pulled in 7,800 lines of image-editing code just to POST a workflow.
import comfy_client
from config import (
    COMFY_URL, OUTPUT_DIR, WORKFLOW_VIDEO_PATH, WORKFLOW_VIDEO_REF_PATH,
    VIDEO_FPS, VIDEO_DEFAULT_FRAMES, VIDEO_MAX_FRAMES, VIDEO_MAX_FRAMES_AUTO,
    VIDEO_SHORT_EDGE, VIDEO_MAX_PIXELS, VIDEO_CANVAS_MULTIPLE,
    VIDEO_STEPS, VIDEO_STEPS_REF2VA, VIDEO_STEPS_REF2VA_VOICES, VIDEO_SPEECH_STRESS, VIDEO_T2VA_LORA_3STEP, VIDEO_CFG, VIDEO_SHIFT_VIDEO, VIDEO_SHIFT_AUDIO,
    VIDEO_JOB_TIMEOUT,
)

logger = logging.getLogger("assistant.video")

# Node ids inside the two workflow graphs (see workflow_video_h3*.json).
N_UNET, N_SHIFT, N_CLIP = "1", "2", "3"
N_VAE_VIDEO, N_VAE_AUDIO = "4", "5"
N_COND, N_SAMPLER, N_SAVE = "6", "8", "12"
N_LORA = "13"  # LightX2V turbo LoRA in both graphs (FL2V v1.2 / Ref2V v0.1)

MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3

# Why the last generation produced nothing, for an honest message to the user.
# Mirrors image._GENERATE_FAILURE.
_VIDEO_FAILURE: dict = {"reason": "server_error", "detail": ""}


class VideoUnavailable(RuntimeError):
    """The engine cannot run at all (missing weights, ComfyUI too old)."""


# --------------------------------------------------------------------------- #
# Geometry and duration — the model's grid, not ours
# --------------------------------------------------------------------------- #
def snap_frames(length: int) -> int:
    """Round a frame count up onto H3's 17k+5 grid, clamped to the trained range.

    The node does this internally too, but doing it here means the duration we
    PROMISE the user ("~5s") is the duration they actually get, and that the
    audio/video latent lengths we reason about match what the sampler builds.
    """
    n = max(5, int(length or 0))
    while n % 17 != 5:
        n += 1
    # The CEILING has to sit on the grid too. VIDEO_MAX_FRAMES is env-overridable,
    # and clamping to an off-grid value (e.g. VIDEO_MAX_FRAMES=300) would hand the
    # node a length it rejects — snapping up here would instead exceed the cap the
    # operator asked for, so walk the ceiling DOWN to the nearest legal count.
    cap = max(5, int(VIDEO_MAX_FRAMES))
    while cap % 17 != 5:
        cap -= 1
    return min(n, cap)


_QUOTED = re.compile(r'«([^»]*)»|"([^"]*)"|“([^”]*)”')
_BEAT = re.compile(r"[,.;:!?—]|\b(?:then|and then|after that|afterwards|while|before|until|"
                   r"затем|потом|после этого|и снова|снова)\b", re.I)
_HOLD = re.compile(r"\b(?:for (?:a few|several|some|\d+) seconds?|for a while|"
                   r"несколько секунд|какое-то время)\b", re.I)


def estimate_seconds(description: str, capped: bool = True) -> float:
    """How long the scripted clip needs to play out, when nobody said a length.

    A fixed 5.2 s cut every multi-action script short (live 10-06: "puts on a suit,
    takes a broom, sweeps for several seconds, stops, looks at the camera, says a
    line, sweeps again" ended mid-way). Spoken lines at ~2.3 words/s, every other
    beat ~1.3 s, a held action ("for several seconds") +2 s; 5.2 s at the least.
    """
    text = description or ""
    for pre in (CONTINUE_PREFIX, CONTINUE_CTX_PREFIX):   # the template, not the script
        if text.startswith(pre):
            text = text[len(pre):]
    spoken = " ".join(next(g for g in m.groups() if g is not None) for m in _QUOTED.finditer(text))
    rest = _QUOTED.sub(" ", text)
    words = len(spoken.split())
    beats = max(1, len([b for b in _BEAT.split(rest) if b and any(c.isalpha() for c in b)]))
    secs = 1.0 + words / 2.3 + beats * 1.3 + 2.0 * len(_HOLD.findall(rest))
    top = frames_to_seconds(VIDEO_MAX_FRAMES_AUTO) if capped else float("inf")
    return round(min(top, max(frames_to_seconds(VIDEO_DEFAULT_FRAMES), secs)), 2)


_SENT = re.compile(r"(?<=[.!?;])\s+|(?<=,)\s+(?=(?:then|after that|afterwards|затем|потом|после этого)\b)", re.I)


def split_script(description: str, max_seconds: float = 0.0) -> list:
    """A script too long for one clip, as consecutive parts that each fit one
    (`estimate_seconds` <= max_seconds, default the auto cap). Cuts only between
    sentences or before "then"; a quoted line is never split. One part = fits as is."""
    top = max_seconds or frames_to_seconds(VIDEO_MAX_FRAMES_AUTO)
    text = (description or "").strip()
    if estimate_seconds(text, capped=False) <= top:
        return [text]
    quotes = []

    def _hide(m):
        quotes.append(m.group(0))
        return f"\x00{len(quotes) - 1}\x00"
    hidden = _QUOTED.sub(_hide, text)
    pieces = [x for x in _SENT.split(hidden) if x and x.strip()]
    restore = lambda t: re.sub(r"\x00(\d+)\x00", lambda m: quotes[int(m.group(1))], t)
    parts, cur = [], ""
    for piece in pieces:
        cand = (cur + " " + piece).strip()
        if cur and estimate_seconds(restore(cand), capped=False) > top:
            parts.append(restore(cur))
            cur = piece.strip()
        else:
            cur = cand
    if cur:
        parts.append(restore(cur))
    return parts


def seconds_to_frames(seconds: float) -> int:
    return snap_frames(int(round(max(0.2, float(seconds)) * VIDEO_FPS)))


def frames_to_seconds(frames: int) -> float:
    return round(int(frames) / float(VIDEO_FPS), 2)


def fit_canvas(width: int, height: int) -> tuple[int, int]:
    """A port of the node's adapt_canvas: 768 short edge, 1MP cap, /32 per axis."""
    width = max(1, int(width or 1))
    height = max(1, int(height or 1))
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = VIDEO_SHORT_EDGE * ratio, float(VIDEO_SHORT_EDGE)
    else:
        nom_w, nom_h = float(VIDEO_SHORT_EDGE), VIDEO_SHORT_EDGE / ratio
    if nom_w * nom_h > VIDEO_MAX_PIXELS:
        s = math.sqrt(VIDEO_MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    m = VIDEO_CANVAS_MULTIPLE
    return (max(m, round(nom_w / m) * m), max(m, round(nom_h / m) * m))


# Named shapes for callers that think in aspect ratios rather than pixels.
ASPECTS = {
    "16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1),
    "4:3": (4, 3), "3:4": (3, 4), "21:9": (21, 9),
    # 4:5 and 5:4 are what a phone-shot still usually is, and they land exactly on
    # 768x960 / 960x768 — both inside the 768*1344 cap with axes already on the /32
    # grid, so nothing is squashed or clamped. Without them an unknown key falls
    # back to the 16:9 DEFAULT, which silently returns a LANDSCAPE canvas: asking
    # for "4:5" used to hand back 1344x768, the opposite shape.
    "4:5": (4, 5), "5:4": (5, 4),
}


def resolve_size(aspect: str = "", source_size: Optional[tuple] = None) -> tuple[int, int]:
    """Canvas for this clip: an explicit aspect wins, else the source image's own
    shape, else 16:9. Always returned already fitted to the model's grid."""
    key = (aspect or "").strip()
    if key in ASPECTS:
        return fit_canvas(*ASPECTS[key])
    if source_size and source_size[0] and source_size[1]:
        return fit_canvas(int(source_size[0]), int(source_size[1]))
    return fit_canvas(*ASPECTS["16:9"])


def _image_size(path: str) -> Optional[tuple]:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Reference vocabulary
# --------------------------------------------------------------------------- #
def reference_tags(n_images: int = 0, n_videos: int = 0, n_audios: int = 0) -> list[str]:
    """The <Picture i>/<Video k>/<Audio j> labels these inputs will carry.

    1-based per type and assigned in attach order, matching the node. The prompt
    has to use exactly these strings or the model cannot tell which reference a
    sentence is talking about.
    """
    tags = [f"<Picture {i}>" for i in range(1, int(n_images) + 1)]
    tags += [f"<Video {i}>" for i in range(1, int(n_videos) + 1)]
    tags += [f"<Audio {i}>" for i in range(1, int(n_audios) + 1)]
    return tags


def pick_mode(images: Sequence[str] = (), videos: Sequence[str] = (),
              audios: Sequence[str] = ()) -> str:
    """Which H3 task this request is, from what was attached.

    In principle one/two images are keyframe work the FL2VA checkpoint does
    better, since those images become actual frames rather than something the
    model is merely reminded of, and 3+ images (or any video/audio reference)
    need the omni-reference checkpoint. In practice, live 2026-09-19: every
    i2va/flf2va job this deployment has ever submitted (3 for 3, across two
    separate real animate requests plus this fix's own verification run) got
    stuck in KSampler and never finished on its own -- each had to be killed
    with /interrupt after 45-50+ minutes at 100% GPU. Ref2VA, run with the
    exact same GPU/model stack, has reliably finished in 8-12 minutes every
    time. The two checkpoints are ~19.8GB GGUF files of near-identical size
    (not a corrupt/truncated download), so something about the FL2VA graph or
    checkpoint itself hangs on this machine -- until that is root-caused, route
    every image-driven request through the checkpoint that actually completes,
    even at the cost of the images being loose references rather than exact
    frames. A silently-broken checkpoint is worse than a lower-fidelity one.
    """
    n_i, n_v, n_a = len(images or ()), len(videos or ()), len(audios or ())
    if n_v or n_a or n_i >= 1:
        return "ref2va"
    return "t2va"


MODE_LABELS = {
    "t2va": "text to video",
    "i2va": "image to video",
    "flf2va": "first and last frame to video",
    "ref2va": "reference to video",
}


# --------------------------------------------------------------------------- #
# Readiness
# --------------------------------------------------------------------------- #
def _models_dir() -> str:
    return os.getenv("COMFY_BASE_DIR", os.path.join(os.path.expanduser("~"), "Documents", "ComfyUI"))


REQUIRED_FILES = {
    # t2va: workflow_video_h3.json; ref2va (every "animate this photo" request):
    # Singularity fine-tune, A/B 2026-09-24
    "diffusion_models": ["MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-convrot.safetensors",
                         "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"],
    "text_encoders": ["qwen3vl_32b_minimax_h3-Q4_K_M.gguf"],
    "vae": ["minimax_h3_video_vae_fp16.safetensors",
            "minimax_h3_audio_vae_fp32.safetensors"],
}


def missing_weights() -> list[str]:
    """Which required model files are not on disk yet (empty = ready)."""
    base = os.path.join(_models_dir(), "models")
    out = []
    for sub, names in REQUIRED_FILES.items():
        for n in names:
            if not os.path.exists(os.path.join(base, sub, n)):
                out.append(f"models/{sub}/{n}")
    return out


def engine_available(ctx=None) -> tuple[bool, str]:
    """Can we generate a video right now? Returns (ok, human reason if not).

    Checked BEFORE the agent promises anything, because the failure modes here are
    slow and expensive: a missing node means the graph is rejected after the user
    has already been told a video is coming, and a missing 20GB checkpoint means a
    download, not a retry.
    """
    if getattr(_config, "VIDEO_ENGINE", "") != "minimax_h3":
        return False, f"VIDEO_ENGINE is {getattr(_config, 'VIDEO_ENGINE', '')!r}, not minimax_h3"
    miss = missing_weights()
    if miss:
        return False, ("the H3 weights are not downloaded yet (missing: "
                       + ", ".join(miss[:3]) + (" …" if len(miss) > 3 else "")
                       + ") — run scripts/fetch_h3_weights.py")
    has = _server_has_h3_nodes()
    if has is None:
        # Down is not "too old" (live 10-02 night: the agent told the user video
        # "is not set up yet" while ComfyUI was merely not running).
        return False, f"ComfyUI is not responding at {COMFY_URL} (not running) — try again shortly"
    if not has:
        return False, ("this ComfyUI does not have the MiniMaxH3 nodes — they need "
                       "ComfyUI >= 0.30.0")
    return True, ""


_NODE_CACHE: dict = {}


def _server_has_h3_nodes():
    """Ask the live ComfyUI whether it knows the H3 nodes (None: it did not answer). Cached per URL."""
    if COMFY_URL in _NODE_CACHE:
        return _NODE_CACHE[COMFY_URL]
    ok = False
    try:
        import requests
        r = requests.get(f"{COMFY_URL}/object_info/MiniMaxH3ImageToVideo", timeout=10)
        ok = r.status_code == 200 and bool(r.json())
    except Exception as exc:
        logger.warning("could not ask ComfyUI about the H3 nodes: %s", exc)
        return None         # not cached: a server that is merely down may come back
    _NODE_CACHE[COMFY_URL] = ok
    return ok


def _server_has_node(cls: str):
    """Whether the live ComfyUI knows node `cls` (None: it did not answer). Cached per URL."""
    key = (COMFY_URL, cls)
    if key in _NODE_CACHE:
        return _NODE_CACHE[key]
    try:
        import requests
        r = requests.get(f"{COMFY_URL}/object_info/{cls}", timeout=10)
        ok = r.status_code == 200 and bool(r.json())
    except Exception as exc:
        logger.warning("could not ask ComfyUI about %s: %s", cls, exc)
        return None
    _NODE_CACHE[key] = ok
    return ok


def _valid_video_file(path: str) -> bool:
    """A real, non-empty video container — not a truncated write or a stub.

    Deliberately NOT image.py's _valid_image_file: a SaveVideo job reports its mp4
    under the same "images" history key, so without a video-shaped check the poller
    would find the right file and then discard it for failing a PNG decode.
    """
    try:
        if not (path and os.path.exists(path)):
            return False
        # Anything under ~16KB is not a real 5s clip; it is a failed write.
        if os.path.getsize(path) < 16 * 1024:
            logger.error("video file is implausibly small: %s (%d bytes)",
                         path, os.path.getsize(path))
            return False
        return os.path.splitext(path)[1].lower() in (
            ".mp4", ".webm", ".mkv", ".mov", ".m4v", ".avi")
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Building the graph
# --------------------------------------------------------------------------- #
def _load_workflow(mode: str) -> dict:
    path = WORKFLOW_VIDEO_REF_PATH if mode == "ref2va" else WORKFLOW_VIDEO_PATH
    if not os.path.exists(path):
        raise VideoUnavailable(f"video workflow missing: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _upload(path: str, kind: str = "image") -> Optional[str]:
    """Put a local file where ComfyUI's LoadImage/LoadVideo can see it."""
    if kind == "image":
        return comfy_client._upload_image_to_comfy(path, COMFY_URL)
    # Videos and audio go through the same /upload/image endpoint (ComfyUI stores
    # them all in input/); only the mime differs.
    try:
        import requests
        ext = os.path.splitext(path)[1].lower()
        mime = {"mp4": "video/mp4", "webm": "video/webm", "mov": "video/quicktime",
                "wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg",
                "flac": "audio/flac", "m4a": "audio/mp4"}.get(ext.lstrip("."),
                                                              "application/octet-stream")
        with open(path, "rb") as fh:
            r = requests.post(f"{COMFY_URL}/upload/image",
                              files={"image": (comfy_client.upload_name(path), fh, mime)},
                              data={"type": "input", "overwrite": "true"}, timeout=120)
        if r.status_code == 200:
            return r.json().get("name") or comfy_client.upload_name(path)
        logger.error("ComfyUI upload HTTP %d for %s", r.status_code, path)
    except Exception as exc:
        logger.error("ComfyUI upload failed for %s: %s", path, exc)
    return None


LATENT_UPSCALER = os.getenv(
    "VIDEO_LATENT_UPSCALER", "night\\minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors")


# 1.5x, not 2x (user pick 2026-09-27, bench/h3_stage2.py): stage 2 + decode
# 45 s vs 93 s; the clip comes out 1008x576 instead of 1344x768.
UPSCALE_FACTOR = _cfg_env.env_float("VIDEO_UPSCALE_FACTOR", 1.5)
REFINE_STEPS = max(1, _cfg_env.env_int("VIDEO_REFINE_STEPS", 2))


def _two_stage_on() -> bool:
    v = os.getenv("VIDEO_TWO_STAGE")
    if v is not None:
        return v == "1"
    return not os.getenv("F5_TEST_RUN")


def _add_latent_upscale(wf: dict, seed: int, factor: float = 0.0) -> None:
    """Render at half size, upscale the video latent 2x, refine 2 steps.

    Night bench 2026-09-24 (bench/night_upscale.py, same seed/prompt): t1 149s vs
    212s, t2 176s vs 211s single-stage at full size; speech still exact (GigaAM),
    hands and faces intact, the green flare artifact gone. Text-to-video only:
    a reference render was not measured. Sigma tail from the upstream LBH
    workflow3. The audio is stage 1's: the refine output keeps only its video.
    """
    cond = wf[N_COND]["inputs"]
    cond["width"] = int(cond["width"]) // 2 // 32 * 32
    cond["height"] = int(cond["height"]) // 2 // 32 * 32
    base = max(int(k) for k in wf) + 1
    n = {k: str(base + i) for i, k in enumerate(
        ("sep", "ups", "cat", "sig", "noise", "guide", "smp", "adv", "sep2", "cat2"))}
    wf[n["sep"]] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": [N_SAMPLER, 0]}}
    wf[n["ups"]] = {"class_type": "MinimaxH3LatentUpscaler3D", "inputs": {
        "latent": [n["sep"], 0], "model_name": LATENT_UPSCALER,
        "mode": "scale by multiplier", "mode.scale": factor or UPSCALE_FACTOR, "align": 32,
        "enable_temporal_chunking": True, "force_unload": True,
        "device": "cuda", "precision": "fp16"}}
    wf[n["cat"]] = {"class_type": "LTXVConcatAVLatent",
                    "inputs": {"video_latent": [n["ups"], 0], "audio_latent": [n["sep"], 1]}}
    # the refine walks the same tail (0.6316 -> 0) in VIDEO_REFINE_STEPS even steps
    top = 0.6316
    sig = ", ".join(f"{top * (1 - i / REFINE_STEPS):.4f}" for i in range(REFINE_STEPS)) + ", 0"
    wf[n["sig"]] = {"class_type": "ManualSigmas", "inputs": {"sigmas": sig}}
    wf[n["noise"]] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed) + 1}}
    smp = wf[N_SAMPLER]["inputs"]
    wf[n["guide"]] = {"class_type": "CFGGuider", "inputs": {
        "model": smp["model"], "positive": smp["positive"], "negative": smp["negative"],
        "cfg": float(smp["cfg"])}}
    wf[n["smp"]] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    wf[n["adv"]] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": [n["noise"], 0], "guider": [n["guide"], 0], "sampler": [n["smp"], 0],
        "sigmas": [n["sig"], 0], "latent_image": [n["cat"], 0]}}
    # The refine re-noises the WHOLE pack, audio included, and rebuilds it in a
    # couple of steps -- that thinned the hits and the score (10-02). The picture
    # comes from the refine, the sound stays the clean stage-1 sound.
    wf[n["sep2"]] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": [n["adv"], 0]}}
    wf[n["cat2"]] = {"class_type": "LTXVConcatAVLatent",
                     "inputs": {"video_latent": [n["sep2"], 0], "audio_latent": [n["sep"], 1]}}
    for node in wf.values():
        for k, v in node["inputs"].items():
            if v == [N_SAMPLER, 0] and node is not wf[n["sep"]]:
                node["inputs"][k] = [n["cat2"], 0]


def build_workflow(prompt: str, *, mode: str, width: int, height: int,
                   frames: int, seed: int,
                   images: Sequence[str] = (), videos: Sequence[str] = (),
                   audios: Sequence[str] = (),
                   steps: Optional[int] = None,
                   ref_image_size: str = "match", ctx=None, two_stage: Optional[bool] = None,
                   context_video: str = "") -> dict:
    """The ComfyUI API graph for one clip, with every reference already uploaded.

    Split out from `generate_video` so the shape of the graph can be asserted in
    tests without a ComfyUI, a GPU or 60GB of weights anywhere in sight.
    """
    wf = _load_workflow(mode)
    # Uploading a handful of stills, or a whole reference clip, is visible dead time
    # before the progress bar starts moving — say what is happening.
    if (images or videos or audios) and ctx is not None and hasattr(ctx, "set_stage"):
        ctx.set_stage("Preparing the references")
    cond = wf[N_COND]["inputs"]
    cond["prompt"] = prompt
    cond["width"], cond["height"] = int(width), int(height)
    cond["length"] = int(frames)

    smp = wf[N_SAMPLER]["inputs"]
    smp["seed"] = int(seed)
    smp["steps"] = int(steps or VIDEO_STEPS)
    smp["cfg"] = float(VIDEO_CFG)

    # The Turbo LoRA baked into workflow_video_h3_ref.json (node 13) is
    # distilled for EXACTLY 4 NFE (huggingface.co/lightx2v/Minimax-h3-Turbo).
    # Live 2026-09-20: ref2va renders came back "оч мыльно" (soft/blurry) and
    # steps was raised past 4 to fix it -- but sampling a 4-step-distilled
    # LoRA for more steps is outside its trained regime and won't sharpen
    # anything (same lesson as the Ideogram turbo LoRA fighting its own
    # polish pass). So whenever ref2va runs at any step count other than 4,
    # the LoRA is zeroed out here rather than left attached and wrong.
    # Both graphs now carry a 4-step turbo LoRA (t2va: FL2V v1.2, ref2va: Ref2V v0.1).
    # t2va at 3 steps swaps in the TaoMate 3-step LoRA (distilled for 3 NFE).
    if N_LORA in wf and mode == "t2va" and smp["steps"] == 3 and VIDEO_T2VA_LORA_3STEP:
        wf[N_LORA]["inputs"]["lora_name"] = VIDEO_T2VA_LORA_3STEP
    # A turbo LoRA stays on ABOVE 4 steps (community range 6-8 sharpens it);
    # zeroed there, the bare model at cfg 1 turned a fist into mush (10-02).
    elif N_LORA in wf and smp["steps"] < 4:
        wf[N_LORA]["inputs"]["strength_model"] = 0.0

    sh = wf[N_SHIFT]["inputs"]
    sh["shift_video"] = float(VIDEO_SHIFT_VIDEO)
    sh["shift_audio"] = float(VIDEO_SHIFT_AUDIO)

    if (_two_stage_on() if two_stage is None else two_stage):
        if mode == "t2va" and not images:
            _add_latent_upscale(wf, seed)
        elif mode == "ref2va":
            # Half size, then 2x back to full (live 10-02, the deds fight at 8
            # steps: 267 s vs 348 s single-stage, hands clean at 1344x768).
            _add_latent_upscale(wf, seed, factor=2.0)

    if mode == "ref2va":
        cond["ref_image_size"] = ref_image_size
        # Autogrow (COMFY_AUTOGROW_V3) slots are ZERO-based AND carry the
        # container name as a dotted prefix: "ref_images.ref_image_0".
        #
        #   comfy_api/latest/_io.py builds the names with
        #       names = [f"{prefix}{i}" for i in range(max)]
        #   and parse_class_inputs() then prefixes each one with the input's own
        #   id, so expected_id = finalize_prefix(["ref_images"], "ref_image_0").
        #   At execution build_nested_inputs() splits on "." to rebuild the dict
        #   that execute(ref_images={...}) actually takes.
        #
        # Posting a BARE "ref_image_0" is the trap: it is not in the schema at
        # all, so validation ignores it as an unknown extra and POST /prompt
        # returns 200 — then execution passes it through as a stray kwarg and
        # dies with "execute() got an unexpected keyword argument 'ref_image_0'"
        # AFTER the encoder and VAEs have loaded. Schema acceptance proves
        # nothing here; only a real reference render does.
        #
        # Do NOT confuse these with the tags used in the PROMPT: the wire names are
        # 0-based, while the model addresses the same references as <Picture 1>,
        # <Video 1>, <Audio 1> — 1-based per type (see reference_tags). Numbering
        # the wire from 1 silently drops the first reference AND asks for a slot 9
        # that does not exist.
        #
        # The node pairs a clip with its soundtrack by the trailing index
        # (ref_video_audio_N belongs to ref_video_N), so the families stay aligned.
        node_id = str(max(int(k) for k in wf) + 1)
        for i, p in enumerate(list(images)[:MAX_REF_IMAGES]):
            name = _upload(p, "image")
            if not name:
                raise VideoUnavailable(f"could not upload reference image {p}")
            wf[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
            cond[f"ref_images.ref_image_{i}"] = [node_id, 0]
            node_id = str(int(node_id) + 1)
        for i, p in enumerate(list(videos)[:MAX_REF_VIDEOS]):
            name = _upload(p, "video")
            if not name:
                raise VideoUnavailable(f"could not upload reference video {p}")
            # LoadVideo yields a VIDEO; the node wants frames, so split it.
            wf[node_id] = {"class_type": "LoadVideo", "inputs": {"file": name}}
            vid_id = node_id
            node_id = str(int(node_id) + 1)
            wf[node_id] = {"class_type": "GetVideoComponents",
                           "inputs": {"video": [vid_id, 0]}}
            cond[f"ref_videos.ref_video_{i}"] = [node_id, 0]                    # frames
            cond[f"ref_video_audios.ref_video_audio_{i}"] = [node_id, 1]        # its sound
            node_id = str(int(node_id) + 1)
        for i, p in enumerate(list(audios)[:MAX_REF_AUDIOS]):
            name = _upload(p, "audio")
            if not name:
                raise VideoUnavailable(f"could not upload reference audio {p}")
            wf[node_id] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
            cond[f"ref_audios.ref_audio_{i}"] = [node_id, 0]
            node_id = str(int(node_id) + 1)
    else:
        # FL2VA keyframes: the first image becomes frame 0, the second the last
        # frame. These are ACTUAL frames of the output, which is why one or two
        # images do not go down the reference path.
        node_id = str(max(int(k) for k in wf) + 1)
        for slot, p in zip(("first_frame", "last_frame"), list(images)[:2]):
            name = _upload(p, "image")
            if not name:
                raise VideoUnavailable(f"could not upload keyframe {p}")
            wf[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
            cond[slot] = [node_id, 0]
            node_id = str(int(node_id) + 1)
    if context_video:
        _add_motion_context(wf, context_video)
    _apply_overrides(wf)
    return wf


# Continuation by pinned frames (ComfyUI-H3-Motion-Context v0.3.1, the release for
# ComfyUI <= 0.33): the last 22 frames and the last second of sound of the previous
# clip sit at the head of the NEW clip's own timeline as never-denoised rows and stay
# there: join_pinned cuts the old clip at that point.
# The old way handed the tail in as a <Video 1> reference -- 60 more
# frames of tokens in a quadratic attention, and the sound restarted instead of going on.
MOTION_CONTEXT_FRAMES = int(os.getenv("VIDEO_MOTION_CONTEXT_FRAMES", "22"))                      # on the VAE grid: 5, 22, 39, 56


def motion_context_on() -> bool:
    if os.getenv("VIDEO_MOTION_CONTEXT", "1") != "1":
        return False
    return bool(_server_has_node("MiniMaxH3MotionContext"))


def _add_motion_context(wf: dict, context_video: str) -> None:
    name = _upload(context_video, "video")
    if not name:
        raise VideoUnavailable(f"could not upload the clip to continue {context_video}")
    base = max(int(k) for k in wf) + 1
    load, comp, mc, trim = (str(base + i) for i in range(4))
    wf[load] = {"class_type": "LoadVideo", "inputs": {"file": name}}
    wf[comp] = {"class_type": "GetVideoComponents", "inputs": {"video": [load, 0]}}
    wf[mc] = {"class_type": "MiniMaxH3MotionContext", "inputs": {
        "conditioning": [N_COND, 0], "vae": [N_VAE_VIDEO, 0], "latent": [N_COND, 1],
        "context_length": str(MOTION_CONTEXT_FRAMES), "audio_context_length": 24,
        "context_frames": [comp, 0], "audio_vae": [N_VAE_AUDIO, 0], "context_audio": [comp, 1]}}
    # Only the first pass reads the pins; a two-stage refine starts from a latent that
    # already holds them, and its guider keeps the plain conditioning.
    wf[N_SAMPLER]["inputs"]["positive"] = [mc, 0]
    out = wf[N_SAVE]["inputs"]["video"][0]                   # CreateVideo
    v = wf[out]["inputs"]
    wf[trim] = {"class_type": "MiniMaxH3MotionContextTrim", "inputs": {
        "images": v["images"], "audio": v["audio"], "trim_frames": 0,
        "fps": 24.0, "match_tail": True}}
    v["images"], v["audio"] = [trim, 0], [trim, 1]


def _apply_overrides(wf: dict) -> None:
    """VIDEO_WF_OVERRIDES='{"13.lora_name": "x", "8.sampler_name": "euler"}' for A/B runs
    without editing the graph files; a node id that is not in this graph is skipped."""
    raw = os.getenv("VIDEO_WF_OVERRIDES")
    if not raw:
        return
    for key, val in json.loads(raw).items():
        node, _, field = key.partition(".")
        if not field:          # "90": {"class_type": ...} adds a node (e.g. an attention patch)
            wf[node] = val
        elif node in wf:
            wf[node]["inputs"][field] = val


def _normalize_save_node(wf: dict) -> dict:
    """Pick a `codec` value for SaveVideo that THIS server will accept.

    `codec` is a COMFY_DYNAMICCOMBO_V3. It is tempting to send the structured
    form {"codec": "auto"} — and the server ACCEPTS that at validation, which is
    what makes the mistake expensive. Execution then fails with

        SaveVideo.execute() missing 1 required positional argument: 'codec'

    after the full render has already been paid for (observed live: 8m26s of
    sampling and VAE decode thrown away). The reason is in
    comfy_api/latest/_io.py: DynamicCombo._expand_schema_for_dynamic does
    `key = live_inputs[finalized_id]` and matches it against `option["key"]`, so
    the value must be the BARE OPTION STRING. A dict matches no option, the input
    is dropped during expansion, and the node is called without it.

    So: always a bare string, and prefer one the server actually lists.
    """
    want = "auto"
    try:
        import requests
        r = requests.get(f"{COMFY_URL}/object_info/SaveVideo", timeout=10)
        if r.status_code == 200:
            entry = ((r.json().get("SaveVideo") or {}).get("input", {})
                     .get("required", {}) or {}).get("codec")
            opts = []
            if entry and isinstance(entry, (list, tuple)) and len(entry) > 1:
                # DynamicCombo: [{"key": "auto", ...}, {"key": "h264", ...}]
                for o in (entry[1] or {}).get("options") or []:
                    if isinstance(o, dict) and o.get("key"):
                        opts.append(o["key"])
            if entry and isinstance(entry[0], list):
                opts += [o for o in entry[0] if isinstance(o, str)]
            if opts and want not in opts:
                want = opts[0]
    except Exception as exc:
        logger.debug("could not inspect SaveVideo schema (%s) — using %r", exc, want)
    wf[N_SAVE]["inputs"]["codec"] = want
    return wf


# --------------------------------------------------------------------------- #
# Generating
# --------------------------------------------------------------------------- #
_FRAMING_LOCK_SUFFIX = (
    " Camera: static, locked off, no camera movement, no zoom, no pan, no crop -- "
    "preserve the original framing and composition of the reference exactly.")


def _lock_framing_unless_requested(description: str) -> str:
    """Append a static-camera directive unless the request itself asks for motion.

    Live, 2026-09-19: "the video model doesn't preserve the original at all on
    complex requests, sometimes crops without being asked." Researched against
    the model's own prompt guide (MiniMax H3): a request with several distinct
    actions but no explicit camera line gets the model improvising a shot --
    zooming, panning or reframing on its own initiative -- and the documented
    fix is exactly this: state "CAMERA: none"/a locked static camera explicitly
    rather than leaving it unsaid. Skipped when the description already asks
    for camera motion, so an explicit "zoom in slowly" or an animate preset
    built around a pan is never fought.
    """
    d = (description or "").strip()
    import intent
    if not d or intent.ask_yes("A video clip is described as: {text}\n\nDoes the description ask "
                               "for the camera to move (zoom, pan, dolly, orbit, tilt, tracking or "
                               "handheld shot)?", d):
        return d
    return d + _FRAMING_LOCK_SUFFIX


# With a picture to start from: the picture IS the opening shot, and its look
# holds for the whole clip (owner 10-03: «где был реализм там только гипер
# реализм, где мультяшно там мультяшно… первый снимок это и есть стартовый
# ракурс»). Said for any picture, so no guess about its style is needed; a
# camera move the user asks for still happens, but from that opening shot.
_START_LOCK_SUFFIX = (
    " The reference image <Picture 1> is the exact first frame: the clip opens on it as it is -- same "
    "framing, crop, zoom and angle; any camera move or reframing happens only later in the "
    "clip and only when asked for above.")
_STYLE_LOCK_SUFFIX = (
    " Keep the reference's visual style unchanged from "
    "the first frame to the last: a photograph stays a photograph (hyper-realistic, "
    "true-to-life detail, real skin and materials), a cartoon, anime, painting or 3D render "
    "stays in exactly that style; never switch between realistic and drawn, never restyle "
    "the faces or the scene.")


def _asks_new_style(description: str) -> bool:
    """«оживи в стиле аниме»: the style lock would fight the request itself."""
    import intent
    return intent.ask_yes("A video clip is made from a picture and described as: {text}\n\n"
                          "Does the description ask to change the picture's visual style (into "
                          "anime, cartoon, painting, 3D, photoreal, another art style)?", description)


def _with_lock(description: str, lock: str) -> str:
    """`lock` restated verbatim at the end of the visual part: the context-IR
    rewrite paraphrases it away («the camera holds a static shot»)."""
    if lock in description:
        return description
    cut = description.find("\n\noverall_soundscape:")
    return description[:cut] + lock + description[cut:] if cut > 0 else description + lock


_GUIDE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "h3_prompt_guides")
_CIR_FIELDS = ("integrated_multimodal_description:", "overall_soundscape:", "non_diegetic_music:")
_CIR_REF_FIELDS = ("subject_definitions", "retention_analysis", "detailed_description")


def _context_ir_on() -> bool:
    """On by default since the night A/B of 2026-09-24 (same seed, same prompts):
    Ref2VA from an illustrated photo kept the illustration style, ONE continuous
    shot and the same face for all 5 s, where the plain prompt switched to
    photoreal within 1.5 s, cut to her back at 3 s and ended on another woman.
    T2VA: equal quality, the asked-for action reads better. No render cost.
    Suites never call the live model: off under F5_TEST_RUN unless set explicitly."""
    v = os.getenv("VIDEO_CONTEXT_IR")
    if v is not None:
        return v == "1"
    return not os.getenv("F5_TEST_RUN")


_ACCENTOR = None
_RU_LINE = re.compile(r'(«[^»]*[а-яё][^»]*»|"[^"]*[а-яё][^"]*"|<d>[^<]*[а-яё][^<]*</d>)', re.I)


def mark_speech_stress(ctx, prompt: str) -> str:
    """Russian lines the people speak carry stress marks (U+0301 after the
    stressed vowel, dictionary style): H3 reads Russian letters, not stress,
    and guessed «пи́дор» as «пидо́р». The same RUAccent the F5 voice uses
    places them; any failure leaves the line as written."""
    global _ACCENTOR
    acc = getattr(getattr(ctx, "models", None), "accentor", None)
    if acc is None or not getattr(getattr(ctx, "models", None), "accentor_loaded", False):
        try:
            if _ACCENTOR is None:
                from stress import BilingualAccentor
                from config import RUACCENT_MODEL, RUACCENT_WORKDIR, RUACCENT_DEVICE
                _ACCENTOR = BilingualAccentor(ru_model=RUACCENT_MODEL, ru_workdir=RUACCENT_WORKDIR,
                                              device=RUACCENT_DEVICE, enable_english=False)
            acc = _ACCENTOR
        except Exception as exc:
            logger.debug("no accentor for video speech: %s", exc)
            return prompt

    def one(m):
        # Only the spoken words go through the accentor: RUAccent normalizes
        # punctuation away and turned the closing "</d>" into "d>" (live 10-02),
        # leaving the line unterminated for H3.
        head, line, tail = re.match(r"(«|\"|<d>(?:\[[^\]]*\]\s*)?)(.*?)(»|\"|</d>)$", m.group(0), re.S).groups()
        try:
            plus = acc(line)
        except Exception:
            return m.group(0)
        # «+» before the stressed vowel -> the vowel followed by U+0301; ё is stressed already
        out = re.sub(r"\+([аеиоуыэюяАЕИОУЫЭЮЯ])", "\\1\u0301", plus)
        # a one-vowel word has nowhere else to put the stress: no mark, less noise
        out = re.sub("[а-яё́]+", lambda w: w.group(0) if len(re.findall("[аеиоуыэюяё]", w.group(0), re.I)) > 1
                     else w.group(0).replace("́", ""), out.replace("+", ""), flags=re.I)
        return head + out + tail
    return _RU_LINE.sub(one, prompt)


def to_context_ir(ctx, description: str, *, mode: str, seconds: float,
                  images: Sequence[str] = (), n_videos: int = 0, n_audios: int = 0) -> str:
    """Rewrite a free-text request into H3's own prompt structure.

    MiniMax ships a prompt-writing skill (skills/h3-prompt-writing, copied to
    h3_prompt_guides/): base modes want integrated_multimodal_description /
    overall_soundscape / non_diegetic_music with [Shot N] timing, Ref2VA six
    sections with <Picture i> labels. The model was trained on prompts of that
    shape. Returns the original description when the rewrite fails or comes
    back without the required fields -- never a half-formed prompt.
    """
    ref = mode == "ref2va"
    try:
        guide = open(os.path.join(_GUIDE_DIR, "ref-en.txt" if ref else "base-en.txt"),
                     encoding="utf-8").read()
    except OSError:
        return description
    task = {"t2va": "T2VA", "i2va": "I2VA", "fl2va": "FL2VA", "ref2va": "Ref2VA"}.get(mode, "T2VA")
    user = (f"Mode: {task}. Target duration: {seconds:.2f} seconds. "
            f"Reference pictures attached in order: {len(images)}; reference videos: "
            f"{n_videos}; reference audios: {n_audios}.\n"
            # Live 10-02 the summary said «[video continuation + audio reference]» with
            # no video at all: the task type names only what the references really do.
            + ("" if n_videos else "There is NO reference video: the summary's task type is "
               "never 'video continuation' or 'video editing' -- pictures are 'reference "
               "generation', voice samples are 'audio reference'.\n") +
            "Rewrite the request below into the final prompt exactly as the guide's final "
            "prompt structure prescribes (instruction line first where the mode needs one, "
            "then the fields in order). Output ONLY the final prompt, no commentary.\n"
            # Live 10-01: «сочные удары, эпичный баян» came out as «muffled thudding
            # hits» in the soundscape and a whisper of accordion. The guide's own
            # rules: a synchronized sound sits next to its action, the mix is stated.
            "SOUND RULES -- H3 places a sound on the frame by the words beside it. "
            "(1) EVERY visible thing that makes a sound gets that sound in the SAME sentence "
            "as its action, with a timing word, whether or not the user mentioned it: 'as "
            "the boots hit the cobbles in step, a heavy rhythmic stomp', 'as she tips the "
            "jug, water splashes loudly into the basin', 'as the fist lands, a loud wet "
            "crack'. A sound listed apart from its action ('marching sounds', 'punch "
            "sounds') plays late or not at all; each repeated hit, step or slam gets its "
            "own. Keep the user's strength words (juicy, loud, crisp); never soften them "
            "into 'muffled' or move them to the soundscape. "
            "(2) An instrument someone plays ON SCREEN is that person's action: hands, "
            "strings or bellows and its sound written in the shot, never non_diegetic_music. "
            "(3) overall_soundscape always names the scene's own continuous background "
            "sources even when the user said nothing -- wind in the grass, a crowd's murmur, "
            "rain on a roof, traffic, room tone -- because H3 fills silence with mumbling; "
            "end it with one mix sentence that follows the request: what is loud and up "
            "front, what sits under it. "
            "(4) non_diegetic_music is only a score nobody in the scene plays, given by "
            "instruments, tempo and dynamics, never mood words like 'epic'; when the user "
            "asked for music it is clearly audible: under the dialogue during lines, "
            "swelling back up between them, never 'faint' or 'distant' unless they said so.\n\n"
            "Request:\n" + description)
    import llm as _llm
    try:
        if images:
            out = _llm.analyze_image_with_llm(ctx, image_path=images[0], user_text=user,
                                              system_prompt=guide, temperature=0.4,
                                              max_tokens=2500)
        else:
            out = _llm.call_llm_simple(ctx, guide, user, temperature=0.4, max_tokens=2500)
    except Exception as exc:
        logger.warning("Context-IR rewrite failed: %s", exc)
        return description
    out = (out or "").strip().strip("`").strip()
    need = _CIR_REF_FIELDS if ref else _CIR_FIELDS[:1]
    if not out or not all(f in out for f in need) or "non_diegetic_music" not in out:
        logger.info("Context-IR rewrite rejected (missing fields); keeping the plain prompt")
        return description
    # The guide's cut time is "At 00:03.500"; the model also wrote "At 04.00".
    out = re.sub(r"\bAt (\d{1,2})\.(\d{1,3})\b(?!:)",
                 lambda m: f"At 00:{int(m.group(1)):02d}.{m.group(2).ljust(3, '0')}", out)
    # The guide separates the fields with one blank line; the model often
    # writes them back to back.
    return re.sub(r"\n+(?=(?:overall_soundscape|non_diegetic_music|summary|retention_analysis|"
                  r"detailed_description|integrated_multimodal_description):)", "\n\n", out)


AUDIO_REF_TOTAL_MS = 15000   # H3: 2-15 s per audio reference, 15 s for all of them


def fit_audio_refs(audios: Sequence[str]) -> list:
    """Each voice sample cut to its densest speech so all of them fit H3's 15 s
    audio budget. Live 10-01: two 43-51 s samples went in whole and the clip
    spoke with neither voice. A sample that cannot be read stays as it is."""
    from pydub import AudioSegment
    import voice_clone
    audios = list(audios)
    cap = max(2000, AUDIO_REF_TOTAL_MS // max(1, len(audios)) - 500)
    out = []
    for p in audios:
        try:
            seg = AudioSegment.from_file(p)
            if len(seg) <= cap + 400:
                out.append(p)
                continue
            try:
                cut = voice_clone.pick_speech(seg, min_ms=min(6000, cap), max_ms=cap)
            except voice_clone.CloneError:
                cut = seg[:cap]
            dst = os.path.splitext(p)[0] + f".h3ref{cap // 1000}s.wav"
            cut.export(dst, format="wav")
            logger.info("audio ref %s: %.1fs -> %.1fs", os.path.basename(p), len(seg) / 1000, len(cut) / 1000)
            out.append(dst)
        except Exception as exc:
            logger.warning("could not fit audio ref %s: %s", p, exc)
            out.append(p)
    return out


REF_MEDIA_MAX = 12           # H3 Ref2VA: no more than 12 reference files of all kinds


def _seconds(path: str) -> float:
    try:
        p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", path], capture_output=True, text=True, timeout=30,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return float(p.stdout.strip() or 0)
    except Exception:
        return 0.0


def fit_video_refs(videos: Sequence[str]) -> list:
    """Reference clips cut to H3's budget (2-15 s each, 15 s for all): the head
    of each clip, which is where a «как в том ролике» motion starts."""
    videos = list(videos)
    cap = max(2.0, AUDIO_REF_TOTAL_MS / 1000 / max(1, len(videos)) - 0.3)
    out = []
    for p in videos:
        if _seconds(p) <= cap + 0.3:
            out.append(p)
            continue
        dst = os.path.splitext(p)[0] + f".h3ref{int(cap)}s.mp4"
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p, "-t", f"{cap:.2f}", "-c:v", "libx264",
                            "-preset", "veryfast", "-c:a", "aac", dst], capture_output=True, timeout=300,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        out.append(dst if r.returncode == 0 and os.path.exists(dst) else p)
        logger.info("video ref %s cut to %.1fs", os.path.basename(p), cap)
    return out


def fit_references(images: Sequence[str], videos: Sequence[str], audios: Sequence[str]) -> tuple:
    """Everything H3 is handed obeys its reference limits, whoever called:
    counts per kind, 12 files in all (the newest pictures kept), and the
    15-second budgets for clips and for voices."""
    videos = fit_video_refs(list(videos)[:MAX_REF_VIDEOS])
    audios = fit_audio_refs(list(audios)[:MAX_REF_AUDIOS])
    room = max(0, REF_MEDIA_MAX - len(videos) - len(audios))
    images = list(images)[-min(MAX_REF_IMAGES, room):] if room else []
    return images, videos, audios


def generate_video(ctx, description: str, *,
                   images: Sequence[str] = (), videos: Sequence[str] = (),
                   audios: Sequence[str] = (),
                   seconds: float = 0.0, aspect: str = "",
                   seed: Optional[int] = None,
                   steps: Optional[int] = None,
                   on_progress: Optional[Callable] = None,
                   context_video: str = "") -> dict:
    """Render one clip. `context_video`: a clip whose end this one continues (pinned
    frames, see _add_motion_context); its pinned head is cut off the result. Returns {path, mode, seconds, width, height, seed, status}.

    `status` is "success" or "fail"; on failure `_VIDEO_FAILURE["reason"]` says
    why in terms the tool layer can turn into an honest sentence.
    """
    _VIDEO_FAILURE.update({"reason": "server_error", "detail": ""})

    ok, why = engine_available(ctx)
    if not ok:
        _VIDEO_FAILURE.update({"reason": "unavailable", "detail": why})
        logger.error("video engine unavailable: %s", why)
        return {"path": None, "status": "fail", "reason": why}

    if not (description or "").strip():
        _VIDEO_FAILURE.update({"reason": "empty_prompt",
                               "detail": "no description was given"})
        return {"path": None, "status": "fail", "reason": "empty description"}

    images = [p for p in (images or ()) if p and os.path.exists(p)]
    videos = [p for p in (videos or ()) if p and os.path.exists(p)]
    audios = [p for p in (audios or ()) if p and os.path.exists(p)]
    images, videos, audios = fit_references(images, videos, audios)

    asked = description
    description = _lock_framing_unless_requested(description)
    mode = pick_mode(images, videos, audios)
    # Ref2VA's workflow carries the LightX2V Turbo LoRA baked in (see
    # workflow_video_h3_ref.json) and is distilled for 4 steps, not the plain
    # model's 8 -- an explicit caller-supplied `steps` still wins.
    if steps is None:
        steps = (VIDEO_STEPS_REF2VA_VOICES if audios else VIDEO_STEPS_REF2VA) if mode == "ref2va" else VIDEO_STEPS
    if not seconds:
        seconds = estimate_seconds(asked)
        logger.info("video length from the script: %.1fs", seconds)
    frames = seconds_to_frames(seconds)
    if context_video:
        # the pinned head is rendered too, then trimmed: the clip grows by it
        frames = snap_frames(frames + MOTION_CONTEXT_FRAMES - 5)
    src_size = _image_size(images[0]) if images else None
    if context_video and not src_size:              # a continuation keeps the clip's own shape
        p = probe(context_video)
        src_size = (p["width"], p["height"]) if p.get("width") else None
    width, height = resolve_size(aspect, src_size)
    if _context_ir_on():
        locked = description.endswith(_FRAMING_LOCK_SUFFIX)
        description = to_context_ir(ctx, description, mode=mode,
                                    seconds=frames_to_seconds(frames), images=images,
                                    n_videos=len(videos), n_audios=len(audios))
        if locked:
            description = _with_lock(description, _FRAMING_LOCK_SUFFIX)
    if 1 <= len(images) <= 2 and not videos:     # a photo to animate, not a set of references
        restyle = _asks_new_style(asked)
        description = _with_lock(description, _START_LOCK_SUFFIX)
        if not restyle:
            description = _with_lock(description, _STYLE_LOCK_SUFFIX)
    if VIDEO_SPEECH_STRESS:
        description = mark_speech_stress(ctx, description)
    if seed is None or int(seed) < 1:
        seed = random.randint(1, 2**31 - 1)

    logger.info("H3 %s: %dx%d, %d frames (%.1fs), seed %d, %d image(s) %d video(s)",
                mode, width, height, frames, frames_to_seconds(frames), seed,
                len(images), len(videos))
    logger.info("H3 prompt: %s", description[:4000])    # what the model actually got
    if ctx is not None and hasattr(ctx, "set_stage"):
        ctx.set_stage("Generating a video")

    try:
        wf = build_workflow(description, mode=mode, width=width, height=height,
                            frames=frames, seed=seed, images=images, videos=videos,
                            audios=audios, steps=steps, ctx=ctx,
                            context_video=context_video)
        if ctx is not None and hasattr(ctx, "set_stage"):
            ctx.set_stage("Generating a video")
        wf = _normalize_save_node(wf)
    except VideoUnavailable as exc:
        _VIDEO_FAILURE.update({"reason": "unavailable", "detail": str(exc)})
        logger.error("could not build the video graph: %s", exc)
        return {"path": None, "status": "fail", "reason": str(exc)}

    # The whole card, nothing less: the H3 DiT is ~19.4 GB of Q4 weights and
    # nothing but ComfyUI runs between submit and file. Left NON-exclusive the
    # chat model stayed resident, the DiT got 13.4 GB of the card, 6.2 GB were
    # offloaded to RAM and a 5-second clip took 25 minutes at 190 s/step
    # (live, 2026-09-12, journey 28) -- the chat turn that followed then
    # fought the render for the card and timed out three times.
    out = comfy_client._submit_and_poll(ctx, wf, timeout=VIDEO_JOB_TIMEOUT,
                                        label=f"H3 {mode}", on_progress=on_progress,
                                        validate=_valid_video_file,
                                        job_timeout=VIDEO_JOB_TIMEOUT,
                                        exclusive=True,
                                        min_free_mb=_config.VIDEO_MIN_FREE_MB)
    if not out:
        if ctx is not None and getattr(ctx, "is_cancelled", None) and ctx.is_cancelled():
            _VIDEO_FAILURE.update({"reason": "cancelled", "detail": "cancelled"})
            return {"path": None, "status": "cancelled", "reason": "cancelled"}
        # The real cause, not a blank: with only "no file" the chat model made
        # one up for the user (live 2026-10-03).
        why = comfy_client.last_failure() or "the render produced no file"
        _VIDEO_FAILURE.update({"reason": "server_error", "detail": why})
        return {"path": None, "status": "fail", "reason": why}

    final = _adopt_output(out)
    if ctx is not None:
        ctx.last_video_path = final
        ctx.last_video_prompt = description
    return {"path": final, "mode": mode, "seconds": frames_to_seconds(frames),
            "frames": frames, "width": width, "height": height, "seed": seed,
            "status": "success"}


def _adopt_output(path: str) -> str:
    """Move the clip out of ComfyUI's output tree into ours (see
    comfy_client.adopt_output). OUTPUT_DIR is passed at call time: it is the
    global the suites rebind."""
    return comfy_client.adopt_output(path, "clip", OUTPUT_DIR)


# --------------------------------------------------------------------------- #
# Delivery helpers
# --------------------------------------------------------------------------- #
def probe(path: str) -> dict:
    """Duration / dimensions / audio presence, for delivery decisions."""
    info = {"seconds": 0.0, "width": 0, "height": 0, "has_audio": False,
            "bytes": 0}
    try:
        info["bytes"] = os.path.getsize(path)
    except OSError:
        return info
    ff = shutil.which("ffprobe")
    if not ff:
        return info
    try:
        out = subprocess.run(
            [ff, "-v", "error", "-print_format", "json", "-show_format",
             "-show_streams", path],
            capture_output=True, text=True, timeout=60)
        data = json.loads(out.stdout or "{}")
        for s in data.get("streams", []):
            if s.get("codec_type") == "video" and not info["width"]:
                info["width"] = int(s.get("width") or 0)
                info["height"] = int(s.get("height") or 0)
            if s.get("codec_type") == "audio":
                info["has_audio"] = True
        info["seconds"] = round(float(data.get("format", {}).get("duration") or 0), 2)
    except Exception as exc:
        logger.debug("ffprobe failed on %s: %s", path, exc)
    return info


CONTINUE_PREFIX = (
    "Continue the shot of <Video 1> directly, with no cut and no reset: the new footage begins "
    "exactly where <Video 1> ends, on <Picture 1>, and keeps its motion going. The same people with "
    "the same faces, location and lighting, and everyone exactly as they are in <Picture 1> (its last "
    "frame): whatever changed during <Video 1> -- clothes taken off or put on, things moved, broken "
    "or picked up -- stays changed. The camera keeps moving at the same speed. "
    "Their voices and the ambient sound of <Video 1> go on unchanged. What happens next: ")

# With pinned frames the previous shot is not a reference at all, it IS the opening of
# this clip, and it carries the faces too. No <Picture 1> on purpose: with the end frame as
# a ref2va reference the model played the pins and then CUT to a fresh shot of the next
# action (bench/motion_context_ab.py, 10-07, twice of twice); the plain FL2V graph with the
# same pins went straight on, in 166 s against the old way's 348 s.
CONTINUE_CTX_PREFIX = (
    "The shot continues with no cut and no reset: the same people, place, framing and "
    "lighting, picking up exactly where the opening frames leave off; whatever already "
    "changed stays changed, and voices and ambient sound go on unchanged. What happens next: ")


def bridge_part(ctx, last_frame: str, part: str) -> str:
    """The next part of a long script with one bridging sentence in front, so it starts
    from `last_frame` (where the previous part really ended). A part that names a place or
    a thing not in that frame ("sweeps the carrots into a steaming pot" while he stands at a
    bare board) made the model conjure the pot out of nothing right after the pinned frames
    (10-07). A bridge saying only where the pot was had it slide into the shot on its
    own; the owner: things do not move by themselves, a hand brings them in. Asked to REWRITE the part, Gemma echoed it unchanged, so it is asked what is
    missing and for the move that brings it in, and that sentence goes first. The part
    itself is never touched; any failure keeps it as it was."""
    if not (part and last_frame and os.path.exists(last_frame)):
        return part
    guide = ("The image is the last frame of a video so far. The next part of its script is "
             "given. List every place, object or person the next part's FIRST action needs that "
             "is not visible in the frame or not within the person's reach. If any, write ONE "
             "short sentence in English of how the person gets it first. Things never move "
             "by themselves: a person picks the object up off-frame with their hands and "
             "brings it into the shot, or walks to the place -- unless the script says "
             "otherwise. E.g. 'He reaches off-frame to the right, lifts a steaming pot with "
             "both hands and sets it on the counter in front of him.' or 'She walks to the "
             "door at the left.' -- only that movement, never the action itself. Reply as JSON only: "
             '{"missing": ["..."], "bridge": "sentence, or empty when nothing is missing"}')
    import llm as _llm
    from utils import safe_json_from_llm
    try:
        raw = _llm.analyze_image_with_llm(ctx, image_path=last_frame, user_text="Next part: " + part,
                                          system_prompt=guide, temperature=0.2, max_tokens=400)
    except Exception as exc:
        logger.warning("bridge_part failed: %s", exc)
        return part
    got = safe_json_from_llm(raw or "", ["bridge"]) or {}
    bridge = str(got.get("bridge") or "").strip()
    if not got.get("missing") or not bridge or len(bridge) > 300:
        logger.info("bridge_part: nothing to bridge (%s)", got.get("missing"))
        return part
    logger.info("bridge_part: missing %s -> %s", got.get("missing"), bridge)
    return bridge[:1].upper() + bridge[1:].rstrip(".") + ". Then " + part[:1].lower() + part[1:]


def new_people_clause(first: int, n: int) -> str:
    """The continuation brings in people who are not in <Video 1>, one per picture
    from <Picture first>: each enters the shot and keeps that picture's face, hair,
    body and clothes (only the person is taken from it, not its background)."""
    tags = [f"<Picture {first + i}>" for i in range(n)]
    who = tags[0] if n == 1 else ", ".join(tags[:-1]) + " and " + tags[-1]
    return (f" NEW PEOPLE: the person shown in {who} is not in <Video 1>"
            if n == 1 else f" NEW PEOPLE: the people shown in {who} are not in <Video 1>") + (
        " -- they come into this same location from off-screen and join the scene as the "
        "request above says, each with exactly the face, hair, body and clothes of their own "
        "picture. Only the person is taken from that picture, never its background or framing; "
        "the people already in the shot stay as they are.")


def join_pinned(src: str, new: str, head: int = MOTION_CONTEXT_FRAMES) -> Optional[str]:
    """`src` without its last `head` frames, then `new` whole -- `new` opens on the model's own
    rendering of exactly those frames (the pinned head), so the cut lands between two
    near-identical frames and the turn into the new action happens inside one clip.
    A crossfade at the end of the pins blended two different poses into a double image
    (live 10-07, the stirring hand). Sound is cut at the same point, with a 10 ms blend.
    """
    ff = shutil.which("ffmpeg")
    a, b = probe(src), probe(new)
    keep = a["seconds"] - head / 24.0
    if not ff or keep <= 0.1 or not (b["width"] and b["height"] and a["has_audio"] and b["has_audio"]):
        return None
    w, h = b["width"], b["height"]
    vnorm = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24,format=yuv420p"
    anorm = "aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"
    fc = (f"[0:v]{vnorm},trim=end={keep:.4f},setpts=PTS-STARTPTS[v0];[1:v]{vnorm}[v1];"
          f"[v0][v1]concat=n=2:v=1:a=0[v];"
          f"[0:a]{anorm},atrim=end={keep + 0.01:.4f},asetpts=PTS-STARTPTS[a0];[1:a]{anorm}[a1];"
          f"[a0][a1]acrossfade=d=0.01[a]")
    fh = tempfile.NamedTemporaryFile(prefix="vidjoin_", suffix=".mp4", delete=False)
    fh.close()
    cmd = [ff, "-y", "-v", "error", "-i", src, "-i", new, "-filter_complex", fc,
           "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", fh.name]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0 or not os.path.exists(fh.name) or os.path.getsize(fh.name) == 0:
        logger.warning("join_pinned failed: %s", (r.stderr or "")[-400:])
        _discard(fh.name)
        return None
    return fh.name


def join_continuation(src: str, new: str, fade: float = 0.25, drop_frames: int = 2) -> Optional[str]:
    """`src` followed by its generated continuation `new`, as one clip, or None when ffmpeg fails.

    The continuation opens on (nearly) the last frame of `src`, so its first frames are
    dropped (the model regenerates the overlap and the first two often glitch, and the seed
    frame would otherwise be held twice), then the picture and the sound crossfade over `fade`
    seconds. Both sides are brought to the continuation's size, 24 fps and 48 kHz stereo; a
    side without sound gets silence so the audio crossfade has something to blend.
    ponytail: no colour matching across the join -- add a per-channel histogram match if grading drifts.
    """
    ff = shutil.which("ffmpeg")
    a, b = probe(src), probe(new)
    if not ff or not (a["seconds"] > fade * 2 and b["seconds"] > fade * 2 and b["width"] and b["height"]):
        return None
    w, h = b["width"], b["height"]
    skip = drop_frames / 24.0
    off = max(0.0, a["seconds"] - fade)
    vnorm = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24,format=yuv420p"
    anorm = "aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"
    cmd = [ff, "-y", "-v", "error", "-i", src, "-i", new]
    sil = []
    for i, p in enumerate((a, b)):
        if not p["has_audio"]:
            sil.append(i)
            cmd += ["-f", "lavfi", "-t", f"{p['seconds']:.2f}", "-i", "anullsrc=r=48000:cl=stereo"]
    # input index of each side's audio: its own stream, or the silence appended after the two clips
    aidx = {}
    nxt = 2
    for i in (0, 1):
        if i in sil:
            aidx[i] = nxt
            nxt += 1
        else:
            aidx[i] = i
    fc = (f"[0:v]{vnorm}[v0];[1:v]trim=start={skip:.4f},setpts=PTS-STARTPTS,{vnorm}[v1];"
          f"[v0][v1]xfade=transition=fade:duration={fade}:offset={off:.3f}[v];"
          f"[{aidx[0]}:a]{anorm}[a0];[{aidx[1]}:a]atrim=start={skip:.4f},asetpts=PTS-STARTPTS,{anorm}[a1];"
          f"[a0][a1]acrossfade=d={fade}:c1=tri:c2=tri[a]")
    fh = tempfile.NamedTemporaryFile(prefix="vidjoin_", suffix=".mp4", delete=False)
    fh.close()
    cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "18", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", fh.name]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0 or not os.path.exists(fh.name) or os.path.getsize(fh.name) == 0:
        logger.warning("join_continuation failed: %s", (r.stderr or "")[-400:])
        _discard(fh.name)
        return None
    return fh.name


def _discard(tmp: Optional[str]) -> None:
    """Delete a scratch file we created and are not going to hand to anyone.

    Only ever called with a path this module made via NamedTemporaryFile — never
    with a render, which belongs to the user.
    """
    if not tmp:
        return
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
    except OSError:
        pass


def make_thumbnail(path: str) -> Optional[str]:
    """A first-frame JPEG, for chat clients that want a poster image."""
    ff = shutil.which("ffmpeg")
    if not ff:
        return None
    tmp = None
    try:
        fh = tempfile.NamedTemporaryFile(prefix="vidthumb_", suffix=".jpg", delete=False)
        fh.close()
        tmp = fh.name
        subprocess.run([ff, "-y", "-v", "error", "-i", path, "-frames:v", "1",
                        "-q:v", "3", tmp], capture_output=True, timeout=120)
        if os.path.getsize(tmp) > 0:
            return tmp
    except Exception as exc:
        logger.debug("thumbnail failed for %s: %s", path, exc)
    # delete=False means a failed or empty ffmpeg run leaves the scratch file
    # behind forever: the caller only cleans up a thumbnail it was GIVEN, and we
    # are returning None.
    _discard(tmp)
    return None


def fit_for_telegram(path: str, max_bytes: Optional[int] = None, ctx=None) -> str:
    """Re-encode down if the clip is over Telegram's bot upload ceiling.

    A 15s 1344x768 clip can land over 50MB, and Telegram rejects the upload
    outright — which historically showed up as the bot narrating a video it never
    sent. Better a slightly softer clip than no clip.
    """
    cap = int(max_bytes or getattr(_config, "VIDEO_TG_MAX_BYTES", 50 * 1024 * 1024))
    try:
        if os.path.getsize(path) <= cap:
            return path
    except OSError:
        return path
    ff = shutil.which("ffmpeg")
    if not ff:
        logger.warning("clip is over the Telegram limit and ffmpeg is not available")
        return path
    tmp = None
    try:
        # Re-encoding a 15s clip is another minute of waiting AFTER the render
        # finished — without a stage the chat looks hung at 100%.
        if ctx is not None and hasattr(ctx, "set_stage"):
            ctx.set_stage("Encoding the video")
        info = probe(path)
        secs = info.get("seconds") or frames_to_seconds(VIDEO_DEFAULT_FRAMES)
        # Leave ~12% headroom for container overhead and the audio track.
        target_kbps = max(300, int((cap * 8 * 0.88) / max(1.0, secs) / 1000) - 128)
        fh = tempfile.NamedTemporaryFile(prefix="vidtg_", suffix=".mp4", delete=False)
        fh.close()
        tmp = fh.name
        subprocess.run([ff, "-y", "-v", "error", "-i", path,
                        "-c:v", "libx264", "-b:v", f"{target_kbps}k",
                        "-preset", "medium", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "128k", tmp],
                       capture_output=True, timeout=900)
        if os.path.exists(tmp) and 0 < os.path.getsize(tmp) <= cap:
            logger.info("re-encoded %s (%.1fMB) to fit Telegram (%.1fMB)",
                        os.path.basename(path), os.path.getsize(path) / 1e6,
                        os.path.getsize(tmp) / 1e6)
            return tmp
        logger.warning("re-encode did not get under the limit; sending as a document")
    except Exception as exc:
        logger.warning("could not re-encode %s for Telegram: %s", path, exc)
    # Every path below here returns the ORIGINAL, so the caller's cleanup (which
    # only deletes send_path when it differs from path) will never see the
    # re-encode. Left alone it strands up to a cap-sized file per oversize clip.
    _discard(tmp)
    return path
