import logging
import os
from pathlib import Path

logger = logging.getLogger("assistant.config")

# Load a project-local .env BEFORE any os.getenv() below runs — every knob in this
# file is read at import time, so a later load would be ignored. Without this the
# only way to set VK_TOKEN or tune the DR_* knobs was a system-wide `setx` plus a
# restart. Existing environment variables always win, so a real env var still
# overrides the file. Missing file or missing package is a silent no-op.
try:
    from dotenv import load_dotenv
    _ENV_FILE = Path(__file__).resolve().parents[1].joinpath(".env")
    # Test runs (F5_TEST_RUN) ignore the operator's .env: its values changed
    # the defaults the suites assert, so a machine that had run the setup
    # failed tests that pass on a clean checkout.
    if _ENV_FILE.exists() and not os.getenv("F5_TEST_RUN"):
        load_dotenv(_ENV_FILE, override=False)
        logger.info("Loaded environment overrides from %s", _ENV_FILE)
except Exception as exc:  # pragma: no cover - defensive
    logger.debug("No .env loaded: %s", exc)


# LM Studio, ComfyUI and the sandbox are local servers. With HTTP(S)_PROXY set
# (a VPN client or a corporate proxy) and no NO_PROXY, `requests` sent even
# 127.0.0.1 through the proxy and every local call failed while the servers
# ran fine. Loopback is always exempt; the user's own entries are kept.
def _exempt_loopback_from_proxy() -> None:
    need = ("127.0.0.1", "localhost", "::1")
    for var in ("NO_PROXY", "no_proxy"):
        have = [h.strip() for h in os.environ.get(var, "").split(",") if h.strip()]
        if "*" in have:
            continue
        missing = [h for h in need if h not in have]
        if missing:
            os.environ[var] = ",".join(have + missing)


_exempt_loopback_from_proxy()

def _env_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back to default on a bad value
    so a malformed override never crashes startup at import time."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        logger.warning("Invalid %s=%r — using default %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        logger.warning("Invalid %s=%r — using default %s", name, raw, default)
        return default


# Public names for the rest of the code base. A bare int(os.getenv(...)) at
# module scope turns one typo in .env (or an empty `TG_WORKERS=`) into an
# ImportError that keeps the whole app from starting, with a traceback that
# does not say which setting was wrong.
env_int = _env_int
env_float = _env_float


# ---------------------------------------------------------------------------
# External service URLs — overridable via environment variables
# ---------------------------------------------------------------------------
# 127.0.0.1, NOT "localhost". Measured on this machine: "localhost" resolves to
# ::1 first, LM Studio only listens on IPv4, and every request eats the ~2.06 s
# IPv6 connect timeout before falling back (127.0.0.1: ~10 ms). That tax is paid
# per LLM call -- translate-in, every tool round, translate-out, vision -- so a
# single 5-round turn was losing ~12 s to name resolution alone.
LM_STUDIO_BASE = os.getenv("LM_STUDIO_BASE", "http://127.0.0.1:1234")
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", f"{LM_STUDIO_BASE}/v1/chat/completions")
# Default LLM id as served by LM Studio's API (see GET /v1/models). The 9B-Q8
# fits fully in 12 GB VRAM alongside Whisper + F5-TTS; the 35B does not. The
# startup picker (model_selector) overrides this at runtime.
# The house model. The old 9B default could not plan an Ideogram layout — it
# returned no layout-shaped JSON on every attempt, so every storyboard silently
# fell back to a flat, box-less layout ("it edited without using the boxes").
# Gemma 4 26B plans it correctly first time.
# The id has to be the one LM Studio SERVES (GET /v1/models), not the repo it
# came from: "google/gemma-4-26b-a4b-qat" is not served here and anything
# comparing against it -- the bench's "am I talking to the house model?"
# check included -- silently answered no.
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-26b-a4b-qat")

# The model to start on when NOBODY CHOOSES ONE: the startup settings dialog
# timing out, or ASSISTANT_AUTOSTART skipping it entirely. Deliberately its own
# setting rather than MODEL_NAME, because the two answer different questions --
# MODEL_NAME is the house model a chosen session runs on, this is what an
# unattended launch falls back to. An app left sitting on that dialog is a bot
# that is simply offline with nothing in the log to explain it, which is how a
# forwarded message came to look like it had crashed the app.
STARTUP_MODEL = os.getenv("STARTUP_MODEL", MODEL_NAME)
# Seconds the dialog waits for a human before starting on STARTUP_MODEL. Any
# key or click cancels the countdown -- someone actively choosing must never be
# yanked out from under. 0 disables the timeout entirely.
STARTUP_DIALOG_TIMEOUT_S = _env_int("STARTUP_DIALOG_TIMEOUT_S", 30)
COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8000")
# Every generation lives under runtime/ (2026-09-27): ComfyUI writes here too, not Documents.
OUTPUT_DIR_COMFY = Path(os.getenv("COMFY_OUTPUT_DIR", str(Path(__file__).resolve().parents[1] / "runtime" / "comfy")))
INPUT_DIR_COMFY = Path(os.getenv("COMFY_INPUT_DIR", str(Path(__file__).resolve().parents[1] / "runtime" / "comfy_in")))

# Fallback city for the "What to wear" button when a user has never picked
# their own (each user's own pick, once made, overrides this -- see
# weather.py / User.prefs["city"]).
WEATHER_DEFAULT_CITY = os.getenv("WEATHER_DEFAULT_CITY", "Saint Petersburg")

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
ASSISTANT_ACTOR = "DC"
DEFAULT_ACTOR_SPEED = 1.0
TARGET_DBFS = -20.0
# Microphone pre-amplification in dB. 0 = none (recommended). Higher values
# boost quiet mics but clip/distort a normal-level mic — keep it small.
MIC_GAIN_DB = _env_float("MIC_GAIN_DB", 0.0)
# Hands-free VAD voice mode (GUI "🎙 VAD" toggle): Silero VAD listens to the mic
# continuously and detects utterance boundaries by itself — no buttons.
VAD_THRESHOLD = _env_float("VAD_THRESHOLD", 0.5)            # speech probability for a voiced frame
VAD_END_SILENCE_S = _env_float("VAD_END_SILENCE_S", 0.9)    # trailing silence that ends an utterance
VAD_MIN_SPEECH_S = _env_float("VAD_MIN_SPEECH_S", 0.4)      # discard shorter voice bursts as noise
VAD_PRE_ROLL_S = _env_float("VAD_PRE_ROLL_S", 0.4)          # audio kept from just before speech onset
VAD_MAX_UTTERANCE_S = _env_float("VAD_MAX_UTTERANCE_S", 45.0)  # force-cut runaway utterances
# Trailing silence appended to every synthesized reply, so playback truncation
# (Windows async audio often clips the last few ms) eats silence, not the last word.
TTS_TAIL_SILENCE_MS = _env_int("TTS_TAIL_SILENCE_MS", 250)
# === DO NOT REMOVE, DO NOT "CLEAN UP" WITH A REGEX ===============================
# Punctuation appended to the text before synthesis. This is the single fix for
# the one defect this Russian F5 fine-tune has: it eats the end of the phrase.
#
# Why it works: F5-TTS does not know how long the output should be. It estimates
# the duration from the BYTE LENGTH of gen_text (utils_infer._infer_basic:
# duration = ref_audio_len + ref_audio_len / ref_text_len * gen_text_len), and
# the estimate runs short, so roughly the last second is never generated and the
# final word is swallowed. A run of trailing periods inflates gen_text_len, so
# the missing tail eats the sacrificial dots instead of real speech. The dots
# themselves are silent.
#
# This has been deleted or regex-swept several times across sessions, and every
# time the endings started clipping again in all three surfaces at once. If you
# are here to tidy trailing punctuation: the padding is appended EXACTLY ONCE, in
# audio.preprocess_text_for_synthesis, which every synthesis path goes through
# (the app via graph.py, Telegram via tg_transport.py, Skyrim via
# mantella_f5_server.py). tests/test_tts_end_padding.py fails if it goes missing.
#
# Do not append it anywhere else -- doubling it is path-dependent and audible as
# a long pause. Do not strip trailing dots after it. Note f5_tts.chunk_text
# splits on punctuation followed by WHITESPACE, so an unspaced run stays welded
# to the last chunk, which is exactly the chunk that clips.
TTS_END_PADDING = os.getenv("TTS_END_PADDING", "." * 15)
# Rewrite a written answer into spoken sentences before it is voiced (graph.speakable).
TTS_SPEECH_ADAPT = os.getenv("TTS_SPEECH_ADAPT", "1") not in ("0", "false", "False", "")
# Empty = auto-detect, so the assistant can reply in whatever language the user
# actually spoke. Set e.g. "en" or "fr" to force a single transcription language.
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "") or None

# Russian ASR engine. GigaAM (Sber, MIT, run through onnx-asr on the CPU) was
# measured against large-v3-turbo on two real corpora, 200 clips each
# (bench/asr_ru_shootout.py): corpus WER 10.4% vs 13.3% on crowd recordings,
# 6.2% vs 8.9% on read speech, and 20.1 s of CPU against 80.3 s of GPU for
# 12.5 minutes of audio.
#
# "auto" uses it ONLY when the language is pinned to Russian. That condition is
# not caution for its own sake: GigaAM is Russian-only and does not detect or
# refuse another language, it renders it as confident Russian nonsense -- and
# this same code transcribes the F5 reference clips, which are English. With
# WHISPER_LANGUAGE unset the app is asking for auto-detection, and only Whisper
# can answer that.
# DEFAULT IS WHISPER, and that is a decision, not an oversight: GigaAM lives in
# the Skyrim stack only (C:\llamacpp\mantella_whisper_proxy.py), where every
# speaker is known to be Russian. This app takes voice notes from arbitrary
# Telegram users, and a Russian-only engine would turn any other language into
# confident Russian nonsense without reporting anything.
#
# "auto" remains available and means "use GigaAM when WHISPER_LANGUAGE is
# pinned to ru" -- but it is not the default, so nobody enables the Russian-only
# engine here as a side effect of pinning a language.
ASR_ENGINE = os.getenv("ASR_ENGINE", "whisper")   # whisper | auto | gigaam
# large-v3-turbo (int8_float16 on CUDA): 1.0 GB VRAM vs 3.8 GB, 2x faster (2026-09-27 A/B).
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
GIGAAM_MODEL = os.getenv("GIGAAM_MODEL", "gigaam-v3-e2e-rnnt")

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so the project is relocatable
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = BASE_DIR / "model_212000.safetensors"
VOCAB_PATH = BASE_DIR / "vocab.txt"
# The F5 vocoder (charactr/vocos-mel-24khz: config.yaml + pytorch_model.bin).
# Absolute, not the relative "vocos" it used to be loaded from: that resolved
# against whatever the CWD was, so a launch from another folder lost the voice.
VOCOS_DIR = Path(os.getenv("VOCOS_DIR", str(BASE_DIR / "vocos")))
DC_REF_WAV = Path(os.getenv("ASSISTANT_REF_WAV", BASE_DIR / "Stepan_short.wav"))
# Runtime working files: uploaded photos, contained crops, TTS wavs, scratch tiles.
# This used to be BASE_DIR/"tests" — the project's own test directory — so every
# session buried the suites in generated artifacts (3927 non-.py files by
# 2026-07-30) and a working crop delivered to a user carried a path inside the
# source tree. Overridable so an existing install can point back at the old
# location if something outside this repo depends on it.
OUTPUT_DIR = Path(os.getenv("ASSISTANT_OUTPUT_DIR", str(BASE_DIR / "runtime")))
# runtime/ is gitignored: a fresh clone has none, and a writer that forgot its own
# mkdir (resolve_ref_audio) failed only there.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def scratch_path(out_dir, name):
    """Where an intermediate render/mask/tile goes: <out_dir>/generated/_intermediate,
    out of the runtime/ root (owner 10-03). Takes the caller's OUTPUT_DIR so suites
    that rebind it to a temp dir keep fixtures out of the real tree."""
    d = Path(out_dir) / "generated" / "_intermediate"
    d.mkdir(parents=True, exist_ok=True)
    return d / name
CACHE_FILE = BASE_DIR / "transcriptions.json"

# ---- drawing engine --------------------------------------------------------
# Ideogram 4 (workflow_ideogram4.json) is the ONLY drawing engine: it composes a
# scene from a structured layout with per-element boxes, and our own pictures
# are later edited by moving those boxes. A user's own photo is edited by
# FireRed alone. The old model, Qwen-Image, BrushNet, FLUX.1-Fill and the ESRGAN
# upscalers were removed from the product. Ideogram REQUIRES the structured
# caption (a plain sentence yields a "blocked by safety filter" card), so
# ideogram.py plans the layout with the LLM before submitting.
IMAGE_ENGINE = "ideogram4"

# Refuse a ComfyUI job while a whole-card job (a LoRA training run) holds
# runtime/gpu.lock. Training needs ~18 of this card's 24 GB; a diffusion model
# loaded on top does not merely queue -- both processes thrash, the training
# step time nearly doubles, and the render itself usually dies on an OOM. So the
# honest answer is "not now", not a picture that costs an hour of training.
# Set COMFY_RESPECT_GPU_LOCK=0 to draw anyway.
COMFY_RESPECT_GPU_LOCK = os.getenv("COMFY_RESPECT_GPU_LOCK", "1").strip() not in ("0", "false", "no")
IDEOGRAM_STEPS = _env_int("IDEOGRAM_STEPS", 20)
IDEOGRAM_CFG = _env_float("IDEOGRAM_CFG", 7.0)
IDEOGRAM_SHIFT = _env_float("IDEOGRAM_SHIFT", 5.0)
# Ideogram 4 does NOT sample at a constant guidance. Confirmed straight from
# the real upstream registry (ideogram-oss/ideogram4, src/ideogram4/
# sampler_configs.py -- fetched and read 2026-09-19, not a third-party guide)
# -- there are only THREE published presets, not five:
#   V4_QUALITY_48: 45 steps @ gw=7, then 3 @ gw=3  (mu=0.0,  std=1.5)
#   V4_DEFAULT_20: 18 steps @ gw=7, then 2 @ gw=3  (mu=0.0,  std=1.75)
#   V4_TURBO_12:   11 steps @ gw=7, then 1 @ gw=3  (mu=0.5,  std=1.75)
# ("High-34"/"Fast-16" seen on a community node-docs page do not exist in the
# real registry -- do not resurrect them.) The low-guidance tail is what
# settles the artefacts high guidance leaves behind. We used to run a flat
# cfg=7 for 28 steps -- 28 wasn't even one of the published presets.
#
# A/B'd 2026-09-19 with bench/ideogram_polish_ab.py (same seed, two captions,
# flat vs polish2 vs polish3, all at 20 steps): polish2/3 read as marginally
# cleaner on fabric grain and skin, the CAFE ROSA sign stayed perfectly legible
# in all three -- that part is a real, measured win over the old flat-28.
#
# Live feedback the same night flagged eyes as wonky on a face-heavy render,
# so this was briefly bumped to V4_QUALITY_48 (45+3). Direct side-by-side
# feedback on the SAME seed/portrait: 48 steps looked no different from 20 --
# the eye issue isn't a step-count problem, so paying 2.3x the render time for
# it was pure waste. Reverted to V4_DEFAULT_20. If eyes are still off, the fix
# is elsewhere (face-region resampling/inpaint pass, not more global steps).
IDEOGRAM_POLISH_STEPS = _env_int("IDEOGRAM_POLISH_STEPS", 2)  # V4_DEFAULT_20's real 2-step tail
IDEOGRAM_POLISH_CFG = _env_float("IDEOGRAM_POLISH_CFG", 3.0)
# ostris/ideogram_4_turbotime_lora (2026-09-20): a continuous-turbo-trained
# LoRA good for 2-4 step, near-CFG-free sampling. Measured 35-40% faster
# across 3 varied scenes (lettering/portrait/busy_scene) with no quality loss
# on lettering and equal-or-better subject count on a crowd scene. On a
# face-heavy portrait it produced a visible colour-block glitch -- traced to
# add_polish_pass() still bolting on its normal 2-step, cfg=3.0 tail on top
# (steps=4, polish=2 < steps, so the polish branch fired regardless of the
# caller's own cfg), fighting a LoRA that is trained for one constant
# low/no-cfg pass throughout. generate() skips the polish pass whenever the
# turbo LoRA is engaged, which is the actual fix, not just a lower cfg.
# 2026-10-02: turbo is opt-in again -- the user judged the full int8 run
# (20 steps + polish tail) clearly more detailed than turbo4.
IDEOGRAM_TURBO = os.getenv("IDEOGRAM_TURBO", "0").strip() in ("1", "true", "yes")
IDEOGRAM_PHOTO_LORA = os.getenv("IDEOGRAM_PHOTO_LORA", "lenovo_ideogram4.safetensors")
IDEOGRAM_PHOTO_LORA_STRENGTH = _env_float("IDEOGRAM_PHOTO_LORA_STRENGTH", 0.8)
IDEOGRAM_TURBO_LORA = os.getenv("IDEOGRAM_TURBO_LORA", "ideogram_4_turbotime_v1.safetensors")
IDEOGRAM_STEPS_TURBO = _env_int("IDEOGRAM_STEPS_TURBO", 4)
IDEOGRAM_CFG_TURBO = _env_float("IDEOGRAM_CFG_TURBO", 1.0)
# When the planned layout asks for LETTERING in the picture, the render is read
# back and the words are compared to what was asked for; a garbled string
# ("POLICE" -> "AVCHKE") buys this many repair rounds — a bigger, correctly
# shaped text box and the spelling restated in the caption. Each round is a full
# re-render (~40s), so 1 by default; 0 turns the check off entirely. Pictures
# with no text never pay for it.
IDEOGRAM_TEXT_ROUNDS = _env_int("IDEOGRAM_TEXT_ROUNDS", 1)
# The model that TRANSCRIBES the lettering for that check. Empty = use whatever
# chat model is already loaded.
#
# This used to default to "google/gemma-4-12b-qat", because measured on eight
# test frames the then-house 9B chat model read a clean "STOP" as "HIT" and
# scored garbled renders higher than correct ones — it would have redrawn good
# pictures at random. That measurement is about the 9B, and the house model is
# now the 26B (MODEL_NAME above), so its premise no longer holds.
#
# Naming a DIFFERENT model here is not free: LM Studio has to make it resident,
# and a render already alternates reader -> critic -> reader across rounds. With
# ComfyUI simultaneously holding Ideogram 4 (two 9GB UNETs + a 10GB text encoder)
# there is no room on a 24GB card for a second chat model as well — observed live
# 2026-08-05 as "it loads a second model for some reason and everything freezes".
# Set it back to a specific model if the loaded chat model turns out to OCR badly
# AND there is VRAM headroom for both.
TEXT_READ_MODEL = os.getenv("TEXT_READ_MODEL", "").strip()
# inpaint_image is mask-free: it edits via FireRed Image Edit 1.1 (Qwen-Image-Edit
# family) — the whole image is re-rendered from a plain English instruction
# ("remove the glasses"), which handles add/remove/recolor/replace cleanly with no
# ghost frames or melted regions (the old CLIPSeg/Florence + controlnet mask
# pipeline is gone). Requires the FireRed transformer, Qwen 2.5-VL text encoder,
# Qwen VAE and the Lightning LoRA in ComfyUI (see workflow_firered_edit.json).
WORKFLOW_FIRERED_EDIT_PATH = BASE_DIR / "workflows" / "image" / "workflow_firered_edit.json"

# Reference-person image mode (OPT-IN, default OFF). When ON, a request to draw a
# real, named public figure first fetches an actual photo of them from the web and
# uses it as a visual identity REFERENCE (plus a read of their appearance), instead
# of a from-scratch text-to-image guess — so the likeness matches the real person.
# Toggle in the GUI settings (Image section) or via REFERENCE_PERSON_MODE=1.
REFERENCE_PERSON_MODE = os.getenv("REFERENCE_PERSON_MODE", "0") not in ("0", "false", "False", "")

# ---------------------------------------------------------------------------
# Video generation — MiniMax H3
# ---------------------------------------------------------------------------
# H3 is an omni-modal generator: it emits video WITH a native 32kHz stereo track
# from a single packed latent, so there is no separate audio pass to bolt on.
#
# It ships as two task-specific checkpoints and between them they cover every
# mode this project exposes:
#   FL2VA  — no image (text-to-video), one image (image-to-video),
#            two images (first-and-last-frame)
#   Ref2VA — omni-reference: up to 9 images, up to 3 video clips (2-15s each),
#            up to 3 audio clips. This is the "several images" and the
#            "video + image" mode.
#
# Requires ComfyUI >= 0.30.0 (the MiniMaxH3* nodes landed there) and the GGUF
# weights fetched by scripts/fetch_h3_weights.py. GGUF rather than the official
# fp8/nvfp4 packaging because this box is an Ampere 3090: fp8 needs sm_89+ and
# nvfp4 is Blackwell-only, so both would be upcast and lose their advantage.
VIDEO_ENGINE = os.getenv("VIDEO_ENGINE", "minimax_h3").strip().lower()
WORKFLOW_VIDEO_PATH = BASE_DIR / "workflows" / "video" / "workflow_video_h3.json"
WORKFLOW_VIDEO_REF_PATH = BASE_DIR / "workflows" / "video" / "workflow_video_h3_ref.json"

# H3 renders at 24 fps on a frame grid of 17k+5 — only those counts are legal, and
# video.py snaps to them. 124 frames = ~5.2s; MiniMax trained the model over roughly
# 124-362 (5-15s) and says longer is untested.
VIDEO_FPS = 24
VIDEO_DEFAULT_FRAMES = _env_int("VIDEO_DEFAULT_FRAMES", 124)
VIDEO_MAX_FRAMES = _env_int("VIDEO_MAX_FRAMES", 362)
# 768 short edge with a 768*1344 area cap, every axis a multiple of 32 — taken from
# the node's own adapt_canvas(); going outside it is not a quality knob, it is a
# shape the model was never trained on.
VIDEO_SHORT_EDGE = _env_int("VIDEO_SHORT_EDGE", 768)
VIDEO_MAX_PIXELS = _env_int("VIDEO_MAX_PIXELS", 768 * 1344)
VIDEO_CANVAS_MULTIPLE = 32
# CFG-distilled weights: guidance is baked in, so cfg stays at 1.0 and a negative
# prompt is mathematically inert. Raising it does not "strengthen" the prompt, it
# just doubles the cost of every step.
# T2VA now runs the LightX2V FL2V Turbo 4-step v1.2 768p LoRA (workflow_video_h3.json
# node 13). Night A/B 2026-09-24, same seed/prompts: 220 s vs 401 s per 5 s clip,
# speech exact, sharper hair/skin at shift 12 than 6, beard kept through the sip
# (the plain 8-step lost it), the asked-for wave reads clearly (plain: barely).
# F5-TTS flow steps. Keep to the EPSS table {5,6,7,10,12,16} (f5_tts.model.cfm):
# any other count falls back to a uniform schedule the model was never tuned for.
TTS_NFE_STEP = _env_int("TTS_NFE_STEP", 10)  # A/B 2026-09-24: 1.16s -> 0.81s, 0 real ASR errors
VIDEO_STEPS = _env_int("VIDEO_STEPS", 3)
# T2VA runs the TaoMate-H3 3-step LoRA (bench 2026-09-25: same look as the 4-step
# FL2V LoRA on t1/t2, 127-129 s vs 131-153 s). Ref2VA stays on its 4-step LoRA:
# the one tao3 ref2va render burned a subtitle into the frame.
VIDEO_T2VA_LORA_3STEP = os.getenv("VIDEO_T2VA_LORA_3STEP", "night\\taomate_h3_3step_comfyui_bf16.safetensors")
# Ref2VA runs the official LightX2V/ModelTC step-distilled Turbo LoRA (see
# workflow_video_h3_ref.json's LoraLoaderModelOnly node): the checkpoint is
# distilled specifically for 4 NFE, so it must be sampled at 4 steps, not the
# plain (non-distilled) FL2VA/T2VA default above. Briefly raised to 8 on a
# live "оч мыльно" complaint 2026-09-20, reverted back to 4 the same session
# at the user's own call -- back to the turbo LoRA path. video.build_workflow
# still zeroes the LoRA's strength if steps is ever explicitly pushed past 4
# (it is out of the LoRA's trained regime there), so that safeguard stays.
# 10-02 night: the graph now carries the lightx2v Ref2V 8-step v1.0 LoRA (distilled
# for 8 NFE, euler/beta, shift 12/3), so 8 is its own regime, not past it.
VIDEO_STEPS_REF2VA = _env_int("VIDEO_STEPS_REF2VA", 8)
# With voice samples the voices, the score and the hits converge last (H3 guide:
# "speech and voice timbre converge last"); live 10-01 at 4 steps the accordion
# was gone, at 8 it came through. +115 s on a 5 s clip (241 s -> 356 s).
VIDEO_SPEECH_STRESS = os.getenv("VIDEO_SPEECH_STRESS", "1") != "0"
# 8 steps WITH the turbo LoRA kept on (10-02, user's call): before, 8 steps
# zeroed the LoRA (build_workflow) and the bare model at cfg 1 turned a fist
# into mush. FL2V LoRA + er_sde/beta at 4 steps took 224 s.
# 12 in two stages (user pick 10-02): 249 s warm vs 267 s at 8, 348 s single-stage 8.
# 8 with the Ref2V 8-step LoRA (night A/B 10-02, user's call "8-step LoRA -> 8 steps"):
# 237 s vs 273 s at 12+2, voice similarity in the same range.
VIDEO_STEPS_REF2VA_VOICES = _env_int("VIDEO_STEPS_REF2VA_VOICES", 8)
VIDEO_CFG = _env_float("VIDEO_CFG", 1.0)
VIDEO_SHIFT_VIDEO = _env_float("VIDEO_SHIFT_VIDEO", 12.0)
VIDEO_SHIFT_AUDIO = _env_float("VIDEO_SHIFT_AUDIO", 3.0)
# A 33B transformer swapping through 24GB of VRAM is slow. This is the ceiling for
# one clip; the job is abandoned (and reported honestly) past it rather than hanging
# a chat turn forever.
VIDEO_JOB_TIMEOUT = _env_int("VIDEO_JOB_TIMEOUT", 5400)
# How much free VRAM comfy_client._submit_and_poll insists on before an H3 job's
# first submit, overriding lora_training.FREE_GPU_MIN_MB (16 GB -- calibrated
# only to confirm the chat model has drained, not sized for this pipeline).
# The DiT alone is ~19.4 GB of Q4 weights (see video.py's generate_video); with
# less than that plus the CLIP encoder and both VAEs free, ComfyUI holds part
# of the graph in system RAM and swaps it in per step instead of keeping
# everything resident. Live, 2026-09-19: a render that started with 16-17 GB
# free took 35-49 minutes instead of the usual 8-12 -- the two fast completions
# in the same window both started at 17.5-20 GB free.
VIDEO_MIN_FREE_MB = _env_int("VIDEO_MIN_FREE_MB", 20500)
# Telegram refuses uploads over 50MB from bots.
VIDEO_TG_MAX_BYTES = _env_int("VIDEO_TG_MAX_BYTES", 50 * 1024 * 1024)

# ---------------------------------------------------------------------------
# Music generation (MiniMax Music 3, text-to-music via a local ComfyUI)
# ---------------------------------------------------------------------------
# Node class names in workflow_music3.json are NOT yet confirmed against a live
# ComfyUI (the model is brand-new; see docs/music_generation.md) — built on the
# same UNETLoader/CLIPLoader/VAELoader shape as workflow_video_h3.json pending
# that check.
WORKFLOW_MUSIC_PATH = BASE_DIR / "workflows" / "music" / "workflow_music3.json"
MUSIC_JOB_TIMEOUT = _env_int("MUSIC_JOB_TIMEOUT", 3600)
MUSIC_DEFAULT_SECONDS = _env_int("MUSIC_DEFAULT_SECONDS", 60)
# 180, not 240: measured end to end, a 240s ask came back at 149.5s with the
# budget's own lyric and at 175.6s with half again as many lines. The planner
# compresses as the lyric grows, so the slot cannot be filled with more words.
# This is the clamp, so it also bounds a caller that passes a number directly
# rather than picking one. Raise it only with a measurement -- the tool is
# bench/music_duration_e2e.py.
MUSIC_MAX_SECONDS = _env_int("MUSIC_MAX_SECONDS", 180)
# The render CEILING may sit above the longest ask: per the model card the
# duration is an upper bound and the model ends on its own end-of-audio token,
# so the ask gets headroom (music.render_ceiling) and the song closes on its
# [outro] instead of being chopped at the slot. 9,000 acoustic frames = 360 s
# is the model's own hard limit.
MUSIC_CEILING_SECONDS = _env_int("MUSIC_CEILING_SECONDS", 240)
# 30 -> 20 after an A/B on one seed (15.09): the DiT stage is ~half of a ⚡
# render and scales with the step count; 20 was not worse on the ear. The
# user can raise it to 50 in the Quality screen (music.STEPS_RANGE).
MUSIC_STEPS = _env_int("MUSIC_STEPS", 20)
MUSIC_CFG = _env_float("MUSIC_CFG", 3.0)
# modulsx/MiniMax-Music-3-Turbo-FP8's 8-step consistency-distillation LoRA
# (2026-09-20): measured 25% faster wall-clock on a 60s song (124.2s -> 93.6s;
# only ~7% on a 20s song, since the DiT share of the render grows with
# duration). Always attached in build_workflow -- see MUSIC_STEPS_TURBO/
# MUSIC_CFG_TURBO below, which build_workflow uses in place of MUSIC_STEPS/
# MUSIC_CFG when steps is not explicitly overridden by the caller.
MUSIC_TURBO_LORA = os.getenv("MUSIC_TURBO_LORA", "minimax_music3_turbo_lora_8step.safetensors")
MUSIC_TURBO_STRENGTH = _env_float("MUSIC_TURBO_STRENGTH", 0.85)
MUSIC_STEPS_TURBO = _env_int("MUSIC_STEPS_TURBO", 8)
MUSIC_CFG_TURBO = _env_float("MUSIC_CFG_TURBO", 1.7)
# Music3 renders up to max_duration and stops there whether or not the song has
# finished, so a song that outlasts its slot is cut off mid-phrase at full
# volume (measured: three 30s renders all ended at 29.99s with last-250ms peaks
# of 0.57-0.999, where a real ending tapers to ~0). The fade makes the end
# sound deliberate instead of severed. The tiny fade-in is just click
# suppression on a waveform that can start at a non-zero sample.
# Seconds; 0 disables either.
MUSIC_FADE_OUT_S = _env_float("MUSIC_FADE_OUT_S", 2.0)
MUSIC_FADE_IN_S = _env_float("MUSIC_FADE_IN_S", 0.03)

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
# DEVICE / WHISPER_DEVICE / WHISPER_COMPUTE_TYPE are computed ON FIRST ACCESS,
# not at import time (PEP 562 module __getattr__ below).
#
# `import torch` costs ~1.9 s and ~250 MB of RSS. config.py is imported by 80+
# modules — the Telegram layer, the tools, every one of the ~190 standalone test
# scripts — and almost none of them ever look at a device string. Paying the
# whole Torch import just to answer "cuda or cpu?" made every process on this box
# start two seconds late and hold a quarter gigabyte it never used, which matters
# on a machine where LM Studio, Whisper and F5-TTS already share 96 GB.
#
# The three names still read exactly as before: `config.DEVICE` and
# `from config import DEVICE` both go through __getattr__, which imports Torch
# once, memoizes the answer into the module globals, and is never consulted for
# that name again. Only the modules that actually need a device pay for Torch,
# and those import Torch themselves anyway.
_DEVICE_NAMES = ("DEVICE", "WHISPER_DEVICE", "WHISPER_COMPUTE_TYPE")


# Set ASSISTANT_GPU_IDLE=1 to start the app WITHOUT taking the card: Whisper
# and F5-TTS go to CPU and the LLM is not loaded at all. This exists because
# the one job that needs the whole 3090 -- training a character LoRA -- runs
# for an hour, and the only way to watch it was to open the app, which loads
# ~20 GB of LLM and kills one side or the other. Voice and chat are then slow
# or unavailable, which is the honest trade and is stated on the loading
# screen rather than discovered.
GPU_IDLE = os.getenv("ASSISTANT_GPU_IDLE", "0").strip().lower() in ("1", "true", "yes")


def _resolve_devices() -> dict:
    """Probe CUDA once and return all three device settings together."""
    if GPU_IDLE:
        logger.warning("ASSISTANT_GPU_IDLE=1 — running on CPU, GPU left free")
        return {"DEVICE": "cpu", "WHISPER_DEVICE": "cpu",
                "WHISPER_COMPUTE_TYPE": "int8"}
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - Torch missing/broken
        logger.warning("Torch unavailable (%s) — falling back to CPU settings", exc)
        cuda = False
    return {
        "DEVICE": "cuda:0" if cuda else "cpu",
        "WHISPER_DEVICE": "cuda" if cuda else "cpu",
        "WHISPER_COMPUTE_TYPE": "int8_float16" if cuda else "int8",
    }


def __getattr__(name: str):
    """Lazily resolve the device settings; anything else is a real AttributeError."""
    if name in _DEVICE_NAMES:
        resolved = _resolve_devices()
        globals().update(resolved)   # memoize: __getattr__ is not called again
        return resolved[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Keep the lazy names discoverable by dir()/introspection."""
    return sorted(set(globals()) | set(_DEVICE_NAMES))


# ---------------------------------------------------------------------------
# Russian stress accentor (RUAccent — heavy char/word Transformer with a
# context-aware homograph ("omograph") model). turbo3.1 is the largest/newest
# omograph transformer. Kept on CPU by default: the 12 GB GPU is already shared
# by the LLM + Whisper + F5 + ComfyUI, and accentuation is text-only (fast on CPU).
# Weights live under the project (models/ruaccent), not inside venv.
# ---------------------------------------------------------------------------
RUACCENT_MODEL = os.getenv("RUACCENT_MODEL", "turbo3.1")
RUACCENT_DEVICE = os.getenv("RUACCENT_DEVICE", "cpu")
RUACCENT_WORKDIR = Path(os.getenv("RUACCENT_WORKDIR", str(BASE_DIR / "models" / "ruaccent")))

# English lexical stress via CharsiuG2P (byT5 Transformer G2P). Its IPA ˈ/ˌ stress
# is mapped back onto the orthographic vowel as a '+' so it matches the same
# character-level F5 vocab (CharsiuG2P does NOT emit Russian stress — RUAccent owns
# Russian). Set STRESS_ENGLISH=0 to disable the English path entirely.
STRESS_ENGLISH = os.getenv("STRESS_ENGLISH", "1") != "0"
CHARSIU_G2P_MODEL = os.getenv("CHARSIU_G2P_MODEL", "charsiu/g2p_multilingual_byT5_small_100")

# Manual per-word stress overrides (edited in the GUI "Stress" tab). Applied AFTER
# the automatic accentor, so a hand-set mark always wins for edge cases. JSON map
# {base_word: stressed_form}; hot-reloaded on change. See stress_overrides.py.
STRESS_OVERRIDES_PATH = Path(os.getenv("STRESS_OVERRIDES_PATH", str(BASE_DIR / "stress_overrides.json")))

# ---------------------------------------------------------------------------
# LLM / agent behaviour
# ---------------------------------------------------------------------------
MAX_TOOL_ROUNDS = 8          # max tool-call iterations per personality turn

# File work is inherently longer: unpack, list, search, read, edit, pack is six
# rounds before a single mistake, and on the sandbox bench a run that had done
# everything right ran out mid-edit and told the user "я выполнил действия, но
# не успел подвести итог". These rounds are also cheap -- no GPU, no network,
# milliseconds each -- so the ceiling that protects an image turn from spending
# minutes is simply the wrong ceiling here.
SANDBOX_TOOL_ROUNDS = _env_int("SANDBOX_TOOL_ROUNDS", 16)
# How many FULL renders a single turn may spend. The agent is told to inspect a
# picture and fix what is wrong "until it is right", which is correct advice with
# no cost attached to it — observed live, one turn ran redraw → inspect → redraw →
# inspect → draw, each a full render, monopolising ComfyUI for minutes while other
# users queued behind it. The loop still self-corrects, it just cannot do so
# indefinitely; on exhaustion the agent is told to deliver what it has and say
# plainly what it could not fix. Counts generate_image and redraw_image (whole-
# frame re-renders), not the localized edits the user explicitly asked for.
IMAGE_MAX_RENDERS_PER_TURN = _env_int("IMAGE_MAX_RENDERS_PER_TURN", 3)
# (room for search -> generate -> inspect -> inpaint -> inspect)
# Adaptive recovery budget: when the round budget is about to run out but the agent
# is actively dealing with a tool FAILURE (a broken path that needs an alternative
# chain), grant up to this many extra rounds so legitimate recovery isn't truncated.
# Only spent while errors persist; a healthy run never touches it.
RECOVERY_EXTRA_ROUNDS = _env_int("RECOVERY_EXTRA_ROUNDS", 4)
# After this many tool results in a row come back as errors (across DIFFERENT tools,
# not just identical retries), inject a "step back and re-plan" signal so the agent
# stops walking broken paths and either routes around them or answers honestly.
REPLAN_FAILURE_THRESHOLD = _env_int("REPLAN_FAILURE_THRESHOLD", 3)
LLM_MAX_RETRIES = 3          # total attempts before giving up on an LLM call
LLM_RETRY_BASE_DELAY = 0.5   # seconds; doubled each retry (exponential backoff)
API_MIN_INTERVAL = 1.0       # minimum seconds between outbound API calls

# --- Streaming watchdog: catch a model that GETS STUCK ----------------------
# Three independent stuck-modes a streaming completion can hit, each with its own
# guard so the worker thread can never block indefinitely (the old single
# timeout=1900 only fired after ~32 min of TOTAL socket silence and never at all
# while a degenerate model kept trickling tokens):
#   1. connect stall  -> can't even reach LM Studio.
#   2. mid-stream stall-> server accepted the request then stopped emitting tokens.
#   3. runaway stream -> model loops emitting tokens forever (never sends [DONE]).
LLM_CONNECT_TIMEOUT = _env_float("LLM_CONNECT_TIMEOUT", 20)   # s to first byte
# Per-read (between-chunks) timeout: no new token for this long => treat as stalled
# and abort the attempt (retriable). Far tighter than the old 1900s.
LLM_STREAM_STALL_TIMEOUT = _env_float("LLM_STREAM_STALL_TIMEOUT", 90)
# Absolute wall-clock cap for ONE streaming attempt regardless of trickle: a model
# stuck in a token loop streams forever (iter_lines never times out because bytes
# keep coming). Hit this and we stop reading and use whatever we have.
LLM_STREAM_MAX_SECONDS = _env_float("LLM_STREAM_MAX_SECONDS", 600)
# Hard cap on accumulated streamed characters (content + reasoning) — a degenerate
# repetition loop ("the the the…") is cut off here even if it streams fast.
LLM_STREAM_MAX_CHARS = _env_int("LLM_STREAM_MAX_CHARS", 200000)

# Repetition guard (llm._looks_degenerate). The char/wall-clock caps above are
# last-resort and let a stuck model burn 600s + 200k chars before giving up; this
# samples the tail every _CHECK_EVERY chars and aborts as soon as the output is
# provably looping. Set _CHECK_EVERY=0 to disable the guard entirely.
# Interval is kept just BELOW the tail width so consecutive windows overlap and
# no stretch of output goes uninspected (at 4000 vs a 1500-char tail, 2500 chars
# of every cycle were never examined). Two strikes are required to abort, so this
# also sets the waste ceiling on a stuck model: ~2x this many chars. Measured cost
# is 0.027 ms per check — ~2 ms of CPU per 100k chars generated, i.e. free.
LLM_REPEAT_CHECK_EVERY = _env_int("LLM_REPEAT_CHECK_EVERY", 1200)
LLM_REPEAT_TAIL_CHARS = _env_int("LLM_REPEAT_TAIL_CHARS", 1500)
LLM_REPEAT_CYCLE_MAX = _env_int("LLM_REPEAT_CYCLE_MAX", 300)
# Minimum total repeated span (cycle_len * repeats) before a cycle counts as a
# loop. Without it, cycle length 1 flags "------------", "....", or a number like
# 1200000 — all legitimate. Real loops repeat far past this.
LLM_REPEAT_MIN_SPAN = _env_int("LLM_REPEAT_MIN_SPAN", 200)

# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------
# Output size is chosen as (aspect ratio × quality tier) rather than a raw pixel
# pair: the ratio is what the user actually cares about, and the tier is what
# costs VRAM/time. Everything downstream still speaks in pixels — resolve_image_size
# is the single place that turns a choice into one.
IMAGE_ASPECTS: dict = {
    "1:1":  (1, 1),
    "4:3":  (4, 3),
    "3:4":  (3, 4),
    "3:2":  (3, 2),
    "2:3":  (2, 3),
    "16:9": (16, 9),
    "9:16": (9, 16),
}
# Tier -> target megapixels. "draft" is the historic 960x544 budget; the old
# default. It was 0.52 MP for EVERY image, which is where "why is it so blurry"
# came from — a 24 GB card has no business rendering at half a megapixel.
IMAGE_QUALITIES: dict = {
    "draft":    0.52,
    "standard": 1.0,
    "high":     1.6,
    "ultra":    2.3,
}
DEFAULT_IMAGE_ASPECT = os.getenv("DEFAULT_IMAGE_ASPECT", "16:9")
# Everyone starts on the CHEAPEST tier and opts up. One 24 GB card is shared by
# the LLM, Whisper, F5-TTS and ComfyUI, so rendering every casual picture at 1.6 MP
# spends the queue's time on people who never asked for it. Anyone who wants more
# raises it in Settings and it sticks for them.
DEFAULT_IMAGE_QUALITY = os.getenv("DEFAULT_IMAGE_QUALITY", "draft")
# Hard rails. Above ~2048 on a side the turbo models start repeating structures
# (two heads, doubled horizons) regardless of how much VRAM is free.
IMAGE_MAX_SIDE = _env_int("IMAGE_MAX_SIDE", 2048)
IMAGE_MIN_SIDE = 256


def resolve_image_size(aspect=None, quality=None):
    """(aspect, quality) -> (width, height), snapped to a multiple of 16.

    Unknown/empty values fall back to the defaults rather than raising: this is
    called with whatever a session or an LLM happened to store.
    """
    rw, rh = IMAGE_ASPECTS.get(aspect or "") or IMAGE_ASPECTS.get(
        DEFAULT_IMAGE_ASPECT) or (16, 9)
    mp = IMAGE_QUALITIES.get(quality or "") or IMAGE_QUALITIES.get(
        DEFAULT_IMAGE_QUALITY) or 1.6

    scale = ((mp * 1_000_000) / float(rw * rh)) ** 0.5
    w, h = rw * scale, rh * scale
    # Respect the ceiling before snapping, so the ratio survives the clamp.
    over = max(w, h) / IMAGE_MAX_SIDE
    if over > 1.0:
        w, h = w / over, h / over

    def _snap(n: float) -> int:
        return max(IMAGE_MIN_SIDE, int(round(n / 16.0)) * 16)

    return _snap(w), _snap(h)


DEFAULT_WIDTH, DEFAULT_HEIGHT = resolve_image_size()
DEFAULT_STEPS = 8
# Carried through the generate signature only; Ideogram takes its own
# IDEOGRAM_STEPS/IDEOGRAM_CFG.
DEFAULT_CFG = 1.0
MAX_IMAGE_REFINEMENT_ATTEMPTS = 3
# Hard wall-clock ceiling on ONE ComfyUI job (seconds). The historic default was
# 1900s (~32 min) threaded through every call site as `timeout=1900`; a wedged job
# (stuck queue, model-load loop) blocked the whole turn for that long with the
# stage frozen on "Drawing a picture". _submit_and_poll/_submit_and_collect now
# CLAMP every job's timeout to this. 900s covers the slowest legitimate workflow
# on the 12 GB 3060 (FireRed edit incl. model swap); raise via env if needed.
COMFY_JOB_TIMEOUT = _env_int("COMFY_JOB_TIMEOUT", 900)

# ---------------------------------------------------------------------------
# Web search (ddgs)
# ---------------------------------------------------------------------------
SEARCH_MAX_RESULTS = _env_int("SEARCH_MAX_RESULTS", 10)   # candidates; the top 5 that open are read
SEARCH_REGION = os.getenv("SEARCH_REGION", "wt-wt")        # wt-wt = worldwide; e.g. "us-en", "fr-fr"
SEARCH_SAFESEARCH = os.getenv("SEARCH_SAFESEARCH", "moderate")  # off | moderate | strict
SEARCH_TIMELIMIT = os.getenv("SEARCH_TIMELIMIT", "") or None    # None | d | w | m | y (recency)
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "auto")      # auto rotates engines
SEARCH_SNIPPET_CHARS = 500       # per-result snippet cap (keeps one result from hogging context)
SEARCH_TOTAL_CHARS = 4000        # total raw-results cap handed to the model
# Results kept per domain. Was effectively 1 (dedupe-by-domain), which meant a news
# query got one link per outlet — and if that outlet's homepage ranked first, the
# actual articles were discarded and the user received a link to the front page.
SEARCH_MAX_PER_DOMAIN = _env_int("SEARCH_MAX_PER_DOMAIN", 2)
# Distill raw results into grounded facts via SEARCHER_PROMPT before the main loop.
# Adds one extra LLM call per search but massively shrinks context and improves grounding.
SEARCH_DISTILL = os.getenv("SEARCH_DISTILL", "1") not in ("0", "false", "False", "")
SEARCH_DISTILL_MAX_TOKENS = 400  # enough for 1-4 sentences of distilled facts
# A snippet is the search engine's teaser, not the answer: the top results are
# OPENED and read, in parallel, and each page's text stands in for its snippet.
SEARCH_READ_PAGES = _env_int("SEARCH_READ_PAGES", 5)
SEARCH_PAGE_CHARS = _env_int("SEARCH_PAGE_CHARS", 1800)   # per opened page
SEARCH_FETCH_WORKERS = _env_int("SEARCH_FETCH_WORKERS", 16)
# One extra search round when the model judges the first one left a gap.
SEARCH_FOLLOWUP_ROUND = os.getenv("SEARCH_FOLLOWUP_ROUND", "1") not in ("0", "false", "False", "")

# ---------------------------------------------------------------------------
# Ultra Search / deep research (deep_research.py)
# A separate, aggressive research path: expand queries -> collect many sources
# -> crawl pages (requests + trafilatura) -> dedupe -> map/reduce synthesis into
# a structured report. Bounded by the caps below so it can't run away.
# ---------------------------------------------------------------------------
# Depth a run uses when the user has not picked one (Telegram: Search ▸ 🎚 Depth).
DR_DEFAULT_DEPTH = os.getenv("DR_DEFAULT_DEPTH", "standard").strip().lower()
if DR_DEFAULT_DEPTH not in ("quick", "standard", "deep"):
    DR_DEFAULT_DEPTH = "standard"
DR_MAX_QUERIES = _env_int("DR_MAX_QUERIES", 8)          # expanded search queries per run
DR_RESULTS_PER_QUERY = _env_int("DR_RESULTS_PER_QUERY", 6)  # search hits collected per query
DR_MAX_PAGES = _env_int("DR_MAX_PAGES", 24)             # hard cap on pages actually crawled
DR_MAX_DEPTH = _env_int("DR_MAX_DEPTH", 1)              # link-following depth (0 = no recursion)
DR_LINKS_PER_PAGE = _env_int("DR_LINKS_PER_PAGE", 3)   # in-domain links followed per page at depth>0
DR_PAGE_TIMEOUT = _env_int("DR_PAGE_TIMEOUT", 15)      # per-page fetch timeout (seconds)
DR_PAGE_CHARS = _env_int("DR_PAGE_CHARS", 32000)       # extracted text fed to the briefer per page. Raised
# 6000->32000: 6000 (~1500 tok) truncated long
# articles/papers, losing prose facts before briefing.
# Each page is briefed in its own call so 32k (~8k tok)
# fits any reasonable context; the 2MB _MAX_HTML_CHARS
# ceiling still guards pathological pages. Lower via env
# DR_PAGE_CHARS if runs get slow.
DR_BRIEF_TOKENS = _env_int("DR_BRIEF_TOKENS", 320)     # floor for a per-source brief; brief_source now sizes
# to the model context (no 320-tok truncation)
DR_REPORT_TOKENS = _env_int("DR_REPORT_TOKENS", 2000)  # max tokens for the synthesized report
DR_FETCH_DELAY = _env_float("DR_FETCH_DELAY", 0.4)     # polite delay between page fetches (seconds)
# I/O parallelism (1 = serial, the historical default). Search queries and page
# fetches are network-bound and independent, so a small pool cuts wall-clock time
# (esp. past the dead-engine ddgs timeout). Kept bounded + per-domain spaced to
# avoid rate-limit bans; LLM calls stay strictly serial (single 9B / 12GB VRAM).
DR_SEARCH_CONCURRENCY = _env_int("DR_SEARCH_CONCURRENCY", 16)   # parallel search queries; 16 measured 2026-10-01: 8 s vs 100 s for 16 queries, no empty results
DR_FETCH_CONCURRENCY = _env_int("DR_FETCH_CONCURRENCY", 16)     # parallel page fetches (per wave; distinct domains) -- at 1 one slow site idled the model 30 s
DR_FETCH_JITTER = _env_float("DR_FETCH_JITTER", 0.3)           # random +jitter on the politeness delay
DR_USER_AGENT = os.getenv(
    "DR_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
)
DR_DIR = BASE_DIR / "memory" / "research"   # persisted runs (progress + final reports)
DR_CACHE_ENABLED = os.getenv("DR_CACHE_ENABLED", "1") not in ("0", "false", "False")
DR_CACHE_DIR = DR_DIR / "_cache"             # URL->extracted-text cache (Phase 6)
DR_ENABLE_ADAPTERS = os.getenv("DR_ENABLE_ADAPTERS", "1") not in ("0", "false", "False")
DR_REPLAN_ENABLED = os.getenv("DR_REPLAN_ENABLED", "1") not in ("0", "false", "False")
# Second-stage reranker (rerank.py): scores crawled pages by semantic relevance
# to the topic and keeps only the top-K for briefing, so the LLM reads the
# strongest evidence, not the whole retrieval set. Relevance is fused with the
# existing authority score (RRF) — set weight 0 for relevance-only ordering.
DR_RERANK_ENABLED = os.getenv("DR_RERANK_ENABLED", "1") not in ("0", "false", "False")
DR_RERANK_BACKEND = os.getenv("DR_RERANK_BACKEND", "auto")   # auto|cross|dense|lexical
# Document library reranker (cross-encoder, CPU): hit@1 0.675 -> 0.863 on the
# 80-question ru eval. Empty string disables it.
LIBRARY_CROSS_ENCODER = os.getenv("LIBRARY_CROSS_ENCODER", "BAAI/bge-reranker-v2-m3")
DR_RERANK_MODEL = os.getenv("DR_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
DR_RERANK_TOP_K = _env_int("DR_RERANK_TOP_K", 0)            # pages kept for briefing; 0 = keep ALL crawled
# pages (rerank only REORDERS by relevance, never
# drops). Was 14 — that silently discarded good
# pages before briefing. The crawl is already
# bounded by DR_MAX_PAGES, so briefing them all is
# cheap. Set a positive N via env to cap again.
DR_RERANK_AUTHORITY_WEIGHT = _env_float("DR_RERANK_AUTHORITY_WEIGHT", 0.5)
# Contradiction search (contradiction.py): a SEPARATE pass that actively tries to
# disprove the provisional findings — limitations, failures, negative results,
# reproducibility/benchmark anomalies, issue-tracker complaints. Kept logically
# apart from confirmation search; its evidence is tagged and surfaced, not hidden.
DR_CONTRADICTION_ENABLED = os.getenv("DR_CONTRADICTION_ENABLED", "1") not in ("0", "false", "False")
DR_CONTRADICTION_QUERIES = _env_int("DR_CONTRADICTION_QUERIES", 4)   # contradiction queries/run
DR_CONTRADICTION_PER_QUERY = _env_int("DR_CONTRADICTION_PER_QUERY", 4)
# Entity-driven multi-hop expansion (entities.py): extract named entities from the
# first-pass briefs, build follow-up queries around the salient ones, search again.
# Bounded by depth and a visited-set so it never loops.
DR_MULTIHOP_ENABLED = os.getenv("DR_MULTIHOP_ENABLED", "1") not in ("0", "false", "False")
DR_MULTIHOP_DEPTH = _env_int("DR_MULTIHOP_DEPTH", 1)         # recursion depth (0 = off)
DR_MULTIHOP_ENTITIES = _env_int("DR_MULTIHOP_ENTITIES", 4)  # top entities expanded per hop
DR_MULTIHOP_PER_QUERY = _env_int("DR_MULTIHOP_PER_QUERY", 4)
# Reflection / error-correction: after the first synthesis-ready brief set, find
# gaps (weak claims, unexpanded entities, no contradiction evidence) and decide
# whether a second bounded pass is warranted.
DR_REFLECTION_ENABLED = os.getenv("DR_REFLECTION_ENABLED", "1") not in ("0", "false", "False")
# Abstain / ask / answer gate (decision_gate.py): turns the already-computed
# confidence + coverage + contradiction signals into an explicit decision BEFORE
# synthesis. ANSWER (optionally with explicit uncertainty) | ASK | ABSTAIN.
DR_DECISION_GATE_ENABLED = os.getenv("DR_DECISION_GATE_ENABLED", "1") not in ("0", "false", "False")
DR_GATE_MIN_STRONG = _env_int("DR_GATE_MIN_STRONG", 1)      # strong sources to allow ANSWER
DR_GATE_MIN_CLUSTERS = _env_int("DR_GATE_MIN_CLUSTERS", 2)  # independent clusters for confident ANSWER
# Hierarchical / graph-guided research (clustering.py, graph_guided.py,
# claim_merge.py, hierarchical.py): broad-first → cluster into sub-threads →
# graph-guided deep dives on important/weak/contradicted threads → claim-level
# merge → per-cluster digests → global synthesis. All bounded + visited-aware.
DR_HIERARCHICAL_ENABLED = os.getenv("DR_HIERARCHICAL_ENABLED", "1") not in ("0", "false", "False")
DR_CLUSTER_THRESHOLD = _env_float("DR_CLUSTER_THRESHOLD", 0.18)  # single-link merge sim
DR_CLUSTER_MAJOR_MIN = _env_int("DR_CLUSTER_MAJOR_MIN", 2)   # min size to be a "major" thread
DR_GRAPH_EXPANSION_ENABLED = os.getenv("DR_GRAPH_EXPANSION_ENABLED", "1") not in ("0", "false", "False")
DR_GRAPH_EXPANSION_QUERIES = _env_int("DR_GRAPH_EXPANSION_QUERIES", 6)  # graph-guided follow-ups/run
DR_CLAIM_MERGE_ENABLED = os.getenv("DR_CLAIM_MERGE_ENABLED", "1") not in ("0", "false", "False")
DR_CLAIM_MERGE_THRESHOLD = _env_float("DR_CLAIM_MERGE_THRESHOLD", 0.55)  # claim shingle-Jaccard
DR_CLUSTER_DIGEST_MAX = _env_int("DR_CLUSTER_DIGEST_MAX", 6)  # clusters summarized into synthesis
# Recursive hierarchical retrieval (hierarchy.py): a cluster can spawn child clusters
# down to a bounded depth, so broad topics produce a real tree (root → cluster →
# subcluster → …) instead of a flat two-level split. Deterministic + loop-safe.
DR_HIERARCHY_MAX_DEPTH = _env_int("DR_HIERARCHY_MAX_DEPTH", 4)      # max nesting levels
DR_HIERARCHY_MIN_CLUSTER_SIZE = _env_int("DR_HIERARCHY_MIN_CLUSTER_SIZE", 2)  # don't split below this
DR_HIERARCHY_SPLIT_THRESHOLD = _env_float("DR_HIERARCHY_SPLIT_THRESHOLD", 0.22)  # base child-split sim
# Graph community detection (communities.py): real modularity-optimizing (Louvain)
# communities over the entity/source/claim/contradiction/provenance graph, distinct
# from the lexical brief clusters. Saved + reported.
DR_COMMUNITIES_ENABLED = os.getenv("DR_COMMUNITIES_ENABLED", "1") not in ("0", "false", "False")
# Active reflection loop: reflection now LAUNCHES new retrieval cycles to close the
# gaps it finds (missing/weak/unsupported claims, unresolved contradictions, thin
# clusters), bounded by iterations + a query budget, with loop protection.
DR_REFLECTION_MAX_ITERATIONS = _env_int("DR_REFLECTION_MAX_ITERATIONS", 2)  # extra cycles
DR_REFLECTION_RESEARCH_BUDGET = _env_int("DR_REFLECTION_RESEARCH_BUDGET", 8)  # total extra queries
# Full phase history: every pipeline phase is appended to history/NNN_phase.json
# (timestamp + inputs/outputs/stats) so an entire run is reconstructable from disk.
DR_PHASE_HISTORY_ENABLED = os.getenv("DR_PHASE_HISTORY_ENABLED", "1") not in ("0", "false", "False")
# Phase 9 — research-grade depth
DR_PDF_ENABLED = os.getenv("DR_PDF_ENABLED", "1") not in ("0", "false", "False")
DR_PDF_MAX_PAGES = _env_int("DR_PDF_MAX_PAGES", 30)     # pages parsed per PDF
DR_PDF_MAX_BYTES = _env_int("DR_PDF_MAX_BYTES", 12_000_000)  # cap PDF download size
# A primary paper deserves a bigger brief budget than an ordinary web page so
# methods/derivations aren't truncated to the abstract (model ctx ~25k tokens).
DR_PDF_TEXT_CHARS = _env_int("DR_PDF_TEXT_CHARS", 16000)
DR_CITATIONS_ENABLED = os.getenv("DR_CITATIONS_ENABLED", "1") not in ("0", "false", "False")
DR_CITATION_MAILTO = os.getenv("DR_CITATION_MAILTO", "research@localhost")  # OpenAlex polite pool
# Report token budget scales with depth so research-grade reports aren't truncated.
# NOTE this is think + write, not write alone. A reasoning model that spends
# 15-20k characters (~4-5k tokens) deliberating cannot produce ANY report inside
# 2000 tokens — the call comes back as pure reasoning, the retry ladder fires
# three times, and a "quick" run burned ~25 minutes in "Building report" only to
# emit a stub. One call that fits is far cheaper than three that cannot.
DR_REPORT_TOKENS_QUICK = _env_int("DR_REPORT_TOKENS_QUICK", 6000)
DR_REPORT_TOKENS_STANDARD = _env_int("DR_REPORT_TOKENS_STANDARD", 8000)
DR_REPORT_TOKENS_DEEP = _env_int("DR_REPORT_TOKENS_DEEP", 10000)

# gpt-oss/harmony spends reasoning AND output from ONE max_tokens budget, so at high
# reasoning the model can burn the whole report budget thinking and emit empty content.
# This is dedicated HEADROOM added on top of the report budget so the model can think
# as much as it wants AND still have room to write the full report. 0 disables it.
DR_REASONING_HEADROOM = _env_int("DR_REASONING_HEADROOM", 40000)

# Direct override of the report's OUTPUT token budget (0 = use the per-depth default
# from _resolve_caps). Exposed as a selectable knob in the Research Manual Control.
DR_REPORT_TOKENS_OVERRIDE = _env_int("DR_REPORT_TOKENS_OVERRIDE", 0)

# "Depth of analysis" for the report synthesis (gpt-oss reasoning_effort). "auto" =
# use the model default chosen in Settings; low/medium/high override it for the run.
DR_REASONING_EFFORT = os.getenv("DR_REASONING_EFFORT", "auto")

# SURVEY MODE: synthesize the final report as a coherent, dynamically-outlined
# long-form scientific document (an LLM first designs the section plan from the
# evidence, then each section is written separately as flowing narrative prose),
# instead of one fixed-template single-pass digest. This is what makes the report
# read like a survey paper / monograph and reach tens of pages. 0 = legacy single
# pass (REPORT_SYNTHESIS_PROMPT).
DR_SURVEY_MODE = _env_int("DR_SURVEY_MODE", 1)

# Per-SECTION output budget when writing the survey section-by-section. The whole
# document = N sections x this, so a 10-section plan at 3500 reaches ~tens of pages.
DR_SURVEY_SECTION_TOKENS = _env_int("DR_SURVEY_SECTION_TOKENS", 3500)

# Safety guard on how many sections the dynamic outline may contain — protects only
# against a malformed/runaway plan, NOT a quality cap. Set high so the LLM can design
# a full monograph (many sections) when the topic warrants it. Subsections unbounded.
DR_SURVEY_MAX_SECTIONS = _env_int("DR_SURVEY_MAX_SECTIONS", 30)

# The loaded model's CONTEXT WINDOW (tokens). The whole section call — system
# prompt + outline + evidence + the generated think-block + prose — must fit in
# this. If the model is loaded with a small context (e.g. LM Studio's 4096 default),
# synthesis MUST trim evidence and cap output to fit, or the server rejects the
# prompt ("n_keep >= n_ctx") and returns EMPTY (→ digest fallback). Set this to the
# context you loaded the model with; bigger = more evidence per section + longer
# sections. The pipeline also self-adapts (shrinks on empty), but a correct value
# avoids wasted retries. Default is LM Studio's conservative 4096 — safe everywhere
# (a too-low value just means less evidence per section; a too-HIGH value overflows
# the prompt and returns empty). STRONGLY recommend loading the model with a larger
# context (>=16384) AND setting this to match — short context is the single biggest
# limiter on report depth.
DR_MODEL_CONTEXT = _env_int("DR_MODEL_CONTEXT", 4096)

# Hard per-section token CEILING (think + prose combined). This is NOT a target —
# the model stops on its own (finish_reason=stop) when the section is done; the
# ceiling only exists so a long section is never TRUNCATED mid-sentence. Set it as
# high as the loaded model's context window allows. The effective send budget is
# max(section_target + reasoning headroom, this ceiling), so the model may spend
# tens of thousands of tokens on a single section if it wants to.
DR_SECTION_TOKEN_CEILING = _env_int("DR_SECTION_TOKEN_CEILING", 32000)

# Ceiling for the SHORT synthesis calls — query planner, outline planner, brief
# consolidation. These emit a small JSON plan or a few paragraphs of notes; the
# only reason they need headroom at all is the chain-of-thought that precedes the
# answer (a 900-token cap truncated inside <think> and yielded ""), so this must
# stay comfortably above CoT length. It must NOT be the full context window:
# handing a model 260k tokens of rope (live-detected window minus prompt) means a
# degenerate repetition loop runs until the 600s wall-clock guard kills it, three
# times per report. 8192 covers think + answer with room to spare.
DR_PLAN_TOKENS = _env_int("DR_PLAN_TOKENS", 8192)

# Ceiling for a single per-source brief (map step). DR_BRIEF_TOKENS (320) was too
# tight — it truncated dense extractions mid-bullet and lost evidence — but the fix
# must not be "the whole window": this call runs once PER SOURCE (20-40x a report),
# so an unbounded budget multiplies a single repetition loop by the source count.
# 2048 fits a dense bullet list with verbatim figures and still stops early.
# Per-source brief budget. Must cover the model's THINKING as well as the brief:
# a reasoning model that always deliberates spends the whole allowance thinking
# and returns nothing when the ceiling is low. Measured on
# google/gemma-4-26b-a4b-qat with the same page and prompt:
#   2048 tokens -> 0/3 usable briefs (8.4k chars of reasoning, truncated, empty)
#   6000 tokens -> 2/3 usable briefs
# With 0 briefs the whole run ends in "pages were crawled but none contained
# facts relevant to the topic" — which is what a 15-minute research was returning.
DR_BRIEF_TOKEN_CEILING = _env_int("DR_BRIEF_TOKEN_CEILING", 6000)

# How much of a page is handed to the per-source brief call. This is the single
# biggest driver of research wall-clock, and of whether the step works at all.
# Measured on google/gemma-4-26b-a4b-qat, same page, same prompt, 3 runs each:
#   32000 chars (the old DR_PAGE_CHARS) -> 0/3 usable briefs, median 474s
#    4000 chars                         -> 2/3 usable briefs, median 225s
# A reasoning model deliberates in proportion to what it is given, and this model
# has no working brake (the "<think></think>" prefill is dropped for Gemma), so a
# 32k dump means 17-20k characters of thinking and no answer. Equations are
# extracted from the FULL text at collection time, so capping here costs no math.
DR_BRIEF_INPUT_CHARS = _env_int("DR_BRIEF_INPUT_CHARS", 4000)

# Optional: run the per-source brief step on a DIFFERENT, smaller model. Briefing
# is mechanical extraction, not reasoning, and it is the research bottleneck —
# every source costs one call. Measured, same page/prompt, 3 runs each:
#   gemma-4-26b : 2/3 usable, median 356s, ~260-char briefs
#   gemma-4-12b : 3/3 usable, median 211s, ~536-char briefs
# The 12B is faster, more reliable AND more detailed here.
#
# OFF by default on purpose: keeping a second model resident costs ~7 GB, and on
# this 24 GB card the LLM already shares the GPU with ComfyUI, Whisper and F5-TTS
# — a second model would starve image generation. Set it only if you would rather
# have fast research than concurrent image work, and expect LM Studio to hold
# both models in memory.
DR_BRIEF_MODEL = os.getenv("DR_BRIEF_MODEL", "").strip()

# Manual control — exhaustive collection targets. 0 = no target (off). Default
# DR_MAX_COLLECTION_ROUNDS=1 reproduces the original "one bounded corrective
# re-plan" behavior exactly; raising it (or DR_FORCE_EXHAUSTIVE) makes the
# coverage loop keep broadening until these targets are met, the source cap is
# hit, or a round adds no new sources (stagnation guard — always bounded).
DR_MIN_SOURCES = _env_int("DR_MIN_SOURCES", 0)
DR_TARGET_SOURCES = _env_int("DR_TARGET_SOURCES", 0)
DR_MAX_SOURCES = _env_int("DR_MAX_SOURCES", 0)
DR_MIN_UNIQUE_DOMAINS = _env_int("DR_MIN_UNIQUE_DOMAINS", 0)
DR_MAX_PAGES_PER_DOMAIN = _env_int("DR_MAX_PAGES_PER_DOMAIN", 0)
DR_MAX_COLLECTION_ROUNDS = _env_int("DR_MAX_COLLECTION_ROUNDS", 1)
DR_FORCE_EXHAUSTIVE = os.getenv("DR_FORCE_EXHAUSTIVE", "0") not in ("0", "false", "False")
# Domain filters (comma-separated substrings, matched against the bare host).
DR_DOMAIN_WHITELIST = os.getenv("DR_DOMAIN_WHITELIST", "")
DR_DOMAIN_BLACKLIST = os.getenv("DR_DOMAIN_BLACKLIST", "")
# Source-type inclusion: comma-separated subset of academic/news/government/
# technical/forum/blog/social/whitepaper to EXCLUDE; empty = include everything.
DR_SOURCE_TYPES_EXCLUDE = os.getenv("DR_SOURCE_TYPES_EXCLUDE", "")
# Retry/resilience: when a query returns zero hits, try up to N deterministic
# mutations (strip punctuation, drop trailing word, add a generic qualifier)
# before giving up on it. 0 = off (original behavior).
DR_QUERY_MUTATION_ATTEMPTS = _env_int("DR_QUERY_MUTATION_ATTEMPTS", 0)

# Social media wall reading (posters/afisha live in VK groups & Telegram channels).
# Telegram public channels are read auth-free via t.me/s/<name>. VK blocks scraping,
# so VK wall reading needs a service/access token (https://dev.vk.com) — set VK_TOKEN
# to enable it; without it the crawler falls back to Telegram / the org's own site.
VK_TOKEN = os.getenv("VK_TOKEN", "") or None
VK_API_VERSION = os.getenv("VK_API_VERSION", "5.199")
SOCIAL_POSTS_LIMIT = _env_int("SOCIAL_POSTS_LIMIT", 10)   # recent posts pulled per wall
SOCIAL_OUTLINKS_PER_PAGE = _env_int("SOCIAL_OUTLINKS_PER_PAGE", 2)  # cross-domain social hops/page

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
# When true, `python assistant.py` launches the PyQt desktop app instead of the
# terminal command loop. Override with USE_GUI=0 to force the terminal.
USE_GUI = os.getenv("USE_GUI", "1") not in ("0", "false", "False", "")

# One GPU, several things that want a model on it: a chat turn keeps the LLM
# resident while the Transfer/Storyboard/Madhouse tabs load a diffusion model on
# top of it. With this on, a chat turn started while a tab job is running is
# QUEUED (visible chip in the queue bar) and runs the moment the tab job ends,
# instead of racing it for VRAM. Set GUI_SERIALIZE_GPU_JOBS=0 to allow the
# overlap — reasonable only on a card with headroom for both at once.
GUI_SERIALIZE_GPU_JOBS = os.getenv("GUI_SERIALIZE_GPU_JOBS", "1") \
    not in ("0", "false", "False", "")

# ---------------------------------------------------------------------------
# Session memory
# ---------------------------------------------------------------------------
MEMORY_LIMIT = 50  # max items kept in Context.session_memory
PINNED_FACTS_LIMIT = 40  # max pinned facts (remember_fact tool); oldest dropped first
MEMORY_DIR = BASE_DIR / "memory"
# (session_memory.json / summary.json paths are built per-profile at
# Context.active_memory_dir / "<name>.json" — see models.py load_memory/save_memory
# — so no standalone MEMORY_FILE/MEMORY_SUMMARY_FILE constant is needed here.)

# ---------------------------------------------------------------------------
# Telegram bot — fairness, quotas, limits
# ---------------------------------------------------------------------------
# One GPU pipeline serves every chat, so a single 40-minute deep research would
# otherwise block everyone behind it. The queue is scheduled round-robin BY CHAT
# and each user may only have a small number of tasks waiting at once.
TG_MAX_QUEUED_PER_USER = _env_int("TG_MAX_QUEUED_PER_USER", 3)

# Daily per-user budgets (UTC day). 0 disables the individual limit. Admins are
# always exempt — they are the ones who have to test the expensive paths.
TG_QUOTA_DEEP_RESEARCH = _env_int("TG_QUOTA_DEEP_RESEARCH", 5)
TG_QUOTA_IMAGE         = _env_int("TG_QUOTA_IMAGE", 40)
TG_QUOTA_TASKS         = _env_int("TG_QUOTA_TASKS", 200)

# Rough per-task seconds used only to turn a queue position into a human ETA.
TG_ETA_TASK_SEC     = _env_int("TG_ETA_TASK_SEC", 45)
TG_ETA_RESEARCH_SEC = _env_int("TG_ETA_RESEARCH_SEC", 900)

# Login throttling for the logged-out password gate.
# A self-hosted Bot API server (telegram-bot-api --local) lifts the 20 MB
# download cap to 2 GB and hands files back as local paths. Point TG_API_BASE
# at it (e.g. http://127.0.0.1:8081); the default is Telegram's cloud API.
# With TG_API_ID + TG_API_HASH (from https://my.telegram.org/apps) the app
# starts that server itself (tg_local_api.py) and the base defaults to it.
TG_API_ID   = (os.getenv("TG_API_ID", "") or "").strip()
TG_API_HASH = (os.getenv("TG_API_HASH", "") or "").strip()
TG_API_PORT = _env_int("TG_API_PORT", 8081)
TG_API_EXE  = os.getenv("TG_API_EXE", "C:/tools/telegram-bot-api/bin/telegram-bot-api.exe")
_tg_api_default = ("http://127.0.0.1:%d" % TG_API_PORT
                   if (TG_API_ID and TG_API_HASH) else "https://api.telegram.org")
TG_API_BASE = (os.getenv("TG_API_BASE", "") or _tg_api_default).rstrip("/")
TG_API_LOCAL = TG_API_BASE != "https://api.telegram.org"
TG_LOGIN_MAX_FAILS  = _env_int("TG_LOGIN_MAX_FAILS", 5)
TG_LOGIN_LOCKOUT_S  = _env_int("TG_LOGIN_LOCKOUT_S", 300)
# scrypt work factor for stored passwords (2**TG_SCRYPT_N_LOG2 iterations).
TG_SCRYPT_N_LOG2    = _env_int("TG_SCRYPT_N_LOG2", 14)

# Documents: anything longer than this is INDEXED into the user's library
# instead of being pasted into the prompt and truncated.
TG_DOC_INLINE_CHARS = _env_int("TG_DOC_INLINE_CHARS", 6000)
# Reports longer than this are also delivered as a .md file attachment.
# Report length above which the research result is delivered as a .md ATTACHMENT.
# 0 = always. A research run produces a document, and at 6000 a short report was
# pasted into the chat instead — through the Markdown-to-HTML converter, so the
# user got a wall of bubbles with the Markdown stripped out of it. Chat text is
# now only the fallback for an upload that actually failed.
TG_REPORT_FILE_CHARS = _env_int("TG_REPORT_FILE_CHARS", 0)

# Supervision: how often the watchdog checks that the poll/consumer threads are
# alive, and how long a single task may run before it is force-cancelled.
TG_WATCHDOG_S       = _env_int("TG_WATCHDOG_S", 15)
TG_TASK_MAX_S       = _env_int("TG_TASK_MAX_S", 3600)

# Hard ceiling on a single graph.invoke() call. graph.py/tools.py never look at
# ctx.cancel_event (only deep_research's own progress loop does), so a hung
# LLM/tool call inside invoke() cannot be unstuck cooperatively the way the
# watchdog above unsticks everything else — it would simply block the
# consumer thread that owns it forever, wedging that chat's admission gate
# permanently since _execute_task's finally never runs. graph.invoke() is run
# on its own daemon thread and joined with this timeout; a Python thread
# cannot be killed, so a genuinely hung call still leaks a background thread
# (same tradeoff already accepted for the poll-thread case above), but the
# consumer thread that was serving it is freed immediately and the user gets
# an explicit failure instead of silence.
# Base deadline for one agent turn. Extended automatically while ComfyUI is
# actually rendering (see tg_tasks), capped by TG_TASK_MAX_S -- a picture
# queued behind a ~70-minute music render must not be killed for waiting.
TG_INVOKE_TIMEOUT_S = _env_int("TG_INVOKE_TIMEOUT_S", 1800)

# ---------------------------------------------------------------------------
# Embeddings (knowledge base, document library, dense rerank)
# ---------------------------------------------------------------------------
# BGE-M3 replaced nomic-embed-text-v1.5 because the corpus is bilingual and
# nomic is English-centric: measured on RU/EN pairs of the SAME sentence, nomic
# separated matches from unrelated text by only 0.046 cosine (it scored one
# unrelated pair HIGHER than a correct match), while BGE-M3 separated them by
# 0.617. It is also the same family as the bge-reranker-v2-m3 already used by
# deep research.
#
# EMBED_DIM must match the model. Changing EMBED_MODEL invalidates every stored
# vector — knowledge.ensure_embeddings() re-embeds chunks whose embed_model no
# longer matches, and searches ignore vectors from a different model, so a
# half-migrated index degrades to FTS-only instead of returning nonsense.
# Run  python reindex_embeddings.py  after switching.
# BGE-M3 runs on the CPU-only llama-server (llama_backend.ensure_embedder starts it):
# LM Studio's `--gpu off` still pinned ~0.9 GB of VRAM, and that is what pushed a
# 16-18 GB chat model into shared memory beside F5. Same vectors (cos 0.9999),
# 0.02 s per query; bulk indexing is ~30x slower (16 chunks 6.4 s).
EMBED_BASE  = os.getenv("EMBED_BASE", "http://127.0.0.1:8096")
EMBED_CPU_SERVER = os.getenv("EMBED_CPU_SERVER", r"C:\llamacpp\build\llama-server.exe")
EMBED_GGUF = os.getenv("EMBED_GGUF", os.path.expanduser(
    r"~\.lmstudio\models\ggml-org\bge-m3-Q8_0-GGUF\bge-m3-q8_0.gguf"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-bge-m3")
EMBED_DIM   = _env_int("EMBED_DIM", 1024)

# How long a chat turn waits for a render that was given the whole card
# (comfy_client._gpu_slot exclusive mode) before calling LM Studio anyway. A
# song is minutes, so this is generous on purpose: giving up early would JIT-load
# 20 GB into what the render left and slow BOTH down.
LLM_CARD_WAIT_S = _env_int("LLM_CARD_WAIT_S", 900)
