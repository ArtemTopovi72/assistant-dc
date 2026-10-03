"""Training LoRA adapters for characters, and being honest about whether it can.

The GUI needs to answer three questions without lying: is a trainer installed,
what will it train on, and how far along is it. Everything here exists to keep
those answers true.

Two hard constraints, both learned the expensive way:

  * THE TRAINER GETS ITS OWN VENV. This project's ./venv pins
    torch 2.8.0+cu128; pip installing a trainer into it has already once
    swapped the CUDA build for a CPU one and taken image generation down with
    it. So the toolkit lives elsewhere and is invoked by ITS OWN python.
  * ONE JOB PER GPU. A 24 GB 3090 cannot train while LM Studio holds a model
    and ComfyUI holds the old model. free_gpu() unloads both, and the caller refuses
    to start if the card is still occupied -- an OOM forty minutes in costs
    more than a refusal now.

Nothing here imports torch, so the GUI can ask about status on a machine where
the trainer was never installed.
"""
from __future__ import annotations

import json
import os
import config as _cfg_env   # env_int/env_float: a bad .env value falls back, never crashes the import
import logging
import re
import time
import subprocess
import urllib.request
from pathlib import Path

logger = logging.getLogger("assistant.lora")

_ROOT = Path(__file__).resolve().parents[1]
# Where adapters must end up to be loadable: ComfyUI's own loras folder.
LORA_OUT_DIR = Path(os.getenv(
    "COMFY_LORA_DIR", os.path.expanduser("~/Documents/ComfyUI/models/loras")))
DATASET_ROOT = _ROOT / "runtime" / "lora_datasets"
TRAIN_ROOT = _ROOT / "runtime" / "lora_training"
# Ostris AI Toolkit, cloned and installed by the operator into its own venv.
TOOLKIT_DIR = Path(os.getenv("LORA_TOOLKIT_DIR", str(_ROOT / "tools" / "ai-toolkit")))

# The 24 GB profile: 1024 px, rank 16, batch 1, gradient checkpointing. The
# model-specific keys (model_path, arch, sampling, prompts) come from
# IDEOGRAM_PROFILE below -- Ideogram 4 is the only drawing engine, so it is the
# only thing a character LoRA can be trained for. (the older drawing
# model it replaced was removed from the product.)
# quantize/low_vram default ON, and they should STAY on even when the card is
# exclusive. The reasoning that says otherwise -- fp8 has no tensor-core path
# on Ampere, low_vram adds offload traffic an evicted card does not need -- is
# sound and was still wrong here. Measured on this 3090:
#   quantize=False, low_vram=False -> 15.2 s/it and 24199/24576 MiB, against
#   a 5.99 s/it baseline. bf16 weights are twice the bytes of fp8 and this
#   workload is memory-bandwidth bound, not compute bound, so the emulation
#   cost is far cheaper than the extra traffic.
# grad_ckpt=False was tried separately and was worse again: 2:10 for the first
# step alone and VRAM at 24147/24576, about to OOM.
_BASE_DEFAULTS = {"steps": 2000, "rank": 16, "lr": 1e-4, "size": 1024, "save_every": 250,
                  "quantize": "true", "low_vram": "true", "grad_ckpt": "true"}


def _json_sample_prompts(specs) -> str:
    """Ideogram sample prompts as YAML list items, brace-escaped for .format().

    Built with ideogram_layout.build_caption -- the project's own caption
    builder, which already enforces key order, the photo/art_style exclusivity
    and compact serialisation, and has a test suite behind it. Hand-writing the
    JSON here would be a second, unverified implementation of the same schema.

    Two traps stacked on top of each other.

    First, the prompts must be STRUCTURED. Ideogram's documentation says a
    plain-text prompt bypasses the schema and will likely trigger a safety
    warning -- and it does. Inheriting the old model's flat prompts made every
    training preview come back as the model's grey refusal card, at step 0 and
    at every save. Training itself was unaffected, since that reads the
    dataset's JSON captions; only the previews were destroyed, which is the one
    thing they exist for.

    Second, the config template is rendered with str.format, so every brace in
    the JSON would be read as a substitution field. They are doubled here --
    except {trigger}, which is the one field we DO want substituted.
    """
    import json

    import ideogram_layout as IL

    lines = []
    for hl, lighting, background, desc in specs:
        caption = IL.build_caption(
            background, [IL.element(desc)],
            high_level=hl,
            aesthetics="photorealistic, natural skin texture",
            lighting=lighting,
            photo="digital photograph, portrait lens",
            medium="photograph")
        blob = json.dumps(caption, separators=(",", ":"), ensure_ascii=False)
        if "'" in blob:
            raise ValueError("a single quote would break the YAML scalar: " + blob)
        blob = blob.replace("{", "{{").replace("}", "}}")
        blob = blob.replace("{{trigger}}", "{trigger}")
        lines.append("          - '%s'" % blob)
    return "\n".join(lines)


def _qwen3_vl_local() -> str:
    """The downloaded Qwen3-VL-8B-Instruct snapshot dir, or the hub id.

    Falls back to the hub id rather than failing here: the trainer will then
    try to download it, which is the right error to see and not a silent
    misconfiguration.
    """
    root = Path("E:/hf-cache") / "models--Qwen--Qwen3-VL-8B-Instruct" / "snapshots"
    if root.is_dir():
        snaps = [p for p in root.iterdir() if (p / "config.json").is_file()]
        if snaps:
            return max(snaps, key=lambda p: p.stat().st_mtime).as_posix()
    return "Qwen/Qwen3-VL-8B-Instruct"


_QWEN3_VL_LOCAL = _qwen3_vl_local()


# Ideogram 4. Its HF repo is GATED, so name_or_path points at the local dir
# bench/build_ideogram4_local.py assembles from the ComfyUI weights.
#
# The text encoder is a SEPARATE download on purpose: ideogram4.py loads stock
# bf16 Qwen3-VL-8B-Instruct rather than the fp8 copy shipped in the model repo
# ("faster and higher precision than dequantizing"), and pointing it at a hub
# id would send it to the default cache on C:, which has no room.
#
# Sampling follows Ideogram's own V4_DEFAULT_20 preset: 20 steps at guidance
# weight 7.
IDEOGRAM_PROFILE = {
    "model_path": r"E:/ideogram4-local",
    "arch": "ideogram4",
    "assistant": "",
    "model_kwargs": ("        model_kwargs:\n"
                      "          text_encoder_path: %s\n"
                      % _QWEN3_VL_LOCAL),
    "prompts": _json_sample_prompts([
        ("A photograph of {trigger}, a man, standing outdoors in daylight.",
         "natural daylight", "an outdoor setting",
         "{trigger}, a man, standing, waist up"),
        ("A studio portrait of {trigger}, a man.",
         "soft studio lighting", "a plain studio backdrop",
         "{trigger}, a man, head and shoulders"),
    ]),
    "sample_steps": 20,
    "guidance": 7.0,
}


DEFAULTS = {**_BASE_DEFAULTS, **IDEOGRAM_PROFILE}


def toolkit_python():
    """The trainer's OWN interpreter, or None. Never this process's executable."""
    for rel in ("venv/Scripts/python.exe", "venv/bin/python", ".venv/Scripts/python.exe",
                ".venv/bin/python"):
        p = TOOLKIT_DIR / rel
        if p.is_file():
            return p
    return None


def toolkit_status() -> dict:
    """What is and is not ready, in terms a user can act on."""
    py = toolkit_python()
    runner = TOOLKIT_DIR / "run.py"
    missing = []
    if not TOOLKIT_DIR.is_dir():
        missing.append("папка тренера не найдена: " + str(TOOLKIT_DIR))
    elif not runner.is_file():
        missing.append("нет run.py в " + str(TOOLKIT_DIR))
    if TOOLKIT_DIR.is_dir() and py is None:
        missing.append("у тренера нет своего venv (в общий ставить нельзя: "
                       "он собьёт torch 2.8.0+cu128)")
    if not LORA_OUT_DIR.is_dir():
        missing.append("нет папки loras ComfyUI: " + str(LORA_OUT_DIR))
    return {"ready": not missing, "dir": str(TOOLKIT_DIR),
            "python": str(py) if py else "", "missing": missing}


def gpu_free_mb():
    """Free VRAM in MiB, or None if nvidia-smi is not answering."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def free_gpu(log=None, min_free_mb: int = None) -> None:
    """Evict LM Studio and ComfyUI so training (or a render) has the card
    to itself.

    Best-effort by design: a machine where neither is running must not be
    blocked from training just because the eviction call had nobody to talk to.

    `min_free_mb` overrides FREE_GPU_MIN_MB for callers whose model needs more
    headroom than "the chat model has definitely drained" -- see its caller in
    comfy_client._claim_card for why H3 video passes a higher number.
    """
    def say(m):
        if log:
            log(m)
    try:
        subprocess.run(["lms", "unload", "--all"], capture_output=True,
                       text=True, timeout=60, shell=(os.name == "nt"))
        say("LM Studio: выгружено")
    except Exception as exc:
        say("LM Studio не отвечает (%s) — продолжаю" % exc)
    try:
        import llama_backend
        if llama_backend.stop():
            say("llama-server: остановлен")
    except Exception as exc:
        say("llama-server не остановлен (%s) — продолжаю" % exc)
    try:
        url = os.getenv("COMFY_URL", "http://127.0.0.1:8000").rstrip("/") + "/free"
        req = urllib.request.Request(
            url, data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()
        say("ComfyUI: выгружено")
    except Exception as exc:
        say("ComfyUI не отвечает (%s) — продолжаю" % exc)
    # `lms unload` returns before the driver has the memory back. A render
    # started 0.2 s later loaded its weights beside the not-yet-released chat
    # model and ran 296 s instead of 48 s (live, 2026-09-12, journey 21: the
    # 6x tax of sharing the card). Wait for the free counter to say so.
    freed = wait_vram_free(min_free_mb or FREE_GPU_MIN_MB, timeout=FREE_GPU_WAIT_S)
    if freed is not None:
        say("VRAM свободно: %d MiB" % freed)


# What "the card is free" means after an eviction: the 26B chat model alone
# is ~17 GB, so anything under this is a model still draining.
FREE_GPU_MIN_MB = _cfg_env.env_int("FREE_GPU_MIN_MB", 16000)
FREE_GPU_WAIT_S = _cfg_env.env_float("FREE_GPU_WAIT_S", 25)


def wait_vram_free(min_free_mb: int, timeout: float = 25.0, poll: float = 0.5):
    """Poll nvidia-smi until at least `min_free_mb` MiB is free or `timeout`
    passes. Returns the last free reading, or None when nvidia-smi is silent
    (a machine without it must not be held up)."""
    deadline = time.monotonic() + timeout
    last = gpu_free_mb()
    while last is not None and last < min_free_mb and time.monotonic() < deadline:
        time.sleep(poll)
        last = gpu_free_mb()
    if last is not None and last < min_free_mb:
        logging.getLogger("assistant.lora").warning(
            "VRAM still only %d MiB free after %.0fs — rendering anyway", last, timeout)
    return last


def _lms_base() -> str:
    """The server root, WITHOUT /v1: lmstudio.py appends /api/v0/... to it.
    The old default here was ".../v1", so every give-back after a render polled
    /v1/api/v0/models (LM Studio: "Unexpected endpoint", once a second), never saw
    the model, and reported the restore as failed."""
    import config
    return config.LM_STUDIO_BASE.rstrip("/").removesuffix("/v1")


def loaded_llm_id() -> str:
    """Which chat model LM Studio is serving right now, or "".

    Captured BEFORE the eviction so the same one can be brought back after
    training. Asking afterwards is too late -- by then nothing is loaded and
    the app would come back with whatever the config default happens to be.
    """
    try:
        with urllib.request.urlopen(_lms_base() + "/v1/models", timeout=10) as r:
            ids = [m["id"] for m in json.loads(r.read())["data"]]
    except Exception:
        return ""
    # Embedding models are routinely resident alongside the chat model; the
    # one worth restoring is the chat model, and those are not it.
    chat = [i for i in ids if "embed" not in i.lower() and "rerank" not in i.lower()]
    return chat[0] if chat else ""


def reload_llm(model_id: str, log=None) -> bool:
    """Bring the chat model back after a training run. Best effort."""
    if not model_id:
        return False
    try:
        from lmstudio import ensure_exclusive
        ok, msg = ensure_exclusive(_lms_base(), model_id)
    except Exception as exc:
        ok, msg = False, str(exc)
    if log:
        log(("модель вернулась: %s" % model_id) if ok
            else ("не удалось вернуть модель %s (%s)" % (model_id, msg)))
    return bool(ok)


def dataset_size(folder) -> int:
    d = Path(folder)
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


_YAML = """job: extension
config:
  name: {slug}
  process:
    - type: sd_trainer
      training_folder: {out}
      device: cuda:0
      trigger_word: "{trigger}"
      network:
        type: lora
        linear: {rank}
        linear_alpha: {rank}
      save:
        dtype: float16
        save_every: {save_every}
        # Keep them ALL. At 8 the early saves are rotated away as later ones
        # land, and on the first neurostepan run that destroyed 250-1750
        # before they could be compared -- the checkpoint that turned out to
        # generalise best was the EARLIEST surviving one, so the ones it ate
        # were exactly the ones worth looking at.
        max_step_saves_to_keep: 40
      datasets:
        - folder_path: {data}
          caption_ext: txt
          shuffle_tokens: false
          cache_latents_to_disk: true
          cache_text_embeddings: true
          resolution: [{size}]
      train:
        batch_size: 1
        steps: {steps}
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: {grad_ckpt}
        # captions are static per image (no shuffle_tokens, no dropout), so the
        # text-encoder output for each one never changes across 9000 steps --
        # caching it once and dropping the encoder from VRAM is a free win,
        # not a quality/speed tradeoff like quantize or grad_ckpt were.
        unload_text_encoder: true
        noise_scheduler: flowmatch
        # "weighted", not the toolkit's "sigmoid" default: which noise levels
        # get sampled during training decides what the adapter gets to learn,
        # and the flow-matching guides call for weighted here, where
        # an object's structure forms. This is the one setting our config
        # actually disagreed with them on.
        timestep_type: weighted
        optimizer: adamw8bit
        lr: {lr}
        dtype: bf16
      model:
        name_or_path: {model_path}
        # "ideogram4", not "ideogram_4": the arch key is <Model>.arch in the toolkit,
        # and a name that merely looks right fails at load time, after the
        # dataset has already been cached.
        arch: {arch}
        quantize: {quantize}
        low_vram: {low_vram}
{assistant}{model_kwargs}
      sample:
        sampler: flowmatch
        sample_every: {save_every}
        width: {size}
        height: {size}
        # Validate in the regime the adapter will actually be USED in. For
        # Turbo that is 8 steps at guidance ~1; a sample that looks right at
        # 30 steps and collapses at 8 has not met the objective. Base is the
        # opposite -- it needs the long trajectory and real CFG.
        sample_steps: {sample_steps}
        guidance_scale: {guidance}
        prompts:
{prompts}
meta:
  name: {slug}
"""


def build_config(slug: str, trigger: str, dataset_dir, **over) -> Path:
    """Write the AI Toolkit YAML for one character and return its path."""
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in over.items() if v is not None})
    # The sample prompts need a format pass of their OWN. _YAML.format inserts
    # each value verbatim and never looks inside it, so a {trigger} sitting in
    # the prompts string would reach the config as the literal text
    # "{trigger}" and every preview would be of a word rather than a person.
    # This is also why _json_sample_prompts doubles the braces of its JSON:
    # they pass through this pass, not the template one.
    cfg["prompts"] = str(cfg["prompts"]).format(trigger=trigger)
    run_dir = TRAIN_ROOT / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    text = _YAML.format(slug=slug, trigger=trigger,
                        data=Path(dataset_dir).as_posix(),
                        out=run_dir.as_posix(), **cfg)
    path = run_dir / (slug + ".yaml")
    path.write_text(text, encoding="utf-8")
    return path


def log_path(slug: str) -> Path:
    """Where a run's output is teed, whoever started it.

    One canonical path so the tab can follow a run it did not launch -- a
    training started from a terminal was otherwise invisible in the app, which
    is how a user ends up staring at a tab that says nothing while the GPU is
    plainly busy.
    """
    return TRAIN_ROOT / slug / "train.log"


_STEP_IN_NAME = re.compile(r"(\d{4,})\.safetensors$", re.I)


def is_training(slug: str) -> bool:
    """Is a trainer process for this character actually alive?

    Log freshness alone is not enough, and the difference is not academic: the
    first phase of a cold run downloads a 12 GB base model and writes nothing
    to the log for many minutes, during which a freshness check calls a
    perfectly healthy run dead. So look for the process, and fall back to
    freshness only when we cannot.
    """
    # Match on the config FILE NAME, not its absolute path: the command line
    # carries whatever the caller typed -- a relative path with forward
    # slashes, most often -- so comparing against a resolved path silently
    # never matches and every live run looks dead.
    cfg_name = (slug + ".yaml").lower()
    try:
        import psutil
    except Exception:
        log = log_path(slug)
        return log.is_file() and (time.time() - log.stat().st_mtime) < 180
    # Match ARGUMENTS, not substrings of the whole command line. Searching the
    # joined line matches any process that merely MENTIONS these names --
    # including the shell command doing the checking, and every lingering
    # shell that once ran `grep neurostepan.yaml`. That is not theoretical:
    # it blocked two consecutive relaunches here with no trainer alive at all,
    # and the Персонажи tab reads the same function to decide whether to show
    # a run as live.
    for proc in psutil.process_iter(["cmdline", "name"]):
        try:
            argv = proc.info.get("cmdline") or []
            name = (proc.info.get("name") or "").lower()
        except Exception:
            continue
        if not argv:
            continue
        # "Is this a python process" is a NARROWING check, never a rejection on
        # ignorance. psutil does not always hand back a name (permissions, a
        # race with exit, a stubbed iterator), and rejecting on a missing name
        # reported a live trainer as dead -- the exact failure c080de0 fixed,
        # where a cold run downloading a 12 GB base model vanished from the tab.
        # When the name is unreadable, the argv match below carries the weight;
        # it is specific enough on its own.
        looks_python = "python" in (name or str(argv[0]).lower())
        if name and not looks_python:
            continue
        args = [str(a).lower().replace("\\", "/") for a in argv]
        if any(a.endswith("/run.py") or a == "run.py" for a in args[1:]) \
                and any(a.endswith("/" + cfg_name) or a == cfg_name for a in args[1:]):
            return True
    return False


# "1466/2500 [3:30:04<2:28:19" -- the trainer's tqdm bar.
_STEP_IN_BAR = re.compile(r"(\d+)/(\d+)\s*\[")
# "[1:52:05<17:35:31,  8.78s/it" -- elapsed, remaining, rate. Kept separate
# from _STEP_IN_BAR: an interrupted or just-started run writes a bar with no
# timing at all ("? it/s"), and one regex for both would drop the step too.
_BAR_TIMING = re.compile(r"\[([\d:]+)<([\d:?]+),\s*([\d.]+\s*(?:s/it|it/s))")


def run_slugs(slug: str) -> list:
    """Every run folder belonging to this character, newest activity first.

    One character can have several runs: the old-model one under its bare slug,
    and the Ideogram ones under `<slug>_ideo`, `<slug>_ideo32` and so on. The
    tab knows only the bare slug, so without this it reported on whichever run
    happened to be named exactly that -- which is how a live rank-32 run showed
    as "не активно" while the numbers came from an old model run that finished
    hours earlier.
    """
    if not TRAIN_ROOT.is_dir():
        return []
    dirs = [d for d in TRAIN_ROOT.iterdir()
            if d.is_dir() and (d.name == slug or d.name.startswith(slug + "_"))]
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [d.name for d in dirs]


def configured_steps(slug: str) -> int:
    """The step count the RUN was configured with, not the one on the form.

    Reading the spin box instead is what produced "шаг 2250 из 2000": the run
    had been extended past the number still sitting in the widget.
    """
    cfg = TRAIN_ROOT / slug / ("%s.yaml" % slug)
    if not cfg.is_file():
        return 0
    try:
        for line in cfg.read_text(encoding="utf-8").splitlines():
            key, _, val = line.strip().partition(":")
            if key == "steps" and val.strip().isdigit():
                return int(val.strip())
    except OSError:
        pass
    return 0


RUN_RUNNING, RUN_DONE, RUN_STOPPED, RUN_NEVER = (
    "running", "done", "stopped", "never")


def run_state(p: dict) -> str:
    """Which of the four endings a progress() dict describes.

    "not running" covered two very different ones: a run that reached its
    configured step count FINISHED, while one that stopped at 1147 of 8000 was
    killed or crashed -- and the difference decides whether the next move is to
    publish a checkpoint or to restart. The trainer writes no verdict, so it is
    read off the numbers, in one place, because the desktop tab and the
    Telegram admin panel both report it.
    """
    if p.get("running"):
        return RUN_RUNNING
    step = int(p.get("step") or 0)
    total = int(p.get("total") or 0)
    if not step:
        return RUN_NEVER
    if total and step >= total:
        return RUN_DONE
    # With no recorded total this cannot be told apart from a finished run.
    # It is reported as stopped because that is the answer that makes someone
    # LOOK; callers that would print the total must check it themselves.
    return RUN_STOPPED


def human_eta(raw: str) -> str:
    """"17:23:40" -> "17ч 23м".

    The trainer prints hours:minutes:seconds; nobody waiting on a day-long run
    reads the seconds, and the raw form is easy to misread as a clock time.
    Lives here rather than in the tab because the Telegram admin panel shows
    the same figure, and two formatters drift.
    """
    parts = [p for p in str(raw).split(":") if p.isdigit()]
    if len(parts) == 3:
        h, m, _ = (int(x) for x in parts)
        return ("%dч %02dм" % (h, m)) if h else ("%dм" % m)
    if len(parts) == 2:
        m, sec = (int(x) for x in parts)
        return ("%dм" % m) if m else ("%dс" % sec)
    return str(raw)


def progress(slug: str) -> dict:
    """What a run has actually produced, read off the disk.

    Deliberately NOT read from a live process handle: the source of truth is
    the files, so the same numbers show whether the run was started from the
    tab, from a terminal, or in a previous session.
    """
    # Resolve to the run this character is ACTUALLY using: a live one if there
    # is one, otherwise the most recently touched.
    variants = run_slugs(slug) or [slug]
    slug = next((v for v in variants if is_training(v)), variants[0])
    run_dir = TRAIN_ROOT / slug
    ckpts = sorted(run_dir.rglob("*.safetensors"), key=lambda p: p.stat().st_mtime)
    samples = sorted(run_dir.rglob("*.jpg")) + sorted(run_dir.rglob("*.png"))
    step = 0
    eta = rate = ""
    for p in ckpts:
        m = _STEP_IN_NAME.search(p.name)
        if m:
            step = max(step, int(m.group(1)))
    log = log_path(slug)
    tail = ""
    live_step = live_total = 0
    if log.is_file():
        try:
            text = log.read_text(encoding="utf-8", errors="replace")[-4000:]
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            tail = lines[-1][-300:] if lines else ""
            # The trainer's own progress bar is the only place the CURRENT step
            # appears; checkpoints land every 250, so between them the step read
            # off the filenames is stale by up to 250 and reads 0 for the first
            # quarter-hour of a run. The bar is carriage-return separated, so
            # the tail is scanned rather than just the last newline.
            flat = text.replace(chr(13), chr(10))
            m = None
            for m in _STEP_IN_BAR.finditer(flat):
                pass
            if m:
                live_step, live_total = int(m.group(1)), int(m.group(2))
            # The bar also carries what the user actually wants to know -- how
            # long is left, and how fast it is going. Reading it back is honest
            # in a way a locally computed estimate is not: it is the trainer's
            # own accounting of its own steps, and it survives a GUI restart.
            b = None
            for b in _BAR_TIMING.finditer(flat):
                pass
            if b:
                eta, rate = b.group(2), b.group(3).strip()
        except OSError:
            pass
    return {"slug": slug, "total": live_total or configured_steps(slug),
            "step": max(step, live_step),
            "checkpoints": len(ckpts), "samples": len(samples),
            "last_sample": str(samples[-1]) if samples else "",
            "tail": tail, "eta": eta, "rate": rate,
            "running": is_training(slug)}


def latest_adapter(slug: str):
    """The newest .safetensors this character's run produced.

    Checkpoints matter: on this model the 1500-1750 step saves are often a
    better likeness than the final one, so the run keeps eight of them and the
    caller picks. This returns the newest, not the best.
    """
    run_dir = TRAIN_ROOT / slug
    if not run_dir.is_dir():
        return None
    cands = sorted(run_dir.rglob("*.safetensors"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def list_checkpoints(slug: str, limit: int = 10) -> list:
    """The run's saved checkpoints, newest first, as (step, path).

    The trainer names intermediate saves `<slug>_000001750.safetensors` and the
    final one plain `<slug>.safetensors`, so the step has to be read from the
    name rather than from mtime -- mtime ordering breaks the moment a file is
    copied or restored, and the whole point of this list is to let a HUMAN pick
    a step by number after comparing renders.

    The final save is reported under the configured step count, because that is
    what it actually is; without it, the last save would be the only one the
    picker could not name.
    """
    run_dir = TRAIN_ROOT / slug / slug
    if not run_dir.is_dir():
        run_dir = TRAIN_ROOT / slug
    if not run_dir.is_dir():
        return []
    out = []
    for p in run_dir.glob("%s*.safetensors" % slug):
        stem = p.stem
        tail = stem[len(slug):].lstrip("_")
        if not tail:
            out.append((None, p))          # the final save; numbered below
        elif tail.isdigit():
            out.append((int(tail), p))
    numbered = sorted((s for s, _ in out if s is not None))
    final_step = (numbered[-1] + (numbered[-1] - numbered[-2])
                  if len(numbered) >= 2 else None)
    rows = [((final_step if s is None else s), p) for s, p in out]
    # A final save we cannot number sorts last rather than crashing the sort.
    rows = [(s if s is not None else 0, p) for s, p in rows]
    rows.sort(key=lambda r: r[0], reverse=True)
    return rows[:limit] if limit else rows


def publish(slug: str, adapter) -> Path:
    """Copy a trained adapter into ComfyUI's loras folder under a stable name."""
    import shutil
    LORA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = LORA_OUT_DIR / (slug + ".safetensors")
    shutil.copy2(str(adapter), str(dest))
    return dest


def train_command(cfg_path) -> list:
    py = toolkit_python()
    if py is None:
        raise RuntimeError("тренер не установлен: "
                           + "; ".join(toolkit_status()["missing"]))
    return [str(py), str(TOOLKIT_DIR / "run.py"), str(cfg_path)]


# A run keeps `max_step_saves_to_keep` checkpoints plus the final one, and
# caches latents and text embeddings beside the dataset. Measured on the
# Ideogram runs: about 1.4 GB for a 3250-step rank-16 run. Two of those, plus
# the headroom Windows itself wants, is the floor below which a run is not
# worth starting.
MIN_FREE_GB = 6.0


def free_gb(path=None) -> float:
    """Free space on the drive the run will write to, in GB (0 if unknown)."""
    import shutil as _sh
    p = Path(path or TRAIN_ROOT)
    # Before the first run the folder does not exist yet; its drive does.
    while not p.exists() and p.parent != p:
        p = p.parent
    try:
        return _sh.disk_usage(str(p)).free / 1024.0 ** 3
    except Exception:
        return 0.0


def disk_warning(path=None) -> str:
    """A sentence to show before starting a run, or "" when there is room.

    A trainer that fills the disk does not fail loudly: it dies somewhere
    inside a checkpoint write, hours in, and what is left behind is a partial
    .safetensors that the resume path will happily pick up as the newest save.
    Checking first costs one stat call.
    """
    free = free_gb(path)
    if not free or free >= MIN_FREE_GB:
        return ""
    return ("На диске осталось %.1f ГБ — обучение пишет чекпоинты и кэш "
            "латентов и может упасть на середине. Нужно хотя бы %.0f ГБ."
            % (free, MIN_FREE_GB))


def spawn(cfg_path):
    """Start the trainer as a child process with line-buffered output."""
    warn = disk_warning()
    if warn:
        # Logged, not raised: the caller decides whether to stop. A refusal
        # here would be a new way for a deliberate run to fail silently.
        logger.warning("starting a training run with low disk: %s", warn)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # hf_xet's "reconstructing file" stage hangs indefinitely on this machine:
    # the 170 MB assistant adapter downloaded in full, then sat at 0% for six
    # minutes with the GPU idle and never finalised out of .incomplete. The
    # classic HTTP downloader fetches the same file in seconds.
    env["HF_HUB_DISABLE_XET"] = "1"
    return subprocess.Popen(
        train_command(cfg_path), cwd=str(TOOLKIT_DIR), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1)
