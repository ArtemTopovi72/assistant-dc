"""Assemble the agent runtime: models, context, base state, graph.

Named app_runtime rather than runtime because a runtime/ DIRECTORY already
exists in this project and a module of the same name beside it is an import
trap waiting to happen.

Extracted from assistant.py to break the last import cycle. assistant.py is the
ENTRY POINT — it launches the terminal loop or the GUI — so it is entirely
reasonable for it to import gui. What was not reasonable is gui importing back
into assistant for build_runtime and make_base_state, which are not entry-point
concerns at all: they are how the runtime gets built, needed identically by the
terminal loop, the GUI and the Telegram bot.

With this module in the middle, the dependency runs one way:

    assistant  ->  gui  ->  app_runtime

and nothing points back.
"""
import logging
import shutil
import threading

from config import MEMORY_DIR, OUTPUT_DIR
from models import AgentState, Context, Models
from prompts import SYSTEM_PROMPT_PERSONALITY
from graph import build_graph
from utils import load_transcription_cache
from audio import transcribe_audio_file

logger = logging.getLogger("assistant")


def make_base_state() -> AgentState:
    return {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT_PERSONALITY}],
        "user_input": "",
        "image_data": None,
        "final_answer": "",
        "session_memory_text": "",
        "vision_summary": "",
        "image_path": "",
        "image_score": 0,
        "image_attempt": 0,
        "image_status": "",
    }


def _migrate_memory_to_profiles(memory_dir) -> None:
    """One-time migration: move flat session_memory.json/summary.json into 'default/' subdir."""
    old_session = memory_dir / "session_memory.json"
    old_summary = memory_dir / "summary.json"
    default_dir = memory_dir / "default"
    if (old_session.exists() or old_summary.exists()) and not default_dir.exists():
        default_dir.mkdir(parents=True, exist_ok=True)
        for f in (old_session, old_summary):
            if f.exists():
                shutil.move(str(f), str(default_dir / f.name))
        logger.info("Memory migrated to 'default' profile at %s", default_dir)


def _load_llm(model_name: str, base_url: str, _status) -> None:
    """Make the chosen model the one resident model in LM Studio.

    Split out of build_runtime so the "no model" path is a plain if/else rather
    than a conditional wrapped around a try block.
    """
    try:
        from lmstudio import ensure_exclusive
        ok, msg = ensure_exclusive(base_url, model_name)
        if ok:
            _status(f"LLM ready: {model_name}")
        else:
            _status(f"⚠ Could not load LLM ({msg}); LM Studio will JIT-load on first message.")
    except Exception as exc:
        logger.warning("ensure_exclusive failed: %s", exc)


def build_runtime(model_name: str, no_think: bool, status_cb=None,
                  reasoning_effort: str = "high"):
    """Load the local ML models and assemble the agent runtime.

    Shared by both the terminal loop and the GUI. Returns (ctx, base_state, graph).
    Heavy: loads Whisper + F5-TTS, so call it once. ``status_cb`` (optional) is
    called with short progress strings so the GUI loading screen can update.
    """
    from config import CACHE_FILE, LM_STUDIO_BASE
    from utils import cleanup_runtime_artifacts

    def _status(msg: str) -> None:
        logger.info(msg)
        if status_cb is not None:
            try:
                status_cb(msg)
            except Exception:
                pass

    cleanup_runtime_artifacts(OUTPUT_DIR)
    try:
        from config import INPUT_DIR_COMFY, BASE_DIR
        from utils import cleanup_comfy_inputs
        cleanup_comfy_inputs(INPUT_DIR_COMFY, BASE_DIR / "runtime")
    except Exception:
        logger.debug("ComfyUI input cleanup skipped", exc_info=True)
    _status("Loading Whisper + F5-TTS…")
    models = Models.load()
    ctx = Context(
        models=models,
        transcription_cache=load_transcription_cache(CACHE_FILE),
        cache_file=CACHE_FILE,
        asr_lock=threading.Lock(),
        tts_lock=threading.Lock(),
        model_name=model_name,
        no_think=no_think,
        reasoning_effort=reasoning_effort,
    )
    # Seed the opt-in reference-person image mode from its config default (env override).
    try:
        from config import REFERENCE_PERSON_MODE
        ctx.reference_person_mode = bool(REFERENCE_PERSON_MODE)
    except Exception:
        pass
    import f5_tts.infer.utils_infer as infer_utils
    infer_utils.transcribe = lambda path: transcribe_audio_file(ctx, path)

    # Actually load the chosen LLM in LM Studio now (unloading any other model),
    # instead of leaving it to JIT-load on the first chat — which leaves the old
    # model resident as a "phantom" and delays the first reply.
    # An EMPTY model_name is a deliberate choice ("do not load a model"), not a
    # mistake: the user wants the window up with the card left alone -- typically
    # to watch a training run. Loading anything here would take back the VRAM
    # they just freed, so skip the load but still build the graph. A missing
    # graph makes _busy() true forever, which is how "no model" turns into
    # messages that queue silently and are never answered.
    if not (model_name or "").strip():
        _status("Starting without an LLM — the card is left alone")
        logger.info("build_runtime: no model requested; skipping the LM Studio load")
    else:
        _status(f"Loading LLM '{model_name}' in LM Studio…")
        _load_llm(model_name, LM_STUDIO_BASE, _status)
        # One known picture, one question: a broken vision projector (it
        # happened -- see vision_selftest.py) must show up in the log at
        # start-up, not as a week of "the judge says everything is dark".
        try:
            import vision_selftest
            vision_selftest.run_in_background(ctx)
        except Exception:
            logger.exception("vision self-test could not start")

    _migrate_memory_to_profiles(MEMORY_DIR)
    default_dir = MEMORY_DIR / "default"
    default_dir.mkdir(parents=True, exist_ok=True)
    ctx.active_memory_dir = default_dir
    ctx.load_memory(default_dir)
    assistant_graph = build_graph(ctx)
    base_state = make_base_state()
    return ctx, base_state, assistant_graph
