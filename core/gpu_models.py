"""Park the auxiliary models around exclusive GPU work."""
import logging

log = logging.getLogger('assistant.gpu_models')


def park():
    import llama_backend
    llama_backend.park_embedder()


def restore():
    import llama_backend
    import models
    ctx = models.LIVE_CTX
    if ctx is not None and ctx.models.whisper is not None:
        try:
            with ctx.asr_lock:
                ctx.models.whisper.wake()
        except Exception:
            log.exception('Could not restore Whisper to GPU')
    if ctx is not None:
        try:
            with ctx.tts_lock:
                models.wake_tts(ctx.models)
        except Exception:
            log.exception('Could not restore F5 to GPU')
    llama_backend.resume_embedder()
