"""Stem separation (Demucs htdemucs) and the stem sums the engine mixes from.

Owns the one heavyweight, cached resource in the engine — the separator model
and its lock — plus MashupUnavailable, the user-facing error every layer
raises. Split out of mashup.py unchanged.
"""
import logging
import os
import threading
import time

import numpy as np

logger = logging.getLogger("assistant.mashup")


STEM_MODEL = "htdemucs"
SAMPLE_RATE = 44100          # htdemucs' native rate; everything resamples to it


# Which stems make up "the instrumental". htdemucs emits four; the backing
# track is simply everything that is not the voice. Summing them is what the
# two-stem models call "no_vocals", so this needs no separate model download.
_BACKING_STEMS = ("drums", "bass", "other")


class MashupUnavailable(RuntimeError):
    """The mashup could not be made -- the message is user-facing."""


# -- stem separation -------------------------------------------------------
_separator = None


_separator_lock = threading.Lock()


def _get_separator(device: str = ""):
    """One cached Separator.

    Loading the model costs ~11s and a mashup needs it twice (once per input),
    so building it per call would double the wait for nothing. Locked because
    both surfaces run this on a background thread, and two concurrent mashups
    would otherwise race to build it.
    """
    global _separator
    with _separator_lock:
        if _separator is None:
            try:
                import torch
                from demucs.api import Separator
            except Exception as exc:
                raise MashupUnavailable(
                    "The stem separator is not installed -- run "
                    "`pip install --no-deps demucs`.") from exc
            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            t0 = time.time()
            try:
                _separator = Separator(model=STEM_MODEL, device=dev)
            except Exception as exc:
                raise MashupUnavailable(
                    "Could not load the stem separator: %s" % exc) from exc
            logger.info("demucs %s ready on %s in %.1fs", STEM_MODEL, dev,
                        time.time() - t0)
        return _separator


def release_separator() -> None:
    """Drop the model and free its VRAM.

    This machine shares 24GB between the LLM, Whisper, F5-TTS and ComfyUI, so a
    separator sitting idle on the GPU is memory the next image or song render
    does not get. Surfaces release it once a mashup is delivered.
    """
    global _separator
    with _separator_lock:
        _separator = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def separate(path: str, device: str = "") -> dict:
    """{stem_name: float32 (samples,) mono} for one audio file, at SAMPLE_RATE.

    Mono because every downstream step (tempo, key, alignment) is a mono
    analysis, and collapsing here means the rest of the pipeline never has to
    care how many channels the input had.
    """
    if not path or not os.path.exists(path):
        raise MashupUnavailable("That audio file is not there any more.")
    sep = _get_separator(device)
    try:
        _origin, stems = sep.separate_audio_file(path)
    except Exception as exc:
        logger.exception("demucs failed on %s", path)
        raise MashupUnavailable("Could not read that audio: %s" % exc) from exc
    out = {}
    for name, tensor in stems.items():
        arr = tensor.detach().cpu().numpy()
        out[name] = (arr.mean(axis=0) if arr.ndim > 1 else arr).astype(np.float32)
    return out


def backing_of(stems: dict) -> np.ndarray:
    """Everything that is not the voice, summed."""
    parts = [stems[k] for k in _BACKING_STEMS if k in stems]
    if not parts:
        raise MashupUnavailable("The separator returned no instrumental stems.")
    n = max(len(p) for p in parts)
    total = np.zeros(n, dtype=np.float32)
    for p in parts:
        total[:len(p)] += p
    return total
