"""Who spoke when, for forwarded voice notes and videos with several people.

Nemotron-3-Diarization runs in its own venv (it needs transformers master;
the app's venv is pinned) as a subprocess on the CPU -- a ~100M model, no GPU
needed, nothing to evict. GigaAM then reads each speaker turn, and the
transcript comes back as "Спикер 1: ... / Спикер 2: ...".

Every failure returns None and the caller keeps its plain transcript: a
missing venv, a single speaker, a clip too short to be a conversation.
"""
import json
import logging
import os
import subprocess
import tempfile
from typing import List, Optional

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.getenv("DIAR_PYTHON", os.path.join(ROOT, "venv_diar", "Scripts", "python.exe"))
MODEL = os.getenv("DIAR_MODEL", os.path.join(ROOT, "models_ext", "nemotron-diar"))
MIN_SECONDS = float(os.getenv("DIAR_MIN_SECONDS", "8"))
MIN_TURN = 0.6            # a "turn" shorter than this is a cough, merged away
LABEL = {"ru": "Спикер", "en": "Speaker"}


def enabled() -> bool:
    return os.getenv("DIARIZE", "1") != "0" and os.path.exists(PY) and os.path.isdir(MODEL)


def segments(wav: str, timeout: int = 300) -> Optional[List[dict]]:
    try:
        p = subprocess.run([PY, os.path.join(ROOT, "scripts", "diar_worker.py"), wav, MODEL],
                           capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        line = (p.stdout or "").strip().splitlines()[-1:] or [""]
        return json.loads(line[0]) if p.returncode == 0 else None
    except Exception:
        logger.warning("diarization worker failed", exc_info=True)
        return None


def turns(segs: List[dict]) -> List[dict]:
    """Sorted, blips dropped, consecutive same-speaker segments merged."""
    out: List[dict] = []
    for s in sorted(segs, key=lambda s: s["start"]):
        if s["end"] - s["start"] < MIN_TURN:
            continue
        if out and out[-1]["speaker"] == s["speaker"] and s["start"] - out[-1]["end"] < 1.5:
            out[-1]["end"] = max(out[-1]["end"], s["end"])
        else:
            out.append(dict(s))
    return out


def label_order(ts: List[dict]) -> dict:
    """Model speaker ids -> 1, 2, 3 in order of first appearance."""
    order: dict = {}
    for t in ts:
        order.setdefault(t["speaker"], len(order) + 1)
    return order


def speaker_transcript(ctx, wav: str, lang: str = "ru", asr_lang: str = "") -> Optional[str]:
    """Labelled transcript, or None when this is not a multi-speaker clip."""
    if not enabled():
        return None
    import soundfile as sf
    try:
        x, sr = sf.read(wav, dtype="float32", always_2d=False)
    except Exception:
        return None
    if getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=1)
    if len(x) / sr < MIN_SECONDS:
        return None
    segs = segments(wav)
    ts = turns(segs or [])
    names = label_order(ts)
    if len(names) < 2:
        return None
    from audio import transcribe_audio_file
    label = LABEL.get(lang, LABEL["en"])
    lines = []
    for t in ts:
        a, b = int(max(0.0, t["start"] - 0.15) * sr), int((t["end"] + 0.15) * sr)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            tmp = fh.name
        try:
            sf.write(tmp, x[a:b], sr)
            said = (transcribe_audio_file(ctx, tmp, lang_hint=asr_lang or lang) or "").strip()
        finally:
            os.unlink(tmp)          # our own temp file
        if said:
            who = f"{label} {names[t['speaker']]}"
            if lines and lines[-1][0] == who:
                lines[-1][1].append(said)
            else:
                lines.append((who, [said]))
    if len({w for w, _ in lines}) < 2:
        return None
    return "\n".join(f"{w}: {' '.join(s)}" for w, s in lines)
