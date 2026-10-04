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
import config as _cfg_env   # env_int/env_float: a bad .env value falls back, never crashes the import
import subprocess
import tempfile
from typing import List, Optional

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.getenv("DIAR_PYTHON", _cfg_env.venv_python(os.path.join(ROOT, "venv_diar")))
MODEL = os.getenv("DIAR_MODEL", os.path.join(ROOT, "models_ext", "nemotron-diar"))
MIN_SECONDS = _cfg_env.env_float("DIAR_MIN_SECONDS", 8)
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
    return name_speakers(ctx, "\n".join(f"{w}: {' '.join(s)}" for w, s in lines), lang)


def name_speakers(ctx, transcript: str, lang: str = "ru") -> str:
    """«Спикер 2» -> «Иван» when the talk itself names them («Иван, подойди»). The LLM proposes,
    a literal check disposes: a name that never occurs in the text is dropped, so nothing is invented."""
    import re
    label = LABEL.get(lang, LABEL["en"])
    ids = sorted(set(re.findall(rf"^{label} (\d+):", transcript or "", re.M)))
    if len(ids) < 2:
        return transcript
    try:
        import llm as _llm
        from utils import safe_json_from_llm
        raw = _llm.call_llm_simple(
            ctx, 'Name the speakers of a transcript. Answer ONLY JSON {"1": "name or empty", ...}. '
                 "A name counts ONLY if it is spoken in the text itself (someone addresses or introduces them); "
                 "otherwise empty. Never guess.", transcript[:6000],
            temperature=0.0, max_tokens=300, force_think=False) or ""
        got = safe_json_from_llm(raw) or {}
    except Exception:
        return transcript
    low = transcript.lower()
    for i in ids:
        name = str(got.get(i) or "").strip()
        if 2 <= len(name) <= 30 and re.fullmatch(r"[\w .-]+", name) and name[: max(3, len(name) - 2)].lower() in low:
            transcript = re.sub(rf"^{label} {i}:", f"{name} ({label} {i}):", transcript, flags=re.M)
    return transcript
