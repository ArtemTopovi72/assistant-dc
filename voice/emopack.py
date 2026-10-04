"""Emotion pack for a cloned voice: the same timbre carrying real Russian emotional delivery.

Why: F5 takes its delivery from the reference clip. A neutral reference gives a neutral voice whatever the text; the
measured fix (docs/f5_artifacts_emotions_2026-10.md) is an emotional reference IN THE SAME VOICE. We have one clip of the
voice, so the pack is made by moving the timbre of real acted Russian emotions (Dusha donors, models_ext/emodonors) onto it
with Seed-VC (F0 kept). Measured on one voice: P(emotion) 0.04-0.09 -> 0.4-0.9, WER and similarity within ~0.1 of plain.

    emopack.build(ref_wav)                 # once per cloned voice, ~2 min, runs Seed-VC in venv_qwen
    wav, text = emopack.ref_for(ref_wav, "joy") or (ref_wav, ref_text)
Emotions: joy, anger, sad.  "neutral" is the original reference.
"""
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import config

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
DONORS = ROOT / "models_ext" / "emodonors"
MANIFEST = ROOT / "voice" / "emodonors.json"
PY = Path(config.venv_python(ROOT / "venv_qwen"))
WORKER = ROOT / "scripts" / "seedvc_batch.py"
EMOTIONS = ("joy", "anger", "sad")


def pack_dir(ref_wav: str) -> Path:
    return Path(str(ref_wav) + ".emopack")


def available() -> bool:
    return PY.exists() and MANIFEST.exists() and DONORS.exists()


def has_pack(ref_wav: str) -> bool:
    return (pack_dir(ref_wav) / "pack.json").exists()


def build(ref_wav: str, ctx=None, timeout: int = 1800) -> bool:
    """Make the pack next to `ref_wav`. Idempotent. ctx (optional) transcribes the converted clips; otherwise the donor text is used."""
    if has_pack(ref_wav):
        return True
    if not available():
        logger.warning("emopack: donors or venv_qwen missing, no pack")
        return False
    donors = json.loads(MANIFEST.read_text(encoding="utf-8"))
    d = pack_dir(ref_wav)
    d.mkdir(parents=True, exist_ok=True)
    jobs = [{"source": str(DONORS / r["file"]), "target": str(Path(ref_wav).resolve()), "out": str(d / r["file"])}
            for rows in donors.values() for r in rows]
    jf = d / "jobs.json"
    jf.write_text(json.dumps(jobs), encoding="utf-8")
    try:
        subprocess.run([str(PY), str(WORKER), str(jf)], cwd=str(ROOT), check=True, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:                                       # noqa: BLE001 -- the pack is an extra, never a blocker
        logger.warning("emopack build failed: %s", e)
        return False
    pack = {}
    for emo, rows in donors.items():
        for r in rows:
            wav = d / r["file"]
            if not wav.exists():
                continue
            text = r["text"]
            if ctx is not None:
                import audio
                text = (audio.transcribe_audio_file(ctx, str(wav), lang_hint="ru") or "").strip() or text
            pack.setdefault(emo, []).append({"wav": str(wav), "text": text})
    if not pack:
        return False
    (d / "pack.json").write_text(json.dumps(pack, ensure_ascii=False, indent=1), encoding="utf-8")
    return True


def ref_for(ref_wav: str, emotion: str, variant: int = 0) -> Optional[Tuple[str, str]]:
    """(wav, text) of the emotional reference, or None (use the plain reference)."""
    p = pack_dir(ref_wav) / "pack.json"
    if not emotion or emotion == "neutral" or not p.exists():
        return None
    rows = json.loads(p.read_text(encoding="utf-8")).get(emotion) or []
    if not rows:
        return None
    r = rows[variant % len(rows)]
    return r["wav"], r["text"]


def build_async(ref_wav: str, ctx=None) -> None:
    """Fire-and-forget build (a cloned voice is usable at once; the pack appears a couple of minutes later). Off under tests."""
    import threading
    if os.environ.get("F5_TEST_RUN") or has_pack(ref_wav) or not available():
        return
    threading.Thread(target=build, args=(ref_wav, ctx), daemon=True, name="emopack").start()
