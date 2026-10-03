"""On-disk state for a research run.

A run persists as it goes so it survives a crash or a session end: state.json
holds the latest snapshot, history/NNN_<phase>.json keeps every phase as an
immutable ordered record (so the whole run is reconstructable from disk), and
report.md + meta.json are the final product.

Every write here is best-effort — a full disk or a locked file must never abort
a research run that is otherwise going fine, so failures are logged and
swallowed.
"""
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import dr_settings as S

logger = logging.getLogger("assistant.research")


def _slug(text: str, maxlen: int = 48) -> str:
    """Filesystem-safe stem for a run directory."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return (s[:maxlen].rstrip("-")) or "research"


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def _save_state(run_dir: Path, **fields) -> None:
    """Best-effort incremental state dump (so a crashed run leaves a trace).

    Writes state.json = the LATEST snapshot, and ALSO appends an immutable, ordered
    history/NNN_<phase>.json so the entire run is reconstructable from disk (spec E):
    every phase keeps its own timestamp + inputs/outputs/stats instead of being
    overwritten."""
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        fields["ts"] = time.time()
        tmp = run_dir / "state.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(fields, f, ensure_ascii=False, indent=2)
        tmp.replace(run_dir / "state.json")
    except Exception as exc:
        logger.debug("save_state failed: %s", exc)
    if S.DR_PHASE_HISTORY_ENABLED and fields.get("phase"):
        _append_history(run_dir, fields)


def _append_history(run_dir: Path, fields: dict) -> None:
    """Append one ordered, never-overwritten phase record under history/."""
    try:
        hdir = run_dir / "history"
        hdir.mkdir(parents=True, exist_ok=True)
        seq = sum(1 for _ in hdir.glob("[0-9][0-9][0-9]_*.json")) + 1
        phase = re.sub(r"[^A-Za-z0-9_-]+", "-", str(fields.get("phase", "phase")))[:48]
        rec = {"seq": seq, "wall_clock": _now_iso(), **fields}
        (hdir / f"{seq:03d}_{phase}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("append_history failed: %s", exc)


def _now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now().isoformat(timespec="seconds")


def _write_report(run_dir: Path, topic: str, report: str, meta: dict) -> Optional[str]:
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "report.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return str(path)
    except Exception as exc:
        logger.error("Failed to write research report: %s", exc)
        return None
