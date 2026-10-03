"""Memory Center — structured, manageable application memory.

The live assistant persists three plain files per profile under ``memory/<profile>/``:
  * ``facts.json``           — durable pinned facts: ``[{ts, text}, ...]``
  * ``session_memory.json``  — rolling notes:        ``[{ts, kind, text, meta}, ...]``
  * ``summary.json``         — one compacted summary: ``{ts, text}``

This module turns those files into first-class, manageable memory WITHOUT changing
what the assistant reads. It is a clean layered design:

  storage layer    — MemoryStore: load/save the canonical files + a sidecar
  editing layer    — add / update / delete (soft) / restore / revisions
  retrieval layer  — Retriever interface (KeywordRetriever now; SemanticRetriever
                     is a documented stub so embeddings can drop in later)
  compaction layer — review-before-commit (no silent loss)
  audit trail      — append-only event log per profile
  diagnostics      — measurable stats + a retrieval simulator (message → rank →
                     select-under-budget → inject)

Canonical entries are enriched IN PLACE with optional keys (``id``, ``importance``,
``source``, ``tags``, ``created``, ``last_accessed``, ``access_count``). The
assistant ignores unknown keys, so the files stay fully compatible. Bulky data
(revision history, audit log, trash) lives in sidecar files prefixed ``.mc_``.

Nothing here is destructive without an explicit call; the UI layer adds the
confirmations. All writes are atomic.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import uuid
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("assistant.memory_center")

# Reuse the assistant's atomic writer so two processes can't truncate a file.
try:
    from models import _atomic_write_json
except Exception:  # pragma: no cover - fallback if import graph changes
    def _atomic_write_json(path: Path, data) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

from config import MEMORY_DIR

META_FILE = ".mc_meta.json"      # {id: {revisions:[...], extra metadata}}
AUDIT_FILE = ".mc_audit.jsonl"   # append-only, one JSON event per line
TRASH_FILE = ".mc_trash.json"    # soft-deleted entries

SOURCES = ("manual", "inferred", "compacted", "imported")
TYPES = ("fact", "session", "summary")
DEFAULT_IMPORTANCE = 50
SUGGESTED_PROFILES = ("default", "research", "coding", "personal")

_TOKEN_CHARS = 4  # rough chars-per-token estimate for budget math


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _tok(text: str) -> int:
    return max(1, round(len(text or "") / _TOKEN_CHARS))


def _semantic_available() -> bool:
    return bool(getattr(SemanticRetriever, "available", False))


# --------------------------------------------------------------------------- #
# Entry model
# --------------------------------------------------------------------------- #
@dataclass
class MemoryEntry:
    id: str
    type: str               # 'fact' | 'session' | 'summary'
    profile: str
    text: str
    kind: str = ""          # session sub-kind ('note', 'inpaint', 'summary', ...)
    created: float = field(default_factory=_now)
    last_accessed: float = 0.0
    access_count: int = 0
    importance: int = DEFAULT_IMPORTANCE
    source: str = "inferred"
    tags: List[str] = field(default_factory=list)
    pinned: bool = False
    meta: dict = field(default_factory=dict)

    def preview(self, n: int = 90) -> str:
        t = " ".join((self.text or "").split())
        return t if len(t) <= n else t[: n - 1] + "…"

    def to_public(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Retrieval layer (pluggable; keyword now, semantic later)
# --------------------------------------------------------------------------- #
class Retriever:
    """Interface so semantic/embedding retrieval can replace keyword cleanly."""
    name = "base"

    def score(self, entry: MemoryEntry, query: str) -> Tuple[float, str]:
        raise NotImplementedError


class KeywordRetriever(Retriever):
    name = "keyword"

    def score(self, entry: MemoryEntry, query: str) -> Tuple[float, str]:
        q = (query or "").strip().lower()
        if not q:
            return 0.0, ""
        text = (entry.text or "").lower()
        terms = [t for t in re.split(r"\W+", q) if t]
        if not terms:
            return 0.0, ""
        hits = [t for t in terms if t in text]
        if not hits:
            # whole-phrase substring as a weak fallback
            if q in text:
                return 0.4, "phrase match"
            return 0.0, ""
        coverage = len(hits) / len(terms)
        # small boost for importance and exact phrase
        boost = 0.15 if q in text else 0.0
        score = min(1.0, coverage * 0.85 + boost + entry.importance / 1000.0)
        return score, "matched: " + ", ".join(sorted(set(hits)))


class SemanticRetriever(Retriever):
    """Placeholder for embedding-based retrieval.

    Wire an embedding model here later (encode entries + query, cosine rank);
    the store will pick it up automatically. Until then callers fall back to
    KeywordRetriever via MemoryStore.search(semantic=...) which degrades safely.
    """
    name = "semantic"
    available = False

    def score(self, entry: MemoryEntry, query: str) -> Tuple[float, str]:
        raise NotImplementedError("semantic retrieval not yet available")


# --------------------------------------------------------------------------- #
# Storage / editing / audit layer
# --------------------------------------------------------------------------- #
class MemoryStore:
    def __init__(self, root: Path = MEMORY_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- profiles ----------------------------------------------------------
    def profiles(self) -> List[str]:
        return sorted(
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ) if self.root.exists() else []

    def profile_dir(self, profile: str) -> Path:
        return self.root / profile

    def ensure_profile(self, profile: str) -> Path:
        d = self.profile_dir(profile)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def create_profile(self, profile: str) -> bool:
        profile = (profile or "").strip()
        if not profile or profile.startswith(".") or "/" in profile or "\\" in profile:
            raise ValueError("invalid profile name")
        d = self.profile_dir(profile)
        if d.exists():
            return False
        d.mkdir(parents=True)
        self.log(profile, "profile_created", {})
        return True

    def rename_profile(self, old: str, new: str) -> bool:
        new = (new or "").strip()
        if not new or new.startswith("."):
            raise ValueError("invalid profile name")
        src, dst = self.profile_dir(old), self.profile_dir(new)
        if not src.exists() or dst.exists():
            return False
        src.rename(dst)
        self.log(new, "profile_renamed", {"from": old})
        return True

    def duplicate_profile(self, src_name: str, dst_name: str) -> bool:
        src, dst = self.profile_dir(src_name), self.profile_dir(dst_name)
        if not src.exists() or dst.exists():
            return False
        shutil.copytree(src, dst)
        self.log(dst_name, "profile_duplicated", {"from": src_name})
        return True

    def delete_profile(self, profile: str) -> bool:
        d = self.profile_dir(profile)
        if not d.exists():
            return False
        shutil.rmtree(d)
        logger.info("Deleted memory profile %s", profile)
        return True

    # ---- canonical file IO -------------------------------------------------
    def _read_json(self, path: Path, default):
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("memory_center: could not read %s: %s", path, exc)
        return default

    def _meta(self, profile: str) -> dict:
        return self._read_json(self.profile_dir(profile) / META_FILE, {})

    def _write_meta(self, profile: str, meta: dict) -> None:
        _atomic_write_json(self.profile_dir(profile) / META_FILE, meta)

    # ---- load entries (join canonical + sidecar) ---------------------------
    def load(self, profile: str) -> List[MemoryEntry]:
        d = self.profile_dir(profile)
        meta = self._meta(profile)
        entries: List[MemoryEntry] = []
        dirty = False

        facts = self._read_json(d / "facts.json", [])
        for f in facts:
            if not isinstance(f, dict) or not str(f.get("text", "")).strip():
                continue
            if not f.get("id"):
                f["id"] = _new_id(); dirty = True
            entries.append(self._entry_from(f, "fact", profile, meta, pinned=True))

        sess = self._read_json(d / "session_memory.json", [])
        for s in sess:
            if not isinstance(s, dict) or not str(s.get("text", "")).strip():
                continue
            if not s.get("id"):
                s["id"] = _new_id(); dirty = True
            entries.append(self._entry_from(s, "session", profile, meta))

        summ = self._read_json(d / "summary.json", {})
        if isinstance(summ, dict) and str(summ.get("text", "")).strip():
            if not summ.get("id"):
                summ["id"] = _new_id(); dirty = True
            entries.append(self._entry_from(summ, "summary", profile, meta, kind="summary"))

        if dirty:
            # backfill the ids we just minted so they remain stable
            self._persist_canonical(profile, entries)
        return entries

    def _entry_from(self, raw: dict, etype: str, profile: str, meta: dict,
                    *, pinned: bool = False, kind: str = "") -> MemoryEntry:
        eid = raw["id"]
        m = meta.get(eid, {})
        return MemoryEntry(
            id=eid, type=etype, profile=profile,
            text=str(raw.get("text", "")),
            kind=kind or raw.get("kind", ""),
            created=float(raw.get("created", raw.get("ts", _now()))),
            last_accessed=float(m.get("last_accessed", raw.get("last_accessed", 0.0))),
            access_count=int(m.get("access_count", raw.get("access_count", 0))),
            importance=int(raw.get("importance", m.get("importance", DEFAULT_IMPORTANCE))),
            source=raw.get("source", m.get("source", "compacted" if etype == "summary" else "inferred")),
            tags=list(raw.get("tags", m.get("tags", []))),
            pinned=pinned,
            meta=raw.get("meta", {}) if isinstance(raw.get("meta"), dict) else {},
        )

    # ---- persist entries back to canonical files ---------------------------
    def _persist_canonical(self, profile: str, entries: List[MemoryEntry]) -> None:
        d = self.ensure_profile(profile)
        facts, sess, summ = [], [], None
        for e in entries:
            if e.type == "fact":
                facts.append({"id": e.id, "ts": e.created, "created": e.created,
                              "text": e.text, "importance": e.importance,
                              "source": e.source, "tags": e.tags})
            elif e.type == "session":
                sess.append({"id": e.id, "ts": e.created, "created": e.created,
                             "kind": e.kind or "note", "text": e.text,
                             "importance": e.importance, "source": e.source,
                             "tags": e.tags, "meta": e.meta})
            elif e.type == "summary":
                summ = {"id": e.id, "ts": e.created, "text": e.text}
        _atomic_write_json(d / "facts.json", facts)
        _atomic_write_json(d / "session_memory.json", sess)
        if summ is not None:
            _atomic_write_json(d / "summary.json", summ)
        elif (d / "summary.json").exists():
            # summary removed
            try:
                (d / "summary.json").unlink()
            except Exception:
                pass

    # ---- single-entry mutations -------------------------------------------
    def get(self, profile: str, entry_id: str) -> Optional[MemoryEntry]:
        for e in self.load(profile):
            if e.id == entry_id:
                return e
        return None

    def add(self, profile: str, text: str, *, etype: str = "fact",
            importance: int = DEFAULT_IMPORTANCE, source: str = "manual",
            tags: Optional[List[str]] = None, kind: str = "note") -> MemoryEntry:
        text = (text or "").strip()
        if not text:
            raise ValueError("empty memory text")
        entries = self.load(profile)
        e = MemoryEntry(id=_new_id(), type=etype, profile=profile, text=text,
                        kind=("summary" if etype == "summary" else kind),
                        created=_now(), importance=int(importance), source=source,
                        tags=list(tags or []), pinned=(etype == "fact"))
        if etype == "summary":
            entries = [x for x in entries if x.type != "summary"] + [e]
        else:
            entries.append(e)
        self._persist_canonical(profile, entries)
        self._add_revision(profile, e.id, "", text)
        self.log(profile, "created", {"id": e.id, "type": etype, "source": source})
        return e

    def update(self, profile: str, entry_id: str, *, text: Optional[str] = None,
               importance: Optional[int] = None, tags: Optional[List[str]] = None,
               source: Optional[str] = None) -> Optional[MemoryEntry]:
        entries = self.load(profile)
        target = None
        old_text = ""
        for e in entries:
            if e.id == entry_id:
                target = e
                old_text = e.text
                if text is not None:
                    e.text = text.strip()
                if importance is not None:
                    e.importance = max(0, min(100, int(importance)))
                if tags is not None:
                    e.tags = list(tags)
                if source is not None:
                    e.source = source
                break
        if target is None:
            return None
        self._persist_canonical(profile, entries)
        if text is not None and text.strip() != old_text:
            self._add_revision(profile, entry_id, old_text, text.strip())
        self.log(profile, "edited", {"id": entry_id})
        return target

    def set_importance(self, profile: str, entry_id: str, importance: int):
        return self.update(profile, entry_id, importance=importance)

    def duplicate(self, profile: str, entry_id: str) -> Optional[MemoryEntry]:
        e = self.get(profile, entry_id)
        if not e:
            return None
        return self.add(profile, e.text + " (copy)", etype=e.type,
                        importance=e.importance, source=e.source, tags=e.tags, kind=e.kind)

    def split(self, profile: str, entry_id: str, parts: List[str]) -> List[MemoryEntry]:
        e = self.get(profile, entry_id)
        if not e:
            return []
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 2:
            return []
        self.delete(profile, [entry_id], soft=True)
        out = []
        for p in parts:
            out.append(self.add(profile, p, etype=e.type, importance=e.importance,
                                source=e.source, tags=e.tags, kind=e.kind))
        self.log(profile, "split", {"from": entry_id, "into": [o.id for o in out]})
        return out

    def merge(self, profile: str, ids: List[str], joiner: str = " ") -> Optional[MemoryEntry]:
        entries = self.load(profile)
        chosen = [e for e in entries if e.id in ids]
        if len(chosen) < 2:
            return None
        etype = chosen[0].type
        text = joiner.join(e.text for e in chosen)
        imp = max(e.importance for e in chosen)
        tags = sorted({t for e in chosen for t in e.tags})
        self.delete(profile, ids, soft=True)
        merged = self.add(profile, text, etype=etype, importance=imp,
                          source="manual", tags=tags, kind=chosen[0].kind)
        self.log(profile, "merged", {"from": ids, "into": merged.id})
        return merged

    def move(self, src_profile: str, entry_id: str, dst_profile: str) -> Optional[MemoryEntry]:
        e = self.get(src_profile, entry_id)
        if not e:
            return None
        self.ensure_profile(dst_profile)
        moved = self.add(dst_profile, e.text, etype=e.type, importance=e.importance,
                         source=e.source, tags=e.tags, kind=e.kind)
        self.delete(src_profile, [entry_id], soft=True)
        self.log(dst_profile, "moved_in", {"id": moved.id, "from": src_profile})
        self.log(src_profile, "moved_out", {"id": entry_id, "to": dst_profile})
        return moved

    # ---- deletion (soft by default) ---------------------------------------
    def delete(self, profile: str, ids: List[str], *, soft: bool = True) -> int:
        entries = self.load(profile)
        ids = set(ids)
        keep = [e for e in entries if e.id not in ids]
        removed = [e for e in entries if e.id in ids]
        if not removed:
            return 0
        if soft:
            trash = self._read_trash(profile)
            for e in removed:
                rec = e.to_public()
                rec["deleted_ts"] = _now()
                trash.append(rec)
            self._write_trash(profile, trash)
        self._persist_canonical(profile, keep)
        self.log(profile, "deleted" if soft else "purged",
                 {"ids": list(ids), "count": len(removed), "soft": soft})
        return len(removed)

    def clear(self, profile: str, *, types: Tuple[str, ...] = TYPES, soft: bool = True) -> int:
        entries = self.load(profile)
        ids = [e.id for e in entries if e.type in types]
        return self.delete(profile, ids, soft=soft) if ids else 0

    # ---- trash -------------------------------------------------------------
    def _read_trash(self, profile: str) -> List[dict]:
        return self._read_json(self.profile_dir(profile) / TRASH_FILE, [])

    def _write_trash(self, profile: str, data: List[dict]) -> None:
        _atomic_write_json(self.profile_dir(profile) / TRASH_FILE, data)

    def trash(self, profile: str) -> List[dict]:
        return self._read_trash(profile)

    def restore(self, profile: str, ids: List[str]) -> int:
        trash = self._read_trash(profile)
        ids = set(ids)
        restore_recs = [r for r in trash if r.get("id") in ids]
        if not restore_recs:
            return 0
        for r in restore_recs:
            self.add(profile, r.get("text", ""), etype=r.get("type", "fact"),
                     importance=int(r.get("importance", DEFAULT_IMPORTANCE)),
                     source=r.get("source", "imported"), tags=r.get("tags", []),
                     kind=r.get("kind", "note"))
        remaining = [r for r in trash if r.get("id") not in ids]
        self._write_trash(profile, remaining)
        self.log(profile, "restored", {"count": len(restore_recs)})
        return len(restore_recs)

    def empty_trash(self, profile: str) -> int:
        trash = self._read_trash(profile)
        self._write_trash(profile, [])
        return len(trash)

    # ---- revisions ---------------------------------------------------------
    def _add_revision(self, profile: str, entry_id: str, before: str, after: str) -> None:
        meta = self._meta(profile)
        node = meta.setdefault(entry_id, {})
        revs = node.setdefault("revisions", [])
        revs.append({"ts": _now(), "before": before, "after": after})
        node["revisions"] = revs[-25:]  # keep last 25
        self._write_meta(profile, meta)

    def revisions(self, profile: str, entry_id: str) -> List[dict]:
        return self._meta(profile).get(entry_id, {}).get("revisions", [])

    def revert(self, profile: str, entry_id: str, index: int) -> Optional[MemoryEntry]:
        revs = self.revisions(profile, entry_id)
        if not (0 <= index < len(revs)):
            return None
        return self.update(profile, entry_id, text=revs[index]["after"])

    # ---- retrieval / search ------------------------------------------------
    def search(self, profile: str, query: str = "", *, semantic: bool = False,
               types: Optional[Tuple[str, ...]] = None,
               sources: Optional[Tuple[str, ...]] = None,
               min_importance: int = 0,
               date_from: Optional[float] = None, date_to: Optional[float] = None,
               tag: str = "") -> List[Tuple[MemoryEntry, float, str]]:
        retriever: Retriever = KeywordRetriever()
        if semantic and _semantic_available():
            retriever = SemanticRetriever()
        out = []
        for e in self.load(profile):
            if types and e.type not in types:
                continue
            if sources and e.source not in sources:
                continue
            if e.importance < min_importance:
                continue
            if date_from and e.created < date_from:
                continue
            if date_to and e.created > date_to:
                continue
            if tag and tag not in e.tags:
                continue
            if query.strip():
                score, reason = retriever.score(e, query)
                if score <= 0:
                    continue
            else:
                score, reason = 1.0, ""
            out.append((e, score, reason))
        out.sort(key=lambda t: (t[1], t[0].importance, t[0].created), reverse=True)
        return out

    # ---- overview / diagnostics -------------------------------------------
    def _dir_size(self, profile: str) -> int:
        total = 0
        d = self.profile_dir(profile)
        if d.exists():
            for p in d.rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
        return total

    def overview(self, profile: str) -> dict:
        entries = self.load(profile)
        by_type = {t: 0 for t in TYPES}
        last_update = 0.0
        for e in entries:
            by_type[e.type] = by_type.get(e.type, 0) + 1
            last_update = max(last_update, e.created)
        summ_file = self.profile_dir(profile) / "summary.json"
        last_compact = 0.0
        sj = self._read_json(summ_file, {})
        if isinstance(sj, dict):
            last_compact = float(sj.get("ts", 0.0))
        return {
            "profile": profile,
            "total": len(entries),
            "facts": by_type.get("fact", 0),
            "session": by_type.get("session", 0),
            "summary": by_type.get("summary", 0),
            "disk_bytes": self._dir_size(profile),
            "last_update": last_update,
            "last_compaction": last_compact,
            "retrieval_mode": KeywordRetriever.name,
            "semantic_enabled": _semantic_available(),
            "trash": len(self.trash(profile)),
        }

    def diagnostics(self, profile: str, *, query: str = "", token_budget: int = 1500) -> dict:
        """Concrete retrieval trace: message → rank → select-under-budget → inject."""
        entries = self.load(profile)
        facts = [e for e in entries if e.type == "fact"]
        # facts are always injected; session/summary compete for the remaining budget
        ranked = self.search(profile, query, types=("session", "summary")) if query \
            else [(e, 1.0, "") for e in entries if e.type in ("session", "summary")]
        fact_tokens = sum(_tok(e.text) for e in facts)
        budget_left = max(0, token_budget - fact_tokens)
        selected, dropped, used = [], [], 0
        for e, score, reason in ranked:
            t = _tok(e.text)
            if used + t <= budget_left:
                selected.append((e, score, reason, t)); used += t
            else:
                dropped.append((e, score, reason, t))
        per_profile = {p: self._dir_size(p) for p in self.profiles()}
        return {
            "query": query,
            "token_budget": token_budget,
            "facts_injected": len(facts),
            "fact_tokens": fact_tokens,
            "candidates": len(ranked),
            "selected": selected,
            "dropped_budget": dropped,
            "tokens_used": fact_tokens + used,
            "retrieval_mode": KeywordRetriever.name,
            "semantic_enabled": _semantic_available(),
            "per_profile_bytes": per_profile,
        }

    # ---- compaction review (no silent loss) -------------------------------
    def compaction_sources(self, profile: str) -> List[MemoryEntry]:
        """The session entries that would feed a compaction (summary excluded)."""
        return [e for e in self.load(profile) if e.type == "session"]

    def commit_compaction(self, profile: str, summary: str, *,
                          remove_sources: bool = True, archive: bool = True) -> None:
        """Apply an APPROVED compaction. Sources are soft-deleted (recoverable from
        trash) — never silently lost. Pinned facts are untouched."""
        summary = (summary or "").strip()
        if not summary:
            return
        if remove_sources:
            self.clear(profile, types=("session",), soft=archive)
        self.add(profile, summary, etype="summary", source="compacted")
        self.log(profile, "compacted", {"summary_tokens": _tok(summary),
                                        "archived_sources": archive})

    # ---- audit log ---------------------------------------------------------
    def log(self, profile: str, event: str, data: dict) -> None:
        try:
            d = self.ensure_profile(profile)
            line = json.dumps({"ts": _now(), "event": event, **data}, ensure_ascii=False)
            with open(d / AUDIT_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            logger.debug("audit log write failed: %s", exc)

    def audit(self, profile: str, limit: int = 500) -> List[dict]:
        path = self.profile_dir(profile) / AUDIT_FILE
        if not path.exists():
            return []
        out = []
        try:
            for ln in path.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if ln:
                    try:
                        out.append(json.loads(ln))
                    except Exception:
                        pass
        except Exception as exc:
            logger.debug("audit read failed: %s", exc)
        return out[-limit:][::-1]  # newest first

    # ---- backup / restore --------------------------------------------------
    def export_json(self, profiles: List[str]) -> dict:
        return {
            "format": "memory-center-export",
            "version": 1,
            "exported_ts": _now(),
            "profiles": {p: [e.to_public() for e in self.load(p)] for p in profiles},
        }

    def export_markdown(self, profiles: List[str]) -> str:
        lines = [f"# Memory export — {time.strftime('%Y-%m-%d %H:%M')}", ""]
        for p in profiles:
            entries = self.load(p)
            lines.append(f"## Profile: {p}  ({len(entries)} entries)")
            for t in TYPES:
                group = [e for e in entries if e.type == t]
                if not group:
                    continue
                lines.append(f"\n### {t} ({len(group)})")
                for e in group:
                    tags = f"  _tags: {', '.join(e.tags)}_" if e.tags else ""
                    lines.append(f"- (imp {e.importance}, {e.source}) {e.text}{tags}")
            lines.append("")
        return "\n".join(lines)

    def export_zip(self, profiles: List[str], dest: Path) -> Path:
        dest = Path(dest)
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("export.json", json.dumps(self.export_json(profiles),
                                                 ensure_ascii=False, indent=2))
            z.writestr("export.md", self.export_markdown(profiles))
            for p in profiles:
                d = self.profile_dir(p)
                for fn in ("facts.json", "session_memory.json", "summary.json"):
                    fp = d / fn
                    if fp.exists():
                        z.write(fp, f"{p}/{fn}")
        return dest

    def preview_import(self, path: Path) -> dict:
        """Read an export (JSON or ZIP) and report what it contains — no commit."""
        path = Path(path)
        data = None
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as z:
                if "export.json" in z.namelist():
                    data = json.loads(z.read("export.json").decode("utf-8"))
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
        if not data or "profiles" not in data:
            raise ValueError("not a Memory Center export")
        return {p: len(v) for p, v in data["profiles"].items()}, data

    def commit_import(self, data: dict, *, target_profile: Optional[str] = None,
                      merge: bool = True) -> int:
        count = 0
        for p, entries in data.get("profiles", {}).items():
            dest = target_profile or p
            self.ensure_profile(dest)
            if not merge:
                self.clear(dest, soft=True)
            for e in entries:
                self.add(dest, e.get("text", ""), etype=e.get("type", "fact"),
                         importance=int(e.get("importance", DEFAULT_IMPORTANCE)),
                         source="imported", tags=e.get("tags", []),
                         kind=e.get("kind", "note"))
                count += 1
            self.log(dest, "imported", {"count": len(entries), "merge": merge})
        return count

    # ---- live-ctx sync -----------------------------------------------------
    def sync_into_ctx(self, ctx, profile: str) -> None:
        """After mutating the active profile, refresh the live assistant Context so
        its in-memory deque matches disk (and the close-save won't clobber edits)."""
        try:
            if ctx is None:
                return
            active = getattr(ctx, "active_memory_dir", None)
            if active is None or Path(active).name != profile:
                return
            with ctx.memory_lock:
                ctx.session_memory.clear()
                ctx.pinned_facts.clear()
            ctx.load_memory(self.profile_dir(profile))
        except Exception as exc:
            logger.warning("sync_into_ctx failed: %s", exc)
