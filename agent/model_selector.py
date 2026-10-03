"""Interactive LM Studio model selection.

Queries the LM Studio REST API for available models and lets the user pick one
(plus a thinking on/off toggle) at startup. Selecting a not-loaded model is fine:
LM Studio JIT-loads it on the first chat request.
"""
import glob
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import requests

logger = logging.getLogger("assistant.model_selector")

# Model ids that are not chat LLMs (won't appear in the picker).
_NON_LLM_HINTS = ("embed", "embedding", "reranker", "whisper")

# Where LM Studio stores GGUF files (the model API does not report sizes).
_MODELS_DIR = Path(os.getenv("LMSTUDIO_MODELS_DIR", str(Path.home() / ".lmstudio" / "models")))
# LM Studio's separate catalog dir: "virtual model" aliases (the id the API
# reports, e.g. google/gemma-4-12b-qat) each get a model.yaml here that points
# at the REAL downloaded repo under _MODELS_DIR by key, not by file path.
_HUB_MODELS_DIR = Path(os.getenv("LMSTUDIO_HUB_DIR", str(Path.home() / ".lmstudio" / "hub" / "models")))
_QUANT_RE = re.compile(r"(bf16|q\d_k_m|q\d_k_s|q\d_\d|q\d_k|iq\d\w*|q\d+)")


def _index_model_sizes(models_dir: Path, hub_dir: Path = _HUB_MODELS_DIR) -> dict:
    """Best-effort map of on-disk GGUF sizes, keyed by (repo, quant) and repo.

    The model API has no size field, so we scan the GGUF files. Repo folders may
    carry a '-GGUF' suffix the API id drops; virtual models (model.yaml) point at
    a base GGUF. mmproj side-files are skipped.
    """
    by_key: dict = {}
    by_repo: dict = {}

    def add(repo: str, quant: str, size: int) -> None:
        by_key[(repo, quant)] = by_key.get((repo, quant), 0) + size
        by_repo[repo] = max(by_repo.get(repo, 0), size)

    try:
        for gg in glob.glob(str(models_dir / "**" / "*.gguf"), recursive=True):
            name = os.path.basename(gg).lower()
            if name.startswith("mmproj"):
                continue
            repo = os.path.basename(os.path.dirname(gg)).lower()
            qm = _QUANT_RE.search(name)
            quant = qm.group(1) if qm else ""
            size = os.path.getsize(gg)
            for r in {repo, re.sub(r"-gguf$", "", repo)}:
                add(r, quant, size)
        for yml in glob.glob(str(models_dir / "**" / "model.yaml"), recursive=True):
            try:
                m = re.search(r"base:\s*(.+\.gguf)", open(yml, encoding="utf-8").read())
                if m:
                    base = models_dir / m.group(1).strip()
                    if base.exists():
                        by_repo[os.path.basename(os.path.dirname(yml)).lower()] = os.path.getsize(base)
            except Exception:
                pass
        # Hub catalog aliases: model.yaml here has no inline "base: x.gguf"
        # path, just a "base: - key: <publisher>/<repo>" reference to a repo
        # already sized above (or, for a not-yet-downloaded alternate quant,
        # not found at all -- left unsized, same as any other unknown model).
        for yml in glob.glob(str(hub_dir / "**" / "model.yaml"), recursive=True):
            try:
                text = open(yml, encoding="utf-8").read()
                mid = re.search(r"^model:\s*(\S+)", text, re.M)
                key = re.search(r"^\s*-\s*key:\s*(\S+)", text, re.M)
                if not (mid and key):
                    continue
                alias = mid.group(1).split("/")[-1].lower()
                ref = key.group(1).split("/")[-1].lower()
                size = by_repo.get(ref) or by_repo.get(re.sub(r"-gguf$", "", ref))
                if size and alias not in by_repo:
                    by_repo[alias] = size
            except Exception:
                pass
        # Non-GGUF formats (MLX, safetensors, ...) have no single file to size,
        # so fall back to the whole repo folder's total size for any repo the
        # GGUF/yaml scan above didn't already cover.
        for org_dir in models_dir.iterdir() if models_dir.is_dir() else []:
            if not org_dir.is_dir():
                continue
            for repo_dir in org_dir.iterdir():
                if not repo_dir.is_dir():
                    continue
                repo = repo_dir.name.lower()
                if repo in by_repo:
                    continue
                total = sum(f.stat().st_size for f in repo_dir.rglob("*") if f.is_file())
                if total:
                    by_repo[repo] = total
    except Exception as exc:
        logger.debug("Could not index model sizes: %s", exc)
    return {"by_key": by_key, "by_repo": by_repo}


def _size_gb(model_id: str, index: dict) -> Optional[float]:
    base, _, quant = model_id.partition("@")
    segments = base.split("/")
    # Most ids are "publisher/repo". Some LM Studio server ids for a
    # multi-quant repo (several .gguf files sharing one folder) are
    # "publisher/repo/repo-quant.gguf" -- a THIRD segment that is the actual
    # filename, not the repo. Settings showed "?" for exactly these: the last
    # segment (the .gguf filename) was looked up against _index_model_sizes'
    # by_repo, which is keyed by the FOLDER name (the second-to-last segment),
    # and a filename never matches a folder name. Try both, and when the last
    # segment IS a filename, pull its quant out of the name itself (there is
    # no "@quant" suffix on this id shape) so two quants sharing one folder
    # resolve to their own sizes via by_key instead of both getting the
    # folder's largest file from by_repo.
    filename_quant = ""
    if segments[-1].lower().endswith(".gguf"):
        qm = _QUANT_RE.search(segments[-1].lower())
        filename_quant = qm.group(1) if qm else ""
    candidates = [segments[-1]]
    if len(segments) >= 2:
        candidates.append(segments[-2])
    candidates = [c[:-5] if c.endswith(".gguf") else c for c in candidates]
    for q in (filename_quant, quant.lower()):
        for cand in candidates:
            nbytes = index["by_key"].get((cand.lower(), q))
            if nbytes:
                return nbytes / 1e9
    for cand in candidates:
        nbytes = index["by_repo"].get(cand.lower())
        if nbytes:
            return nbytes / 1e9
    return None


def _is_non_llm(model: dict) -> bool:
    """True if ``model``'s id hints at an embedding/reranker/whisper model."""
    return any(h in model.get("id", "").lower() for h in _NON_LLM_HINTS)


def _disk_models() -> List[dict]:
    """The models on disk, from `lms ls --json`.

    Needed because the REST API only advertises models the server has actually
    LOADED when just-in-time loading is off -- so the picker went from "every
    model you own" to "the one that happens to be running", which looks like the
    app lost the list. The CLI reads the same local catalog LM Studio itself
    shows, so this is the honest full set. Failure is silent on purpose: no CLI,
    no problem, the API answer still stands.
    """
    import json as _json
    import shutil as _shutil
    import subprocess as _subprocess
    exe = _shutil.which("lms")
    if not exe:
        return []
    try:
        out = _subprocess.run([exe, "ls", "--json"], capture_output=True,
                              text=True, timeout=15,
                              stdin=_subprocess.DEVNULL)
        rows = _json.loads(out.stdout or "[]")
    except Exception as exc:
        logger.debug("lms ls --json failed: %s", exc)
        return []
    got = []
    for r in rows:
        if not isinstance(r, dict) or r.get("type") != "llm":
            continue
        mid = r.get("modelKey") or r.get("indexedModelIdentifier") or ""
        if mid:
            got.append({"id": mid, "type": "llm", "state": "not-loaded"})
    return got


def _merge_disk(models: List[dict]) -> List[dict]:
    """API models first (they carry the load state), then whatever else is on
    disk. Never drops an API entry: the server is the authority on state."""
    seen = {m.get("id", "") for m in models}
    extra = [m for m in _disk_models()
             if m["id"] not in seen and not _is_non_llm(m)]
    return list(models) + extra


def list_models(base_url: str) -> List[dict]:
    """Return available chat models, preferring the native API (has load state).

    Each dict has at least an ``id``; native API also provides ``state`` and
    ``type``. Embedding/reranker models are filtered out.
    """
    # Native API exposes state + type; fall back to OpenAI-compatible endpoint.
    try:
        r = requests.get(f"{base_url}/api/v0/models", timeout=10)
        r.raise_for_status()
        data = r.json().get("data", [])
        models = [
            m for m in data
            if m.get("type") in (None, "llm", "vlm") and not _is_non_llm(m)
        ]
        if models:
            return _merge_disk(models)
    except Exception as exc:
        logger.debug("Native /api/v0/models failed, falling back: %s", exc)

    try:
        r = requests.get(f"{base_url}/v1/models", timeout=10)
        r.raise_for_status()
        return _merge_disk(
            [m for m in r.json().get("data", []) if not _is_non_llm(m)])
    except Exception as exc:
        logger.error("Could not list models from %s: %s", base_url, exc)
        return _disk_models()


def choose_model_interactive(base_url: str, default_model: str) -> Tuple[str, bool]:
    """Show a numbered menu and return ``(model_id, no_think)``.

    Falls back to ``(default_model, True)`` if the server is unreachable or the
    user just presses Enter.
    """
    models = list_models(base_url)
    if not models:
        print(f"⚠️  Could not reach LM Studio at {base_url}. Using default: {default_model}")
        return default_model, True

    # Put the default first if present, so Enter picks something sensible.
    ids = [m.get("id", "") for m in models]
    default_idx = ids.index(default_model) + 1 if default_model in ids else 1

    sizes = _index_model_sizes(_MODELS_DIR)
    print("\n" + "=" * 68)
    print("Available models (● loaded / ○ on disk):")
    for i, m in enumerate(models, 1):
        marker = "●" if m.get("state") == "loaded" else "○"
        gb = _size_gb(m["id"], sizes)
        size_str = f"{gb:5.1f} GB" if gb is not None else "    ?   "
        tag = " (default)" if i == default_idx else ""
        print(f"  {i:2}. {marker} {size_str}  {m['id']}{tag}")
    print("=" * 68)

    chosen_id = models[default_idx - 1]["id"]
    while True:
        raw = input(f"Select model [1-{len(models)}] (Enter = {chosen_id}): ").strip()
        if not raw:
            break
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            chosen_id = models[int(raw) - 1]["id"]
            break
        print("  Invalid choice, try again.")

    answer = input("Enable thinking/reasoning? (y/N): ").strip().lower()
    no_think = answer not in ("y", "yes")

    print(f"→ Using '{chosen_id}' with thinking {'OFF' if no_think else 'ON'}\n")
    return chosen_id, no_think
