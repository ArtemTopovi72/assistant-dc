"""LM Studio model management.

Primary path: lms CLI (unload --all, then load with -c / --gpu).
Fallback path: HTTP API POST /api/v0/models/{id}/load (available in LM Studio 0.3.6+).
KV cache quantization and Flash Attention have no CLI flags — they are collected
in the config and either sent via the HTTP API (if available) or returned as
manual instructions for the user.
"""
import logging
import os
import config as _cfg_env   # env_int/env_float: a bad .env value falls back, never crashes the import
from typing import Optional
import re
import subprocess
import threading
import time
from pathlib import Path

import requests

logger = logging.getLogger("assistant.lmstudio")

_T_LIST = 5
_T_RELOAD = 300

# Smallest context the agent can actually work in. The system prompt is ~2.4k
# tokens and the full tool-schema list adds ~7k, so anything at or below 8192
# rejects every tool-bearing call outright ("n_keep >= n_ctx") — measured with
# google/gemma-4-26b-a4b-qat at 9905 needed vs 8192 loaded. 32768 leaves room for
# history and tool results on top of that. House default (user, 2026-09-23):
# 40k context and ONE parallel slot for every model the app loads (raised from
# 20k on 2026-09-24: 48k q8 measured, no speed or VRAM cost; the loader kept 20k).
MIN_CONTEXT_TOKENS = _cfg_env.env_int("LM_MIN_CONTEXT", 40960)
DEFAULT_PARALLEL = _cfg_env.env_int("LM_PARALLEL", 1)

# Virtual model.yaml lives here: ~/.lmstudio/models/<publisher>/<name>/model.yaml
_LMS_MODELS_DIR = Path(os.path.expanduser("~")) / ".lmstudio" / "models"

# Models the loader had to resolve from a virtual (model.yaml) to a raw base GGUF
# because LM Studio cannot programmatically load virtual models (verified on
# 0.4.14: lms CLI, REST /load, the lmstudio SDK, and JIT all return
# `generic.pathNotFound` for every virtual when it is not already loaded; only the
# GUI loads them). When we fall back to the base GGUF we LOSE the virtual's
# `enable_thinking=false`, so the app must apply its own runtime no-think (the
# `<think></think>` prefill). Map: virtual_id -> True. Callers can consult
# `virtual_requires_no_think()` to force no_think on for a resolved virtual.
_RESOLVED_VIRTUAL_NO_THINK: dict = {}


def _virtual_yaml_path(model_id: str):
    """Return the model.yaml Path for a virtual id like 'Pub/Name', or None."""
    if not model_id or "/" not in model_id or "@" in model_id:
        return None
    p = _LMS_MODELS_DIR / model_id / "model.yaml"
    return p if p.exists() else None


def _parse_virtual_yaml(yaml_path: Path) -> dict:
    """Parse the few fields we need from a virtual model.yaml.

    Returns {'base': <relative gguf path>, 'reasoning': bool, 'enable_thinking': bool}.
    Uses PyYAML when available, else a minimal line parser (the file is tiny and
    flat enough that this is reliable for our generated virtuals)."""
    out = {"base": "", "reasoning": None, "enable_thinking": None}
    text = yaml_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text) or {}
        out["base"] = (data.get("base") or "").strip()
        mo = data.get("metadataOverrides") or {}
        if "reasoning" in mo:
            out["reasoning"] = bool(mo["reasoning"])
        for cf in (data.get("customFields") or []):
            if cf.get("key") == "enableThinking":
                out["enable_thinking"] = bool(cf.get("defaultValue", False))
        return out
    except Exception:
        pass
    # Minimal fallback parser.
    m = re.search(r"^\s*base:\s*(.+?)\s*$", text, re.MULTILINE)
    if m:
        out["base"] = m.group(1).strip().strip('"').strip("'")
    m = re.search(r"reasoning:\s*(true|false)", text, re.IGNORECASE)
    if m:
        out["reasoning"] = m.group(1).lower() == "true"
    m = re.search(r"defaultValue:\s*(true|false)", text, re.IGNORECASE)
    if m:
        out["enable_thinking"] = m.group(1).lower() == "true"
    return out


def _base_to_raw_key(base_url: str, base_rel: str):
    """Map a virtual's `base:` gguf path to LM Studio's raw loadable key.

    `base_rel` is like 'Pub/Folder/Folder-Q8_0.gguf'. LM Studio keys raw GGUFs as
    '<folder-name-lowercased>@<quant-from-filename-suffix>'. We derive that and,
    when possible, confirm it against the live /api/v0/models list so we return a
    key that actually exists. Returns (raw_key, quant) or (None, None)."""
    if not base_rel:
        return None, None
    fname = os.path.basename(base_rel)
    stem = re.sub(r"\.gguf$", "", fname, flags=re.IGNORECASE)
    # Quant is the trailing token after the last '-', e.g. '...-Q8_0' -> 'q8_0'.
    quant = stem.rsplit("-", 1)[-1].lower() if "-" in stem else ""
    parent = os.path.basename(os.path.dirname(base_rel))  # the model folder name
    derived = f"{parent.lower()}@{quant}" if quant else parent.lower()

    # Confirm against the live catalog: prefer an id that matches our derivation,
    # else any non-loaded raw id whose folder name matches (quant differences).
    ids = [m.get("id", "") for m in fetch_models(base_url)]
    if derived in ids:
        return derived, quant
    folder_lc = parent.lower()
    for mid in ids:
        if mid.split("@")[0] == folder_lc:
            logger.info("virtual-resolve: derived %r absent; using catalog match %r",
                        derived, mid)
            return mid, (mid.split("@")[1] if "@" in mid else quant)
    # Not found in catalog — return the derivation anyway; the load will surface
    # a clear error if it is wrong.
    return derived, quant


def resolve_virtual_model(base_url: str, model_id: str):
    """If `model_id` is a virtual (model.yaml) that LM Studio cannot load
    programmatically, resolve it to the raw base GGUF key.

    Returns (raw_key, no_think) where `raw_key` is the loadable base id and
    `no_think` is True when the virtual requested thinking-off (so the caller must
    apply the runtime no-think prefill, the base GGUF having no such setting).
    Returns (None, False) when `model_id` is not a resolvable virtual."""
    yaml_path = _virtual_yaml_path(model_id)
    if not yaml_path:
        return None, False
    try:
        info = _parse_virtual_yaml(yaml_path)
    except Exception as exc:
        logger.warning("virtual-resolve: cannot parse %s: %s", yaml_path, exc)
        return None, False
    raw_key, _quant = _base_to_raw_key(base_url, info.get("base", ""))
    if not raw_key:
        logger.warning("virtual-resolve: could not map base %r to a raw key",
                       info.get("base"))
        return None, False
    # thinking-off if either signal says so (reasoning:false or enable_thinking:false)
    no_think = (info.get("reasoning") is False) or (info.get("enable_thinking") is False)
    logger.info("virtual-resolve: %s -> base raw key %s (no_think=%s)",
                model_id, raw_key, no_think)
    _RESOLVED_VIRTUAL_NO_THINK[model_id] = no_think
    return raw_key, no_think


def virtual_requires_no_think(model_id: str) -> bool:
    """True if `model_id` was resolved from a thinking-off virtual to a base GGUF,
    meaning the app should force the runtime no-think prefill."""
    return bool(_RESOLVED_VIRTUAL_NO_THINK.get(model_id, False))


def effective_model_id(base_url: str, model_id: str) -> str:
    """The id LM Studio will actually load for `model_id`: the raw base key if it
    is an unloadable virtual, else `model_id` unchanged. Lets callers compare
    against `loaded_model_ids()` without a phantom-reload loop."""
    raw_key, _ = resolve_virtual_model(base_url, model_id)
    return raw_key or model_id


# ── model info ───────────────────────────────────────────────────────────────

def fetch_models(base_url: str) -> list:
    try:
        r = requests.get(f"{base_url}/api/v0/models", timeout=_T_LIST)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as exc:
        logger.warning("fetch_models: %s", exc)
        return []


def fetch_model(base_url: str, model_id: str) -> dict:
    try:
        r = requests.get(f"{base_url}/api/v0/models/{model_id}", timeout=_T_LIST)
        if r.ok:
            data = r.json()
            if not data.get("error"):
                return data
    except Exception:
        pass
    for m in fetch_models(base_url):
        if m.get("id") == model_id:
            return m
    return {}


def loaded_model_ids(base_url: str) -> list:
    """Ids of every model LM Studio currently has loaded in memory."""
    return [m.get("id") for m in fetch_models(base_url) if m.get("state") == "loaded"]


def loaded_context_length(base_url: str, model_id: str) -> int:
    """Tokens of context the model is CURRENTLY loaded with (0 if unknown).

    Not cosmetic: the agent's system prompt plus the tool schemas needs roughly
    10k tokens, so a model loaded at 8192 rejects every tool-bearing call with
    "n_keep >= n_ctx". Nothing else in the stack notices — the calls come back
    empty and the agent tells the user its tools are unavailable.
    """
    for m in fetch_models(base_url):
        if m.get("id") == model_id and m.get("state") == "loaded":
            try:
                return int(m.get("loaded_context_length") or 0)
            except (TypeError, ValueError):
                return 0
    return 0


_REAL_SUBPROCESS_RUN = subprocess.run


def _would_reach_the_real_cli() -> bool:
    """True when a CLI call from here would actually shell out for real.

    A suite that patches `lmstudio.subprocess` is exercising the CLI logic
    without touching the machine, and must keep working -- two such suites
    exist and the first version of this guard broke both. What matters is
    whether the real subprocess is still in place, not whether we are in a
    test.
    """
    return getattr(subprocess, "run", None) is _REAL_SUBPROCESS_RUN


def _refuse_under_tests(what: str) -> Optional[tuple]:
    """Block model management inside a test run, and say why.

    tests/run_all.py sets F5_TEST_RUN. One suite drove the context-overflow
    path with requests.post patched but subprocess not, so the self-heal ran
    `lms unload --all` for real and every full test run left the machine with
    no model. The damage was invisible at the time and surfaced much later as a
    live research run reporting that successfully fetched pages "contained no
    facts relevant to the topic".

    Returns a (False, message) tuple to return, or None when it is safe.
    """
    if not os.getenv("F5_TEST_RUN"):
        return None
    if not _would_reach_the_real_cli():
        return None                 # subprocess is patched; nothing can escape
    msg = (f"refusing to {what} during a test run (F5_TEST_RUN is set): this "
           f"would move the operator's real model. Patch the boundary you meant "
           f"to exercise instead.")
    logger.error(msg)
    return False, msg


def _lms_unload_all() -> tuple:
    """Run `lms unload --all`. Returns (ok, message). Shared by every CLI path
    that needs to free VRAM before loading a model."""
    blocked = _refuse_under_tests("unload every model")
    if blocked:
        return blocked
    try:
        r = subprocess.run(
            ["lms", "unload", "--all"],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            return False, f"lms unload --all failed: {(r.stderr or r.stdout).strip()}"
        return True, "ok"
    except FileNotFoundError:
        return False, "lms не найден — убедитесь, что LM Studio CLI установлен и доступен в PATH."
    except Exception as exc:
        return False, f"lms unload error: {exc}"


_REAL_UNLOAD_ALL = _lms_unload_all


# ── exclusive load (unload everything else, then load the chosen model) ────────

def load_model_exclusive(
    base_url: str,
    model_id: str,
    context_length: int = 0,
    gpu_offload_pct: int = -1,
) -> tuple:
    """Unload ALL loaded models, then load `model_id` so it is the only one in VRAM.

    This is what "switch model" should do: it frees the previously loaded model
    (the "phantom") instead of relying on LM Studio's JIT load, which leaves the
    old model resident. Returns (ok: bool, message: str).
    """
    ok, msg = _lms_unload_all()
    if not ok:
        return False, msg

    time.sleep(1)  # let VRAM actually free before the load

    # Try the requested id first (a future LM Studio may load virtuals via CLI),
    # then, if that fails and it is a virtual, fall back to its resolved base key.
    # LM Studio 0.4.14 cannot load virtuals programmatically (CLI/REST/SDK/JIT all
    # return pathNotFound), so this fallback is what makes a NO-THINK virtual
    # usable from the app — see resolve_virtual_model().
    candidates = [model_id]
    raw_key, _no_think = resolve_virtual_model(base_url, model_id)
    if raw_key and raw_key != model_id:
        candidates.append(raw_key)

    last_err = ""
    for idx, cand in enumerate(candidates):
        if gpu_offload_pct < 0:
            ok, last_err = _rest_load(base_url, cand, context_length)
            if ok:
                return True, last_err
            logger.warning("REST load of %r failed (%s); trying lms load", cand, last_err)
        cmd = ["lms", "load", cand, "-y",
               "-c", str(context_length if context_length and context_length > 0
                         else MIN_CONTEXT_TOKENS),
               "--parallel", str(DEFAULT_PARALLEL)]
        if gpu_offload_pct >= 0:
            cmd += ["--gpu", _gpu_arg(gpu_offload_pct)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=_T_RELOAD)
            if r.returncode == 0 and not _wait_served(base_url, cand):
                # Live, 2026-09-12: the model crashed, `lms load` exited 0 in
                # nine seconds, and the server then answered "Model reloaded."
                # followed by "No models loaded" for the next minute. The
                # revive path trusted the exit code, cached "ok", and every
                # turn inside its cooldown died. Exit 0 is a promise; only the
                # model list is the fact.
                last_err = f"lms load exited 0 but {cand!r} is not being served"
                logger.warning(last_err)
                continue
            if r.returncode == 0:
                if cand != model_id:
                    note = (f"✓ Loaded base GGUF '{cand}' for virtual '{model_id}' "
                            f"(LM Studio cannot load virtuals directly). "
                            f"Runtime no-think {'ON' if virtual_requires_no_think(model_id) else 'as set'}.")
                    logger.info(note)
                    return True, note
                return True, (r.stdout or "OK").strip()
            last_err = (r.stderr or r.stdout).strip()
            if idx < len(candidates) - 1:
                logger.warning("lms load %r failed (%s); trying base fallback %r",
                               cand, last_err, candidates[idx + 1])
        except subprocess.TimeoutExpired:
            return False, "Timeout — модель может всё ещё загружаться в фоне."
        except Exception as exc:
            last_err = str(exc)
    return False, f"lms load failed: {last_err}"


# A picture's tokens must fit ONE llama.cpp micro-batch (Gemma 4's image
# attention is non-causal: GGML_ASSERT n_ubatch >= n_tokens). The default
# ubatch is 512, so anything over ~1300 px killed the model and every picture
# had to be shrunk to 1024 px; Gemma 4 spends up to 1120 tokens on one image.
# `lms load` cannot set it, the REST load can (physical_batch_size, accepted
# though undocumented in 0.4.x -- echo_load_config shows it applied).
BATCH_TOKENS = _cfg_env.env_int("LM_BATCH_TOKENS", 2048)


def _rest_load(base_url: str, model: str, context_length: int = 0) -> tuple:
    try:
        r = requests.post(base_url.rstrip("/") + "/api/v1/models/load", json={
            "model": model,
            "context_length": context_length if context_length and context_length > 0 else MIN_CONTEXT_TOKENS,
            "parallel": DEFAULT_PARALLEL,
            "eval_batch_size": BATCH_TOKENS, "physical_batch_size": BATCH_TOKENS,
            "echo_load_config": True}, timeout=_T_RELOAD)
        body = r.json() if r.content else {}
        if r.status_code != 200 or body.get("status") != "loaded":
            return False, str(body.get("error") or body or r.status_code)[:300]
        if not _wait_served(base_url, model):
            return False, f"{model!r} loaded but is not being served"
        cfg = body.get("load_config") or {}
        return True, (f"loaded {model} ctx={cfg.get('context_length')} "
                      f"ubatch={cfg.get('physical_batch_size')} parallel={cfg.get('parallel')}")
    except Exception as exc:
        return False, str(exc)


LOAD_SERVED_WAIT_S = 30.0


def _wait_served(base_url: str, model_id: str, timeout: float = None) -> bool:
    """True once LM Studio lists `model_id` as loaded, polling up to `timeout`."""
    deadline = time.monotonic() + (LOAD_SERVED_WAIT_S if timeout is None else timeout)
    while True:
        try:
            if model_id in loaded_model_ids(base_url):
                return True
        except Exception:
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(1.0)


# One loader at a time, process-wide. Live, 2026-09-12: a crash revive from
# the LLM retry path and the card release's reload ran together, LM Studio
# answered "2 models match the provided model key on the same device ...
# Operation canceled", then Internal Server Error, and four turns died. The
# second caller now waits and, if the first one already loaded the model,
# finds it loaded and does nothing.
_ENSURE_LOCK = threading.RLock()


def ensure_exclusive(
    base_url: str,
    model_id: str,
    context_length: int = 0,
    gpu_offload_pct: int = -1,
) -> tuple:
    """Make `model_id` the single loaded model. No-op if it already is.

    Returns (ok, message). On a server we cannot reach we report failure so the
    caller can surface it rather than silently leaving a phantom model loaded.
    Serialised: concurrent callers queue behind one load.
    """
    import llama_backend
    if llama_backend.handles(model_id):
        with _ENSURE_LOCK:
            if model_id not in llama_backend.served_ids():
                llama_backend.unload_lmstudio_chat()
            res = llama_backend.start(model_id)
            llama_backend.ensure_embedder()
            return res
    with _ENSURE_LOCK:
        llama_backend.stop()        # an LM Studio model needs the card back
        res = _ensure_exclusive_locked(base_url, model_id, context_length, gpu_offload_pct)
        # A real (re)load follows `lms unload --all` -- in free_gpu before a
        # render, or in load_model_exclusive itself -- which took BGE-M3 out
        # too, and the knowledge base then silently fell back to keywords.
        # Also on "already loaded": after an app restart the chat model survives
        # while BGE-M3 does not (night 2026-09-24). ensure_embedder throttles itself.
        if res[0]:
            drop_phantoms(model_id)
            llama_backend.ensure_embedder()
        return res


def drop_phantoms(model_id: str) -> list:
    """Unload every "<id>:N" copy beside the real instance. A request that lands
    while the loader has the model unloaded makes LM Studio JIT-load its own
    copy, and the loader's copy then arrives as a second one (live 10-02: an
    app restart mid-revive left two 18 GB instances). Returns what it unloaded."""
    gone = []
    for ident in loaded_instances(model_id):
        if ":" in ident and ident.rsplit(":", 1)[1].isdigit():
            try:
                r = subprocess.run(["lms", "unload", ident], capture_output=True, text=True, timeout=60)
                if r.returncode == 0:
                    gone.append(ident)
                    logger.warning("unloaded phantom instance %s", ident)
            except Exception:
                logger.debug("could not unload %s", ident, exc_info=True)
    return gone


def _ensure_exclusive_locked(base_url, model_id, context_length=0, gpu_offload_pct=-1) -> tuple:
    loaded = loaded_model_ids(base_url)
    # A virtual that LM Studio can't load programmatically is served under its base
    # raw key, so compare against the EFFECTIVE id to avoid a phantom-reload loop.
    target = effective_model_id(base_url, model_id)
    # Membership, not exact list equality: an embedding/reranker model (e.g.
    # EMBED_MODEL served through the same LM Studio instance) is routinely
    # ALSO resident alongside the chat model, so `loaded` legitimately has more
    # than one entry during normal use. The old `loaded == [model_id]` check
    # only matched when the chat model was the ONLY thing loaded, so
    # reselecting the model that's already active triggered a full
    # unload-everything-and-reload the instant anything else (like the
    # embedding model) was also loaded — silently undoing the docstring's own
    # "No-op if it already is" contract.
    active = model_id if model_id in loaded else (target if target in loaded else None)
    if active is not None:
        # "Loaded" is not enough — it must be loaded with enough CONTEXT. The right
        # model sitting at 8192 tokens rejects every call that carries the tool
        # schemas, and the only symptom the user sees is the assistant claiming its
        # tools are unavailable. Reload rather than accept it.
        want = context_length or MIN_CONTEXT_TOKENS
        have = loaded_context_length(base_url, active)
        # NOT a parallelism problem — that was measured and disproved: ONE
        # instance at 32768 with PARALLEL 4 serves two concurrent tool-bearing
        # requests fine. The context is per-instance, not divided between slots.
        if have and have < want:
            logger.warning(
                "%s is loaded with only %d tokens of context; the agent's prompt "
                "plus tool schemas needs ~%d. Reloading it at %d.",
                active, have, want, want)
            return load_model_exclusive(base_url, model_id, want, gpu_offload_pct)
        return True, "already loaded"
    return load_model_exclusive(base_url, model_id, context_length, gpu_offload_pct)


# ── CLI reload (context length + GPU offload) ─────────────────────────────────

def _gpu_arg(pct: int) -> str:
    if pct >= 100:
        return "max"
    if pct <= 0:
        return "off"
    return f"{pct / 100:.2f}"


def loaded_instances(model_id: str = "") -> list:
    """Identifiers of every resident instance of `model_id`.

    LM Studio JIT-loads a SECOND copy ("<id>:2") at the model's default context
    whenever a request arrives for a model it does not consider loaded, and then
    serves requests from either. One healthy 32768 instance next to an 8192
    phantom is indistinguishable from a flaky server: the same prompt succeeds or
    fails depending on which one answers, and concurrency makes it constant.
    """
    out = []
    try:
        r = subprocess.run(["lms", "ps"], capture_output=True, text=True,
                           timeout=30)
        if r.returncode != 0:
            return out
        for line in (r.stdout or "").splitlines()[1:]:
            ident = line.split()[0] if line.split() else ""
            if ident and (not model_id or ident.split(":")[0] == model_id
                          or ident == model_id):
                out.append(ident)
    except Exception:
        logger.debug("could not list loaded instances", exc_info=True)
    return out


def loaded_parallel(model_id: str = "") -> int:
    """How many predictions the loaded model may run at once (1 if unknown).

    Diagnostic only. The obvious theory — that the context is divided between
    the slots, so PARALLEL 4 at 32768 gives each request 8192 — was MEASURED AND
    DISPROVED: one instance at 32768 with PARALLEL 4 served two concurrent
    tool-bearing requests fine. The context is per-instance. The real cause of
    "works alone, fails when two people talk at once" is a phantom second
    instance at the default 8192 (see `loaded_instances`).
    """
    try:
        r = subprocess.run(["lms", "ps"], capture_output=True, text=True,
                           timeout=30)
        if r.returncode != 0:
            return 1
        head, *rows = [l for l in (r.stdout or "").splitlines() if l.strip()]
        try:
            col = head.upper().index("PARALLEL")
        except ValueError:
            return 1
        for row in rows:
            if model_id and model_id not in row:
                continue
            tail = row[col:].split()
            if tail and tail[0].isdigit():
                return max(1, int(tail[0]))
    except Exception:
        logger.debug("could not read the parallel slot count", exc_info=True)
    return 1


def reload_via_cli(model_id: str, context_length: int, gpu_offload_pct: int,
                   parallel: int = 0) -> tuple:
    """Use lms CLI: unload --all, then load with -c and --gpu.
    Returns (ok: bool, message: str).

    `parallel` is passed through when given; `context_length` is the TOTAL, so
    the caller is responsible for multiplying it by the slot count it wants.
    """
    # Only guard the path that would really run: a suite that patched the unload
    # helper is testing this function's own logic, and the unload it "performs"
    # goes nowhere.
    if _lms_unload_all is _REAL_UNLOAD_ALL:
        blocked = _refuse_under_tests("reload the model via the CLI")
        if blocked:
            return blocked
    ok, msg = _lms_unload_all()
    if not ok:
        return False, msg

    time.sleep(1)  # brief pause between unload and load

    cmd = [
        "lms", "load", model_id,
        "-c", str(context_length),
        "--gpu", _gpu_arg(gpu_offload_pct),
        "-y",
    ]
    cmd += ["--parallel", str(int(parallel or DEFAULT_PARALLEL))]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=_T_RELOAD)
        if r.returncode == 0:
            return True, (r.stdout or "OK").strip()
        err = (r.stderr or r.stdout).strip()
        return False, f"lms load failed: {err}"
    except subprocess.TimeoutExpired:
        return False, "Timeout — модель может всё ещё загружаться в фоне."
    except Exception as exc:
        return False, str(exc)


# ── runtime self-heal for an under-sized context ──────────────────────────────
#
# One reload at startup is NOT enough and this was verified the hard way: LM
# Studio JIT-loads a SECOND instance of the same model (identifier "<id>:2") at
# the model's DEFAULT context whenever a request arrives for a model it does not
# consider loaded. A model reloaded at 32768 was silently replaced by an 8192 one
# mid-session, and every tool-bearing call started failing again.
#
# So the fix has to be reactive: when llm.py sees "n_keep >= n_ctx", it asks here
# to fix the model and retries. Guarded per model+size so a genuinely impossible
# request cannot start a reload loop.
# model_id -> (monotonic time of the last repair, attempts so far)
_heal_state: dict = {}
_heal_lock = threading.Lock()
_HEAL_COOLDOWN_S = _cfg_env.env_float("LM_HEAL_COOLDOWN_S", 90)
_HEAL_MAX_ATTEMPTS = _cfg_env.env_int("LM_HEAL_MAX_ATTEMPTS", 5)


def heal_context(model_id: str, needed_tokens: int) -> tuple:
    """Reload `model_id` big enough to fit `needed_tokens`. (ok, message).

    Returns (False, reason) without touching anything if this exact repair has
    already been attempted — one reload per model per size, per process.
    """
    want = max(int(needed_tokens) + 2048, MIN_CONTEXT_TOKENS)
    # Round up to a power-of-two-ish step so repeated near-misses collapse to one
    # reload instead of a ladder of them.
    for step in (20480, 32768, 40960, 65536, 131072, 262144):
        if want <= step:
            want = step
            break
    # The old guard was "one reload per model per SIZE, for the life of the
    # process", and that is what made this failure permanent. LM Studio keeps
    # JIT-loading a SECOND instance ("<id>:2") at the model's default 8192
    # alongside the healthy one, and requests land on whichever — so after a
    # successful heal the fault comes back, the guard says "already attempted a
    # reload at 32768", and nothing is ever repaired again. Seen live: two
    # instances resident at once (8192 and 32768), and every concurrent turn
    # failing while single turns worked.
    #
    # So the guard is now a RATE LIMIT, not a latch: repair whenever the server
    # is actually broken, but never more often than _HEAL_COOLDOWN_S and never
    # more than _HEAL_MAX_ATTEMPTS times per process, so a genuinely impossible
    # request cannot spin the GPU in a reload loop.
    now = time.monotonic()
    with _heal_lock:
        last = _heal_state.get(model_id, (0.0, 0))
        since, attempts = now - last[0], last[1]
        if last[0] and since < _HEAL_COOLDOWN_S:
            return False, (f"a reload of {model_id} was attempted {since:.0f}s ago "
                           f"— waiting out the cooldown")
        if attempts >= _HEAL_MAX_ATTEMPTS:
            return False, (f"{model_id} has already been reloaded "
                           f"{attempts} times in this process — not trying again")
        _heal_state[model_id] = (now, attempts + 1)

    extra = loaded_instances(model_id)
    if len(extra) > 1:
        logger.warning("%d instances of %s are resident (%s) — requests land on "
                       "whichever, so an 8192 phantom breaks calls at random",
                       len(extra), model_id, ", ".join(extra))
    logger.warning("Context too small for %s — reloading a SINGLE instance at %d "
                   "tokens (attempt %d/%d)",
                   model_id, want, attempts + 1, _HEAL_MAX_ATTEMPTS)
    # reload_via_cli unloads EVERYTHING first, which is what clears the phantom.
    ok, msg = reload_via_cli(model_id, want, -1)
    if ok:
        logger.warning("Reloaded %s at %d tokens; retrying the call", model_id, want)
    else:
        logger.error("Could not reload %s at %d tokens: %s", model_id, want, msg)
    return ok, msg


# ── HTTP API reload (all fields, newer LM Studio versions) ────────────────────

def _http_reload(base_url: str, model_id: str, config: dict) -> tuple:
    """Try HTTP POST /api/v0/models/{id}/load. Returns (ok, msg)."""
    try:
        r = requests.post(
            f"{base_url}/api/v0/models/{model_id}/load",
            json={"config": config},
            timeout=_T_RELOAD,
        )
        body = _json_or_empty(r)
        if body.get("error"):
            return False, body["error"]
        return r.ok, "OK" if r.ok else f"HTTP {r.status_code}"
    except Exception as exc:
        return False, str(exc)


# ── unified entry point ───────────────────────────────────────────────────────

class ReloadResult:
    """Outcome of a reload attempt."""
    __slots__ = ("ok", "message", "manual_settings")

    def __init__(self, ok: bool, message: str, manual_settings: str = ""):
        self.ok = ok
        self.message = message
        self.manual_settings = manual_settings  # non-empty → show to user


def reload_model_full(
    base_url: str,
    model_id: str,
    context_length: int,
    gpu_offload_pct: int,
    kv_quant: str,       # "auto" | "f16" | "q8_0" | "q4_0"
    flash_attn: bool,
) -> ReloadResult:
    """
    Reload strategy:
      1. Try lms CLI (context + GPU only).
      2. If CLI succeeds but kv_quant / flash_attn are non-default,
         try the HTTP API to apply them; if that also fails, return manual instructions.
      3. If CLI itself fails, try the HTTP API for everything.
      4. If both fail, return failure with manual instructions.
    """
    needs_manual = _manual_fields(kv_quant, flash_attn)

    # ── attempt 1: lms CLI ──
    cli_ok, cli_msg = reload_via_cli(model_id, context_length, gpu_offload_pct)

    if cli_ok:
        # Some lms versions return before the model is fully resident; wait so the
        # caller's refresh reads the real loaded_context_length, not a transient null.
        wait_for_loaded(base_url, model_id, timeout=120.0)
        actual = fetch_model(base_url, model_id).get("loaded_context_length")
        if actual and int(actual) != int(context_length):
            cli_msg = (f"{cli_msg}\n⚠ Запрошено {context_length:,} токенов, "
                       f"но модель загрузилась с {int(actual):,} (возможно, ограничено моделью).")
        if not needs_manual:
            return ReloadResult(True, f"✓ Перезагружено через lms CLI.\n{cli_msg}")
        # CLI worked, but we still need kv/fa — try HTTP for those
        cfg = {"contextLength": context_length,
               "gpuOffload": gpu_offload_pct / 100.0 if gpu_offload_pct < 100 else 1.0}
        if kv_quant != "auto":
            cfg["kvCacheQuantization"] = kv_quant
        cfg["flashAttention"] = flash_attn
        http_ok, http_msg = _http_reload(base_url, model_id, cfg)
        if http_ok:
            return ReloadResult(True, "✓ Перезагружено (CLI + HTTP API).")
        # HTTP failed — model is loaded via CLI but without kv/fa
        return ReloadResult(
            True,
            "✓ Перезагружено через lms CLI.\n"
            "Контекст и GPU применены. KV-кэш и Flash Attention требуют ручной настройки:",
            manual_settings=_format_manual(kv_quant, flash_attn),
        )

    # ── attempt 2: HTTP API for everything ──
    cfg = build_config(context_length, kv_quant, gpu_offload_pct, flash_attn)
    http_ok, http_msg = _http_reload(base_url, model_id, cfg)
    if http_ok:
        return ReloadResult(True, "✓ Перезагружено через HTTP API.")

    # ── both failed ──
    return ReloadResult(
        False,
        f"CLI: {cli_msg}\nHTTP API: {http_msg}",
        manual_settings=_format_manual_full(context_length, gpu_offload_pct, kv_quant, flash_attn),
    )


def _manual_fields(kv_quant: str, flash_attn: bool) -> bool:
    return kv_quant != "auto" or flash_attn


def _format_manual(kv_quant: str, flash_attn: bool) -> str:
    lines = ["В LM Studio → Edit Model Config:"]
    if kv_quant != "auto":
        lines.append(f"  KV Cache Quantization : {kv_quant}")
    if flash_attn:
        lines.append("  Flash Attention       : вкл")
    return "\n".join(lines)


def _format_manual_full(ctx: int, gpu: int, kv: str, fa: bool) -> str:
    lines = ["Примените вручную в LM Studio → Edit Model Config:"]
    lines.append(f"  Context Length        : {ctx:,}")
    lines.append(f"  GPU Offload           : {gpu}%")
    if kv != "auto":
        lines.append(f"  KV Cache Quantization : {kv}")
    if fa:
        lines.append("  Flash Attention       : вкл")
    return "\n".join(lines)


# ── helpers ───────────────────────────────────────────────────────────────────

def build_config(context_length: int, kv_quant: str,
                 gpu_offload_pct: int, flash_attn: bool) -> dict:
    cfg: dict = {"contextLength": context_length}
    if kv_quant and kv_quant != "auto":
        cfg["kvCacheQuantization"] = kv_quant
    cfg["gpuOffload"] = 1.0 if gpu_offload_pct >= 100 else gpu_offload_pct / 100.0
    cfg["flashAttention"] = flash_attn
    return cfg


def supported_fields(compatibility_type: str) -> set:
    ct = (compatibility_type or "").lower()
    if ct == "gguf":
        return {"contextLength", "kvCacheQuantization", "gpuOffload", "flashAttention"}
    if ct == "mlx":
        return {"contextLength"}
    return {"contextLength"}


def wait_for_loaded(base_url: str, model_id: str, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        m = fetch_model(base_url, model_id)
        if m.get("state") == "loaded":
            return True
        time.sleep(2)
    return False


def _json_or_empty(r: requests.Response) -> dict:
    try:
        return r.json()
    except Exception:
        return {}
