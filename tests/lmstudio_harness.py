"""
Hardened LM Studio test harness.

Background: a vision investigation (docs/vision_pipeline_forensics.md) reached a WRONG
conclusion because of testing confounds, not real behaviour:
  * phantom model id        — requesting an id LM Studio doesn't have...
  * silent substitution     — ...made it answer with the LOADED model instead, undetected.
  * "Model unloaded"        — an oversized model errored instead of serving.
  * global model state      — concurrent tests loaded/evicted each other's models.

This module makes every one of those IMPOSSIBLE by construction. Use it for ANY
integration test that talks to LM Studio.

Guarantees
----------
1. Served-id assertion: every chat() checks that the response ``model`` field EXACTLY
   equals the requested id. A mismatch raises ModelSubstitution and stops the test.
2. Preflight availability: the model must exist in the catalog and be loaded (or
   loadable) before a request is sent — else ModelNotFound / ModelUnavailable.
3. Strict serialization: a cross-process file lock guards every request, so two tests
   can never load models concurrently.
4. Error bodies are failures: an ``{"error": ...}`` body (e.g. "Model unloaded") raises
   ModelUnavailable; it is never mistaken for a valid empty answer.

Nothing here is silent: every guard raises rather than returning a degraded result.
"""
from __future__ import annotations

import base64
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import json
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

BASE = os.getenv("LMSTUDIO_BASE", "http://localhost:1234")
REST_MODELS = f"{BASE}/api/v0/models"      # reports load STATE (loaded / not-loaded)
CHAT = f"{BASE}/v1/chat/completions"
_LOCK_PATH = os.path.join(tempfile.gettempdir(), "lmstudio_test.lock")


# --------------------------------------------------------------------------- errors
class HarnessError(RuntimeError):
    """Base class for all harness failures."""


class ModelNotFound(HarnessError):
    """Requested model id is not in LM Studio's catalog (phantom id)."""


class ModelUnavailable(HarnessError):
    """Model exists but could not be served (not loaded / 'Model unloaded' / load failed)."""


class ModelSubstitution(HarnessError):
    """LM Studio served a DIFFERENT model than requested. The cardinal confound."""


class EmptyResponse(HarnessError):
    """Served the right model but returned no content (caller decides if that's fatal)."""


# --------------------------------------------------------------------------- result
@dataclass
class ChatResult:
    requested_model: str
    served_model: str
    content: str
    reasoning_content: str
    finish_reason: Optional[str]
    raw: dict

    @property
    def empty(self) -> bool:
        return not (self.content.strip() or self.reasoning_content.strip())


# ----------------------------------------------------------------------- model state
def list_models() -> Dict[str, str]:
    """{model_id: state} from LM Studio's REST endpoint. state ∈ {loaded, not-loaded}."""
    r = requests.get(REST_MODELS, timeout=30)
    r.raise_for_status()
    return {m["id"]: m.get("state", "unknown") for m in r.json().get("data", [])}


def model_state(model_id: str) -> Optional[str]:
    """Load state for one model, or None if it's not in the catalog at all."""
    return list_models().get(model_id)


def _find_lms() -> Optional[str]:
    """Locate the `lms` CLI (used to load models deterministically)."""
    for cand in (shutil.which("lms"),
                 os.path.expanduser("~/.lmstudio/bin/lms.exe"),
                 os.path.expanduser("~/.lmstudio/bin/lms")):
        if cand and os.path.exists(cand):
            return cand
    return None


def ensure_loaded(model_id: str, *, ttl: int = 3600, load_timeout: int = 600) -> None:
    """Guarantee `model_id` is loaded and serving, or raise.

    - Not in catalog               -> ModelNotFound (phantom id).
    - In catalog but not loaded     -> load it via the `lms` CLI, then re-verify.
    - Still not loaded after trying -> ModelUnavailable.
    """
    st = model_state(model_id)
    if st is None:
        raise ModelNotFound(
            f"{model_id!r} is NOT in LM Studio's catalog. Requesting it would make LM "
            f"Studio silently serve the loaded model instead. Available: "
            f"{sorted(list_models())}"
        )
    if st == "loaded":
        return
    lms = _find_lms()
    if not lms:
        raise ModelUnavailable(
            f"{model_id!r} is present but not loaded, and the `lms` CLI was not found to "
            f"load it. Load it manually in LM Studio, then re-run."
        )
    # Explicit, deterministic load (JIT-on-request is unreliable for big models).
    subprocess.run([lms, "load", model_id, "--ttl", str(ttl), "-y"],
                   capture_output=True, text=True, timeout=load_timeout)
    if model_state(model_id) != "loaded":
        raise ModelUnavailable(
            f"{model_id!r} failed to load (likely too large for available memory — the "
            f"'Model unloaded' class of failure). Do NOT continue; results would be invalid."
        )


@contextmanager
def serial_lock(timeout: float = 1800.0, poll: float = 0.5):
    """Cross-process exclusive lock. LM Studio has GLOBAL model state, so every test that
    may load/evict a model must hold this. Concurrent acquisition is impossible."""
    deadline = time.time() + timeout
    fd = None
    while True:
        try:
            fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode())
            break
        except FileExistsError:
            if time.time() > deadline:
                raise HarnessError(
                    f"Could not acquire the LM Studio serial lock within {timeout}s "
                    f"({_LOCK_PATH}). Another model test is running — they must NOT run "
                    f"concurrently. If stale, delete the lock file."
                )
            time.sleep(poll)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.remove(_LOCK_PATH)
        except OSError:
            pass


# ------------------------------------------------------------------------------ chat
def chat(model: str, messages: List[dict], *, temperature: float = 0.0,
         max_tokens: int = 400, assert_served: bool = True,
         preflight: bool = True, allow_empty: bool = True, **payload_extra) -> ChatResult:
    """Single non-streaming chat completion with ALL guards on.

    Raises ModelNotFound / ModelUnavailable / ModelSubstitution before it ever trusts a
    response. Serialized via the global lock. `assert_served=False` is provided ONLY so a
    test can deliberately demonstrate the substitution guard; never disable it in a real
    test.
    """
    with serial_lock():
        if preflight:
            ensure_loaded(model)
        payload = {"model": model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens, "stream": False, "reasoning": "off"}
        payload.update(payload_extra)
        r = requests.post(CHAT, json=payload, timeout=1800)
        try:
            body = r.json()
        except ValueError:
            raise ModelUnavailable(f"Non-JSON response (HTTP {r.status_code}): {r.text[:300]}")

        if isinstance(body, dict) and body.get("error"):
            raise ModelUnavailable(f"LM Studio error for {model!r}: {body['error']!r}")

        served = body.get("model")
        if assert_served and served != model:
            raise ModelSubstitution(
                f"Requested {model!r} but LM Studio SERVED {served!r}. This is the silent-"
                f"substitution confound — stopping immediately so no invalid conclusion is "
                f"drawn."
            )
        msg = (body.get("choices") or [{}])[0].get("message", {}) or {}
        content = str(msg.get("content", "") or "")
        reasoning = str(msg.get("reasoning_content", "") or "")
        finish = (body.get("choices") or [{}])[0].get("finish_reason")
        result = ChatResult(model, served, content, reasoning, finish, body)
        if result.empty and not allow_empty:
            raise EmptyResponse(f"{model!r} served but returned empty content.")
        return result


def vision_chat(model: str, image_path: str, user_text: str, system_prompt: str = "",
                **kw) -> ChatResult:
    """chat() with an image attached. Same guards apply."""
    with open(image_path, "rb") as f:
        data_url = "data:image/png;base64," + base64.b64encode(f.read()).decode()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]})
    return chat(model, messages, **kw)
