"""Chat models that LM Studio cannot run, served by our own llama-server.

Muse Glimmer (architecture `muse-glimmer`) needs llama.cpp b10353+; LM Studio's
runtime 2.43.0 refuses it. The app's model lifecycle goes through two funnels --
lmstudio.ensure_exclusive (load) and lora_training.free_gpu (evict for a
render or training) -- and both call in here for a model listed in MODELS, so
renders still get the card to themselves and the model still comes back.

The CUDA 12.3 runtime must be on PATH for this build or it silently falls back
to the CPU (4.8 tok/s); it is added for the child process only.
"""
import logging
import os
import config as _cfg_env   # env_int/env_float: a bad .env value falls back, never crashes the import
import subprocess
import time
import urllib.request
from pathlib import Path

try:
    import config  # noqa: F401  (loads .env before the env reads below)
except Exception:
    pass

logger = logging.getLogger(__name__)

SERVER_EXE = os.getenv("LLAMA_SERVER_EXE", r"C:\llamacpp\build\llama-server.exe")
CUDA_BIN = os.getenv("LLAMA_CUDA_BIN", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin")
PORT = _cfg_env.env_int("LLAMA_PORT", 8091)  # 8081 is the local Telegram Bot API
_LOG = Path(__file__).resolve().parents[1] / "runtime" / "llama_server.log"

_GLIMMER_DIR = Path.home() / ".lmstudio" / "models" / "lmstudio-community" / "Muse-Glimmer-30B-GGUF"
# Empty by default: LM Studio runtime 2.44.0 loads Glimmer itself. Set
# LLAMA_BACKEND_GLIMMER=1 to serve it from our llama-server instead.
MODELS = {} if os.getenv("LLAMA_BACKEND_GLIMMER") != "1" else {
    "muse-glimmer-30b": {
        "gguf": _GLIMMER_DIR / "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
        "mmproj": _GLIMMER_DIR / "mmproj-Muse-Glimmer-30B-Q4_K_M.gguf",
        "ctx": _cfg_env.env_int("GLIMMER_CTX", 32768),
    },
}


def handles(model_id: str) -> bool:
    return (model_id or "") in MODELS


def base_url() -> str:
    return f"http://127.0.0.1:{PORT}"


def served_ids() -> list:
    try:
        import json
        with urllib.request.urlopen(base_url() + "/v1/models", timeout=3) as r:
            return [m["id"] for m in json.loads(r.read()).get("data", [])]
    except Exception:
        return []


def _pids() -> list:
    """llama-server processes listening on OUR port only. The same build also
    serves Mantella (Skyrim) on another port; that one is never touched."""
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True,
                             text=True, timeout=15).stdout
        pids = {int(p[-1]) for p in (l.split() for l in out.splitlines())
                if len(p) >= 5 and p[1].endswith(f":{PORT}") and p[3] == "LISTENING"}
        tl = subprocess.run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"],
                            capture_output=True, text=True, timeout=15).stdout
        mine = {int(l.split('","')[1]) for l in tl.splitlines() if l.startswith('"llama-server')}
        return sorted(pids & mine)
    except Exception:
        return []


def _under_tests() -> bool:
    # Same rule as lmstudio._refuse_under_tests: a suite must never kill or
    # start the operator's live model.
    return bool(os.getenv("F5_TEST_RUN")) and not os.getenv("LLAMA_ALLOW_IN_TESTS")


def stop() -> bool:
    """Kill every llama-server. True if one was running."""
    if _under_tests():
        return False
    pids = _pids()
    for p in pids:
        subprocess.run(["taskkill", "/PID", str(p), "/F"], capture_output=True, timeout=15)
    if pids:
        # The driver returns the memory a moment after the process exits.
        for _ in range(40):
            if not _pids():
                break
            time.sleep(0.25)
        logger.info("llama-server stopped")
    return bool(pids)


def start(model_id: str, timeout: float = 240.0) -> tuple:
    """Serve `model_id`. No-op if it already is. Returns (ok, message)."""
    spec = MODELS.get(model_id)
    if not spec:
        return False, f"{model_id} is not a llama-server model"
    if model_id in served_ids():
        return True, "already loaded"
    if _under_tests():
        return False, "refused under tests"
    stop()
    if not Path(SERVER_EXE).exists() or not Path(spec["gguf"]).exists():
        return False, f"missing {SERVER_EXE} or {spec['gguf']}"
    env = dict(os.environ)
    env["PATH"] = CUDA_BIN + os.pathsep + env.get("PATH", "")
    args = [SERVER_EXE, "-m", str(spec["gguf"]), "-ngl", "99", "-c", str(spec["ctx"]),
            "-np", "1", "--jinja", "--host", "127.0.0.1", "--port", str(PORT), "-a", model_id]
    if spec.get("mmproj") and Path(spec["mmproj"]).exists():
        args += ["--mmproj", str(spec["mmproj"])]
    _LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_LOG, "ab") as log:     # the child keeps its own handle
        proc = subprocess.Popen(args, env=env, stdout=log, stderr=log,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(base_url() + "/health", timeout=3) as r:
                if r.status == 200:
                    logger.info("llama-server serving %s (%.0fs)", model_id, time.time() - t0)
                    return True, "loaded"
        except Exception:
            pass
        if proc.poll() is not None:
            return False, f"llama-server exited; see {_LOG}"
        time.sleep(1.0)
    # Not left loading in the background: it would hold the card while the
    # caller falls back, and the next start() could not tell it was ours.
    proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    return False, f"llama-server did not come up in {timeout:.0f}s"


def unload_lmstudio_chat() -> None:
    """Free the card of LM Studio chat models, keeping the embedding model."""
    try:
        out = subprocess.run(["lms", "ps"], capture_output=True, text=True, timeout=30, shell=(os.name == "nt")).stdout
        for line in out.splitlines()[1:]:
            ident = line.split()[0] if line.split() else ""
            if ident and "embed" not in ident.lower() and "rerank" not in ident.lower() \
                    and ident != "IDENTIFIER":
                subprocess.run(["lms", "unload", ident], capture_output=True, text=True,
                               timeout=60, shell=(os.name == "nt"))
    except Exception as exc:
        logger.info("lms unload skipped: %s", exc)


_EMBED_CHECKED = [0.0]


def ensure_embedder(min_interval: float = 60.0) -> None:
    """Keep the embedding server up: BGE-M3 on the CPU-only llama-server at
    config.EMBED_BASE. Without it the knowledge base silently falls back to
    keyword-only search. A non-local EMBED_BASE is left alone."""
    if _under_tests():
        return
    import time as _t
    if _t.monotonic() - _EMBED_CHECKED[0] < min_interval:
        return
    _EMBED_CHECKED[0] = _t.monotonic()
    try:
        import urllib.error
        import urllib.parse
        import urllib.request
        base = config.EMBED_BASE.rstrip("/")
        try:
            urllib.request.urlopen(base + "/health", timeout=3)
            return
        except urllib.error.HTTPError:
            return          # 503 while the model loads: it IS up, don't spawn a second one
        except Exception:
            pass
        u = urllib.parse.urlparse(base)
        if u.hostname not in ("127.0.0.1", "localhost") or not os.path.exists(config.EMBED_CPU_SERVER):
            return
        log = open(os.path.join(os.path.dirname(config.EMBED_CPU_SERVER), "bge_cpu.log"), "ab")
        subprocess.Popen([config.EMBED_CPU_SERVER, "-m", config.EMBED_GGUF, "--embedding",
                          "--pooling", "cls", "-c", "8192", "-b", "8192", "-ub", "8192",
                          "--alias", config.EMBED_MODEL, "--host", "127.0.0.1",
                          "--port", str(u.port or 8096), "-t", "8"],
                         stdout=log, stderr=subprocess.STDOUT,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        logger.info("started CPU embedding server on %s", base)
    except Exception as exc:
        logger.info("embedder start skipped: %s", exc)
