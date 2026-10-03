"""Running the agent's own code, inside real isolation or not at all.

Two separate jobs here, and only the second one is interesting.

The first is the brakes. A local 26B writing code emits infinite loops, runaway
prints and scripts that block on stdin; that is the normal output distribution,
not a bug to fix later, and the complaint that started this work was exactly
that ("goes mad in recursion and falls over"). So: a hard deadline, the whole
process tree killed when it expires, no stdin to wait on, and clipped output.

The second is containment, and it is the reason this module has a backend at
all. Path checking (code_sandbox) keeps the agent's TOOLS inside a directory.
It cannot keep arbitrary code there — a script runs as whoever started it and
can touch anything that user can. Running model-written code from a Telegram
chat directly on the host is a straight path to a wrecked machine, so it is not
offered:

  * DOCKER backend (used whenever the engine is reachable) — no network, a
    read-only root filesystem, every capability dropped, no privilege
    escalation, memory / pid / cpu ceilings, and only the one sandbox directory
    mounted writable. A script inside cannot see the host filesystem, the other
    users' sandboxes, the LAN, or the GPU.

  * HOST backend — the brakes but no walls. Refused by default and enabled only
    per user by an administrator who has read what that means.

The failure mode that matters is silent degradation: if the engine is down, the
answer is "code execution is unavailable", never a quiet fall back to running it
on the host. Everything else in this file exists to keep that one rule true.

Dependencies install into a venv belonging to the SANDBOX, never the project's.
That is not tidiness: this project pins a CUDA torch build, and a resolver that
decides to reinstall torch hands the machine a CPU wheel instead, silently
costing it the GPU stack.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("assistant.code_runner")

DEFAULT_TIMEOUT = 60
MAX_TIMEOUT = 300

# Output handed back to the model. A runaway print loop produces megabytes,
# which would blow the whole per-request budget and say nothing the first and
# last few lines do not.
MAX_OUTPUT_CHARS = 8_000
_HEAD_CHARS = 5_000

AGENT_DIR = ".agent"

# Resource ceilings for the container. Generous enough to unpack and rewrite a
# modpack, small enough that a fork bomb or a runaway allocation is contained.
# The house image ships pillow/numpy/opencv/matplotlib/pandas/... (see
# docker/sandbox/Dockerfile); python:3.12-slim is the bare fallback the build
# is made FROM. ensure_image() builds the house image when it is missing.
HOUSE_IMAGE = "assistant-sandbox:latest"
CONTAINER_IMAGE = os.getenv("SANDBOX_IMAGE", HOUSE_IMAGE)
SANDBOX_DOCKERFILE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docker", "sandbox")
CONTAINER_MEMORY = os.getenv("SANDBOX_MEMORY", "2g")
CONTAINER_PIDS = os.getenv("SANDBOX_PIDS", "256")
CONTAINER_CPUS = os.getenv("SANDBOX_CPUS", "2")

_DOCKER_PATHS = (
    "docker",
    r"C:\Program Files\Docker\Docker\resources\bin\docker.exe",
)


@dataclass
class RunResult:
    ok: bool
    code: int
    output: str
    seconds: float
    script: str = ""
    backend: str = ""

    def as_tool_result(self) -> str:
        head = (f"exit={self.code} in {self.seconds:.1f}s"
                if self.ok else
                f"FAILED exit={self.code} after {self.seconds:.1f}s")
        if self.backend:
            head += f" [{self.backend}]"
        return f"{head}\n{self.output}".strip()


def _clip(text: str) -> str:
    """Keep the head AND the tail: the traceback is at the end, and whatever
    caused it is usually at the beginning."""
    text = text or ""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    tail = MAX_OUTPUT_CHARS - _HEAD_CHARS
    return (text[:_HEAD_CHARS]
            + f"\n… [{len(text) - MAX_OUTPUT_CHARS} chars cut] …\n"
            + text[-tail:])


# -- backend selection ------------------------------------------------------

def docker_binary() -> str | None:
    for cand in _DOCKER_PATHS:
        found = shutil.which(cand) or (cand if os.path.exists(cand) else None)
        if found:
            return found
    return None


def docker_available(timeout: int = 12) -> bool:
    """Is the ENGINE reachable, not merely installed?

    Docker Desktop leaves docker.exe on disk with the engine stopped, and in
    that state every run fails. Checked with `docker info`, which talks to the
    daemon, rather than `--version`, which does not. An engine in Windows-container
    mode is reachable but cannot run the Linux sandbox image, so it counts as absent.
    """
    exe = docker_binary()
    if not exe:
        return False
    try:
        proc = subprocess.run([exe, "info", "--format", "{{.ServerVersion}} {{.OSType}}"],
                              capture_output=True, text=True, timeout=timeout)
        return proc.returncode == 0 and (proc.stdout or "").strip().endswith("linux")
    except Exception:
        return False


_DESKTOP_PATHS = (
    r"C:\Program Files\Docker\Docker\Docker Desktop.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Docker\Docker\Docker Desktop.exe"),
)


def desktop_binary() -> str | None:
    for cand in _DESKTOP_PATHS:
        if cand and os.path.exists(cand):
            return cand
    return None


def ensure_docker(wait_s: float = 90.0, log=None) -> bool:
    """Start Docker Desktop if the engine is down; True when it answers.

    Live 2026-09-14: «собери коллаж скриптом» -- the bot found every bridge
    and then had to say "Docker не запущен", because Docker Desktop was
    installed and healthy but simply not launched. Code execution is part of
    the app, so the app starts its engine the way it starts LM Studio and
    ComfyUI. Measured: the engine answers ~5 s after the exe is spawned.

    The exe is spawned detached and silently; if it is not installed or the
    engine never answers within `wait_s`, this returns False and run_code
    keeps its honest "engine is not running" error.
    """
    if docker_available(timeout=6):
        return True
    exe = desktop_binary()
    if not exe:
        if log: log("Docker Desktop is not installed; code execution stays off")
        return False
    try:
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen([exe], creationflags=flags, close_fds=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    except Exception as exc:
        if log: log(f"Docker Desktop could not be started: {exc}")
        return False
    if log: log("Docker Desktop started; waiting for the engine")
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if docker_available(timeout=6):
            if log: log("Docker engine is up")
            return True
        time.sleep(3)
    if log: log(f"Docker engine did not answer within {wait_s:.0f}s")
    return False


def image_present(image: str = None, timeout: int = 15) -> bool:
    exe = docker_binary()
    if not exe:
        return False
    try:
        proc = subprocess.run([exe, "image", "inspect", image or CONTAINER_IMAGE,
                               "--format", "{{.Id}}"],
                              capture_output=True, text=True, timeout=timeout,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return proc.returncode == 0
    except Exception:
        return False


_image_build_lock = threading.Lock()
_DOCKERFILE_LABEL = "assistant.dockerfile"


def dockerfile_digest() -> str:
    """A short hash of the house Dockerfile (and the helpers it copies in)."""
    import hashlib
    h = hashlib.sha1()
    for name in ("Dockerfile", "assistant_tools.py"):
        p = os.path.join(SANDBOX_DOCKERFILE_DIR, name)
        if os.path.exists(p):
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()[:12]


def image_stale(timeout: int = 15) -> bool:
    """The house image was built from an older Dockerfile (2026-09-18: graphviz
    was added for flowcharts and the image was never rebuilt -- ensure_image
    only knew "missing", not "outdated")."""
    exe = docker_binary()
    if not exe or CONTAINER_IMAGE != HOUSE_IMAGE:
        return False
    try:
        proc = subprocess.run([exe, "image", "inspect", HOUSE_IMAGE, "--format",
                               "{{index .Config.Labels \"" + _DOCKERFILE_LABEL + "\"}}"],
                              capture_output=True, text=True, timeout=timeout,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        return False
    if proc.returncode != 0:
        return False
    return proc.stdout.strip() != dockerfile_digest()


def ensure_image(log=None, timeout: int = 1800) -> bool:
    """Build the house sandbox image if it is not there yet (once, ~2-4 min).

    The user's ask, 2026-09-14: "собрать докер образ с предустановленными
    либами" -- so that a collage or a chart runs on the first script instead
    of after a minute of pip inside a 64 MB tmpfs. Only the house image is
    built here; a custom SANDBOX_IMAGE from .env is the operator's own.
    """
    if CONTAINER_IMAGE != HOUSE_IMAGE:
        return image_present()
    stale = image_stale()
    if image_present() and not stale:
        return True
    exe = docker_binary()
    if not exe or not os.path.exists(os.path.join(SANDBOX_DOCKERFILE_DIR, "Dockerfile")):
        return image_present()
    with _image_build_lock:
        if image_present() and not image_stale():
            return True
        if log: log(f"Building the sandbox image {HOUSE_IMAGE} "
                    + ("(Dockerfile changed, a few minutes)" if stale else "(first run, a few minutes)"))
        try:
            proc = subprocess.run([exe, "build", "-t", HOUSE_IMAGE,
                                   "--label", f"{_DOCKERFILE_LABEL}={dockerfile_digest()}",
                                   SANDBOX_DOCKERFILE_DIR],
                                  capture_output=True, text=True, timeout=timeout,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception as exc:
            if log: log(f"sandbox image build failed: {exc}")
            return False
        ok = proc.returncode == 0
        if log: log("sandbox image ready" if ok else
                    f"sandbox image build failed: {(proc.stderr or proc.stdout or '')[-600:]}")
        return ok


def backend_status() -> tuple[str, str]:
    """(backend, human explanation) — for the admin panel and the tool errors."""
    if docker_available():
        return "docker", (f"Isolated: {CONTAINER_IMAGE}, no network, read-only "
                          f"root, {CONTAINER_MEMORY} RAM, {CONTAINER_PIDS} pids.")
    if docker_binary():
        return "", ("Docker is installed but the engine is not running. Start "
                    "Docker Desktop; code execution stays off until it answers.")
    return "", ("No isolated backend available. Start Docker Desktop, "
                "or have an administrator enable host execution for this user.")


# -- public API -------------------------------------------------------------

def run_python(sandbox, code: str, timeout: int = DEFAULT_TIMEOUT,
               allow_host: bool = False) -> RunResult:
    """Write the script into the sandbox and run it under the best backend.

    `allow_host` is the administrator's explicit decision, not a fallback this
    function may take on its own.
    """
    if not (code or "").strip():
        return RunResult(False, -1, "No code given.", 0.0)
    timeout = max(1, min(int(timeout or DEFAULT_TIMEOUT), MAX_TIMEOUT))
    scripts = sandbox.root / AGENT_DIR
    scripts.mkdir(parents=True, exist_ok=True)
    rel = f"{AGENT_DIR}/run_{int(time.time() * 1000)}.py"
    (sandbox.root / rel).write_text(code, encoding="utf-8")

    # The engine may still be booting (launch_all spawns Docker Desktop and
    # does not wait); a request that lands in that window waits here instead
    # of failing with "engine is not running" on the user's first script.
    if docker_available() or ensure_docker(wait_s=60):
        ensure_image()
        # A HOME the script can write. The root filesystem is read-only, so any
        # library that keeps a dotfile fails on first use: matplotlib prints
        # "mkdir -p failed for path /root/.config/matplotlib ... Read-only file
        # system" before every single plot. Nothing breaks, but the model reads
        # that as an error and starts working around a problem it does not
        # have. Pointed at the sandbox, which is the one writable mount.
        (sandbox.root / AGENT_DIR / "home").mkdir(parents=True, exist_ok=True)
        res = _run_in_container(sandbox, ["python", "-u", rel], timeout,
                                network=False,
                                env={"HOME": f"/work/{AGENT_DIR}/home",
                                     "XDG_CACHE_HOME": f"/work/{AGENT_DIR}/home/.cache",
                                     "MPLCONFIGDIR": f"/work/{AGENT_DIR}/home/.mpl"})
    elif allow_host:
        res = _spawn([sys.executable, "-u", str(sandbox.root / rel)],
                     cwd=sandbox.root, timeout=timeout)
        res.backend = "host (UNISOLATED)"
    else:
        _, why = backend_status()
        return RunResult(False, -1,
                         f"Code execution is unavailable: {why}", 0.0, script=rel)
    res.script = rel
    # A missing import is the one failure with an obvious next move, and the
    # model was not making it. Measured: asked to chart a CSV, it hit
    # ModuleNotFoundError for matplotlib and -- rather than calling
    # install_packages, which it had -- computed the averages by hand and
    # presented THOSE as the answer. A tool error that names the cure turns a
    # dead end into one more round; the same fix as unpack_archive's "does not
    # exist" carrying the real filename.
    # Installing from INSIDE the script cannot work and never will: script runs
    # are given no network on purpose. Measured: told twice that a module was
    # missing, the model's third attempt was subprocess pip install rather than
    # the install_packages tool it had all along -- and when that failed too it
    # told the user the chart had been saved. Naming the one path that works is
    # cheaper than any amount of description.
    if not res.ok and _PIP_IN_SCRIPT_RE.search(code or ""):
        res.output += ("\n[a script run has NO network, so pip cannot work from "
                       "inside the code. Installing is a separate tool: call "
                       "install_packages with the package names, then run the "
                       "script again unchanged.]")
    _missing = _missing_module(res.output)
    _local = _missing and any(
        (sandbox.root / p).exists() for p in (_missing + ".py", _missing))
    if _local and not res.ok:
        res.output += ("\n['" + _missing + "' is a file in the working folder, "
                       "not a package -- do NOT install it. Check the name and "
                       "that it is at the top level of the folder.]")
    elif _missing and not res.ok:
        res.output += ("\n[the sandbox has no '" + _missing + "' module. Call "
                       "install_packages with packages=['" + _missing + "'] "
                       "and run this again. Do NOT work around it by hand and "
                       "do NOT tell the user it cannot be done.]")
    return res


def install(sandbox, packages, timeout: int = 300,
            allow_host: bool = False) -> RunResult:
    """pip install into the SANDBOX's venv.

    This is the one operation that needs the network, so it is the one run with
    networking enabled — narrowly, for this call only, with the same ceilings.
    Ordinary script runs stay on `--network none`.
    """
    names = [str(p).strip() for p in (packages or []) if str(p).strip()]
    if not names:
        return RunResult(False, -1, "No package names given.", 0.0)
    bad = [n for n in names if n.startswith("-")]
    if bad:
        # `--target /elsewhere`, `--index-url http://…` and friends. This tool
        # takes package names; anything option-shaped is a mistake or an escape.
        return RunResult(False, -1,
                         f"Refusing option-shaped arguments: {bad}. Package "
                         f"names only.", 0.0)
    if docker_available() or ensure_docker(wait_s=60):
        # pip unpacks into TMPDIR, and the container's /tmp is a 64 MB tmpfs --
        # which is RAM, and nowhere near enough. Measured: `install_packages
        # matplotlib` died with "[Errno 28] No space left on device" after
        # downloading all 41 MB of wheels, because numpy + pillow + matplotlib
        # unpack to several times that. So the sandbox advertised an install
        # tool that could not install the most ordinary package anyone would
        # ask for, and the model, told only "no space", answered the user with
        # a hand-computed table instead of a chart.
        #
        # Staged on the MOUNTED sandbox instead: real disk, no RAM cost, and it
        # is cleaned up with everything else. /tmp stays small for script runs.
        (sandbox.root / AGENT_DIR / "tmp").mkdir(parents=True, exist_ok=True)
        return _run_in_container(
            sandbox,
            ["python", "-m", "pip", "install", "--no-input",
             "--target", f"{AGENT_DIR}/site-packages", *names],
            timeout, network=True,
            env={"TMPDIR": f"/work/{AGENT_DIR}/tmp",
                 "PIP_CACHE_DIR": f"/work/{AGENT_DIR}/tmp/pipcache"})
    if not allow_host:
        _, why = backend_status()
        return RunResult(False, -1, f"Installing is unavailable: {why}", 0.0)
    target = sandbox.root / AGENT_DIR / "site-packages"
    target.mkdir(parents=True, exist_ok=True)
    res = _spawn([sys.executable, "-m", "pip", "install", "--no-input",
                  "--target", str(target), *names],
                 cwd=sandbox.root, timeout=timeout)
    res.backend = "host (UNISOLATED)"
    return res


_MISSING_MODULE_RE = re.compile(
    r"ModuleNotFoundError: No module named ['\"]([A-Za-z_][A-Za-z_0-9]*)")


_PIP_IN_SCRIPT_RE = re.compile(r"(pip[^\n]{0,40}install|ensurepip)",
                               re.IGNORECASE)


def _missing_module(output: str) -> str:
    """The import that failed, if that is why the run failed."""
    m = _MISSING_MODULE_RE.search(output or "")
    return m.group(1) if m else ""


# -- backends ---------------------------------------------------------------

def _run_in_container(sandbox, argv, timeout: int, *, network: bool,
                      env: dict | None = None) -> RunResult:
    exe = docker_binary()
    root = str(sandbox.root)
    # Named, so a timeout can stop the CONTAINER. Killing the docker client
    # (all _kill_tree reaches) leaves a `--rm` container running: three
    # `while True` scripts from a bench sat at 99% CPU each for 20+ minutes
    # after their runs had been reported "killed after 120s" (2026-09-24).
    import uuid as _uuid
    name = f"sbx_{_uuid.uuid4().hex[:12]}"
    cmd = [
        exe, "run", "--rm", "--name", name,
        "--network", "bridge" if network else "none",
        "--memory", CONTAINER_MEMORY, "--memory-swap", CONTAINER_MEMORY,
        "--pids-limit", CONTAINER_PIDS, "--cpus", CONTAINER_CPUS,
        # The container's own filesystem is immutable; only the sandbox and a
        # small tmpfs are writable, so a script cannot install itself anywhere
        # that survives the run.
        "--read-only",
        "--tmpfs", "/tmp:rw,size=64m,exec",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "-v", f"{root}:/work",
        "-w", "/work",
        # /work first: the script lives in .agent/, so without it the
        # project's own modules are invisible ("import shop" -> not found,
        # and the hint then told the model to pip-install a stranger's
        # package of that name).
        "-e", "PYTHONPATH=/work:/work/" + AGENT_DIR + "/site-packages",
        "-e", "PYTHONUTF8=1", "-e", "PYTHONIOENCODING=utf-8",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
    ]
    for _k, _v in (env or {}).items():
        cmd += ["-e", f"{_k}={_v}"]
    cmd += [
        CONTAINER_IMAGE, *argv,
    ]
    res = _spawn(cmd, cwd=sandbox.root, timeout=timeout)
    if not res.ok:
        # Harmless when the container already exited (--rm removed it).
        try:
            subprocess.run([exe, "rm", "-f", name], capture_output=True, timeout=30)
        except Exception:
            logger.warning("could not stop container %s after its timeout", name)
    res.backend = "docker"
    if not res.ok and "Unable to find image" in (res.output or ""):
        res.output += (f"\n[the image is being pulled on first use; run it "
                       f"again once `docker pull {CONTAINER_IMAGE}` finishes]")
    return res


def _spawn(cmd, cwd, timeout: int) -> RunResult:
    """One subprocess, one deadline, no stdin, whole tree killed on expiry."""
    t0 = time.monotonic()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Same reason as the container's PYTHONPATH: the project root, not the
    # script's .agent/ folder, is where the user's modules are.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(cwd)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=env,
            stdin=subprocess.DEVNULL,      # never block on input that cannot come
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=creationflags,
        )
    except OSError as exc:
        return RunResult(False, -1, f"Could not start: {exc}", time.monotonic() - t0)

    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out, _ = proc.communicate(timeout=15)
        except Exception:
            out = ""
        return RunResult(
            False, -9,
            _clip((out or "") + f"\n[killed after {timeout}s — it did not finish. "
                                f"Endless loops and waiting for input are the "
                                f"usual causes.]"),
            time.monotonic() - t0)
    return RunResult(proc.returncode == 0, proc.returncode, _clip(out),
                     time.monotonic() - t0)


def _kill_tree(proc) -> None:
    """Kill the child AND its children.

    proc.kill() alone leaves grandchildren running: a script that spawned pip,
    or another interpreter, keeps holding the CPU after the deadline was
    supposed to have ended it.
    """
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=15)
            return
        except Exception:
            logger.debug("taskkill failed; falling back to kill()", exc_info=True)
    try:
        proc.kill()
    except Exception:
        logger.debug("could not kill the run", exc_info=True)
