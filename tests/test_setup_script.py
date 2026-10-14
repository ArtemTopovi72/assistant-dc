"""The one-command setup (scripts/setup.py) and its health check.

The parts that must hold on every machine: a re-run never overwrites the
user's .env, a missing required part makes the health check exit non-zero,
and the Windows wrapper turns PowerShell-style switches into setup flags.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_config_step_creates_env_once_and_never_overwrites(tmp_path):
    setup = _load("setup")
    (tmp_path / ".env.example").write_text("A=1\nB=2\n", encoding="utf-8")
    setup.STEPS.clear()
    setup.s_config(SimpleNamespace(root=tmp_path))
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "A=1\nB=2\n"

    (tmp_path / ".env").write_text("A=mine\n", encoding="utf-8")
    setup.s_config(SimpleNamespace(root=tmp_path))
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "A=mine\n"
    _, status, detail = setup.STEPS[-1]
    assert status == "ok" and "kept" in detail and "B" in detail, detail


def _health(*args, env=None):
    e = dict(os.environ, **(env or {}))
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "healthcheck.py"), *args],
                          capture_output=True, text=True, timeout=300, env=e,
                          encoding="utf-8", errors="replace")


def test_healthcheck_passes_on_present_parts():
    p = _health("--only", "config")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "READY" in p.stdout


def test_healthcheck_fails_when_the_chat_server_is_missing():
    # Port 9 (discard) has no HTTP server: lmstudio is a required part.
    p = _health("--only", "lmstudio", env={"LM_STUDIO_BASE": "http://127.0.0.1:9"})
    assert p.returncode == 1, p.stdout + p.stderr
    assert "FAIL" in p.stdout and "lmstudio" in p.stdout


def test_wrappers_call_setup_py_and_launcher():
    ps1 = (ROOT / "setup.ps1").read_text(encoding="utf-8")
    sh = (ROOT / "setup.sh").read_text(encoding="utf-8")
    assert r"scripts\setup.py" in ps1 and "exit $LASTEXITCODE" in ps1
    assert "scripts/setup.py" in sh and "set -euo pipefail" in sh
    assert "wget -qO-" in sh                  # images without curl can still get uv
    assert "launch_all.py" in (ROOT / "start.cmd").read_text(encoding="utf-8")
    assert "launch_all.py" in (ROOT / "start.sh").read_text(encoding="utf-8")


def test_deb_package_names():
    setup = _load("setup")
    assert setup._deb_package("libxcb-icccm.so.4") == "libxcb-icccm4"
    assert setup._deb_package("libxkbcommon-x11.so.0") == "libxkbcommon-x11-0"
    assert setup._deb_package("libGL.so.1") == "libgl1"


def test_qt_libs_step_reports_what_to_install():
    setup = _load("setup")
    if setup.IS_WIN:
        return
    setup.STEPS.clear()
    setup.s_qt_libs(SimpleNamespace(no_install=True, qt_missing_libs=lambda: ["libxcb-icccm.so.4"],
                                    apt_install=lambda pkgs, args: False))
    _, status, detail = setup.STEPS[-1]
    assert status == "warn" and "apt install libxcb-icccm4" in detail, detail
    # once installed the step re-checks and passes
    left = [["libxcb-icccm.so.4"], []]
    setup.s_qt_libs(SimpleNamespace(no_install=False, qt_missing_libs=lambda: left.pop(0),
                                    apt_install=lambda pkgs, args: True))
    assert setup.STEPS[-1][1] == "ok"


def test_apt_is_never_run_with_no_install():
    setup = _load("setup")
    assert setup._apt_install(["ffmpeg"], SimpleNamespace(no_install=True)) is False


def test_voice_hint_names_the_setup_for_downloadable_parts(tmp_path):
    hc = _load("healthcheck")
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    cfg = SimpleNamespace(WEIGHTS_PATH=tmp_path / "none.safetensors", VOCOS_DIR=tmp_path / "vocos", DC_REF_WAV=ref)
    r = hc.check_voice(cfg)
    text = " ".join(str(x) for x in vars(r).values()) if hasattr(r, "__dict__") else str(r)
    assert "setup script again" in text and "ASSISTANT_REF_WAV" not in text, text
    ref.unlink()
    r = hc.check_voice(cfg)
    text = " ".join(str(x) for x in vars(r).values()) if hasattr(r, "__dict__") else str(r)
    assert "ASSISTANT_REF_WAV" in text, text


def test_docker_hint_on_linux_names_the_group_or_the_service():
    hc = _load("healthcheck")
    cr = SimpleNamespace(docker_available=lambda timeout=0: False, docker_binary=lambda: "/usr/bin/docker")
    r = hc.check_docker(cr, "posix", lambda exe: (
        "permission denied while trying to connect to the Docker daemon socket"))
    assert "usermod -aG docker" in str(vars(r)), vars(r)
    r = hc.check_docker(cr, "posix", lambda exe: (
        "Cannot connect to the Docker daemon. Is the docker daemon running?"))
    assert "systemctl" in str(vars(r)) and "Docker Desktop" not in str(vars(r)), vars(r)
