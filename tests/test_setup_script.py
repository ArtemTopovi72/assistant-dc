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


def test_config_step_creates_env_once_and_never_overwrites(monkeypatch, tmp_path):
    setup = _load("setup")
    (tmp_path / ".env.example").write_text("A=1\nB=2\n", encoding="utf-8")
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    setup.STEPS.clear()
    setup.s_config(SimpleNamespace())
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "A=1\nB=2\n"

    (tmp_path / ".env").write_text("A=mine\n", encoding="utf-8")
    setup.s_config(SimpleNamespace())
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "A=mine\n"
    name, status, detail = setup.STEPS[-1]
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
    assert "launch_all.py" in (ROOT / "start.cmd").read_text(encoding="utf-8")
    assert "launch_all.py" in (ROOT / "start.sh").read_text(encoding="utf-8")
