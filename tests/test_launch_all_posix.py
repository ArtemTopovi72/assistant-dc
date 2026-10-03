"""scripts/launch_all.py starts ComfyUI and LM Studio on Linux too.

Bug: every spawn passed creationflags=..., which POSIX Popen rejects with
ValueError for any non-zero value. The callers swallow exceptions, so on
Linux launch_all silently started nothing -- and the venv/log paths were
Windows-only (Scripts\\python.exe, r"~\\Documents\\ComfyUI").
"""
import importlib.util
import os
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX spawn path")


def _load(monkeypatch, base):
    monkeypatch.setenv("COMFY_BASE_DIR", str(base))
    monkeypatch.delenv("COMFY_SRC", raising=False)
    monkeypatch.delenv("COMFY_PY", raising=False)
    monkeypatch.delenv("COMFY_LOG", raising=False)
    spec = importlib.util.spec_from_file_location(
        "launch_all_posix", os.path.join(ROOT, "scripts", "launch_all.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_tree(base):
    tree = base / "ComfyUI-0.38.2"
    (tree / ".venv" / "bin").mkdir(parents=True)
    os.symlink(sys.executable, tree / ".venv" / "bin" / "python")
    marker = base / "started.txt"
    (tree / "main.py").write_text(
        "import sys, pathlib\n"
        f"pathlib.Path({str(marker)!r}).write_text(' '.join(sys.argv[1:]))\n"
        "print('fake comfy up')\n")
    return tree, marker


def test_paths_are_native(tmp_path, monkeypatch):
    tree, _ = _fake_tree(tmp_path)
    m = _load(monkeypatch, tmp_path)
    assert m.COMFY_SRC == str(tree)
    assert m.COMFY_PY == str(tree / ".venv" / "bin" / "python")
    assert m.COMFY_LOG == str(tmp_path / "comfyui_launch.log")
    assert "\\" not in m.LMS_CLI and m.LMS_CLI.endswith(os.path.join(".lmstudio", "bin", "lms"))


def test_comfy_source_really_starts(tmp_path, monkeypatch):
    _, marker = _fake_tree(tmp_path)
    m = _load(monkeypatch, tmp_path)
    assert m._spawn_comfy_source() is True
    for _ in range(100):
        if marker.exists() and (tmp_path / "comfyui_launch.log").read_text():
            break
        time.sleep(0.1)
    args = marker.read_text()
    assert "--base-directory" in args and str(tmp_path) in args
    assert "fake comfy up" in (tmp_path / "comfyui_launch.log").read_text()


def test_spawn_detached_binary(tmp_path, monkeypatch):
    m = _load(monkeypatch, tmp_path)
    out = tmp_path / "ran.txt"
    script = tmp_path / "app.sh"
    script.write_text(f"#!/bin/sh\necho ok > {out}\n")
    script.chmod(0o755)
    assert m._spawn(str(script)) is True
    for _ in range(50):
        if out.exists():
            break
        time.sleep(0.1)
    assert out.read_text().strip() == "ok"
