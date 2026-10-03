"""Paths the app builds must be valid on Linux as well as Windows.

Bugs found 2026-10-03: the side venvs (venv_yue2, venv_mula, venv_diar,
venv_eraser, ComfyUI-qi21) were addressed only as Scripts\\python.exe, and the
ComfyUI model folder as r"~\\Documents\\ComfyUI" -- on Linux that is one
directory literally named "~\\Documents\\ComfyUI" under home, so video, songs,
covers and diarisation all read as "not installed" there.
"""
import ast
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import config  # noqa: E402

PROD = ["agent", "bot", "core", "gui", "imaging", "knowledge", "media", "research",
        "sandbox", "services", "voice"]


def test_venv_python_matches_the_os(tmp_path):
    want = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    assert config.venv_python(tmp_path / "v") == os.path.join(str(tmp_path / "v"), *want)


def test_no_backslash_home_paths_in_production_code():
    """r"~\\..." under expanduser is only right on Windows."""
    bad = []
    for d in PROD:
        for f in glob.glob(os.path.join(ROOT, d, "**", "*.py"), recursive=True):
            tree = ast.parse(open(f, encoding="utf-8").read())
            for n in ast.walk(tree):
                if (isinstance(n, ast.Constant) and isinstance(n.value, str)
                        and n.value.startswith("~\\")):
                    bad.append(f"{os.path.relpath(f, ROOT)}:{n.lineno} {n.value!r}")
    assert not bad, bad


def test_no_windows_only_venv_interpreters_in_production_code():
    """A Scripts/python.exe venv path must come with its POSIX twin."""
    bad = []
    for d in PROD:
        for f in glob.glob(os.path.join(ROOT, d, "**", "*.py"), recursive=True):
            tree = ast.parse(open(f, encoding="utf-8").read())
            for n in ast.walk(tree):
                if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "join":
                    lits = [a.value for a in n.args if isinstance(a, ast.Constant)]
                    if "Scripts" in lits and "python.exe" in lits:
                        bad.append(f"{os.path.relpath(f, ROOT)}:{n.lineno}")
    assert not bad, bad


def test_media_model_dirs_resolve_under_home(monkeypatch):
    monkeypatch.delenv("COMFY_BASE_DIR", raising=False)
    import music
    import video
    want = os.path.join(os.path.expanduser("~"), "Documents", "ComfyUI")
    assert music._models_dir() == want
    assert video._models_dir() == want
