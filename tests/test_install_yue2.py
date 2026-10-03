"""scripts/install_yue2.py: one command makes exactly what yue2_available()
checks, and a second run skips what is done (owner 10-03)."""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("F5_TEST_RUN", "1")

import install_yue2 as I  # noqa: E402


def _fake_machine(monkeypatch, cuda_ok=True):
    root = tempfile.mkdtemp(prefix="yue2inst_")
    monkeypatch.setattr(I, "VENV", os.path.join(root, "venv_yue2"))
    monkeypatch.setattr(I, "MODELS", os.path.join(root, "models_ext"))
    ran = []
    state = {"pkgs": False}

    def check_call(cmd):
        ran.append(cmd)
        if cmd[-2:] == ["venv", I.VENV]:
            os.makedirs(os.path.dirname(I.venv_python()), exist_ok=True)
            open(I.venv_python(), "w").close()
        elif I.YUE2_PKG in cmd:
            state["pkgs"] = True
        elif "snapshot_download" in " ".join(cmd):
            dest = cmd[-1]
            open(os.path.join(dest, "config.json"), "w").close()
            open(os.path.join(dest, "model.safetensors"), "w").close()

    monkeypatch.setattr(I.subprocess, "check_call", check_call)
    monkeypatch.setattr(I.subprocess, "call", lambda *a, **k: 0 if state["pkgs"] and cuda_ok else 1)
    return ran


def test_one_run_makes_what_the_app_checks(monkeypatch):
    ran = _fake_machine(monkeypatch)
    assert I.main(["--python", "py -3.12"]) == 0
    flat = [" ".join(c) for c in ran]
    assert flat[0].startswith("py -3.12 -m venv")
    torch_i = next(i for i, c in enumerate(flat) if I.TORCH in c)
    pkg_i = next(i for i, c in enumerate(flat) if I.YUE2_PKG in c)
    assert torch_i < pkg_i and "download.pytorch.org/whl/cu128" in flat[torch_i]   # CUDA torch first
    assert "@1dc1c50" in I.YUE2_PKG           # the commit yue2_render.py patches
    for name in ("YuE2-3B", "YuE2-Vae"):
        assert I.has_weights(os.path.join(I.MODELS, name))


def test_a_second_run_skips_what_is_done(monkeypatch):
    ran = _fake_machine(monkeypatch)
    I.main([])
    ran.clear()
    assert I.main([]) == 0 and ran == []


def test_no_cuda_card_is_said(monkeypatch, capsys):
    _fake_machine(monkeypatch, cuda_ok=False)
    assert I.main(["--no-models"]) == 1
    assert "no CUDA" in capsys.readouterr().out


def test_paths_match_the_apps_check():
    import config
    import music
    assert music.YUE2_PYTHON == config.venv_python(os.path.join(ROOT, "venv_yue2"))
    assert I.VENV == os.path.join(ROOT, "venv_yue2") and I.MODELS == os.path.join(ROOT, "models_ext")
