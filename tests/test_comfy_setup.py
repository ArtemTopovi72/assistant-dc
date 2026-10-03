"""scripts/comfy_setup.py: pinned custom nodes and model downloads.

Exercised with real git (local repos stand in for GitHub) and a faked
Hugging Face hub; nothing leaves the machine.
"""
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import comfy_setup as C  # noqa: E402


def _git(*a, cwd):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


def _upstream(path: Path) -> list:
    """A repo with two commits; returns their shas, oldest first."""
    path.mkdir()
    _git("init", "-q", cwd=path)
    _git("config", "user.email", "t@t", cwd=path)
    _git("config", "user.name", "t", cwd=path)
    _git("config", "uploadpack.allowReachableSHA1InWant", "true", cwd=path)
    shas = []
    for i in range(2):
        (path / "f.txt").write_text(str(i))
        _git("add", ".", cwd=path)
        _git("commit", "-qm", f"c{i}", cwd=path)
        shas.append(_git("rev-parse", "HEAD", cwd=path))
    return shas


@pytest.fixture
def base(tmp_path, monkeypatch):
    b = tmp_path / "ComfyUI"
    monkeypatch.setenv("COMFY_BASE_DIR", str(b))
    return b


def test_nodes_are_pinned_and_rerun_is_a_no_op(tmp_path, base, monkeypatch):
    up = tmp_path / "up"
    old, new = _upstream(up)
    monkeypatch.setattr(C, "NODES", [("NodeA", up.as_uri(), old)])
    st, detail = C.install_nodes("uv", lambda *_: None)
    d = base / "custom_nodes" / "NodeA"
    assert st == "ok", detail
    assert _git("rev-parse", "HEAD", cwd=d) == old      # the pin, not upstream's tip
    st, detail = C.install_nodes("uv", lambda *_: None)
    assert st == "ok" and "added" not in detail
    # a new pin moves an existing checkout
    monkeypatch.setattr(C, "NODES", [("NodeA", up.as_uri(), new)])
    assert C.install_nodes("uv", lambda *_: None)[0] == "ok"
    assert _git("rev-parse", "HEAD", cwd=d) == new


def test_local_edits_are_never_overwritten(tmp_path, base, monkeypatch):
    up = tmp_path / "up"
    old, new = _upstream(up)
    monkeypatch.setattr(C, "NODES", [("NodeA", up.as_uri(), old)])
    C.install_nodes("uv", lambda *_: None)
    d = base / "custom_nodes" / "NodeA"
    (d / "f.txt").write_text("mine")
    monkeypatch.setattr(C, "NODES", [("NodeA", up.as_uri(), new)])
    st, detail = C.install_nodes("uv", lambda *_: None)
    assert st == "warn" and "local changes" in detail
    assert (d / "f.txt").read_text() == "mine"


def test_staged_file_is_not_installed(base):
    name = C.MODELS[0][2]
    stage = base / "models" / ".download" / "sub"
    stage.mkdir(parents=True)
    (stage / name).write_bytes(b"x")
    assert any(m[2] == name for m in C.missing_models())
    real = base / "models" / C.MODELS[0][1]
    real.mkdir(parents=True)
    (real / name).write_bytes(b"x")
    assert not any(m[2] == name for m in C.missing_models())


def test_empty_file_counts_as_missing(base):
    sub, name = C.MODELS[0][1], C.MODELS[0][2]
    (base / "models" / sub).mkdir(parents=True)
    (base / "models" / sub / name).write_bytes(b"")
    assert any(m[2] == name for m in C.missing_models())


def _fake_hub(monkeypatch, files: dict, fail_repos=()):
    hub = types.ModuleType("huggingface_hub")

    def list_repo_files(repo):
        if repo in fail_repos:
            raise OSError("404")
        return files.get(repo, [])

    def hf_hub_download(repo_id, filename, local_dir):
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"weights")
        return str(p)
    hub.list_repo_files = list_repo_files
    hub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(C, "_LISTINGS", {})


def test_download_falls_through_repos_and_lands_in_place(base, monkeypatch):
    m = ("image", "vae", "a.safetensors", ["dead/repo", "good/repo"], ["a_alt.safetensors"], 0.001)
    monkeypatch.setattr(C, "MODELS", [m])
    _fake_hub(monkeypatch, {"good/repo": ["split_files/vae/a_alt.safetensors"]},
              fail_repos={"dead/repo"})
    st, detail = C.download_models(("image",), lambda *_: None)
    assert st == "ok", detail
    assert (base / "models" / "vae" / "a.safetensors").read_bytes() == b"weights"
    assert C.missing_models(("image",)) == []
    assert C.download_models(("image",), lambda *_: None)[0] == "ok"   # rerun: nothing to do


def test_unfindable_file_is_a_warning_not_a_crash(base, monkeypatch):
    m = ("music", "x", "nowhere.safetensors", ["r/r"], [], 0.001)
    monkeypatch.setattr(C, "MODELS", [m])
    _fake_hub(monkeypatch, {"r/r": ["other.bin"]})
    st, detail = C.download_models(("music",), lambda *_: None)
    assert st == "warn" and "nowhere.safetensors" in detail


def test_not_enough_disk_is_reported_before_downloading(base, monkeypatch):
    m = ("video", "x", "huge.safetensors", ["r/r"], [], 10 ** 6)
    monkeypatch.setattr(C, "MODELS", [m])
    _fake_hub(monkeypatch, {})
    st, detail = C.download_models(("video",), lambda *_: None)
    assert st == "warn" and "GB" in detail and "--media" in detail
    assert not (base / "models" / "x").exists()


def test_media_filter(base, monkeypatch):
    monkeypatch.setattr(C, "MODELS", [("image", "a", "i", [], [], 1), ("video", "b", "v", [], [], 1)])
    assert [m[2] for m in C.missing_models(("video",))] == ["v"]
