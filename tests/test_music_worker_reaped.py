"""media.music.run_gpu_worker reaps a killed worker before releasing the GPU.

Bug: on Stop the worker was kill()ed and the GPU slot released at once, with
the process still dying (and holding its VRAM) -- the next queued job started
on a card that was not free yet, and on POSIX the child stayed a zombie.
"""
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import pytest  # noqa: E402

import music  # noqa: E402


class _Ctx:
    def __init__(self):
        self.cancel = threading.Event()

    def is_cancelled(self):
        return self.cancel.is_set()


def test_cancel_reaps_before_returning(monkeypatch):
    seen = {}
    import subprocess
    orig = subprocess.Popen

    class P(orig):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            seen["p"] = self
    monkeypatch.setattr(subprocess, "Popen", P)
    ctx = _Ctx()
    threading.Timer(1.0, ctx.cancel.set).start()
    with pytest.raises(music.MusicUnavailable):
        music.run_gpu_worker(ctx, sys.executable, "", {}, "t", timeout=60,
                             cmd=[sys.executable, "-c", "import time; time.sleep(30)"])
    assert seen["p"].returncode is not None      # reaped, not left running/zombie


def test_timeout_kills_and_reports(monkeypatch):
    t0 = time.time()
    ok, _tail = music.run_gpu_worker(None, sys.executable, "", {}, "t", timeout=1,
                                     cmd=[sys.executable, "-c", "import time; time.sleep(30)"])
    assert ok is False and time.time() - t0 < 10


def test_render_time_log_survives_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(music, "_eta_path", lambda: tmp_path / "eta.json")
    (tmp_path / "eta.json").write_text('[{"preset": "a", "asked": 60, "wa')   # torn write
    music.record_render_time("a", 60, 120.0)
    music.record_render_time("a", 30, 50.0)
    import json
    rows = json.loads((tmp_path / "eta.json").read_text())
    assert [r["wall"] for r in rows] == [120.0, 50.0]
    # one malformed row does not hide the rows after it
    rows.insert(0, {"preset": "a"})
    (tmp_path / "eta.json").write_text(json.dumps(rows))
    got = music._eta_runs().get("a", [])
    assert (60, 120.0, music._STEPS_REF) in got and (30, 50.0, music._STEPS_REF) in got


def test_yue2_weights_may_be_sharded(tmp_path):
    d = tmp_path / "YuE2-3B"
    d.mkdir()
    assert not music._has_weights(str(d))
    (d / "config.json").write_text("{}")
    (d / "model-00001-of-00002.safetensors").write_bytes(b"x")
    assert music._has_weights(str(d))
    (d / "model-00001-of-00002.safetensors").write_bytes(b"")
    assert not music._has_weights(str(d))       # an empty stub is not a model


def test_yue2_python_matches_the_venv_layout():
    tail = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    assert music.YUE2_PYTHON.endswith(os.path.join("venv_yue2", *tail))
