"""agent/llama_backend.start() does not leave a server that never came up.

Bug: on timeout the half-started llama-server kept running (holding the card
while the caller fell back), and the parent's log handle was never closed.
"""
import os
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import llama_backend as L  # noqa: E402


def test_a_server_that_never_answers_is_killed(tmp_path, monkeypatch):
    fake = tmp_path / "server.py"
    fake.write_text(textwrap.dedent("""
        import sys, time, pathlib
        pathlib.Path(sys.argv[1]).write_text("up")
        time.sleep(60)
    """))
    flag = tmp_path / "up.txt"
    seen = {}
    import subprocess
    real = subprocess.Popen

    class P(real):
        def __init__(self, args, **k):
            super().__init__([sys.executable, str(fake), str(flag)], **k)
            seen["p"] = self
    monkeypatch.setattr(L.subprocess, "Popen", P)
    monkeypatch.setattr(L, "_under_tests", lambda: False)
    monkeypatch.setattr(L, "stop", lambda: False)
    monkeypatch.setattr(L, "served_ids", lambda: [])
    monkeypatch.setattr(L, "SERVER_EXE", sys.executable)
    monkeypatch.setattr(L, "PORT", 9)               # nothing answers there
    monkeypatch.setattr(L, "_LOG", tmp_path / "llama.log")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    monkeypatch.setattr(L, "MODELS", {"m": {"gguf": gguf, "ctx": 8}})
    ok, msg = L.start("m", timeout=2)
    assert not ok and "did not come up" in msg
    assert seen["p"].returncode is not None        # killed and reaped
