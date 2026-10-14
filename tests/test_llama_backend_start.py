"""agent/llama_backend.start() does not leave a server that never came up.

Bug: on timeout the half-started llama-server kept running (holding the card
while the caller fell back), and the parent's log handle was never closed.
"""
import os
import subprocess
import sys
import textwrap
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.environ.setdefault("F5_TEST_RUN", "1")

import llama_backend as L  # noqa: E402


def test_a_server_that_never_answers_is_killed(tmp_path):
    fake = tmp_path / "server.py"
    fake.write_text(textwrap.dedent("""
        import sys, time, pathlib
        pathlib.Path(sys.argv[1]).write_text("up")
        time.sleep(60)
    """))
    flag = tmp_path / "up.txt"
    seen = {}
    real = subprocess.Popen

    class P(real):
        def __init__(self, args, **k):
            super().__init__([sys.executable, str(fake), str(flag)], **k)
            seen["p"] = self
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    from fakes import Over
    ctx = types.SimpleNamespace(deps=types.SimpleNamespace(
        subprocess=Over(subprocess, Popen=P), _under_tests=lambda: False, stop=lambda: False,
        served_ids=lambda: [], SERVER_EXE=sys.executable,
        PORT=9,                                     # nothing answers there
        _LOG=tmp_path / "llama.log", MODELS={"m": {"gguf": gguf, "ctx": 8}}))
    ok, msg = L.start("m", timeout=2, ctx=ctx)
    assert not ok and "did not come up" in msg
    assert seen["p"].returncode is not None        # killed and reaped
