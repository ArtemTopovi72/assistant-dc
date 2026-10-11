"""imaging/ocr_reader: a worker that does not report ready is killed.

When ocr_worker.py prints anything other than its "ready" line, the constructor
raises and must also kill and reap the process: _get() marks OCR failed and
never retries, so a leaked worker would keep the EasyOCR model loaded until
the app exits.
"""
import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import pytest  # noqa: E402

import ocr_reader as O  # noqa: E402


def test_a_worker_that_never_reports_ready_is_killed(tmp_path, monkeypatch):
    fake = tmp_path / "worker.py"
    fake.write_text(textwrap.dedent("""
        import sys, time
        print("Traceback: something went wrong", flush=True)
        time.sleep(60)
    """))
    seen = {}
    real = subprocess.Popen

    class P(real):
        def __init__(self, args, **k):
            super().__init__([sys.executable, str(fake)], **k)
            seen["p"] = self
    monkeypatch.setattr(subprocess, "Popen", P)
    with pytest.raises(RuntimeError, match="did not start"):
        O._WorkerReader()
    assert seen["p"].returncode is not None        # killed and reaped
