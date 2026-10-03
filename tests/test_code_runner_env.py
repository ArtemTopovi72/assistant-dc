"""Scripts run by run_code do not inherit the app's credentials.

.env is loaded into os.environ; on the host backend a model-written script
could read VK_TOKEN or TG_API_HASH straight from its environment.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import code_runner as R  # noqa: E402


def test_secrets_are_stripped_but_the_rest_is_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("VK_TOKEN", "vk-secret")
    monkeypatch.setenv("TG_API_HASH", "hash-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "key-secret")
    monkeypatch.setenv("HARMLESS_SETTING", "kept")
    res = R._spawn([sys.executable, "-c",
                    "import os; print(sorted(k for k in os.environ if 'SECRET' in os.environ[k]"
                    " or k == 'HARMLESS_SETTING'))"],
                   cwd=tmp_path, timeout=30)
    assert res.ok, res.output
    assert "HARMLESS_SETTING" in res.output
    assert "VK_TOKEN" not in res.output and "TG_API_HASH" not in res.output
    assert "OPENAI_API_KEY" not in res.output
