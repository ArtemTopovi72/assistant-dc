"""A malformed or empty numeric setting must not stop the app from importing.

Every knob in config.py (and several module-level constants elsewhere) is read
at import time. A bare int(os.getenv(...)) turned `IDEOGRAM_CFG=abc` or an empty
`TG_WORKERS=` in .env into a ValueError at import: the app did not start, and
the traceback named a line, not the setting.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BAD = {
    "IDEOGRAM_CFG": "abc", "STARTUP_DIALOG_TIMEOUT_S": "", "SANDBOX_TOOL_ROUNDS": "x",
    "TG_WORKERS": "", "LM_MIN_CONTEXT": "lots", "CONTEXT_MASK_AT": "?",
    "COMFY_MAX_CONCURRENT": " ", "SANDBOX_MAX_UNPACK_BYTES": "big",
}


def test_bad_values_fall_back_to_defaults():
    code = ("import config, lmstudio, context_v2, code_sandbox, tg_bot;"
            "print(config.IDEOGRAM_CFG, config.STARTUP_DIALOG_TIMEOUT_S, "
            "config.SANDBOX_TOOL_ROUNDS, tg_bot._MAX_CONSUMERS, "
            "lmstudio.MIN_CONTEXT_TOKENS, context_v2.MASK_AT, "
            "code_sandbox.MAX_UNPACK_BYTES)")
    env = dict(os.environ, **BAD)
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=180)
    assert p.returncode == 0, p.stderr[-2000:]
    vals = p.stdout.split()
    assert vals == ["7.0", "30", "16", "3", "40960", "0.4", str(2 * 1024 ** 3)], vals


def test_env_helpers_trim_and_accept_good_values(monkeypatch):
    import config
    monkeypatch.setenv("X_TEST_INT", " 12 ")
    monkeypatch.setenv("X_TEST_FLOAT", "2.5")
    assert config.env_int("X_TEST_INT", 1) == 12
    assert config.env_float("X_TEST_FLOAT", 1.0) == 2.5
    monkeypatch.setenv("X_TEST_INT", "1.5")
    assert config.env_int("X_TEST_INT", 7) == 7
