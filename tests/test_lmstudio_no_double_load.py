"""A slow REST load must not be followed by a second load through the CLI.

Bringing an 18 GB chat model back after a render can outlast the REST call's
timeout (or the 30 s "is it served yet" wait). load_model_exclusive treated
that as a failure and ran `lms load` for the same model while the first load
was still reading the file: two full loads, two instances on one card, and
the reload took twice as long.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
import lmstudio as L


def _setup(monkeypatch, rest_result, served_after):
    calls = {"cli": 0, "rest": 0, "wait": []}
    monkeypatch.setattr(L, "_lms_unload_all", lambda: (True, "ok"))
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    monkeypatch.setattr(L, "resolve_virtual_model", lambda b, m: (None, False))

    def rest(*a, **k):
        calls["rest"] += 1
        return rest_result

    def wait(base, mid, timeout=None):
        calls["wait"].append(timeout)
        return served_after

    def run(cmd, **k):
        calls["cli"] += 1
        raise AssertionError("a second (CLI) load was started: %r" % (cmd,))
    monkeypatch.setattr(L, "_rest_load", rest)
    monkeypatch.setattr(L, "_wait_served", wait)
    monkeypatch.setattr(L.subprocess, "run", run)
    return calls


def test_rest_still_loading_waits_instead_of_cli(monkeypatch):
    calls = _setup(monkeypatch, (False, L._StillLoading("not served yet")), True)
    ok, msg = L.load_model_exclusive("http://x", "big-model")
    assert ok and calls["cli"] == 0, (ok, msg, calls)


def test_rest_still_loading_that_never_finishes_reports_failure(monkeypatch):
    calls = _setup(monkeypatch, (False, L._StillLoading("timeout")), False)
    ok, msg = L.load_model_exclusive("http://x", "big-model")
    assert not ok and calls["cli"] == 0 and "did not finish" in msg


def test_rest_endpoint_missing_still_falls_back_to_cli(monkeypatch):
    calls = _setup(monkeypatch, (False, "404 Unexpected endpoint"), True)

    class R:
        returncode = 0; stdout = "Model loaded"; stderr = ""

    def run(cmd, **k):
        calls["cli"] += 1
        return R()
    monkeypatch.setattr(L.subprocess, "run", run)
    ok, _ = L.load_model_exclusive("http://x", "big-model")
    assert ok and calls["cli"] == 1


def test_rest_read_timeout_is_classified_as_still_loading(monkeypatch):
    def post(*a, **k):
        raise requests.exceptions.ReadTimeout("read timed out")
    monkeypatch.setattr(L.requests, "post", post)
    ok, msg = L._rest_load("http://x", "big-model")
    assert not ok and isinstance(msg, L._StillLoading)


def test_rest_connection_error_is_a_plain_failure(monkeypatch):
    def post(*a, **k):
        raise requests.exceptions.ConnectionError("refused")
    monkeypatch.setattr(L.requests, "post", post)
    ok, msg = L._rest_load("http://x", "big-model")
    assert not ok and not isinstance(msg, L._StillLoading)
