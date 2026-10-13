"""A slow REST load must not be followed by a second load through the CLI.

Bringing an 18 GB chat model back after a render can outlast the REST call's
timeout (or the 30 s "is it served yet" wait). load_model_exclusive treated
that as a failure and ran `lms load` for the same model while the first load
was still reading the file: two full loads, two instances on one card, and
the reload took twice as long.
"""
import os
import sys
import time
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
import lmstudio as L


def _setup(rest_result, served_after):
    calls = {"cli": 0, "rest": 0, "wait": []}

    def rest(*a, **k):
        calls["rest"] += 1
        return rest_result

    def wait(base, mid, timeout=None, ctx=None):
        calls["wait"].append(timeout)
        return served_after

    def run(cmd, **k):
        calls["cli"] += 1
        raise AssertionError("a second (CLI) load was started: %r" % (cmd,))
    ctx = NS(deps=NS(_lms_unload_all=lambda ctx=None: (True, "ok"),
                     time=NS(sleep=lambda s: None, time=time.time, monotonic=time.monotonic),
                     resolve_virtual_model=lambda b, m, ctx=None: (None, False),
                     _rest_load=rest, _wait_served=wait, subprocess=NS(run=run)))
    return calls, ctx


def test_rest_still_loading_waits_instead_of_cli():
    calls, ctx = _setup((False, L._StillLoading("not served yet")), True)
    ok, msg = L.load_model_exclusive("http://x", "big-model", ctx=ctx)
    assert ok and calls["cli"] == 0, (ok, msg, calls)


def test_rest_still_loading_that_never_finishes_reports_failure():
    calls, ctx = _setup((False, L._StillLoading("timeout")), False)
    ok, msg = L.load_model_exclusive("http://x", "big-model", ctx=ctx)
    assert not ok and calls["cli"] == 0 and "did not finish" in msg


def test_rest_endpoint_missing_still_falls_back_to_cli():
    calls, ctx = _setup((False, "404 Unexpected endpoint"), True)

    class R:
        returncode = 0; stdout = "Model loaded"; stderr = ""

    def run(cmd, **k):
        calls["cli"] += 1
        return R()
    ctx.deps.subprocess = NS(run=run)
    ok, _ = L.load_model_exclusive("http://x", "big-model", ctx=ctx)
    assert ok and calls["cli"] == 1


def test_rest_read_timeout_is_classified_as_still_loading():
    def post(*a, **k):
        raise requests.exceptions.ReadTimeout("read timed out")
    ok, msg = L._rest_load("http://x", "big-model", ctx=NS(deps=NS(requests=NS(post=post))))
    assert not ok and isinstance(msg, L._StillLoading)


def test_rest_connection_error_is_a_plain_failure():
    def post(*a, **k):
        raise requests.exceptions.ConnectionError("refused")
    ok, msg = L._rest_load("http://x", "big-model", ctx=NS(deps=NS(requests=NS(post=post))))
    assert not ok and not isinstance(msg, L._StillLoading)
