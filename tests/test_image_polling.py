"""Mock-'requests' coverage for image.py's ComfyUI polling loops
(_submit_and_poll / _submit_and_collect). These branches (HTTP errors,
cancel-mid-poll, connection-error grace period, corrupt-file, errored/empty
job, timeout) don't need a live GPU -- just controlled fake HTTP responses.
Run: venv/Scripts/python.exe tests/test_image_polling.py
"""
import os, sys, time, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path
from PIL import Image
# The ComfyUI polling loop moved out of image.py into comfy_client.py; this
# suite was always testing the transport, not the image editing around it.
import comfy_client as I

_TMP = Path(tempfile.mkdtemp(prefix="imgpoll_"))
RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))

class FakeResp:
    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code; self._data = data; self.text = text
    def json(self):
        return self._data

def _save_real_png(name="out.png"):
    import numpy as np
    p = I.OUTPUT_DIR_COMFY / name
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.random.rand(64, 64, 3) * 255).astype("uint8")
    Image.fromarray(arr).save(p)
    return str(p)

def _install(get_seq=None, post=None):
    calls = {"get": 0}
    # _submit_and_poll now runs a liveness probe first, because a WEDGED ComfyUI
    # (accepts the socket, never answers) otherwise costs the full read timeout on
    # every call -- ~5 min for one draw, reported to the user as mere slowness.
    # These tests are about the POLL loop, not the probe (tests/test_comfy_wedged.py
    # owns that), so present a healthy server. Stubbing the function rather than
    # answering /system_stats through fake_get keeps the job's request sequence AND
    # the faked time.time() call counts in these tests exactly as they were.
    I.server_healthy = lambda force=False: True
    def fake_post(url, json=None, timeout=None):
        return post
    def fake_get(url, timeout=None):
        i = calls["get"]; calls["get"] += 1
        seq = get_seq[min(i, len(get_seq)-1)]
        if isinstance(seq, Exception): raise seq
        return seq
    I.requests.post = fake_post
    I.requests.get = fake_get
    return calls


def test_http_error_on_submit():
    orig_post, orig_get = I.requests.post, I.requests.get
    try:
        _install(get_seq=[FakeResp(200, {})], post=FakeResp(500, {"error": "boom"}, "boom"))
        out = I._submit_and_poll(None, {}, timeout=5)
        check("submit_http_error_none", out is None)
    finally:
        I.requests.post, I.requests.get = orig_post, orig_get

def test_missing_prompt_id():
    orig_post, orig_get = I.requests.post, I.requests.get
    try:
        _install(get_seq=[FakeResp(200, {})], post=FakeResp(200, {}))
        out = I._submit_and_poll(None, {}, timeout=5)
        check("submit_missing_promptid_none", out is None)
    finally:
        I.requests.post, I.requests.get = orig_post, orig_get

def test_post_raises():
    orig_post, orig_get = I.requests.post, I.requests.get
    try:
        def bad_post(*a, **k): raise ConnectionError("no server")
        I.requests.post = bad_post
        out = I._submit_and_poll(None, {}, timeout=5)
        check("submit_post_raises_none", out is None)
    finally:
        I.requests.post, I.requests.get = orig_post, orig_get

def test_cancel_mid_poll():
    orig_post, orig_get = I.requests.post, I.requests.get
    try:
        class C:
            def is_cancelled(self): return True
        _install(get_seq=[FakeResp(200, {})], post=FakeResp(200, {"prompt_id": "abc"}))
        out = I._submit_and_poll(C(), {}, timeout=5)
        check("submit_cancel_none", out is None)
    finally:
        I.requests.post, I.requests.get = orig_post, orig_get

def test_saved_image_success():
    orig_post, orig_get = I.requests.post, I.requests.get
    try:
        p = _save_real_png("z_pollok.png")
        hist = {"abc": {"outputs": {"9": {"images": [{"filename": "z_pollok.png", "subfolder": ""}]}},
                        "status": {}}}
        _install(get_seq=[FakeResp(200, hist)], post=FakeResp(200, {"prompt_id": "abc"}))
        out = I._submit_and_poll(None, {}, timeout=5)
        check("submit_success_path", out == p, out)
    finally:
        I.requests.post, I.requests.get = orig_post, orig_get

def test_saved_audio_success():
    """Audio-producing jobs (SaveAudio, SaveAudioAdvanced) report their file
    under an "audio" history key, not "images" -- confirmed live: a real
    MiniMax Music3 render completed successfully server-side but the client
    reported "no output" because _poll_history only ever looked at
    node_out["images"]. Regression for that bug.
    """
    orig_post, orig_get = I.requests.post, I.requests.get
    try:
        p = I.OUTPUT_DIR_COMFY / "z_pollok_audio.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"RIFF....WAVEfmt ")  # content irrelevant, validate() is stubbed below
        hist = {"abc": {"outputs": {"9": {"audio": [{"filename": "z_pollok_audio.wav", "subfolder": ""}]}},
                        "status": {}}}
        _install(get_seq=[FakeResp(200, hist)], post=FakeResp(200, {"prompt_id": "abc"}))
        out = I._submit_and_poll(None, {}, timeout=5, validate=lambda path: True)
        check("audio_output_key_recognized", out == str(p), out)
    finally:
        I.requests.post, I.requests.get = orig_post, orig_get


def test_corrupt_reported_image():
    orig_post, orig_get = I.requests.post, I.requests.get
    try:
        badp = I.OUTPUT_DIR_COMFY / "z_corrupt.png"
        badp.parent.mkdir(parents=True, exist_ok=True)
        badp.write_bytes(b"not a real png")
        hist = {"abc": {"outputs": {"9": {"images": [{"filename": "z_corrupt.png", "subfolder": ""}]}},
                        "status": {}}}
        _install(get_seq=[FakeResp(200, hist)], post=FakeResp(200, {"prompt_id": "abc"}))
        out = I._submit_and_poll(None, {}, timeout=5)
        check("submit_corrupt_file_none", out is None)
    finally:
        I.requests.post, I.requests.get = orig_post, orig_get

def test_temp_file_skipped_then_real_wins():
    orig_post, orig_get = I.requests.post, I.requests.get
    try:
        real = _save_real_png("z_realwin.png")
        hist = {"abc": {"outputs": {
            "3": {"images": [{"filename": "ComfyUI_temp_0001.png", "subfolder": ""}]},
            "9": {"images": [{"filename": "z_realwin.png", "subfolder": ""}]}},
            "status": {}}}
        _install(get_seq=[FakeResp(200, hist)], post=FakeResp(200, {"prompt_id": "abc"}))
        out = I._submit_and_poll(None, {}, timeout=5)
        check("submit_temp_skipped", out == real, out)
    finally:
        I.requests.post, I.requests.get = orig_post, orig_get

def test_errored_job_no_output():
    orig_post, orig_get = I.requests.post, I.requests.get
    try:
        hist = {"abc": {"outputs": {}, "status": {"status_str": "error",
                 "messages": [["execution_error", "node blew up"]]}}}
        _install(get_seq=[FakeResp(200, hist)], post=FakeResp(200, {"prompt_id": "abc"}))
        out = I._submit_and_poll(None, {}, timeout=5)
        check("submit_errored_none", out is None)
    finally:
        I.requests.post, I.requests.get = orig_post, orig_get

def test_completed_no_output():
    orig_post, orig_get = I.requests.post, I.requests.get
    try:
        hist = {"abc": {"outputs": {}, "status": {"completed": True, "messages": []}}}
        _install(get_seq=[FakeResp(200, hist)], post=FakeResp(200, {"prompt_id": "abc"}))
        out = I._submit_and_poll(None, {}, timeout=5)
        check("submit_completed_empty_none", out is None)
    finally:
        I.requests.post, I.requests.get = orig_post, orig_get

def test_connection_error_grace_then_recovers():
    orig_post, orig_get = I.requests.post, I.requests.get
    try:
        p = _save_real_png("z_recov.png")
        hist = {"abc": {"outputs": {"9": {"images": [{"filename": "z_recov.png", "subfolder": ""}]}},
                        "status": {}}}
        seq = [I.requests.exceptions.ConnectionError("down"), FakeResp(200, hist)]
        _install(get_seq=seq, post=FakeResp(200, {"prompt_id": "abc"}))
        out = I._submit_and_poll(None, {}, timeout=10)
        check("submit_conn_recovers", out == p, out)
    finally:
        I.requests.post, I.requests.get = orig_post, orig_get

def test_connection_error_exceeds_grace():
    orig_post, orig_get = I.requests.post, I.requests.get
    try:
        exc = I.requests.exceptions.ConnectionError("down")
        _install(get_seq=[exc], post=FakeResp(200, {"prompt_id": "abc"}))
        # shrink grace window for the test
        orig_grace = None
        import image as Imod
        # monkeypatch time.time to fast-forward past the 60s grace on 2nd call
        real_time = time.time
        calls = {"n": 0}
        def fake_time():
            calls["n"] += 1
            return real_time() + (0 if calls["n"] < 3 else 61)
        time.time = fake_time
        try:
            out = I._submit_and_poll(None, {}, timeout=120)
            check("submit_conn_grace_exceeded_none", out is None)
        finally:
            time.time = real_time
    finally:
        I.requests.post, I.requests.get = orig_post, orig_get

def test_generic_polling_exception():
    orig_post, orig_get = I.requests.post, I.requests.get
    try:
        def bad_get(*a, **k): raise RuntimeError("weird")
        I.requests.post = lambda *a, **k: FakeResp(200, {"prompt_id": "abc"})
        real_time = time.time
        calls = {"n": 0}
        def fake_time():
            calls["n"] += 1
            return real_time() + (0 if calls["n"] < 2 else 999)
        I.requests.get = bad_get
        time.time = fake_time
        try:
            out = I._submit_and_poll(None, {}, timeout=5)
            check("submit_generic_poll_exc_none_or_timeout", out is None)
        finally:
            time.time = real_time
    finally:
        I.requests.post, I.requests.get = orig_post, orig_get

def test_timeout_expires():
    orig_post, orig_get = I.requests.post, I.requests.get
    try:
        hist = {"other": {}}  # our prompt_id never appears
        _install(get_seq=[FakeResp(200, hist)], post=FakeResp(200, {"prompt_id": "abc"}))
        real_time = time.time
        calls = {"n": 0}
        def fake_time():
            calls["n"] += 1
            return real_time() + (0 if calls["n"] < 2 else 999)
        time.time = fake_time
        try:
            out = I._submit_and_poll(None, {}, timeout=5)
            check("submit_timeout_none", out is None)
        finally:
            time.time = real_time
    finally:
        I.requests.post, I.requests.get = orig_post, orig_get


# ---- _submit_and_collect variants ----

def test_collect_http_error():
    orig_post, orig_get = I.requests.post, I.requests.get
    try:
        _install(get_seq=[FakeResp(200, {})], post=FakeResp(500, {}, "err"))
        out = I._submit_and_collect(None, {}, timeout=5)
        check("collect_http_error_none", out is None)
    finally:
        I.requests.post, I.requests.get = orig_post, orig_get

def test_collect_missing_prompt_id():
    orig_post, orig_get = I.requests.post, I.requests.get
    try:
        _install(get_seq=[FakeResp(200, {})], post=FakeResp(200, {}))
        out = I._submit_and_collect(None, {}, timeout=5)
        check("collect_missing_promptid_none", out is None)
    finally:
        I.requests.post, I.requests.get = orig_post, orig_get

def test_collect_post_raises():
    orig_post, orig_get = I.requests.post, I.requests.get
    try:
        def bad_post(*a, **k): raise ConnectionError("x")
        I.requests.post = bad_post
        out = I._submit_and_collect(None, {}, timeout=5)
        check("collect_post_raises_none", out is None)
    finally:
        I.requests.post, I.requests.get = orig_post, orig_get

def test_collect_cancel():
    orig_post, orig_get = I.requests.post, I.requests.get
    try:
        class C:
            def is_cancelled(self): return True
        _install(get_seq=[FakeResp(200, {})], post=FakeResp(200, {"prompt_id": "abc"}))
        out = I._submit_and_collect(C(), {}, timeout=5)
        check("collect_cancel_none", out is None)
    finally:
        I.requests.post, I.requests.get = orig_post, orig_get

def test_collect_success_multi_output():
    orig_post, orig_get = I.requests.post, I.requests.get
    try:
        p1 = _save_real_png("z_c1.png"); p2 = _save_real_png("z_c2.png")
        hist = {"abc": {"outputs": {
            "1": {"images": [{"filename": "z_c1.png", "subfolder": ""}]},
            "2": {"images": [{"filename": "z_c2.png", "subfolder": ""}]},
            "3": {"images": [{"filename": "ComfyUI_temp_1.png", "subfolder": ""}]},
        }, "status": {"completed": True}}}
        _install(get_seq=[FakeResp(200, hist)], post=FakeResp(200, {"prompt_id": "abc"}))
        out = I._submit_and_collect(None, {}, timeout=5)
        check("collect_multi_output", out == {"1": p1, "2": p2}, out)
    finally:
        I.requests.post, I.requests.get = orig_post, orig_get

def test_collect_errored_no_images():
    orig_post, orig_get = I.requests.post, I.requests.get
    try:
        hist = {"abc": {"outputs": {}, "status": {"status_str": "error"}}}
        _install(get_seq=[FakeResp(200, hist)], post=FakeResp(200, {"prompt_id": "abc"}))
        out = I._submit_and_collect(None, {}, timeout=5)
        check("collect_errored_none", out is None)
    finally:
        I.requests.post, I.requests.get = orig_post, orig_get

def test_collect_generic_exception_then_timeout():
    orig_post, orig_get = I.requests.post, I.requests.get
    try:
        def bad_get(*a, **k): raise RuntimeError("weird")
        I.requests.post = lambda *a, **k: FakeResp(200, {"prompt_id": "abc"})
        I.requests.get = bad_get
        real_time = time.time
        calls = {"n": 0}
        def fake_time():
            calls["n"] += 1
            return real_time() + (0 if calls["n"] < 2 else 999)
        time.time = fake_time
        try:
            out = I._submit_and_collect(None, {}, timeout=5)
            check("collect_generic_exc_none", out is None)
        finally:
            time.time = real_time
    finally:
        I.requests.post, I.requests.get = orig_post, orig_get


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print("  ERROR in " + fn.__name__ + ": " + type(e).__name__ + ": " + str(e))
    passed = sum(1 for _, c in RESULTS if c)
    print(f"\n{len(fns)-failed}/{len(fns)} functions, {passed}/{len(RESULTS)} checks passed")
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)
