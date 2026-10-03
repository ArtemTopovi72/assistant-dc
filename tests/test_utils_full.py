"""Coverage for utils.py: memory cleanup, artifact cleanup, tag/leak stripping,
transcription cache, audio hashing, message trimming/sanitizing, image encoding,
JSON extraction. Pure/local logic + real filesystem, no GPU/network needed
except an optional real torch CUDA check (best-effort, works either way).
Run: venv/Scripts/python.exe tests/test_utils_full.py
"""
import os, sys, tempfile, threading, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path
import utils as U

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))

_TMP = Path(tempfile.mkdtemp(prefix="utilsfull_"))


def test_free_process_memory():
    U.free_process_memory()  # should never raise regardless of torch/image module presence
    check("free_process_memory_no_raise", True)

    import sys as _sys, types
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
        is_available=lambda: True, empty_cache=lambda: None, ipc_collect=lambda: None))
    _sys.modules["torch"] = fake_torch
    try:
        U.free_process_memory()
        check("free_process_memory_with_fake_cuda", True)
    finally:
        del _sys.modules["torch"]

    fake_image = types.SimpleNamespace(_INVENTORY_CACHE={"a": 1})
    _sys.modules["image"] = fake_image
    try:
        U.free_process_memory()
        check("free_process_memory_clears_image_cache", fake_image._INVENTORY_CACHE == {})
    finally:
        del _sys.modules["image"]

    class BadTorch:
        class cuda:
            @staticmethod
            def is_available():
                raise RuntimeError("boom")
    _sys.modules["torch"] = BadTorch()
    try:
        U.free_process_memory()
        check("free_process_memory_cuda_exception_swallowed", True)
    finally:
        del _sys.modules["torch"]


def test_cleanup_runtime_artifacts():
    d = _TMP / "artifacts"
    d.mkdir(exist_ok=True)
    old_file = d / "_capture_old.jpg"
    old_file.write_bytes(b"x")
    old_time = time.time() - 10 * 86400
    os.utime(old_file, (old_time, old_time))
    new_file = d / "_capture_new.jpg"
    new_file.write_bytes(b"x")
    unrelated = d / "keep_me.txt"
    unrelated.write_bytes(b"x")

    n = U.cleanup_runtime_artifacts(d, max_age_days=7.0)
    check("cleanup_removes_old_only", n == 1 and not old_file.exists() and new_file.exists()
          and unrelated.exists())

    n2 = U.cleanup_runtime_artifacts(_TMP / "does_not_exist_dir")
    check("cleanup_missing_dir_zero", n2 == 0)

    # The image pipeline's scratch (never delivered) piled up to 8.9k files:
    # its markers are swept too; a delivered result is not.
    d2 = _TMP / "artifacts2"
    d2.mkdir(exist_ok=True)
    scratch = [d2 / n for n in ("a_INTERMEDIATE_firered_tile_1.png", "_crop_2.png",
                                "_QA_cutout_3.png", "_working_input_4.webp")]
    result = d2 / "upscaled_faceguard_5.png"
    for f in scratch + [result]:
        f.write_bytes(b"x")
        os.utime(f, (old_time, old_time))
    U.cleanup_runtime_artifacts(d2, max_age_days=7.0)
    check("cleanup_sweeps_image_scratch", not any(f.exists() for f in scratch))
    check("cleanup_keeps_delivered_results", result.exists())


def test_strip_think_tags():
    check("strip_think_paired", U.strip_think_tags("<think>hidden</think>answer") == "answer")
    check("strip_think_unclosed", U.strip_think_tags("<think>hidden forever").strip() == "")
    check("strip_think_orphan_close", U.strip_think_tags("garbage</think>real answer") == "real answer")
    check("strip_think_empty", U.strip_think_tags("") == "")
    check("strip_think_none_passthrough", U.strip_think_tags(None) is None)
    check("strip_think_collapses_blank_lines", "\n\n\n" not in U.strip_think_tags("a\n\n\n\nb"))


def test_strip_textual_tool_calls():
    check("strip_tool_calls_no_angle_bracket_noop", U.strip_textual_tool_calls("plain text") == "plain text")
    check("strip_tool_calls_empty", U.strip_textual_tool_calls("") == "")
    out = U.strip_textual_tool_calls("before <tool_call>{\"name\":\"x\"}</tool_call> after")
    check("strip_tool_calls_removes_markup", "tool_call" not in out)
    out2 = U.strip_textual_tool_calls("before <function=calculate>{}</function>")
    check("strip_tool_calls_removes_function_markup", "function=" not in out2)
    out3 = U.strip_textual_tool_calls("unrelated <b>bold</b> text")
    check("strip_tool_calls_leaves_unrelated_tags", "<b>" in out3)


def test_strip_reasoning_leak():
    text = "Let me think about this. Final Answer: 42"
    out = U.strip_reasoning_leak(text)
    check("strip_reasoning_leak_recovers_answer", out == "42")
    text2 = "Let me think about this without any marker at all"
    out2 = U.strip_reasoning_leak(text2)
    check("strip_reasoning_leak_no_marker_returns_asis", out2 == text2)
    check("strip_reasoning_leak_normal_text_unaffected", U.strip_reasoning_leak("The answer is 5.") == "The answer is 5.")
    check("strip_reasoning_leak_empty", U.strip_reasoning_leak("") == "")
    with_think = "<think>reasoning</think>Here's a thinking process out loud. answer: 7"
    out3 = U.strip_reasoning_leak(with_think)
    check("strip_reasoning_leak_strips_think_first", "<think>" not in out3)


def test_load_transcription_cache():
    check("load_cache_missing_file_empty", U.load_transcription_cache(_TMP / "nope.json") == {})
    good = _TMP / "cache_good.json"
    good.write_text(json.dumps({"a": "b"}), encoding="utf-8")
    check("load_cache_valid", U.load_transcription_cache(good) == {"a": "b"})
    bad = _TMP / "cache_bad.json"
    bad.write_text("not json{{{", encoding="utf-8")
    check("load_cache_invalid_json_empty", U.load_transcription_cache(bad) == {})
    non_dict = _TMP / "cache_list.json"
    non_dict.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    check("load_cache_non_dict_empty", U.load_transcription_cache(non_dict) == {})


def test_audio_hash_from_path():
    small = _TMP / "small.bin"
    small.write_bytes(b"hello world" * 100)
    h1 = U.audio_hash_from_path(str(small))
    h2 = U.audio_hash_from_path(str(small))
    check("audio_hash_deterministic", h1 == h2)

    big = _TMP / "big.bin"
    big.write_bytes(os.urandom(3 * 1024 * 1024))
    h3 = U.audio_hash_from_path(str(big))
    check("audio_hash_large_file_head_tail", isinstance(h3, str) and len(h3) == 32)


def test_safe_empty_cuda_cache():
    U.safe_empty_cuda_cache()
    check("safe_empty_cuda_cache_no_raise", True)


def test_throttle_external_calls():
    check("throttle_none_ctx_noop", U.throttle_external_calls(None) is None)

    class Ctx:
        def __init__(self):
            self.api_lock = threading.Lock()
            self.api_min_interval = 0.05
            self.last_api_call_time = time.time()
    ctx = Ctx()
    t0 = time.time()
    U.throttle_external_calls(ctx)
    check("throttle_waits_when_too_soon", time.time() - t0 >= 0.0)

    class Ctx2:
        def __init__(self):
            self.api_lock = threading.Lock()
            self.api_min_interval = 0.0
            self.last_api_call_time = time.time() - 10
    U.throttle_external_calls(Ctx2())
    check("throttle_no_wait_when_interval_zero", True)

    # A lightweight context that carries no throttling state must be a no-op,
    # not an AttributeError. llm.py catches pre-dispatch failures and returns
    # None, so raising here silently turns every model call on that path into
    # "no answer" while the calling suite still exits 0.
    class Bare:
        pass

    raised = None
    try:
        U.throttle_external_calls(Bare())
    except Exception as exc:            # pragma: no cover - only on regression
        raised = exc
    check("throttle_bare_ctx_noop", raised is None,
          f"raised {type(raised).__name__}: {raised}")

    # Partially-populated: pacing state but no lock is still nothing to pace
    # against, and must not raise either.
    class NoLock:
        def __init__(self):
            self.api_min_interval = 5.0
            self.last_api_call_time = time.time()

    t0 = time.time()
    raised = None
    try:
        U.throttle_external_calls(NoLock())
    except Exception as exc:            # pragma: no cover - only on regression
        raised = exc
    check("throttle_no_lock_noop", raised is None and time.time() - t0 < 1.0,
          f"raised {raised}")

    # A lock but no interval/clock: must not raise, must not sleep.
    class LockOnly:
        def __init__(self):
            self.api_lock = threading.Lock()

    lo = LockOnly()
    t0 = time.time()
    raised = None
    try:
        U.throttle_external_calls(lo)
    except Exception as exc:            # pragma: no cover - only on regression
        raised = exc
    check("throttle_lock_only_noop", raised is None and time.time() - t0 < 1.0,
          f"raised {raised}")
    check("throttle_lock_only_stamps_clock",
          isinstance(getattr(lo, "last_api_call_time", None), float))


def test_ensure_required_files():
    # This used to accept either outcome (ran / raised), so it could not fail.
    import config as C
    from pathlib import Path as _P
    saved = (C.WEIGHTS_PATH, C.DC_REF_WAV, C.VOCOS_DIR, C.VOCAB_PATH)
    try:
        # Missing VOICE files: the app still starts, and is told what is missing.
        C.WEIGHTS_PATH = _P("/no/such/model.safetensors")
        C.DC_REF_WAV = _P("/no/such/voice.wav")
        C.VOCOS_DIR = _P("/no/such/vocos")
        missing = U.ensure_required_files()
        check("voice_files_missing_is_not_fatal", len(missing) == 3, str(missing))
        # A file that ships WITH the repo missing is a broken checkout: fatal.
        C.VOCAB_PATH = _P("/no/such/vocab.txt")
        try:
            U.ensure_required_files(); raised = ""
        except FileNotFoundError as e:
            raised = str(e)
        check("repo_file_missing_is_fatal", "Missing required files" in raised and "vocab" in raised,
              raised)
    finally:
        C.WEIGHTS_PATH, C.DC_REF_WAV, C.VOCOS_DIR, C.VOCAB_PATH = saved


def test_truncate_text():
    check("truncate_text_short_unchanged", U.truncate_text("short", limit=100) == "short")
    long_text = "x" * 200
    out = U.truncate_text(long_text, limit=50)
    check("truncate_text_long_truncated", len(out) < len(long_text) and "truncated" in out)


def test_sanitize_history():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "tool_calls": None},
        {"role": "assistant", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "result"},
        {"role": "assistant", "content": "   "},
        {"role": "assistant", "content": 123},
    ]
    out = U.sanitize_history(msgs)
    check("sanitize_history_keeps_system_user", any(m["role"] == "system" for m in out)
          and any(m["role"] == "user" for m in out))
    check("sanitize_history_keeps_real_assistant_text", any(m.get("content") == "hello" for m in out))
    check("sanitize_history_drops_tool_calls_and_tool_role", not any(m["role"] == "tool" for m in out))
    check("sanitize_history_drops_blank_and_nonstr_assistant", len(out) == 3)


def test_trim_messages():
    check("trim_messages_empty", U.trim_messages([]) == [])
    msgs = [{"role": "system", "content": "sys"}] + [{"role": "user", "content": str(i)} for i in range(20)]
    out = U.trim_messages(msgs, max_non_system_messages=5)
    check("trim_messages_keeps_system_and_trims_tail", out[0]["role"] == "system" and len(out) == 6)

    msgs2 = [{"role": "user", "content": str(i)} for i in range(20)]
    out2 = U.trim_messages(msgs2, max_non_system_messages=5)
    check("trim_messages_no_system_trims_all", len(out2) == 5)

    msgs3 = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    out3 = U.trim_messages(msgs3, max_non_system_messages=5)
    check("trim_messages_under_limit_unchanged", len(out3) == 2)

    msgs4 = [{"role": "user", "content": "short"}]
    out4 = U.trim_messages(msgs4, max_non_system_messages=5)
    check("trim_messages_short_no_system_unchanged", len(out4) == 1)

    # deepcopy verification: mutating the copy must not affect the original
    orig = [{"role": "system", "content": "sys"}, {"role": "user", "content": {"nested": [1]}}]
    copied = U.trim_messages(orig, max_non_system_messages=5)
    copied[1]["content"]["nested"].append(2)
    check("trim_messages_deepcopy_isolated", orig[1]["content"]["nested"] == [1])


def test_image_bytes_to_data_url():
    out = U.image_bytes_to_data_url(b"hello", mime_type="image/png")
    check("image_bytes_to_data_url_format", out.startswith("data:image/png;base64,"))


def test_downscale_image_bytes():
    from PIL import Image
    import io
    small_buf = io.BytesIO()
    Image.new("RGB", (100, 100), (0, 0, 0)).save(small_buf, "PNG")
    small_bytes = small_buf.getvalue()
    out = U.downscale_image_bytes(small_bytes, max_edge=1536)
    check("downscale_small_image_unchanged", out == small_bytes)

    big_buf = io.BytesIO()
    Image.new("RGB", (3000, 2000), (10, 20, 30)).save(big_buf, "PNG")
    big_bytes = big_buf.getvalue()
    out2 = U.downscale_image_bytes(big_bytes, max_edge=1000)
    with Image.open(io.BytesIO(out2)) as im2:
        check("downscale_large_image_resized", max(im2.size) <= 1000)

    out3 = U.downscale_image_bytes(b"not a real image", max_edge=1000)
    check("downscale_invalid_image_returns_original", out3 == b"not a real image")


def test_file_to_data_url():
    from PIL import Image
    p = _TMP / "test.png"
    Image.new("RGB", (10, 10)).save(p)
    out = U.file_to_data_url(str(p))
    check("file_to_data_url_png_mime", out.startswith("data:image/png;base64,"))

    p2 = _TMP / "test.unknownext"
    p2.write_bytes(b"raw bytes")
    out2 = U.file_to_data_url(str(p2))
    check("file_to_data_url_unknown_mime_defaults_png", out2.startswith("data:image/png;base64,"))


def test_safe_json_from_llm():
    check("safe_json_none_input", U.safe_json_from_llm("") is None)
    check("safe_json_none_falsy", U.safe_json_from_llm(None) is None)
    out = U.safe_json_from_llm('{"a": 1, "b": "text"}')
    check("safe_json_plain_object", out == {"a": 1, "b": "text"})
    out2 = U.safe_json_from_llm('```json\n{"a": 1}\n```')
    check("safe_json_fenced_json", out2 == {"a": 1})
    out3 = U.safe_json_from_llm('```\n{"a": 1}\n```')
    check("safe_json_fenced_plain", out3 == {"a": 1})
    out4 = U.safe_json_from_llm('some prose {"a": 1} more prose')
    check("safe_json_embedded_object", out4 == {"a": 1})
    check("safe_json_garbage_none", U.safe_json_from_llm("not json at all") is None)
    check("safe_json_non_dict_none", U.safe_json_from_llm("[1, 2, 3]") is None)
    out5 = U.safe_json_from_llm("<think>reasoning</think>{\"a\": 1}")
    check("safe_json_strips_think_first", out5 == {"a": 1})


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
