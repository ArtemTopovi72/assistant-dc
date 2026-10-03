"""Coverage for lmstudio.py: virtual-model YAML resolution, exclusive-load
orchestration, CLI/HTTP reload strategy, config building. Uses the REAL live
LM Studio HTTP API for read-only calls (fetch_models/fetch_model/
loaded_model_ids — safe, no state change) and stubs `subprocess.run` for every
`lms` CLI invocation (load/unload would actually disrupt the running model, so
those are never executed for real).
Run: venv/Scripts/python.exe tests/test_lmstudio_module.py
"""
import os, sys, tempfile, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path
import lmstudio as L
from config import LM_STUDIO_BASE

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))

_TMP = Path(tempfile.mkdtemp(prefix="lmstudio_"))


class _Patches:
    def __init__(self, **kw):
        self.kw = kw; self.orig = {}
    def __enter__(self):
        for k, v in self.kw.items():
            self.orig[k] = getattr(L, k)
            setattr(L, k, v)
        return self
    def __exit__(self, *a):
        for k, v in self.orig.items():
            setattr(L, k, v)


class FakeRun:
    def __init__(self, returncode=0, stdout="OK", stderr="", raises=None):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self._raises = raises
    def __call__(self, *a, **k):
        if self._raises:
            raise self._raises
        return self


TimeoutExpired = __import__("subprocess").TimeoutExpired


class FakeSubprocess:
    TimeoutExpired = TimeoutExpired
    def __init__(self, run_result=None):
        self._run_result = run_result or FakeRun()
    def run(self, *a, **k):
        return self._run_result(*a, **k) if callable(self._run_result) else self._run_result


def test_virtual_yaml_path_and_parse():
    check("virtual_yaml_no_slash_none", L._virtual_yaml_path("noSlashHere") is None)
    check("virtual_yaml_has_at_none", L._virtual_yaml_path("pub/name@q8") is None)
    check("virtual_yaml_missing_file_none", L._virtual_yaml_path("nonexistent-pub-xyz/nonexistent-name-xyz") is None)

    yaml_path = _TMP / "model.yaml"
    yaml_path.write_text(
        "base: Pub/Folder/Folder-Q8_0.gguf\n"
        "metadataOverrides:\n"
        "  reasoning: false\n"
        "customFields:\n"
        "  - key: enableThinking\n"
        "    defaultValue: false\n",
        encoding="utf-8")
    out = L._parse_virtual_yaml(yaml_path)
    check("parse_virtual_yaml_base", out["base"] == "Pub/Folder/Folder-Q8_0.gguf")
    check("parse_virtual_yaml_reasoning_false", out["reasoning"] is False)
    check("parse_virtual_yaml_enable_thinking_false", out["enable_thinking"] is False)

    # force the fallback line-parser path by making PyYAML import fail
    import builtins
    real_import = builtins.__import__
    def fail_yaml(name, *a, **k):
        if name == "yaml":
            raise ImportError("no yaml")
        return real_import(name, *a, **k)
    builtins.__import__ = fail_yaml
    try:
        out2 = L._parse_virtual_yaml(yaml_path)
        check("parse_virtual_yaml_fallback_parser_base", out2["base"] == "Pub/Folder/Folder-Q8_0.gguf")
        check("parse_virtual_yaml_fallback_parser_reasoning", out2["reasoning"] is False)
    finally:
        builtins.__import__ = real_import

    minimal_yaml = _TMP / "minimal.yaml"
    minimal_yaml.write_text("base: X/Y/Y-Q4_0.gguf\n", encoding="utf-8")
    out3 = L._parse_virtual_yaml(minimal_yaml)
    check("parse_virtual_yaml_no_reasoning_key_none", out3["reasoning"] is None)


def test_base_to_raw_key():
    with _Patches(fetch_models=lambda base_url: [{"id": "folder@q8_0"}, {"id": "other@q4_0"}]):
        key, quant = L._base_to_raw_key("http://x", "Pub/Folder/Folder-Q8_0.gguf")
        check("base_to_raw_key_derived_match", key == "folder@q8_0" and quant == "q8_0")
    with _Patches(fetch_models=lambda base_url: [{"id": "folder@q4_k_m"}]):
        key2, quant2 = L._base_to_raw_key("http://x", "Pub/Folder/Folder-Q8_0.gguf")
        check("base_to_raw_key_catalog_fallback_match", key2 == "folder@q4_k_m")
    with _Patches(fetch_models=lambda base_url: []):
        key3, quant3 = L._base_to_raw_key("http://x", "Pub/Folder/Folder-Q8_0.gguf")
        check("base_to_raw_key_no_catalog_uses_derivation", key3 == "folder@q8_0")
    check("base_to_raw_key_empty_base_none", L._base_to_raw_key("http://x", "") == (None, None))
    with _Patches(fetch_models=lambda base_url: []):
        key4, quant4 = L._base_to_raw_key("http://x", "Pub/NoQuantFolder/NoQuantFolder.gguf")
        check("base_to_raw_key_no_quant_suffix", quant4 == "" and key4 == "noquantfolder")


def test_resolve_virtual_model():
    check("resolve_virtual_not_virtual_none", L.resolve_virtual_model("http://x", "not/a/virtual@q8") == (None, False))
    check("resolve_virtual_missing_none", L.resolve_virtual_model("http://x", "no-such-pub/no-such-name") == (None, False))

    with _Patches(_virtual_yaml_path=lambda mid: _TMP / "model.yaml",
                  _parse_virtual_yaml=lambda p: {"base": "Pub/F/F-Q8_0.gguf", "reasoning": False, "enable_thinking": None},
                  fetch_models=lambda base_url: [{"id": "f@q8_0"}]):
        raw, no_think = L.resolve_virtual_model("http://x", "Pub/Virtual")
        check("resolve_virtual_success", raw == "f@q8_0" and no_think is True)
        check("virtual_requires_no_think_true", L.virtual_requires_no_think("Pub/Virtual"))

    with _Patches(_virtual_yaml_path=lambda mid: _TMP / "model.yaml",
                  _parse_virtual_yaml=lambda p: (_ for _ in ()).throw(RuntimeError("boom"))):
        check("resolve_virtual_parse_exception_none", L.resolve_virtual_model("http://x", "Pub/Virtual2") == (None, False))

    with _Patches(_virtual_yaml_path=lambda mid: _TMP / "model.yaml",
                  _parse_virtual_yaml=lambda p: {"base": "", "reasoning": None, "enable_thinking": None}):
        check("resolve_virtual_no_raw_key_none", L.resolve_virtual_model("http://x", "Pub/Virtual3") == (None, False))

    check("virtual_requires_no_think_unknown_false", not L.virtual_requires_no_think("never-resolved"))


def test_effective_model_id():
    with _Patches(resolve_virtual_model=lambda base_url, mid: ("raw@q8", True)):
        check("effective_model_id_uses_raw", L.effective_model_id("http://x", "virt") == "raw@q8")
    with _Patches(resolve_virtual_model=lambda base_url, mid: (None, False)):
        check("effective_model_id_passthrough", L.effective_model_id("http://x", "plain") == "plain")


def test_fetch_models_real_and_failure():
    out = L.fetch_models(LM_STUDIO_BASE)
    check("fetch_models_real_returns_list", isinstance(out, list))
    out2 = L.fetch_models("http://127.0.0.1:1")  # unreachable port
    check("fetch_models_unreachable_empty", out2 == [])


def test_fetch_model_real_and_fallback():
    models = L.fetch_models(LM_STUDIO_BASE)
    if models:
        mid = models[0]["id"]
        out = L.fetch_model(LM_STUDIO_BASE, mid)
        check("fetch_model_real_found", out.get("id") == mid)
    else:
        check("fetch_model_real_skipped_no_models", True)
    out2 = L.fetch_model(LM_STUDIO_BASE, "definitely-nonexistent-model-xyz")
    check("fetch_model_not_found_empty", out2 == {})

    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))):
        with _Patches(fetch_models=lambda base_url: [{"id": "m1"}]):
            out3 = L.fetch_model("http://x", "m1")
            check("fetch_model_http_fails_falls_back_to_list", out3 == {"id": "m1"})


def test_loaded_model_ids_real():
    out = L.loaded_model_ids(LM_STUDIO_BASE)
    check("loaded_model_ids_real_is_list", isinstance(out, list))


def test_lms_unload_all():
    with _Patches(subprocess=FakeSubprocess(FakeRun(returncode=0))):
        ok, msg = L._lms_unload_all()
        check("lms_unload_all_success", ok is True)
    with _Patches(subprocess=FakeSubprocess(FakeRun(returncode=1, stderr="failed"))):
        ok2, msg2 = L._lms_unload_all()
        check("lms_unload_all_failure", ok2 is False and "failed" in msg2)
    with _Patches(subprocess=FakeSubprocess(FakeRun(raises=FileNotFoundError()))):
        ok3, msg3 = L._lms_unload_all()
        check("lms_unload_all_not_found", ok3 is False)
    with _Patches(subprocess=FakeSubprocess(FakeRun(raises=RuntimeError("boom")))):
        ok4, msg4 = L._lms_unload_all()
        check("lms_unload_all_generic_exception", ok4 is False and "boom" in msg4)


def test_load_model_exclusive():
    with _Patches(subprocess=FakeSubprocess(FakeRun(returncode=1, stderr="unload fail"))):
        ok, msg = L.load_model_exclusive("http://x", "m1")
        check("load_model_exclusive_unload_fails", ok is False)

    # exit 0 alone is no longer success: the loader confirms the model is
    # SERVED (see test_load_verifies_served.py); here the server says yes.
    with _Patches(_lms_unload_all=lambda: (True, "ok"),
                  resolve_virtual_model=lambda base_url, mid: (None, False),
                  _wait_served=lambda base_url, mid, timeout=None: True,
                  subprocess=FakeSubprocess(FakeRun(returncode=0, stdout="loaded"))):
        ok2, msg2 = L.load_model_exclusive("http://x", "m1", context_length=4096, gpu_offload_pct=50)
        check("load_model_exclusive_success", ok2 is True)

    with _Patches(_lms_unload_all=lambda: (True, "ok"),
                  resolve_virtual_model=lambda base_url, mid: ("raw@q8", True),
                  subprocess=FakeSubprocess(FakeRun(returncode=1, stderr="fail"))):
        # both candidates fail
        ok3, msg3 = L.load_model_exclusive("http://x", "virt")
        check("load_model_exclusive_both_candidates_fail", ok3 is False)

    calls = {"n": 0}
    def scripted_run(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeRun(returncode=1, stderr="first fails")
        return FakeRun(returncode=0, stdout="second works")
    with _Patches(_lms_unload_all=lambda: (True, "ok"),
                  resolve_virtual_model=lambda base_url, mid: ("raw@q8", True),
                  _wait_served=lambda base_url, mid, timeout=None: True,
                  subprocess=types.SimpleNamespace(run=scripted_run)):
        ok4, msg4 = L.load_model_exclusive("http://x", "virt")
        check("load_model_exclusive_fallback_candidate_succeeds", ok4 is True and "virt" in msg4)

    import subprocess as _sp
    with _Patches(_lms_unload_all=lambda: (True, "ok"),
                  resolve_virtual_model=lambda base_url, mid: (None, False),
                  subprocess=FakeSubprocess(FakeRun(raises=_sp.TimeoutExpired(cmd="lms", timeout=1)))):
        ok5, msg5 = L.load_model_exclusive("http://x", "m1")
        check("load_model_exclusive_timeout", ok5 is False and "Timeout" in msg5)

    with _Patches(_lms_unload_all=lambda: (True, "ok"),
                  resolve_virtual_model=lambda base_url, mid: (None, False),
                  subprocess=FakeSubprocess(FakeRun(raises=RuntimeError("weird")))):
        ok6, msg6 = L.load_model_exclusive("http://x", "m1")
        check("load_model_exclusive_generic_exception", ok6 is False)


def test_ensure_exclusive():
    with _Patches(loaded_model_ids=lambda base_url: ["m1"],
                  effective_model_id=lambda base_url, mid: "m1"):
        ok, msg = L.ensure_exclusive("http://x", "m1")
        check("ensure_exclusive_already_loaded", ok is True and msg == "already loaded")
    with _Patches(loaded_model_ids=lambda base_url: ["m1"],
                  effective_model_id=lambda base_url, mid: "m2",
                  load_model_exclusive=lambda *a, **k: (True, "loaded fresh")):
        ok2, msg2 = L.ensure_exclusive("http://x", "m2")
        check("ensure_exclusive_loads_when_different", ok2 is True and msg2 == "loaded fresh")
    # An embedding/reranker model (EMBED_MODEL, served through the same LM
    # Studio instance) is routinely ALSO resident alongside the chat model --
    # `loaded_model_ids` legitimately returns more than one entry during
    # normal use. Reselecting the chat model that's already active must still
    # be recognized as a no-op instead of unloading everything and reloading.
    with _Patches(loaded_model_ids=lambda base_url: ["m1", "text-embedding-bge-m3"],
                  effective_model_id=lambda base_url, mid: "m1",
                  load_model_exclusive=lambda *a, **k: (True, "should not be called")):
        ok3, msg3 = L.ensure_exclusive("http://x", "m1")
        check("ensure_exclusive_already_loaded_alongside_other_model",
              ok3 is True and msg3 == "already loaded")


def test_gpu_arg():
    check("gpu_arg_max", L._gpu_arg(100) == "max")
    check("gpu_arg_off", L._gpu_arg(0) == "off")
    check("gpu_arg_negative_off", L._gpu_arg(-5) == "off")
    check("gpu_arg_percent", L._gpu_arg(50) == "0.50")


def test_reload_via_cli():
    with _Patches(_lms_unload_all=lambda: (False, "unload failed")):
        ok, msg = L.reload_via_cli("m1", 4096, 50)
        check("reload_via_cli_unload_fails", ok is False and msg == "unload failed")
    with _Patches(_lms_unload_all=lambda: (True, "ok"),
                  subprocess=FakeSubprocess(FakeRun(returncode=0, stdout="loaded"))):
        ok2, msg2 = L.reload_via_cli("m1", 4096, 50)
        check("reload_via_cli_success", ok2 is True)
    with _Patches(_lms_unload_all=lambda: (True, "ok"),
                  subprocess=FakeSubprocess(FakeRun(returncode=1, stderr="load fail"))):
        ok3, msg3 = L.reload_via_cli("m1", 4096, 50)
        check("reload_via_cli_load_fails", ok3 is False)
    import subprocess as _sp
    with _Patches(_lms_unload_all=lambda: (True, "ok"),
                  subprocess=FakeSubprocess(FakeRun(raises=_sp.TimeoutExpired(cmd="lms", timeout=1)))):
        ok4, msg4 = L.reload_via_cli("m1", 4096, 50)
        check("reload_via_cli_timeout", ok4 is False and "Timeout" in msg4)
    with _Patches(_lms_unload_all=lambda: (True, "ok"),
                  subprocess=FakeSubprocess(FakeRun(raises=RuntimeError("weird")))):
        ok5, msg5 = L.reload_via_cli("m1", 4096, 50)
        check("reload_via_cli_generic_exception", ok5 is False)


def test_http_reload():
    class FakeResp:
        def __init__(self, ok=True, status=200, body=None):
            self.ok = ok; self.status_code = status
            self._body = body or {}
        def json(self):
            return self._body
    with _Patches(requests=types.SimpleNamespace(post=lambda *a, **k: FakeResp(ok=True))):
        ok, msg = L._http_reload("http://x", "m1", {})
        check("http_reload_success", ok is True and msg == "OK")
    with _Patches(requests=types.SimpleNamespace(post=lambda *a, **k: FakeResp(ok=False, status=500))):
        ok2, msg2 = L._http_reload("http://x", "m1", {})
        check("http_reload_http_error", ok2 is False and "500" in msg2)
    with _Patches(requests=types.SimpleNamespace(post=lambda *a, **k: FakeResp(ok=True, body={"error": "bad config"}))):
        ok3, msg3 = L._http_reload("http://x", "m1", {})
        check("http_reload_body_error", ok3 is False and msg3 == "bad config")
    with _Patches(requests=types.SimpleNamespace(post=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))):
        ok4, msg4 = L._http_reload("http://x", "m1", {})
        check("http_reload_exception", ok4 is False and "boom" in msg4)


def test_reload_model_full():
    with _Patches(reload_via_cli=lambda *a, **k: (True, "cli ok"),
                  wait_for_loaded=lambda *a, **k: True,
                  fetch_model=lambda *a, **k: {"loaded_context_length": 4096}):
        out = L.reload_model_full("http://x", "m1", 4096, 50, "auto", False)
        check("reload_full_cli_success_no_manual", out.ok is True and "lms CLI" in out.message)

    with _Patches(reload_via_cli=lambda *a, **k: (True, "cli ok"),
                  wait_for_loaded=lambda *a, **k: True,
                  fetch_model=lambda *a, **k: {"loaded_context_length": 2048}):
        out2 = L.reload_model_full("http://x", "m1", 4096, 50, "auto", False)
        check("reload_full_cli_context_mismatch_warns", "ограничено" in out2.message)

    with _Patches(reload_via_cli=lambda *a, **k: (True, "cli ok"),
                  wait_for_loaded=lambda *a, **k: True,
                  fetch_model=lambda *a, **k: {"loaded_context_length": 4096},
                  _http_reload=lambda *a, **k: (True, "http ok")):
        out3 = L.reload_model_full("http://x", "m1", 4096, 50, "q8_0", True)
        check("reload_full_cli_plus_http_success", out3.ok is True and "HTTP API" in out3.message)

    with _Patches(reload_via_cli=lambda *a, **k: (True, "cli ok"),
                  wait_for_loaded=lambda *a, **k: True,
                  fetch_model=lambda *a, **k: {"loaded_context_length": 4096},
                  _http_reload=lambda *a, **k: (False, "http fail")):
        out4 = L.reload_model_full("http://x", "m1", 4096, 50, "q8_0", True)
        check("reload_full_cli_ok_http_fails_manual", out4.ok is True and out4.manual_settings != "")

    with _Patches(reload_via_cli=lambda *a, **k: (False, "cli fail"),
                  _http_reload=lambda *a, **k: (True, "http ok")):
        out5 = L.reload_model_full("http://x", "m1", 4096, 50, "auto", False)
        check("reload_full_cli_fails_http_succeeds", out5.ok is True and "HTTP API" in out5.message)

    with _Patches(reload_via_cli=lambda *a, **k: (False, "cli fail"),
                  _http_reload=lambda *a, **k: (False, "http fail")):
        out6 = L.reload_model_full("http://x", "m1", 4096, 50, "q8_0", True)
        check("reload_full_both_fail", out6.ok is False and out6.manual_settings != "")


def test_manual_fields_and_format():
    check("manual_fields_default_false", not L._manual_fields("auto", False))
    check("manual_fields_kv_true", L._manual_fields("q8_0", False))
    check("manual_fields_fa_true", L._manual_fields("auto", True))
    out = L._format_manual("q8_0", True)
    check("format_manual_has_kv_and_fa", "KV" in out and "Flash" in out)
    out2 = L._format_manual_full(4096, 50, "q4_0", False)
    check("format_manual_full_has_context_gpu", "4,096" in out2 and "50%" in out2)


def test_build_config_and_supported_fields():
    cfg = L.build_config(4096, "q8_0", 100, True)
    check("build_config_full_gpu_offload_1", cfg["gpuOffload"] == 1.0)
    cfg2 = L.build_config(4096, "auto", 50, False)
    check("build_config_no_kv_when_auto", "kvCacheQuantization" not in cfg2)
    check("supported_fields_gguf", "kvCacheQuantization" in L.supported_fields("GGUF"))
    check("supported_fields_mlx", L.supported_fields("mlx") == {"contextLength"})
    check("supported_fields_unknown_default", L.supported_fields("weird") == {"contextLength"})


def test_wait_for_loaded():
    with _Patches(fetch_model=lambda *a, **k: {"state": "loaded"}):
        check("wait_for_loaded_immediate_true", L.wait_for_loaded("http://x", "m1", timeout=5))
    with _Patches(fetch_model=lambda *a, **k: {"state": "unloading"}, time=types.SimpleNamespace(
            time=lambda: 0, sleep=lambda s: None)):
        check("wait_for_loaded_timeout_false", not L.wait_for_loaded("http://x", "m1", timeout=-1))


def test_json_or_empty():
    class GoodResp:
        def json(self): return {"a": 1}
    class BadResp:
        def json(self): raise RuntimeError("no json")
    check("json_or_empty_good", L._json_or_empty(GoodResp()) == {"a": 1})
    check("json_or_empty_bad", L._json_or_empty(BadResp()) == {})


# ── one loader at a time ──────────────────────────────────────────────────────
# Live, 2026-09-12: a crash revive and the card release's reload ran together;
# LM Studio said "2 models match ... Operation canceled", then 500, and four
# turns died. ensure_exclusive is serialised process-wide.
def test_ensure_exclusive_is_serialised():
    import threading, time as _t
    import lmstudio as L
    order = []
    real = L._ensure_exclusive_locked
    def slow(*a, **k):
        order.append("in"); _t.sleep(0.3); order.append("out"); return True, "ok"
    L._ensure_exclusive_locked = slow
    try:
        ts = [threading.Thread(target=L.ensure_exclusive, args=("http://x", "m")) for _ in range(3)]
        [t.start() for t in ts]; [t.join() for t in ts]
    finally:
        L._ensure_exclusive_locked = real
    check("ensure_exclusive_serialised", order == ["in", "out"] * 3, order)


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
