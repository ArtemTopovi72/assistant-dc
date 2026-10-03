"""Coverage for model_selector.py: GGUF size indexing, model listing (native +
OpenAI-compatible fallback), and the interactive picker. Stubs requests/input/
filesystem so every branch runs deterministically.
Run: venv/Scripts/python.exe tests/test_model_selector.py
"""
import os, sys, tempfile, builtins, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path
import model_selector as M

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))

_TMP = Path(tempfile.mkdtemp(prefix="modelsel_"))


class _Patches:
    def __init__(self, **kw):
        self.kw = kw; self.orig = {}
    def __enter__(self):
        for k, v in self.kw.items():
            self.orig[k] = getattr(M, k)
            setattr(M, k, v)
        return self
    def __exit__(self, *a):
        for k, v in self.orig.items():
            setattr(M, k, v)


class FakeResp:
    def __init__(self, ok=True, data=None, raises=None):
        self._ok = ok
        self._data = data if data is not None else []
        self._raises = raises
    def raise_for_status(self):
        if self._raises:
            raise self._raises
        if not self._ok:
            raise RuntimeError("http error")
    def json(self):
        return {"data": self._data}


def test_index_model_sizes():
    models_dir = _TMP / "models"
    repo_dir = models_dir / "Pub" / "Model-GGUF"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "model-q8_0.gguf").write_bytes(b"x" * 1000)
    (repo_dir / "mmproj-model-f16.gguf").write_bytes(b"y" * 500)  # skipped

    virt_dir = models_dir / "Pub" / "Virtual"
    virt_dir.mkdir(parents=True, exist_ok=True)
    base_gguf = repo_dir / "model-q8_0.gguf"
    (virt_dir / "model.yaml").write_text(
        f"base: ../Model-GGUF/model-q8_0.gguf\n", encoding="utf-8")

    idx = M._index_model_sizes(models_dir)
    check("index_model_sizes_by_key", ("model-gguf", "q8_0") in idx["by_key"] or ("model", "q8_0") in idx["by_key"])
    check("index_model_sizes_skips_mmproj", all(v != 500 for v in idx["by_key"].values()))
    check("index_model_sizes_by_repo_populated", len(idx["by_repo"]) > 0)

    idx_empty = M._index_model_sizes(_TMP / "nonexistent_dir_xyz")
    check("index_model_sizes_missing_dir_empty", idx_empty["by_key"] == {} and idx_empty["by_repo"] == {})


def test_index_model_sizes_non_gguf_fallback():
    # MLX/safetensors repos have no .gguf file to size, so the indexer must
    # fall back to the repo folder's total on-disk size instead of leaving it
    # unsized (the GUI shows "?" for anything _size_gb can't find).
    models_dir = _TMP / "models_mlx"
    repo_dir = models_dir / "google" / "gemma-4-12b-qat"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "weights.safetensors").write_bytes(b"z" * 3000)
    (repo_dir / "config.json").write_bytes(b"{}")

    idx = M._index_model_sizes(models_dir)
    check("index_model_sizes_non_gguf_fallback_sized",
          idx["by_repo"].get("gemma-4-12b-qat") == 3002)
    gb = M._size_gb("google/gemma-4-12b-qat", idx)
    check("size_gb_non_gguf_fallback_not_none", gb is not None)


def test_index_model_sizes_hub_catalog_alias():
    # LM Studio's "virtual model" aliases (the id the API actually reports)
    # live in a separate hub/models catalog dir, each with a model.yaml that
    # points at the REAL downloaded repo by key -- not by a "base: x.gguf"
    # file path like the older-style virtual model.yaml already handled.
    models_dir = _TMP / "models_hub"
    repo_dir = models_dir / "lmstudio-community" / "gemma-4-12B-it-QAT-GGUF"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "gemma-4-12B-it-QAT-Q4_0.gguf").write_bytes(b"x" * 5000)

    hub_dir = _TMP / "hub_models"
    alias_dir = hub_dir / "google" / "gemma-4-12b-qat"
    alias_dir.mkdir(parents=True, exist_ok=True)
    (alias_dir / "model.yaml").write_text(
        "model: google/gemma-4-12b-qat\n"
        "base:\n"
        "  - key: lmstudio-community/gemma-4-12b-it-qat-gguf\n"
        "    sources:\n"
        "      - type: huggingface\n",
        encoding="utf-8")

    idx = M._index_model_sizes(models_dir, hub_dir)
    check("index_model_sizes_hub_alias_resolved",
          idx["by_repo"].get("gemma-4-12b-qat") == 5000)
    check("size_gb_hub_alias_not_none",
          M._size_gb("google/gemma-4-12b-qat", idx) is not None)


def test_index_model_sizes_bad_yaml():
    models_dir = _TMP / "models_bad_yaml"
    virt_dir = models_dir / "Pub" / "BadVirtual"
    virt_dir.mkdir(parents=True, exist_ok=True)
    (virt_dir / "model.yaml").write_text("base: nonexistent/path.gguf\n", encoding="utf-8")
    idx = M._index_model_sizes(models_dir)
    check("index_model_sizes_yaml_missing_base_noerror", isinstance(idx, dict))


def test_size_gb():
    index = {"by_key": {("model", "q8_0"): 2_000_000_000}, "by_repo": {"model": 1_500_000_000}}
    check("size_gb_by_key_match", abs(M._size_gb("Pub/model@q8_0", index) - 2.0) < 0.01)
    check("size_gb_by_repo_fallback", abs(M._size_gb("Pub/model@unknownquant", index) - 1.5) < 0.01)
    check("size_gb_no_match_none", M._size_gb("Pub/unknownmodel@q8_0", index) is None)


def test_size_gb_three_segment_multiquant_repo():
    # A repo folder holding several .gguf quants side by side gets a server id
    # shaped "publisher/repo/repo-quant.gguf" -- a THIRD segment that is the
    # actual FILENAME, not the repo. Live, 2026-09-22: the Settings model
    # picker showed "?" for both quants of exactly this shape, because
    # _size_gb's old base.split("/")[-1] grabbed the filename and looked IT
    # up in by_repo (keyed by the folder name) -- a filename never matches a
    # folder name. Two quants sharing one folder must also resolve to their
    # OWN sizes, not both collapse onto the folder's largest file.
    index = {
        "by_key": {("gemma-repo", "q5_k"): 5_000_000_000,
                   ("gemma-repo", "q8_k"): 8_000_000_000},
        "by_repo": {"gemma-repo": 8_000_000_000},
    }
    id5 = "Pub/gemma-repo/gemma-repo-q5_k_p.gguf"
    id8 = "Pub/gemma-repo/gemma-repo-q8_k_p.gguf"
    check("size_gb_three_segment_q5", abs(M._size_gb(id5, index) - 5.0) < 0.01,
          M._size_gb(id5, index))
    check("size_gb_three_segment_q8", abs(M._size_gb(id8, index) - 8.0) < 0.01,
          M._size_gb(id8, index))
    check("size_gb_three_segment_quants_differ",
          M._size_gb(id5, index) != M._size_gb(id8, index))


def test_list_models_native_api():
    with _Patches(_disk_models=lambda: [], requests=types.SimpleNamespace(
            get=lambda url, timeout=10: FakeResp(data=[
                {"id": "chat-model", "type": "llm"},
                {"id": "embed-model", "type": "embeddings"},
                {"id": "vlm-model", "type": "vlm"},
            ]))):
        out = M.list_models("http://x")
        check("list_models_native_filters_embed", all("embed" not in m["id"] for m in out))
        check("list_models_native_keeps_llm_vlm", len(out) == 2)


def test_list_models_native_fails_falls_back():
    calls = {"n": 0}
    def fake_get(url, timeout=10):
        calls["n"] += 1
        if "/api/v0/" in url:
            raise RuntimeError("native down")
        return FakeResp(data=[{"id": "fallback-model"}])
    with _Patches(_disk_models=lambda: [], requests=types.SimpleNamespace(get=fake_get)):
        out = M.list_models("http://x")
        check("list_models_falls_back_to_v1", out == [{"id": "fallback-model"}])



def test_the_picker_shows_models_that_are_installed_but_not_loaded():
    """The defect that put this here: the Settings dialog listed ONE model.

    LM Studio's endpoints report what is loaded (or JIT-loadable), not what is
    on disk, so a machine with eight models offered whichever one happened to
    be resident. list_models now merges `lms ls` on top of whatever the server
    says, and a model already reported by the server is not repeated.
    """
    disk = [{"id": "disk-only-model"}, {"id": "served-model"}]
    with _Patches(_disk_models=lambda: disk,
                  requests=types.SimpleNamespace(
                      get=lambda url, timeout=10: FakeResp(
                          data=[{"id": "served-model"}]))):
        ids = [m["id"] for m in M.list_models("http://x")]
    check("picker_includes_the_unloaded_model", "disk-only-model" in ids, ids)
    check("picker_does_not_duplicate_the_served_one",
          ids.count("served-model") == 1, ids)


def test_an_unreachable_server_still_lists_what_is_installed():
    with _Patches(_disk_models=lambda: [{"id": "disk-only-model"}],
                  requests=types.SimpleNamespace(
                      get=lambda url, timeout=10:
                          (_ for _ in ()).throw(RuntimeError("boom")))):
        ids = [m["id"] for m in M.list_models("http://x")]
    check("offline_picker_is_not_empty", ids == ["disk-only-model"], ids)

def test_list_models_both_fail():
    with _Patches(_disk_models=lambda: [], requests=types.SimpleNamespace(
            get=lambda url, timeout=10: (_ for _ in ()).throw(RuntimeError("boom")))):
        out = M.list_models("http://x")
        check("list_models_both_fail_empty", out == [])


def test_list_models_native_empty_falls_through():
    calls = {"n": 0}
    def fake_get(url, timeout=10):
        calls["n"] += 1
        if "/api/v0/" in url:
            return FakeResp(data=[{"id": "embed-only", "type": "embeddings"}])  # filtered to empty
        return FakeResp(data=[{"id": "v1-model"}])
    with _Patches(_disk_models=lambda: [], requests=types.SimpleNamespace(get=fake_get)):
        out = M.list_models("http://x")
        check("list_models_native_empty_after_filter_falls_through", out == [{"id": "v1-model"}])


def test_list_models_native_raises_status():
    with _Patches(_disk_models=lambda: [], requests=types.SimpleNamespace(
            get=lambda url, timeout=10: FakeResp(ok=False))):
        # both native (raises via raise_for_status) and v1 (also raises) -> empty
        out = M.list_models("http://x")
        check("list_models_native_and_v1_error_status_empty", out == [])


def test_choose_model_interactive_no_models():
    with _Patches(list_models=lambda base_url: []):
        model, no_think = M.choose_model_interactive("http://x", "default-model")
        check("choose_model_no_models_uses_default", model == "default-model" and no_think is True)


def test_choose_model_interactive_enter_defaults():
    real_input = builtins.input
    inputs = iter(["", "n"])
    builtins.input = lambda prompt="": next(inputs)
    try:
        with _Patches(list_models=lambda base_url: [{"id": "model-a", "state": "loaded"},
                                                     {"id": "model-b"}],
                      _index_model_sizes=lambda d: {"by_key": {}, "by_repo": {}}):
            model, no_think = M.choose_model_interactive("http://x", "model-b")
            check("choose_model_enter_picks_default", model == "model-b")
            check("choose_model_no_think_default_true_on_n", no_think is True)
    finally:
        builtins.input = real_input


def test_choose_model_interactive_numeric_selection():
    real_input = builtins.input
    inputs = iter(["2", "y"])
    builtins.input = lambda prompt="": next(inputs)
    try:
        with _Patches(list_models=lambda base_url: [{"id": "model-a"}, {"id": "model-b"}],
                      _index_model_sizes=lambda d: {"by_key": {}, "by_repo": {"model-b": 4_000_000_000}}):
            model, no_think = M.choose_model_interactive("http://x", "model-a")
            check("choose_model_numeric_selection", model == "model-b")
            check("choose_model_no_think_false_on_y", no_think is False)
    finally:
        builtins.input = real_input


def test_choose_model_interactive_invalid_then_valid():
    real_input = builtins.input
    inputs = iter(["99", "abc", "1", "no"])
    builtins.input = lambda prompt="": next(inputs)
    try:
        with _Patches(list_models=lambda base_url: [{"id": "model-a"}, {"id": "model-b"}],
                      _index_model_sizes=lambda d: {"by_key": {}, "by_repo": {}}):
            model, no_think = M.choose_model_interactive("http://x", "model-x")
            check("choose_model_invalid_retries_then_picks", model == "model-a")
    finally:
        builtins.input = real_input


def test_choose_model_interactive_default_not_in_list():
    real_input = builtins.input
    inputs = iter(["", "y"])
    builtins.input = lambda prompt="": next(inputs)
    try:
        with _Patches(list_models=lambda base_url: [{"id": "model-a"}, {"id": "model-b"}],
                      _index_model_sizes=lambda d: {"by_key": {}, "by_repo": {}}):
            model, no_think = M.choose_model_interactive("http://x", "nonexistent-default")
            check("choose_model_default_not_in_list_uses_first", model == "model-a")
    finally:
        builtins.input = real_input


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
