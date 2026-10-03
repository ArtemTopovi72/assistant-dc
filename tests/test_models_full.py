"""Full branch coverage for models.py: _atomic_write_json (retry + cleanup paths),
Models.load (success + accentor-failure via stubbed heavy imports), and every
Context method (memory load/save with corrupt inputs, remember/pin/facts, cache,
stage callback, cancellation).

Run: venv/Scripts/python.exe tests/test_models_full.py
"""
import os, sys, json, types, tempfile, threading, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path
from collections import deque

import models as M

_TMP = Path(tempfile.mkdtemp(prefix="models_"))

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    assert cond, name + ": " + detail


@contextlib.contextmanager
def fake_modules(**mods):
    saved = {}
    for name, mod in mods.items():
        saved[name] = sys.modules.get(name)
        if mod is None:
            sys.modules[name] = None
        else:
            m = types.ModuleType(name)
            for k, v in mod.items():
                setattr(m, k, v)
            sys.modules[name] = m
    try:
        yield
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _ctx(**kw):
    c = M.Context(
        models=types.SimpleNamespace(),
        transcription_cache={},
        cache_file=_TMP / "cache.json",
        asr_lock=threading.Lock(),
        tts_lock=threading.Lock(),
    )
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# ---------------------------------------------------------- _atomic_write_json

def test_atomic_write_normal():
    p = _TMP / "aw.json"
    M._atomic_write_json(p, {"a": 1})
    check("atomic_normal", json.loads(p.read_text(encoding="utf-8")) == {"a": 1})


def test_atomic_write_retry_then_success():
    p = _TMP / "aw2.json"
    calls = {"n": 0}
    real = os.replace
    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("locked")   # first attempt fails -> retry
        return real(src, dst)
    os.replace = flaky
    M.time.sleep = lambda *_a, **_k: None
    try:
        M._atomic_write_json(p, {"b": 2})
        check("atomic_retry", p.exists() and calls["n"] == 2)
    finally:
        os.replace = real


def test_atomic_write_permission_all_attempts_raise():
    p = _TMP / "aw3.json"
    real = os.replace
    os.replace = lambda s, d: (_ for _ in ()).throw(PermissionError("always"))
    M.time.sleep = lambda *_a, **_k: None
    raised = False
    try:
        M._atomic_write_json(p, {"c": 3})
    except PermissionError:
        raised = True
    finally:
        os.replace = real
    check("atomic_all_fail_raises", raised)
    # the tmp file must have been cleaned up in the finally block
    leftover = list(_TMP.glob("aw3.json.*.tmp"))
    check("atomic_tmp_cleaned", leftover == [])


def test_atomic_write_tmp_unlink_oserror_swallowed():
    p = _TMP / "aw4.json"
    real_replace = os.replace
    real_unlink = os.unlink
    os.replace = lambda s, d: (_ for _ in ()).throw(PermissionError("x"))
    M.time.sleep = lambda *_a, **_k: None
    # Path.unlink is what the finally calls; patch os-level too. The finally uses
    # tmp.unlink() (Path.unlink). Patch Path.unlink to raise OSError -> swallowed.
    import pathlib
    real_path_unlink = pathlib.Path.unlink
    pathlib.Path.unlink = lambda self, *a, **k: (_ for _ in ()).throw(OSError("cannot"))
    raised = False
    try:
        M._atomic_write_json(p, {"d": 4})
    except PermissionError:
        raised = True   # the replace error still propagates
    except OSError:
        raised = "wrong"
    finally:
        os.replace = real_replace
        pathlib.Path.unlink = real_path_unlink
    check("atomic_unlink_oserror_swallowed", raised is True)


# ---------------------------------------------------------- Models.load

def _load_stubs():
    return dict(
        faster_whisper={"WhisperModel": lambda *a, **k: "WHISPER"},
        f5_tts=types.SimpleNamespace(),  # placeholder; sub-packages injected separately
    )

def test_models_load_success_and_accentor_fail():
    def stub_env(accentor_ok):
        fw = {"WhisperModel": lambda *a, **k: "WHISPER"}
        uinfer = {"load_model": lambda *a, **k: "TTS", "load_vocoder": lambda *a, **k: "VOCODER"}
        f5model = {"DiT": object}
        if accentor_ok:
            class BA:
                def __init__(self, **kw): pass
            stress = {"BilingualAccentor": BA}
        else:
            stress = {"BilingualAccentor": lambda **kw: (_ for _ in ()).throw(RuntimeError("no accentor"))}
        return fw, uinfer, f5model, stress

    # success path incl. accentor loaded
    fw, uinfer, f5model, stress = stub_env(True)
    with fake_modules(faster_whisper=fw):
        # f5_tts.infer.utils_infer and f5_tts.model are dotted; inject directly
        with fake_modules(**{"f5_tts.infer.utils_infer": uinfer, "f5_tts.model": f5model, "stress": stress}):
            # the intermediate packages must exist for `from f5_tts.infer.utils_infer import ...`
            with fake_modules(**{"f5_tts": {}, "f5_tts.infer": {}}):
                m = M.Models.load()
    check("load_success", m.whisper == "WHISPER" and m.tts_model == "TTS"
          and m.vocoder == "VOCODER" and m.accentor_loaded is True)

    # accentor-failure branch
    fw, uinfer, f5model, stress = stub_env(False)
    with fake_modules(faster_whisper=fw):
        with fake_modules(**{"f5_tts.infer.utils_infer": uinfer, "f5_tts.model": f5model, "stress": stress}):
            with fake_modules(**{"f5_tts": {}, "f5_tts.infer": {}}):
                m2 = M.Models.load()
    check("load_accentor_fail", m2.accentor is None and m2.accentor_loaded is False
          and m2.whisper == "WHISPER")


# ---------------------------------------------------------- Context methods

def test_is_cancelled_and_stage():
    c = _ctx()
    check("not_cancelled", c.is_cancelled() is False)
    c.cancel_event.set()
    check("cancelled", c.is_cancelled() is True)
    # set_stage with no callback -> no-op
    c.set_stage("x"); check("stage_no_cb", True)
    seen = []
    c.stage_callback = lambda s: seen.append(s)
    c.set_stage("Searching"); check("stage_cb", seen == ["Searching"])
    # callback raising is swallowed
    c.stage_callback = lambda s: (_ for _ in ()).throw(RuntimeError("boom"))
    c.set_stage("Y"); check("stage_cb_except", True)


def test_remember_and_memory_text():
    c = _ctx()
    c.remember("note", "  hello  ", {"src": "test"})
    check("remember_strips", c.session_memory[-1]["text"] == "hello")
    c.remember("note", "   ")   # empty after strip -> not stored
    check("remember_empty_skipped", len(c.session_memory) == 1)
    for i in range(20):
        c.remember("note", f"item {i}")
    txt = c.memory_text(limit=5)
    check("memory_text_limit", txt.count("\n") == 4)   # last 5 items -> 5 lines
    # a stray non-dict entry must be skipped by memory_text
    with c.memory_lock:
        c.session_memory.append("STRAY_STRING")
    txt2 = c.memory_text(limit=50)
    check("memory_text_skips_nondict", "STRAY_STRING" not in txt2)
    # memory_text with an item whose text is empty -> skipped
    c2 = _ctx()
    with c2.memory_lock:
        c2.session_memory.append({"kind": "note", "text": "   "})
        c2.session_memory.append({"kind": "note", "text": "real"})
    check("memory_text_empty_item", c2.memory_text() == "1. [note] real" or "real" in c2.memory_text())


def test_pin_fact_and_facts_text():
    c = _ctx()
    check("pin_empty_false", c.pin_fact("   ") is False)
    check("pin_ok", c.pin_fact("user likes dark themes") is True)
    check("pin_dup_false", c.pin_fact("USER LIKES DARK THEMES") is False)  # casefold dup
    for i in range(M._FACTS_LIMIT + 5):
        c.pin_fact(f"fact number {i}")
    check("pin_capped", len(c.pinned_facts) <= M._FACTS_LIMIT)
    ft = c.facts_text()
    check("facts_text_bullets", ft.startswith("- ") and "\n- " in ft)
    # facts_text with a blank fact filtered
    c3 = _ctx()
    with c3.memory_lock:
        c3.pinned_facts.append({"ts": 0, "text": "  "})
        c3.pinned_facts.append({"ts": 0, "text": "keep"})
    check("facts_text_filters_blank", c3.facts_text() == "- keep")


def test_save_and_load_memory_roundtrip():
    d = _TMP / "prof1"; d.mkdir(parents=True, exist_ok=True)
    c = _ctx()
    c.remember("note", "session item one")
    c.remember("summary", "this is a summary")   # summary excluded from session file
    c.pin_fact("a pinned fact")
    c.save_memory(d)
    check("session_file_written", (d / "session_memory.json").exists())
    check("facts_file_written", (d / "facts.json").exists())
    saved = json.loads((d / "session_memory.json").read_text(encoding="utf-8"))
    check("summary_excluded", all(i["kind"] != "summary" for i in saved))
    # load into a fresh context
    c2 = _ctx()
    (d / "summary.json").write_text(json.dumps({"text": "loaded summary", "ts": 1.0}), encoding="utf-8")
    c2.load_memory(d)
    kinds = [i.get("kind") for i in c2.session_memory]
    check("load_summary", "summary" in kinds)
    # save_memory persists only kind="fact", and load_memory restores only
    # durable kinds, so an ordinary session item is deliberately NOT expected back.
    check("load_drops_non_durable_session_items",
          not any(i.get("text") == "session item one" for i in c2.session_memory))
    check("load_keeps_durable_fact",
          any(i.get("kind") == "fact" for i in c2.session_memory)
          or any(f.get("text") == "a pinned fact" for f in c2.pinned_facts))
    check("load_facts", any(f.get("text") == "a pinned fact" for f in c2.pinned_facts))


def test_load_memory_corrupt_inputs():
    d = _TMP / "corrupt"; d.mkdir(parents=True, exist_ok=True)
    # summary.json invalid JSON -> warning, skipped
    (d / "summary.json").write_text("{ not json", encoding="utf-8")
    # session_memory.json is a JSON object (wrong shape) -> ignored with warning
    (d / "session_memory.json").write_text(json.dumps({"unexpected": "object"}), encoding="utf-8")
    # facts.json invalid -> warning
    (d / "facts.json").write_text("also not json", encoding="utf-8")
    c = _ctx()
    c.load_memory(d)
    check("corrupt_no_crash", len(c.session_memory) == 0 and c.pinned_facts == [])
    # session_memory.json a list with mixed dict + non-dict -> only dicts kept
    (d / "summary.json").unlink()
    # kind="fact" so the dict filter is what decides the outcome — with a
    # non-durable kind the durability filter would drop everything and the
    # assertion below would pass without testing anything.
    (d / "session_memory.json").write_text(json.dumps([{"text": "ok", "kind": "fact"}, "stray", 42]),
                                           encoding="utf-8")
    (d / "facts.json").write_text(json.dumps([{"text": "f1"}, "notdict", {"text": "  "}]), encoding="utf-8")
    c2 = _ctx()
    c2.load_memory(d)
    check("mixed_list_dicts_only", len(c2.session_memory) == 1
          and c2.session_memory[0]["text"] == "ok")
    check("facts_filter_blank_and_nondict", [f["text"] for f in c2.pinned_facts] == ["f1"])
    # summary present but empty text -> not added
    (d / "summary.json").write_text(json.dumps({"text": "", "ts": 1.0}), encoding="utf-8")
    (d / "session_memory.json").unlink(); (d / "facts.json").unlink()
    c3 = _ctx()
    c3.load_memory(d)
    check("empty_summary_skipped", len(c3.session_memory) == 0)
    # session_memory.json is INVALID JSON -> json.load raises -> except branch (242-243)
    d2 = _TMP / "corrupt2"; d2.mkdir(parents=True, exist_ok=True)
    (d2 / "session_memory.json").write_text("{ broken json ]", encoding="utf-8")
    c4 = _ctx()
    c4.load_memory(d2)
    check("session_invalid_json_except", len(c4.session_memory) == 0)


def test_save_memory_error_swallowed_and_cache():
    c = _ctx()
    # save to a path that can't be created (mkdir raises) -> logged, no crash
    bad = types.SimpleNamespace(mkdir=lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
    c.save_memory(bad)   # exception swallowed
    check("save_memory_error_swallowed", True)
    # save_cache normal
    c.transcription_cache = {"k": "v"}
    c.cache_file = _TMP / "cachedir" / "c.json"
    c.save_cache()
    check("save_cache_written", c.cache_file.exists())
    # save_cache error swallowed
    c.cache_file = types.SimpleNamespace(parent=types.SimpleNamespace(
        mkdir=lambda *a, **k: (_ for _ in ()).throw(OSError("ro"))))
    c.save_cache()
    check("save_cache_error_swallowed", True)


if __name__ == "__main__":
    _real_replace = os.replace
    _real_sleep = M.time.sleep
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print("  ERROR in " + fn.__name__ + ": " + type(e).__name__ + ": " + str(e))
        finally:
            os.replace = _real_replace
            M.time.sleep = _real_sleep
    print("\n" + str(len(fns) - failed) + "/" + str(len(fns)) + " model test functions passed (" +
          str(sum(1 for _, c in RESULTS if c)) + "/" + str(len(RESULTS)) + " checks)")
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)
