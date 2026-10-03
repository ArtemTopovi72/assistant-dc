"""Drive every QThread worker in gui.py through success + failure paths WITHOUT
threads or heavy deps: run() is called synchronously on the main thread and each
lazily-imported module (assistant/lmstudio/image/library/deep_research/graph/llm)
is replaced with a fake in sys.modules for the duration of the call.

Signals are captured via direct connection (same-thread emit == synchronous slot).

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_workers.py
"""
import os, sys, threading, types, contextlib
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("USE_GUI", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

from PyQt5.QtWidgets import QApplication
import gui
import gui_voice_tab
import gui_workers
import gui_database_tab

_app = QApplication.instance() or QApplication(sys.argv)

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    assert cond, f"{name}: {detail}"


@contextlib.contextmanager
def fake_modules(**mods):
    """Temporarily inject fake modules into sys.modules."""
    saved = {}
    for name, mod in mods.items():
        saved[name] = sys.modules.get(name)
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


def cap(worker, *signals):
    """Connect each named signal to a recording list. Returns {signal: [args...]}."""
    out = {}
    for s in signals:
        out[s] = []
        getattr(worker, s).connect(lambda *a, _s=s, _o=out: _o[_s].append(a if len(a) != 1 else a[0]))
    return out


class Ctx(types.SimpleNamespace):
    pass


def _ctx(**kw):
    c = Ctx()
    c.gui_mode = False
    c.web_search_enabled = True
    c.set_stage = lambda *a, **k: None
    c.remember = lambda *a, **k: None
    c.memory_text = lambda *a, **k: "some memory"
    c.last_image_prompt = "prev prompt"
    # The real context is models.AppContext, and the audio path takes these:
    # without them transcription died inside its own except with "'Ctx' object
    # has no attribute 'asr_lock'" and simply returned "", so three worker
    # tests failed for a reason that had nothing to do with the workers.
    c.asr_lock = threading.Lock()
    c.tts_lock = threading.Lock()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# ------------------------------------------------------------------ ModelLoader

def test_model_loader_success_and_failure():
    w = gui.ModelLoader("m", True, "high")
    rec = cap(w, "ready", "failed", "progress")
    ctx = _ctx()
    # app_runtime, not assistant: ModelLoader imports build_runtime from
    # app_runtime -- its own comment says "not from assistant: that is the entry
    # point" -- so faking `assistant` stubbed a module the worker never loads.
    with fake_modules(app_runtime={"build_runtime":
            lambda name, no_think, status_cb=None, reasoning_effort="high": (ctx, {"messages": []}, "G")}):
        w.run()
    check("model_loader_ready", len(rec["ready"]) == 1 and ctx.gui_mode is True)

    w2 = gui.ModelLoader("m", False)
    rec2 = cap(w2, "ready", "failed")
    with fake_modules(app_runtime={"build_runtime":
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model"))}):
        w2.run()
    check("model_loader_failed", rec2["failed"] == ["no model"])


# ------------------------------------------------------------ ModelSwitchWorker

def test_model_switch_success_and_failure():
    w = gui.ModelSwitchWorker("http://x", "id")
    rec = cap(w, "done")
    with fake_modules(lmstudio={"ensure_exclusive": lambda url, mid: (True, "loaded")}):
        w.run()
    check("switch_ok", rec["done"] == [(True, "loaded")])
    w2 = gui.ModelSwitchWorker("http://x", "id")
    rec2 = cap(w2, "done")
    with fake_modules(lmstudio={"ensure_exclusive": lambda *a: (_ for _ in ()).throw(OSError("boom"))}):
        w2.run()
    check("switch_fail", rec2["done"][0][0] is False and "boom" in rec2["done"][0][1])


# ------------------------------------------------------------------ RedrawWorker

def _image_mod(**over):
    base = {
        "inpaint_region_with_comfy": lambda ctx, src, region, prompt, engine="firered": "out_inpaint.png",
        "fix_hands": lambda ctx, src, mask_override=None, engine="firered": "out_hands.png",
        "edit_region_contained_via_firered": lambda *a, **k: "out_fix.png",
        "assert_deliverable": lambda path, **k: path,
        "log_edit_decision": lambda **k: None,
    }
    base.update(over)
    return base


def _handlers_mod(seen=None, **over):
    """redraw/enhance go through tool_image_handlers.redraw_whole_image (layout
    edit for our own Ideogram picture, FireRed whole-frame for a photo)."""
    def _redraw(ctx, src, instructions):
        if seen is not None:
            seen.append(instructions)
        return "out_redraw.png", "firered-whole"
    base = {"redraw_whole_image": _redraw}
    base.update(over)
    return base


def test_redraw_worker_all_modes_and_failure():
    for mode, expect in [("inpaint", "out_inpaint.png"), ("redraw", "out_redraw.png"),
                         ("enhance", "out_redraw.png")]:
        w = gui.RedrawWorker(_ctx(), "src.png", "make it blue", mode, region="face")
        rec = cap(w, "done", "failed")
        with fake_modules(image=_image_mod(), tool_image_handlers=_handlers_mod()):
            w.run()
        check(f"redraw_{mode}", rec["done"] == [expect], str(rec))
    # enhance with an empty prompt sends a default quality instruction, never ""
    seen = []
    w = gui.RedrawWorker(_ctx(), "src.png", "", "enhance")
    rec = cap(w, "done", "failed")
    with fake_modules(image=_image_mod(), tool_image_handlers=_handlers_mod(seen)):
        w.run()
    check("redraw_enhance_fallback",
          rec["done"] == ["out_redraw.png"] and seen and seen[0].strip(), str(seen))
    # failure path
    w = gui.RedrawWorker(_ctx(), "src.png", "x", "redraw")
    rec = cap(w, "done", "failed")
    with fake_modules(image=_image_mod(), tool_image_handlers=_handlers_mod(
            redraw_whole_image=lambda *a: (_ for _ in ()).throw(ValueError("gpu oom")))):
        w.run()
    check("redraw_failed", rec["failed"] == ["gpu oom"])
    # log_edit_decision raising is swallowed (inner try/except)
    w = gui.RedrawWorker(_ctx(), "src.png", "x", "redraw")
    rec = cap(w, "done", "failed")
    with fake_modules(image=_image_mod(
            log_edit_decision=lambda **k: (_ for _ in ()).throw(IOError("log fail"))),
            tool_image_handlers=_handlers_mod()):
        w.run()
    check("redraw_log_swallowed", rec["done"] == ["out_redraw.png"])


# --------------------------------------------------------------- FixHandsWorker

def test_fix_hands_worker():
    for mask in (None, "mask.png"):
        w = gui.FixHandsWorker(_ctx(), "src.png", mask_path=mask)
        rec = cap(w, "done", "failed")
        with fake_modules(image=_image_mod()):
            w.run()
        check(f"fixhands_mask_{mask is not None}", rec["done"] == ["out_hands.png"])
    w = gui.FixHandsWorker(_ctx(), "src.png")
    rec = cap(w, "done", "failed")
    with fake_modules(image=_image_mod(fix_hands=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nohands")))):
        w.run()
    check("fixhands_failed", rec["failed"] == ["nohands"])


# ------------------------------------------------------------ FixArtifactWorker

def test_fix_artifact_worker():
    w = gui.FixArtifactWorker(_ctx(), "src.png", "mask.png", "remove smear")
    rec = cap(w, "done", "failed")
    with fake_modules(image=_image_mod()):
        w.run()
    check("fixartifact_ok", rec["done"] == ["out_fix.png"])
    w = gui.FixArtifactWorker(_ctx(), "src.png", "mask.png", "x", protect_face=False)
    rec = cap(w, "done", "failed")
    with fake_modules(image=_image_mod(
            edit_region_contained_via_firered=lambda *a, **k: (_ for _ in ()).throw(Exception("edit fail")))):
        w.run()
    check("fixartifact_failed", rec["failed"] == ["edit fail"])


# ----------------------------------------------------------- LibraryBuildWorker

def test_library_build_worker():
    class FakeLib:
        def __init__(self, db_path=None): pass
        def build(self, paths, progress=None, cancel=None):
            progress("extract", 1, 2); progress("embed", 2, 2)
            return {"docs": len(paths)}
        def close(self): pass
    w = gui_database_tab.LibraryBuildWorker(["a.pdf", "b.txt"], db_path="d.db")
    rec = cap(w, "progress", "done", "failed")
    with fake_modules(library={"Library": FakeLib}):
        w.run()
    check("libbuild_done", rec["done"] == [{"docs": 2}] and len(rec["progress"]) == 2)
    w2 = gui_database_tab.LibraryBuildWorker(["a"])
    w2.cancel(); check("libbuild_cancel_flag", w2._cancel is True)
    rec2 = cap(w2, "done", "failed")
    with fake_modules(library={"Library": lambda db_path=None: (_ for _ in ()).throw(OSError("db locked"))}):
        w2.run()
    check("libbuild_failed", rec2["failed"] == ["db locked"])


# ---------------------------------------------------------- DeepResearchWorker

def test_deep_research_worker():
    restored = {"n": 0}
    # The worker also passes out_lang (the Research tab writes the document in the
    # language the topic was typed in) and resolves it via dr.lang_of_text. A stub
    # missing either raises TypeError/AttributeError inside the worker, which it
    # reports as `failed` — so the suite failed on its own stub, not on the code.
    seen = {}
    dr = {
        "apply_overrides": lambda ov: {"saved": True},
        "restore_overrides": lambda s: restored.__setitem__("n", restored["n"] + 1),
        "lang_of_text": lambda t: "ru",
        "run_deep_research": lambda ctx, topic, depth="standard", progress=None,
                                    out_lang="en": (
            seen.__setitem__("out_lang", out_lang)
            or progress("search", {"hits": 3}, "searching") or {"report": "done"}),
    }
    w = gui.DeepResearchWorker(_ctx(), "quantum", overrides={"DR_MAX": 5})
    rec = cap(w, "progress", "done", "failed")
    with fake_modules(deep_research=dr):
        w.run()
    check("dr_done", rec["done"] == [{"report": "done"}] and restored["n"] == 1)
    check("dr_out_lang", seen.get("out_lang") == "ru")
    check("dr_progress", rec["progress"] == [("search", {"hits": 3}, "searching")])
    # no overrides path + failure, finally still restores
    w2 = gui.DeepResearchWorker(_ctx(), "x")
    rec2 = cap(w2, "done", "failed")
    dr2 = dict(dr, run_deep_research=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("dr boom")))
    with fake_modules(deep_research=dr2):
        w2.run()
    check("dr_failed", rec2["failed"] == ["dr boom"])


# --------------------------------------------------------------- TranscribeWorker

def test_transcribe_worker():
    # TranscribeWorker imports transcribe_audio_array from `audio` INSIDE run()
    # (gui_common keeps the module light that way), so `audio` is the only
    # binding a stub can reach -- patching gui's or gui_workers' copy left the
    # real ASR path running against a stub context.
    import audio as _audio
    with fake_modules():
        _audio.transcribe_audio_array = lambda ctx, audio: "hello world"
        w = gui_voice_tab.TranscribeWorker(_ctx(), b"audio")
        rec = cap(w, "recognized", "failed")
        w.run()
        check("transcribe_ok", rec["recognized"] == ["hello world"])
        _audio.transcribe_audio_array = lambda ctx, audio: ""
        w2 = gui_voice_tab.TranscribeWorker(_ctx(), b"a"); rec2 = cap(w2, "recognized", "failed")
        w2.run()
        check("transcribe_empty", len(rec2["failed"]) == 1)
        _audio.transcribe_audio_array = lambda ctx, audio: (_ for _ in ()).throw(RuntimeError("asr crash"))
        w3 = gui_voice_tab.TranscribeWorker(_ctx(), b"a"); rec3 = cap(w3, "recognized", "failed")
        w3.run()
        check("transcribe_crash", rec3["failed"] == ["asr crash"])


# --------------------------------------------------------------- CompactMemoryWorker

def test_compact_memory_worker():
    ctx = _ctx(memory_text=lambda limit=50: "profile text")
    w = gui.CompactMemoryWorker(ctx)
    rec = cap(w, "done", "failed")
    with fake_modules(prompts={"COMPACT_MEMORY_PROMPT": "sys"},
                      llm={"send_to_lm_studio": lambda *a, **k: {"content": "<think>x</think>summary"}},
                      utils={"strip_think_tags": lambda t: t.replace("<think>x</think>", ""),
                             "strip_reasoning_leak": lambda t: t}):
        w.run()
    check("compact_ok", rec["done"] == ["summary"], str(rec))
    # empty memory short-circuits to done("")
    ctx2 = _ctx(memory_text=lambda limit=50: "")
    w2 = gui.CompactMemoryWorker(ctx2); rec2 = cap(w2, "done", "failed")
    with fake_modules(prompts={"COMPACT_MEMORY_PROMPT": "s"}, llm={"send_to_lm_studio": lambda *a, **k: None},
                      utils={"strip_think_tags": lambda t: t, "strip_reasoning_leak": lambda t: t}):
        w2.run()
    check("compact_empty", rec2["done"] == [""])
    # exception
    ctx3 = _ctx(memory_text=lambda limit=50: (_ for _ in ()).throw(RuntimeError("mem boom")))
    w3 = gui.CompactMemoryWorker(ctx3); rec3 = cap(w3, "done", "failed")
    with fake_modules(prompts={"COMPACT_MEMORY_PROMPT": "s"}, llm={"send_to_lm_studio": lambda *a, **k: None},
                      utils={"strip_think_tags": lambda t: t, "strip_reasoning_leak": lambda t: t}):
        w3.run()
    check("compact_crash", rec3["failed"] == ["mem boom"])


# --------------------------------------------------------------- ReloadModelWorker

def test_reload_model_worker():
    Res = types.SimpleNamespace
    w = gui.ReloadModelWorker("http://x", "id", 8192, 90, "q8_0", True)
    rec = cap(w, "done")
    with fake_modules(lmstudio={"reload_model_full":
            lambda *a: Res(ok=True, message="reloaded", manual_settings="")}):
        w.run()
    check("reload_ok", rec["done"] == [(True, "reloaded", "")])
    w2 = gui.ReloadModelWorker("http://x", "id", 8192, 90, "q8_0", True)
    rec2 = cap(w2, "done")
    with fake_modules(lmstudio={"reload_model_full": lambda *a: (_ for _ in ()).throw(OSError("nope"))}):
        w2.run()
    check("reload_fail", rec2["done"][0][0] is False and "nope" in rec2["done"][0][1])


# --------------------------------------------------------------- RequestWorker

def _graph(final):
    return types.SimpleNamespace(invoke=lambda state: final)


def test_request_worker_text_and_empty():
    ctx = _ctx()
    final = {"final_answer": "hi", "messages": [{"role": "assistant", "content": "hi"}]}
    w = gui_workers.RequestWorker(ctx, _graph(final), {"messages": []}, text="hello")
    rec = cap(w, "recognized", "info", "done", "failed")
    with fake_modules(graph={"compact_history_if_needed": lambda ctx, msgs: msgs}):
        w.run()
    check("req_text_done", len(rec["done"]) == 1 and rec["done"][0]["final_answer"] == "hi")
    # empty text -> failed
    w2 = gui_workers.RequestWorker(ctx, _graph(final), {"messages": []}, text="")
    rec2 = cap(w2, "done", "failed")
    with fake_modules(graph={"compact_history_if_needed": lambda c, m: m}):
        w2.run()
    check("req_empty", rec2["failed"] == ["Empty input."])


def test_request_worker_audio_paths():
    ctx = _ctx()
    final = {"final_answer": "a", "messages": []}
    # audio transcribes to text
    gui_workers.transcribe_audio_array = lambda c, a: "spoken text"
    w = gui_workers.RequestWorker(ctx, _graph(final), {"messages": []}, audio=b"aud")
    rec = cap(w, "recognized", "done", "failed")
    with fake_modules(graph={"compact_history_if_needed": lambda c, m: m}):
        w.run()
    check("req_audio_recognized", rec["recognized"] == ["spoken text"] and len(rec["done"]) == 1)
    # audio transcribes empty -> failed
    gui_workers.transcribe_audio_array = lambda c, a: ""
    w2 = gui_workers.RequestWorker(ctx, _graph(final), {"messages": []}, audio=b"aud")
    rec2 = cap(w2, "done", "failed")
    with fake_modules(graph={"compact_history_if_needed": lambda c, m: m}):
        w2.run()
    check("req_audio_empty", len(rec2["failed"]) == 1)


def test_request_worker_use_db_rag():
    ctx = _ctx()
    final = {"final_answer": "doc answer", "messages": []}
    class FakeLib:
        def __init__(self): pass
        def corpus_scripts(self): return {"Cyrillic"}
        def close(self): pass
    lib_mod = {
        "Library": FakeLib,
        "cross_lingual_targets": lambda text, scripts: ["Russian"],
        "build_rag_prompt": lambda lib, text, k=None, extra_queries=None: ("AUGMENTED: " + text, "RAG note"),
    }
    w = gui_workers.RequestWorker(ctx, _graph(final), {"messages": []}, text="question", use_db=True, db_k=5)
    rec = cap(w, "info", "done", "failed")
    # _translate_query uses llm; provide it
    with fake_modules(library=lib_mod, graph={"compact_history_if_needed": lambda c, m: m},
                      llm={"send_to_lm_studio": lambda *a, **k: {"content": "вопрос"}}):
        w.run()
    check("req_db_done", len(rec["done"]) == 1)
    check("req_db_web_restored", ctx.web_search_enabled is True)  # restored in finally
    check("req_db_info", any("RAG note" in str(i) for i in rec["info"]))
    # RAG augmentation raising is swallowed (answers without retrieval)
    w2 = gui_workers.RequestWorker(ctx, _graph(final), {"messages": []}, text="q", use_db=True)
    rec2 = cap(w2, "done", "failed")
    with fake_modules(library={"Library": lambda: (_ for _ in ()).throw(OSError("no db"))},
                      graph={"compact_history_if_needed": lambda c, m: m}):
        w2.run()
    check("req_db_rag_fail_answers_anyway", len(rec2["done"]) == 1)


def test_request_worker_translate_query():
    ctx = _ctx()
    w = gui_workers.RequestWorker(ctx, None, {"messages": []}, text="x")
    with fake_modules(llm={"send_to_lm_studio": lambda *a, **k: {"content": "<think>t</think>перевод"}}):
        out = w._translate_query("hello", "Russian")
    check("translate_strips_think", out == "перевод", repr(out))
    with fake_modules(llm={"send_to_lm_studio": lambda *a, **k: (_ for _ in ()).throw(RuntimeError())}):
        out2 = w._translate_query("hello", "Russian")
    check("translate_error_none", out2 is None)


def test_request_worker_top_level_exception():
    ctx = _ctx(remember=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("mem fail")))
    w = gui_workers.RequestWorker(ctx, _graph({"messages": []}), {"messages": []}, text="hi")
    rec = cap(w, "done", "failed")
    with fake_modules(graph={"compact_history_if_needed": lambda c, m: m}):
        w.run()
    check("req_toplevel_fail", rec["failed"] == ["mem fail"])


# --------------------------------------------------------------- ScanWorker

def test_scan_worker():
    ctx = _ctx()
    class FakeLib:
        def __init__(self): pass
        def all_chunks(self, paths): return ["chunk1", "chunk2"]
        def close(self): pass
    def map_reduce_scan(chunks, q, mapf, reducef, progress=None, cancel=None):
        notes = [mapf(c, q) for c in chunks]
        progress(1, 2); progress(2, 2)
        return reducef([n for n in notes if n], q)
    lib_mod = {"Library": FakeLib, "map_reduce_scan": map_reduce_scan}
    w = gui_database_tab.ScanWorker(ctx, "find all names")
    rec = cap(w, "progress", "info", "done", "failed")
    with fake_modules(library=lib_mod,
                      llm={"send_to_lm_studio": lambda *a, **k: {"content": "- Alice\n- Bob"}}):
        w.run()
    check("scan_done", len(rec["done"]) == 1 and "Alice" in rec["done"][0], str(rec))
    # empty db
    w2 = gui_database_tab.ScanWorker(ctx, "q")
    rec2 = cap(w2, "done", "failed")
    with fake_modules(library={"Library": type("L", (), {"__init__": lambda s: None,
                      "all_chunks": lambda s, p: [], "close": lambda s: None})}):
        w2.run()
    check("scan_empty_db", len(rec2["failed"]) == 1)
    # cancelled mid-scan
    w3 = gui_database_tab.ScanWorker(ctx, "q"); w3.cancel()
    rec3 = cap(w3, "done", "failed", "info")
    def mrs_cancel(chunks, q, mapf, reducef, progress=None, cancel=None):
        mapf(chunks[0], q)  # exercises the cancel-guard NONE return
        return reducef([], q)
    with fake_modules(library={"Library": FakeLib, "map_reduce_scan": mrs_cancel},
                      llm={"send_to_lm_studio": lambda *a, **k: {"content": "x"}}):
        w3.run()
    check("scan_cancelled", rec3["failed"] == ["Scan cancelled."])
    # reducef with no notes -> "nothing relevant"
    w4 = gui_database_tab.ScanWorker(ctx, "q")
    rec4 = cap(w4, "done", "failed")
    def mrs_empty(chunks, q, mapf, reducef, progress=None, cancel=None):
        return reducef([], q)
    with fake_modules(library={"Library": FakeLib, "map_reduce_scan": mrs_empty},
                      llm={"send_to_lm_studio": lambda *a, **k: {"content": "NONE"}}):
        w4.run()
    check("scan_nothing_relevant", "nothing relevant" in rec4["done"][0])
    # top-level exception
    w5 = gui_database_tab.ScanWorker(ctx, "q")
    rec5 = cap(w5, "done", "failed")
    with fake_modules(library={"Library": lambda: (_ for _ in ()).throw(OSError("boom"))}):
        w5.run()
    check("scan_exception", rec5["failed"] == ["boom"])


def test_scan_worker_mapf_reducef_units():
    """Directly exercise mapf NONE-return and reducef join through map_reduce_scan."""
    ctx = _ctx()
    class FakeLib:
        def __init__(self): pass
        def all_chunks(self, paths): return ["c1", "c2"]
        def close(self): pass
    seq = iter([{"content": "NONE"}, {"content": "- real fact"}])
    def mrs(chunks, q, mapf, reducef, progress=None, cancel=None):
        notes = [mapf(c, q) for c in chunks]   # first NONE (dropped), second kept
        return reducef([n for n in notes if n], q)
    with fake_modules(library={"Library": FakeLib, "map_reduce_scan": mrs},
                      llm={"send_to_lm_studio": lambda *a, **k: next(seq, {"content": "final synth"})}):
        w = gui_database_tab.ScanWorker(ctx, "q"); rec = cap(w, "done", "failed")
        w.run()
    check("scan_mapf_none_dropped", len(rec["done"]) == 1, str(rec))


if __name__ == "__main__":
    _real_tx = gui_workers.transcribe_audio_array
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
        finally:
            gui_workers.transcribe_audio_array = _real_tx
    print(f"\n{len(fns)-failed}/{len(fns)} worker test functions passed "
          f"({sum(1 for _,c in RESULTS if c)}/{len(RESULTS)} checks)")
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)
