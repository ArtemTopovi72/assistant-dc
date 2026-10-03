"""Seventh supplement: precise remaining gui.py branches — worker log-except paths,
RequestWorker cross-lingual append, TransferWorker plan/identity excepts, and a
batch of AssistantWindow / MemoryCenterTab / ImageViewerDialog / SystemInfoTab
guard branches identified line-by-line from the coverage map.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_supplement7.py
"""
import os, sys, types, tempfile, threading, contextlib
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("USE_GUI", "0")
os.environ["GUI_REPORT_HTML"] = "0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _isolate_library  # noqa: F401  — never touch the live library.db
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import QApplication, QDialog, QInputDialog, QMessageBox
from PyQt5.QtWidgets import QFileDialog as _QFD
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtCore import Qt
import gui
import gui_workers
import gui_voice_tab

_app = QApplication.instance() or QApplication(sys.argv)
_TMP = tempfile.mkdtemp(prefix="guisupp7_")

def _png(name="s.png"):
    p = os.path.join(_TMP, name)
    img = QImage(40, 30, QImage.Format_RGB32); img.fill(QColor("#709060")); img.save(p)
    return p

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    assert cond, f"{name}: {detail}"

@contextlib.contextmanager
def fake_modules(**mods):
    saved = {}
    for name, mod in mods.items():
        saved[name] = sys.modules.get(name)
        if mod is None:
            sys.modules[name] = None
        else:
            m = types.ModuleType(name)
            for k, v in mod.items(): setattr(m, k, v)
            sys.modules[name] = m
    try: yield
    finally:
        for name, old in saved.items():
            if old is None: sys.modules.pop(name, None)
            else: sys.modules[name] = old

def cap(worker, *sig):
    out = {s: [] for s in sig}
    for s in sig:
        getattr(worker, s).connect(lambda *a, _s=s: out[_s].append(a if len(a) != 1 else a[0]))
    return out


# ---- worker log-except + cross-lingual + transfer excepts --------------------

def _img_ok(**over):
    base = {
        "fix_hands": lambda ctx, s, mask_override=None, engine="firered": _png("fh.png"),
        "edit_region_contained_via_firered": lambda *a, **k: _png("fa.png"),
        "log_edit_decision": lambda **k: (_ for _ in ()).throw(IOError("log fail")),
    }
    base.update(over); return base

def test_worker_log_excepts():
    ctx = types.SimpleNamespace()
    w = gui.FixHandsWorker(ctx, "src.png", mask_path="m.png")
    rec = cap(w, "done", "failed")
    with fake_modules(image=_img_ok()):
        w.run()
    check("fixhands_log_except", rec["done"] and rec["done"][0].endswith("fh.png"))
    w2 = gui.FixArtifactWorker(ctx, "src.png", "m.png", "fix")
    rec2 = cap(w2, "done", "failed")
    with fake_modules(image=_img_ok()):
        w2.run()
    check("fixartifact_log_except", rec2["done"] and rec2["done"][0].endswith("fa.png"))


def test_request_worker_crosslingual_append():
    ctx = types.SimpleNamespace(web_search_enabled=True, set_stage=lambda *a: None,
                                remember=lambda *a, **k: None, memory_text=lambda: "")
    final = {"final_answer": "a", "messages": []}
    graph = types.SimpleNamespace(invoke=lambda s: final)
    class FakeLib:
        def __init__(self): pass
        def corpus_scripts(self): return {"Cyrillic"}
        def close(self): pass
    lib = {"Library": FakeLib, "cross_lingual_targets": lambda t, s: ["Russian", "German"],
           "build_rag_prompt": lambda l, t, k=None, extra_queries=None: ("AUG " + t, "note")}
    w = gui_workers.RequestWorker(ctx, graph, {"messages": []}, text="q", use_db=True, db_k=3)
    # _translate_query returns a translation -> extra.append + info.emit("Also searching")
    w._translate_query = lambda text, lang: f"{text}-{lang}"
    rec = cap(w, "info", "done", "failed")
    with fake_modules(library=lib, graph={"compact_history_if_needed": lambda c, m: m}):
        w.run()
    check("req_crosslingual", any("Also searching" in str(i) for i in rec["info"]))


def test_transfer_worker_mask_and_identity_excepts():
    ctx = types.SimpleNamespace(is_cancelled=lambda: False)
    tgt = _png("tw7.png")
    class Ref:
        def __init__(self, p, r):
            self.path = p; self.role = r; self.effective_role = "object_source"; self.extracted_asset_path = None
        def is_extractable(self): return False
    im = {
        "ReferenceImage": Ref, "crop_to_mask": lambda p, m: None,
        "infer_reference_roles": lambda i, r: None, "extract_reference_asset": lambda c, r: None,
        "_ROLE_TARGET_REGION": {"object_source": "torso"},
        "_upload_image_to_comfy": lambda t, u: (_ for _ in ()).throw(RuntimeError("upload boom")),
        "_region_mask_file": lambda *a, **k: None, "COMFY_URL": "http://c",
        "build_edit_plan": lambda t, r, i: {"n_passes": 1, "steps": [
            {"roles": ["object_source"], "extract": False, "instruction": "s"}]},
        "assert_deliverable": lambda out, where=None, source_path=None: out,
    }
    # plan mode: _upload raises -> except at "plan: target mask failed"
    w = gui.TransferWorker(ctx, tgt, [(_png("r7.png"), None)], "x", "plan")
    rec = cap(w, "preview", "done", "failed")
    with fake_modules(image=im):
        w.run()
    check("tw_mask_except", len(rec["done"]) == 1)
    # contained mode: identity_cosine raises -> except pass
    im2 = dict(im, plan_and_execute_transfer=lambda *a, **k: _png("out7.png"),
               assert_deliverable=lambda out, where=None, source_path=None: out)
    w2 = gui.TransferWorker(ctx, tgt, [(_png("r7b.png"), None)], "x", "contained")
    rec2 = cap(w2, "done", "failed")
    with fake_modules(image=im2, identity_metrics={"identity_cosine": lambda a, b: (_ for _ in ()).throw(ValueError())}):
        w2.run()
    check("tw_identity_except", rec2["done"] and rec2["done"][0]["info"] == "")


# ---- window / tab guard branches --------------------------------------------

_orig_loader = gui.ModelLoader
def _win():
    gui.ModelLoader = type("NoopLoader", (_orig_loader,), {"start": lambda self: None, "run": lambda self: None})
    w = gui.AssistantWindow("m", True, "high")
    gui.ModelLoader = _orig_loader
    d = Path(_TMP) / "mem7"; d.mkdir(parents=True, exist_ok=True)
    w.ctx = types.SimpleNamespace(cancel_event=threading.Event(), memory_lock=threading.RLock(),
                                  reference_images=[], last_image_path=None, model_name="m",
                                  no_think=True, mic_disabled=False, tts_disabled=False,
                                  reasoning_effort="high", response_length="auto",
                                  active_memory_dir=d, save_memory=lambda *a, **k: None,
                                  load_memory=lambda *a, **k: None, session_memory=[])
    w.graph = object(); w.base_state = {"messages": []}
    w.stack.setCurrentWidget(w.dashboard); w._set_busy(False)
    return w

def _close(w):
    for a in ("worker", "compact_worker", "redraw_worker", "model_switch_worker", "research_worker",
              "_lib_build_worker", "_scan_worker", "transcribe_worker", "recorder", "cap", "vad_listener"):
        setattr(w, a, None)
    try: w.close()
    except Exception: pass


def test_load_pasted_qimage_save_fail():
    w = _win()
    fake_qimg = types.SimpleNamespace(isNull=lambda: False, save=lambda p, fmt: False)
    check("pasted_save_fail", w._load_pasted_qimage(fake_qimg) is False)
    _close(w)


def test_start_model_switch_in_progress():
    w = _win()
    w._active_model_id = "m"
    w.model_switch_worker = object()          # already in flight
    w._start_model_switch("different-model")  # -> "already in progress" branch
    check("switch_in_progress", w.model_switch_worker is not None)
    w.model_switch_worker = None
    _close(w)


def test_toggle_recording_short_audio():
    orig = gui_voice_tab.MicRecorder
    class ShortRec:
        def __init__(self, *a, **k): pass
        def start(self): pass
        def stop(self): return np.zeros(10, dtype=np.float32)   # < 0.5s
        def level(self): return 0.1
    gui_voice_tab.MicRecorder = ShortRec
    try:
        w = _win()
        w._toggle_recording()                 # start
        w._toggle_recording()                 # stop -> too short branch
        check("rec_too_short", w.recorder is None)
        _close(w)
    finally:
        gui_voice_tab.MicRecorder = orig


def test_on_vad_utterance_guards():
    w = _win()
    # vad_listener None -> early return
    w.vad_listener = None
    w._on_vad_utterance(np.zeros(int(gui.SAMPLE_RATE), dtype=np.float32))
    check("vad_utt_none", True)
    _close(w)


def test_prof_suggested_and_create_errors():
    import memory_center
    store = memory_center.MemoryStore(root=tempfile.mkdtemp(prefix="mc7_"))
    store.create_profile("default")
    host = types.SimpleNamespace(ctx=types.SimpleNamespace(active_memory_dir=store.profile_dir("default"),
                                 session_memory=[]), _set_status=lambda m: None,
                                 _memory_center_set_profile=lambda p: None)
    tab = gui.MemoryCenterTab(host); tab.store = store; tab.refresh_all()
    real = store.create_profile
    K = len(tab._SUGGESTED)
    # (a) create always returns True -> loop2 hits the append branch (2790->2791)
    store.create_profile = lambda name: True
    tab._prof_suggested(); check("prof_suggested_append", True)
    # (b) one suggestion blows up -> the except branch swallows it, the rest are
    # still created. The old version counted calls on the assumption of TWO loops
    # (a comprehension plus a guarded one); _prof_suggested is a single guarded
    # loop now, so the count could never exceed K and the check failed on a
    # refactor. Assert the BEHAVIOUR instead: one bad profile must not abort the
    # batch, and must not be reported as created.
    seen = []
    def cp(name):
        seen.append(name)
        if name == tab._SUGGESTED[0]:
            raise RuntimeError("boom")
        return True
    store.create_profile = cp
    toasts = []
    tab._toast = lambda m: toasts.append(m)
    tab._prof_suggested()
    check("prof_suggested_except_does_not_abort_batch", len(seen) == K,
          f"attempted {len(seen)} of {K}")
    check("prof_suggested_except_excludes_failed",
          toasts and tab._SUGGESTED[0] not in toasts[-1], str(toasts[-1:]))
    check("prof_suggested_except_keeps_the_rest",
          toasts and all(p in toasts[-1] for p in tab._SUGGESTED[1:]), str(toasts[-1:]))
    store.create_profile = real
    # _prof_create: create raises ValueError -> warning branch
    QInputDialog.getText = staticmethod(lambda *a, **k: ("Bad/Name", True))
    store.create_profile = lambda name: (_ for _ in ()).throw(ValueError("invalid"))
    ow = QMessageBox.warning; QMessageBox.warning = staticmethod(lambda *a, **k: None)
    try:
        tab._prof_create(); check("prof_create_valueerror", True)
    finally:
        QMessageBox.warning = ow; store.create_profile = real
    # _move_menu with no selection
    tab._sel = None; tab._move_menu(); check("move_menu_nosel", True)


def test_on_compact_done_writefail_and_success():
    o = {n: getattr(gui, n) for n in ("CompactMemoryWorker",)}
    try:
        w = _win()
        # write-fail: active_memory_dir is a FILE -> mkdir raises -> "save failed"
        badfile = os.path.join(_TMP, "notadir"); open(badfile, "w").write("x")
        w.ctx.active_memory_dir = Path(badfile)
        w._on_compact_done("summary text")
        check("compact_writefail", "save failed" in w.chat.toPlainText().lower())
        # success with memory_center_tab.refresh_all
        good = Path(_TMP) / "cok"; good.mkdir(parents=True, exist_ok=True)
        w.ctx.active_memory_dir = good
        w._on_compact_done("good summary")
        check("compact_success", (good / "summary.json").exists())
        _close(w)
    finally:
        pass


def test_save_copy_missing_path_valid_pix():
    p = _png("sc7.png")
    dlg = gui.ImageViewerDialog([p])
    dlg._paths = ["C:/no/such_missing.png"]     # path missing but _pix still valid
    dest = os.path.join(_TMP, "sc7out.png")
    _QFD.getSaveFileName = staticmethod(lambda *a, **k: (dest, ""))
    dlg._save_copy()
    check("save_copy_pix_fallback", os.path.exists(dest))
    dlg.close()


def test_runtime_ready_step_and_final_except():
    w = _win()
    import lmstudio; lmstudio.fetch_model = lambda url, mid: {"id": mid, "state": "loaded",
                                                             "compatibility_type": "gguf"}
    ctx = types.SimpleNamespace(cancel_event=threading.Event(), memory_lock=threading.RLock(),
                                model_name="m", no_think=True, mic_disabled=False, tts_disabled=False,
                                reasoning_effort="high", response_length="auto",
                                active_memory_dir=Path(_TMP) / "rr", save_memory=lambda *a, **k: None,
                                session_memory=5)   # int -> len() raises in the final block (except)
    (Path(_TMP) / "rr").mkdir(parents=True, exist_ok=True)
    ctx.stage_callback = None
    # make one step raise so the _step except path runs
    w.model_config_tab = types.SimpleNamespace(set_context=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("step boom")))
    w._on_runtime_ready(ctx, {"messages": []}, types.SimpleNamespace(invoke=lambda s: {}))
    check("runtime_step_except", w.ctx is ctx)
    _close(w)


def test_db_build_guards():
    import gui_database_tab
    o = {n: getattr(gui_database_tab, n) for n in ("LibraryBuildWorker",)}
    gui_database_tab.LibraryBuildWorker = type("LBW", (o["LibraryBuildWorker"],), {"start": lambda self: None})
    try:
        w = _win()
        # already running -> early return
        w._lib_build_worker = object()
        w._db_build(); check("db_build_running", w._lib_build_worker is not None)
        w._lib_build_worker = None
        # library.close raises -> swallowed
        f = os.path.join(_TMP, "d7.txt"); open(f, "w").write("x")
        w._db_add_list_item(f)
        w._library = types.SimpleNamespace(close=lambda: (_ for _ in ()).throw(OSError("locked")))
        w._db_build(); check("db_build_close_except", w._lib_build_worker is not None)
        w._lib_build_worker = None
        _close(w)
    finally:
        gui_database_tab.LibraryBuildWorker = o["LibraryBuildWorker"]


def test_clear_context_ctx_none_and_systeminfo_tools_error():
    w = _win()
    w.ctx = None
    w._clear_context(); check("clear_ctx_none", True)
    _close(w)
    # SystemInfoTab.refresh with tools import failing -> error branch
    tab = gui.SystemInfoTab()
    ctx = types.SimpleNamespace(model_name="m", no_think=False, tts_disabled=False, mic_disabled=False,
                                web_search_enabled=True, custom_personality_path=None,
                                custom_personality_text="", custom_ref_wav=None, session_memory=[],
                                active_memory_dir=Path(_TMP), models=None)
    tab._ctx = ctx; tab._base_url = "x"
    with fake_modules(tools=None):
        tab.refresh()
    check("sysinfo_tools_error", "error loading tools" in tab._text.toPlainText())


if __name__ == "__main__":
    _saved = {"gof": _QFD.getOpenFileName, "gsf": _QFD.getSaveFileName, "gt": QInputDialog.getText}
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
    _QFD.getOpenFileName = staticmethod(_saved["gof"])
    _QFD.getSaveFileName = staticmethod(_saved["gsf"])
    QInputDialog.getText = staticmethod(_saved["gt"])
    print(f"\n{len(fns)-failed}/{len(fns)} supplement7 functions passed "
          f"({sum(1 for _,c in RESULTS if c)}/{len(RESULTS)} checks)")
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)
