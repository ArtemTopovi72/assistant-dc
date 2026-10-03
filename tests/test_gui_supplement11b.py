"""Supplement 11b: TransferTab busy-guard + gather-no-refs + MemoryCenter
row-select None-guard. Split from supplement11.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_supplement11b.py
"""
import os, sys, types, tempfile, threading
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("USE_GUI", "0")
os.environ["GUI_REPORT_HTML"] = "0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _isolate_library  # noqa: F401  — never touch the live library.db
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage, QColor
import gui

_app = QApplication.instance() or QApplication(sys.argv)
_TMP = tempfile.mkdtemp(prefix="guisupp11b_")

def _png(name="s.png"):
    p = os.path.join(_TMP, name)
    img = QImage(40, 30, QImage.Format_RGB32); img.fill(QColor("#204050")); img.save(p)
    return p

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name)
    assert cond, name


def test_transfer_run_busy_gather_and_mc_rowselect():
    host = types.SimpleNamespace(ctx=types.SimpleNamespace(reference_images=[], last_image_path=None,
                                 cancel_event=threading.Event()),
                                 images_panel=types.SimpleNamespace(add_image=lambda p: None),
                                 _add_system=lambda m: None)
    tab = gui.TransferTab(host)
    class RunningWorker:
        def isRunning(self): return True
    tab.worker = RunningWorker()
    before = tab.status.text()
    tab._run("plan")   # busy -> immediate return
    check("transfer_busy11", tab.status.text() == before)
    tab.worker = None
    tab._add_row(_png("gr11.png")); tab._rows[0]["combo"].setCurrentIndex(0)
    check("gather_norefs11", tab._gather() is None)
    import memory_center
    store = memory_center.MemoryStore(root=tempfile.mkdtemp(prefix="mc11_")); store.create_profile("default")
    store.add("default", "row", etype="fact", source="manual", importance=50)
    mh = types.SimpleNamespace(ctx=types.SimpleNamespace(active_memory_dir=store.profile_dir("default"),
                               session_memory=[]), _set_status=lambda m: None,
                               _memory_center_set_profile=lambda p: None)
    mc = gui.MemoryCenterTab(mh); mc.store = store; mc.refresh_all()
    if mc.table.rowCount():
        store.get = lambda prof, rid: None
        mc.table.clearSelection(); mc.table.selectRow(0); mc._on_row_selected()
    check("mc_rowselect_none", True)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try: fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
    print(str(len(fns) - failed) + "/" + str(len(fns)) + " supplement11b passed (" +
          str(sum(1 for _, c in RESULTS if c)) + "/" + str(len(RESULTS)) + " checks)")
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)
