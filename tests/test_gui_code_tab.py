"""The Code tab, driven headless.

Builds the REAL panel and calls the REAL slots. Qt swallows nothing on purpose:
a slot that raises under offscreen looks like a button that did nothing, and
this project has lost an afternoon to exactly that.

The sandbox base is redirected FIRST -- the real one lives under runtime/ and a
test must not leave folders in it, nor read the user's actual chat files.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_code_tab.py
"""
import os, sys, tempfile
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("USE_GUI", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

from pathlib import Path
import code_sandbox as CS
CS.SANDBOX_BASE = Path(tempfile.mkdtemp(prefix="codetab_root_"))

import gui_code_tab as CT
import sandbox_access as A
from PyQt5.QtWidgets import QApplication, QMessageBox

_app = QApplication.instance() or QApplication(sys.argv)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


def _read(box, rel):
    """Reading a file that is not there is a FAILED CHECK, not a traceback:
    a crash here stops every check after it from running at all."""
    try:
        return box.read_text(rel)
    except Exception as exc:
        return "<%s>" % exc


class _Host:
    def __init__(self, ctx=None):
        self.ctx = ctx


# ── it builds, and it starts on the desktop folder ───────────────────────────
tab = CT.CodeTab(_Host())
check("the panel builds", hasattr(tab, "files") and hasattr(tab, "prompt"))
check("it starts on the owner's own folder",
      tab.who.currentText() == CT.DESKTOP_KEY, tab.who.currentText())
check("an empty folder says so", "Empty" in tab.status.text(), tab.status.text())


# ── a chat's folder shows up once it exists ──────────────────────────────────
CS.sandbox_for(4242).write_text("notes.txt", "hello sandbox")
tab._refresh_all()
check("a chat folder is offered",
      "4242" in [tab.who.itemText(i) for i in range(tab.who.count())],
      [tab.who.itemText(i) for i in range(tab.who.count())])

tab.who.setCurrentText("4242")
names = [tab.files.item(i).text() for i in range(tab.files.count())]
check("its files are listed", any(n.startswith("notes.txt") for n in names), names)


# ── clicking a file shows its contents, decoration and all ───────────────────
# list_dir writes "notes.txt (13 B)"; the panel has to strip that back off or
# every click is a "does not exist".
tab._show_file(names[0])
check("clicking a file shows the contents",
      "hello sandbox" in tab.viewer.toPlainText(), tab.viewer.toPlainText()[:80])

CS.sandbox_for(4242).write_text("sub/inner.txt", "deep")
tab._refresh_files()
dirs = [tab.files.item(i).text() for i in range(tab.files.count())]
tab._show_file([d for d in dirs if d.startswith("sub")][0])
check("clicking a directory explains itself instead of raising",
      "directory" in tab.viewer.toPlainText().lower(), tab.viewer.toPlainText()[:80])


# ── adding files goes through the sandbox, not around it ─────────────────────
src = Path(tempfile.mkdtemp()) / "dropped.txt"
src.write_text("from the picker", encoding="utf-8")
CT.QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([str(src)], ""))
tab._add_files()
check("a picked file lands in the folder",
      _read(CS.sandbox_for(4242), "dropped.txt") == "from the picker")

# The basename is all that is used, so a path from anywhere stays contained.
outside = Path(tempfile.mkdtemp()) / "escape.txt"
outside.write_text("x", encoding="utf-8")
CT.QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([str(outside)], ""))
tab._add_files()
inside = CS.sandbox_for(4242).root / "escape.txt"
check("it is written inside the sandbox root", inside.exists())
check("and nowhere above it",
      not (Path(CS.SANDBOX_BASE) / "escape.txt").exists())


# A DROPPED file must take the same route as a picked one.
from PyQt5.QtCore import QMimeData, QUrl, QPoint
from PyQt5.QtGui import QDropEvent
_drop_src = Path(tempfile.mkdtemp()) / "dropped_in.txt"
_drop_src.write_text("dragged", encoding="utf-8")
_mime = QMimeData(); _mime.setUrls([QUrl.fromLocalFile(str(_drop_src))])
from PyQt5.QtCore import Qt as _Qt
tab.dropEvent(QDropEvent(QPoint(1, 1), _Qt.CopyAction, _mime,
                         _Qt.LeftButton, _Qt.NoModifier))
def _read(box, rel):
    """Reading a file that is not there is a FAILED CHECK, not a traceback:
    a crash here stops every check after it from running at all."""
    try:
        return box.read_text(rel)
    except Exception as exc:
        return "<%s>" % exc

check("a dropped file lands in the folder",
      _read(CS.sandbox_for(4242), "dropped_in.txt") == "dragged",
      _read(CS.sandbox_for(4242), "dropped_in.txt")[:60])


# ── reset asks first ─────────────────────────────────────────────────────────
CT.QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Cancel)
tab._reset()
check("cancelling the reset keeps the files",
      CS.sandbox_for(4242).list_dir("."))

CT.QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
tab._reset()
check("confirming empties the folder", not CS.sandbox_for(4242).list_dir("."))
check("and only that folder", CS.sandbox_for(4242).root.exists())


# ── the agent is not reachable before the assistant exists ───────────────────
tab.prompt.setText("do something")
tab._run()
check("with no assistant it says so instead of crashing",
      "not started" in tab.status.text(), tab.status.text())


# ── the sandbox rides on a PRIVATE view of the context ───────────────────────
# Setting ctx.sandbox on the shared context would hand the main chat the coding
# tools as a side effect of opening this tab.
class _Ctx:
    def __init__(self):
        self.sandbox = None
        self.sandbox_user = None

import threading
real = _Ctx()
view = CT._SandboxCtx(real, threading.Event(), CS.sandbox_for(99), CT._Owner())
check("the view carries the sandbox", view.sandbox is not None)
check("the real context does not", real.sandbox is None, real.sandbox)
check("the desktop user may run code", A.may_run_code(CT._Owner()))
# The panel can open ANY chat's folder, so a file uploaded by a stranger would
# be what drives execution. At `code` level that means a container or a refusal;
# `host` here would mean running it unisolated on this machine.
check("but NOT unisolated on the host",
      not A.allow_host_execution(CT._Owner()))

import code_runner as R
_denied = R.run_python(CS.sandbox_for(99), "print(1)", allow_host=False)
check("with no container, execution is refused rather than run here",
      R.docker_available() or (not _denied.ok and "unavailable" in _denied.output),
      _denied)

# A write that is NOT the sandbox still belongs to the real context.
view.some_bookkeeping = 7
check("other writes still reach the real context",
      getattr(real, "some_bookkeeping", None) == 7)


# ── the window can find and stop it ──────────────────────────────────────────
# Both are hand-maintained lists in the window, and a tab missing from either
# is a worker that outlives QApplication (segfault) or a Stop button that does
# not reach it.
# Driven, not grepped: a `False and` in front of a condition leaves a source
# check green, and this project has already been fooled by one.
import gui_busy_state

class _Win(gui_busy_state.BusyStateMixin):
    def __init__(self, tab):
        self.code_tab = tab

_win = _Win(tab)
check("the window counts this tab among its thread hosts",
      tab in _win._thread_hosts(), _win._thread_hosts())
check("the tab exposes cancel(), the name the window calls",
      callable(getattr(tab, "cancel", None)))
check("the worker stays an attribute, which is how it is discovered",
      "worker" in vars(tab))
tab.cancel()
check("cancel sets the tab's own token", tab._cancel.is_set())



# -- walking into a folder ---------------------------------------------------
# list_dir decorates its lines, and an unpacked modpack is a tree: a flat
# listing of its top level is a listing of one folder name.
CS.sandbox_for(4242).write_text("mod/data/tags/medium.json", '{"values": []}')
tab.who.setCurrentText("4242")
tab._cwd = ""
tab._refresh_files()


class _Item:
    def __init__(self, t): self._t = t
    def text(self): return self._t


_dir_line = [tab.files.item(i).text() for i in range(tab.files.count())
             if tab.files.item(i).text().startswith("mod")][0]
tab._open_entry(_Item(_dir_line))
check("double-clicking a folder walks into it", tab._cwd == "mod", tab._cwd)
check("the listing follows",
      any("data" in tab.files.item(i).text() for i in range(tab.files.count())),
      [tab.files.item(i).text() for i in range(tab.files.count())])
check("the crumb says where we are", "mod" in tab.where_lbl.text(), tab.where_lbl.text())
check("Up becomes available", tab.up_btn.isEnabled())

tab._open_entry(_Item("mod/data/"))
check("it keeps descending", tab._cwd == "mod/data", tab._cwd)
tab._go_up()
check("Up climbs one level", tab._cwd == "mod", tab._cwd)
tab._go_up()
check("and back to the root", tab._cwd == "", repr(tab._cwd))
check("where Up is disabled again", not tab.up_btn.isEnabled())

tab._cwd = "mod/data/tags"
tab._refresh_files()
_file_line = [tab.files.item(i).text() for i in range(tab.files.count())][0]
tab._open_entry(_Item(_file_line))
check("double-clicking a FILE shows it, and does not walk",
      tab._cwd == "mod/data/tags" and "values" in tab.viewer.toPlainText(),
      (tab._cwd, tab.viewer.toPlainText()[:40]))

tab._cwd = "mod"
tab.who.setCurrentText(CT.DESKTOP_KEY)
check("switching folders returns to the root", tab._cwd == "", repr(tab._cwd))
tab.who.setCurrentText("4242")
tab._cwd = ""
tab._refresh_files()


# -- the conversation is carried, and dropped when it stops applying ---------
# "Unpack it" -> "now add the block id" -> "pack it up" is how this work goes;
# a fresh history each turn made every follow-up a non-sequitur.
tab._history = []
tab._on_done({"messages": [{"role": "user", "content": "unpack it"},
                           {"role": "assistant", "content": "done"}],
              "final_answer": "done"}, ["unpack_archive"])
check("the turn's history is kept for the next one", len(tab._history) == 2,
      tab._history)
check("the tools that ran are shown", "unpack_archive" in tab.log.toPlainText())

# An artifact is what the user actually came for: a "successful" turn that
# produced no path delivered nothing.
_made = CS.sandbox_for(4242).root / "fixed.zip"
_made.write_bytes(b"PK" + bytes(20))
tab._on_done({"messages": [], "final_answer": "packed",
              "document_path": str(_made)}, ["pack_archive"])
check("a produced file is surfaced", tab.open_btn.isEnabled())
check("and named in the status", "fixed.zip" in tab.status.text(), tab.status.text())

_made.unlink()
tab._reveal()
check("a file that has since gone says so rather than opening nothing",
      "no longer there" in tab.status.text(), tab.status.text())

tab._history = [{"role": "user", "content": "x"}]
tab._on_folder_changed()
check("switching folders starts a new conversation", tab._history == [])
check("and forgets the artifact", not tab.open_btn.isEnabled())

tab._history = [{"role": "user", "content": "x"}]
tab._clear_chat()
check("New chat drops the history", tab._history == [])
check("but leaves the files alone", CS.sandbox_for(4242).root.exists())


print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(0 if BAD == 0 else 1)
