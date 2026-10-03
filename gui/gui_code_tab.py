"""Code tab: the sandbox, driven from the desktop.

The same working folder, the same tools and the same agent loop Telegram uses --
this is a second surface onto `code_sandbox` / `tool_code_handlers`, not a second
implementation. That matters more here than usual: the whole point of the
sandbox is that it is the ONE place untrusted files are opened, so a desktop
panel that reached the filesystem its own way would be a hole with a nice
window on it.

What it offers: pick whose folder to work in (the desktop's own, or any chat's),
see what is in it, read a file, drop files in, ask for work in plain words, and
reset the folder when something goes wrong.

The agent call is a QThread. A tool round is tens of seconds of LLM plus
subprocesses, and this project has lost afternoons to Qt slots raising on the
worker thread -- so nothing here touches a widget from `run()`, and every
outcome comes back as a signal.
"""
import os
from pathlib import Path

from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QMessageBox, QPushButton,
                             QSplitter, QTextEdit, QVBoxLayout, QWidget)

import code_sandbox as _cs
import sandbox_access as _sa
from gui_common import MUTED, _ScopedCtx, _section
from ui_scale import px

import logging
logger = logging.getLogger("assistant.gui")

DESKTOP_KEY = "desktop"          # the owner's own folder, always available


class _SandboxCtx(_ScopedCtx):
    """A scoped context that also carries the sandbox -- locally.

    _ScopedCtx forwards writes to the REAL context on purpose (throttle
    bookkeeping belongs there). Setting ctx.sandbox through it would therefore
    hand the main chat the coding tools as a side effect of opening this tab,
    which is exactly the kind of quiet privilege spread the grant system exists
    to prevent. These two attributes stay on the view.
    """

    def __init__(self, ctx, cancel_event, sandbox, user):
        super().__init__(ctx, cancel_event)
        object.__setattr__(self, "sandbox", sandbox)
        object.__setattr__(self, "sandbox_user", user)

    def __setattr__(self, name, value):
        if name in ("sandbox", "sandbox_user"):
            object.__setattr__(self, name, value)
            return
        super().__setattr__(name, value)


class _Owner:
    """The desktop user, at `code` level -- deliberately NOT `host`.

    `host` means "run scripts straight on this machine, unisolated", which
    code_runner falls back to only when an administrator has said so for a
    specific account. Granting it here looked reasonable (the owner is sitting
    at the machine and could open a terminal anyway) and was wrong: this panel
    can work in ANY chat's folder, so a file a stranger uploaded to the bot
    would be driving unisolated execution on the owner's computer through an
    LLM. At `code` level the runner uses the container or refuses -- which is
    the guarantee the sandbox is for. Host execution stays where it was: an
    explicit per-user grant in the Telegram admin panel.
    """
    prefs = {"sandbox": _sa.CODE}


class CodeWorker(QThread):
    """One turn of the real agent loop, off the UI thread."""
    done = pyqtSignal(dict, list)         # the finished state, tool names
    failed = pyqtSignal(str)

    def __init__(self, ctx, text, history):
        super().__init__()
        self.ctx = ctx
        self.text = text
        self.history = history

    def run(self):
        calls = []
        import graph
        _orig = graph.execute_tool

        def _rec(ctx_, state_, name, args):
            calls.append(name)
            return _orig(ctx_, state_, name, args)

        graph.execute_tool = _rec
        try:
            state = graph.personality_node(
                self.ctx,
                {"messages": list(self.history), "user_input": self.text,
                 "image_data": None})
            self.done.emit(dict(state), calls)
        except Exception as exc:
            logger.exception("Code tab agent turn failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            graph.execute_tool = _orig


class CodeTab(QWidget):
    def __init__(self, host):
        super().__init__()
        self.host = host
        self.worker = None
        self._cwd = ""                    # relative to the sandbox root
        # The conversation, carried between turns. "Unpack it" then "now add
        # the block id" then "pack it back up" is how this work is actually
        # done, and starting from an empty history each time made every
        # follow-up a non-sequitur -- the agent re-listed the folder and asked
        # what the user meant.
        self._history = []
        self._last_artifact = ""
        import threading
        self._cancel = threading.Event()

        root = QVBoxLayout(self)
        root.setContentsMargins(px(6), px(6), px(6), px(6))
        root.setSpacing(px(8))
        root.addWidget(_section("Code sandbox"))

        hint = QLabel("A working folder the assistant can open files in: "
                      "archives, source, configs, photos. Drop something in, "
                      "say what to do with it in plain words, and it unpacks, "
                      "reads, edits and packs the result back up. Code only "
                      "ever runs in a container; without one it refuses rather "
                      "than running on this machine.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{MUTED};")
        root.addWidget(hint)

        pick = QHBoxLayout()
        pick.addWidget(QLabel("Folder:"))
        self.who = QComboBox()
        # Switching folders starts a new conversation: the history talks about
        # files that are not in the new folder, and an agent told "now pack it"
        # would be reasoning about someone else's.
        self.who.currentIndexChanged.connect(self._on_folder_changed)
        pick.addWidget(self.who, 1)
        self.refresh_btn = QPushButton("↻"); self.refresh_btn.setObjectName("ghost")
        self.refresh_btn.setToolTip("Re-read the folder list")
        self.refresh_btn.clicked.connect(self._refresh_all)
        pick.addWidget(self.refresh_btn)
        self.add_btn = QPushButton("＋ Add files")
        self.add_btn.clicked.connect(self._add_files)
        pick.addWidget(self.add_btn)
        self.reset_btn = QPushButton("Reset"); self.reset_btn.setObjectName("ghost")
        self.reset_btn.setToolTip("Delete everything in this folder")
        self.reset_btn.clicked.connect(self._reset)
        pick.addWidget(self.reset_btn)
        root.addLayout(pick)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(px(4))
        # Where we are, and the way back out. An unpacked modpack is a tree and
        # a flat listing of its top level is a listing of one folder name.
        self.where_lbl = QLabel("/")
        self.where_lbl.setStyleSheet(f"color:{MUTED};")
        self.up_btn = QPushButton("↑ Up"); self.up_btn.setObjectName("ghost")
        self.up_btn.clicked.connect(self._go_up)
        self.up_btn.setEnabled(False)
        crumb = QHBoxLayout()
        crumb.addWidget(self.up_btn)
        crumb.addWidget(self.where_lbl, 1)
        ll.addLayout(crumb)
        self.files = QListWidget()
        self.files.currentTextChanged.connect(self._show_file)
        # Double-click descends. Single click still previews, so opening a file
        # never costs a round trip the user did not ask for.
        self.files.itemDoubleClicked.connect(self._open_entry)
        ll.addWidget(self.files, 1)
        split.addWidget(left)
        self.viewer = QTextEdit(); self.viewer.setReadOnly(True)
        self.viewer.setPlaceholderText("Pick a file to look inside it.")
        split.addWidget(self.viewer)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([px(220), px(520)])
        root.addWidget(split, 1)

        ask = QHBoxLayout()
        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText(
            "e.g. unpack the jar, add morevillagers:trading_table to the "
            "break_protected/medium tag, pack it back up")
        self.prompt.returnPressed.connect(self._run)
        ask.addWidget(self.prompt, 1)
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._run)
        ask.addWidget(self.run_btn)
        self.open_btn = QPushButton("Show file"); self.open_btn.setObjectName("ghost")
        self.open_btn.setToolTip("Open the folder containing what the last turn produced")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._reveal)
        ask.addWidget(self.open_btn)
        self.clear_btn = QPushButton("New chat"); self.clear_btn.setObjectName("ghost")
        self.clear_btn.setToolTip("Forget the conversation so far (the files stay)")
        self.clear_btn.clicked.connect(self._clear_chat)
        ask.addWidget(self.clear_btn)
        root.addLayout(ask)

        self.status = QLabel("Idle.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color:{MUTED};")
        root.addWidget(self.status)

        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setPlaceholderText("What the agent did will appear here.")
        self.log.setMinimumHeight(px(120))
        root.addWidget(self.log)

        # Dropping a file on the panel is the same gesture as sending one to
        # the bot, and it is what people try first.
        self.setAcceptDrops(True)

        self._refresh_all()

    # ---- drag and drop ------------------------------------------------------

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls()
                 if u.isLocalFile() and u.toLocalFile()]
        if paths:
            self._take_files(paths)
            event.acceptProposedAction()

    # ---- folders -----------------------------------------------------------

    def _known_folders(self) -> list:
        """The desktop folder plus every chat folder that already exists.

        Read off the disk rather than the user store, so this works with the
        bot stopped -- the same reason the Telegram tab's admin controls do.
        """
        out = [DESKTOP_KEY]
        try:
            base = Path(_cs.SANDBOX_BASE)
            if base.exists():
                out += sorted(p.name for p in base.iterdir()
                              if p.is_dir() and p.name != DESKTOP_KEY)
        except Exception:
            logger.exception("could not list sandbox folders")
        return out

    def _refresh_all(self):
        want = self.who.currentText() or DESKTOP_KEY
        self.who.blockSignals(True)
        self.who.clear()
        self.who.addItems(self._known_folders())
        idx = self.who.findText(want)
        self.who.setCurrentIndex(idx if idx >= 0 else 0)
        self.who.blockSignals(False)
        self._refresh_files()

    def _on_folder_changed(self, _index=0):
        self._cwd = ""
        self._history = []
        self._last_artifact = ""
        self.open_btn.setEnabled(False)
        self._refresh_files()

    def _box(self):
        return _cs.sandbox_for(self.who.currentText() or DESKTOP_KEY)

    def _entry_name(self, text: str) -> str:
        """The path an entry stands for. list_dir decorates its lines --
        "notes.txt (128 B)" for a file, "data/" for a directory -- and the
        decoration has to come back off before the sandbox is asked for it."""
        return text.rsplit(" (", 1)[0].rstrip("/\\").strip()

    def _go_up(self):
        self._cwd = str(Path(self._cwd).parent) if self._cwd not in ("", ".") else ""
        if self._cwd == ".":
            self._cwd = ""
        self._refresh_files()

    def _open_entry(self, item):
        rel = self._entry_name(item.text())
        try:
            box = self._box()
            if box.resolve(rel).is_dir():
                self._cwd = rel
                self._refresh_files()
                return
        except Exception as exc:
            self.status.setText(str(exc))
            return
        self._show_file(item.text())

    def _refresh_files(self):
        self.files.clear()
        try:
            entries = self._box().list_dir(self._cwd or ".")
        except Exception as exc:
            self.status.setText(f"Could not read the folder: {exc}")
            self._cwd = ""
            return
        self.where_lbl.setText("/" + (self._cwd or ""))
        self.up_btn.setEnabled(bool(self._cwd))
        self.files.addItems(entries)
        self.status.setText("Empty." if not entries
                            else f"{len(entries)} entries.")

    def _show_file(self, name: str):
        if not name:
            return
        target = self._entry_name(name)
        try:
            self.viewer.setPlainText(self._box().read_text(target))
        except Exception as exc:
            # Binary, too large, a directory: the sandbox already words all
            # three, and its wording is the same one the model is given.
            self.viewer.setPlainText(str(exc))

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add files to the sandbox")
        if paths:
            self._take_files(paths)

    def _take_files(self, paths):
        """Copy files in. One path for the picker and for a drop, so a dropped
        file cannot take a route the picked one does not."""
        box = self._box()
        added = 0
        for p in paths:
            try:
                # Through resolve(), by BASENAME: the same containment check the
                # Telegram upload path uses. A local file picker cannot smuggle
                # a traversal, but there is no second way in for anything.
                dest = box.resolve(Path(p).name)
                dest.write_bytes(Path(p).read_bytes())
                added += 1
            except Exception as exc:
                self.status.setText(f"{Path(p).name}: {exc}")
        self._refresh_files()
        if added:
            self.status.setText(f"Added {added} file(s).")

    def _reset(self):
        who = self.who.currentText() or DESKTOP_KEY
        if QMessageBox.question(
                self, "Reset the sandbox",
                f"Delete everything in the “{who}” folder? This cannot be undone.",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel) != QMessageBox.Yes:
            return
        try:
            removed = self._box().reset()
        except Exception as exc:
            self.status.setText(f"Reset failed: {exc}")
            return
        self.viewer.clear()
        self._refresh_files()
        self.status.setText(f"Reset — removed {removed} object(s).")

    # ---- the agent ---------------------------------------------------------

    def _run(self):
        if self.worker is not None and self.worker.isRunning():
            self.status.setText("Still working on the previous request.")
            return
        text = self.prompt.text().strip()
        if not text:
            return
        ctx = getattr(self.host, "ctx", None)
        if ctx is None:
            self.status.setText("The assistant is not started yet.")
            return
        self._cancel.clear()
        self.run_btn.setEnabled(False)
        self.status.setText("Working…")
        self.log.append(f"> {text}")
        self.worker = CodeWorker(
            _SandboxCtx(ctx, self._cancel, self._box(), _Owner()),
            text, self._history)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()
        self.prompt.clear()

    def _on_done(self, state: dict, calls: list):
        self.run_btn.setEnabled(True)
        self._history = list(state.get("messages") or [])
        if calls:
            self.log.append("tools: " + ", ".join(calls))
        self.log.append(str(state.get("final_answer") or "") or "(no answer)")
        # The artifact contract: a tool that produced a file puts its path in
        # one of these. A "successful" turn that sets none delivered nothing,
        # and on this surface that is the difference between a packed archive
        # the user can find and a packed archive they cannot.
        made = next((str(state.get(k)) for k in
                     ("document_path", "image_path", "video_path")
                     if state.get(k)), "")
        self._last_artifact = made
        self.open_btn.setEnabled(bool(made))
        if made:
            self.log.append("file: " + made)
        # Refresh FIRST: it writes its own count into the status line, so
        # setting the outcome before it meant the one thing worth reading --
        # the name of the file that was just produced -- was replaced by
        # "3 entries." before the user could see it.
        self._refresh_files()
        self.status.setText(f"Done — {Path(made).name}" if made else "Done.")

    def _on_failed(self, why: str):
        self.run_btn.setEnabled(True)
        self.log.append("failed: " + why)
        self.status.setText(why)

    def _reveal(self):
        target = self._last_artifact
        if not target or not Path(target).exists():
            self.status.setText("That file is no longer there.")
            self.open_btn.setEnabled(False)
            return
        try:
            # The platform's own file manager, via the stdlib -- no dependency
            # and no shell string to quote wrong.
            os.startfile(str(Path(target).parent))       # noqa: S606 (Windows)
        except Exception as exc:
            self.status.setText(f"Could not open the folder: {exc}")

    def _clear_chat(self):
        self._history = []
        self.log.clear()
        self.status.setText("New conversation. The files are untouched.")

    def cancel(self):
        """Named `cancel` because that is the name the window looks for.

        AssistantWindow._cancel_current walks a list of tabs and calls
        `.cancel()` on each; the worker itself is found by _thread_hosts /
        _live_threads, which inspect this widget's attributes -- so `self.worker`
        has to stay an attribute for the close handler to join it.
        """
        self._cancel.set()
