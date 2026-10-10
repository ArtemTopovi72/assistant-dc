"""Document-database tab: file list, index build, scan and retrieval knobs.

The Database tab owns a whole small workflow — choose files, build a hybrid
index, watch progress, report stats, and expose the two knobs (use-db, top-k)
that change how retrieval behaves at query time. It shares nothing with the
rest of AssistantWindow except the chat log and the busy gate.

Needs from the host: _add_assistant, _add_system, _set_busy, _set_status,
_on_failed and _maybe_drain_queue.
"""
import logging
import os
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QAbstractItemView, QCheckBox, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QSlider, QVBoxLayout, QWidget
from gui_common import ACCENT2, MUTED, _section
from gui_workers import LibraryBuildWorker, ScanWorker
from ui_scale import px

logger = logging.getLogger("assistant")


class DatabaseTabMixin:
    def _refresh_search_panel(self):
        if self.ctx is None:
            return
        # Show every search this session, newest first, not just the last call —
        # each entry's text already starts with "Query: ..." (see tools._handle_search),
        # so the queries themselves are visible alongside their results, not just
        # the final search's results overwriting the ones before it.
        searches = [it for it in self.ctx.session_memory if it.get("kind") == "search"]
        if searches:
            sep = "\n" + "─" * 40 + "\n"
            self.search_view.setPlainText(
                sep.join(it.get("text", "") for it in reversed(searches)))
    # ---- Ultra Search / deep research ----
    # ---- document database (RAG) ----------------------------------------- #

    def _get_library(self):
        """Lazy GUI-thread reader connection to the knowledge layer.

        Goes through the knowledge_client boundary (knowledge_api.KnowledgeClient),
        which defaults to the in-process backend — i.e. library.default_library()
        — so this is the same connection it always was.
        """
        if self._library is None:
            import knowledge_client
            self._library = knowledge_client.open_library(default=True)
        return self._library

    def _build_database_tab(self):
        """The Database tab: upload documents, build the hybrid search index, and
        see what's indexed. Building runs on a worker thread with a progress bar."""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(_section("Documents"))
        hint = QLabel("Upload large text documents (TXT, Markdown, PDF, EPUB), then press "
                      "Build Database to index them. Turn on “Use Database” (left panel) to "
                      "ask questions answered from their contents.")
        hint.setWordWrap(True); hint.setStyleSheet(f"color:{MUTED};")
        lay.addWidget(hint)

        self.db_file_list = QListWidget()
        self.db_file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.db_file_list.setToolTip("Documents to include in the database. Already-indexed "
                                     "files are skipped on rebuild (unless changed).")
        lay.addWidget(self.db_file_list, 1)

        btns = QHBoxLayout()
        add_btn = QPushButton("➕  Add files"); add_btn.setObjectName("ghost")
        add_btn.clicked.connect(self._db_add_files)
        rm_btn = QPushButton("✕  Remove"); rm_btn.setObjectName("ghost")
        rm_btn.clicked.connect(self._db_remove_files)
        clear_btn = QPushButton("🗑  Clear DB"); clear_btn.setObjectName("ghost")
        clear_btn.clicked.connect(self._db_clear)
        btns.addWidget(add_btn); btns.addWidget(rm_btn); btns.addWidget(clear_btn)
        lay.addLayout(btns)

        self.db_build_btn = QPushButton("🛠  Build Database")
        self.db_build_btn.setToolTip("Extract, chunk, and index every listed document "
                                     "(keyword BM25 + semantic embeddings).")
        self.db_build_btn.clicked.connect(self._db_build)
        lay.addWidget(self.db_build_btn)

        self.db_scan_chk = QCheckBox("Whole-book scan for the next question (slow, thorough)")
        self.db_scan_chk.setToolTip(
            "For exhaustive 'find all X' questions. With Use Database ON, your next question "
            "reads EVERY passage of the book and aggregates an answer, instead of the fast "
            "top-passages search. Complete but slow (many model calls) — use it when the "
            "normal search misses things that are in the book.")
        lay.addWidget(self.db_scan_chk)

        # How many retrieved passages to feed the model per question (Use Database mode).
        krow = QHBoxLayout()
        self.db_k_slider = QSlider(Qt.Horizontal)
        self.db_k_slider.setRange(1, 100)
        self.db_k_slider.setValue(25)
        self.db_k_slider.setToolTip(
            "Number of passages retrieved and given to the model for each question in "
            "Use Database mode. More = better recall but more context used. Default 25.")
        self.db_k_slider.valueChanged.connect(self._on_db_k_changed)
        self.db_k_label = QLabel("25")
        self.db_k_label.setMinimumWidth(px(28))
        krow.addWidget(QLabel("Passages per question:"))
        krow.addWidget(self.db_k_slider, 1)
        krow.addWidget(self.db_k_label)
        lay.addLayout(krow)

        self.db_progress = QProgressBar(); self.db_progress.hide()
        lay.addWidget(self.db_progress)
        self.db_status = QLabel(""); self.db_status.setWordWrap(True)
        lay.addWidget(self.db_status)
        self.db_stats_label = QLabel(""); self.db_stats_label.setStyleSheet(f"color:{MUTED};")
        self.db_stats_label.setWordWrap(True)
        lay.addWidget(self.db_stats_label)

        self._db_load_existing()
        self._db_update_stats()
        return w

    def _db_load_existing(self):
        """Show documents already in the database so the user sees current contents."""
        try:
            for d in self._get_library().documents():
                self._db_add_list_item(d.get("path") or d.get("title"),
                                       label=d.get("title"), indexed=True)
        except Exception:
            logger.exception("database tab: failed to load existing documents")

    def _db_add_list_item(self, path, *, label=None, indexed=False):
        if not path:
            return
        ap = os.path.abspath(path) if os.path.exists(path) else path
        for i in range(self.db_file_list.count()):
            if self.db_file_list.item(i).data(Qt.UserRole) == ap:
                return  # already listed
        it = QListWidgetItem(("✓ " if indexed else "📄  ") + (label or os.path.basename(ap)))
        it.setData(Qt.UserRole, ap)
        it.setToolTip(ap + ("  (indexed)" if indexed else ""))
        self.db_file_list.addItem(it)

    def _db_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add documents to the database", "",
            "Documents (*.txt *.text *.log *.md *.markdown *.pdf *.epub);;All files (*)")
        for p in paths:
            self._db_add_list_item(p)

    def _db_remove_files(self):
        for it in self.db_file_list.selectedItems():
            self.db_file_list.takeItem(self.db_file_list.row(it))

    def _db_listed_paths(self):
        out = []
        for i in range(self.db_file_list.count()):
            p = self.db_file_list.item(i).data(Qt.UserRole)
            if p and os.path.exists(p):
                out.append(p)
        return out

    def _db_build(self):
        if self._lib_build_worker is not None:
            return
        paths = self._db_listed_paths()
        if not paths:
            self.db_status.setText("Add at least one document file first.")
            return
        # Release the GUI reader connection so the build worker has the DB file to
        # itself (avoids SQLite "database is locked" between the two connections).
        if self._library is not None:
            try:
                self._library.close()
            except Exception:
                pass
            self._library = None
        self.db_build_btn.setEnabled(False)
        self.db_progress.setRange(0, 0)   # indeterminate until the first progress tick
        self.db_progress.show()
        self.db_status.setText("Building… extracting and indexing documents.")
        w = LibraryBuildWorker(paths)
        w.progress.connect(self._on_db_progress)
        w.done.connect(self._on_db_built)
        w.failed.connect(self._on_db_failed)
        w.finished.connect(self._on_db_finished)
        self._lib_build_worker = w
        w.start()

    def _on_db_progress(self, stage, done, total):
        self.db_progress.setRange(0, max(1, total))
        self.db_progress.setValue(done)
        if stage == "extract":
            self.db_status.setText(f"Extracting documents… {done}/{total}")
        else:
            self.db_status.setText(f"Embedding passages… {done}/{total}")

    def _on_db_built(self, stats):
        self._last_db_stats = stats
        errs = stats.get("errors") or []
        cov = int((stats.get("embed_coverage") or 0) * 100)
        msg = (f"✅ Database ready — {stats.get('documents', '?')} documents, "
               f"{stats.get('chunks', '?')} passages, {cov}% embedded.")
        if not stats.get("embed_available"):
            msg += (" Embedding model offline → keyword (BM25) search only. Load the "
                    "embedding model in LM Studio and Build again for semantic search.")
        if errs:
            msg += f"  ⚠ files with errors: {len(errs)}: " + "; ".join(errs[:3])
        self.db_status.setText(msg)

    def _on_db_failed(self, err):
        self.db_status.setText(f"❌ Build failed: {err}")

    def _on_db_finished(self):
        self._lib_build_worker = None
        self.db_build_btn.setEnabled(True)
        self.db_progress.hide()
        self._set_busy(False)  # unblocks the queue — messages typed during the build now drain
        # Drop the stale reader connection so the next query sees the freshly built data.
        if self._library is not None:
            try:
                self._library.close()
            except Exception:
                pass
            self._library = None
        self._db_update_stats()
        # The build held _busy() True; messages typed during it are queued — run them now.
        self._maybe_drain_queue()

    def _db_clear(self):
        if QMessageBox.question(
                self, "Clear database",
                "Remove ALL indexed documents from the database?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            n = self._get_library().purge_all()
            self.db_status.setText(f"Documents removed from the database: {n}.")
        except Exception as exc:
            self.db_status.setText(f"Clear failed: {exc}")
        self.db_file_list.clear()
        self._db_update_stats()

    def _db_update_stats(self):
        try:
            st = self._get_library().stats()
            online = "online" if st.get("embed_available") else "offline"
            self.db_stats_label.setText(
                f"Indexed: {st.get('documents', 0)} documents · "
                f"{st.get('chunks', 0)} passages · {int(st.get('embed_coverage', 0) * 100)}% "
                f"embedded · embedding model {online}")
        except Exception:
            self.db_stats_label.setText("Indexed: (database unavailable)")

    def _on_db_k_changed(self, value):
        self.db_k_label.setText(str(value))

    def _db_k(self):
        """Slider-chosen number of passages per Use-Database question (default 25)."""
        sl = getattr(self, "db_k_slider", None)
        return sl.value() if sl is not None else 25

    def _start_scan(self, text):
        """Whole-book map-reduce scan for an exhaustive question (Database tab checkbox)."""
        if self._scan_worker is not None:
            # Silent before: a second question during a scan simply disappeared.
            self._add_system('A database scan is already running — wait for it to finish.')
            return
        paths = self._db_listed_paths() or None
        w = ScanWorker(self.ctx, text, paths=paths)
        w.progress.connect(self._on_scan_progress)
        w.info.connect(self._add_system)
        w.done.connect(self._on_scan_done)
        w.failed.connect(self._on_failed)
        w.finished.connect(self._on_scan_finished)
        self._scan_worker = w
        self._set_busy(True)
        self.stage.set_stage("Scanning the book")
        self._add_system("🔍 Whole-book scan started — reading every passage. This can take "
                         "a while; press Stop to abort.")
        w.start()

    def _on_scan_progress(self, done, total):
        self._set_status(f"Scanning the book… batch {done}/{total}")

    def _on_scan_done(self, answer):
        self._add_assistant(answer or "(no answer produced)")

    def _on_scan_finished(self):
        self._scan_worker = None
        self._set_busy(False)
        self._set_status("")

    def _toggle_usedb(self):
        self.usedb_on = not self.usedb_on
        if self.usedb_on:
            try:
                empty = self._get_library().is_empty()
            except Exception:
                empty = False
            self.usedb_btn.setText("📚  Use Database: ON")
            self.usedb_btn.setStyleSheet(f"background:{ACCENT2}; color:#06231d;")
            if empty:
                self._set_status("Use Database ON — but the database is empty. Add documents "
                                 "in the Database tab and press Build.")
                self._show_tab("database", True)
                self._focus_tab("database")
            else:
                self._set_status("Use Database ON — your questions are answered from your documents.")
                self._show_tab("database", True)
                self._focus_tab("database")
        else:
            self.usedb_btn.setText("📚  Use Database: OFF")
            self.usedb_btn.setStyleSheet("")
            self._set_status("Use Database OFF.")
