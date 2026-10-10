"""The staged-task queue: its bar, its list model and its drain state machine.

A mixin, not free functions: the queue is UI state on THIS window and its
state machine reads the same busy/cancelled flags the worker plumbing sets.
Extracted from gui.py because nothing here knows what a task IS — it only
decides which text gets dispatched next.

The mixin expects from the host class: `_add_system`, `_busy`,
`_dispatch_user_text`, `_set_status` and the queue state attributes
(`_task_queue`, `_queue_paused`, `_running_task`, `_running_dispatched`,
`_turn_had_error`, `_turn_cancelled` and the three counters), all created in
AssistantWindow.__init__.
"""
import gui_i18n
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget,
)

from ui_scale import px


class TaskQueueMixin:
    """Queue bar construction, list rendering and the drain state machine."""

    def _build_queue_bar(self):
        """The task-queue column beside the chat: a list of staged actions plus
        reorder / remove / clear / pause controls. Collapsible, so the chat can take
        the whole width when nothing is queued (toggle with _toggle_queue)."""
        wrap = QWidget()
        wlay = QVBoxLayout(wrap); wlay.setContentsMargins(0, 0, 0, 0); wlay.setSpacing(px(2))
        # The header row is always visible; it carries the show/hide toggle so the queue
        # can be folded away even while it holds tasks.
        head = QHBoxLayout(); head.setSpacing(px(6))
        self.q_toggle = QPushButton("▾  Queue"); self.q_toggle.setObjectName("ghost")
        self.q_toggle.setCheckable(True); self.q_toggle.setChecked(True)
        self.q_toggle.setToolTip("Hide the task queue to give the chat more room.")
        self.q_toggle.toggled.connect(self._toggle_queue)
        self.q_head_lbl = QLabel(""); self.q_head_lbl.setObjectName("chip")
        head.addWidget(self.q_toggle); head.addWidget(self.q_head_lbl); head.addStretch(1)
        wlay.addLayout(head)

        self._queue_body = QWidget()
        bar = QVBoxLayout(self._queue_body); bar.setContentsMargins(0, 0, 0, 0)
        self.queue_list = QListWidget()
        self.queue_list.setObjectName("queuelist")
        self.queue_list.setToolTip("Queued tasks — run top-to-bottom whenever the assistant "
                                   "is free. Select one to reorder or remove it.")
        bar.addWidget(self.queue_list, 1)
        col = QHBoxLayout(); col.setSpacing(px(2))
        self.q_up = QPushButton("▲"); self.q_up.setObjectName("ghost")
        self.q_up.setToolTip("Move the selected task up"); self.q_up.clicked.connect(lambda: self._queue_move(-1))
        self.q_down = QPushButton("▼"); self.q_down.setObjectName("ghost")
        self.q_down.setToolTip("Move the selected task down"); self.q_down.clicked.connect(lambda: self._queue_move(1))
        self.q_del = QPushButton("✕"); self.q_del.setObjectName("ghost")
        self.q_del.setToolTip("Remove the selected task"); self.q_del.clicked.connect(self._queue_remove)
        self.q_clear = QPushButton("Clear"); self.q_clear.setObjectName("ghost")
        self.q_clear.setToolTip("Remove all queued tasks"); self.q_clear.clicked.connect(self._queue_clear)
        self.q_pause = QPushButton("⏸"); self.q_pause.setCheckable(True); self.q_pause.setObjectName("ghost")
        self.q_pause.setToolTip("Pause/resume automatic execution of the queue")
        self.q_pause.toggled.connect(self._queue_set_paused)
        for b in (self.q_up, self.q_down, self.q_del, self.q_clear, self.q_pause):
            col.addWidget(b)
        bar.addLayout(col)
        self._queue_bar_widgets = (self.q_up, self.q_down, self.q_del, self.q_clear, self.q_pause)
        wlay.addWidget(self._queue_body, 1)
        self._refresh_queue_ui()
        return wrap

    def _toggle_queue(self, shown):
        """Fold the queue body away (chat grows) or bring it back."""
        self._queue_body.setVisible(shown)
        self.q_toggle.setText("▾  Queue" if shown else "▸  Queue")
        self._refresh_queue_ui()
        # Hand the freed width to the chat. Deferred: the column's sizeHint only
        # drops once Qt has processed the hidden body.
        QTimer.singleShot(0, self._rebalance_queue_split)

    def _rebalance_queue_split(self):
        """Shrink the queue column to its natural width, give the rest to the chat."""
        sp = getattr(self, "splitter", None)
        if sp is None or sp.count() < 2:
            return
        sizes = sp.sizes()
        want = sp.widget(1).sizeHint().width() if not self._queue_body.isVisible() else max(sizes[1], self._default_splitter_sizes()[1])
        sp.setSizes([max(px(200), sum(sizes) - want), want])

    _Q_ICON = {"pending": "•", "running": "▶", "done": "✓", "failed": "✗",
               "cancelled": "⃠"}

    def _refresh_queue_ui(self):
        """Re-render the queue list + button enabled-states from self._task_queue, showing
        each task's execution state."""
        if not hasattr(self, "queue_list"):
            return
        sel = self.queue_list.currentRow()
        self.queue_list.clear()
        n_pending = 0
        for item in self._task_queue:
            st = item.get("status", "pending")
            if st == "pending":
                n_pending += 1
                label = f"{n_pending}."
            else:
                label = self._Q_ICON.get(st, "•")
            self.queue_list.addItem(f"{label} {item['text']}")
        if 0 <= sel < len(self._task_queue):
            self.queue_list.setCurrentRow(sel)
        has_pending = n_pending > 0
        for b in getattr(self, "_queue_bar_widgets", ()):
            if b is not self.q_pause:
                b.setEnabled(has_pending)
        self.q_pause.setText("▶" if self._queue_paused else "⏸")
        self.q_pause.setToolTip("Resume queue" if self._queue_paused else "Pause queue")
        done = f"  ·  ✓{self._completed_count}" if self._completed_count else ""
        failed = f" ✗{self._failed_count}" if self._failed_count else ""
        stopped = f" ⃠{self._cancelled_count}" if self._cancelled_count else ""
        # the pieces translated here: glued into one template they split arbitrarily
        running = " " + gui_i18n.tr("— running") if self._running_task else ""
        paused = " " + gui_i18n.tr("— PAUSED") if self._queue_paused else ""
        # Header summary so a folded queue still shows it holds work.
        if hasattr(self, "q_head_lbl"):
            collapsed = not self.q_toggle.isChecked()
            if n_pending or self._running_task:
                self.q_head_lbl.setText(
                    f"{n_pending} pending{' ▶' if self._running_task else ''}"
                    f"{paused}")
                self.q_head_lbl.setVisible(True)
            else:
                self.q_head_lbl.setVisible(collapsed and bool(self._completed_count))
                self.q_head_lbl.setText(f"✓{self._completed_count}" if self._completed_count else "")
        self.queue_list.setToolTip(
            (f"{n_pending} pending{running}{paused}"
             f"{done}{failed}{stopped}. Runs top-to-bottom when the assistant is free.")
            if (self._task_queue or self._completed_count) else
            "Task queue is empty. Messages sent while the assistant is busy are staged here.")

    def _enqueue(self, text: str, *, announce=True):
        text = (text or "").strip()
        if not text:
            return
        self._task_queue.append({"text": text, "status": "pending"})
        self._refresh_queue_ui()
        if announce:
            n = sum(1 for it in self._task_queue if it.get("status") == "pending")
            self._add_system(f"⏳ Queued ({n} pending): {text}")
        # If we're idle and not paused, start draining immediately.
        if not self._busy():
            self._maybe_drain_queue()

    def _pending_indices(self):
        return [i for i, it in enumerate(self._task_queue) if it.get("status") == "pending"]

    def _queue_move(self, delta):
        """Reorder among PENDING tasks only (the running task can't be moved)."""
        i = self.queue_list.currentRow()
        if not (0 <= i < len(self._task_queue)):
            return
        if self._task_queue[i].get("status") != "pending":
            return
        j = i + delta
        if 0 <= j < len(self._task_queue) and self._task_queue[j].get("status") == "pending":
            self._task_queue[i], self._task_queue[j] = self._task_queue[j], self._task_queue[i]
            self._refresh_queue_ui()
            self.queue_list.setCurrentRow(j)

    def _queue_remove(self):
        """Remove the selected task — but never the one currently running."""
        i = self.queue_list.currentRow()
        if 0 <= i < len(self._task_queue) and self._task_queue[i].get("status") == "pending":
            self._task_queue.pop(i)
            self._refresh_queue_ui()

    def _queue_clear(self):
        """Clear all PENDING tasks (a running task keeps executing)."""
        before = len(self._task_queue)
        self._task_queue = [it for it in self._task_queue if it.get("status") == "running"]
        if len(self._task_queue) != before:
            self._refresh_queue_ui()
            self._add_system("Queue cleared (pending tasks removed).")

    def _queue_set_paused(self, paused):
        self._queue_paused = bool(paused)
        self._refresh_queue_ui()
        if not paused and not self._busy():
            self._maybe_drain_queue()

    def _maybe_drain_queue(self):
        """Execution-state machine for the queue. Re-invoked after every worker finishes
        (via _set_busy(False)) and after enqueue/resume.

        1. If a task was running and we're now idle, finalize it (done/failed) and drop it.
        2. If not paused and idle, mark the next pending task running and dispatch it.
        """
        # 1) finalize a finished running task.
        # _running_dispatched matters: dispatch is deferred by one event-loop tick,
        # so between "marked running" and "worker started" the window is still idle.
        # Without the flag a re-entry in that gap (another _enqueue, a resume) reads
        # "running task + not busy" as "it finished", tallies a task that never ran
        # and promotes the next one — so a burst of enqueues fires them all at once.
        if self._running_task is not None and self._running_dispatched and not self._busy():
            # Not ctx.cancel_event: _worker_finished clears it before _set_busy(False)
            # gets here, so by now a cancelled turn looks identical to a clean one.
            cancelled = self._turn_cancelled
            failed = self._turn_had_error
            self._running_task["status"] = ("cancelled" if cancelled else
                                            "failed" if failed else "done")
            if cancelled:
                self._cancelled_count += 1
                # The user asked for this to stop. Launching the next task the
                # instant this one winds down makes Stop look broken — they would
                # have to keep clicking to get out of a long queue. Hold the queue
                # instead; the ⏸/▶ button resumes it.
                if not self._queue_paused and any(
                        it.get("status") == "pending" for it in self._task_queue):
                    self._add_system("Stopped. The queue is on hold — press ▶ in the "
                                     "queue bar to continue, or ✕ to drop the rest.")
                self._queue_paused = True
            elif failed:
                self._failed_count += 1
            else:
                self._completed_count += 1
            try:
                self._task_queue.remove(self._running_task)
            except ValueError:
                pass
            self._running_task = None
            self._running_dispatched = False
            self._refresh_queue_ui()
        # 2) start the next pending task
        if self._queue_paused or self._busy() or self._running_task is not None:
            return
        # Nothing can answer without an LLM, and the drain would otherwise walk
        # the whole queue, refusing every item in turn and filling the chat with
        # the same notice N times. Hold instead, say it once, and let ▶ resume
        # after a model is chosen.
        if getattr(self, "_model_missing", None) is not None and self._model_missing() and any(it.get("status") == "pending" for it in self._task_queue):
            self._queue_paused = True
            self._refresh_queue_ui()
            self._add_system(getattr(self, "NO_MODEL_NOTICE", 'No model loaded.')
                             + ' The queue is paused — press ▶ after picking a model.')
            return
        nxt = next((it for it in self._task_queue if it.get("status") == "pending"), None)
        if nxt is None:
            return
        nxt["status"] = "running"
        self._running_task = nxt
        self._running_dispatched = False
        self._turn_had_error = False
        self._turn_cancelled = False
        self._refresh_queue_ui()
        # Defer one event-loop tick so the just-finished worker is fully torn down.
        QTimer.singleShot(0, lambda t=nxt["text"]: self._dispatch_queued(t))

    def _dispatch_queued(self, text):
        """Run a queued task. Separate from _dispatch_user_text so the queue can
        record that the deferred dispatch actually happened."""
        self._running_dispatched = True
        self._dispatch_user_text(text)
