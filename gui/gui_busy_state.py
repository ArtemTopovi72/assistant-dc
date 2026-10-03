"""Is the window busy, which threads are alive, and how they get joined.

A mixin, not free functions: "busy" is a property of THIS window — its own
worker plus whatever the tabs are running. Extracted from gui.py because it is
the one place that answers that question, and because the teardown join is
safety code that deserves to be readable on its own.

_thread_hosts / _live_threads / _join_live_threads exist because closeEvent
used to join five of roughly twelve workers, so the rest were still running
when the QApplication went away — a segfault, not an exception. They enumerate
the hosts instead of naming threads one by one, which is what stops the list
from going stale the next time a tab grows a worker.

config is read at CALL time inside _tab_job_running: GUI_SERIALIZE_GPU_JOBS is
a switch a suite flips, and binding it at import would freeze it.
"""
import logging
import time

from PyQt5.QtCore import QThread

logger = logging.getLogger("assistant.gui")


class BusyStateMixin:
    """The busy gate, live-thread enumeration and the teardown join."""

    def _busy(self) -> bool:
        """True if the assistant can't take a turn right now, so incoming messages must be
        queued rather than run. Covers: a blocking background op (request, model switch,
        redraw, research, transcribe, compact), the document-database build, AND the
        initial model load (graph not ready yet). The queue auto-drains when this clears."""
        if self.graph is None:                     # still loading the model
            return True
        if getattr(self, "_lib_build_worker", None) is not None:   # building the database
            return True
        # A Model Config "Apply" unloads and reloads the LLM. _set_busy alone only
        # greys buttons out; _busy() is what stops a message from being dispatched
        # into a model that is mid-reload.
        if getattr(getattr(self, "model_config_tab", None), "_reload_worker", None) is not None:
            return True
        if any(w is not None for w in (self.worker, self.model_switch_worker,
                                       self.redraw_worker, self.research_worker,
                                       self.transcribe_worker, self.compact_worker,
                                       getattr(self, "_scan_worker", None))):
            return True
        return self._tab_job_running()

    def _reject_if_busy(self, what: str) -> bool:
        """True (and say so) when `what` cannot start because something is running.

        A sweep of the launch handlers found sixteen of them shaped
        `if self._busy(): return` -- a button press that produced no worker, no
        message and no log line. From the user's side that is indistinguishable
        from a dead button, and it is most likely to happen exactly when the app
        is slowest and they are most likely to click again.

        Deliberately returns a bool rather than raising: every caller is a Qt
        slot, and an exception in one of those is a native abort with no
        traceback (see crash_diag).
        """
        if not self._busy():
            return False
        try:
            if getattr(self, "graph", None) is None:
                # _busy() is also True during the initial load. "Занят, останови
                # кнопкой ⛔" is wrong there -- there is nothing to stop, and
                # the honest instruction is to wait.
                self._add_system('Still loading — %s can start in a few seconds.' % what)
            else:
                self._add_system('Busy — %s starts after the current task. Stop it with ⛔ if you do not need it.'
                                 % what)
        except Exception:
            logger.exception("_reject_if_busy could not report")
        return True

    def _tab_job_running(self) -> bool:
        """True while a TAB is running its own long job (transfer, storyboard,
        madhouse, memory compaction).

        The chat turn and those jobs share one 24GB GPU: the turn holds the LLM
        resident while a transfer or storyboard render loads a diffusion model on
        top of it. Nothing stopped them overlapping, so a message sent mid-render
        raced the render for VRAM. This does not refuse the message — _busy() makes
        it QUEUE, with its chip in the queue bar, and the queue auto-drains the
        moment the tab job ends.

        Deliberately asks isRunning() rather than testing the attribute for None:
        a worker that died without clearing its own reference would otherwise wedge
        the chat permanently, and there is no done-handler to trust here.
        """
        import config as _cfg
        if not getattr(_cfg, "GUI_SERIALIZE_GPU_JOBS", True):
            return False
        for host in self._thread_hosts():
            if host is self:
                continue          # window workers are covered above
            for _name, val in list(vars(host).items()):
                if isinstance(val, QThread):
                    try:
                        if val.isRunning():
                            return True
                    except RuntimeError:
                        pass      # already destroyed C++-side
        return False

    # ---- background-thread lifecycle --------------------------------------- #
    def _thread_hosts(self):
        """Every object that parks a QThread on an attribute. Tabs run their own
        workers (transfer, storyboard, madhouse, model-config, memory-center), and
        those are just as capable of outliving the window as the main-window ones."""
        hosts = [self]
        for attr in ("transfer_tab", "storyboard_tab", "madhouse_tab",
                     "model_config_tab", "memory_center_tab", "code_tab",
                     "voice_clone_tab", "mashup_tab"):
            h = getattr(self, attr, None)
            if h is not None:
                hosts.append(h)
        return hosts

    def _live_threads(self):
        """All QThreads currently running anywhere in the window. Discovered by
        inspection rather than a hand-maintained list — every previous version of
        that list went stale the moment a new worker was added."""
        out, seen = [], set()
        for host in self._thread_hosts():
            for _name, val in list(vars(host).items()):
                if isinstance(val, QThread) and id(val) not in seen:
                    seen.add(id(val))
                    try:
                        if val.isRunning():
                            out.append(val)
                    except RuntimeError:
                        pass      # already destroyed C++-side
        return out

    def _join_live_threads(self, timeout_ms: int = 5000):
        """Ask every running worker to stop, then wait for it. Returns the ones
        that were still running when the budget ran out."""
        threads = self._live_threads()
        for t in threads:
            cancel = getattr(t, "cancel", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception:
                    logger.exception("worker cancel failed: %s", type(t).__name__)
        deadline = time.time() + timeout_ms / 1000.0
        for t in threads:
            remaining = max(0, int((deadline - time.time()) * 1000))
            try:
                t.wait(max(remaining, 50))
            except RuntimeError:
                pass
        stuck = []
        for t in threads:
            try:
                if t.isRunning():
                    stuck.append(type(t).__name__)
            except RuntimeError:
                pass
        # A thread that refuses to stop must not be allowed to fire a slot into
        # torn-down widgets; severing its signals is the last line of defence.
        for t in threads:
            try:
                t.disconnect()
            except (TypeError, RuntimeError):
                pass
        if stuck:
            logger.warning("closeEvent: workers still running after %dms: %s",
                           timeout_ms, ", ".join(stuck))
        return stuck

    def _set_busy(self, busy, status=""):
        # The input box, Send and Add-to-queue stay live even while busy so the user can
        # stage follow-up tasks; submitting while busy enqueues instead of dropping.
        for w in (self.input, self.send_btn):
            w.setEnabled(True)
        for w in (self.cam_btn, self.redraw_btn, self.style_preset_btn, self.fixhands_btn,
                  self.fixartifact_btn, self.whole_frame_btn):
            w.setEnabled(not busy)
        # Retry also requires a prior hand fix to exist.
        self.retryhands_btn.setEnabled(not busy and bool(getattr(self, "_last_handfix_source", None)))
        mic_ok = not busy and not (self.ctx and self.ctx.mic_disabled)
        self.talk_btn.setEnabled(mic_ok)
        self.capture_btn.setEnabled(not busy and self.cap is not None)
        # Stop is the one control that is enabled *only* while something is running.
        self.stop_btn.setEnabled(busy)
        if status:
            self._set_status(status)
        if not busy:
            self._maybe_drain_queue()
