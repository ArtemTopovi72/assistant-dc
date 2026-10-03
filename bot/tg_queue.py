"""Work queue, cancellation and the worker/watchdog loops.

Split out of tg_bot.py. Owns the path a message takes from "accepted" to
"running": the debounce buffer, _enqueue_item, the consumer threads, the
per-chat and per-task cancellation bookkeeping (_request_stop, _cancel_task,
_is_cancelled, _stop_requested_after, _mark_interruptible), the crash-recovery
in-flight journal, the poll watchdog and the queue statistics the GUI reads.

_INFLIGHT_FILE and friends are read as tg_bot.<name> rather than imported by
value -- redirect_data_dir() rebinds them at runtime, and a by-value import
here would have pinned this module to the LIVE data directory even in a test
that redirected it.
"""
from __future__ import annotations

import json
import os
import queue as _pyqueue
import threading
import turn_trace
import time
from pathlib import Path


class QueueMixin:
    def _run_busy(self, chat_id: int, fn, *args) -> None:
        """Run fn(*args) on a daemon thread with the chat marked busy until it
        returns. Work started outside the queue (weather lookup, voice-clone
        prep and synthesis) otherwise left the turn looking finished at its
        "Checking…" line (live journeys #7, #38, 2026-09-25)."""
        with self._task_lock:
            self._chat_busy[chat_id] = self._chat_busy.get(chat_id, 0) + 1

        def run():
            try:
                fn(*args)
            finally:
                with self._task_lock:
                    n = self._chat_busy.get(chat_id, 1) - 1
                    if n <= 0:
                        self._chat_busy.pop(chat_id, None)
                    else:
                        self._chat_busy[chat_id] = n
        turn_trace.spawn(run, name=f"chat-work-{chat_id}")

    def get_queue_stats(self) -> dict:
        return {
            "depth":        self._backend.depth(),
            "active_chats": dict(self._active_stages),
            "backend":      self._backend.name(),
        }

    def _watchdog_loop(self):
        """Keep the poll and consumer threads alive, and unstick hung tasks."""
        interval = max(5, tg_bot._cfg_int("TG_WATCHDOG_S", 15))
        task_max = max(60, tg_bot._cfg_int("TG_TASK_MAX_S", 3600))
        while self._running:
            time.sleep(interval)
            if not self._running:
                break
            try:
                # 1) poll thread
                if self._poll_thread is None or not self._poll_thread.is_alive():
                    tg_bot.logger.error("WATCHDOG: poll thread is dead — restarting")
                    self._activity.log(0, "error", "[watchdog] poll thread restarted")
                    self._poll_beat = time.monotonic()
                    self._poll_thread = threading.Thread(
                        target=self._poll_loop, daemon=True, name="tg-poll")
                    self._poll_thread.start()
                elif time.monotonic() - self._poll_beat > max(120, tg_bot._POLL_TIMEOUT * 4):
                    # Alive but not completing requests: almost always a socket
                    # wedged past its timeout. Logged, not killed — a thread cannot
                    # be killed safely, and the next request usually recovers.
                    tg_bot.logger.warning("WATCHDOG: no successful getUpdates for %.0fs",
                                   time.monotonic() - self._poll_beat)

                # 2) consumer threads
                for i, t in enumerate(list(self._consumers)):
                    if t.is_alive():
                        continue
                    tg_bot.logger.error("WATCHDOG: consumer %d is dead — restarting", i)
                    self._activity.log(0, "error",
                                       f"[watchdog] consumer {i} restarted")
                    nt = threading.Thread(target=self._consumer_loop, daemon=True,
                                          name=f"tg-consumer-{i}")
                    nt.start()
                    self._consumers[i] = nt

                # 3) runaway task — cancel cooperatively so the queue drains
                with self._task_lock:
                    stale = [(tid, started) for tid, started in self._task_started.items()
                             if time.monotonic() - started > task_max]
                for tid, started in stale:
                    tg_bot.logger.error("WATCHDOG: task %s running for %.0fs — cancelling",
                                 tid[:8], time.monotonic() - started)
                    self._activity.log(0, "error",
                                       f"[watchdog] task {tid[:8]} exceeded {task_max}s")
                    with self._task_lock:
                        self._cancelled.add(tid)
                        self._task_started.pop(tid, None)
                        ev = self._task_cancels.get(tid)
                    # The task runs on its OWN scoped context (_execute_task),
                    # not the shared desktop ctx — setting the shared event here
                    # used to abort whatever the desktop assistant was doing
                    # while leaving the actual runaway task running forever.
                    if ev is not None:
                        try: ev.set()
                        except Exception: pass
            except Exception:
                tg_bot.logger.exception("watchdog iteration failed")

    def _write_inflight(self) -> None:
        """Persist the set of tasks currently executing OR merely queued.

        A "running" entry means a crash killed an active graph.invoke() — the
        pre-existing case. A "pending" entry means the task was pushed onto
        the backend but never popped: with the default InMemoryBackend that
        queue lives only in this process's RAM, so a crash there is just as
        fatal to the task as a crash mid-execution, even though nothing was
        ever "running". Both are recorded so a restart is visible either way.
        """
        try:
            with self._task_lock:
                data = ([dict(t.to_dict(), _state="running")
                          for tasks in self._running_task.values() for t in tasks]
                         + [dict(t.to_dict(), _state="pending")
                            for t in self._pending_journal.values()])
            # Atomic: two threads writing at once used to be able to leave a torn
            # file, and recovery then read [] and silently lost every task.
            path = tg_bot._INFLIGHT_FILE
            tmp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except Exception as exc:
            tg_bot.logger.debug("inflight write failed: %s", exc)

    def _recover_inflight(self) -> None:
        """Notify users whose task was killed by a restart, and offer a retry.

        Without this a crash mid-task left the user staring at a "⚙️ Working…"
        status message that would never change — the most confusing possible
        failure mode, because the bot looks alive and simply never answers.

        A "pending" (queued-but-not-yet-popped) entry is only ever meaningful
        here for a non-durable backend (InMemoryBackend): a Redis/Kafka-backed
        queue keeps its own copy outside this process, so a task still sitting
        there at restart will simply be popped and run normally by the fresh
        consumer threads once they start — reporting it "interrupted" here
        would be an outright lie (the user would see BOTH an "interrupted,
        retry?" message AND, moments later, the real answer). Only the
        in-memory backend actually loses a pending task across a restart, so
        only that backend's leftover "pending" entries get the same
        interrupted/retry treatment as a genuinely running one.
        """
        try:
            if not tg_bot._INFLIGHT_FILE.exists():
                return
            data = json.loads(tg_bot._INFLIGHT_FILE.read_text(encoding="utf-8") or "[]")
        except Exception:
            data = []
        try:
            tg_bot._INFLIGHT_FILE.unlink()
        except Exception:
            pass
        durable_backend = not isinstance(self._backend, tg_bot.InMemoryBackend)
        # One notice per chat, for its LAST task: two in-flight entries of one
        # chat sent the same "restarted" message twice (live 2026-09-27 16:52).
        last = {}
        for d in data or []:
            if d.get("_state") == "pending" and durable_backend:
                continue        # still safely queued in the external backend
            last[d.get("chat_id")] = d
        for d in last.values():
            try:
                task = tg_bot._Task.from_dict(d)
                sess = self._get_session(task.chat_id)
                lang = self._lang(sess)
                sess.last_task_text = task.user_text
                sess.last_task_id = task.task_id
                self._store.put(sess)
                self._send_text(task.chat_id, tg_bot._t("interrupted", lang),
                                keyboard={"inline_keyboard": [[
                                    {"text": tg_bot._t("retry_btn", lang),
                                     "callback_data": f"retry:{task.task_id}"}]]})
                self._activity.log(task.chat_id, "system",
                                   "[recovery] interrupted task reported")
            except Exception:
                tg_bot.logger.exception("inflight recovery failed for %s", d)

    def _request_stop(self, chat_id: int) -> int:
        """Cancel the in-flight operation for `chat_id` and discard its queued work.

        Returns how many not-yet-resolved inbound items were dropped. Cancellation is
        cooperative: ctx.cancel_event is what long loops (ComfyUI poll, deep-research
        crawl, the tool round loop) actually check, so a running task stops at its next
        iteration boundary rather than being killed mid-write.
        """
        with self._stop_lock:
            self._stop_requests[chat_id] = time.time()
        # Only cancel the running task if it belongs to THIS chat. ctx is shared,
        # so an unconditional cancel_event would kill whoever happens to be running
        # — under round-robin that is regularly somebody else.
        with self._task_lock:
            # A chat may have TWO tasks in flight (a slow one plus an admitted
            # interject) — Stop means "stop everything happening in this chat",
            # so every running task's own event is set, not just the first.
            running = [t for t in (self._running_task.get(chat_id) or [])
                       if not getattr(t, "delivered", False)]
            evs = [self._task_cancels.get(t.task_id) for t in running]
        for ev in evs:
            if ev is not None:
                # THIS task's own event — not the shared one. Under concurrency the
                # shared event cancelled whichever request happened to be in flight.
                ev.set()
        # Drain items still sitting in this chat's debounce queue (not yet tasks).
        dropped = 0
        with self._mgmt_lock:
            q = self._queues.get(chat_id)
        if q is not None:
            while True:
                try:
                    q.get_nowait(); dropped += 1
                except _pyqueue.Empty:
                    break
        # …and the already-resolved tasks waiting in the backend.
        try:
            dropped += self._backend.drop_chat(chat_id)
        except Exception as exc:
            tg_bot.logger.warning("drop_chat(%s) failed: %s", chat_id, exc)
        # Purge this chat's entries from the pending journal too — otherwise a
        # task the user just explicitly cancelled reappears after the next
        # restart as "interrupted, want to retry?", resurrecting work the user
        # deliberately threw away.
        with self._task_lock:
            stale = [tid for tid, t in self._pending_journal.items()
                     if t.chat_id == chat_id]
            for tid in stale:
                self._pending_journal.pop(tid, None)
        if stale:
            self._write_inflight()
        tg_bot.logger.info("STOP requested chat=%s — dropped %d queued item(s)", chat_id, dropped)
        return dropped

    def _stop_and_report(self, chat_id: int, sess, lang: str) -> None:
        """Stop whatever this chat has in flight or queued, and say what happened.

        Shared by the ⛔ Stop button and /cancel so the two cannot drift. Telling
        the user "Stopping…" when nothing is running is the bot claiming work it
        is not doing — the same false-success family as narrating an image it
        never sent.
        """
        with self._task_lock:
            # A task whose reply already landed is only compacting history —
            # to the user nothing is running (live 2026-09-13, mega step 78:
            # Stop one second after the picture arrived got "Stopping…").
            was_running = any(not getattr(t, "delivered", False)
                              for t in (self._running_task.get(chat_id) or []))
        was_running = was_running or chat_id in self.__dict__.get("_debouncing", ())
        dropped = self._request_stop(chat_id)
        # A long-video job (tg_video) runs outside the task queue.
        _lv = self._long_video_stop(chat_id) if hasattr(self, "_long_video_stop") else False
        if not was_running and not dropped and not _lv:
            self._send_text(chat_id, tg_bot._t("stop_idle", lang),
                            keyboard=self._main_menu_kb(sess, lang))
            self._activity.log(chat_id, "system", "[stop] nothing to stop",
                               str(chat_id))
            return
        extra = tg_bot._t("stop_dropped", lang, n=dropped) if dropped else ""
        self._send_text(chat_id, tg_bot._t("stopping", lang, extra=extra),
                        parse_mode="HTML",
                        keyboard=self._main_menu_kb(sess, lang))
        self._on_stage(chat_id, "⛔ Stopped", True)
        self._activity.log(chat_id, "system", "[stop] requested by user",
                           str(chat_id))

    def _cancel_task(self, chat_id: int, task_id: str) -> str:
        """Cancel ONE request (the inline ⛔ button under its status message).

        Unlike Stop this leaves the rest of the chat's queue alone, which is what
        a user means when they cancel the specific thing they are looking at.

        Returns one of:
          "running" — the task was in flight; its own unwind guard
                       (`_is_cancelled`/`_stop_requested_after` in
                       `_run_task_inner`) will send the user-facing
                       confirmation once it actually stops. The caller must
                       NOT also send `cancel_done`, or the user sees it twice.
          "dropped"  — the task was only queued and is gone for good; nothing
                       else will ever report on it, so the caller must confirm.
          ""         — (falsy, matching the old bool contract) already
                       cancelled/finished by an earlier press, or never
                       existed/failed to drop: an honest "already finished"
                       reply, not a repeated false "cancelled".
        """
        with self._task_lock:
            if task_id in self._cancelled:
                return ""
            # A task cancelled while queued is dropped and never reaches the
            # consumer that would clear its id, so bound the set rather than let
            # it grow for the life of the process.
            if len(self._cancelled) > 512:
                self._cancelled.clear()
            self._cancelled.add(task_id)
            running_ids = {t.task_id for t in (self._running_task.get(chat_id) or [])}
            ev = self._task_cancels.get(task_id)
        if task_id in running_ids:
            if ev is not None:
                ev.set()
            self._activity.log(chat_id, "system", f"[cancel] running task {task_id[:8]}")
            return "running"
        dropped = False
        try:
            dropped = self._backend.drop_task(task_id)
        except Exception as exc:
            tg_bot.logger.warning("drop_task(%s) failed: %s", task_id, exc)
        if dropped:
            with self._task_lock:
                had = self._pending_journal.pop(task_id, None) is not None
            if had:
                self._write_inflight()
        self._activity.log(chat_id, "system",
                           f"[cancel] queued task {task_id[:8]} dropped={dropped}")
        return "dropped" if dropped else ""

    def _mark_interruptible(self, chat_id: int) -> None:
        """Flag that this chat's running task has reached a slow phase.

        Called from a task's own `ctx.stage_callback` (see _INTERRUPTIBLE_STAGE_
        KEYWORDS and the deep-research bypass). Read by the consumer-loop
        admission gate: a chat with exactly one running task and this flag set
        may accept a second, quick task instead of queueing it behind the whole
        slow operation.
        """
        with self._task_lock:
            self._chat_interruptible[chat_id] = True

    def _is_cancelled(self, task_id: str) -> bool:
        with self._task_lock:
            return task_id in self._cancelled

    def _stop_requested_after(self, task: tg_bot._Task) -> bool:
        """True when a Stop arrived AFTER this task was enqueued (so it is stale)."""
        with self._stop_lock:
            ts = self._stop_requests.get(task.chat_id, 0.0)
        return bool(ts) and task.enqueue_ts <= ts

    def _enqueue_item(self, chat_id: int, item: dict):
        # Stamped on ARRIVAL, before the debounce delay (_DEBOUNCE_S) — not at
        # task-creation time in _resolve_and_push, which runs after the delay.
        # A Stop pressed during the debounce window used to find nothing
        # running/queued yet (the item was still sitting here un-stamped) and
        # report "Nothing was running", while the batch went on to become a
        # task with a post-Stop enqueue_ts that _stop_requested_after let
        # through — the render happened anyway.
        item.setdefault("_arrival_ts", time.time())
        with self._mgmt_lock:
            if chat_id not in self._queues:
                self._queues[chat_id] = _pyqueue.Queue()
            q = self._queues[chat_id]
            # The put happens INSIDE the lock, and the worker's idle-exit path
            # re-checks the queue under the same lock. Otherwise there is a window
            # where the worker has already timed out but is not dead yet:
            # is_alive() says True, so no replacement is started, the item is put,
            # and then the worker deregisters and exits — leaving the message
            # sitting in a queue nobody serves. The user simply got no reply, and
            # it only unwedged when they sent something else.
            q.put(item)
            w = self._workers.get(chat_id)
            if w is None or not w.is_alive():
                t = threading.Thread(target=self._debounce_loop,
                                     args=(chat_id,), daemon=True,
                                     name=f"tg-debounce-{chat_id}")
                self._workers[chat_id] = t; t.start()

    def _debounce_loop(self, chat_id: int):
        q = self._queues[chat_id]
        while True:
            try: first = q.get(timeout=tg_bot._WORKER_IDLE_S)
            except _pyqueue.Empty:
                with self._mgmt_lock:
                    if not q.empty():
                        continue          # an item raced in — keep serving it
                    self._workers.pop(chat_id, None)
                    return
            batch = [first]
            # Stop during this window must know a batch is pending (it said
            # "nothing is running" while dropping an essay request, live).
            self.__dict__.setdefault("_debouncing", set()).add(chat_id)
            # A QUIET period, not a fixed window: the deadline moves with every
            # fragment that arrives, capped from the first one. Five fragments
            # of one thought 0.6 s apart used to be cut after the third and
            # answered as two separate requests (live, 2026-09-12).
            t_first = time.monotonic()
            deadline = t_first + tg_bot._DEBOUNCE_S
            while True:
                left = min(deadline, t_first + tg_bot._DEBOUNCE_MAX_S) - time.monotonic()
                if left <= 0: break
                try: batch.append(q.get(timeout=left))
                except _pyqueue.Empty: break
                deadline = time.monotonic() + tg_bot._DEBOUNCE_S
            # The batch is a turn of its own (a forwarded voice + кружки are
            # transcribed and looked at right here): traced, and logged to
            # the chat's transcript, like an update in _dispatch_logged.
            import chatlog
            _trace = turn_trace.start(chat=chat_id, task="batch", text=f"{len(batch)} item(s)")
            try:
                with chatlog.bind(chat_id):
                    self._resolve_and_push(chat_id, batch)
            except Exception: tg_bot.logger.exception("debounce_loop error chat=%s", chat_id)
            finally:
                self._debouncing.discard(chat_id)
                turn_trace.finish(_trace, keep_idle=False)

    def _consumer_loop(self):
        while self._running:
            try:
                task = self._backend.pop(timeout=5.0)
            except Exception:
                # A backend hiccup (Redis reconnect, a parse error) must never end
                # the only thread that runs work — back off briefly and keep going.
                tg_bot.logger.exception("queue pop failed")
                time.sleep(2.0)
                continue
            if task is None: continue
            if self._is_cancelled(task.task_id) or self._stop_requested_after(task):
                # Dropped here, never executed: its journal entry must go too, or
                # the next restart reports it "interrupted" (cancel raced the pop).
                with self._task_lock:
                    had = self._pending_journal.pop(task.task_id, None) is not None
                if had:
                    self._write_inflight()
            if self._is_cancelled(task.task_id):
                tg_bot.logger.info("Dropping task %s — cancelled before it started",
                            task.task_id[:8])
                with self._task_lock:
                    self._cancelled.discard(task.task_id)
                continue
            if self._stop_requested_after(task):
                tg_bot.logger.info("Dropping task %s chat=%s — Stop was pressed after it "
                            "was queued", task.task_id[:8], task.chat_id)
                self._activity.log(task.chat_id, "system",
                                   "[stop] dropped queued task", str(task.chat_id))
                continue
            # ── AT MOST TWO TASKS PER CHAT, AND ONLY WHEN THE FIRST IS SLOW ──
            # With a single consumer "one at a time" was implicit. With several,
            # the round-robin backend happily hands two tasks from the SAME chat
            # to two workers, and the user gets the same turn answered twice —
            # seen live: two identical "Я задумался и не выдал ответ" replies in
            # the same second, plus a pile of orphaned Cancel buttons. That gate
            # stays: a chat starts at 0 running tasks and a fresh task is always
            # admitted alone. A SECOND task for the same chat is only admitted
            # once the first has announced (via ctx.set_stage, see
            # _mark_interruptible) that it is now in a genuinely slow,
            # backgroundable phase — an image render, a web/deep-research crawl —
            # so a quick new message can be answered without waiting for the
            # whole thing, while the slow task keeps running untouched on its
            # own thread. A third task for the same chat always still queues.
            with self._task_lock:
                n = self._chat_busy.get(task.chat_id, 0)
                interruptible = self._chat_interruptible.get(task.chat_id, False)
                # A candidate whose text is BYTE-IDENTICAL to the task already
                # running for this chat is never a genuine "quick interject" —
                # it is the same request arriving twice (a client-side retry
                # sending the same message again after the first looked stuck,
                # observed live during a slow image-generation turn: the exact
                # same translate call, tool round and prompt-engineering steps
                # all ran a second time, CONCURRENTLY, doubling GPU/LLM load and
                # producing two near-simultaneous identical replies). Running
                # the identical text twice at once can never be "answer this
                # quickly while the slow thing continues" — it just wastes a
                # full second turn, so it still queues and waits, exactly as
                # before this concurrency mechanism existed.
                running = self._running_task.get(task.chat_id) or []
                is_exact_repeat = any(r.user_text == task.user_text for r in running)
                admit = (n == 0) or (n == 1 and interruptible and not is_exact_repeat)
                if admit:
                    self._chat_busy[task.chat_id] = n + 1
            if not admit:
                # Back to the HEAD of that chat's queue so the user's order is
                # kept, then let another chat's work proceed.
                self._backend.requeue(task)
                time.sleep(0.25)
                continue
            try: self._execute_task(task)
            except Exception:
                tg_bot.logger.exception("consumer error task=%s", task.task_id[:8])
            finally:
                # Release the chat gate FIRST, then clean up the temp upload.
                # Both must happen even when the task raised, or the chat stays
                # wedged for the life of the process. Only clear
                # _chat_interruptible once the LAST task for this chat is gone —
                # otherwise a still-running slow task loses its own flag and a
                # THIRD arrival that should still be admitted gets queued instead.
                with self._task_lock:
                    n = self._chat_busy.get(task.chat_id, 1) - 1
                    if n <= 0:
                        self._chat_busy.pop(task.chat_id, None)
                        self._chat_interruptible.pop(task.chat_id, None)
                    else:
                        self._chat_busy[task.chat_id] = n
                # task.image_path now lives under tg_bot._IMAGE_DIR (see
                # tg_resolve.py) precisely because it is ALSO the path
                # registered into sess.image_log for later "the one I sent
                # earlier" lookups -- deleting it here as if it were a
                # throwaway temp file left every register entry for a plain
                # photo upload dangling the instant its own task finished.
                # Only clean up a path that is genuinely NOT part of that
                # persistent register (e.g. an older/foreign temp path).
                if task.image_path and os.path.exists(task.image_path):
                    try:
                        _under_image_dir = Path(task.image_path).resolve().is_relative_to(
                            tg_bot._IMAGE_DIR.resolve())
                    except Exception:
                        _under_image_dir = False
                    if not _under_image_dir:
                        try: os.unlink(task.image_path)
                        except Exception: pass


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)
