"""Work queue, cancellation and the worker/watchdog loops.

Owns the path a message takes from "accepted" to
"running": the debounce buffer, _enqueue_item, the consumer threads, the
per-chat and per-task cancellation bookkeeping (_request_stop, _cancel_task,
_is_cancelled, _stop_requested_after, _mark_interruptible), the crash-recovery
in-flight journal, the poll watchdog and the queue statistics the GUI reads.

_INFLIGHT_FILE and friends are read as tg_bot.<name> rather than imported by
value -- redirect_data_dir() rebinds them at runtime, and a by-value import
would pin this module to the LIVE data directory even in a test that
redirected it.
"""
from __future__ import annotations

import json
import os
import queue as _pyqueue
import threading
import tg_transport
import turn_trace
import time
import uuid
from pathlib import Path
from tg_incoming import BATCH_IDS, BotStopping, current_update_id


# [last snapshot taken, last snapshot written] and the lock that orders writes.
_INFLIGHT_SEQ = [0, 0]
_INFLIGHT_IO = threading.Lock()


class QueueMixin:
    def _run_busy(self, chat_id: int, fn, *args) -> None:
        """Run fn(*args) on a daemon thread with the chat marked busy until it
        returns. Work started outside the queue (weather lookup, voice-clone
        prep and synthesis) otherwise left the turn looking finished at its
        "Checking…" line."""
        with self._task_lock:
            self._chat_busy[chat_id] = self._chat_busy.get(chat_id, 0) + 1

        def run():
            try:
                with tg_transport.result_menu(self, chat_id):
                    fn(*args)
            finally:
                with self._task_lock:
                    n = self._chat_busy.get(chat_id, 1) - 1
                    if n <= 0:
                        self._chat_busy.pop(chat_id, None)
                    else:
                        self._chat_busy[chat_id] = n
        turn_trace.spawn(run, name=f"chat-work-{chat_id}")

    def _run_cancellable(self, chat_id: int, note: str, fn, *args, lang: str = "ru") -> None:
        """Long work a button starts outside the queue (✍️ / ✨ lyrics, ✨ polish
        and sing...), with ⛔ under its «working» note -- one call instead of a
        cancel button wired by hand for every new flow. fn(ctx, *args) gets its own ctx copy: is_cancelled()
        turns true on ⛔, and the work stops at its next check. The note
        loses its button when the work ends."""
        import uuid
        base = self._get_ctx()
        if base is None:                       # no app context (suites): the flag alone
            ctx = type("JobCtx", (), {"is_cancelled": lambda s: s.cancel_event.is_set()})()
        else:
            ctx = tg_bot._scoped_ctx(base)
        ctx.cancel_event = threading.Event()
        jid = uuid.uuid4().hex[:8]
        jobs = self.__dict__.setdefault("_busy_jobs", {})
        jobs[jid] = (chat_id, ctx.cancel_event)
        kb = {"inline_keyboard": [[{"text": tg_bot._t("cancel_btn", lang),
                                    "callback_data": f"busy:cancel:{jid}"}]]}
        mid = self._send_get_id(chat_id, note, keyboard=kb)

        def show(text: str) -> None:
            if mid:
                try:
                    self._edit_text(chat_id, mid, text, keyboard=kb)
                except Exception:
                    pass
        # the copy carried the desktop's own bar; this job draws its own under its note
        _, ctx.progress_callback = self._with_progress(chat_id, lang, show, shown=note,
                                                       alive=lambda: jid in jobs)

        def work():
            try:
                fn(ctx, *args)
            finally:
                jobs.pop(jid, None)
                if mid:
                    self._api_post("editMessageReplyMarkup", {
                        "chat_id": chat_id, "message_id": mid,
                        "reply_markup": json.dumps({"inline_keyboard": []})})
        self._run_busy(chat_id, work)

    def _cb_busy_cancel(self, chat_id: int, data: str) -> None:
        job = getattr(self, "_busy_jobs", {}).get(data.rsplit(":", 1)[-1])
        if job and job[0] == chat_id:
            job[1].set()
        self._send_text(chat_id, tg_bot._t("cancelled", self._lang(self._get_session(chat_id))))

    def get_queue_stats(self) -> dict:
        return {
            "depth":        self._backend.depth(),
            "active_chats": dict(self._active_stages),
            "backend":      self._backend.name(),
        }

    def _watchdog_loop(self):
        """Keep the poll and consumer threads alive, and unstick hung tasks."""
        interval = max(5, tg_bot._cfg_int("TG_WATCHDOG_S", 15, self.settings))
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
                self._cancel_runaways(task_max)
            except Exception:
                tg_bot.logger.exception("watchdog iteration failed")

    def _cancel_runaways(self, task_max: int) -> None:
        abs_max = max(task_max, tg_bot._cfg_int("TG_TASK_ABS_MAX_S", 4 * 3600))
        with self._task_lock:
            chat_of = {t.task_id: cid for cid, lst in self._running_task.items() for t in lst}
            stale = [(tid, started) for tid, started in self._task_started.items()
                     if time.monotonic() - started > task_max]
        # a render still counting its steps is not a runaway (see _still_rendering):
        # this cancelled part 6 of an approved six-part clip at the hour (2026-10-20)
        stale = [(tid, started) for tid, started in stale
                 if time.monotonic() - started > abs_max or tid not in chat_of
                 or not self._still_rendering(chat_of[tid])]
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
            # would abort whatever the desktop assistant was doing
            # while leaving the actual runaway task running forever.
            if ev is not None:
                try: ev.set()
                except Exception: pass

    def _inflight_path(self):
        return tg_bot._INFLIGHT_FILE

    def _write_inflight(self) -> bool:
        """Persist the set of tasks currently executing OR merely queued.

        A "running" entry means a crash killed an active graph.invoke(). A
        "pending" entry means the task was pushed onto the backend but never
        popped: with the default InMemoryBackend that queue lives only in this
        process's RAM, so a crash there is just as fatal to the task. Both are
        recorded so a restart is visible either way.
        """
        try:
            with self._task_lock:
                data = ([dict(t.to_dict(), _state="running")
                          for tasks in self._running_task.values() for t in tasks]
                         + [dict(t.to_dict(), _state="pending")
                            for t in self._pending_journal.values()])
                seq = _INFLIGHT_SEQ[0] = _INFLIGHT_SEQ[0] + 1
            # Atomic: two threads writing at once could leave a torn
            # file, and recovery would then read [] and silently lose every task.
            # Ordered: the snapshot is taken under the lock but written after it,
            # so an older snapshot could land last and bring back a finished task
            # (a crash then told its user it was "interrupted").
            path = self._inflight_path()
            with _INFLIGHT_IO:
                if seq < _INFLIGHT_SEQ[1]:
                    return True
                tmp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
                with tmp.open("w", encoding="utf-8") as stream:
                    json.dump(data, stream, ensure_ascii=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(tmp, path)
                _INFLIGHT_SEQ[1] = seq
            return True
        except Exception as exc:
            tg_bot.logger.debug("inflight write failed: %s", exc)
            return False

    def _take_inflight(self) -> list:
        """Read the previous process's journal without opening a loss window.

        start() calls this BEFORE the poll and consumer threads exist: the
        first message after a restart enqueues a task, which rewrites the
        journal, and a recovery reading it afterwards found only that new
        task -- the users whose tasks died with the crash never heard. Keep the
        old file until its replacement snapshot has been written."""
        try:
            if not tg_bot._INFLIGHT_FILE.exists():
                return []
            data = json.loads(tg_bot._INFLIGHT_FILE.read_text(encoding="utf-8") or "[]")
        except Exception:
            data = []
        return data if isinstance(data, list) else []

    def _queued_work_allowed(self, chat_id):
        user = self._user_store.get(chat_id)
        return (user is not None and user.status == "approved"
                and self._get_session(chat_id).reg_state != "awaiting_login")

    def _restore_pending(self, data: list) -> None:
        """Restore never-started work before any consumer can claim it."""
        missing = []
        for d in data or []:
            if not isinstance(d, dict) or d.get("_state") != "pending":
                continue
            try:
                task = tg_bot._Task.from_dict(d)
            except Exception:
                tg_bot.logger.exception("inflight requeue failed for %s", d.get("task_id"))
                continue
            if not self._queued_work_allowed(task.chat_id):
                self._backend.drop_task(task.task_id)
                continue
            if not self._backend.position_of(task.task_id):
                missing.append(task)
            with self._task_lock:
                self._pending_journal[task.task_id] = task
        for task in reversed(missing):
            self._backend.requeue(task)
        self._write_inflight()

    def _restore_incoming(self, tasks):
        self._incoming.ack([key for task in tasks if isinstance(task, dict)
                            for key in task.get("incoming_ids", [])])
        for entry in self._incoming.pending():
            item = entry["item"]
            chat_id = entry["chat_id"]
            if not self._queued_work_allowed(chat_id):
                self._incoming.ack([entry["id"]])
                continue
            if item.get("type") == "album_pending":
                photo = item["photos"][0]
                self._buffer_album(chat_id, item["group_id"], photo["file_id"],
                    photo.get("caption", ""), item.get("forwarded", False), saved=item)
            else:
                item.setdefault('_message_id', 0)
                item.setdefault('_target_captured', True)
                self._enqueue_item(chat_id, dict(item, _inbox_id=entry["id"]))

    def _persist_task(self, task, incoming_ids=None):
        import intent
        if not task.intent_snapshot:
            task.intent_snapshot = intent.snapshot_turn()
        if getattr(self, "_stopping", False):
            raise BotStopping("the bot stopped before this task was accepted")
        if self._stop_requested_after(task):
            return False
        sess = self._get_session(task.chat_id)
        if task.context_revision is None:
            if not task.translation and not task.image_path and not task.image_id:
                task.image_id = sess.turn_image
                task.image_selection = 'context'
            token = sess.begin_context_turn(task.context_generation)
            if token is None:
                return False
            task.context_generation, task.context_revision = token
            entry = (tg_bot._image_by_path(sess, task.image_path) if task.image_path
                     else tg_bot._image_by_id(sess, task.image_id))
            if entry:
                sess.update_turn_media(token, turn_image=entry['id'])
            self._store.put(sess)
        ids = list(incoming_ids if incoming_ids is not None else BATCH_IDS.get() or
                   ([current_update_id()] if current_update_id() else []))
        task.incoming_ids = list(dict.fromkeys(task.incoming_ids + ids))
        with self._task_lock:
            self._pending_journal[task.task_id] = task
            if task.replaces_task_id:
                self._pending_journal.pop(task.replaces_task_id, None)
        if self._write_inflight() is False:
            raise OSError("could not persist the accepted task")
        incoming = getattr(self, "_incoming", None)
        if incoming is not None:
            incoming.ack(task.incoming_ids)
        self._backend.push(task)
        if self._stop_requested_after(task):
            self._cancel_task(task.chat_id, task.task_id)
            return False
        return True

    def _recover_inflight(self, data: list | None = None) -> None:
        """Notify users whose task was killed by a restart, and offer a retry.

        Otherwise the user is left staring at a "⚙️ Working…" status message
        that never changes.

        Never-started tasks were restored before consumers started. Only work
        that had already started needs an interrupted/retry notice.
        """
        if data is None:
            data = self._take_inflight()
            self._restore_pending(data)
            self._restore_incoming(data)
        # One notice per chat, for its LAST task: two in-flight entries of one
        # chat sent the same "restarted" message twice.
        last = {}
        for d in data or []:
            if not isinstance(d, dict):
                continue
            if d.get("_state") == "pending":
                # Already reconciled by _restore_pending before the consumers started.
                continue
            last[d.get("chat_id")] = d
        for d in last.values():
            try:
                task = tg_bot._Task.from_dict(d)
                if not self._queued_work_allowed(task.chat_id):
                    continue
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
        stopped_queue_note = self.__dict__.get('_queue_note_tasks', {}).get(chat_id)
        sess = self._get_session(chat_id)
        with sess._lock:
            for name in sess.FLOW_STATES:
                # Rendering may already have disarmed its prompt; retire its token too.
                setattr(sess, name, getattr(sess, name))
        self._store.put(sess)
        for cid, event in list(self.__dict__.get('_busy_jobs', {}).values()):
            if cid == chat_id:
                event.set()
        incoming_ids = [entry["id"] for entry in self._incoming.pending() if entry["chat_id"] == chat_id]
        self._incoming.stop_chat(chat_id, self._stop_requests[chat_id])
        with self._mgmt_lock:
            self._incoming_enqueued.difference_update(incoming_ids)
        with self._album_lock:
            for key, album in list(self._albums.items()):
                if album["chat_id"] == chat_id:
                    if album.get("timer"):
                        album["timer"].cancel()
                    del self._albums[key]
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
            if stopped_queue_note:
                threading.Thread(target=self._drop_busy_note, args=(chat_id, stopped_queue_note[0]), daemon=True).start()
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
        the user "Stopping…" when nothing is running would claim work the bot is
        not doing.
        """
        with self._task_lock:
            # A task whose reply already landed is only compacting history —
            # to the user nothing is running (a Stop right after the picture
            # arrived must not answer "Stopping…").
            was_running = any(not getattr(t, "delivered", False)
                              for t in (self._running_task.get(chat_id) or []))
            was_running = was_running or bool(self._chat_busy.get(chat_id))
        was_running = was_running or any(
            cid == chat_id for cid, _ in list(self.__dict__.get('_busy_jobs', {}).values()))
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
          ""         — (falsy) already cancelled/finished by an earlier
                       press, or never existed/failed to drop: an honest
                       "already finished" reply, not a repeated "cancelled".
        """
        with self._task_lock:
            running_ids = {t.task_id for t in (self._running_task.get(chat_id) or [])}
            pending = self._pending_journal.get(task_id)
            owned = task_id in running_ids or (pending is not None and pending.chat_id == chat_id)
        if not owned:
            # A restored durable queue may have no entry in this process's journal yet.
            try:
                owned = any(t.task_id == task_id and t.chat_id == chat_id
                            for t in self._backend.service_order())
            except Exception as exc:
                tg_bot.logger.warning("cancel ownership lookup failed: %s", exc)
                return ""
            if not owned:
                return ""
        with self._task_lock:
            if task_id in self._cancelled:
                return ""
            # A task cancelled while queued is dropped and never reaches the
            # consumer that would clear its id, so the set is bounded: it
            # forgets the oldest ids, never the one just cancelled
            # (tg_bot._RecentIds).
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
            self._drop_busy_note(chat_id, task_id)
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
        # Stamped on ARRIVAL, before the debounce delay (_DEBOUNCE_S), not at
        # task-creation time in _resolve_and_push (after the delay): otherwise a
        # Stop pressed during the debounce window would not see this item, and
        # the task it becomes would carry a post-Stop enqueue_ts that
        # _stop_requested_after lets through.
        item.setdefault("_arrival_ts", time.time())
        # When the USER sent it (Telegram's message date), not when we got it: a stalled poll delivers a burst late
        item.setdefault("_sent_ts", (getattr(self, "_msg_date", {}) or {}).get(chat_id))
        item.setdefault('_message_id', (getattr(self, '_msg_id', {}) or {}).get(chat_id, 0))
        sess = self._get_session(chat_id)
        item.setdefault('_context_generation', sess.context_generation)
        if sess.menu in ('translator', 'translator_languages'):
            item.setdefault('_translation', (sess.translation_target, sess.translation_mode))
        if sess.menu.startswith('teacher'):
            item.setdefault('_teacher_epoch',sess.teacher_epoch)
        item.setdefault('_input_context', {
            'prefix': sess.pending_prefix, 'menu': sess.menu,
            'revisions': {name: sess.flow_revision(name) for name in sess.INPUT_STATES}})
        target_to_consume = ''
        if '_target_captured' not in item:
            if item.get('type') in ('photo', 'album'):
                target_to_consume = sess.target_image
            if (item.get('type') in ('text', 'voice', 'armed_media') and not item.get('forwarded')
                    and not tg_bot._LABEL2KEY.get((item.get('text') or '').strip())):
                if not item.get('image_id') and not item.get('reply_photo') and sess.target_image:
                    item['image_id'] = sess.target_image
                target_to_consume = sess.target_image if item.get('reply_photo') else item.get('image_id', '')
            item['_target_captured'] = True
        with self._mgmt_lock:
            key = item.setdefault("_inbox_id", current_update_id() or uuid.uuid4().hex)
            if key in self._incoming_enqueued or not self._incoming.put(chat_id, item, key):
                return
            self._incoming_enqueued.add(key)
            if target_to_consume and sess.target_image == target_to_consume:
                sess.target_image = ''
                self._store.put(sess)
            if chat_id not in self._queues:
                self._queues[chat_id] = _pyqueue.Queue()
            q = self._queues[chat_id]
            # The put happens INSIDE the lock, and the worker's idle-exit path
            # re-checks the queue under the same lock. Otherwise a worker that has
            # timed out but is not dead yet (is_alive() still True) would get no
            # replacement, then exit, leaving the message in a queue nobody serves.
            q.put(item)
            w = self._workers.get(chat_id)
            if w is None or not w.is_alive():
                t = threading.Thread(target=self._debounce_loop,
                                     args=(chat_id,), daemon=True,
                                     name=f"tg-debounce-{chat_id}")
                self._workers[chat_id] = t; t.start()

    def _debounce_loop(self, chat_id: int):
        q = self._queues[chat_id]
        carry = None
        while not getattr(self, "_stopping", False):
            try: first, carry = (carry or q.get(timeout=self.settings.get("_WORKER_IDLE_S", tg_bot._WORKER_IDLE_S))), None
            except _pyqueue.Empty:
                with self._mgmt_lock:
                    if not q.empty():
                        continue          # an item raced in — keep serving it
                    self._workers.pop(chat_id, None)
                    return
            if first is None or getattr(self, "_stopping", False):
                return
            batch = [first]
            # Stop during this window must know a batch is pending (see
            # _stop_and_report).
            self.__dict__.setdefault("_debouncing", set()).add(chat_id)
            # A QUIET period, not a fixed window: the deadline moves with every
            # fragment that arrives, capped from the first one. Five fragments
            # of one thought 0.6 s apart must not be cut after the third and
            # answered as two separate requests.
            # A bare forward (only forwarded material, nothing the user typed) waits longer: the instruction
            # ("объясни Степану прикол") is typed a few seconds later, and answering the forward alone gives
            # two replies to one request.
            t_first = time.monotonic()
            def _fwd(): return all(i.get("forwarded") or i.get("type") == "fwd_voice" for i in batch)
            # a forwarded post's videos/voices come with pictures that land a couple of seconds later (an album is
            # flushed after its own quiet period); a lone forwarded video must not wait the full text-forward window
            def _quiet(): return ((tg_bot._DEBOUNCE_FWDV_S if any(i.get("type") == "fwd_voice" for i in batch) else tg_bot._DEBOUNCE_FWD_S)
                                  if _fwd() else tg_bot._DEBOUNCE_S)
            def _cap(): return ((tg_bot._DEBOUNCE_FWDV_MAX_S if any(i.get("type") == "fwd_voice" for i in batch) else tg_bot._DEBOUNCE_FWD_MAX_S) if _fwd() else tg_bot._DEBOUNCE_MAX_S)
            def _lag(it):
                # Forwards that reach us late (a stalled getUpdates: pieces sent together arrive
                # minutes apart) have more of the same burst still on the way: wait that long again.
                s = it.get("_sent_ts")
                return min(30.0, max(0.0, time.time() - s)) if s and it.get("forwarded") and time.time() - s > 3 else 0.0
            extra = _lag(first)
            deadline = t_first + _quiet() + extra
            while True:
                left = min(deadline, t_first + _cap() + extra) - time.monotonic()
                if left <= 0: break
                try: nxt = q.get(timeout=left)
                except _pyqueue.Empty: break
                if nxt is None or getattr(self, "_stopping", False):
                    self._debouncing.discard(chat_id)
                    return
                # Sent further apart than any window waits: two turns that a downtime delivered together
                # (a link at 14:33 and a screenshot at 14:57 got one reply about the screenshot, 2026-10-22).
                s0, s1 = batch[-1].get("_sent_ts"), nxt.get("_sent_ts")
                if s0 and s1 and abs(s1 - s0) > tg_bot._DEBOUNCE_FWD_MAX_S:
                    carry = nxt
                    break
                batch.append(nxt)
                extra = max(extra, _lag(batch[-1]))
                deadline = time.monotonic() + _quiet() + _lag(batch[-1])
            # The batch is a turn of its own (a forwarded voice + кружки are
            # transcribed and looked at right here): traced, and logged to
            # the chat's transcript, like an update in _dispatch_logged.
            import chatlog
            if getattr(self, "_stopping", False):
                self._debouncing.discard(chat_id)
                return
            _trace = turn_trace.start(chat=chat_id, task="batch", text=f"{len(batch)} item(s)")
            try:
                with chatlog.bind(chat_id):
                    self._resolve_and_push(chat_id, batch)
            except BotStopping:
                return
            except Exception as exc:
                tg_bot.logger.exception("debounce_loop error chat=%s", chat_id)
                # A song with a caption crashed here and the user heard nothing at all.
                try:
                    _lang = self._lang(self._get_session(chat_id))
                    self._send_text(chat_id, tg_bot._t("err_generic", _lang, err=type(exc).__name__),
                                    parse_mode="HTML")
                except Exception:
                    tg_bot.logger.exception("could not report the batch error chat=%s", chat_id)
            finally:
                self._debouncing.discard(chat_id)
                turn_trace.finish(_trace, keep_idle=False)

    def _claim_task(self):
        # Pop and admission must be ordered: two held candidates can reverse a chat's FIFO.
        with self._queue_claim_lock:
            if not self._running or self._stopping:
                return None
            task = self._backend.pop(timeout=5.0)
            if task is None: return None
            if not self._running or self._stopping:
                self._backend.requeue(task)
                return None
            if not self._queued_work_allowed(task.chat_id):
                self._request_stop(task.chat_id)
                return None
            if self._is_cancelled(task.task_id) or self._stop_requested_after(task):
                # Dropped here, never executed: its journal entry must go too, or
                # the next restart reports it "interrupted" (cancel raced the pop).
                with self._task_lock:
                    had = self._pending_journal.pop(task.task_id, None) is not None
                if had:
                    self._write_inflight()
            if self._is_cancelled(task.task_id):
                threading.Thread(target=self._drop_busy_note, args=(task.chat_id, task.task_id), daemon=True).start()
                tg_bot.logger.info("Dropping task %s — cancelled before it started",
                            task.task_id[:8])
                with self._task_lock:
                    self._cancelled.discard(task.task_id)
                return None
            if self._stop_requested_after(task):
                threading.Thread(target=self._drop_busy_note, args=(task.chat_id, task.task_id), daemon=True).start()
                tg_bot.logger.info("Dropping task %s chat=%s — Stop was pressed after it "
                            "was queued", task.task_id[:8], task.chat_id)
                self._activity.log(task.chat_id, "system",
                                   "[stop] dropped queued task", str(task.chat_id))
                return None
            # ── AT MOST TWO TASKS PER CHAT, AND ONLY WHEN THE FIRST IS SLOW ──
            # With several consumers the round-robin backend would hand two
            # tasks from the SAME chat to two workers and the user would get the
            # same turn answered twice. A fresh task is therefore admitted
            # alone; a SECOND task for the chat is admitted only once the first
            # has announced (via ctx.set_stage, see _mark_interruptible) that it
            # is in a genuinely slow, backgroundable phase -- an image render, a
            # web/deep-research crawl -- so a quick new message is answered
            # without waiting for the whole thing. A third task always queues.
            with self._task_lock:
                n = self._chat_busy.get(task.chat_id, 0)
                interruptible = self._chat_interruptible.get(task.chat_id, False)
                # A candidate whose text is BYTE-IDENTICAL to the task already
                # running for this chat is never a genuine "quick interject":
                # it is the same request arriving twice (a client-side retry),
                # and running it concurrently would only double the GPU/LLM load
                # and produce two identical replies, so it still queues and waits.
                running = self._running_task.get(task.chat_id) or []
                is_exact_repeat = any(r.user_text == task.user_text for r in running)
                admit = (n == 0) or (n == 1 and interruptible and not is_exact_repeat)
                if admit:
                    self._chat_busy[task.chat_id] = n + 1
            if not admit:
                # Back to the HEAD of that chat's queue so the user's order is
                # kept, then let another chat's work proceed.
                self._backend.requeue(task)
                return None
            return task

    def _consumer_loop(self):
        while self._running:
            try:
                task = self._claim_task()
            except Exception:
                # Backend outages must not end the worker that drains the queue.
                tg_bot.logger.exception("queue claim failed")
                time.sleep(2.0)
                continue
            if task is None:
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
                # task.image_path lives under tg_bot._IMAGE_DIR (see tg_resolve.py)
                # because it is ALSO the path registered into sess.image_log for
                # later lookups, so it must not be deleted like a temp file. Only
                # clean up a path that is NOT part of that persistent register.
                if task.image_path and os.path.exists(task.image_path):
                    try:
                        _under_image_dir = Path(task.image_path).resolve().is_relative_to(
                            tg_bot._IMAGE_DIR.resolve())
                    except Exception:
                        _under_image_dir = False
                    # A bot result (runtime/lettering_clean_*.png) is registered where it was
                    # made: 🎬 under it deleted it, and the second 🎬 said «картинки больше нет».
                    if not _under_image_dir and not tg_bot._image_by_path(
                            self._get_session(task.chat_id), task.image_path):
                        try: os.unlink(task.image_path)
                        except Exception: pass


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)
