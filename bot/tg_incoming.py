"""Durable, already-authorized inputs waiting for the debounce/resolve stage."""
import copy
import contextvars
import json
import os
import threading
import weakref
from pathlib import Path

UPDATE_ID = contextvars.ContextVar("tg_update_id", default=("", None))
BATCH_IDS = contextvars.ContextVar("tg_batch_ids", default=())
_LOCKS = {}
_OWNERS = weakref.WeakValueDictionary()
_REGISTRY_LOCK = threading.Lock()


class BotStopping(RuntimeError):
    pass


def current_update_id():
    key, thread_id = UPDATE_ID.get()
    return key if thread_id == threading.get_ident() else ""


class IncomingStore:
    def __init__(self, path):
        self.path = Path(path)
        with _REGISTRY_LOCK:
            self._lock = _LOCKS.setdefault(str(self.path.resolve()), threading.RLock())
        self._load()

    def _load(self):
        data = json.loads(self.path.read_text("utf-8")) if self.path.is_file() else {}
        self._pending = data.get("pending", {})
        self._done = list(data.get("done", []))[-512:]
        self._stops = data.get("stops", {})
        if not isinstance(self._pending, dict):
            raise ValueError("invalid incoming journal")

    def _save(self, pending, done, stops=None):
        stops = self._stops if stops is None else stops
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as stream:
                json.dump({"pending": pending, "done": done, "stops": stops}, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)
        self._pending, self._done = pending, done
        self._stops = stops

    def seen(self, key):
        with self._lock:
            self._load()
            return key in self._pending or key in self._done

    def put(self, chat_id, item, key):
        with self._lock:
            self._load()
            if key in self._done:
                return False
            pending = dict(self._pending)
            pending[key] = {"chat_id": chat_id, "item": copy.deepcopy(item), "id": key}
            self._save(pending, list(self._done))
            return True

    def pending(self):
        with self._lock:
            self._load()
            return copy.deepcopy(list(self._pending.values()))

    def ack(self, keys):
        keys = list(dict.fromkeys(k for k in keys if k))
        if not keys:
            return
        with self._lock:
            self._load()
            pending = {k: v for k, v in self._pending.items() if k not in keys}
            done = [k for k in self._done if k not in keys] + keys
            self._save(pending, done[-512:])

    def drop_chat(self, chat_id):
        self.ack([entry["id"] for entry in self.pending() if entry["chat_id"] == chat_id])

    def stop_chat(self, chat_id, timestamp):
        with self._lock:
            self._load()
            keys = [k for k, v in self._pending.items() if v["chat_id"] == chat_id]
            pending = {k: v for k, v in self._pending.items() if k not in keys}
            done = [k for k in self._done if k not in keys] + keys
            self._save(pending, done[-512:], dict(self._stops, **{str(chat_id): timestamp}))

    def stops(self):
        with self._lock:
            self._load()
            return {int(k): float(v) for k, v in self._stops.items()}

    def claim(self, bot):
        with _REGISTRY_LOCK:
            key = str(self.path.resolve())
            prior = _OWNERS.get(key)
            if prior is not None and (prior is not bot or getattr(prior, "_stopping", False)):
                threads = list(prior._workers.values()) + list(prior._consumers)
                if prior._running or any(t.is_alive() for t in threads) or prior._chat_busy:
                    return False
            _OWNERS[key] = bot
            return True
