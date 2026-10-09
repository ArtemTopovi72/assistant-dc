"""🔐 Админка: the Telegram bot's live console on the desktop.

Same snapshot and actions as the Telegram panel (tg_admin.AdminMixin), so the
two never disagree. The snapshot is taken on a worker thread (nvidia-smi and
Redis are not for the GUI thread); every action runs on a plain thread, and
every slot catches its own errors -- an exception escaping a Qt slot aborts
the whole app with no traceback (see native-crash notes).
"""
import threading
import logging

from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QAbstractItemView, QComboBox, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                             QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget, QCheckBox)

from gui_common import ACCENT2, BORDER, MUTED, PANEL2, REC, TEXT, _card, _section
from tg_admin import _dur
from ui_scale import px

logger = logging.getLogger("assistant.gui")
_STATUS_COLOR = {"approved": "#3ddc84", "pending": "#f0b429", "banned": REC}


class _Poller(QThread):
    snap = pyqtSignal(object)

    def __init__(self, get_bot):
        super().__init__()
        self._get_bot, self._stop = get_bot, threading.Event()

    def run(self):
        while not self._stop.is_set():
            bot = self._get_bot()
            try:
                self.snap.emit(bot.admin_snapshot() if bot else None)
            except Exception:
                logger.debug("admin snapshot failed", exc_info=True)
            self._stop.wait(2.0)

    def stop(self):
        self._stop.set()


def _table(headers, stretch: int, rows: int = 4) -> QTableWidget:
    """`stretch` is the free-text column that takes the spare width and elides;
    the rest fit their contents. Tall enough for `rows` rows plus the header --
    it used to collapse to a sliver that cut its only row in half."""
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setSelectionMode(QAbstractItemView.SingleSelection)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setWordWrap(False)
    t.setTextElideMode(Qt.ElideRight)
    t.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    h = t.horizontalHeader()
    h.setSectionResizeMode(QHeaderView.ResizeToContents)
    h.setSectionResizeMode(stretch, QHeaderView.Stretch)
    h.setMinimumSectionSize(40)
    # The app's QSS pads cells and the header (a header is ~1.5 rows): measured
    # from the font alone, the 4th user row came out half hidden.
    rh = t.fontMetrics().height() + 22
    t.verticalHeader().setDefaultSectionSize(rh)
    t.setMinimumHeight(int(rh * (rows + 1.6)) + 8)
    t.setStyleSheet(f"QTableWidget{{background:{PANEL2};border:1px solid {BORDER};border-radius:8px;}}")
    return t


def _buttons(spec, per_row: int = 2) -> QGridLayout:
    """Buttons in a grid, `per_row` across: one long row pushed the whole tab
    into a horizontal scroll and cut the last button in half on a narrow window."""
    g = QGridLayout()
    g.setHorizontalSpacing(8); g.setVerticalSpacing(6)
    for i, (label, slot) in enumerate(spec):
        b = QPushButton(label); b.clicked.connect(slot)
        b.setObjectName("ghost")      # row actions, not the page's primary action: all-blue read as four "Send" buttons
        g.addWidget(b, i // per_row, i % per_row)
    return g


class AdminTab(QWidget):
    def __init__(self, host):
        super().__init__()
        self.host = host
        self._snap = None
        self._build()
        self._poller = _Poller(self._bot)
        self._poller.snap.connect(self._on_snap)
        self._poller.start()

    def _bot(self):
        return getattr(getattr(self.host, "telegram_tab", None), "_bot", None)

    # ── layout ───────────────────────────────────────────────────────────────
    def _build(self):
        lay = QVBoxLayout(self)
        head = QHBoxLayout()
        title = QLabel('🔐 <b>Bot admin</b>')
        title.setStyleSheet(f"font-size:{px(18)}px;color:{TEXT};")
        self.meta = QLabel('bot is not running')
        self.meta.setStyleSheet(f"color:{MUTED};")
        head.addWidget(title); head.addStretch(1); head.addWidget(self.meta)
        lay.addLayout(head)

        self.gpu = QLabel("")
        self.gpu.setStyleSheet(f"color:{ACCENT2};font-weight:600;")
        self.gpu.setWordWrap(True)
        lay.addWidget(self.gpu)

        lay.addWidget(_section('⚡ Running'))
        self.run_t = _table(['User', 'Stage', "⏱", 'Request'], stretch=3, rows=3)
        lay.addWidget(self.run_t)
        lay.addLayout(_buttons([('⛔ Cancel request', self._cancel_running),
                                ('🛑 Everything of this user', self._stop_running_chat)]))

        lay.addWidget(_section('📋 Queue'))
        self.q_t = _table(["#", 'User', 'Waiting', 'Request'], stretch=3, rows=3)
        lay.addWidget(self.q_t)
        lay.addLayout(_buttons([('⬆️ Next up', self._bump),
                                ('✖ Remove', self._drop)]))

        lay.addWidget(_section('👥 Users'))
        self.u_t = _table(['Name', "@", "id", 'Status', 'Today', 'Now'], stretch=0, rows=6)
        lay.addWidget(self.u_t, 1)
        lay.addLayout(_buttons([('🚫 Ban', lambda: self._set_status("banned")),
                                ('✅ Approve', lambda: self._set_status("approved")),
                                ('🛑 Stop their requests', self._stop_user)]))

        card, cl = _card()
        cl.addWidget(_section('📢 Message from the administration'))
        row = QHBoxLayout()
        self.to = QComboBox(); self.to.setMinimumWidth(120)
        self.all_cb = QCheckBox('all approved users')
        self.msg = QLineEdit(); self.msg.setPlaceholderText('Message text…')
        self.msg.returnPressed.connect(self._say)
        send = QPushButton('Send'); send.clicked.connect(self._say)
        row.addWidget(self.to, 1); row.addWidget(self.all_cb)
        cl.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.msg, 1); row.addWidget(send)
        cl.addLayout(row)
        self.note = QLabel(""); self.note.setStyleSheet(f"color:{MUTED};")
        cl.addWidget(self.note)
        lay.addWidget(card)

    # ── data ─────────────────────────────────────────────────────────────────
    def _on_snap(self, s):
        try:
            self._snap = s
            if not s:
                self.meta.setText('bot is not running')
                return
            self.meta.setText(f"🔄 {s['ts']} · {s['backend']}")
            self.gpu.setText("   ".join(x for x in (s["gpu"], _plain(s["training"])) if x))
            self._fill(self.run_t, [(r["user"], r["stage"], _dur(r["elapsed"]), r["text"]) for r in s["running"]],
                       [r["task_id"] for r in s["running"]], [r["chat_id"] for r in s["running"]])
            self._fill(self.q_t, [(str(i), q["user"], _dur(q["waited"]), q["text"])
                                  for i, q in enumerate(s["queued"], 1)],
                       [q["task_id"] for q in s["queued"]], [q["chat_id"] for q in s["queued"]])
            busy = {r["chat_id"]: "⚡" for r in s["running"]}
            for q in s["queued"]:
                busy.setdefault(q["chat_id"], "⏳")
            self._fill(self.u_t, [(("👑 " if u["is_admin"] else "") + (u["name"] or "—"), u["username"] or "",
                                   str(u["chat_id"]), u["status"], str(u["today"]), busy.get(u["chat_id"], ""))
                                  for u in s["users"]],
                       [u["chat_id"] for u in s["users"]], [u["chat_id"] for u in s["users"]],
                       color_col=3)
            cur = self.to.currentData()
            self.to.blockSignals(True)
            self.to.clear()
            for u in s["users"]:
                if u["status"] == "approved":
                    self.to.addItem(u["name"] or str(u["chat_id"]), u["chat_id"])
            i = self.to.findData(cur)
            if i >= 0:
                self.to.setCurrentIndex(i)
            self.to.blockSignals(False)
        except Exception:
            logger.exception("admin tab refresh failed")

    @staticmethod
    def _fill(t, rows, keys, chats, color_col=None):
        sel = t.item(t.currentRow(), 0).data(Qt.UserRole) if t.currentRow() >= 0 and t.item(t.currentRow(), 0) else None
        t.setRowCount(len(rows))
        for r, (vals, key, chat) in enumerate(zip(rows, keys, chats)):
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c == 0:
                    it.setData(Qt.UserRole, key)
                    it.setData(Qt.UserRole + 1, chat)
                if color_col is not None and c == color_col:
                    it.setForeground(QColor(_STATUS_COLOR.get(v, TEXT)))
                t.setItem(r, c, it)
            if key == sel:
                t.selectRow(r)

    @staticmethod
    def _picked(t):
        it = t.item(t.currentRow(), 0) if t.currentRow() >= 0 else None
        return (it.data(Qt.UserRole), it.data(Qt.UserRole + 1)) if it else (None, None)

    # ── actions: off the GUI thread, errors reported, never raised ───────────
    def _act(self, label, fn, *args):
        bot = self._bot()
        if not bot:
            self.note.setText('The bot is not running.')
            return
        self.note.setText(label + "…")

        def run():
            try:
                fn(bot, *args)
            except Exception:
                logger.exception("admin action %s failed", label)
        threading.Thread(target=run, daemon=True).start()

    def _cancel_running(self):
        tid, _ = self._picked(self.run_t)
        if tid:
            self._act('⛔ Cancelling', lambda b, t: b.admin_cancel(t, "desktop"), tid)

    def _stop_running_chat(self):
        _, chat = self._picked(self.run_t)
        if chat:
            self._act('🛑 Stopping', lambda b, c: b.admin_stop_chat(c, "desktop"), chat)

    def _bump(self):
        tid, _ = self._picked(self.q_t)
        if tid:
            self._act('⬆️ Moving up', lambda b, t: b.admin_bump(t, "desktop"), tid)

    def _drop(self):
        tid, _ = self._picked(self.q_t)
        if tid:
            self._act('✖ Removing', lambda b, t: b.admin_cancel(t, "desktop"), tid)

    def _set_status(self, status):
        cid, _ = self._picked(self.u_t)
        if cid:
            self._act('Changing status', lambda b, c: b.admin_set_status(c, status, "desktop"), cid)

    def _stop_user(self):
        cid, _ = self._picked(self.u_t)
        if cid:
            self._act('🛑 Stopping', lambda b, c: b.admin_stop_chat(c, "desktop"), cid)

    def _say(self):
        text = self.msg.text().strip()
        if not text:
            return
        if self.all_cb.isChecked():
            ids = [u["chat_id"] for u in (self._snap or {}).get("users", []) if u["status"] == "approved"]
        else:
            ids = [self.to.currentData()] if self.to.currentData() else []
        if not ids:
            self.note.setText('To whom? Pick a user.')
            return
        self.msg.clear()
        self._act('📢 Sending (%d)' % len(ids),
                  lambda b, xs: [b.admin_say(c, text, "desktop") for c in xs], ids)

    def shutdown(self):
        self._poller.stop()
        self._poller.wait(3000)


def _plain(html: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", html or "")
