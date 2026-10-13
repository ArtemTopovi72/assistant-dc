"""🧪 Debug chat: a Telegram-like chat with the RUNNING bot, as the debug user
(bot/debug_user.py). Its messages wait in the same queue as everyone's, so what it
shows is what a real user gets -- statuses, edits, buttons, pictures -- without a
phone and without stopping the bot. Everything goes through the debug user's HTTP API
on 127.0.0.1, the same one scripts use.
"""
from seams import file_dialog
import html
import json
import os
import time
import urllib.request

from PyQt5.QtCore import QThread, Qt, QTimer, QUrl, pyqtSignal
from PyQt5.QtWidgets import (QTabWidget, QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QTextBrowser, QVBoxLayout, QWidget)

from ui_scale import px
from gui_common import open_in_os, ACCENT, BORDER, MUTED, PANEL, PANEL2, TEXT, _section

import logging
logger = logging.getLogger("assistant.gui")

_BOT_BUBBLE = "#2a2d3a"
_ME_BUBBLE = "#2b5278"          # Telegram dark theme's outgoing bubble


def _api():
    import debug_user
    return f"http://127.0.0.1:{debug_user.API_PORT}"


def _call(path: str, body: dict = None, timeout: float = 5.0) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(_api() + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class _Poller(QThread):
    """Asks the API for new bot messages twice a second, off the UI thread."""
    got = pyqtSignal(list)
    state = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.since = time.time() - 3600
        self.user = 0
        self._stop = False

    def run(self):
        while not self._stop:
            try:
                msgs = _call("/messages?since=%f&user=%d" % (self.since, self.user)).get("messages") or []
                self.state.emit(True)
                if msgs:
                    self.since = max(m["t"] for m in msgs) + 1e-6
                    self.got.emit(msgs)
            except Exception:
                self.state.emit(False)
            self.msleep(500)

    def stop(self):
        self._stop = True


class DebugChatTab(QWidget):
    def __init__(self, host=None, poller=None, shot=None):
        super().__init__()
        self.host = host
        self.shot = shot
        self._msgs = []                  # [{"id", "me", "text", "html", "files"}], in order
        self._by_id = {}
        self._reply_kb = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(px(12), px(12), px(12), px(12))
        lay.setSpacing(px(8))

        head = QHBoxLayout()
        head.addWidget(_section("Debug chat"))
        head.addStretch(1)
        self.status = QLabel("○ offline")
        self.status.setStyleSheet(f"color:{MUTED};")
        head.addWidget(self.status)
        self.who = QComboBox()
        self.who.addItems([f"User {i}" for i in range(5)])
        self.who.setToolTip("Which debug user you are; open several to write in parallel")
        self.who.currentIndexChanged.connect(self._switch_user)
        head.addWidget(self.who)
        self.enable = QCheckBox("Debug user on")
        self.enable.setToolTip("Lets the bot read this chat. Its messages queue with everyone's.")
        self.enable.toggled.connect(self._toggle)
        head.addWidget(self.enable)
        clear = QPushButton("Clear view")
        clear.clicked.connect(self._clear)
        head.addWidget(clear)
        lay.addLayout(head)

        hint = QLabel("You are an ordinary user of the running bot: same queue, same answers, "
                      "nothing goes to Telegram.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{MUTED};")
        lay.addWidget(hint)

        self.view = QTextBrowser()
        self.view.setOpenLinks(False)
        self.view.anchorClicked.connect(self._open)
        self.view.setStyleSheet(f"QTextBrowser{{background:{PANEL};border:1px solid {BORDER};"
                                f"border-radius:{px(10)}px;padding:{px(8)}px;color:{TEXT};}}")
        lay.addWidget(self.view, 1)

        self.buttons = QWidget()
        self.btn_grid = QGridLayout(self.buttons)
        self.btn_grid.setContentsMargins(0, 0, 0, 0)
        self.btn_grid.setSpacing(px(6))
        lay.addWidget(self.buttons)

        row = QHBoxLayout()
        attach = QPushButton("📎")
        attach.setToolTip("Send pictures (several = one album); the text goes as the caption")
        attach.setFixedWidth(px(44))
        attach.clicked.connect(self._attach)
        row.addWidget(attach)
        self.input = QLineEdit()
        self.input.setPlaceholderText("Message")
        self.input.returnPressed.connect(self._send)
        row.addWidget(self.input, 1)
        send = QPushButton("Send")
        send.setStyleSheet(f"QPushButton{{background:{ACCENT};color:white;border-radius:{px(6)}px;"
                           f"padding:{px(6)}px {px(16)}px;}}")
        send.clicked.connect(self._send)
        row.addWidget(send)
        lay.addLayout(row)

        try:
            import debug_user
            self.enable.setChecked(debug_user.enabled())
        except Exception:
            pass

        self._poller = poller or _Poller()
        self._poller.got.connect(self._on_bot)
        self._poller.state.connect(self._on_state)
        self._poller.start()
        self._shot_timer = QTimer(self)
        self._shot_timer.timeout.connect(self._take_shot)
        self._shot_timer.start(1000)

    def _take_shot(self):
        """POST /shot: this tab brought to the front and the whole window saved, so a
        debugging session can see the live GUI without a desktop grant."""
        try:
            import debug_user
        except Exception:
            return
        shot = self.shot or debug_user.SHOT
        if not os.path.exists(shot + ".req"):
            return
        try:
            os.remove(shot + ".req")
        except OSError:
            return
        w = self
        while w is not None:
            stack = w.parentWidget()
            tabs = stack.parentWidget() if stack is not None else None
            if isinstance(tabs, QTabWidget):
                tabs.setCurrentWidget(w)
            w = stack
        QTimer.singleShot(400, lambda: self.window().grab().save(shot))

    # ---- the user side ----

    def _toggle(self, on: bool):
        import debug_user
        flag = os.path.join(debug_user.ROOT, "enabled")
        os.makedirs(debug_user.ROOT, exist_ok=True)
        if on and not os.path.exists(flag):
            open(flag, "w").close()
        elif not on and os.path.exists(flag):
            os.remove(flag)

    def _switch_user(self, i: int):
        self._poller.user = i
        self._poller.since = time.time() - 3600
        self._clear()

    def _post(self, path: str, body: dict) -> bool:
        try:
            _call(path, dict(body, user=self.who.currentIndex()))
            return True
        except Exception as exc:
            self._add(None, False, f"⚠️ The bot did not take it ({exc}). Is the bot running "
                                   "and «Debug user on» checked?", [], plain=True)
            return False

    def _send(self):
        text = self.input.text().strip()
        if not text:
            return
        if self._post("/say", {"text": text}):
            self._add(None, True, text, [], plain=True)
            self.input.clear()

    def _attach(self):
        paths, _ = file_dialog(self).getOpenFileNames(self, "Pictures", "",
                                                "Images (*.png *.jpg *.jpeg *.webp)")
        if not paths:
            return
        caption = self.input.text().strip()
        ok = (self._post("/photo", {"path": paths[0], "caption": caption}) if len(paths) == 1
              else self._post("/album", {"paths": paths, "caption": caption}))
        if ok:
            self._add(None, True, caption, paths, plain=True)
            self.input.clear()

    def _press(self, data: str, label: str):
        if self._post("/press", {"data": data}):
            self._add(None, True, f"[{label}]", [], plain=True)

    def _say_button(self, label: str):
        if self._post("/say", {"text": label}):
            self._add(None, True, label, [], plain=True)

    # ---- the bot side ----

    def _on_state(self, up: bool):
        self.status.setText("● live" if up else "○ offline")
        self.status.setStyleSheet(f"color:{'#3ccf91' if up else MUTED};")

    def _on_bot(self, events: list):
        for e in events:
            m, p = e["method"], e.get("payload") or {}
            text = p.get("text") or p.get("caption") or ""
            markup = p.get("reply_markup")
            if isinstance(markup, str):
                try:
                    markup = json.loads(markup)
                except ValueError:
                    markup = None
            if m == "deleteMessage":
                old = self._by_id.pop(p.get("message_id"), None)
                if old in self._msgs:
                    self._msgs.remove(old)
            elif m.startswith("edit"):
                old = self._by_id.get(p.get("message_id"))
                if old is not None:
                    if text:
                        old["text"], old["plain"] = text, p.get("parse_mode") != "HTML"
                    if "reply_markup" in p:
                        old["markup"] = markup
            elif m.startswith("send") and m != "sendChatAction":
                self._add(e.get("message_id"), False, text, e.get("files") or [],
                          plain=p.get("parse_mode") != "HTML", markup=markup, render=False)
        self._render()

    def _add(self, mid, me, text, files, plain=False, markup=None, render=True):
        msg = {"id": mid, "me": me, "text": text, "files": files, "plain": plain, "markup": markup}
        self._msgs.append(msg)
        if mid is not None:
            self._by_id[mid] = msg
        if render:
            self._render()

    def _clear(self):
        self._msgs.clear()
        self._by_id.clear()
        self._render()

    def _render(self):
        parts = []
        for m in self._msgs[-200:]:
            body = (html.escape(m["text"]).replace("\n", "<br>") if m["plain"]
                    else m["text"].replace("\n", "<br>"))
            imgs = "".join(
                f'<a href="{QUrl.fromLocalFile(f).toString()}"><img src="{QUrl.fromLocalFile(f).toString()}" '
                f'width="{px(260)}"></a><br>' for f in m["files"]
                if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg", ".webp", ".gif"))
            other = "".join(f'<a href="{QUrl.fromLocalFile(f).toString()}">📎 {html.escape(os.path.basename(f))}</a><br>'
                            for f in m["files"] if os.path.splitext(f)[1].lower()
                            not in (".png", ".jpg", ".jpeg", ".webp", ".gif"))
            align, bg = ("right", _ME_BUBBLE) if m["me"] else ("left", _BOT_BUBBLE)
            parts.append(
                f'<table width="100%" cellspacing="0" cellpadding="0"><tr><td align="{align}">'
                f'<table {'width="78%" ' if len(m["text"]) > 60 else ''}cellpadding="{px(8)}" style="background:{bg};border-radius:{px(10)}px;'
                f'margin:{px(3)}px;"><tr><td style="color:{TEXT};">{imgs}{other}{body}</td></tr>'
                f'</table></td></tr></table>')
        self.view.setHtml("".join(parts))
        self.view.verticalScrollBar().setValue(self.view.verticalScrollBar().maximum())
        self._render_buttons()

    def _render_buttons(self):
        while self.btn_grid.count():
            w = self.btn_grid.takeAt(0).widget()
            if w:
                w.deleteLater()
        last = next((m for m in reversed(self._msgs) if not m["me"] and m.get("markup")), None)
        mk = (last or {}).get("markup") or {}
        rows = mk.get("inline_keyboard") or []
        if mk.get("keyboard"):
            self._reply_kb = mk["keyboard"]
        if not rows:
            rows = self._reply_kb
        for r, row in enumerate(rows[:8]):
            for c, b in enumerate(row[:4]):
                label = b.get("text", "") if isinstance(b, dict) else str(b)
                btn = QPushButton(label)
                btn.setStyleSheet(f"QPushButton{{background:{PANEL2};color:{TEXT};border:1px solid {BORDER};"
                                  f"border-radius:{px(6)}px;padding:{px(5)}px;}}"
                                  f"QPushButton:hover{{border-color:{ACCENT};}}")
                if isinstance(b, dict) and b.get("callback_data"):
                    btn.clicked.connect(lambda _=False, d=b["callback_data"], t=label: self._press(d, t))
                elif isinstance(b, dict) and b.get("url"):
                    btn.setToolTip(b["url"])
                    btn.setEnabled(False)
                else:
                    btn.clicked.connect(lambda _=False, t=label: self._say_button(t))
                self.btn_grid.addWidget(btn, r, c)

    def _open(self, url: QUrl):
        try:
            open_in_os(url.toLocalFile())
        except Exception:
            logger.info("debug chat: cannot open %s", url.toString())

    def closeEvent(self, e):
        self._poller.stop()
        self._poller.wait(1500)
        super().closeEvent(e)
