"""Telegram bot control panel, and the settings scope that protects the token.

Fourth tab out of gui.py. It moved together with its ENTIRE settings unit —
_TG_SETTINGS_SCOPE, _TG_SETTINGS_DIR, redirect_settings() and _tg_settings() —
and that is the important part of this file.

The panel PERSISTS the bot token when it starts, so any test that drives a real
TelegramTab writes into the user's live QSettings store and destroys their real
BotFather token. redirect_settings(tmpdir) is what prevents that: it points
_tg_settings() at a throwaway INI instead.

The guard only works if the flag and the function that reads it live in the
SAME module. Had redirect_settings stayed in gui.py while _TG_SETTINGS_DIR
moved here, gui.redirect_settings() would have set a variable nobody reads —
the protection would have gone silently dead and the suites would have gone
back to clobbering the real token, which is a failure this project has already
suffered once. gui.py re-exports redirect_settings so existing callers are
unaffected.

tg_bot.redirect_data_dir is the matching protection over the user database.
"""
import copy
import os
import threading
import time
from pathlib import Path

from PyQt5.QtCore import QSettings, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QCheckBox, QHBoxLayout, QHeaderView, QInputDialog,
                             QLabel, QLineEdit, QListWidget, QListWidgetItem,
                             QComboBox, QMessageBox, QPushButton, QScrollArea,
                             QSizePolicy, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from ui_scale import pt, px
from gui_common import (ACCENT2, BG, MUTED, PANEL2, TEXT, _FlowWidget,
                        _ScopedCtx, _flow, _section)

import logging
logger = logging.getLogger("assistant.gui")   # same channel as gui.py


_TG_TOKEN_KEY   = "telegram/bot_token"
_TG_TTS_KEY     = "telegram/tts_enabled"
_TG_AUTO_KEY    = "telegram/autostart"
_TG_ADMINS_KEY  = "telegram/admin_ids"
_TG_SILENT_KEY  = "telegram/silent_mode"
# my.telegram.org app credentials for the self-hosted Bot API server (2 GB
# files). Remembered like the token: saved on Start, masked in the field.
_TG_API_ID_KEY   = "telegram/api_id"
_TG_API_HASH_KEY = "telegram/api_hash"

# The panel PERSISTS the token on _start(), so any test that drives a real
# TelegramTab writes into the user's live store and destroys their real token.
# Tests call redirect_settings(tmpdir) first; see tg_bot.redirect_data_dir for
# the same protection over the user database.
_TG_SETTINGS_SCOPE = ("AssistantApp", "TelegramBot")
_TG_SETTINGS_DIR: str | None = None


def redirect_settings(dir_path) -> None:
    """Point the Telegram panel at a throwaway INI file instead of the real store.

    Pass None to restore the live scope.
    """
    global _TG_SETTINGS_DIR
    _TG_SETTINGS_DIR = str(dir_path) if dir_path else None


def _tg_settings() -> QSettings:
    # A suite that built the main window without redirect_settings() read the
    # REAL token + autostart and started the live bot for its lifetime: real
    # users' queued messages were answered from a test run (2026-09-28 23:15).
    # Under a test the live store is unreachable, redirected or not.
    if (not _TG_SETTINGS_DIR and os.environ.get("F5_TEST_RUN")
            and tuple(_TG_SETTINGS_SCOPE) == ("AssistantApp", "TelegramBot")):
        import tempfile
        redirect_settings(tempfile.mkdtemp(prefix="tg_settings_test_"))
    if _TG_SETTINGS_DIR:
        return QSettings(os.path.join(_TG_SETTINGS_DIR, "telegram_test.ini"),
                         QSettings.IniFormat)
    return QSettings(*_TG_SETTINGS_SCOPE)


# `host` is red on purpose: it means model-written code runs on this machine
# with no container around it, and it must never be quietly the same colour as
# an ordinary setting.
_SANDBOX_COLORS = {
    "files": "#7aa2f7",
    "code":  "#00c97a",
    "host":  "#cc3333",
}

_STATUS_COLORS = {
    "approved": "#00c97a",
    "pending":  "#f0a500",
    "rejected": "#e05050",
    "banned":   "#cc3333",
    "admin":    "#7b6cff",
}


def _feed_line(text: str, limit: int) -> str:
    """One feed row: whitespace collapsed, a long text cut on a word boundary with «…»."""
    one = " ".join((text or "").split())
    if len(one) <= limit:
        return one
    cut = one[:limit].rsplit(" ", 1)[0] or one[:limit]
    return cut.rstrip(",.;:—- ") + "…"


class TelegramTab(QWidget):
    """Telegram bot control panel.

    Sections
    ────────
    1. Bot settings (token, TTS, auto-start, admin IDs)
    2. Active dialogs + queue stats (live, refreshes every 2 s)
    3. Users — pending approval badge, table, approve/reject/ban/admin buttons
    4. Live log — real-time message/stage feed
    """

    # Signals emitted from the bot thread back to the Qt main thread
    # Telegram chat_ids are 64-bit and routinely exceed the +-2^31 range a plain
    # PyQt `int` signal arg silently truncates to (C `int`, not Python's
    # arbitrary-precision int) -- a large chat_id wrapped into a negative,
    # unrelated-looking number everywhere this signal's payload was displayed
    # (the live log, in particular). 'qint64' is PyQt's signal-type spelling
    # for a real 64-bit integer.
    _sig_status     = pyqtSignal(str)
    _sig_message    = pyqtSignal('qint64', str, str)   # chat_id, user_text, bot_reply
    _sig_stage      = pyqtSignal('qint64', str, bool)  # chat_id, stage_text, done
    _sig_user_change = pyqtSignal('qint64', str, str)  # chat_id, name, status

    def __init__(self, host):
        super().__init__()
        self.host = host
        self._bot  = None
        self._settings = _tg_settings()
        self._build_ui()
        self._sig_status.connect(self._on_status)
        self._sig_message.connect(self._on_message)
        self._sig_stage.connect(self._on_stage)
        self._sig_user_change.connect(self._on_user_change)
        self._active_chats: dict[int, str] = {}
        # The bot's OWN cancel token (see _ScopedCtx and _bot_ctx below).
        self._bot_cancel = threading.Event()
        self._bot_ctx_view = None

        saved = self._settings.value(_TG_TOKEN_KEY, "")
        if saved: self.token_in.setText(saved)
        import config as _cfg
        self.api_id_in.setText(str(self._settings.value(_TG_API_ID_KEY, "") or _cfg.TG_API_ID or ""))
        self.api_hash_in.setText(str(self._settings.value(_TG_API_HASH_KEY, "") or _cfg.TG_API_HASH or ""))
        self.tts_box.setChecked(self._settings.value(_TG_TTS_KEY, True, type=bool))
        self.auto_box.setChecked(self._settings.value(_TG_AUTO_KEY, False, type=bool))
        saved_admins = self._settings.value(_TG_ADMINS_KEY, "")
        if saved_admins: self.admin_ids_in.setText(saved_admins)
        self.silent_box.setChecked(self._settings.value(_TG_SILENT_KEY, False, type=bool))
        self.silent_box.stateChanged.connect(
            lambda v: self._settings.setValue(_TG_SILENT_KEY, bool(v)))

        # Refresh queue stats every 2 s while bot is running
        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._refresh_stats)
        self._stats_timer.setInterval(2000)

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.NoFrame)
        inner = QWidget()
        root = QVBoxLayout(inner)
        root.setContentsMargins(px(10), px(10), px(10), px(10))
        root.setSpacing(px(8))
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # ── 1. Bot settings ────────────────────────────────────────────────
        root.addWidget(_section("Telegram Bot"))

        trow = QHBoxLayout()
        trow.addWidget(QLabel("Bot token"))
        self.token_in = QLineEdit()
        self.token_in.setPlaceholderText("Paste token from @BotFather  e.g. 1234567890:AAH...")
        self.token_in.setEchoMode(QLineEdit.Password)
        self.show_btn = QPushButton("👁")
        self.show_btn.setFixedWidth(px(44))
        self.show_btn.setStyleSheet("padding: 0;")   # the QSS button padding left the eye a dot
        self.show_btn.setToolTip("Show the token")
        self.show_btn.setCheckable(True)
        self.show_btn.toggled.connect(
            lambda on: self.token_in.setEchoMode(
                QLineEdit.Normal if on else QLineEdit.Password))
        trow.addWidget(self.token_in, 1)
        trow.addWidget(self.show_btn)
        root.addLayout(trow)

        # Local Bot API server credentials (https://my.telegram.org/apps).
        # Same 👁 toggle as the token: they are secrets of the same class.
        krow = QHBoxLayout()
        krow.addWidget(QLabel("API id / hash"))
        self.api_id_in = QLineEdit()
        self.api_id_in.setPlaceholderText("api_id from my.telegram.org")
        self.api_id_in.setEchoMode(QLineEdit.Password)
        self.api_id_in.setFixedWidth(px(160))
        self.api_hash_in = QLineEdit()
        self.api_hash_in.setPlaceholderText("api_hash (32 hex) — enables files up to 2 GB via the local Bot API server")
        self.api_hash_in.setEchoMode(QLineEdit.Password)
        for w in (self.api_id_in, self.api_hash_in):
            w.setToolTip("App configuration block at https://my.telegram.org/apps. "
                         "With both set the app starts telegram-bot-api --local itself.")
        self.show_btn.toggled.connect(
            lambda on: [w.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password)
                        for w in (self.api_id_in, self.api_hash_in)])
        krow.addWidget(self.api_id_in)
        krow.addWidget(self.api_hash_in, 1)
        root.addLayout(krow)

        arow = QHBoxLayout()
        arow.addWidget(QLabel("Admin chat IDs"))
        self.admin_ids_in = QLineEdit()
        self.admin_ids_in.setPlaceholderText(
            "Comma-separated Telegram chat IDs of admins — e.g. 123456789,987654321")
        self.admin_ids_in.setToolTip(
            "These users are auto-approved as admins on first contact.\n"
            "Find your chat ID: send any message to the bot and look at the Live log.")
        arow.addWidget(self.admin_ids_in, 1)
        root.addLayout(arow)

        orow = _flow(spacing=px(22))   # a box nearer the previous label read as its box
        self.tts_box = QCheckBox("🔊  Send voice replies")
        self.tts_box.setToolTip(
            "After each text reply, synthesize with F5-TTS and send as a voice note.")
        self.tts_box.setChecked(True)
        self.auto_box = QCheckBox("🚀  Auto-start with app")
        self.auto_box.setToolTip("Start the bot automatically when the app opens.")
        self.silent_box = QCheckBox("🔕  Silent mode")
        self.silent_box.setToolTip(
            "Don't notify users when the bot goes online or offline.")
        orow.addWidget(self.tts_box)
        orow.addWidget(self.auto_box)
        orow.addWidget(self.silent_box)
        # The language a new user meets before they pick one (TG_DEFAULT_LANG in
        # .env). tg_strings binds it at import, so it takes effect on the next start.
        orow.addWidget(QLabel("Default bot language:"))
        self.bot_lang = QComboBox()
        for code, name in (("ru", "Русский"), ("en", "English")):
            self.bot_lang.addItem(name, code)
        self.bot_lang.setCurrentIndex(0 if os.getenv("TG_DEFAULT_LANG", "ru") != "en" else 1)
        self.bot_lang.setToolTip("Language of the bot for users who have not chosen one. "
                                 "Each user can still switch in the bot's Settings. Applies after a restart.")
        self.bot_lang.currentIndexChanged.connect(self._save_bot_lang)
        orow.addWidget(self.bot_lang)
        root.addWidget(_FlowWidget(orow))

        crow = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start bot")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("⏹  Stop")
        self.stop_btn.setObjectName("ghost")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        self.status_lbl = QLabel("Not running")
        self.status_lbl.setStyleSheet(f"color:{MUTED};")
        crow.addWidget(self.start_btn)
        crow.addWidget(self.stop_btn)
        crow.addStretch(1)
        crow.addWidget(self.status_lbl)
        root.addLayout(crow)

        hint = QLabel(
            "<b>How to get a token:</b> open Telegram → search <b>@BotFather</b> → "
            "send <code>/newbot</code> → follow the steps → copy the token above.<br>"
            "First-time users must register (name + password) and await admin approval.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{MUTED}; font-size:{pt(13)}px;")
        hint.setTextFormat(Qt.RichText)
        root.addWidget(hint)

        # ── 2. Active dialogs + queue stats ────────────────────────────────
        root.addWidget(_section("Active dialogs"))

        self.status_board = QLabel("—  no active dialogs")
        self.status_board.setWordWrap(True)
        self.status_board.setTextFormat(Qt.RichText)
        self.status_board.setStyleSheet(
            f"font-family:monospace; font-size:{pt(13)}px; "
            f"padding:{px(6)}px; background:rgba(0,0,0,0.18); "
            f"border-radius:{px(6)}px;")
        self.status_board.setMinimumHeight(px(48))
        root.addWidget(self.status_board)

        self.queue_lbl = QLabel("Queue: —")
        self.queue_lbl.setStyleSheet(f"color:{MUTED}; font-size:{pt(13)}px;")
        root.addWidget(self.queue_lbl)

        # ── 3. User management ─────────────────────────────────────────────
        root.addWidget(_section("Users"))

        # Pending badge
        self.pending_badge = QLabel("")
        self.pending_badge.setStyleSheet(
            f"color:{_STATUS_COLORS['pending']}; font-weight:700; font-size:{pt(14)}px;")
        root.addWidget(self.pending_badge)

        # User table: chat_id | Name | @username | Status | Admin | Registered
        self.user_table = QTableWidget(0, 7)
        self.user_table.setHorizontalHeaderLabels(
            ["Chat ID", "Name", "@username", "Status", "Admin", "Sandbox",
             "Registered"])
        self.user_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.user_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.user_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.user_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.user_table.setAlternatingRowColors(False)
        self.user_table.verticalHeader().setVisible(False)
        self.user_table.setMinimumHeight(px(140))
        self.user_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        root.addWidget(self.user_table)

        # Action buttons
        ubrow = QHBoxLayout()
        self.approve_btn = QPushButton("✅  Approve")
        self.approve_btn.setToolTip(
            "Select a user row, then click here to approve their account.\n"
            "You can also approve directly in Telegram if your chat ID is set as Admin.")
        self.approve_btn.clicked.connect(self._approve_selected)
        self.reject_btn  = QPushButton("❌  Reject")
        self.reject_btn.setObjectName("ghost")
        self.reject_btn.setToolTip("Reject selected user's account request")
        self.reject_btn.clicked.connect(self._reject_selected)
        self.ban_btn     = QPushButton("🚫  Ban")
        self.ban_btn.setObjectName("ghost")
        self.ban_btn.setToolTip("Ban selected user (blocks all future messages)")
        self.ban_btn.clicked.connect(self._ban_selected)
        self.admin_btn   = QPushButton("👑  Make admin")
        self.admin_btn.setObjectName("ghost")
        self.admin_btn.setToolTip("Grant admin panel access to selected user")
        self.admin_btn.clicked.connect(self._make_admin_selected)
        self.revoke_admin_btn = QPushButton("👤  Remove admin")
        self.revoke_admin_btn.setObjectName("ghost")
        self.revoke_admin_btn.setToolTip(
            "Revoke admin panel access from selected user (demotes to a "
            "regular approved user). Telegram admins can approve/reject account "
            "requests, but only this GUI can promote or demote admins.")
        self.revoke_admin_btn.clicked.connect(self._revoke_admin_selected)
        self.refresh_users_btn = QPushButton("🔄  Refresh")
        self.refresh_users_btn.setObjectName("ghost")
        self.refresh_users_btn.clicked.connect(self._refresh_users)
        ubrow.addWidget(self.approve_btn)
        ubrow.addWidget(self.reject_btn)
        ubrow.addWidget(self.ban_btn)
        ubrow.addWidget(self.admin_btn)
        ubrow.addWidget(self.revoke_admin_btn)
        ubrow.addStretch(1)
        ubrow.addWidget(self.refresh_users_btn)
        root.addLayout(ubrow)

        # Sandbox rights, on their own row. These are not account management:
        # they decide what a person may do to this machine, and the step from
        # "files" to "code" should be taken deliberately rather than by
        # clicking along a row of similar-looking buttons.
        sbrow = QHBoxLayout()
        sblbl = QLabel("Sandbox:")
        sblbl.setStyleSheet(f"color:{MUTED}; font-size:{pt(13)}px;")
        self.sbx_combo = QComboBox()
        self.sbx_combo.addItems(["off", "files", "code", "host"])
        self.sbx_combo.setToolTip(
            "off    — no working folder at all\n"
            "files  — keep, read and edit files; nothing executes\n"
            "code   — plus running code inside an isolated container\n"
            "host   — plus running code WITHOUT isolation when the\n"
            "         container engine is down. Only for someone you\n"
            "         trust with this PC itself.")
        self.sbx_apply_btn = QPushButton("Apply to selected")
        self.sbx_apply_btn.setObjectName("ghost")
        self.sbx_apply_btn.clicked.connect(self._apply_sandbox_level)
        self.sbx_reset_btn = QPushButton("♻  Reset their folder")
        self.sbx_reset_btn.setObjectName("ghost")
        self.sbx_reset_btn.setToolTip(
            "Delete everything in the selected user's working folder. "
            "Their access level is left unchanged.")
        self.sbx_reset_btn.clicked.connect(self._reset_sandbox_selected)
        self.sbx_status_lbl = QLabel("")
        self.sbx_status_lbl.setStyleSheet(f"color:{MUTED}; font-size:{pt(13)}px;")
        sbrow.addWidget(sblbl)
        sbrow.addWidget(self.sbx_combo)
        sbrow.addWidget(self.sbx_apply_btn)
        sbrow.addWidget(self.sbx_reset_btn)
        sbrow.addWidget(self.sbx_status_lbl, 1)
        root.addLayout(sbrow)

        # DB backup controls
        dbrow = QHBoxLayout()
        self.backup_info_lbl = QLabel("DB: tg_users.db  (SQLite WAL)")
        self.backup_info_lbl.setStyleSheet(f"color:{MUTED}; font-size:{pt(13)}px;")
        self.restore_btn = QPushButton("⏮  Restore backup")
        self.restore_btn.setObjectName("ghost")
        self.restore_btn.setToolTip("Pick a snapshot from tg_users_backups/ and restore it")
        self.restore_btn.clicked.connect(self._restore_backup)
        dbrow.addWidget(self.backup_info_lbl, 1)
        dbrow.addWidget(self.restore_btn)
        root.addLayout(dbrow)

        # ── 4. Live log ────────────────────────────────────────────────────
        root.addWidget(_section("Live log"))

        self.log = QListWidget()
        self.log.setWordWrap(True)
        self.log.setMinimumHeight(px(160))
        self.log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.log, 1)

        lbrow = QHBoxLayout()
        clr = QPushButton("🗑  Clear log"); clr.setObjectName("ghost")
        clr.clicked.connect(self.log.clear)
        exp_btn = QPushButton("📁  Export log"); exp_btn.setObjectName("ghost")
        exp_btn.setToolTip("Copy path to activity log file")
        exp_btn.clicked.connect(self._export_log)
        lbrow.addWidget(clr)
        lbrow.addWidget(exp_btn)
        lbrow.addStretch(1)
        root.addLayout(lbrow)

    # ── control ────────────────────────────────────────────────────────────

    def _parse_admin_ids(self) -> list[int]:
        raw = self.admin_ids_in.text().strip()
        ids = []
        for part in raw.replace(";", ",").split(","):
            part = part.strip()
            if part.lstrip("-").isdigit():
                ids.append(int(part))
        return ids

    def _bot_ctx(self):
        """A private view of the assistant context for the bot (see _ScopedCtx).

        ctx.cancel_event is a single global flag, and the bot both SETS it (⛔ Stop,
        the per-request Cancel button, the watchdog's runaway check) and CLEARS it
        at the start of every task. On the raw ctx each of those reaches straight
        into the desktop chat:

          · a Telegram user's Stop killed the desktop user's in-flight turn;
          · the desktop Stop button aborted a remote user's task mid-answer —
            _cancel_current reaches transfer_tab and storyboard_tab by name and
            never knew about the bot;
          · and a Telegram message arriving just after a desktop Stop *un-cancelled*
            the desktop turn, because starting a task clears the flag.

        Every other tab that runs long jobs already takes a scoped view; the bot
        was the one that did not.
        """
        ctx = getattr(self.host, "ctx", None)
        if ctx is None:
            return None
        view = self._bot_ctx_view
        # Rebuild when the host swaps its context (profile switch) — but keep the
        # same cancel token, or work in flight becomes uncancellable.
        if view is None or object.__getattribute__(view, "_ctx") is not ctx:
            view = _ScopedCtx(ctx, self._bot_cancel)
            self._bot_ctx_view = view
        return view

    def _save_bot_lang(self):
        from dotenv import set_key
        env = Path(__file__).resolve().parents[1] / ".env"
        env.touch()
        set_key(str(env), "TG_DEFAULT_LANG", self.bot_lang.currentData(), quote_mode="never")
        from gui_i18n import tr
        self._on_status(tr("Default bot language saved — restart the app to apply it."))

    def _start(self):
        token = self.token_in.text().strip()
        if not token:
            QMessageBox.warning(self, "No token",
                                "Paste your BotFather token first.")
            return
        self._settings.setValue(_TG_TOKEN_KEY,  token)
        self._settings.setValue(_TG_TTS_KEY,    self.tts_box.isChecked())
        self._settings.setValue(_TG_AUTO_KEY,   self.auto_box.isChecked())
        self._settings.setValue(_TG_ADMINS_KEY,  self.admin_ids_in.text().strip())
        self._settings.setValue(_TG_SILENT_KEY, self.silent_box.isChecked())
        self._settings.setValue(_TG_API_ID_KEY,   self.api_id_in.text().strip())
        self._settings.setValue(_TG_API_HASH_KEY, self.api_hash_in.text().strip())
        try:
            import tg_local_api
            tg_local_api.apply_credentials(self.api_id_in.text().strip(),
                                           self.api_hash_in.text().strip())
        except Exception as exc:
            logger.warning("local Bot API credentials not applied: %s", exc)

        from tg_bot import TelegramBot
        self._bot = TelegramBot(
            token=token,
            get_ctx=self._bot_ctx,
            get_graph=lambda: getattr(self.host, "graph", None),
            get_base_state=lambda: copy.deepcopy(
                getattr(self.host, "base_state", None) or {}),
            on_status=self._sig_status.emit,
            on_message=self._sig_message.emit,
            on_stage=self._sig_stage.emit,
            tts_enabled_fn=self.tts_box.isChecked,
            admin_chat_ids=self._parse_admin_ids(),
            on_user_change=self._sig_user_change.emit,
            silent_mode=self.silent_box.isChecked(),
        )
        # Only claim the bot is running if it actually started. This used to flip
        # the whole panel into the running state regardless, so a bad token or a
        # dropped network left the UI insisting the bot was up: Start greyed out,
        # Stop live, and every message the user sent silently unanswered.
        if not self._bot.start():
            self._bot = None
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.token_in.setEnabled(True)
            self.admin_ids_in.setEnabled(True)
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.token_in.setEnabled(False)
        self.admin_ids_in.setEnabled(False)
        self._stats_timer.start()
        self._refresh_users()

    def _stop(self):
        self._stats_timer.stop()
        if self._bot:
            self._bot.stop()
            self._bot = None
        self._active_chats.clear()
        self.status_board.setText("—  no active chats")
        self.queue_lbl.setText("Queue: —")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.token_in.setEnabled(True)
        self.admin_ids_in.setEnabled(True)

    def maybe_autostart(self):
        if self.auto_box.isChecked() and self.token_in.text().strip() and self._bot is None:
            self._start()

    def shutdown(self):
        if self._bot:
            self._bot.stop()

    # ── user management slots ─────────────────────────────────────────────

    def _selected_chat_id(self) -> int | None:
        rows = self.user_table.selectedItems()
        if not rows: return None
        row = self.user_table.currentRow()
        item = self.user_table.item(row, 0)
        if not item: return None
        try: return int(item.text())
        except ValueError: return None

    def _approve_selected(self):
        cid = self._selected_chat_id()
        if cid is None:
            QMessageBox.information(self, "No selection", "Select a user row first."); return
        if self._bot: self._bot.approve_user(cid, allow_from_any_status=True)
        self._refresh_users()

    def _reject_selected(self):
        cid = self._selected_chat_id()
        if cid is None:
            QMessageBox.information(self, "No selection", "Select a user row first."); return
        if self._bot: self._bot.reject_user(cid, allow_from_any_status=True)
        self._refresh_users()

    def _ban_selected(self):
        cid = self._selected_chat_id()
        if cid is None:
            QMessageBox.information(self, "No selection", "Select a user row first."); return
        reply = QMessageBox.question(self, "Confirm ban",
            f"Ban user {cid}? They will no longer be able to use the bot.",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self._bot: self._bot.ban_user(cid, allow_from_any_status=True)
            self._refresh_users()

    def _make_admin_selected(self):
        cid = self._selected_chat_id()
        if cid is None:
            QMessageBox.information(self, "No selection", "Select a user row first."); return
        if self._bot: self._bot.make_admin(cid)
        self._refresh_users()

    def _revoke_admin_selected(self):
        cid = self._selected_chat_id()
        if cid is None:
            QMessageBox.information(self, "No selection", "Select a user row first."); return
        reply = QMessageBox.question(self, "Confirm removal",
            f"Remove admin access from user {cid}?",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self._bot: self._bot.revoke_admin(cid)
            self._refresh_users()

    # -- sandbox rights --------------------------------------------------------
    def _apply_sandbox_level(self):
        """Grant or revoke the selected user's sandbox level.

        `host` asks for confirmation and the other levels do not, because it is
        the only one that removes the container: at that level a script the
        model wrote runs on this machine as this user. That is worth one extra
        click.
        """
        cid = self._selected_chat_id()
        if cid is None:
            self.sbx_status_lbl.setText("Select a user row first.")
            return
        level = self.sbx_combo.currentText()
        if level == "host":
            ok = QMessageBox.warning(
                self, "Unisolated execution",
                f"Give {cid} UNISOLATED code execution?\n\n"
                f"At this level, whenever the container engine is not running, "
                f"code the model writes runs directly on this PC as you, and can "
                f"read or change anything you can. Only for someone you trust "
                f"with the machine itself.",
                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
            if ok != QMessageBox.Yes:
                return
        store = self._sandbox_user_store()
        if store is None:
            self.sbx_status_lbl.setText("User store unavailable.")
            return
        try:
            import sandbox_access as _sa
            _sa.set_level(store, cid, level)
        except Exception as exc:
            self.sbx_status_lbl.setText(f"Failed: {exc}")
            return
        self.sbx_status_lbl.setText(f"{cid}: sandbox = {level}")
        self._refresh_users()

    def _reset_sandbox_selected(self):
        cid = self._selected_chat_id()
        if cid is None:
            self.sbx_status_lbl.setText("Select a user row first.")
            return
        if QMessageBox.question(
                self, "Reset working folder",
                f"Delete everything in {cid}'s working folder?\n"
                f"Their uploads and any work in progress go with it.",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel) != QMessageBox.Yes:
            return
        try:
            import code_sandbox as _cs
            removed = _cs.sandbox_for(cid).reset()
        except Exception as exc:
            self.sbx_status_lbl.setText(f"Failed: {exc}")
            return
        self.sbx_status_lbl.setText(f"{cid}: removed {removed} item(s)")

    def _sandbox_user_store(self):
        """The user store, whether or not the bot happens to be running.

        Rights have to be revocable while the bot is stopped. Requiring it to be
        started before access can be taken away is exactly backwards.
        """
        bot = getattr(self, "_bot", None)
        store = getattr(bot, "_user_store", None) if bot is not None else None
        if store is not None:
            return store
        try:
            import tg_bot as _T
            return _T._UserStore(_T._USERS_DB)
        except Exception:
            return None

    def _refresh_users(self):
        if not self._bot: return
        users = self._bot.get_users()
        self.user_table.setRowCount(0)
        pending_count = 0
        for u in sorted(users, key=lambda x: x.registered_at, reverse=True):
            row = self.user_table.rowCount()
            self.user_table.insertRow(row)
            reg_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(u.registered_at))
            row_bg = QColor(PANEL2 if row % 2 else BG)
            # The badge is the admin's own customization (picked via /account in
            # Telegram, not assigned here) -- shown as a name prefix so it's
            # visible at a glance without adding another column.
            badge = (getattr(u, "prefs", None) or {}).get("badge") if u.is_admin else ""
            name_disp = f"{badge} {u.name}" if badge else u.name
            # The sandbox level is the most consequential per-user setting on
            # this screen -- `code` means that person can execute code on this
            # machine -- so it gets a column instead of hiding inside prefs.
            import sandbox_access as _sa
            _lvl = _sa.level_for(u)
            for col, val in enumerate([
                str(u.chat_id), name_disp, f"@{u.tg_username}" if u.tg_username else "",
                u.status, "👑" if u.is_admin else "",
                "" if _lvl == _sa.OFF else _lvl, reg_str
            ]):
                cell = QTableWidgetItem(val)
                cell.setBackground(row_bg)
                cell.setForeground(QColor(TEXT))
                if col == 3:   # status column — colour-code by status
                    color = _STATUS_COLORS.get(u.status, TEXT)
                    if u.is_admin and u.status == "approved":
                        color = _STATUS_COLORS["admin"]
                    cell.setForeground(QColor(color))
                if col == 5 and val:   # sandbox — host must never blend in
                    cell.setForeground(QColor(_SANDBOX_COLORS.get(val, TEXT)))
                self.user_table.setItem(row, col, cell)
            if u.status == "pending":
                pending_count += 1
        if pending_count:
            self.pending_badge.setText(
                f"🔔  {pending_count} user{'s' if pending_count > 1 else ''} "
                f"awaiting approval — select a row below and click ✅ Approve")
        else:
            self.pending_badge.setText("")

        # Update the tab title to show pending count as a badge
        tabs = getattr(self.host, "tabs", None)
        if tabs is not None:
            for i in range(tabs.count()):
                if tabs.tabText(i).startswith("Telegram"):
                    label = f"Telegram ({pending_count})" if pending_count else "Telegram"
                    tabs.setTabText(i, label)
                    break

    def _on_user_change(self, chat_id: int, name: str, status: str):
        """Called from bot thread via signal when any user's status changes."""
        stamp = time.strftime("%H:%M:%S")
        import html as _html
        icon = {"approved": "✅", "pending": "⏳", "rejected": "❌",
                "banned": "🚫", "admin": "👑", "admin_revoked": "👤"}.get(status, "👤")
        item = QListWidgetItem(
            f"[{stamp}]  {icon} user {chat_id} ({_html.escape(name)}): {status}")
        color = _STATUS_COLORS.get(status, MUTED)
        item.setForeground(QColor(color))
        self._log_append(item)
        self._refresh_users()

    def _refresh_stats(self):
        if not self._bot: return
        try:
            stats = self._bot.get_queue_stats()
            depth = stats.get("depth", 0)
            self.queue_lbl.setText(
                f"Queue: <b>{depth}</b> task{'s' if depth != 1 else ''} waiting  "
                f"·  backend: {stats.get('backend', '?')}")
            self.queue_lbl.setTextFormat(Qt.RichText)
        except Exception:
            pass

    def _restore_backup(self):
        if not self._bot:
            QMessageBox.warning(self, "Bot not running",
                                "Start the bot first so the DB path is known.")
            return
        backups = self._bot.list_db_backups()
        if not backups:
            QMessageBox.information(self, "No backups",
                "No backup snapshots found yet.\n"
                "Snapshots are written automatically on every user change.")
            return
        # Build a list of labels for the picker
        labels = [f"{p.name}  ({p.stat().st_size // 1024} KB)" for p in backups]
        choice, ok = QInputDialog.getItem(self, "Restore backup",
            "Select snapshot to restore (newest first):", labels, 0, False)
        if not ok: return
        idx = labels.index(choice)
        chosen = backups[idx]
        reply = QMessageBox.question(self, "Confirm restore",
            f"Replace the live user database with:\n{chosen.name}\n\n"
            "This will overwrite all current user records. Continue?",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes: return
        ok2 = self._bot.restore_db_from_backup(chosen)
        if ok2:
            QMessageBox.information(self, "Restored",
                f"Database restored from {chosen.name}.\nRefreshing user list…")
            self._refresh_users()
        else:
            QMessageBox.critical(self, "Restore failed",
                "Could not restore the backup. Check the log for details.")

    def _export_log(self):
        from pathlib import Path
        log_path = Path(__file__).resolve().parents[1].joinpath("tg_activity.jsonl")
        if log_path.exists():
            try:
                n_lines = sum(1 for _ in open(log_path, encoding="utf-8",
                                              errors="replace"))
            except Exception:
                n_lines = "?"
            QMessageBox.information(self, "Activity log",
                f"Log file location:\n{log_path}\n\n"
                f"Size: {log_path.stat().st_size // 1024} KB  ·  {n_lines} entries\n\n"
                "Copy the path above to open it in any text editor or spreadsheet.")
        else:
            QMessageBox.information(self, "Activity log",
                "No activity logged yet. Send some messages first.")

    # ── live log helpers ──────────────────────────────────────────────────

    _LOG_MAX = 500

    def _log_append(self, item):
        self.log.addItem(item)
        while self.log.count() > self._LOG_MAX:
            self.log.takeItem(0)
        self.log.scrollToBottom()

    def _on_status(self, text: str):
        self.status_lbl.setText(text)
        item = QListWidgetItem(f"[bot]  {text}")
        item.setForeground(QColor(ACCENT2))
        self._log_append(item)

    def _on_stage(self, chat_id: int, stage: str, done: bool):
        import html as _html
        from gui_i18n import tr
        stage = tr("Done" if done else stage)
        stamp = time.strftime("%H:%M:%S")
        cid   = str(chat_id)
        if done:
            self._active_chats.pop(chat_id, None)
        else:
            self._active_chats[chat_id] = stage
        if self._active_chats:
            rows = [f"👤 <b>{_html.escape(str(c))}</b>  →  {_html.escape(str(s))}"
                    for c, s in self._active_chats.items()]
            self.status_board.setText("<br>".join(rows))
        else:
            self.status_board.setText("—  no active chats")
        icon = "✅" if done else "⚙️"
        item = QListWidgetItem(f"[{stamp}]  {icon} {tr('user')} {cid}:  {stage}")
        item.setForeground(QColor("#888888"))
        self._log_append(item)
        # Mirror Telegram stage into the main header indicator so the user
        # can see what the bot is doing without opening the Telegram tab.
        main_stage = getattr(self.host, "stage", None)
        if main_stage is not None:
            main_stage.set_stage("Ready" if done else stage)

    def _on_message(self, chat_id: int, user_text: str, bot_reply: str):
        stamp = time.strftime("%H:%M:%S")
        cid   = str(chat_id)
        # A hard [:150] cut the reply mid-word with no mark («…делает схему ещё чётч», 10-09):
        # a longer line ends on a word with «…», and the whole message is on hover.
        item_u = QListWidgetItem(f"[{stamp}]  👤 {cid}:  {_feed_line(user_text, 300)}")
        item_b = QListWidgetItem(f"[{stamp}]  🤖 {cid}:  {_feed_line(bot_reply, 500)}")
        item_u.setToolTip((user_text or "")[:4000])
        item_b.setToolTip((bot_reply or "")[:4000])
        item_b.setForeground(QColor(ACCENT2))
        self._log_append(item_u)
        self._log_append(item_b)
