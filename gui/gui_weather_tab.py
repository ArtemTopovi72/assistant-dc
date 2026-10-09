"""Weather tab: the Telegram forecast, drawn as a real desktop table.

Modeled on MusicTab (a self-contained QWidget(host) tab plus one QThread that
does the slow call and reports back via signals). The lookup makes an HTTP
call AND normally an LLM call, so it cannot run on the UI thread.

What it deliberately does NOT reuse is weather.format_forecast: that renders a
monospace <pre> block sized for a phone, where the conditions had to become an
icon with a legend underneath just to keep rows from wrapping. A desktop tab
has the width for a real QTableWidget with the description in the row, so this
surface renders from weather.forecast_data's buckets instead. Everything that
is a DECISION rather than a layout -- which advice applies to a bucket, how
identical advice is grouped, what the conditions are called -- comes from
weather.py, so the two surfaces cannot disagree about the forecast itself.
"""
import datetime as _dt

import gui_i18n

from PyQt5.QtCore import QDate, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QAbstractItemView, QDateEdit, QHBoxLayout,
                             QHeaderView, QLabel, QLineEdit, QPushButton,
                             QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)

import config as _cfg
from ui_scale import px
from gui_common import MUTED, _section

import logging
logger = logging.getLogger("assistant.gui")   # same channel as gui.py


class WeatherWorker(QThread):
    """Geocode, fetch, and ask the LLM for clothing advice — all off the UI
    thread. One worker covers the whole lookup because there is nothing worth
    showing in between: a resolved city with no forecast yet is not a usable
    interim result."""
    done = pyqtSignal(dict, list, dict, str)   # loc, buckets, advice_map, error
    failed = pyqtSignal(str)

    def __init__(self, ctx, city: str, lang: str = "ru", *,
                 hours: int = 24, date=None):
        super().__init__()
        self.ctx, self.city, self.lang = ctx, city, lang
        # Snapshotted at construction, never read off the widgets in run():
        # run() is on the worker thread, and touching Qt widgets from there is
        # the unhandled-slot-exception path that aborts the process with no
        # traceback.
        self.hours, self.date = hours, date

    def run(self):
        try:
            import weather as _w
            advise_fn = correct_fn = None
            if self.ctx is not None:
                # No ctx means no model loaded: the forecast is still worth
                # showing, it just falls back to the fixed temperature bands
                # and cannot fix a misspelled city.
                advise_fn = _w.periods_advise_fn(self.ctx)
                correct_fn = _w.city_correct_fn(self.ctx)
            loc = _w.resolve_city(self.city, correct_fn=correct_fn)
            if not loc:
                self.done.emit({}, [], {},
                               _w.not_found_message(self.city, self.lang))
                return
            buckets, advice_map, error = _w.forecast_data(
                loc, self.lang, hours=self.hours, date=self.date,
                advise_fn=advise_fn)
            self.done.emit(loc, buckets, advice_map, error)
        except Exception as exc:
            logger.exception("Weather lookup failed")
            self.failed.emit(str(exc))


class WeatherTab(QWidget):
    """A city, four windows to look at it through, and what to wear.

    The four buttons mirror Telegram's reworked 🌤 keyboard: each window is one
    press, rather than 48h being reachable only as a follow-up under a 24h
    result (which meant sitting through a lookup nobody asked for).
    """

    def __init__(self, host):
        super().__init__()
        self.host = host                  # AssistantWindow (for ctx)
        self.worker = None

        root = QVBoxLayout(self)
        root.setContentsMargins(px(6), px(6), px(6), px(6))
        root.setSpacing(px(8))

        root.addWidget(_section("Weather"))
        help_lbl = QLabel("Pick a city and a window. Humidity and wind are "
                          "shown alongside the temperature, and the clothing "
                          "advice comes from the model when one is loaded.")
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet(f"color:{MUTED};")
        root.addWidget(help_lbl)

        row = QHBoxLayout()
        self.city_in = QLineEdit()
        self.city_in.setText(getattr(_cfg, "WEATHER_DEFAULT_CITY", "") or "")
        self.city_in.setPlaceholderText("City")
        self.city_in.returnPressed.connect(lambda: self._lookup(hours=24))
        row.addWidget(QLabel("City:"))
        row.addWidget(self.city_in, 1)
        root.addLayout(row)

        btn_row = QHBoxLayout()
        self._buttons = []
        for label, hours in (("Now", 0), ("24 h", 24), ("48 h", 48)):
            b = QPushButton(label)
            # hours=hours binds the value per button; a bare closure over the
            # loop variable would give every button the last window.
            b.clicked.connect(lambda _checked, h=hours: self._lookup(hours=h))
            btn_row.addWidget(b)
            self._buttons.append(b)

        self.date_edit = QDateEdit(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("dd.MM.yyyy")
        self.date_edit.setMinimumWidth(px(140))   # the date was cut to «10.10.202»
        # Open-Meteo's hourly forecast runs about two weeks out; offering dates
        # past that just produces the no-data message.
        self.date_edit.setDateRange(QDate.currentDate(),
                                    QDate.currentDate().addDays(14))
        date_btn = QPushButton("On date")
        date_btn.clicked.connect(self._lookup_date)
        self._buttons.append(date_btn)
        btn_row.addWidget(QLabel("Date:"))
        btn_row.addWidget(self.date_edit)
        btn_row.addWidget(date_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        self.status = QLabel("Idle.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color:{MUTED};")
        root.addWidget(self.status)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Date", "When", "°C", "Humidity", "Wind", "Conditions"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setAlternatingRowColors(True)
        hh = self.table.horizontalHeader()
        for _c in range(5):
            hh.setSectionResizeMode(_c, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(5, QHeaderView.Stretch)   # conditions take the slack
        root.addWidget(self.table, 1)

        self.advice = QLabel("")
        self.advice.setWordWrap(True)
        self.advice.setTextFormat(Qt.PlainText)   # never interpret model output as markup
        root.addWidget(self.advice)

    # ----- actions -----------------------------------------------------------
    def _busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def _lookup_date(self):
        qd = self.date_edit.date()
        self._lookup(date=_dt.date(qd.year(), qd.month(), qd.day()))

    def _lookup(self, hours: int = 24, date=None):
        city = self.city_in.text().strip()
        if not city:
            self.status.setText("Type a city first.")
            return
        if self._busy():
            return
        for b in self._buttons:
            b.setEnabled(False)
        window = ("current conditions" if date is None and not hours
                  else f"{hours}h" if date is None else date.strftime("%d.%m.%Y"))
        self.status.setText(f"Looking up {city} — {window}…")
        ctx = getattr(self.host, "ctx", None)
        lang = gui_i18n.LANG
        # hours=0 ("Now") has no separate snapshot path here: the nearest
        # period bucket IS the current conditions, and asking for a 6h window
        # keeps one code path instead of two renderers.
        self.worker = WeatherWorker(ctx, city, lang,
                                    hours=(hours or 6), date=date)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_done(self, loc: dict, buckets: list, advice_map: dict, error: str):
        import weather as _w
        self.table.setRowCount(0)
        self.advice.setText("")
        if error or not buckets:
            self.status.setText(error or "No forecast data.")
            return
        lang = gui_i18n.LANG
        self.status.setText(f"📍 {loc.get('name', '')}"
                            + (f", {loc['country']}" if loc.get("country") else ""))

        shown_date = None      # the date already printed, so it is not repeated
        self.table.setRowCount(len(buckets))
        for r, b in enumerate(buckets):
            hum = b.get("humidity")
            date_txt = ""
            if b["date"] != shown_date:
                date_txt = b["date"].strftime("%d.%m")
                shown_date = b["date"]
            cells = [date_txt,
                     _w._period_label(b["period"], lang),
                     f'{b["temp_c"]:+.0f}',
                     f'{hum:.0f}%' if hum is not None else "—",
                     f'{b["wind_kmh"]:.0f} ' + ("км/ч" if lang == "ru" else "km/h"),
                     f'{_w._code_icon(b["code"])}  {_w._code_desc(b["code"], lang)}']
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if 2 <= c <= 4:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, c, item)

        # Grouped through weather.group_advice, not regrouped here: identical
        # advice for consecutive periods is stated once, the same way the chat
        # reply states it.
        lines = ["👕 " + ("Что надеть" if lang == "ru" else "What to wear")]
        for advice, whens in _w.group_advice(buckets, advice_map, lang):
            span = ", ".join([whens[0]] + [w[:1].lower() + w[1:] for w in whens[1:]])
            lines.append(f"{span} — {advice}")
        self.advice.setText("\n".join(lines))

    def _on_failed(self, msg: str):
        self.status.setText(f"Weather lookup failed: {msg}")

    def _on_finished(self):
        for b in self._buttons:
            b.setEnabled(True)
        self.worker = None
