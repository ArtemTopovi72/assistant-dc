"""Feedback tab: what people thought of the bot's answers (👍/👎 from Telegram).

Read-only view over bot/tg_feedback: totals, a daily bar chart (up green, down red)
with the satisfaction line over it, a per-kind breakdown and the latest disliked
answers. Charts are plain QPainter (the Qt build has no QtChart). The store is
passed in so the tab can be shown over a temp file.
"""
import time

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton,
                             QTextEdit, QVBoxLayout, QWidget)

from ui_scale import px
from gui_common import (ACCENT2, BORDER, MUTED, PANEL, REC, TEXT, _section)

_RANGES = (("7 days", 7), ("30 days", 30), ("90 days", 90))


class _DailyChart(QWidget):
    """Stacked bars per day (👍 below, 👎 above) and the % satisfied line."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._days = []
        self.setMinimumHeight(px(220))

    def set_days(self, days):
        self._days = list(days)
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(PANEL))
        left, right, top, bottom = px(34), px(40), px(12), px(26)
        area = QRectF(left, top, w - left - right, h - top - bottom)
        days = self._days
        votes = [d["up"] + d["down"] for d in days]
        if not days or not any(votes):
            p.setPen(QColor(MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "No votes yet")
            return
        peak = max(max(votes), 1)
        p.setPen(QPen(QColor(BORDER), 1))
        for i in range(5):
            y = area.top() + area.height() * i / 4
            p.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
        small = QFont(p.font())
        small.setPointSize(max(7, small.pointSize() - 2))
        p.setFont(small)
        p.setPen(QColor(MUTED))
        p.drawText(QRectF(0, area.top() - 6, left - 4, 14), Qt.AlignRight, str(peak))
        p.drawText(QRectF(0, area.bottom() - 8, left - 4, 14), Qt.AlignRight, "0")
        p.drawText(QRectF(area.right() + 4, area.top() - 6, right, 14), Qt.AlignLeft, "100%")
        p.drawText(QRectF(area.right() + 4, area.bottom() - 8, right, 14), Qt.AlignLeft, "0%")

        n = len(days)
        slot = area.width() / n
        bar = max(2.0, slot * 0.7)
        line = []
        for i, d in enumerate(days):
            x = area.left() + slot * i + (slot - bar) / 2
            up_h = area.height() * d["up"] / peak
            dn_h = area.height() * d["down"] / peak
            p.fillRect(QRectF(x, area.bottom() - up_h, bar, up_h), QColor(ACCENT2))
            p.fillRect(QRectF(x, area.bottom() - up_h - dn_h, bar, dn_h), QColor(REC))
            if d["up"] + d["down"]:
                line.append(QPointF(x + bar / 2,
                                    area.bottom() - area.height() * d["up"] / (d["up"] + d["down"])))
        p.setPen(QPen(QColor(TEXT), 2))
        for a, b in zip(line, line[1:]):
            p.drawLine(a, b)
        for pt in line:
            p.drawEllipse(pt, 2.5, 2.5)
        p.setPen(QColor(MUTED))
        step = max(1, n // 6)
        for i in range(0, n, step):
            x = area.left() + slot * i
            p.drawText(QRectF(x - 10, area.bottom() + 4, slot * step + 20, 16),
                       Qt.AlignLeft, days[i]["day"][5:])


def _card_label(title: str):
    box = QWidget()
    box.setStyleSheet(f"background:{PANEL};border-radius:{px(8)}px;")
    lay = QVBoxLayout(box)
    lay.setContentsMargins(px(14), px(10), px(14), px(10))
    lay.setSpacing(2)
    cap = QLabel(title)
    cap.setStyleSheet(f"color:{MUTED};font-size:{px(11)}px;background:transparent;")
    val = QLabel("–")
    val.setStyleSheet(f"color:{TEXT};font-size:{px(24)}px;font-weight:600;background:transparent;")
    lay.addWidget(cap)
    lay.addWidget(val)
    return box, val


class FeedbackTab(QWidget):
    def __init__(self, parent=None, store=None):
        super().__init__(parent)
        self._store_fn = store
        self._build_ui()

    def _store(self):
        if self._store_fn is not None:
            return self._store_fn()
        import tg_feedback
        return tg_feedback.store()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(px(10), px(10), px(10), px(10))
        root.setSpacing(px(10))
        hdr = QHBoxLayout()
        hdr.addWidget(_section("Feedback"))
        hdr.addStretch(1)
        self._range = QComboBox()
        for label, days in _RANGES:
            self._range.addItem(label, days)
        self._range.setCurrentIndex(1)
        self._range.currentIndexChanged.connect(self.refresh)
        hdr.addWidget(self._range)
        ref = QPushButton("↻ Refresh")
        ref.setObjectName("ghost")
        ref.clicked.connect(self.refresh)
        hdr.addWidget(ref)
        root.addLayout(hdr)

        cards = QHBoxLayout()
        cards.setSpacing(px(10))
        self._vals = {}
        for key, title in (("replies", "Answers"), ("up", "👍"), ("down", "👎"), ("rate", "Satisfied")):
            box, val = _card_label(title)
            self._vals[key] = val
            cards.addWidget(box, 1)
        root.addLayout(cards)

        self._chart = _DailyChart()
        root.addWidget(self._chart)

        root.addWidget(_section("By kind"))
        self._kinds = QTextEdit()
        self._kinds.setReadOnly(True)
        self._kinds.setObjectName("terminal")
        self._kinds.setMaximumHeight(px(110))
        root.addWidget(self._kinds)

        root.addWidget(_section("Latest 👎"))
        self._bad = QTextEdit()
        self._bad.setReadOnly(True)
        self._bad.setObjectName("terminal")
        self._bad.setPlaceholderText("Answers somebody disliked show up here.")
        root.addWidget(self._bad, 1)
        self.refresh()

    def refresh(self, *_):
        days = self._range.currentData() or 30
        try:
            st = self._store()
            s, daily, kinds, bad = st.summary(days), st.daily(days), st.by_kind(days), st.recent_down(30)
        except Exception as e:
            self._bad.setPlainText(f"Feedback store unavailable: {e}")
            return
        self._vals["replies"].setText(str(s["replies"]))
        self._vals["up"].setText(str(s["up"]))
        self._vals["down"].setText(str(s["down"]))
        self._vals["rate"].setText("–" if s["rate"] is None else f"{round(s['rate'] * 100)}%")
        self._chart.set_days(daily)
        self._kinds.setPlainText("\n".join(
            f"{k['kind']:<10} 👍 {k['up']:<4} 👎 {k['down']:<4} answers {k['replies']}" for k in kinds))
        self._bad.setPlainText("\n\n".join(
            f"{time.strftime('%m-%d %H:%M', time.localtime(b['ts']))}  [{b['kind']}]\n"
            f"Q: {b['user']}\nA: {b['reply']}" for b in bad))
