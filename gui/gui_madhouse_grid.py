"""MadhouseGrid: the room itself -- avatars on a grid, speech bubbles, wandering.

Split out of gui_madhouse_tab.py. This is a self-contained QWidget renderer:
its AST closure reaches only Qt classes, the shared theme colours and the ui
scale helper. Nothing outside it reads its internals -- MadhouseTab constructs
one and calls set_characters / show_speech / clear_all / show_note.

Re-exported from gui_madhouse_tab so both `gui_madhouse_tab.MadhouseGrid` and
gui.py's existing import keep resolving.
"""
import random

from PyQt5.QtCore import QPoint, QRect, QTimer, Qt
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPolygon
from PyQt5.QtWidgets import QSizePolicy, QWidget

from ui_scale import px
from gui_common import BORDER, MUTED, PANEL, PANEL2, TEXT


class MadhouseGrid(QWidget):
    """Animated character minimap for the Madhouse room.

    Characters wander a tile grid between dialogue lines and perform idle
    actions (eat, swap items, fight). The speaking character gets a speech
    bubble with their truncated line. Click any character to cycle their
    avatar emoji.
    """

    COLS = 7
    ROWS = 4
    ANIM_MS   = 1000   # autonomous action tick (ms)
    BUBBLE_MS = 4500   # how long a speech bubble stays visible

    _AVATARS = [
        "🧙", "🧛", "🤖", "👾", "🐱", "🦊", "🐸", "🧟",
        "👻", "🦸", "🧜", "🐺", "🦁", "🐯", "🦄", "🐲",
        "🧝", "🧞", "🧚", "🤡",
    ]
    _FOOD  = ["🍕", "🍔", "☕", "🍣", "🍪", "🥐", "🍉", "🍎", "🌮", "🍜"]
    _ITEMS = ["⚔", "📜", "💎", "🔑", "🎩", "📦", "🧪", "🏆"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._chars: dict = {}    # cid -> state dict
        self._grid:  dict = {}    # (col, row) -> cid
        self._notes: list = []    # rolling system notes (last 3 shown)
        self._anim = QTimer(self)
        self._anim.setInterval(self.ANIM_MS)
        self._anim.timeout.connect(self._tick)
        self.setMinimumHeight(px(160))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click a character to change their avatar")

    # ── public API ────────────────────────────────────────────────────────

    def set_characters(self, characters: list):
        old = {cid: dict(s) for cid, s in self._chars.items()}
        self._chars.clear()
        self._grid.clear()
        occupied: set = set()
        for i, c in enumerate(characters or ()):
            # Defensive: a cast entry without id/name would raise inside a Qt slot.
            # _load_characters normalises both today, so this guards future callers.
            if not isinstance(c, dict):
                continue
            cid = c.get("id") or f"char{i + 1}"
            prev = old.get(cid, {})
            col, row = prev.get("col"), prev.get("row")
            if col is None or (col, row) in occupied:
                col, row = self._empty_cell(occupied)
            if col is None:
                col, row = i % self.COLS, min(i // self.COLS, self.ROWS - 1)
            occupied.add((col, row))
            self._chars[cid] = {
                "avatar":        c.get("_avatar") or self._AVATARS[i % len(self._AVATARS)],
                "name":          c.get("name") or cid,
                "col":           col,
                "row":           row,
                "bubble":        prev.get("bubble"),
                "bubble_ticks":  prev.get("bubble_ticks", 0),
                "action":        "idle",
                "action_data":   None,
                "action_ticks":  0,
            }
            self._grid[(col, row)] = cid
        if self._chars and not self._anim.isActive():
            self._anim.start()
        elif not self._chars:
            self._anim.stop()
        self.update()

    def show_speech(self, cid: str, text: str):
        if cid not in self._chars:
            return
        short = (text[:58] + "…") if len(text) > 60 else text
        s = self._chars[cid]
        s["bubble"] = short
        s["bubble_ticks"] = max(4, round(self.BUBBLE_MS / self.ANIM_MS))
        self.update()

    def clear_all(self):
        for s in self._chars.values():
            s["bubble"] = None
            s["bubble_ticks"] = 0
            s["action"] = "idle"
            s["action_data"] = None
            s["action_ticks"] = 0
        self.update()

    def show_note(self, text: str):
        self._notes = (self._notes + [text])[-3:]
        self.update()

    # ── internal ─────────────────────────────────────────────────────────

    def _cell_dims(self) -> tuple:
        """Return (cell_width, cell_height) filling the widget area."""
        w = max(self.width(), 1)
        h = max(self.height() - px(20), 1)   # reserve bottom strip for notes
        return max(28, w // self.COLS), max(28, h // self.ROWS)

    def _empty_cell(self, extra: set = None):
        occ = set(self._grid) | (extra or set())
        candidates = [(c, r) for c in range(self.COLS) for r in range(self.ROWS)
                      if (c, r) not in occ]
        if not candidates:
            return None, None
        return random.choice(candidates)

    def _tick(self):
        changed = False
        # Age bubbles and actions
        for s in self._chars.values():
            if s["bubble_ticks"] > 0:
                s["bubble_ticks"] -= 1
                if s["bubble_ticks"] <= 0:
                    s["bubble"] = None
                changed = True
            if s["action_ticks"] > 0:
                s["action_ticks"] -= 1
                if s["action_ticks"] <= 0:
                    s["action"] = "idle"
                    s["action_data"] = None
                changed = True
        # Random autonomous action (skip 25% of ticks for variety)
        if self._chars and random.random() > 0.25:
            cids = list(self._chars.keys())
            action = random.choices(
                ["walk", "eat", "fight", "swap"],
                weights=[50, 20, 15, 15],
            )[0]
            if action == "walk":
                self._do_walk(random.choice(cids))
            elif action == "eat":
                s = self._chars[random.choice(cids)]
                s["action"] = "eat"
                s["action_data"] = random.choice(self._FOOD)
                s["action_ticks"] = 2
            elif action == "fight":
                pair = self._adjacent_pair(cids)
                if pair:
                    a, b = pair
                    for cid in (a, b):
                        self._chars[cid]["action"] = "fight"
                        self._chars[cid]["action_data"] = None
                        self._chars[cid]["action_ticks"] = 3
            elif action == "swap":
                pair = self._adjacent_pair(cids)
                if pair:
                    a, b = pair
                    item = random.choice(self._ITEMS)
                    for cid in (a, b):
                        self._chars[cid]["action"] = "swap"
                        self._chars[cid]["action_data"] = item
                        self._chars[cid]["action_ticks"] = 2
            changed = True
        if changed:
            self.update()

    def _do_walk(self, cid: str):
        s = self._chars[cid]
        col, row = s["col"], s["row"]
        moves = [
            (col + dc, row + dr)
            for dc, dr in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= col + dc < self.COLS
            and 0 <= row + dr < self.ROWS
            and (col + dc, row + dr) not in self._grid
        ]
        if not moves:
            return
        del self._grid[(col, row)]
        nc, nr = random.choice(moves)
        s["col"], s["row"] = nc, nr
        s["action"] = "walk"
        s["action_ticks"] = 1
        self._grid[(nc, nr)] = cid

    def _adjacent_pair(self, cids: list):
        """Return a random (a, b) pair that are neighbours (Manhattan dist == 1), or None."""
        pairs = []
        for i, a in enumerate(cids):
            sa = self._chars[a]
            for b in cids[i + 1:]:
                sb = self._chars[b]
                if abs(sa["col"] - sb["col"]) + abs(sa["row"] - sb["row"]) == 1:
                    pairs.append((a, b))
        return random.choice(pairs) if pairs else None

    # ── painting ──────────────────────────────────────────────────────────

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cw, ch = self._cell_dims()
        self._draw_grid(p, cw, ch)
        self._draw_characters(p, cw, ch)
        self._draw_notes(p, cw, ch)
        p.end()

    def _draw_grid(self, p: QPainter, cw: int, ch: int):
        p.fillRect(self.rect(), QColor(PANEL))
        pen = QPen(QColor(BORDER))
        pen.setWidth(1)
        p.setPen(pen)
        gw, gh = self.COLS * cw, self.ROWS * ch
        for c in range(self.COLS + 1):
            p.drawLine(c * cw, 0, c * cw, gh)
        for r in range(self.ROWS + 1):
            p.drawLine(0, r * ch, gw, r * ch)

    def _draw_characters(self, p: QPainter, cw: int, ch: int):
        cs = min(cw, ch)   # font sizing still based on the smaller dimension
        ef = QFont("Segoe UI Emoji", max(9, cs // 2 - 2))
        nf = QFont("Segoe UI", max(6, cs // 7))
        nf.setBold(False)
        sf = QFont("Segoe UI Emoji", max(7, cs // 3))
        bubble_font = QFont("Segoe UI", max(7, cs // 8))

        _action_bg = {
            "fight": QColor(200, 60, 60, 90),
            "eat":   QColor(60, 180, 80, 80),
            "swap":  QColor(80, 130, 220, 80),
            "walk":  QColor(180, 160, 60, 60),
        }

        for cid, s in self._chars.items():
            col, row = s["col"], s["row"]
            x, y = col * cw, row * ch
            action = s["action"]

            # Cell tint
            if action in _action_bg:
                p.fillRect(x + 1, y + 1, cw - 2, ch - 2, _action_bg[action])

            # Avatar emoji (centred, upper ~80% of cell height)
            np_h = max(10, ch // 5)
            avatar_h = ch - np_h
            p.setFont(ef)
            p.setPen(QColor(TEXT))
            p.drawText(QRect(x, y, cw, avatar_h), Qt.AlignCenter, s["avatar"])

            # Action overlay (top-right quadrant of cell)
            overlay = None
            if action == "eat":
                overlay = s.get("action_data") or "🍔"
            elif action == "fight":
                overlay = "⚔"
            elif action == "swap":
                overlay = s.get("action_data") or "📦"
            if overlay:
                p.setFont(sf)
                p.drawText(QRect(x + cw // 2, y, cw // 2, ch // 2),
                           Qt.AlignCenter, overlay)

            # Name plate (bottom strip)
            p.setFont(nf)
            p.setPen(QColor(MUTED))
            p.drawText(QRect(x, y + ch - np_h, cw, np_h),
                       Qt.AlignCenter, s["name"][:12])

            # Speech bubble
            if s.get("bubble"):
                self._draw_bubble(p, x + cw // 2, y, s["bubble"], bubble_font)

    def _draw_bubble(self, p: QPainter, tip_x: int, char_top: int,
                     text: str, font: QFont):
        fm = QFontMetrics(font)
        words = text.split()
        lines: list = []
        cur = ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > 26:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            lines.append(cur)
        lines = lines[:3]

        lh = fm.height() + 2
        bw = max(fm.horizontalAdvance(l) for l in lines) + 14
        bh = len(lines) * lh + 10

        bx = max(2, min(tip_x - bw // 2, self.width() - bw - 2))
        by = max(2, char_top - bh - 8)

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(PANEL2))
        p.drawRoundedRect(bx, by, bw, bh, 6, 6)

        pts = QPolygon([
            QPoint(tip_x - 4, by + bh),
            QPoint(tip_x + 4, by + bh),
            QPoint(tip_x,     by + bh + 6),
        ])
        p.drawPolygon(pts)

        p.setPen(QColor(TEXT))
        p.setFont(font)
        for i, line in enumerate(lines):
            p.drawText(bx + 7, by + 5 + i * lh + fm.ascent(), line)

    def _draw_notes(self, p: QPainter, cw: int, ch: int):
        if not self._notes:
            return
        nf = QFont("Segoe UI", max(7, min(cw, ch) // 8))
        fm = QFontMetrics(nf)
        p.setFont(nf)
        p.setPen(QColor(MUTED))
        y = self.ROWS * ch + 4
        for note in self._notes[-2:]:
            p.drawText(4, y + fm.ascent(), f"— {note}")
            y += fm.height() + 2

    def mousePressEvent(self, ev):
        """Click a character to cycle their avatar emoji."""
        cw, ch = self._cell_dims()
        col = ev.x() // cw
        row = ev.y() // ch
        cid = self._grid.get((col, row))
        if cid and cid in self._chars:
            s = self._chars[cid]
            try:
                idx = self._AVATARS.index(s["avatar"])
                s["avatar"] = self._AVATARS[(idx + 1) % len(self._AVATARS)]
            except ValueError:
                s["avatar"] = self._AVATARS[0]
            self.update()
        super().mousePressEvent(ev)
