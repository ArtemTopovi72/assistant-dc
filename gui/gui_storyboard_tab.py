"""Storyboard tab: plan a picture as BOXES, then draw and edit it through them.

Third tab out of gui.py — 939 lines with its four background workers and the
box canvas.

The whole point of this tab is the layout: a picture is planned as labelled
rectangles, and an edit rewords a BOX rather than re-rendering the frame. That
is what keeps an edit from becoming a re-roll — the seed is reused, and
"make the man blond" changes one box instead of blending a new face over him.
_BoxCanvas is the direct-manipulation view of those rectangles.

The four workers exist because each stage is a separate long call that must
stay off the UI thread: IdeogramWorker renders, DrawAgentWorker plans the
layout, LayoutEditWorker turns an instruction into box operations, and
PictureEditWorker runs a pixel-level edit when no box applies.

Each takes a _ScopedCtx (gui_common) so the main chat's Stop button cannot
cancel this tab's work, and vice versa.
"""
import copy
import json
import os
import random
import threading

from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QLabel,
                             QLineEdit, QListWidget, QPushButton, QSpinBox,
                             QSplitter, QVBoxLayout, QWidget)

from ui_scale import pt, px
from gui_common import _FlowWidget, _ScopedCtx, _flow, _section

import logging
logger = logging.getLogger("assistant.gui")   # same channel as gui.py

# The four background workers and the box canvas live in their own modules.
# Re-exported BY VALUE here on purpose: StoryboardTab._plan/._draw resolve these
# as globals of THIS module, so the suites that patch gui_storyboard_tab.
# IdeogramWorker (and gui.IdeogramWorker) to install a spy keep working.
from gui_storyboard_workers import (  # noqa: E402,F401
    IdeogramWorker, DrawAgentWorker, LayoutEditWorker, PictureEditWorker,
)
from gui_storyboard_canvas import _BoxCanvas  # noqa: E402,F401


class StoryboardTab(QWidget):
    """Ideogram 4 drawing mode with visible, editable boxes.

    Ideogram 4 composes from a structured caption in which every element may carry
    a bounding box. That is a layout the user can SEE, so here it is one: plan a
    scene with the LLM, drag the boxes where they belong, reword an element or its
    lettering, then draw. The finished picture becomes the backdrop, so the next
    pass is a repaint of a composition you can still adjust.
    """

    _SIZES = (("1:1  1024²", 1024, 1024), ("3:2  1216×832", 1216, 832),
              ("2:3  832×1216", 832, 1216), ("16:9  1344×768", 1344, 768),
              ("9:16  768×1344", 768, 1344))

    def __init__(self, host):
        super().__init__()
        self.host = host
        self.worker = None
        self.plan_worker = None
        self.agent_worker = None
        self.edit_worker = None
        self.picedit_worker = None
        self._last_image = None            # the picture a contained edit works on
        self._agent_cancel = None          # the agent's own Stop flag (see _ScopedCtx)
        # ONE seed, reused across edits and repaints. Ideogram box mode is text-to-image:
        # every draw is a fresh generation, so a new seed reshuffles the WHOLE picture and
        # an "edit" becomes an unrelated image. Holding the seed keeps the composition and
        # lets a box change alter only its own part of the frame. Repaint / 🎲 pick a new one.
        self._seed = 0
        self.layout_data = {"high_level_description": "", "aesthetics": "", "lighting": "",
                            "photo": "", "medium": "", "background": "", "elements": []}
        self._agent_serial = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(px(6), px(6), px(6), px(6))
        outer.setSpacing(px(6))
        self.vsplit = QSplitter(Qt.Vertical)
        outer.addWidget(self.vsplit, 1)

        top = QWidget(); root = QVBoxLayout(top)
        root.setContentsMargins(0, 0, 0, px(2)); root.setSpacing(px(6))
        root.addWidget(_section("Scene"))

        self.prompt_in = QLineEdit()
        self.prompt_in.setPlaceholderText("What should the picture show?")
        self.prompt_in.returnPressed.connect(self._plan)
        root.addWidget(self.prompt_in)

        # The agent line: one field for "now change this" — the layout it edits is the
        # one on the canvas, so a conversation and the boxes stay the same object.
        self.tell_in = QLineEdit()
        self.tell_in.setPlaceholderText("Tell the agent what to change: “drop the dog, "
                                        "put a sign top-left saying OPEN”")
        self.tell_in.returnPressed.connect(self._agent_edit)
        root.addWidget(self.tell_in)

        arow = _flow(spacing=px(6))
        self.edit_btn = QPushButton("✏  Edit boxes"); self.edit_btn.setObjectName("ghost")
        self.edit_btn.setToolTip("Change the BOXES from that instruction (then Draw to "
                                 "regenerate). Good for moving/adding/removing elements.")
        self.edit_btn.clicked.connect(self._agent_edit)
        self.picedit_btn = QPushButton("🩹  Change on picture"); self.picedit_btn.setObjectName("ghost")
        self.picedit_btn.setToolTip(
            "Edit ONLY the named part of the CURRENT picture and keep everything else "
            "exactly as it is — the right tool for “swap the grandpa for a grandma” "
            "without touching the tractor. Needs a drawn picture first.")
        self.picedit_btn.clicked.connect(self._edit_on_picture)
        self.agent_btn = QPushButton("🤖  Draw with agent")
        self.agent_btn.setToolTip(
            "Plan or edit the boxes, draw, LOOK at the result, and rearrange the boxes\n"
            "if the picture does not match the layout — repeating up to the round limit.")
        self.agent_btn.clicked.connect(self._agent_run)
        self.rounds_spin = QSpinBox()
        self.rounds_spin.setRange(0, 5); self.rounds_spin.setValue(2)
        self.rounds_spin.setPrefix("fix ×")
        self.rounds_spin.setToolTip("How many repair passes the agent may spend after "
                                    "the first draw. 0 = draw once, judge, stop.")
        self.autofix_box = QCheckBox("Fix bad arrangements")
        self.autofix_box.setChecked(True)
        self.autofix_box.setToolTip(
            "Before drawing, repair arrangements known to break the render:\n"
            "boxes that overlap so much they merge, boxes too small to survive,\n"
            "a box that fills the frame and leaves the background nothing to be.")
        self.stop_btn = QPushButton("⏹  Stop"); self.stop_btn.setObjectName("ghost")
        self.stop_btn.clicked.connect(self._agent_stop)
        self.stop_btn.setEnabled(False)
        for wgt in (self.edit_btn, self.picedit_btn, self.agent_btn, self.rounds_spin,
                    self.autofix_box, self.stop_btn):
            arow.addWidget(wgt)
        root.addWidget(_FlowWidget(arow))

        row = _flow(spacing=px(6))
        self.plan_btn = QPushButton("🧠  Plan layout")
        self.plan_btn.setToolTip("Let the model break the scene into elements and place them.")
        self.plan_btn.clicked.connect(self._plan)
        self.draw_btn = QPushButton("🎨  Draw")
        self.draw_btn.clicked.connect(self._draw)
        self.repaint_btn = QPushButton("🔁  Repaint"); self.repaint_btn.setObjectName("ghost")
        self.repaint_btn.setToolTip("Draw the same layout again with a new seed.")
        self.repaint_btn.clicked.connect(lambda: self._draw(new_seed=True))
        self.size_combo = QComboBox()
        for label, w, h in self._SIZES:
            self.size_combo.addItem(label, (w, h))
        self.size_combo.currentIndexChanged.connect(self._on_size)
        self.seed_btn = QPushButton("🎲  New seed"); self.seed_btn.setObjectName("ghost")
        self.seed_btn.setToolTip("Pick a fresh seed. The seed is held across edits so the "
                                 "composition stays put and only your box changes move the "
                                 "picture — press this when you want a different composition.")
        self.seed_btn.clicked.connect(self._new_seed)
        self.seed_lbl = QLabel("seed —"); self.seed_lbl.setObjectName("chip")
        for wgt in (self.plan_btn, self.draw_btn, self.repaint_btn, self.size_combo,
                    self.seed_btn, self.seed_lbl):
            row.addWidget(wgt)
        root.addWidget(_FlowWidget(row))

        row2 = _flow(spacing=px(6))
        add_btn = QPushButton("➕  Box"); add_btn.setObjectName("ghost")
        add_btn.clicked.connect(self._add_box)
        del_btn = QPushButton("✕  Delete"); del_btn.setObjectName("ghost")
        del_btn.clicked.connect(self._delete_box)
        up_btn = QPushButton("⬆  Front"); up_btn.setObjectName("ghost")
        up_btn.setToolTip("Move this element to the front of the list (drawn on top).")
        up_btn.clicked.connect(self._to_front)
        self.boxes_box = QCheckBox("Show boxes"); self.boxes_box.setChecked(True)
        self.boxes_box.toggled.connect(self._on_show_boxes)
        for wgt in (add_btn, del_btn, up_btn, self.boxes_box):
            row2.addWidget(wgt)
        root.addWidget(_FlowWidget(row2))

        # The switch the agent's own drawing goes through.
        row3 = _flow(spacing=px(6))
        self.agent_box = QCheckBox("🤖  Agent draws here")
        self.agent_box.setToolTip(
            "Switch the assistant's image generation to Ideogram 4 and pick up every\n"
            "picture it draws as an editable layout — so when it needs a repaint you\n"
            "drag the boxes, reword an element, and draw again.")
        self.agent_box.toggled.connect(self._on_agent_mode)
        pick_btn = QPushButton("⤵  Agent's last layout"); pick_btn.setObjectName("ghost")
        pick_btn.setToolTip("Load the composition the assistant drew most recently.")
        pick_btn.clicked.connect(self._pick_agent_layout)
        load_btn = QPushButton("📂  Load…"); load_btn.setObjectName("ghost")
        load_btn.clicked.connect(self._load_layout)
        save_btn = QPushButton("💾  Save…"); save_btn.setObjectName("ghost")
        save_btn.clicked.connect(self._save_layout)
        for wgt in (self.agent_box, pick_btn, load_btn, save_btn):
            row3.addWidget(wgt)
        root.addWidget(_FlowWidget(row3))
        self.vsplit.addWidget(top)

        # ---- canvas (gets the slack) ----------------------------------------
        mid = QWidget(); midl = QVBoxLayout(mid)
        midl.setContentsMargins(0, 0, 0, 0); midl.setSpacing(px(4))
        midl.addWidget(_section("Frame"))
        self.canvas = _BoxCanvas()
        self.canvas.elements = self.layout_data["elements"]
        self.canvas.selected.connect(self._on_canvas_select)
        self.canvas.changed.connect(self._on_canvas_changed)
        midl.addWidget(self.canvas, 1)
        self.vsplit.addWidget(mid)

        # ---- element editor --------------------------------------------------
        bot = QWidget(); botl = QVBoxLayout(bot)
        botl.setContentsMargins(0, 0, 0, 0); botl.setSpacing(px(4))
        botl.addWidget(_section("Element"))
        self.el_list = QListWidget()
        self.el_list.setMinimumHeight(px(52))
        self.el_list.currentRowChanged.connect(self._on_row)
        botl.addWidget(self.el_list, 1)
        self.desc_in = QLineEdit()
        self.desc_in.setPlaceholderText("What is in this box (described on its own)")
        self.desc_in.textEdited.connect(self._on_desc)
        self.text_in = QLineEdit()
        self.text_in.setPlaceholderText("Lettering to render inside the box (optional)")
        self.text_in.textEdited.connect(self._on_text)
        botl.addWidget(self.desc_in)
        botl.addWidget(self.text_in)
        self.bg_in = QLineEdit()
        self.bg_in.setPlaceholderText("Background / setting behind everything")
        self.bg_in.textEdited.connect(
            lambda s: self.layout_data.__setitem__("background", s))
        botl.addWidget(self.bg_in)
        # What the agent did and why — the arrangement reasoning, not just the picture.
        self.agent_log = QListWidget()
        self.agent_log.setWordWrap(True)
        self.agent_log.setMinimumHeight(px(46))
        self.agent_log.setToolTip("What the agent changed about the boxes, and what it "
                                  "saw wrong in the render.")
        botl.addWidget(self.agent_log, 1)
        self.status = QLabel("Type a scene and press Plan layout."); self.status.setObjectName("chip")
        self.status.setWordWrap(True)
        botl.addWidget(self.status)
        self.vsplit.addWidget(bot)
        self.vsplit.setSizes([px(150), px(420), px(190)])
        self.vsplit.setCollapsible(1, False)

        self._on_size()
        self._refresh_list()
        # Poll for pictures the agent drew (the render happens on the agent's own
        # thread, in another module — a 1.5 s poll of a serial counter is simpler and
        # safer than wiring a cross-thread signal through image.py).
        self._poll = QTimer(self)
        self._poll.setInterval(1500)
        self._poll.timeout.connect(self._poll_agent)
        self._poll.start()

    # ---- helpers ---------------------------------------------------------
    @property
    def ctx(self):
        return getattr(self.host, "ctx", None)

    def _set_layout(self, layout, *, note=""):
        import ideogram
        self.layout_data = ideogram.normalize_layout(layout, self.prompt_in.text())
        self.canvas.elements = self.layout_data["elements"]
        self.bg_in.setText(self.layout_data.get("background", ""))
        self._refresh_list()
        self.canvas.update()
        if note:
            self.status.setText(note)

    def _refresh_list(self):
        self.el_list.blockSignals(True)
        self.el_list.clear()
        for i, el in enumerate(self.layout_data["elements"]):
            tag = f'  ✎ “{el["text"]}”' if el.get("text") else ""
            self.el_list.addItem(
                f'{i + 1}. {el.get("desc", "")[:60]}{tag}   '
                f'[{el["x"]:.2f} {el["y"]:.2f} {el["w"]:.2f}×{el["h"]:.2f}]')
        self.el_list.blockSignals(False)
        sel = self.canvas.sel
        if 0 <= sel < self.el_list.count():
            self.el_list.setCurrentRow(sel)
        self.draw_btn.setEnabled(bool(self.layout_data["elements"]) or
                                 bool(self.layout_data.get("background")))

    def _current(self):
        i = self.canvas.sel
        els = self.layout_data["elements"]
        return els[i] if 0 <= i < len(els) else None

    # ---- slots -----------------------------------------------------------
    def _on_size(self):
        w, h = self.size_combo.currentData() or (1024, 1024)
        self.canvas.aspect = w / h
        self.canvas.update()

    def _on_show_boxes(self, on):
        self.canvas.show_boxes = bool(on)
        self.canvas.update()

    def _on_canvas_select(self, i):
        el = self._current()
        self.desc_in.setText(el.get("desc", "") if el else "")
        self.text_in.setText(el.get("text", "") if el else "")
        if 0 <= i < self.el_list.count():
            self.el_list.blockSignals(True)
            self.el_list.setCurrentRow(i)
            self.el_list.blockSignals(False)

    def _on_canvas_changed(self):
        self._refresh_list()

    def _on_row(self, i):
        self.canvas.sel = i
        self.canvas.update()
        self._on_canvas_select(i)

    def _on_desc(self, s):
        el = self._current()
        if el is not None:
            el["desc"] = s
            self._refresh_list()
            self.canvas.update()

    def _on_text(self, s):
        el = self._current()
        if el is not None:
            el["text"] = s
            self._refresh_list()
            self.canvas.update()

    def _add_box(self):
        els = self.layout_data["elements"]
        # stagger new boxes so a second one is not hidden exactly under the first
        off = 0.05 * (len(els) % 5)
        els.append({"desc": "new element", "text": "",
                    "x": 0.25 + off, "y": 0.25 + off, "w": 0.3, "h": 0.3})
        self.canvas.sel = len(els) - 1
        self._refresh_list()
        self._on_canvas_select(self.canvas.sel)
        self.canvas.update()

    def _delete_box(self):
        els = self.layout_data["elements"]
        i = self.canvas.sel
        if 0 <= i < len(els):
            els.pop(i)
            self.canvas.sel = min(i, len(els) - 1)
            self._refresh_list()
            self._on_canvas_select(self.canvas.sel)
            self.canvas.update()

    def _to_front(self):
        els = self.layout_data["elements"]
        i = self.canvas.sel
        if 0 <= i < len(els) - 1:
            els.append(els.pop(i))
            self.canvas.sel = len(els) - 1
            self._refresh_list()
            self.canvas.update()

    # ---- planning / drawing ---------------------------------------------
    def _plan(self):
        prompt = self.prompt_in.text().strip()
        if not prompt:
            self.status.setText("Describe the scene first.")
            return
        if self.ctx is None:
            self.status.setText("No model loaded yet — add boxes by hand with ➕ Box.")
            return
        if self.plan_worker is not None and self.plan_worker.isRunning():
            self.status.setText("Already planning…")
            return
        self.status.setText("Laying the scene out…")
        self.plan_btn.setEnabled(False)

        class _PlanWorker(QThread):
            done = pyqtSignal(dict)
            failed = pyqtSignal(str)

            def __init__(self, ctx, prompt):
                super().__init__()
                self.ctx, self.prompt = ctx, prompt

            def run(self):
                try:
                    import ideogram
                    self.done.emit(ideogram.plan_layout(self.ctx, self.prompt))
                except Exception as exc:
                    logger.exception("storyboard planning failed")
                    self.failed.emit(str(exc))

        self.plan_worker = _PlanWorker(self.ctx, prompt)
        self.plan_worker.done.connect(self._on_planned)
        self.plan_worker.failed.connect(self._on_plan_failed)
        self.plan_worker.finished.connect(self._on_plan_finished)
        self.plan_worker.start()

    def _on_plan_finished(self):
        self.plan_worker = None

    def _on_planned(self, layout):
        self.plan_btn.setEnabled(True)
        n = len(layout.get("elements") or [])
        self._set_layout(layout, note=f"{n} element(s) placed — drag them, then Draw.")

    def _on_plan_failed(self, msg):
        self.plan_btn.setEnabled(True)
        self.status.setText(f"Planning failed: {msg}")

    def _draw(self, *_a, new_seed=False):
        if self.worker is not None and self.worker.isRunning():
            self.status.setText("Already drawing…")
            return
        import ideogram
        if not self.layout_data["elements"] and not self.layout_data.get("background"):
            self.status.setText("Nothing to draw — plan a scene or add a box.")
            return
        caption = ideogram.layout_to_caption(self.layout_data)
        w, h = self.size_combo.currentData() or (1024, 1024)
        # Reuse the held seed so redrawing after an edit keeps the composition; only
        # Repaint (or 🎲) asks for a fresh one.
        if new_seed or self._seed <= 0:
            self._seed = random.randint(1, 999_999_999)
        self._update_seed_label()
        self.status.setText(f"Drawing {w}×{h} (seed {self._seed})…")
        self.draw_btn.setEnabled(False); self.repaint_btn.setEnabled(False)
        self.worker = IdeogramWorker(self.ctx, caption, w, h, self._seed)
        self.worker.done.connect(self._on_drawn)
        self.worker.failed.connect(self._on_draw_failed)
        self.worker.progress.connect(self._on_draw_progress)
        self.worker.finished.connect(self._on_draw_worker_finished)
        self.worker.start()

    def _on_draw_progress(self, step, total):
        if total > 0:
            self.status.setText(f"Drawing… step {step}/{total}")

    def _on_edit_progress(self, step, total):
        # Same live counter as drawing, worded for the contained edit. The edit fans
        # out to a couple of quick segmentation jobs before the inpaint sampler, so
        # the counter settles on the long one (28 steps) — the short ones flick past.
        if total > 0:
            self.status.setText(f"Editing the picture… step {step}/{total}")

    def _new_seed(self):
        self._seed = random.randint(1, 999_999_999)
        self._update_seed_label()
        self.status.setText(f"New seed {self._seed} — Draw to see this composition.")

    def _update_seed_label(self):
        self.seed_lbl.setText(f"seed {self._seed}" if self._seed > 0 else "seed —")

    def _on_drawn(self, path):
        self.draw_btn.setEnabled(True); self.repaint_btn.setEnabled(True)
        self.set_backdrop(path)
        self.status.setText(f"Drawn: {os.path.basename(path)} — adjust the boxes and Repaint.")
        panel = getattr(self.host, "images_panel", None)
        if panel is not None:
            try:
                panel.add_image(path)
            except Exception:
                logger.exception("adding the storyboard render to the Images panel failed")

    def _on_draw_worker_finished(self):
        self.worker = None

    def _on_draw_failed(self, msg):
        self.draw_btn.setEnabled(True); self.repaint_btn.setEnabled(True)
        self.status.setText(f"Draw failed: {msg}")

    def set_backdrop(self, path):
        try:
            pm = QPixmap(path)
            if not pm.isNull():
                self.canvas.backdrop = pm
                self._last_image = path      # the file a picture-edit works on
            else:
                self.canvas.backdrop = None
        except Exception:
            logger.exception("loading the storyboard backdrop failed")
            self.canvas.backdrop = None
        self.canvas.update()

    def _edit_on_picture(self):
        """Contained edit of the CURRENT render: change only the named region, keep
        the rest of the photo. This is the identity-preserving path (segment → inpaint
        → composite back) — the box regenerate cannot swap a subject without either
        redrawing the whole scene or leaving the old one barely changed."""
        instruction = self.tell_in.text().strip()
        img = getattr(self, "_last_image", None)
        if not img or not os.path.exists(img):
            self.status.setText("Draw a picture first — this edits the picture you see.")
            return
        if not instruction:
            self.status.setText("Type what to change (e.g. “swap the grandpa for a grandma”).")
            return
        if self.ctx is None:
            self.status.setText("No model loaded.")
            return
        if self.picedit_worker is not None and self.picedit_worker.isRunning():
            return
        self._agent_busy(True)
        self.status.setText("Editing the picture (keeping everything else)…")
        self._note(f"picture edit: {instruction}", kind="edited")
        self.picedit_worker = PictureEditWorker(
            _ScopedCtx(self.ctx, threading.Event()), img, instruction)
        self.picedit_worker.progress.connect(self._on_edit_progress)
        self.picedit_worker.done.connect(self._on_picture_edited)
        self.picedit_worker.failed.connect(self._on_agent_failed)
        self.picedit_worker.finished.connect(self._on_picedit_worker_finished)
        self.picedit_worker.start()

    def _on_picedit_worker_finished(self):
        self.picedit_worker = None

    def _on_picture_edited(self, path):
        self._agent_busy(False)
        if not path:
            self.status.setText("Picture edit failed — the region could not be found or drawn.")
            return
        self.set_backdrop(path)
        self._note(f"edited {os.path.basename(path)} (rest of the picture unchanged)", kind="drawn")
        self.status.setText("Done — only that part changed. Keep editing, or Draw for a fresh scene.")
        self.tell_in.clear()
        panel = getattr(self.host, "images_panel", None)
        if panel is not None:
            try:
                panel.add_image(path)
            except Exception:
                logger.exception("adding the picture edit to the Images panel failed")

    # ---- the drawing agent ------------------------------------------------
    def _note(self, text, *, kind=""):
        icon = {"stage": "▸", "fixed": "🔧", "edited": "✏", "rearranged": "🧩", "preflight": "🧭",
                "judged": "👁", "drawn": "🖼", "refused": "⛔", "failed": "⚠"}.get(kind, "·")
        self.agent_log.addItem(f"{icon}  {text}")
        self.agent_log.scrollToBottom()
        while self.agent_log.count() > 200:          # a long session must not grow forever
            self.agent_log.takeItem(0)

    def _agent_busy(self, busy):
        for b in (self.agent_btn, self.edit_btn, self.picedit_btn, self.draw_btn,
                  self.repaint_btn, self.plan_btn):
            b.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)

    def _agent_edit(self):
        """Apply one instruction to the CURRENT boxes, without drawing."""
        instruction = self.tell_in.text().strip()
        if not instruction:
            self.status.setText("Type what to change first.")
            return
        if self.ctx is None:
            self.status.setText("No model loaded — edit the boxes by hand.")
            return
        if self.edit_worker is not None and self.edit_worker.isRunning():
            return
        self._agent_busy(True)
        self.status.setText("Editing the boxes…")
        self.edit_worker = LayoutEditWorker(self.ctx, self.layout_data, instruction)
        self.edit_worker.done.connect(self._on_edited)
        self.edit_worker.failed.connect(self._on_agent_failed)
        self.edit_worker.finished.connect(self._on_edit_worker_finished)
        self.edit_worker.start()

    def _on_edit_worker_finished(self):
        self.edit_worker = None

    def _on_edited(self, layout, notes):
        self._agent_busy(False)
        self._set_layout(layout)
        for n in (notes or []):
            self._note(n, kind="edited")
        self.status.setText("Boxes updated — Draw, or keep telling the agent what to change.")
        self.tell_in.clear()

    def _agent_run(self):
        """The whole loop: plan/edit → draw → look → rearrange → draw again."""
        if self.agent_worker is not None and self.agent_worker.isRunning():
            self.status.setText("The agent is already working.")
            return
        if self.ctx is None:
            self.status.setText("No model loaded — the agent needs one to plan and judge.")
            return
        instruction = self.tell_in.text().strip()
        have_layout = bool(self.layout_data.get("elements"))
        request = self.prompt_in.text().strip()
        if not have_layout and not request:
            self.status.setText("Describe the scene (or arrange some boxes) first.")
            return
        w, h = self.size_combo.currentData() or (1024, 1024)
        self._agent_cancel = threading.Event()
        self.agent_log.clear()
        self._agent_busy(True)
        self.status.setText("Agent working…")
        self.agent_worker = DrawAgentWorker(
            _ScopedCtx(self.ctx, self._agent_cancel),
            request=request, layout=copy.deepcopy(self.layout_data) if have_layout else None,
            instruction=instruction, width=w, height=h, seed=self._seed,
            rounds=self.rounds_spin.value(), auto_fix=self.autofix_box.isChecked())
        self.agent_worker.event.connect(self._on_agent_event)
        self.agent_worker.progress.connect(self._on_draw_progress)
        self.agent_worker.done.connect(self._on_agent_done)
        self.agent_worker.failed.connect(self._on_agent_failed)
        self.agent_worker.finished.connect(self._on_agent_worker_finished)
        self.agent_worker.start()

    def _agent_stop(self):
        if self._agent_cancel is not None:
            self._agent_cancel.set()
        self.status.setText("Stopping after the current step…")

    def _on_agent_event(self, kind, data):
        data = data or {}
        if kind == "stage":
            self.status.setText(str(data.get("text", "")))
            self._note(str(data.get("text", "")), kind="stage")
        elif kind in ("edited", "fixed", "rearranged"):
            for why in (data.get("problems") or []):
                self._note(why, kind="judged")
            for n in (data.get("notes") or []):
                self._note(n, kind=kind)
            if data.get("layout"):
                self._set_layout(data["layout"])
        elif kind == "drawn":
            self.set_backdrop(data.get("image"))
            self._note(f"drew {os.path.basename(data.get('image') or '')}", kind="drawn")
            panel = getattr(self.host, "images_panel", None)
            if panel is not None and data.get("image"):
                try:
                    panel.add_image(data["image"])
                except Exception:
                    logger.exception("adding the agent's render to the Images panel failed")
        elif kind == "preflight":
            # The plan checked as a sketch, before the render (draw_preview).
            if data.get("source") != "vision":
                self._note("sketch check: no vision answer — drawing the plan as is", kind="preflight")
            else:
                self._note("sketch check: " + ("the plan matches the request"
                                               if data.get("ok") else "fixing the plan"), kind="preflight")
            for p in (data.get("problems") or []):
                self._note(p, kind="preflight")
            if data.get("layout"):
                self._set_layout(data["layout"])
        elif kind == "judged":
            src = data.get("source", "")
            if src == "unavailable":
                self._note("could not look at the result (no vision answer) — keeping it",
                           kind="judged")
            else:
                verdict = "looks right" if data.get("ok") else "not right yet"
                self._note(f"{verdict} (score {data.get('score', 0)}/10)", kind="judged")
            for p in (data.get("problems") or []):
                self._note(p, kind="judged")
        elif kind in ("refused", "failed"):
            self._note(str(data.get("text", "")), kind=kind)

    def _on_agent_done(self, result):
        self._agent_busy(False)
        result = result or {}
        if result.get("seed"):
            self._seed = int(result["seed"])       # keep the agent's seed for follow-up edits
            self._update_seed_label()
        if result.get("layout"):
            self._set_layout(result["layout"])
        if result.get("image"):
            self.set_backdrop(result["image"])
        stopped = result.get("stopped", "")
        rounds = len(result.get("history") or [])
        msg = {"ok": "Agent finished — the picture matches the layout.",
               "out of rounds": "Agent stopped at the round limit; the boxes are its "
                                "best arrangement so far.",
               "refused": "Ideogram refused this prompt and drew its safety card.",
               "failed": "The renderer returned nothing.",
               "cancelled": "Stopped."}.get(stopped, f"Agent finished ({stopped}).")
        self.status.setText(f"{msg}  [{rounds} attempt(s)]")
        self.tell_in.clear()

    def _on_agent_worker_finished(self):
        self.agent_worker = None

    def _on_agent_failed(self, msg):
        self._agent_busy(False)
        self._note(msg, kind="failed")
        self.status.setText(f"Agent failed: {msg}")

    # ---- agent hand-off ---------------------------------------------------
    def _on_agent_mode(self, on):
        import config as _cfg
        if on:
            self._prev_engine = getattr(_cfg, "IMAGE_ENGINE", "ideogram4")
            _cfg.IMAGE_ENGINE = "ideogram4"
            self.status.setText("Agent drawing routed through Ideogram 4 — its layouts "
                                "land here automatically.")
        else:
            _cfg.IMAGE_ENGINE = getattr(self, "_prev_engine", "ideogram4")
            self.status.setText(f"Agent drawing back on '{_cfg.IMAGE_ENGINE}'.")

    def _pick_agent_layout(self):
        import ideogram
        if not ideogram.LAST_CAPTION:
            self.status.setText("The assistant has not drawn anything with Ideogram 4 yet.")
            return
        self._agent_serial = ideogram.LAST_SERIAL
        if ideogram.LAST_PROMPT and not self.prompt_in.text().strip():
            self.prompt_in.setText(ideogram.LAST_PROMPT)
        self._set_layout(ideogram.caption_to_layout(ideogram.LAST_CAPTION),
                         note="Loaded the assistant's last composition.")
        if ideogram.LAST_IMAGE:
            self.set_backdrop(ideogram.LAST_IMAGE)

    def _poll_agent(self):
        """Auto-pick-up while the agent-draws-here switch is on."""
        if not self.agent_box.isChecked():
            return
        try:
            import ideogram
        except Exception:
            return
        if ideogram.LAST_SERIAL and ideogram.LAST_SERIAL != self._agent_serial:
            if self.worker is not None and self.worker.isRunning():
                return                       # our own render, not the agent's
            self._pick_agent_layout()

    # ---- files -----------------------------------------------------------
    def _save_layout(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save layout", "layout.json",
                                              "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.layout_data, fh, ensure_ascii=False, indent=2)
            self.status.setText(f"Saved {os.path.basename(path)}")
        except Exception as exc:
            self.status.setText(f"Save failed: {exc}")

    def _load_layout(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load layout", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            self.status.setText(f"Load failed: {exc}")
            return
        import ideogram
        # accept both an editable layout and a raw Ideogram caption
        if isinstance(data, dict) and "compositional_deconstruction" in data:
            data = ideogram.caption_to_layout(data)
        self._set_layout(data, note=f"Loaded {os.path.basename(path)}")

    def shutdown(self):
        self._poll.stop()
        if self._agent_cancel is not None:
            self._agent_cancel.set()
        for w in (self.worker, self.plan_worker, self.agent_worker, self.edit_worker,
                  self.picedit_worker):
            if w is not None and w.isRunning():
                w.wait(200)
