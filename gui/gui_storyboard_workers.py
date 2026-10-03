"""Storyboard background workers.

Lifted out of gui_storyboard_tab.py. Four QThreads that run planning, drawing
and the two edit paths off the UI thread.

They are re-exported from gui_storyboard_tab by value ON PURPOSE. The callers
(StoryboardTab._plan / ._draw) stay in that module and resolve the class as a
global THERE, so the existing seam -- suites patch gui_storyboard_tab.
IdeogramWorker and gui.IdeogramWorker to install a spy -- keeps working. Moving
the callers instead would have orphaned both patches silently.
"""
import copy
import json
import os
import random
import threading
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QLabel, QLineEdit, QListWidget, QPushButton, QSpinBox,
    QWidget,
)
from ui_scale import pt, px
from gui_common import _FlowWidget, _flow, _section
import logging

logger = logging.getLogger("assistant.gui")


class IdeogramWorker(QThread):
    """Render one storyboard layout off the UI thread."""
    done = pyqtSignal(str)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)     # (step, total) live from ComfyUI

    def __init__(self, ctx, caption, width, height, seed):
        super().__init__()
        self.ctx, self.caption = ctx, caption
        self.width, self.height, self.seed = width, height, seed

    def run(self):
        try:
            import ideogram
            path = ideogram.generate(self.ctx, "", width=self.width, height=self.height,
                                     seed=self.seed, caption=self.caption,
                                     on_progress=lambda k, n: self.progress.emit(k, n))
            if not path:
                self.failed.emit("ComfyUI returned no image (is it running on the Ideogram "
                                 "build, with the Ideogram 4 models present?)")
                return
            self.done.emit(path)
        except Exception as exc:
            # ContentRefused lands here too: the model drew its safety card. Report it —
            # do NOT silently reroute to another engine.
            logger.exception("Ideogram storyboard render failed")
            self.failed.emit(str(exc))


class DrawAgentWorker(QThread):
    """Run the box-arranging draw agent off the UI thread.

    Every step is forwarded as a signal rather than accumulated, so the canvas shows
    the boxes the agent is working with WHILE it works — the point of the mode is
    watching the arrangement change, not waiting for a final picture.
    """
    event = pyqtSignal(str, object)     # (kind, data dict)
    progress = pyqtSignal(int, int)     # (step, total) live from ComfyUI
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, ctx, *, request="", layout=None, instruction="",
                 width=1024, height=1024, rounds=2, auto_fix=True, seed=0):
        super().__init__()
        self.ctx = ctx
        self.kw = dict(request=request, layout=layout, instruction=instruction,
                       width=width, height=height, rounds=rounds, auto_fix=auto_fix,
                       seed=seed)

    def run(self):
        try:
            import draw_agent
            req = self.kw.pop("request")
            result = draw_agent.run(self.ctx, req, on_event=lambda k, d: self.event.emit(k, d),
                                    on_progress=lambda k, n: self.progress.emit(k, n),
                                    **self.kw)
            self.done.emit(result)
        except Exception as exc:
            logger.exception("draw agent failed")
            self.failed.emit(str(exc))


class LayoutEditWorker(QThread):
    """Apply one natural-language edit to a layout (no rendering)."""
    done = pyqtSignal(object, object)   # (layout, notes)
    failed = pyqtSignal(str)

    def __init__(self, ctx, layout, instruction):
        super().__init__()
        self.ctx, self.layout, self.instruction = ctx, layout, instruction

    def run(self):
        try:
            import draw_agent
            layout, notes = draw_agent.edit_layout(self.ctx, self.layout, self.instruction)
            self.done.emit(layout, notes)
        except Exception as exc:
            logger.exception("layout edit failed")
            self.failed.emit(str(exc))


class PictureEditWorker(QThread):
    """Contained edit of one region of an existing picture, off the UI thread.

    Splits the instruction into region + new content, then runs the app's contained
    inpaint (segment that region, redraw only it, composite back over the original)
    so identity and everything outside the region are preserved by construction.
    """
    done = pyqtSignal(str)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, ctx, image_path, instruction):
        super().__init__()
        self.ctx, self.image_path, self.instruction = ctx, image_path, instruction

    def run(self):
        try:
            import draw_agent
            from image import inpaint_region_with_comfy
            spec = draw_agent.split_edit(self.ctx, self.instruction)
            path = inpaint_region_with_comfy(
                self.ctx, self.image_path, spec["region"],
                spec["content"] or self.instruction, removal=spec.get("removal", False),
                on_progress=lambda k, n: self.progress.emit(k, n))
            self.done.emit(path or "")
        except Exception as exc:
            logger.exception("picture region edit failed")
            self.failed.emit(str(exc))
