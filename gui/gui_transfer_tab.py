"""Transfer tab: move a subject, a garment or a face from one picture to another.

Fifth tab out of gui.py. TransferWorker keeps the transfer off the UI thread
and takes a _ScopedCtx, so the main chat's Stop button cannot cancel a transfer
and a transfer cannot un-cancel the main chat.

The viewer and mask dialogs it opens live in gui_dialogs.py rather than here,
because AssistantWindow and ImagesPanel use them too.

MaskDrawDialog is the one of those that must be reached through `gui` at CALL
time (`_g.MaskDrawDialog`), never by the by-value name imported below. It is a
MODAL dialog, and every suite that drives _draw_mask stubs it as
`gui.MaskDrawDialog`; against a by-value binding that patch is a silent no-op,
the real dialog opens, and `exec_()` enters a nested event loop nothing will
ever quit -- offscreen included. gui_image_fix.py reaches it the same way for
the same reason.
"""
import os
import threading
from pathlib import Path

from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QComboBox, QDialog, QFileDialog, QFrame,
                             QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QScrollArea, QTextEdit, QVBoxLayout,
                             QWidget)

from ui_scale import px
from gui_common import (ACCENT, ACCENT2, BG, BORDER, MUTED, PANEL2, TEXT,
                        _ScopedCtx)
from gui_dialogs import (ImageViewerDialog, show_image_viewer, MaskCanvas,
                         MaskDrawDialog, OUTPUT_DIR_GUI_MASK)

import logging
logger = logging.getLogger("assistant.gui")


# --------------------------------------------------------------------------- #
# Multi-image transfer tab (Phase 9 GUI)
# --------------------------------------------------------------------------- #
# Display label -> backend role. "TARGET" and None are special (target / auto).
# FireRed's own documented style-transfer phrasing (prompt_augment.py's rewrite
# rules for "match picture 2's style"): identical wording to the one Telegram's
# 🎭 button sends (tg_resolve.py._push_style_transfer_task), so the same request
# behaves the same way from either surface.
_QUICK_STYLE_PROMPT = (
    "Change the visual style of the target image to match the reference image: "
    "lighting, colour palette, contrast, texture, atmosphere and photographic "
    "character. Preserve the exact subject identity, pose, composition and "
    "framing of the target -- do not replace the subject with anything from "
    "the reference."
)

_TRANSFER_ROLES = [
    ("Target (edit this)", "TARGET"),
    ("Auto-infer role", None),
    ("Clothing source", "clothing_source"),
    ("Object source", "object_source"),
    ("Hairstyle reference", "hairstyle_reference"),
    ("Face reference", "face_reference"),
    ("Pose reference", "pose_reference"),
    ("Style reference", "style_reference"),
    ("Identity reference", "identity_reference"),
    ("Scene reference", "scene_reference"),
    ("Lighting reference", "lighting_reference"),
]


class TransferWorker(QThread):
    """Run the multi-image reference-transfer pipeline off the UI thread.

    mode='plan'      -> infer roles, extract reference assets + a target mask for
                        each reference, build the plan; emits previews, no output.
    mode='contained' -> plan_and_execute_transfer (identity-preserving, the tool path).
    mode='whole'     -> transfer_with_references (single whole-frame pass; for debug).
    """
    progress = pyqtSignal(str)
    preview = pyqtSignal(dict)   # {path, role, asset, mask, region}
    done = pyqtSignal(dict)      # {output, info}
    failed = pyqtSignal(str)

    def __init__(self, ctx, target, refs_spec, instruction, mode, target_mask=None,
                 protect_face=True):
        super().__init__()
        self.ctx, self.target = ctx, target
        self.refs_spec = refs_spec          # list of (path, role_or_None)
        self.instruction = instruction
        self.mode = mode
        self.target_mask = target_mask      # optional hand-drawn target mask (L PNG)
        self.protect_face = protect_face    # protect face under a non-facial manual mask

    def run(self):
        try:
            import image as im
            refs = []
            for spec in self.refs_spec:
                p, r = spec[0], spec[1]
                src_mask = spec[2] if len(spec) > 2 else None
                eff_path = p
                if src_mask:
                    cropped = im.crop_to_mask(p, src_mask)
                    if cropped:
                        eff_path = cropped
                        self.progress.emit("Using your hand-drawn source region…")
                        logger.info("Transfer: source region cropped for %s",
                                    os.path.basename(p))
                refs.append(im.ReferenceImage(eff_path, r))
            if self.mode == "plan":
                im.infer_reference_roles(self.instruction, refs)
                for ref in refs:
                    if self.ctx.is_cancelled():
                        break
                    if ref.is_extractable():
                        self.progress.emit(f"Extracting {ref.effective_role}…")
                        im.extract_reference_asset(self.ctx, ref)
                    region = im._ROLE_TARGET_REGION.get(ref.effective_role)
                    mask = None
                    if region:
                        self.progress.emit(f"Masking target region '{region}'…")
                        try:
                            up = im._upload_image_to_comfy(self.target, im.COMFY_URL)
                            if up:
                                mask = im._region_mask_file(
                                    self.ctx, up, region, 4, seed=1, timeout=600)
                        except Exception:
                            logger.warning("plan: target mask failed", exc_info=True)
                    self.preview.emit({"path": ref.path, "role": ref.effective_role,
                                       "asset": ref.extracted_asset_path,
                                       "mask": mask, "region": region})
                plan = im.build_edit_plan(self.target, refs, self.instruction)
                lines = [f"Target : {os.path.basename(self.target)}",
                         f"Passes : {plan['n_passes']}", ""]
                for n, step in enumerate(plan["steps"], 1):
                    lines.append(f"Pass {n}: roles={step['roles']}  extract={step['extract']}")
                    lines.append(f"    “{step['instruction']}”")
                self.done.emit({"output": None, "info": "\n".join(lines)})
                return

            self.progress.emit("Running transfer — this can take a minute…")
            if self.mode == "whole":
                out = im.transfer_with_references(
                    self.ctx, self.target, refs, self.instruction or None)
            else:
                out = im.plan_and_execute_transfer(
                    self.ctx, self.target, refs, self.instruction or "",
                    mask_override=self.target_mask, protect_face=self.protect_face)
            out = im.assert_deliverable(out, where="gui.TransferWorker",
                                        source_path=self.target)
            if not out or not os.path.exists(out):
                self.failed.emit("The editor returned no result.")
                return
            info = ""
            try:
                import identity_metrics as idm
                c = idm.identity_cosine(self.target, out)
                if c is not None:
                    info = f"identity cosine = {c:.3f}" + (
                        "  ⚠ DRIFT" if c < 0.80 else "  ✓ identity preserved")
            except Exception:
                pass
            self.done.emit({"output": out, "info": info})
        except Exception as exc:
            logger.exception("Transfer worker failed")
            self.failed.emit(str(exc))


class TransferTab(QWidget):
    """Phase 9 — upload images, assign roles visually, run the transfer, view results.

    Drives the same backend the agent tool uses (image.plan_and_execute_transfer over
    ctx.reference_images), so the GUI result matches the contained-transfer behavior."""

    def __init__(self, host):
        super().__init__()
        self.host = host                 # AssistantWindow (for ctx, images_panel, _add_system)
        self._rows = []                  # [{path, combo, frame, mask, mask_btn, ...}]
        self._sel = None                 # currently selected row dict (or None)
        self.worker = None
        # This tab's OWN cancel token (see _ScopedCtx). A transfer must never
        # clear ctx.cancel_event: that flag may be carrying a Stop the user
        # pressed for the chat turn or the research run.
        self._cancel = threading.Event()

        root = QVBoxLayout(self)
        root.setContentsMargins(px(6), px(6), px(6), px(6))

        # ---- toolbar ----
        bar = QHBoxLayout()
        add_btn = QPushButton("➕  Add images…"); add_btn.clicked.connect(self._add_files)
        use_btn = QPushButton("⤵  Use loaded"); use_btn.setObjectName("ghost")
        use_btn.setToolTip("Pull in every image loaded/pasted this session (ctx.reference_images)")
        use_btn.clicked.connect(self._pull_loaded)
        clr_btn = QPushButton("✕  Clear"); clr_btn.setObjectName("ghost"); clr_btn.clicked.connect(self._clear)
        bar.addWidget(add_btn); bar.addWidget(use_btn); bar.addWidget(clr_btn); bar.addStretch(1)
        root.addLayout(bar)

        hint = QLabel("Upload images → assign one as <b>Target</b> and the rest as references "
                      "→ run. Newest is Target by default.<br>"
                      "Use the <b>Region</b> button on a row to <b>draw a region</b>: on the Target = exactly where the edit "
                      "lands (pick one person in a group photo, or a tattoo spot); on a reference = "
                      "the source area to take from. Manual masks override auto-detection.")
        hint.setWordWrap(True); hint.setStyleSheet(f"color:{MUTED};")
        root.addWidget(hint)

        # ---- image list (role assignment) ----
        self._list_area = QScrollArea(); self._list_area.setWidgetResizable(True)
        self._list_host = QWidget(); self._list_host.setObjectName("trowhost")
        self._list_host.setStyleSheet(f"QWidget#trowhost {{ background:{BG}; }}")
        self._list_lay = QVBoxLayout(self._list_host)
        self._list_lay.setAlignment(Qt.AlignTop)
        self._placeholder = QLabel("No images yet — add at least two (one Target + one reference).")
        self._placeholder.setStyleSheet(f"color:{MUTED};"); self._placeholder.setAlignment(Qt.AlignCenter)
        self._list_lay.addWidget(self._placeholder)
        self._list_area.setWidget(self._list_host)
        self._list_area.setMinimumHeight(px(150))
        root.addWidget(self._list_area, 1)

        # ---- instruction ----
        self.instr = QLineEdit()
        self.instr.setPlaceholderText("What to transfer? e.g. “put the jacket from the reference on the person”")
        root.addWidget(self.instr)

        # ---- run controls ----
        ctr = QHBoxLayout()
        self.style_btn = QPushButton("🎭  Quick style transfer")
        self.style_btn.setToolTip(
            "One click: every non-Target image becomes a Style reference and the "
            "instruction box is filled in with FireRed's own documented "
            "style-transfer prompt, then runs the contained transfer.")
        self.style_btn.clicked.connect(self._quick_style)
        ctr.addWidget(self.style_btn)
        self.plan_btn = QPushButton("◉  Preview plan"); self.plan_btn.setObjectName("ghost")
        self.plan_btn.clicked.connect(lambda: self._run("plan"))
        self.run_btn = QPushButton("▶  Run contained transfer")
        self.run_btn.setToolTip("Identity-preserving (mask + composite-back). Recommended.")
        self.run_btn.clicked.connect(lambda: self._run("contained"))
        self.whole_btn = QPushButton("⚠  Run whole-frame"); self.whole_btn.setObjectName("ghost")
        self.whole_btn.setToolTip("Single whole-frame pass — faster but can drift the face. Debug.")
        self.whole_btn.clicked.connect(lambda: self._run("whole"))
        ctr.addWidget(self.plan_btn); ctr.addWidget(self.run_btn); ctr.addWidget(self.whole_btn)
        root.addLayout(ctr)

        self.status = QLabel(""); self.status.setStyleSheet(f"color:{ACCENT2};"); self.status.setWordWrap(True)
        root.addWidget(self.status)

        # ---- visualization ----
        viz = QGridLayout(); viz.setSpacing(px(6))
        self._viz = {}
        for col, key in enumerate(("target", "asset", "mask", "output")):
            cap = QLabel({"target": "Target", "asset": "Extracted asset",
                          "mask": "Target region mask", "output": "Output"}[key])
            cap.setAlignment(Qt.AlignCenter); cap.setStyleSheet(f"color:{MUTED}; font-size:{px(11)}px;")
            lbl = QLabel(); lbl.setAlignment(Qt.AlignCenter); lbl.setMinimumHeight(px(150))
            lbl.setStyleSheet(f"border:1px solid {BORDER}; border-radius:6px;")
            lbl.setCursor(Qt.PointingHandCursor)
            lbl._fullpath = None
            lbl.mouseDoubleClickEvent = (lambda e, l=lbl: l._fullpath and show_image_viewer(
                [l._fullpath], 0, self.window()))
            viz.addWidget(cap, 0, col); viz.addWidget(lbl, 1, col)
            self._viz[key] = lbl
        root.addLayout(viz, 1)

        self.plan_view = QTextEdit(); self.plan_view.setReadOnly(True)
        self.plan_view.setObjectName("terminal"); self.plan_view.setMaximumHeight(px(110))
        self.plan_view.setPlaceholderText("Plan / metrics will appear here.")
        root.addWidget(self.plan_view)

    # ---- helpers ----------------------------------------------------------- #
    @property
    def ctx(self):
        return getattr(self.host, "ctx", None)

    def _set_viz(self, key, path):
        lbl = self._viz[key]
        lbl._fullpath = path if (path and os.path.exists(path)) else None
        if lbl._fullpath:
            pix = QPixmap(path)
            if not pix.isNull():
                lbl.setPixmap(pix.scaled(px(190), px(190), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                return
        lbl.clear()

    def _busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    # ---- image list -------------------------------------------------------- #
    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add images", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        for p in paths:
            self._add_row(p)

    def _pull_loaded(self):
        ctx = self.ctx
        if ctx is None:
            return
        for p in (ctx.reference_images or []):
            self._add_row(p)
        if not self._rows:
            self.status.setText("No images have been loaded/pasted this session yet.")

    def _add_row(self, path: str):
        if not path or not os.path.exists(path):
            return
        if any(r["path"] == path for r in self._rows):
            return
        if self._placeholder is not None:
            self._placeholder.hide(); self._placeholder.deleteLater(); self._placeholder = None

        frame = QFrame(); frame.setObjectName("trow")
        frame.setStyleSheet(
            f"QFrame#trow {{ background:{PANEL2}; border:1px solid {BORDER}; border-radius:6px; }}"
            f"QFrame#trow QLabel {{ background:transparent; border:none; color:{TEXT}; }}")
        h = QHBoxLayout(frame); h.setContentsMargins(px(6), px(4), px(6), px(4))
        thumb = QLabel(); pix = QPixmap(path)
        if not pix.isNull():
            thumb.setPixmap(pix.scaled(px(56), px(56), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        thumb.setFixedWidth(px(60)); thumb.setAlignment(Qt.AlignCenter)
        name = QLabel(os.path.basename(path))
        name.setToolTip(path)
        combo = QComboBox()
        for label, _role in _TRANSFER_ROLES:
            combo.addItem(label)
        combo.currentIndexChanged.connect(lambda _i, c=combo: self._on_role_changed(c))
        mask_btn = QPushButton("Region"); mask_btn.setObjectName("ghost")
        mask_btn.setToolTip("Draw a mask on this image (region to edit / source region)")
        mask_btn.clicked.connect(lambda: self._draw_mask(frame))
        rm = QPushButton("Remove"); rm.setObjectName("ghost")
        rm.setToolTip("Remove this image from the transfer")
        rm.clicked.connect(lambda: self._remove_row(frame))
        h.addWidget(thumb); h.addWidget(name, 1); h.addWidget(combo); h.addWidget(mask_btn); h.addWidget(rm)
        self._list_lay.addWidget(frame)
        row = {"path": path, "combo": combo, "frame": frame,
               "mask": None, "mask_btn": mask_btn, "protect_face": True}
        self._rows.append(row)
        # Clicking anywhere on the row (not the buttons/combo) selects it.
        frame._row = row
        frame.mousePressEvent = lambda _e, fr=frame: self._select_row(getattr(fr, "_row", None))
        # Default: newest row is the Target, demote any previous target to auto.
        for r in self._rows:
            r["combo"].blockSignals(True)
            r["combo"].setCurrentIndex(0 if r["frame"] is frame else
                                       (1 if self._role_of(r["combo"]) == "TARGET" else r["combo"].currentIndex()))
            r["combo"].blockSignals(False)
        self._select_row(row)

    def _draw_mask(self, frame):
        row = next((r for r in self._rows if r["frame"] is frame), None)
        if row is None:
            return
        is_target = self._role_of(row["combo"]) == "TARGET"
        title = ("Draw TARGET region (where the edit lands)" if is_target
                 else "Draw SOURCE region (what to take from this image)")
        # Resolved through `gui` at call time, not the module-level name — see
        # the module docstring: a by-value binding makes every gui.MaskDrawDialog
        # stub a no-op and hangs the caller in the real modal's event loop.
        import gui as _g
        dlg = _g.MaskDrawDialog(row["path"], existing_mask=row.get("mask"),
                                parent=self.window(), title=title)
        if dlg.exec_() == QDialog.Accepted:
            row["mask"] = dlg.mask_path
            row["protect_face"] = dlg.protect_face
            row["mask_btn"].setText("Region ✓" if dlg.mask_path else "Region")
            row["mask_btn"].setToolTip(
                f"Mask saved: {os.path.basename(dlg.mask_path)}" if dlg.mask_path
                else "No mask drawn")
            logger.info("Transfer: %s mask %s for %s",
                        "target" if is_target else "source",
                        "saved" if dlg.mask_path else "cleared", os.path.basename(row["path"]))
            if dlg.mask_path:
                self._set_viz("mask", dlg.mask_path)

    def _restore_placeholder_if_empty(self):
        if not self._rows and self._placeholder is None:
            self._placeholder = QLabel(
                "No images yet — add at least two (one Target + one reference).")
            self._placeholder.setStyleSheet(f"color:{MUTED};")
            self._placeholder.setAlignment(Qt.AlignCenter)
            self._list_lay.addWidget(self._placeholder)

    # ---- selection --------------------------------------------------------- #
    def _style_row(self, row, selected: bool):
        frame = row.get("frame")
        if frame is None:
            return
        border = ACCENT if selected else BORDER
        frame.setStyleSheet(
            f"QFrame#trow {{ background:{PANEL2}; border:1px solid {border}; border-radius:6px; }}"
            f"QFrame#trow QLabel {{ background:transparent; border:none; color:{TEXT}; }}")

    def _select_row(self, row):
        """Mark `row` selected (highlight + remember). None clears the selection."""
        if row is not None and row not in self._rows:
            row = None
        self._sel = row
        for r in self._rows:
            self._style_row(r, r is row)

    def _owns_mask(self, mask_path) -> bool:
        """True only for masks WE wrote (outputs/masks/transfer_mask_*.png), so we
        never delete a user's original image or an unrelated file."""
        if not mask_path:
            return False
        try:
            mp = Path(mask_path).resolve()
            return (mp.parent == OUTPUT_DIR_GUI_MASK().resolve()
                    and mp.name.startswith("transfer_mask_"))
        except Exception:
            return False

    def _drop_mask_file(self, row):
        mp = row.get("mask")
        if self._owns_mask(mp) and os.path.exists(mp):
            try:
                os.remove(mp)
                logger.info("Transfer: cleaned up orphaned mask %s", os.path.basename(mp))
            except OSError as exc:
                logger.debug("mask cleanup failed for %s: %s", mp, exc)
        row["mask"] = None

    def _remove_row(self, frame):
        """Remove exactly the image owning `frame`. Other rows (paths, roles, masks,
        prompt, settings) are untouched. Cleans the row's mask file + any viz pane it
        populated, and re-selects the next available row if the deleted one was selected."""
        idx = next((i for i, r in enumerate(self._rows) if r["frame"] is frame), None)
        if idx is None:
            return
        row = self._rows[idx]
        was_selected = row is self._sel
        # Clear viz panes that were showing THIS row's image/mask (avoid stale previews).
        for key, ref in (("target", row["path"]), ("mask", row.get("mask"))):
            if ref and self._viz[key]._fullpath == ref:
                self._set_viz(key, None)
        self._drop_mask_file(row)              # free the orphaned mask PNG on disk
        del self._rows[idx]
        self._sel = None if was_selected else self._sel
        frame.setParent(None); frame.deleteLater()
        logger.info("Transfer: removed image %s (%d remain)",
                    os.path.basename(row["path"]), len(self._rows))
        # Auto-select the next available row (same slot, clamped) if we removed the
        # selected one; otherwise leave the existing selection highlighted.
        if was_selected and self._rows:
            self._select_row(self._rows[min(idx, len(self._rows) - 1)])
        else:
            self._select_row(self._sel)
        self._restore_placeholder_if_empty()

    def _clear(self):
        for r in self._rows:
            self._drop_mask_file(r)
            r["frame"].setParent(None); r["frame"].deleteLater()
        self._rows = []
        self._sel = None
        for k in self._viz:
            self._set_viz(k, None)
        self.plan_view.clear(); self.status.clear()
        self._restore_placeholder_if_empty()

    def _role_of(self, combo):
        return _TRANSFER_ROLES[combo.currentIndex()][1]

    def _on_role_changed(self, combo):
        # Enforce a single Target: if this row became Target, demote the others.
        if self._role_of(combo) != "TARGET":
            return
        for r in self._rows:
            if r["combo"] is not combo and self._role_of(r["combo"]) == "TARGET":
                r["combo"].blockSignals(True); r["combo"].setCurrentIndex(1); r["combo"].blockSignals(False)

    # ---- execution --------------------------------------------------------- #
    def _gather(self):
        if self.ctx is None:
            self.status.setText("Model still loading — try again in a moment.")
            return None
        if len(self._rows) < 2:
            self.status.setText("Add at least two images (one Target + one reference).")
            return None
        target = None
        target_mask = None
        protect_face = True
        refs = []
        for r in self._rows:
            role = self._role_of(r["combo"])
            if role == "TARGET":
                if target is not None:
                    self.status.setText("Only one image can be the Target."); return None
                target = r["path"]; target_mask = r.get("mask")
                protect_face = r.get("protect_face", True)
            else:
                refs.append((r["path"], role, r.get("mask")))  # role None=auto; mask=source region
        if target is None:                       # default to the last row
            last = self._rows[-1]
            target = last["path"]; target_mask = last.get("mask")
            protect_face = last.get("protect_face", True)
            refs = [(r["path"], self._role_of(r["combo"]), r.get("mask"))
                    for r in self._rows[:-1]]
        if not refs:
            self.status.setText("Need at least one reference (non-Target) image."); return None
        return target, refs, target_mask, protect_face

    def _quick_style(self):
        """One-click FireRed style transfer: every non-Target row becomes a
        Style reference and the instruction box gets the canonical prompt,
        then runs exactly like pressing 'Run contained transfer' by hand.
        Nothing new is built -- style_reference has been a selectable role
        here all along; this just removes the "pick the role, type the
        prompt" steps for the single most common use of the tab."""
        if self._busy():
            return
        if len(self._rows) < 2:
            self.status.setText(
                "Add a target photo and a style-reference photo first "
                "(➕ Add images… or ⤵ Use loaded), then press this again.")
            return
        style_idx = next(i for i, (_, role) in enumerate(_TRANSFER_ROLES)
                         if role == "style_reference")
        for r in self._rows:
            if self._role_of(r["combo"]) != "TARGET":
                r["combo"].blockSignals(True)
                r["combo"].setCurrentIndex(style_idx)
                r["combo"].blockSignals(False)
        self.instr.setText(_QUICK_STYLE_PROMPT)
        self._run("contained")

    def _run(self, mode):
        if self.ctx is None:
            self.status.setText("Model not loaded yet — wait for startup to finish.")
            return
        if self._busy():
            return
        g = self._gather()
        if not g:
            return
        target, refs, target_mask, protect_face = g
        self._set_viz("target", target)
        if target_mask:
            self._set_viz("mask", target_mask)
        # Only clear the panes THIS run will repopulate, so Preview and Output
        # coexist: Preview fills asset+mask (keeps any output); a transfer fills
        # output (keeps the preview's asset+mask). A hand-drawn mask is kept.
        if mode == "plan":
            self._set_viz("asset", None)
            if not target_mask:
                self._set_viz("mask", None)
        else:
            self._set_viz("output", None)
        if target_mask:
            self.status.setText("Using your hand-drawn target mask.")
        self.plan_view.clear()
        self._cancel.clear()
        self.status.setText({"plan": "Building plan…", "contained": "Transferring…",
                             "whole": "Transferring (whole-frame)…"}[mode])
        for b in (self.style_btn, self.plan_btn, self.run_btn, self.whole_btn):
            b.setEnabled(False)
        self.worker = TransferWorker(_ScopedCtx(self.ctx, self._cancel),
                                     target, refs, self.instr.text().strip(),
                                     mode, target_mask=target_mask, protect_face=protect_face)
        self.worker.progress.connect(self.status.setText)
        self.worker.preview.connect(self._on_preview)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_preview(self, d):
        if d.get("asset"):
            self._set_viz("asset", d["asset"])
        if d.get("mask"):
            self._set_viz("mask", d["mask"])

    def _on_done(self, d):
        if d.get("info"):
            self.plan_view.setPlainText(d["info"])
        out = d.get("output")
        if out and os.path.exists(out):
            self._set_viz("output", out)
            ctx = self.ctx
            if ctx is not None:
                ctx.last_image_path = out
                ctx.reference_images = (ctx.reference_images or []) + [out]
            try:
                self.host.images_panel.add_image(out)
                self.host._add_system(f"Transfer complete — saved {os.path.basename(out)}.")
            except Exception:
                pass
            self.status.setText("Done. " + (d.get("info") or ""))
        else:
            self.status.setText("Plan ready." if d.get("info") else "Done.")

    def _on_failed(self, msg):
        self.status.setText(f"Failed: {msg}")

    def _on_finished(self):
        self.worker = None
        self._cancel.clear()          # our own token only — never the global one
        for b in (self.style_btn, self.plan_btn, self.run_btn, self.whole_btn):
            b.setEnabled(True)

    def cancel(self):
        """Abort this tab's transfer. Called by the main Stop button and by the
        window's shutdown join."""
        self._cancel.set()

    def shutdown(self):
        self._cancel.set()
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(5000)
