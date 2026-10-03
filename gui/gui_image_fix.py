"""Redraw / fix-hands / fix-artifact on the last generated image.

Split out of gui.py. Reaches RedrawWorker, FixHandsWorker, FixArtifactWorker
and MaskDrawDialog back through `import gui as _g` at call time rather than
by value: all four are patched directly ON the gui module by test_gui_window
and half a dozen supplement suites (RedrawWorker/FixHandsWorker/
FixArtifactWorker by test_gui_supplement{5,9,10}; MaskDrawDialog by
test_gui_supplement{2,4,5,6}, test_gui_tabs). Binding them by value here
would have frozen the real classes at import time and made every one of
those patches a silent no-op — the same dead-seam failure fixed the same way
throughout this refactor (tools.py's render seam, gui.py's research tab,
graph.py's personality_node): reach a dependency through the module that
owns it, never by value across a split.
"""
import logging
import os

from PyQt5.QtWidgets import QDialog, QInputDialog

import style_presets as _style_presets

logger = logging.getLogger("assistant.gui")


class ImageFixMixin:
    """Redraw / fix-hands / fix-artifact actions on the last generated image."""

    # Same English prefixes as the Telegram buttons (tg_bot._PROMPT_KB /
    # tg_callbacks._cb_change_clothes): one route, one narrowed tool list
    # (graph_fastpath._INTENT_TOOL_MAP), whichever surface pressed it.
    REMOVE_PREFIX = "remove from the image: "
    OUTFIT_PREFIX = "change the outfit to: "

    def _ask_and_send_image_edit(self, title: str, prompt: str, prefix: str):
        if self.ctx is None:
            return
        source = getattr(self.ctx, "last_image_path", "")
        if not source or not os.path.exists(source):
            self._add_system('No picture — draw or paste one first.')
            return
        text, ok = QInputDialog.getText(self, title, prompt)
        if not ok or not text.strip():
            return
        self.input.setText(prefix + text.strip())
        self._send_text()          # model gate + busy queue live there

    def _remove_object_last_image(self):
        self._ask_and_send_image_edit(
            'Remove object', 'What to remove? (e.g. «the cup on the table», «all lettering»)',
            self.REMOVE_PREFIX)

    def _change_outfit_last_image(self):
        self._ask_and_send_image_edit(
            'Outfit', 'What outfit? (e.g. «a white summer dress»)',
            self.OUTFIT_PREFIX)

    def _style_preset_last_image(self):
        """🎭 Style preset: same RedrawWorker path as Redraw last,
        but picking a named preset (or typing a custom one) instead of
        describing an edit -- no second reference photo needed."""
        import gui as _g
        if self.ctx is None or self._reject_if_busy('style change'):
            return
        source = self.ctx.last_image_path
        if not source or not os.path.exists(source):
            self._add_system("No image to restyle yet — generate one first.")
            return
        custom_label = "✏️ Custom (type my own)…"
        items = [_style_presets.preset_label(k, "en")
                 for k in _style_presets.STYLE_PRESET_ORDER] + [custom_label]
        choice, ok = QInputDialog.getItem(
            self, "Style preset", "Convert the last image to:", items, 0, False)
        if not ok:
            return
        if choice == custom_label:
            text, ok2 = QInputDialog.getText(self, "Custom style",
                                             "Describe the style (e.g. \"1980s polaroid photo\"):")
            if not ok2 or not text.strip():
                return
            instructions = text.strip()
        else:
            key = _style_presets.STYLE_PRESET_ORDER[items.index(choice)]
            instructions = _style_presets.preset_prompt(key)
        self._add_system(f"Changing style: {choice}…")
        self.ctx.cancel_event.clear()
        self.redraw_worker = _g.RedrawWorker(self.ctx, source, instructions, "redraw", "",
                                             engine="firered")
        self.redraw_worker.done.connect(self._on_redraw_done)
        self.redraw_worker.failed.connect(self._on_redraw_failed)
        self.redraw_worker.finished.connect(self._on_redraw_finished)
        self.redraw_worker.start()
        self._set_busy(True)
        self.stage.set_stage("Changing style")

    def _redraw_last_image(self):
        import gui as _g
        if self.ctx is None or self._reject_if_busy('redraw'):
            return
        source = self.ctx.last_image_path
        if not source or not os.path.exists(source):
            self._add_system("No image to redraw yet — generate one first.")
            return
        text, ok = QInputDialog.getText(
            self, "Edit image",
            "What to change? (leave empty to just enhance quality)")
        if not ok:
            return
        instructions = text.strip()
        # If the user names a region, do a precise regional inpaint (only that area
        # changes); otherwise fall back to whole-image redraw / quality enhance.
        region = ""
        if instructions:
            region_text, ok2 = QInputDialog.getText(
                self, "Edit region",
                "Which part to edit? e.g. face, shirt, hand, hair, background\n"
                "(leave empty to re-render the whole image)")
            if not ok2:
                return
            region = region_text.strip()
        if instructions and region:
            mode, verb = "inpaint", "Editing"
        elif instructions:
            mode, verb = "redraw", "Redrawing"
        else:
            mode, verb = "enhance", "Enhancing"
        detail = f" ({region}: {instructions})" if region else (f" ({instructions})" if instructions else "")
        self._add_system(f"{verb} the last image…" + detail)
        self.ctx.cancel_event.clear()
        self.redraw_worker = _g.RedrawWorker(self.ctx, source, instructions, mode, region,
                                             engine="firered")
        self.redraw_worker.done.connect(self._on_redraw_done)
        self.redraw_worker.failed.connect(self._on_redraw_failed)
        self.redraw_worker.finished.connect(self._on_redraw_finished)
        self.redraw_worker.start()
        self._set_busy(True)
        self.stage.set_stage(verb)

    def _fix_hands_last_image(self):
        import gui as _g
        if self.ctx is None or self._reject_if_busy('hand fix'):
            return
        source = self.ctx.last_image_path
        if not source or not os.path.exists(source):
            self._add_system("No image to fix yet — generate or load one first.")
            return
        # Let the user draw over the bad hand (the reliable route for hands that
        # auto-detection can't localize). Cancelling the dialog still runs the
        # auto path (MeshGraphormer + SAM/Florence).
        mask_path = None
        try:
            dlg = _g.MaskDrawDialog(source, parent=self,
                                    title="Draw over the hand to fix (or Cancel for auto-detect)")
            if dlg.exec_() == QDialog.Accepted:
                mask_path = dlg.mask_path
        except Exception:
            logger.warning("fix-hands: mask dialog failed; using auto-detect", exc_info=True)
        # Remember the ORIGINAL source + mask so "Retry" can re-run with a fresh
        # seed (fix_hands randomizes its seed each call) WITHOUT re-fixing the
        # already-fixed result or making the user redraw the mask.
        self._last_handfix_source = source
        self._last_handfix_mask = mask_path
        self.retryhands_btn.setEnabled(True)
        self._run_fix_hands(source, mask_path, retry=False)

    def _retry_hands(self):
        if self.ctx is None or self._reject_if_busy('hand fix retry'):
            return
        source = getattr(self, "_last_handfix_source", None)
        if not source or not os.path.exists(source):
            self._add_system("Nothing to retry yet — run Fix hands first.")
            return
        self._run_fix_hands(source, getattr(self, "_last_handfix_mask", None), retry=True)

    def _fix_artifact_last_image(self):
        import gui as _g
        if self.ctx is None or self._reject_if_busy('artifact removal'):
            return
        source = self.ctx.last_image_path
        if not source or not os.path.exists(source):
            self._add_system("No image to fix yet — generate or load one first.")
            return
        # A mask is REQUIRED here (unlike Fix hands there's no auto-detect): the user
        # must point at the artifact. Cancelling or drawing nothing aborts.
        mask_path, protect_face = None, True
        try:
            dlg = _g.MaskDrawDialog(source, parent=self,
                                    title="Paint over the artifact / awkward area to fix")
            if dlg.exec_() != QDialog.Accepted:
                return
            mask_path, protect_face = dlg.mask_path, dlg.protect_face
        except Exception:
            logger.warning("fix-artifact: mask dialog failed", exc_info=True)
            return
        if not mask_path or not os.path.exists(mask_path):
            self._add_system("Nothing selected — paint over the area you want fixed, then Save.")
            return
        text, ok = QInputDialog.getText(
            self, "Describe the fix",
            "What's wrong / what should it look like?\n"
            "e.g. 'smooth this harsh transition', 'remove this smear',\n"
            "'fix this melted edge' (leave empty for a generic clean-up)")
        if not ok:
            return
        instruction = text.strip() or (
            "Repair the marked area: fix any artifact, awkward transition, smear or "
            "distortion and make it look natural and seamless, matching the surrounding "
            "texture, colour and lighting.")
        self._add_system(f"Fixing the selected area… ({instruction[:60]})")
        self.ctx.cancel_event.clear()
        self.redraw_worker = _g.FixArtifactWorker(self.ctx, source, mask_path, instruction,
                                                  protect_face=protect_face, engine="firered")
        self.redraw_worker.done.connect(self._on_redraw_done)
        self.redraw_worker.failed.connect(self._on_redraw_failed)
        self.redraw_worker.finished.connect(self._on_redraw_finished)
        self.redraw_worker.start()
        self._set_busy(True)
        self.stage.set_stage("Fixing artifact")

    def _toggle_whole_frame(self):
        """Toggle MASKLESS whole-frame mode: ON =
        the entire image is re-rendered by the instruction edit (no segmentation,
        no composite; face pasted back automatically) — for stubborn edits where
        masking keeps failing. OFF = normal masked/contained pipeline."""
        on = (getattr(self.ctx, "image_edit_engine", "auto") == "firered_whole"
              if self.ctx is not None else False)
        on = not on
        if self.ctx is not None:
            self.ctx.image_edit_engine = "firered_whole" if on else "auto"
        self.whole_frame_btn.setText(f"🖼  Whole frame: {'ON' if on else 'OFF'}")
        self._add_system(
            "Whole-frame mode ON — edits re-render the ENTIRE image with no mask and "
            "no identity guards: the model has complete freedom, face included." if on
            else "Whole-frame mode OFF — edits use the normal masked pipeline.")

    def _run_fix_hands(self, source, mask_path, *, retry):
        import gui as _g
        region = " (drawn region)" if mask_path else " (auto-detect)"
        self._add_system(("Retrying hand fix with a new seed…" if retry
                          else "Fixing the hand…") + region)
        self.ctx.cancel_event.clear()
        self.redraw_worker = _g.FixHandsWorker(self.ctx, source, mask_path, engine="firered")
        self.redraw_worker.done.connect(self._on_redraw_done)
        self.redraw_worker.failed.connect(self._on_redraw_failed)
        self.redraw_worker.finished.connect(self._on_redraw_finished)
        self.redraw_worker.start()
        self._set_busy(True)
        self.stage.set_stage("Fixing hands")

    def _on_redraw_done(self, path: str):
        import image as _img
        if path and _img.is_intermediate_artifact(path):
            # Same gate as _on_done: a cropped working tile must never be delivered.
            logger.error("UI DELIVERY REJECTED: intermediate tile reached _on_redraw_done: %s", path)
            self._add_system('Internal error: the editor returned an intermediate fragment, not the finished image. The result is not shown.')
            self._set_status("Redraw failed (intermediate tile rejected).")
        elif path and os.path.exists(path):
            if self.ctx is not None:
                self.ctx.last_image_path = path
            self.images_panel.add_image(path)
            self._add_system("Done — see the Images tab.")
            self._set_status("Redraw complete.")
        else:
            self._add_system("Redraw failed — the image server returned no result.")
            self._set_status("Redraw failed.")

    def _on_redraw_failed(self, msg: str):
        self._add_system(f"Redraw error: {msg}")
        self._set_status("Redraw failed.")

    def _on_redraw_finished(self):
        self.redraw_worker = None
        if self.ctx is not None:
            self.ctx.cancel_event.clear()
        self._set_busy(False)
        self.stage.set_stage("Ready")
