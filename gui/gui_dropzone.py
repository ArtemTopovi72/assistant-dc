"""Drag-and-drop, clipboard paste and the chat-viewport event filter.

A mixin, not free functions: every method here reaches into THIS window's
chat, input box and agent context. Extracted from gui.py because none of it
touches the agent loop, the workers or the tabs — it is purely "a file/image
arrived, turn it into a working image or a request".

The mixin expects from the host class: `_add_system`, `_add_user`,
`_add_user_image`, `_set_status`, `_busy`, `_start_worker`, `_image_url_at`,
`_url_to_local`, `_open_external`, and the `graph` / `ctx` / `chat` / `input`
attributes.  `eventFilter` calls `super().eventFilter`, so the mixin must
precede QMainWindow in the MRO.
"""
import io
import logging
import os
from prompt_guard import wrap_document
import time

from PyQt5.QtCore import QEvent, Qt, QTimer
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import QApplication

import config
from config import LM_STUDIO_BASE
import gui_workers
# Reached through the MODULE, never by value: the suites patch
# gui_dialogs.show_image_viewer to count viewer opens, and a by-value import
# here would pin the original function no matter what they patch.
import gui_dialogs as _dialogs
from ui_scale import px

logger = logging.getLogger("assistant.gui")


class DropPasteMixin:
    """File/image drops, clipboard paste and the chat/input event filter."""

    _IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif")

    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md.hasUrls():
            paths = [u.toLocalFile().lower() for u in md.urls()]
            if any(p.endswith((".txt", ".pdf") + self._IMAGE_EXTS) for p in paths):
                event.acceptProposedAction()
        elif md.hasImage():
            event.acceptProposedAction()

    def _mime_is_droppable(self, md) -> bool:
        if md.hasImage():
            return True
        if md.hasUrls():
            exts = (".txt", ".pdf") + self._IMAGE_EXTS
            return any(u.toLocalFile().lower().endswith(exts) for u in md.urls())
        return False

    def _route_dropped_mime(self, md) -> bool:
        """Handle a dropped/pasted payload (text, pdf, or image). Returns True if
        something was consumed. Shared by the window dropEvent and the child-widget
        event filter (so drops onto the chat/input box work too — those widgets
        accept drops themselves and would otherwise swallow the event)."""
        if md.hasUrls():
            for url in md.urls():
                path = url.toLocalFile()
                low = path.lower()
                if low.endswith(".txt"):
                    self._drop_text_file(path); return True
                if low.endswith(".pdf"):
                    self._drop_pdf_file(path); return True
                if low.endswith(self._IMAGE_EXTS):
                    self._load_image_file_as_working(path); return True
        if md.hasImage():
            return self._load_pasted_qimage(QImage(md.imageData()))
        return False

    def dropEvent(self, event):
        # A dropped file that lands on a busy window used to vanish -- no
        # message, no chip, nothing on screen moved. Dropping again is the
        # natural response, and it does nothing either.
        # No graph check of its own: _busy() already covers the initial
        # load, and _reject_if_busy words that case as "ещё загружаюсь".
        if self._reject_if_busy('processing the file'):
            return
        if self._route_dropped_mime(event.mimeData()):
            event.acceptProposedAction()

    # ---- image input (drag-drop / clipboard) ----
    def _model_has_vision(self) -> bool:
        """True if the active LM Studio model can analyze images. Cached per model;
        returns True on any uncertainty so we never warn falsely."""
        if self.ctx is None:
            return True
        mid = self.ctx.model_name
        cache = getattr(self, "_vision_cap_cache", None)
        if cache is None:
            cache = self._vision_cap_cache = {}
        if mid in cache:
            return cache[mid]
        has = True
        try:
            from lmstudio import fetch_model
            info = fetch_model(LM_STUDIO_BASE, mid)
            # LM Studio flags vision models by type "vlm". The `capabilities` list is
            # always just ["tool_use"] and NEVER contains "vision", so a string check
            # there marks every model as blind. Trust `type`; only declare "no vision"
            # when the model is positively a known text-only type.
            mtype = (info.get("type") or "").lower()
            caps = info.get("capabilities") or []
            if mtype == "vlm" or "vision" in caps:
                has = True
            elif mtype in ("llm", "embeddings"):
                has = False
            else:
                has = True   # unknown/unreachable — never warn falsely
        except Exception:
            has = True
        cache[mid] = has
        return has

    def _load_image_file_as_working(self, path: str):
        """Load an image from disk as the current working image: previewed, set as
        ctx.last_image_path (so inpaint_image/redraw act on it) and attached to the
        next message as image_data (so the agent can also see it)."""
        if not path or not os.path.exists(path) or QPixmap(path).isNull():
            self._add_system('Could not load the image.')
            return
        if self.ctx is not None:
            self.ctx.last_image_path = path
            # Track every loaded/pasted image for multi-image transfer_image.
            # Newest = target; earlier = references. Dedupe, cap to 8.
            refs = [p for p in (self.ctx.reference_images or []) if p != path]
            self.ctx.reference_images = (refs + [path])[-8:]
        try:
            from PIL import Image
            im = Image.open(path).convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=92)
            self.captured_image = buf.getvalue()  # attached to the next message for vision
        except Exception as exc:
            logger.warning("Could not encode dropped image for vision: %s", exc)
            self.captured_image = None
        self._add_user_image(path)
        if not self._model_has_vision():
            self._add_system('⚠ The current model cannot «see» images (no vision). It can EDIT the picture (inpaint, redraw, remove the background) but not describe it. To ask about its content, pick a VLM (vision) model in Settings.')
        self._set_status('🖼 Image loaded — write what to change (e.g. «put a fork in the hand»), or ask what is in it.')

    def _load_pasted_qimage(self, qimg) -> bool:
        if qimg is None or qimg.isNull():
            self._set_status('There is no image on the clipboard.')
            return False
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = str(config.OUTPUT_DIR / f"_pasted_{int(time.time() * 1000)}.png")
        if not qimg.save(path, "PNG"):
            self._add_system('Could not save the pasted image.')
            return False
        self._load_image_file_as_working(path)
        return True

    def _paste_image_from_clipboard(self):
        if self._reject_if_busy('clipboard paste'):
            return
        cb = QApplication.clipboard()
        md = cb.mimeData()
        if md.hasImage():
            if self._load_pasted_qimage(QImage(md.imageData())):
                return
        if md.hasUrls():
            for u in md.urls():
                p = u.toLocalFile()
                if p.lower().endswith(self._IMAGE_EXTS):
                    self._load_image_file_as_working(p)
                    return
        self._set_status('There is no image on the clipboard.')

    def eventFilter(self, obj, event):
        et = event.type()
        # The chat (QTextEdit) and input (QLineEdit) accept drops themselves, so a
        # file/image dropped ONTO them never reaches the window's dropEvent. Catch
        # it here and route it through the same handler.
        # NB: this filter is installed on the chat viewport before self.input is
        # created, and the real (non-offscreen) platform delivers events during the
        # rest of construction. Use getattr so an early event can't raise
        # AttributeError inside this Qt callback (which crashes the process).
        chat = getattr(self, "chat", None)
        inp = getattr(self, "input", None)
        watched = (inp is not None and obj is inp) or \
                  (chat is not None and obj is chat.viewport())
        # Reposition the floating scroll-to-bottom button when the chat resizes
        if chat is not None and obj is chat.viewport() and et == QEvent.Resize:
            btn = getattr(self, "_scroll_down_btn", None)
            if btn is not None and btn.isVisible():
                vp = chat.viewport()
                btn.move(vp.width() - px(40), vp.height() - px(40))
        # Double-click an image in the chat → open it in the full-size viewer.
        # Everything here runs inside a Qt C++ callback, so any exception would
        # fast-fail the whole process (0xC0000409). Two safeguards: swallow/log
        # any error, and open the modal viewer via a 0-ms timer instead of calling
        # exec_() synchronously — a nested modal event loop started while the
        # double-click event is still being dispatched re-enters event delivery
        # and can crash the app. Deferring runs it cleanly after this returns.
        if chat is not None and obj is chat.viewport() and et == QEvent.MouseButtonDblClick:
            try:
                url = self._image_url_at(event.pos())
                if url:
                    local = self._url_to_local(url)
                    # A poster frame stands in for a CLIP: play the clip, do not
                    # open the still. Checked before the existence test so a
                    # deleted poster still reports the video's name.
                    _clip = getattr(self, "_video_by_poster", {}).get(
                        os.path.abspath(local))
                    if _clip:
                        if os.path.exists(_clip):
                            QTimer.singleShot(0, lambda p=_clip: self._open_external(p))
                        else:
                            self._set_status(
                                f"Video file no longer exists: {os.path.basename(_clip)}")
                        return True
                    if os.path.exists(local):
                        QTimer.singleShot(
                            0, lambda p=local: _dialogs.show_image_viewer(p, parent=self))
                    else:
                        # show_image_viewer silently no-ops on a missing path — tell
                        # the user why nothing opened instead of appearing broken.
                        self._set_status(f"Image file no longer exists: {os.path.basename(local)}")
                    return True
            except Exception:
                logger.exception("Chat image double-click handler failed")
                return True
        # Hover feedback: pointing hand over a chat image, the normal I-beam elsewhere
        # (a read-only QTextEdit shows the text cursor everywhere, which made images
        # look non-interactive). Same in-Qt-callback safety rules as above.
        if chat is not None and obj is chat.viewport() and et == QEvent.MouseMove:
            try:
                over_img = self._image_url_at(event.pos()) is not None
                chat.viewport().setCursor(
                    Qt.PointingHandCursor if over_img else Qt.IBeamCursor)
            except Exception:
                logger.exception("Chat image hover-cursor handler failed")
        if watched and et in (QEvent.DragEnter, QEvent.DragMove):
            if self._mime_is_droppable(event.mimeData()):
                event.acceptProposedAction()
                return True
        if watched and et == QEvent.Drop:
            if self.graph is not None and not self._busy() and \
                    self._route_dropped_mime(event.mimeData()):
                event.acceptProposedAction()
                return True
        # Ctrl+V in the text box: if the clipboard holds an image, load it as the
        # working image; otherwise fall through so normal text paste still works.
        if inp is not None and obj is inp and et == QEvent.KeyPress and event.matches(QKeySequence.Paste):
            md = QApplication.clipboard().mimeData()
            if md.hasImage() or (md.hasUrls() and any(
                    u.toLocalFile().lower().endswith(self._IMAGE_EXTS) for u in md.urls())):
                self._paste_image_from_clipboard()
                return True
        return super().eventFilter(obj, event)

    # Dropped-file text cap. 30k chars (~10k tokens) could evict the system prompt
    # on a small context window; ~12k chars (~3-4k tokens) stays safe and is still
    # plenty for "перескажи/объясни этот файл".
    _DROP_CHARS = 12000

    def _drop_text_file(self, path: str):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                content = f.read(self._DROP_CHARS + 1)
        except Exception as exc:
            self._add_system(f"Could not read the file: {exc}")
            return
        truncated = len(content) > self._DROP_CHARS
        if truncated:
            content = content[:self._DROP_CHARS] + '\n…(file truncated)'
        name = os.path.basename(path)
        self._add_user(f"[File: {name}]" + (' (truncated)' if truncated else ""))
        self._start_worker(gui_workers.RequestWorker(self.ctx, self.graph, self.base_state,
                                         text=wrap_document(name, content)))

    def _drop_pdf_file(self, path: str):
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            pages = [p.extract_text() or "" for p in reader.pages]
            content = "\n\n".join(pages)
        except ImportError:
            self._add_system('PDF needs pypdf: pip install pypdf')
            return
        except Exception as exc:
            self._add_system(f"PDF read error: {exc}")
            return
        truncated = len(content) > self._DROP_CHARS
        if truncated:
            content = content[:self._DROP_CHARS] + '\n…(document truncated)'
        name = os.path.basename(path)
        self._add_user(f"[PDF: {name}]" + (' (truncated)' if truncated else ""))
        self._start_worker(gui_workers.RequestWorker(self.ctx, self.graph, self.base_state,
                                         text=wrap_document(name, content)))
