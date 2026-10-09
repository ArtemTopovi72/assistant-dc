"""Everything that puts a line into the chat transcript.

A mixin, not free functions: every method writes into THIS window's chat
widget. Extracted from gui.py because it is the presentation half of the
conversation — it renders a user line, an assistant line, a system notice, an
image, a video card — and knows nothing about who produced any of it.

Deliberately excludes the double-click handler and the viewer: those live in
gui_dropzone with the rest of the event filter. This module only WRITES the
transcript; reading a click out of it is a different job.

The video import stays at call time inside _add_result_video, so a machine
without the video stack still opens the window.
"""
import logging
import os
import re
import sys

from PyQt5.QtGui import QTextCursor

import gui_i18n
from gui_common import ACCENT, ACCENT2, MUTED, PANEL2, TEXT, _esc

logger = logging.getLogger("assistant.gui")


def _md(text: str) -> str:
    """Escaped text with the model's light markdown drawn: **bold**, *italic*, `code`, line breaks.
    The reply showed «**рыжий кот**» with its stars (Qt draws no markdown in insertHtml)."""
    t = _esc(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t, flags=re.S)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return t.replace("\n", "<br>")


def _bubble(inner: str, bg: str, fg: str, align: str) -> str:
    """A chat bubble Qt can draw: rich text ignores padding and radius on a span (the line
    looked like a text selection), a one-cell table keeps its cellpadding and background."""
    return (f'<table align="{align}" cellpadding="10" cellspacing="0" '
            f'style="background-color:{bg}; margin:4px 0;"><tr><td style="color:{fg};">{inner}</td></tr></table>')


class ChatViewMixin:
    """Renders user/assistant/system lines, images and video cards."""

    def _append_html(self, html):
        cur = self.chat.textCursor(); cur.movePosition(QTextCursor.End)
        cur.insertHtml(html + "<br>"); self.chat.setTextCursor(cur); self.chat.ensureCursorVisible()

    def _add_user(self, text):
        self._append_html(_bubble(_esc(text).replace("\n", "<br>"), ACCENT, "#ffffff", "right"))

    def _add_assistant(self, text):
        self._append_html(_bubble(_md(text), PANEL2, TEXT, "left"))

    def _add_system(self, text):
        text = gui_i18n.tr(text)
        self._append_html(f'<div align="center"><span style="color:{MUTED};">{_esc(text)}</span></div>')

    def _add_room_line(self, name, text, *, is_user=False):
        """Mirror one Madhouse line into the main conversation pane.

        The Madhouse tab lives in the narrow right panel, which is painful to read
        on a TV; the room's transcript belongs in the big centre pane. Each speaker
        gets a name label so a four-way conversation stays readable, and your own
        lines are right-aligned like your normal chat messages.
        """
        align = "right" if is_user else "left"
        bubble = ACCENT if is_user else PANEL2
        colour = "#fff" if is_user else TEXT
        self._append_html(_bubble(f'<span style="color:{ACCENT2};font-size:11px;">{_esc(name)}</span><br>'
                                  + _md(text), bubble, colour, align))

    def _add_user_image(self, path):
        url = "file:///" + path.replace("\\", "/")
        self._append_html(
            f'<div align="right"><img src="{url}" width="240" '
            f'title="Double-click to view full size"></div>')

    def _add_result_image(self, path):
        """Embed a tool-produced image inline in the chat (left-aligned), so it can be
        double-clicked to open full size — not only shown in the Images tab."""
        url = "file:///" + path.replace("\\", "/")
        self._append_html(
            f'<div align="left"><img src="{url}" width="240" '
            f'title="Double-click to view full size"></div>')

    def _add_result_video(self, path):
        """Show a generated clip in the chat as its poster frame.

        A QTextEdit cannot play video, so the honest options are a still the user
        can click or nothing at all. The poster is embedded and double-clicking it
        hands the CLIP to the OS player (see the eventFilter) rather than opening
        the poster in the image viewer — which is what would otherwise happen, and
        would look like the video was silently replaced by a screenshot.
        """
        try:
            import video as _vid
            poster = _vid.make_thumbnail(path)
        except Exception:
            logger.exception("could not build a poster frame for %s", path)
            poster = None

        secs = ""
        try:
            info = _vid.probe(path)
            if info.get("seconds"):
                secs = f" · {info['seconds']:.1f}s"
                if info.get("has_audio"):
                    secs += " · with sound"
        except Exception:
            pass

        if poster and os.path.exists(poster):
            if not hasattr(self, "_video_by_poster"):
                self._video_by_poster = {}
            self._video_by_poster[os.path.abspath(poster)] = path
            url = "file:///" + poster.replace("\\", "/")
            self._append_html(
                f'<div align="left"><img src="{url}" width="320" '
                f'title="Double-click to play"></div>')
        self._add_system(f"🎬 Video ready{secs}: {path}"
                         + ("" if poster else " (double-click unavailable — open it "
                                              "from the file above)"))

    def _add_result_document(self, path):
        """Announce a tool-produced document (deck, report...) and open it.

        create_presentation set state["document_path"] correctly and _on_done
        used it -- but only to decide whether the turn "produced nothing";
        nothing ever told the user the file existed or where it was. A
        finished .pptx sat on disk with no path shown and no way to open it
        from the window at all. Unlike a generated video, opening a document
        immediately in its OS-registered app is the one action the user
        wanted right after asking for it, and isn't the autoplay-style
        surprise a video would be.
        """
        name = os.path.basename(path)
        opened = self._open_external(path)
        self._add_system(f"📄 {name} ready" + ("" if opened else f": {path}"))

    def _open_external(self, path):
        """Hand a file to whatever the OS plays it with. Never raises into Qt."""
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)                      # noqa: S606 - user's own file
            else:
                import subprocess
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open",
                                  path])
            return True
        except Exception:
            logger.exception("could not open %s in an external player", path)
            return False

    def _image_url_at(self, pos):
        """Return the file:// URL of a chat image under the given viewport point,
        or None. Used to open the full viewer on double-click."""
        cur = self.chat.cursorForPosition(pos)
        # An inline image is a single object-replacement character, and
        # QTextCursor.charFormat() describes the char immediately BEFORE position().
        # cursorForPosition lands the cursor before OR after the image char depending
        # on which half of the image was clicked, so check both neighbours:
        #   * unmoved cursor  -> the char to the LEFT of the click point;
        #   * moved one right -> the char to the RIGHT of the click point.
        # (The previous code moved PreviousCharacter for the left case, which made
        # charFormat() describe the char TWO back — clicks on the right half of an
        # image found nothing, so the double-click viewer "didn't work".)
        right = QTextCursor(cur)
        right.movePosition(QTextCursor.NextCharacter)
        for c in (cur, right):
            fmt = c.charFormat()
            if fmt.isImageFormat():
                name = fmt.toImageFormat().name()
                if name:
                    return name
        return None

    @staticmethod
    def _url_to_local(url: str) -> str:
        path = url
        if path.startswith("file:///"):
            path = path[len("file:///"):]
        elif path.startswith("file://"):
            path = path[len("file://"):]
        return path.replace("/", os.sep)

    def _clear_chat(self):
        self.chat.clear()
