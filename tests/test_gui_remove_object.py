"""GUI 🧽/👗 buttons send the same prefixed request as the Telegram buttons."""
import os, sys, tempfile
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen"); os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt5.QtWidgets import QApplication, QInputDialog
app = QApplication.instance() or QApplication([])
from PIL import Image
import gui, gui_image_fix


def test_buttons_send_prefixed_request():
    w = gui.AssistantWindow("x", True)
    assert w.remove_obj_btn.text().startswith("🧽") and w.outfit_btn.text().startswith("👗")
    p = os.path.join(tempfile.mkdtemp(), "a.png"); Image.new("RGB", (8, 8)).save(p)
    import threading
    class C: last_image_path = p; cancel_event = threading.Event()
    w.ctx = C()
    sent = []
    w._send_text = lambda: sent.append(w.input.text())
    orig = QInputDialog.getText
    QInputDialog.getText = staticmethod(lambda *a, **k: ("чашку на столе", True))
    try:
        w._remove_object_last_image(); w._change_outfit_last_image()
    finally:
        QInputDialog.getText = orig
    assert sent == ["remove from the image: чашку на столе", "change the outfit to: чашку на столе"], sent
    import tg_bot
    assert gui_image_fix.ImageFixMixin.REMOVE_PREFIX == tg_bot._PROMPT_KB["remove_obj"]
    w.ctx = type("N", (), {"last_image_path": "", "cancel_event": threading.Event()})(); sent.clear()
    w._remove_object_last_image()
    assert not sent                      # no picture: nothing sent, the user is told
    # A window destroyed after QApplication is a native crash at exit (0xC0000409)
    # with the test already green: close it and run the deferred delete now.
    w.ctx = None; w.close(); w.deleteLater(); app.processEvents()


if __name__ == "__main__":
    test_buttons_send_prefixed_request(); print("ok")
