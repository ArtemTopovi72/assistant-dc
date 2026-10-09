"""Screenshot every tab of the main window on the real Windows platform (text rasterizes),
for visual review of a GUI change. No model is loaded.

Run: venv/Scripts/python tests/shot_gui_tabs.py <out_dir> [width height]
"""
import os, sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.pop("QT_QPA_PLATFORM", None)
os.environ.setdefault("F5_TEST_RUN", "1")

from PyQt5.QtWidgets import QApplication, QTabWidget
import ui_scale, gui

out = Path(sys.argv[1] if len(sys.argv) > 1 else "runtime/gui_shots")
out.mkdir(parents=True, exist_ok=True)
w, h = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (1600, 950)
app = QApplication(sys.argv)
gui.ModelLoader.start = lambda self: None
import gui_i18n
gui_i18n.install(os.environ.get("SHOT_LANG", "en"))
ui_scale._effective = 1.0; ui_scale._font_mult = 1.0
gui.apply_ui_scale(app)
win = gui.AssistantWindow("shot-model", True)
win.stack.setCurrentWidget(win.dashboard)
win.resize(w, h); win.show()
for _ in range(3): app.processEvents()


def snap(name):
    for _ in range(3): app.processEvents()
    win.grab().save(str(out / f"{name}.png"))


snap("00_main")
seen = set()
for tabs in win.findChildren(QTabWidget):
    if not tabs.isVisible() and tabs.parent() is not None and tabs.window() is not win:
        continue
    for i in range(tabs.count()):
        label = tabs.tabText(i).strip()
        key = (id(tabs), label)
        if key in seen:
            continue
        seen.add(key)
        tabs.setCurrentIndex(i)
        if not tabs.isVisible():
            continue
        safe = "".join(c for c in label if c.isalnum() or c in " _-").strip().replace(" ", "_")[:30] or f"tab{i}"
        snap(f"{len(seen):02d}_{safe}")
from gui_settings_dialog import SettingsDialog
dlg = SettingsDialog("shot-model", ctx=None, parent=win)
dlg.resize(1000, 800); dlg.show()
for t in dlg.findChildren(QTabWidget):
    for i in range(t.count()):
        t.setCurrentIndex(i)
        for _ in range(3): app.processEvents()
        safe = "".join(c for c in t.tabText(i) if c.isalnum() or c in " _-").strip().replace(" ", "_")[:30]
        dlg.grab().save(str(out / f"settings_{i:02d}_{safe}.png"))
if not dlg.findChildren(QTabWidget):
    for _ in range(3): app.processEvents()
    dlg.grab().save(str(out / "settings.png"))
dlg.close()
print("saved", len(list(out.glob('*.png'))), "shots to", out)
