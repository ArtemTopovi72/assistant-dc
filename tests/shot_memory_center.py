"""Render the redesigned Memory Center on the REAL Windows platform (not offscreen)
so text actually rasterizes — for visual inspection of the redesign.

Saves docs/validation/real_mc_<scale>.png at the 4 required scales plus a narrow
width (to prove responsive column hiding + button reflow). Run with the venv
python on a desktop session.
"""
import os, sys, tempfile, shutil
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
# IMPORTANT: do NOT force offscreen — we want real text rendering.
os.environ.pop("QT_QPA_PLATFORM", None)

from PyQt5.QtWidgets import QApplication
import ui_scale, gui, memory_center, crash_diag
crash_diag.install()

OUT = _ROOT / "docs" / "validation"
OUT.mkdir(parents=True, exist_ok=True)
app = QApplication(sys.argv)
gui.ModelLoader.start = lambda self: None


def seed(store, profile, nf, ns):
    store.create_profile(profile)
    samples = [
        "User prefers concise, dark-themed answers and minimal preamble.",
        "Hardware: 12GB RTX 3060 shared by the LLM, Whisper and F5-TTS.",
        "Research crawler caches extracted pages under memory/research/_cache.",
        "Image edits must preserve original resolution unless resize is requested.",
        "FireRed edit workflow downscaled to 1MP; now rescaled back to source.",
    ]
    for i in range(nf):
        store.add(profile, samples[i % len(samples)] + f" (note {i})",
                  etype="fact", importance=40 + (i * 7) % 60, source="manual",
                  tags=["pref", f"topic{i%4}"])
    for i in range(ns):
        store.add(profile, f"Session: discussed ComfyUI the old model workflow params, batch {i}",
                  etype="session", source="inferred")
    store.add(profile, "Summary: local voice AI assistant with image editing.",
              etype="summary", source="compacted")


def build(store, profile):
    win = gui.AssistantWindow("shot-model", True)
    win.ctx = type("C", (), {})()
    win.ctx.session_memory = []
    win.ctx.cancel_event = type("E", (), {"set": lambda self: None})()
    win.ctx.active_memory_dir = store.profile_dir(profile)
    win.stack.setCurrentWidget(win.dashboard)
    mc = win.memory_center_tab
    mc.store = store
    mc.refresh_all()
    idx = [win.tabs.tabText(i) for i in range(win.tabs.count())].index("Memory")
    win.tabs.setCurrentIndex(idx)
    mc.subtabs.setCurrentIndex(1)  # Explorer
    return win, mc


tmp = Path(tempfile.mkdtemp())
store = memory_center.MemoryStore(root=tmp)
seed(store, "default", nf=24, ns=30)

shots = [
    ("real_mc_1080p_100pct", 1.0, 1840, 1010),
    ("real_mc_1440p_125pct", 1.25, 2300, 1280),
    ("real_mc_4K_150pct", 1.5, 2800, 1560),
    ("real_mc_4K_200pct", 2.0, 3400, 1900),
    ("real_mc_narrow", 1.0, 620, 1000),   # stress: reflow + column hiding
]
for label, eff, ww, hh in shots:
    ui_scale._effective = eff; ui_scale._font_mult = 1.0
    gui.apply_ui_scale(app)
    win, mc = build(store, "default")
    win.resize(min(ww, 3600), min(hh, 2000))
    win.show(); app.processEvents(); app.processEvents()
    # select a row so the editor populates for the shot
    if mc.table.rowCount() > 1:
        mc.table.selectRow(1); app.processEvents()
    win.grab().save(str(OUT / f"{label}.png"))
    # also capture the Overview sub-tab (profile header + stats + actions)
    mc.subtabs.setCurrentIndex(0); app.processEvents()
    win.grab().save(str(OUT / f"{label}_overview.png"))
    mc.subtabs.setCurrentIndex(1); app.processEvents()
    cols = [c for c in range(mc.table.columnCount()) if not mc.table.isColumnHidden(c)]
    print(f"{label}: {ww}x{hh} eff={eff} rows={mc.table.rowCount()} visible_cols={cols}")
    win.close(); app.processEvents()

shutil.rmtree(tmp, ignore_errors=True)
print("saved to", OUT)
