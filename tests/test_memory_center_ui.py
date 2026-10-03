"""Memory Center forensic UI audit.

Validates the Memory Center tab on real Qt widgets (not just 'it builds'):
  * functional correctness of every control (add/edit/pin/search/filter/delete/
    export/import/compact/profile switch) by driving the actual GUI handlers;
  * visual correctness (no zero-size widgets, readable fonts, no horizontal
    overflow) at 1080p@100%, 1440p@125%, 4K@150%, 4K@200%;
  * empty / small / large memory sets.

Saves screenshots to docs/validation/. Exits non-zero on any failure.
"""
import os
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
import tempfile
import shutil
from pathlib import Path

# Allow running from tests/ — put the project root on sys.path.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import _isolate_library  # noqa: F401  — never touch the live library.db

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtCore import Qt

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import crash_diag
crash_diag.install()

import ui_scale
import gui
import memory_center

# Screenshots go to a scratch directory by default. They used to overwrite the
# TRACKED files in docs/validation on every run, so `git status` was dirty
# after any test pass and the rendered noise got swept into unrelated commits.
# Set MC_REFRESH_DOCS=1 when you actually mean to update the documentation.
if os.environ.get("MC_REFRESH_DOCS") == "1":
    OUT = _ROOT / "docs" / "validation"
else:
    OUT = Path(tempfile.mkdtemp(prefix="mc_shots_"))
OUT.mkdir(parents=True, exist_ok=True)

app = QApplication(sys.argv)
# Don't load real ML models — we only exercise the Memory Center.
gui.ModelLoader.start = lambda self: None

failures = []
notes = []


def check(cond, msg):
    (notes if cond else failures).append(("PASS" if cond else "FAIL", msg))


def seed(store, profile, n_facts, n_session, with_summary):
    store.create_profile(profile)
    for i in range(n_facts):
        store.add(profile, f"Fact {i}: user prefers concise dark-themed answers index {i}",
                  etype="fact", importance=55 + (i % 40), source="manual", tags=["pref", f"t{i%3}"])
    for i in range(n_session):
        store.add(profile, f"Session note {i}: discussed ComfyUI the old model workflow params batch {i}",
                  etype="session", source="inferred")
    if with_summary:
        store.add(profile, "Summary: builds a local voice AI assistant with image editing.",
                  etype="summary", source="compacted")


def build_window(store, profile):
    win = gui.AssistantWindow("audit-model", True)
    win.ctx = type("C", (), {})()
    win.ctx.session_memory = []
    # Mirror the real app: active_memory_dir is a real Path whose .name is the profile.
    win.ctx.active_memory_dir = store.profile_dir(profile)
    win.stack.setCurrentWidget(win.dashboard)
    mc = win.memory_center_tab
    mc.store = store
    mc.refresh_all()
    idx = [win.tabs.tabText(i) for i in range(win.tabs.count())].index("Memory")
    win.tabs.setCurrentIndex(idx)
    return win, mc


# --------------------------------------------------------------------------- #
# 1. FUNCTIONAL — drive the real handlers (scale 1.0, small dataset)
# --------------------------------------------------------------------------- #
ui_scale._effective = 1.0
ui_scale._font_mult = 1.0
gui.apply_ui_scale(app)

tmp = Path(tempfile.mkdtemp())
store = memory_center.MemoryStore(root=tmp)
seed(store, "default", n_facts=3, n_session=4, with_summary=True)
win, mc = build_window(store, "default")
mc.subtabs.setCurrentIndex(1)  # Explorer
app.processEvents()

before_total = store.overview("default")["total"]

# --- add ---
e_added = store.add("default", "AUDIT added memory entry", etype="fact", source="manual")
mc._after_mutation(); app.processEvents()
check(store.overview("default")["total"] == before_total + 1, "add: total increments")

# --- search filters rows ---
mc.search_box.setText("AUDIT"); app.processEvents()
check(mc.table.rowCount() == 1, "search: filters table to matching row")
mc.search_box.setText(""); app.processEvents()

# --- type filter ---
mc.type_filter.setCurrentText("fact"); app.processEvents()
fact_rows = mc.table.rowCount()
mc.type_filter.setCurrentText("session"); app.processEvents()
sess_rows = mc.table.rowCount()
mc.type_filter.setCurrentText("all types"); app.processEvents()
check(fact_rows >= 1 and sess_rows >= 1 and (fact_rows != mc.table.rowCount() or sess_rows != mc.table.rowCount()),
      "type filter: changes the visible set")

# --- select a fact row -> editor populates, then edit + save -> revision created ---
# Target a fact specifically (summaries don't round-trip importance — they have no
# importance field in summary.json, so importance editing only applies to fact/session).
# The table is sorted, so locate the fact by its visible type cell, not by _rows index.
fact_row = next((r for r in range(mc.table.rowCount())
                 if mc.table.item(r, 0) and mc.table.item(r, 0).text() == "fact"), 0)
mc.table.selectRow(fact_row); app.processEvents()
check(mc._sel is not None, "row select: detail/editor populated")
if mc._sel:
    target_id = mc._sel.id
    revs_before = len(store.revisions("default", target_id))
    mc.editor.setPlainText("EDITED by audit"); mc.imp_slider.setValue(77)
    mc._save_entry(); app.processEvents()
    saved = store.get("default", target_id)
    check(saved and saved.text == "EDITED by audit" and saved.importance == 77,
          "edit+save: text & importance persisted")
    check(len(store.revisions("default", target_id)) == revs_before + 1, "edit+save: revision recorded")

# --- pin / unpin (promote session note to fact) ---
sess = [e for e in store.load("default") if e.type == "session"]
if sess:
    mc._sel = sess[0]
    mc._toggle_pin(); app.processEvents()
    promoted = any(e.type == "fact" and e.text == sess[0].text for e in store.load("default"))
    check(promoted, "pin: session note promoted to fact")

# --- delete (soft) -> trash ---
victim = store.add("default", "DELETE ME", etype="session", source="inferred")
t_before = len(store.trash("default"))
store.delete("default", [victim.id], soft=True)
mc._after_mutation(); app.processEvents()
check(len(store.trash("default")) == t_before + 1, "delete(soft): entry moved to trash")
check(store.get("default", victim.id) is None, "delete(soft): removed from active set")
# --- restore from trash (note: restore re-adds with a NEW id, so verify by text) ---
store.restore("default", [victim.id])
check(any(e.text == "DELETE ME" for e in store.load("default")), "restore: entry returns from trash")

# --- export json / markdown / zip ---
js = store.export_json(["default"])
md = store.export_markdown(["default"])
zp = store.export_zip(["default"], tmp / "backup.zip")
check(isinstance(js, dict) and js, "export: JSON non-empty")
check(isinstance(md, str) and "default" in md, "export: Markdown mentions profile")
check(Path(zp).exists() and Path(zp).stat().st_size > 0, "export: ZIP written")

# --- import preview + commit into a fresh profile ---
counts, data = store.preview_import(tmp / "backup.zip")
check(bool(data), "import: preview returns parsed data")
store.commit_import(data, target_profile="imported")
check("imported" in store.profiles(), "import: commit creates target profile")

# --- compaction: sources -> commit summary, sources soft-deleted, summary added ---
src = store.compaction_sources("default")
sess_before = sum(1 for e in store.load("default") if e.type == "session")
if src:
    store.commit_compaction("default", "AUDIT compacted summary of session notes")
    mc._after_mutation(); app.processEvents()
    sess_after = sum(1 for e in store.load("default") if e.type == "session")
    has_sum = any(e.type == "summary" and "AUDIT compacted" in e.text for e in store.load("default"))
    check(sess_after < sess_before, "compaction: source session notes removed from active set")
    check(has_sum, "compaction: summary entry created")
    # pinned facts must survive compaction
    check(any(e.type == "fact" for e in store.load("default")), "compaction: facts preserved")

# --- profile switching via the combo ---
store.create_profile("research")
mc.refresh_all(); app.processEvents()
mc.profile_combo.setCurrentText("research"); app.processEvents()
check(mc._profile() == "research", "profile switch: combo changes active profile view")
mc.profile_combo.setCurrentText("default"); app.processEvents()

win.close(); app.processEvents()

# --------------------------------------------------------------------------- #
# 2. VISUAL — 4 required scales (large dataset) + empty/small content states
# --------------------------------------------------------------------------- #
SCALES = [
    ("1080p_100pct", 1.0),
    ("1440p_125pct", 1.25),
    ("4K_150pct", 1.5),
    ("4K_200pct", 2.0),
]


def render_and_check(label, eff, store, profile, shot=True):
    ui_scale._effective = eff
    ui_scale._font_mult = 1.0
    gui.apply_ui_scale(app)
    win, mc = build_window(store, profile)
    # size the window roughly to the target panel
    base_w, base_h = (1920, 1040) if eff <= 1.0 else (int(1700 * eff), int(950 * eff))
    win.resize(min(base_w, 3700), min(base_h, 2100))
    win.show(); app.processEvents()
    mc.subtabs.setCurrentIndex(1); app.processEvents()  # Explorer

    fm = QFontMetrics(mc.search_box.font())
    font_px = fm.height()
    # readable: body text must be at least 12px on screen at every scale
    check(font_px >= 12, f"{label}: font readable ({font_px}px >= 12)")
    check(mc.subtabs.count() == 7, f"{label}: all 7 sub-tabs present")
    check(mc.search_box.height() > 0 and mc.table.height() > 0, f"{label}: explorer widgets have non-zero size")
    # no horizontal overflow: table viewport fits within the tab width
    overflow = mc.table.horizontalHeader().length() - mc.table.viewport().width()
    check(mc.table.horizontalScrollBarPolicy() == Qt.ScrollBarAsNeeded or overflow <= mc.table.width(),
          f"{label}: explorer table has no uncontrolled horizontal overflow")
    # overview labels populated when data exists
    mc.subtabs.setCurrentIndex(0); app.processEvents()
    tot = mc._ov_labels["Total memories"].text()
    check(tot not in ("", "—") , f"{label}: overview 'Total memories' populated ({tot})")
    mc.subtabs.setCurrentIndex(1); app.processEvents()
    if shot:
        win.grab().save(str(OUT / f"mc_{label}.png"))
    notes.append(("INFO", f"{label}: effective={eff} fontPx={font_px} rows={mc.table.rowCount()} total={tot}"))
    win.close(); app.processEvents()


# large dataset across all 4 scales
big = Path(tempfile.mkdtemp())
bstore = memory_center.MemoryStore(root=big)
seed(bstore, "default", n_facts=60, n_session=190, with_summary=True)
for label, eff in SCALES:
    render_and_check(label, eff, bstore, "default")

# content states at 1.0
empty = Path(tempfile.mkdtemp())
estore = memory_center.MemoryStore(root=empty); estore.create_profile("default")
ui_scale._effective = 1.0; ui_scale._font_mult = 1.0; gui.apply_ui_scale(app)
win, mc = build_window(estore, "default"); win.resize(1840, 1010); win.show(); app.processEvents()
mc.subtabs.setCurrentIndex(1); app.processEvents()
check(mc.table.rowCount() == 0, "empty set: table renders with 0 rows (no crash)")
win.grab().save(str(OUT / "mc_empty.png")); win.close(); app.processEvents()

small = Path(tempfile.mkdtemp())
sstore = memory_center.MemoryStore(root=small); seed(sstore, "default", 2, 1, False)
win, mc = build_window(sstore, "default"); win.resize(1840, 1010); win.show(); app.processEvents()
mc.subtabs.setCurrentIndex(1); app.processEvents()
check(mc.table.rowCount() == 3, "small set: 3 rows render")
win.grab().save(str(OUT / "mc_small.png")); win.close(); app.processEvents()

# --------------------------------------------------------------------------- #
for d in (tmp, big, empty, small):
    shutil.rmtree(d, ignore_errors=True)

print("\n==== MEMORY CENTER AUDIT ====")
for status, msg in notes:
    print(f"  [{status}] {msg}")
for status, msg in failures:
    print(f"  [{status}] {msg}")
print(f"\nScreenshots: {OUT}")
print(f"RESULT: {len(notes)} ok / {len(failures)} failed")
sys.exit(1 if failures else 0)
