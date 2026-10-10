"""Drive the discrete tab/dialog classes in gui.py: MaskCanvas, MaskDrawDialog,
StressTab, TransferTab, TransferWorker, and MemoryCenterTab's dialog/action
methods. Blocking dialogs (exec_, QFileDialog, QInputDialog, QMessageBox) are
monkeypatched; heavy image ops are faked in sys.modules.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_tabs.py
"""
import os, sys, types, tempfile, shutil, contextlib
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("USE_GUI", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _isolate_library  # noqa: F401  — never touch the live library.db
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

from PyQt5.QtWidgets import (QApplication, QDialog, QFileDialog, QInputDialog,
                             QMessageBox, QWidget)
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtCore import Qt, QPoint
import gui

_app = QApplication.instance() or QApplication(sys.argv)
_TMP = tempfile.mkdtemp(prefix="guitabs_")

def _png(name="t.png", w=80, h=60, col="#4488cc"):
    p = os.path.join(_TMP, name)
    img = QImage(w, h, QImage.Format_RGB32); img.fill(QColor(col)); img.save(p)
    return p

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    assert cond, f"{name}: {detail}"


@contextlib.contextmanager
def fake_modules(**mods):
    saved = {}
    for name, mod in mods.items():
        saved[name] = sys.modules.get(name)
        m = types.ModuleType(name)
        for k, v in mod.items(): setattr(m, k, v)
        sys.modules[name] = m
    try: yield
    finally:
        for name, old in saved.items():
            if old is None: sys.modules.pop(name, None)
            else: sys.modules[name] = old

def cap(worker, *signals):
    out = {s: [] for s in signals}
    for s in signals:
        getattr(worker, s).connect(lambda *a, _s=s: out[_s].append(a if len(a) != 1 else a[0]))
    return out


# ------------------------------------------------------------------ MaskCanvas

def test_mask_canvas():
    mc = gui.MaskCanvas(_png("mc.png"), max_side=64)
    check("mask_empty_initial", mc.is_empty())
    check("mask_export_empty_none", mc.export_mask(os.path.join(_TMP, "m0.png")) is None)
    mc.stroke_at(20, 20); mc.stroke_at(30, 30)   # continuous stroke (interp branch)
    check("mask_not_empty", not mc.is_empty())
    out = mc.export_mask(os.path.join(_TMP, "m1.png"))
    check("mask_export_path", out and os.path.exists(out))
    mc.erase_at(20, 20)                            # erase branch
    mc.clear(); check("mask_cleared", mc.is_empty())
    # mouse events
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtCore import QEvent, QPointF
    ev = QMouseEvent(QEvent.MouseButtonPress, QPointF(10, 10), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    mc.mousePressEvent(ev)
    mv = QMouseEvent(QEvent.MouseMove, QPointF(15, 15), Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
    mc.mouseMoveEvent(mv)
    rel = QMouseEvent(QEvent.MouseButtonRelease, QPointF(15, 15), Qt.LeftButton, Qt.NoButton, Qt.NoModifier)
    mc.mouseReleaseEvent(rel)
    mc.repaint()
    check("mask_mouse_driven", True)


def test_mask_draw_dialog():
    img = _png("md.png")
    dlg = gui.MaskDrawDialog(img, title="Test mask")
    dlg.canvas.stroke_at(30, 20)
    orig = QDialog.accept; QDialog.accept = lambda self: None
    try:
        dlg._save()
    finally:
        QDialog.accept = orig
    check("maskdlg_saved", dlg.mask_path is not None and os.path.exists(dlg.mask_path))
    check("maskdlg_protect_face", dlg.protect_face is True)
    # brush slider drives brush size
    dlg.bslider.setValue(30); check("maskdlg_brush", dlg.canvas.brush == 30)
    # preload an existing mask
    existing = dlg.mask_path
    dlg2 = gui.MaskDrawDialog(img, existing_mask=existing, title="Reload")
    check("maskdlg_preload", not dlg2.canvas.is_empty())
    # empty-save -> mask_path None
    dlg3 = gui.MaskDrawDialog(img)
    QDialog.accept = lambda self: None
    try: dlg3._save()
    finally: QDialog.accept = orig
    check("maskdlg_empty_save", dlg3.mask_path is None)
    check("output_dir_gui_mask", gui.OUTPUT_DIR_GUI_MASK().exists())


# ------------------------------------------------------------------ StressTab

class _Host(types.SimpleNamespace):
    pass

def test_stress_tab():
    host = _Host(ctx=types.SimpleNamespace(models=types.SimpleNamespace(accentor_loaded=False)))
    tab = gui.StressTab(host)
    # add valid
    tab.add_in.setText("звон+ит"); tab._add()
    check("stress_added", tab.table.rowCount() >= 1)
    # add without plus -> warning (stub QMessageBox.warning)
    warned = {"n": 0}
    orig = QMessageBox.warning; QMessageBox.warning = staticmethod(lambda *a, **k: warned.__setitem__("n", warned["n"]+1))
    try:
        tab.add_in.setText("noplus"); tab._add()
        check("stress_noplus_warned", warned["n"] == 1)
        tab.add_in.setText(""); tab._add()   # empty -> no-op, no warning
        check("stress_empty_noop", warned["n"] == 1)
    finally:
        QMessageBox.warning = orig
    # preview (overrides-only path since accentor not loaded)
    tab.preview_in.setText("звонит утром"); tab._preview()
    check("stress_preview", tab.preview_out.text() != "")
    tab.preview_in.setText(""); tab._preview()
    check("stress_preview_empty", tab.preview_out.text() == "")
    # edit selected + remove selected
    tab.table.selectRow(0)
    it = tab.table.item(0, 1)
    if it: tab._edit_selected(it)
    tab.table.selectRow(0); tab._remove_selected()
    check("stress_removed", True)
    tab._reload(); check("stress_reload", True)
    # preview error path + accentor-loaded path
    host.ctx.models.accentor_loaded = True
    with fake_modules(audio={"stress_plus": lambda ctx, txt: "зв+онит"}):
        tab.preview_in.setText("звонит"); tab._preview()
    check("stress_preview_accentor", "+" in tab.preview_out.text())
    with fake_modules(audio={"stress_plus": lambda ctx, txt: (_ for _ in ()).throw(RuntimeError("boom"))}):
        tab.preview_in.setText("x"); tab._preview()
    check("stress_preview_error", "error" in tab.preview_out.text())


# ----------------------------------------------------------------- WeatherTab

def test_weather_tab():
    import datetime as _dt
    import weather as _w

    host = _Host(ctx=types.SimpleNamespace(lang="ru"))
    tab = gui.WeatherTab(host)

    # An empty city is a no-op with a nudge, not a lookup for "".
    tab.city_in.setText("  "); tab._lookup(hours=24)
    check("weather_empty_city_noop",
          tab.worker is None and "city" in tab.status.text().lower(),
          tab.status.text())

    d0, d1 = _dt.date(2026, 8, 17), _dt.date(2026, 8, 18)
    def bk(date, period, t, h, wind, code):
        return {"date": date, "period": period, "temp_c": t, "humidity": h,
                "wind_kmh": wind, "code": code}
    buckets = [bk(d0, "afternoon", 21, 53, 16, 3), bk(d0, "evening", 17, 76, 19, 3),
               bk(d1, "morning", 13, None, 6, 66), bk(d1, "afternoon", 17, 67, 6, 3)]

    # The worker does the network + LLM work; drive it synchronously the way the
    # other worker tests in this file do.
    with fake_modules(weather={
        "resolve_city": lambda city, correct_fn=None: {
            "lat": 1, "lon": 2, "name": "Bezhanitsy", "country": "Russia"},
        "forecast_data": lambda loc, lang, hours=None, date=None, advise_fn=None: (
            buckets, {}, ""),
        "periods_advise_fn": lambda ctx: None,
        "city_correct_fn": lambda ctx: None,
        "not_found_message": _w.not_found_message,
    }):
        wk = gui.WeatherWorker(host.ctx, "Bezhanitsy", "ru", hours=24)
        caps = cap(wk, "done", "failed")
        wk.run()
    check("weather_worker_done", len(caps["done"]) == 1, caps)
    tab._on_done(*caps["done"][0])

    check("weather_table_row_per_bucket", tab.table.rowCount() == len(buckets),
          tab.table.rowCount())
    check("weather_status_names_place", "Bezhanitsy" in tab.status.text(),
          tab.status.text())
    # The date is printed once per day, not on every row -- same de-duplication
    # the chat table does with its group lines.
    dates = [tab.table.item(r, 0).text() for r in range(tab.table.rowCount())]
    check("weather_date_shown_once_per_day",
          [d.split(" ")[0] for d in dates] == ["17.08", "", "18.08", ""], dates)
    check("weather_missing_humidity_is_a_dash",
          tab.table.item(2, 3).text() == "—", tab.table.item(2, 3).text())
    check("weather_humidity_is_a_percentage",
          tab.table.item(0, 3).text() == "53%", tab.table.item(0, 3).text())
    # The desktop table has the width for the words, so unlike the chat block it
    # carries the description in the row rather than a legend underneath.
    conditions = tab.table.item(0, 5).text()
    check("weather_conditions_spell_out_the_description",
          _w._code_desc(3, "en") in conditions and _w._code_icon(3) in conditions,
          conditions)
    # Freezing rain must not read as ordinary rain: it gets the ice icon.
    check("weather_freezing_rain_gets_the_ice_icon",
          _w._code_icon(66) in tab.table.item(2, 5).text(),
          tab.table.item(2, 5).text())

    advice = tab.advice.text()
    check("weather_advice_has_a_heading", "👕" in advice, advice)
    check("weather_advice_drops_the_verb", "надень" not in advice.lower(), advice)
    # Two consecutive buckets share the 17-21C band, so their advice is stated
    # ONCE covering both -- via weather.group_advice, the same grouping the chat
    # reply uses.
    groups = _w.group_advice(buckets, {}, "ru")
    check("weather_advice_grouped_not_repeated",
          len([l for l in advice.splitlines() if l and "👕" not in l]) == len(groups),
          advice)

    # A city that cannot be geocoded comes back as the translated not-found
    # message, and must clear the previous result rather than leave stale rows.
    with fake_modules(weather={
        "resolve_city": lambda city, correct_fn=None: None,
        "forecast_data": lambda *a, **k: ([], {}, ""),
        "periods_advise_fn": lambda ctx: None,
        "city_correct_fn": lambda ctx: None,
        "not_found_message": _w.not_found_message,
    }):
        wk2 = gui.WeatherWorker(host.ctx, "Nowhereville", "ru", hours=24)
        caps2 = cap(wk2, "done", "failed")
        wk2.run()
    tab._on_done(*caps2["done"][0])
    check("weather_unknown_city_message", "Не нашёл" in tab.status.text(),
          tab.status.text())
    check("weather_unknown_city_clears_the_table", tab.table.rowCount() == 0)
    check("weather_unknown_city_clears_the_advice", tab.advice.text() == "")

    # No model loaded is not an error: the forecast still renders, on the fixed
    # temperature bands, and the worker must not try to build an LLM hook.
    asked = {"advise": 0, "correct": 0}
    with fake_modules(weather={
        "resolve_city": lambda city, correct_fn=None: {
            "lat": 1, "lon": 2, "name": "Bezhanitsy", "country": ""},
        "forecast_data": lambda loc, lang, hours=None, date=None, advise_fn=None: (
            buckets, {}, ""),
        "periods_advise_fn": lambda ctx: asked.__setitem__("advise", asked["advise"] + 1),
        "city_correct_fn": lambda ctx: asked.__setitem__("correct", asked["correct"] + 1),
        "not_found_message": _w.not_found_message,
    }):
        wk3 = gui.WeatherWorker(None, "Bezhanitsy", "ru", hours=24)
        caps3 = cap(wk3, "done", "failed")
        wk3.run()
    check("weather_no_ctx_still_returns_a_forecast",
          len(caps3["done"]) == 1 and caps3["done"][0][1], caps3)
    check("weather_no_ctx_skips_the_llm_hooks", asked == {"advise": 0, "correct": 0},
          asked)

    # A raising lookup surfaces as a status line, never a crash -- run() is on a
    # worker thread, where an escaping exception aborts the process.
    with fake_modules(weather={
        "resolve_city": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network down")),
        "forecast_data": lambda *a, **k: ([], {}, ""),
        "periods_advise_fn": lambda ctx: None,
        "city_correct_fn": lambda ctx: None,
        "not_found_message": _w.not_found_message,
    }):
        wk4 = gui.WeatherWorker(host.ctx, "Bezhanitsy", "ru", hours=24)
        caps4 = cap(wk4, "done", "failed")
        wk4.run()
    check("weather_worker_failure_is_reported_not_raised",
          len(caps4["failed"]) == 1 and not caps4["done"], caps4)
    tab._on_failed(caps4["failed"][0])
    check("weather_failure_status", "failed" in tab.status.text().lower(),
          tab.status.text())

    # Buttons are re-enabled after every lookup, however it ended -- a disabled
    # keyboard after one failure would strand the tab.
    tab._on_finished()
    check("weather_buttons_reenabled", all(b.isEnabled() for b in tab._buttons))
    check("weather_worker_released", tab.worker is None)


# ------------------------------------------------------------------ MusicTab

def test_music_tab():
    host = _Host(ctx=types.SimpleNamespace(lang="ru"))
    tab = gui.MusicTab(host)
    check("music_play_disabled_initial", not tab.play_btn.isEnabled())

    # empty topic -> no-op, no worker spawned
    tab.topic_in.setText(""); tab._generate()
    check("music_empty_noop", tab.music_worker is None)

    # no ctx -> status message, no worker spawned
    host2 = _Host(ctx=None)
    tab_noctx = gui.MusicTab(host2)
    tab_noctx.topic_in.setText("a song about rain"); tab_noctx._generate()
    # Re-anchored: "No model loaded yet — load one first." conflated two states
    # once the app grew a "start without a model" option. ctx is None means the
    # runtime is still coming up (waiting fixes it); a ctx with an empty
    # model_name means no model was loaded ON PURPOSE (only Settings fixes it),
    # and each says so separately now. What is pinned is that BOTH speak.
    check("music_no_ctx", bool(tab_noctx.status.text().strip())
          and tab_noctx.music_worker is None)
    import types as _types
    tab_nomodel = gui.MusicTab(_Host(ctx=_types.SimpleNamespace(model_name="")))
    tab_nomodel.topic_in.setText("a song about rain"); tab_nomodel._generate()
    check("music_no_model_points_at_settings",
          "Settings" in tab_nomodel.status.text()
          and tab_nomodel.music_worker is None)

    # success path: stub music.build_structured_caption/generate_music, run the
    # worker synchronously (as the other worker tests in this file do) and drive
    # the tab's own signal handlers.
    wav_path = os.path.join(_TMP, "song.wav")
    open(wav_path, "wb").close()
    with fake_modules(music={
        "build_structured_caption": lambda ctx, topic, lang, **kw: {"lyrics": "la la", "style": "pop"},
        "generate_music": lambda ctx, lyrics, style, **kw: wav_path,
    }):
        w = gui.MusicWorker(host.ctx, "a happy tune", "ru")
        caps = cap(w, "done", "failed")
        w.run()
    check("music_worker_done", len(caps["done"]) == 1 and caps["done"][0][0] == wav_path)
    tab._on_done(*caps["done"][0])
    check("music_status_done", "Done" in tab.status.text())
    check("music_play_enabled", tab.play_btn.isEnabled())

    # failure path: MusicUnavailable (or any stub NotImplementedError) surfaces
    # as a status message, not a crash.
    import music as _music_mod
    with fake_modules(music={
        "build_structured_caption": lambda ctx, topic, lang, **kw: (_ for _ in ()).throw(
            _music_mod.MusicUnavailable("engine down")),
        "generate_music": lambda *a, **k: "",
    }):
        w2 = gui.MusicWorker(host.ctx, "a sad tune", "ru")
        caps2 = cap(w2, "done", "failed")
        w2.run()
    check("music_worker_failed", len(caps2["failed"]) == 1)
    tab._on_failed(caps2["failed"][0])
    check("music_status_failed", "unavailable" in tab.status.text().lower())

    # ── the settings pickers ────────────────────────────────────────────────
    # A picker whose value never reaches the prompt is a lie told with a
    # dropdown, so these follow the selection all the way into the worker's
    # arguments rather than stopping at "the widget exists".
    import music as _music_mod2
    check("music_has_pickers",
          set(tab._combos) == {"genre", "tempo", "vocal"} and tab.dur_combo is not None,
          list(tab._combos))
    check("music_pickers_offer_every_engine_value",
          tab._combos["genre"].count() == len(_music_mod2.GENRES)
          and tab._combos["vocal"].count() == len(_music_mod2.VOCALS),
          (tab._combos["genre"].count(), len(_music_mod2.GENRES)))
    # Steps sit next to Quality: default 20, range 20-50, and the ETA line
    # quotes every preset at the chosen count (the same numbers as Telegram).
    check("music_steps_default_20", tab.steps_spin.value() == _music_mod2.MUSIC_STEPS == 20
          and (tab.steps_spin.minimum(), tab.steps_spin.maximum()) == (20, 50))
    eta20 = tab.eta_lbl.text()
    tab.steps_spin.setValue(50)
    eta50 = tab.eta_lbl.text()
    check("music_eta_line_follows_steps",
          "Fast" in eta20 and "Max" in eta20 and eta20 != eta50
          and _music_mod2.eta_label(_music_mod2.eta_seconds("fast", steps=50), "en") in eta50, (eta20, eta50))
    tab.qual_combo.setCurrentIndex(list(_music_mod2.WEIGHT_PRESETS).index("max"))
    check("music_eta_line_bolds_the_chosen_preset", "<b>Max" in tab.eta_lbl.text(), tab.eta_lbl.text())
    tab.qual_combo.setCurrentIndex(list(_music_mod2.WEIGHT_PRESETS).index(_music_mod2.DEFAULT_PRESET))
    tab.steps_spin.setValue(20)
    # Auto is the default, and Auto must mean "no opinion" -- not an empty
    # instruction handed to the songwriter.
    prefs0, dur0, preset0 = tab._settings()
    check("music_defaults_to_auto", prefs0 == {}, prefs0)
    check("music_default_duration_is_real", dur0 in _music_mod2.DURATIONS, dur0)

    def _pick(box, key):
        box.setCurrentIndex([box.itemData(i) for i in range(box.count())].index(key))

    _pick(tab._combos["genre"], "jazz")
    _pick(tab._combos["tempo"], "slow")
    _pick(tab._combos["vocal"], "male")
    _pick(tab.dur_combo, 180)
    prefs1, dur1, preset1 = tab._settings()
    check("music_prefs_resolved_through_engine",
          "jazz" in prefs1.get("genre", "").lower()
          and "65-75" in prefs1.get("tempo", "")
          and "MALE" in prefs1.get("vocal", ""), prefs1)
    check("music_duration_read_from_picker", dur1 == 180, dur1)

    # Instrumental is a different shape, not a kind of voice.
    _pick(tab._combos["vocal"], "instrumental")
    prefs2, _, _ = tab._settings()
    check("music_instrumental_is_its_own_flag",
          prefs2.get("instrumental") is True and "vocal" not in prefs2, prefs2)
    _pick(tab._combos["vocal"], "male")

    # ...and the choices actually reach the generation call.
    seen = {}
    with fake_modules(music={
        "build_structured_caption": lambda ctx, topic, lang, **kw: (
            seen.update(cap_kw=kw), {"lyrics": "la", "style": "pop"})[-1],
        "generate_music": lambda ctx, lyrics, style, **kw: (
            seen.update(gen_kw=kw), wav_path)[-1],
    }):
        w3 = gui.MusicWorker(host.ctx, "a tune", "ru", prefs=prefs1, duration_s=dur1)
        w3.run()
    check("music_prefs_reach_the_songwriter",
          "jazz" in str(seen.get("cap_kw", {}).get("prefs", {})).lower(), seen.get("cap_kw"))
    check("music_duration_reaches_the_songwriter",
          seen.get("cap_kw", {}).get("duration_s") == 180, seen.get("cap_kw"))
    check("music_duration_reaches_the_render",
          seen.get("gen_kw", {}).get("duration_s") == 180, seen.get("gen_kw"))

    # An unavailable engine must surface as a clean "failed", not a crashed worker.
    #
    # This used to run the worker fully unpatched and assert it failed -- which
    # only held because nothing was installed. With ComfyUI actually up and the
    # Music3 weights present, engine_available() returns True, the worker starts
    # a REAL render (minutes of GPU) and never emits "failed", so the check went
    # red on a HEALTHY machine and made an offline suite do live work. The intent
    # was always "the unavailable path fails cleanly", so say that explicitly.
    with fake_modules(music={
        "engine_available": lambda ctx=None, preset=None: (False, "weights not downloaded"),
    }):
        w3 = gui.MusicWorker(host.ctx, "a real stub call", "ru")
        caps3 = cap(w3, "done", "failed")
        w3.run()
    check("music_unavailable_engine_fails_cleanly", len(caps3["failed"]) == 1, caps3)
    check("music_unavailable_engine_emits_no_done", len(caps3["done"]) == 0, caps3)

    # -- typed values and the bot's own picks (parity with the Telegram menu) --
    import gui_music_tab as _gmt
    _pick(tab._combos["tempo"], _gmt._CUSTOM); tab.bpm_spin.setValue(118)
    _pick(tab.dur_combo, _gmt._CUSTOM); tab.dur_spin.setValue(100)
    prefs4, dur4, _ = tab._settings()
    check("music_custom_bpm_and_length_reach_settings",
          prefs4.get("tempo") == "exactly 118 BPM" and dur4 == 100
          and tab.bpm_spin.isVisibleTo(tab) and tab.dur_spin.isVisibleTo(tab), (prefs4, dur4))
    check("music_spins_stay_inside_the_engine_range",
          (tab.bpm_spin.minimum(), tab.bpm_spin.maximum()) == tuple(_music_mod2.BPM_RANGE)
          and (tab.dur_spin.minimum(), tab.dur_spin.maximum()) == tuple(_music_mod2.DURATION_RANGE))
    _pick(tab.dur_combo, _gmt._BOT_DECIDES)
    _pick(tab._combos["tempo"], "auto")
    prefs5, dur5, _ = tab._settings()
    check("music_bot_decides_hands_the_worker_no_length", dur5 == 0 and "tempo" not in prefs5, (prefs5, dur5))
    asked = {}
    with fake_modules(music={
        "build_structured_caption": lambda ctx, topic, lang, **kw: (
            seen.update(cap_kw=kw), {"lyrics": "la", "style": "pop"})[-1],
        "generate_music": lambda ctx, lyrics, style, **kw: (
            seen.update(gen_kw=kw), wav_path)[-1],
        "choose_auto_params": lambda ctx, topic, lang, fixed=None, **want: (
            asked.update(want=want, fixed=fixed), {"bpm": 92, "seconds": 75, "why": "calm"})[-1],
        "prefs_from": _music_mod2.prefs_from,
    }):
        w5 = gui.MusicWorker(host.ctx, "a lullaby", "ru", prefs=prefs5, duration_s=dur5)
        caps5 = cap(w5, "done", "chose")
        w5.run()
    check("music_worker_asks_the_bot_for_the_blanks_only",
          asked.get("want", {}).get("tempo") is True and asked["want"].get("duration") is True
          and asked["want"].get("genre") is False and "genre" in (asked.get("fixed") or {}), asked)
    check("music_bot_picks_reach_the_brief_and_the_render",
          seen["cap_kw"]["prefs"].get("tempo") == "exactly 92 BPM" and seen["cap_kw"]["duration_s"] == 75
          and seen["gen_kw"]["duration_s"] == 75, seen)
    tab._on_chose(caps5["chose"][0]); tab._on_done(*caps5["done"][0])
    check("music_status_reports_what_the_bot_chose",
          "The bot chose:" in tab.status.text() and "92 BPM" in tab.status.text()
          and "75 s" in tab.status.text() and "calm" in tab.status.text(), tab.status.text())
    _pick(tab.dur_combo, 60)


# ------------------------------------------------------------------ TransferTab

def _tt_host():
    ctx = types.SimpleNamespace()
    import threading
    ctx.cancel_event = threading.Event()
    ctx.reference_images = []
    ctx.is_cancelled = lambda: ctx.cancel_event.is_set()
    ctx.last_image_path = None
    host = _Host(ctx=ctx,
                 images_panel=types.SimpleNamespace(add_image=lambda p: None),
                 _add_system=lambda m: None)
    return host

def test_transfer_tab_rows_and_selection():
    host = _tt_host()
    tab = gui.TransferTab(host)
    p1, p2, p3 = _png("tt1.png"), _png("tt2.png"), _png("tt3.png")
    tab._add_row(p1); tab._add_row(p2); tab._add_row(p3)
    check("tt_three_rows", len(tab._rows) == 3)
    tab._add_row(p1)                       # duplicate -> ignored
    tab._add_row("C:/no/such.png")         # missing -> ignored
    check("tt_dedup", len(tab._rows) == 3)
    # newest is target (index 0)
    check("tt_newest_target", tab._role_of(tab._rows[-1]["combo"]) == "TARGET")
    # role change enforces single target
    tab._rows[0]["combo"].setCurrentIndex(0)   # make first the target -> demotes last
    tab._on_role_changed(tab._rows[0]["combo"])
    targets = [r for r in tab._rows if tab._role_of(r["combo"]) == "TARGET"]
    check("tt_single_target", len(targets) == 1)
    # select + style
    tab._select_row(tab._rows[1]); check("tt_selected", tab._sel is tab._rows[1])
    tab._select_row(QWidget()); check("tt_select_foreign_none", tab._sel is None)
    # _set_viz with good + bad path
    tab._set_viz("target", p1); check("tt_viz_set", tab._viz["target"]._fullpath == p1)
    tab._set_viz("target", None); check("tt_viz_clear", tab._viz["target"]._fullpath is None)
    tab._set_viz("target", "C:/no/x.png"); check("tt_viz_missing", tab._viz["target"]._fullpath is None)
    # remove a row
    tab._select_row(tab._rows[0]); tab._remove_row(tab._rows[0]["frame"])
    check("tt_removed", len(tab._rows) == 2)
    tab._remove_row(QWidget())             # unknown frame -> no-op
    tab._clear(); check("tt_cleared", tab._rows == [] and tab._placeholder is not None)


def test_transfer_tab_pull_and_files():
    host = _tt_host()
    p1, p2 = _png("pl1.png"), _png("pl2.png")
    host.ctx.reference_images = [p1, p2]
    tab = gui.TransferTab(host)
    tab._pull_loaded(); check("tt_pull_loaded", len(tab._rows) == 2)
    tab._clear()
    host.ctx.reference_images = []
    tab._pull_loaded(); check("tt_pull_empty_status", "loaded" in tab.status.text().lower())
    # _add_files stubbed
    QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([p1, p2], ""))
    tab._add_files(); check("tt_add_files", len(tab._rows) == 2)


def test_transfer_tab_gather_and_run():
    host = _tt_host()
    tab = gui.TransferTab(host)
    # gather with <2 rows
    check("tt_gather_few", tab._gather() is None)
    p1, p2 = _png("g1.png"), _png("g2.png")
    tab._add_row(p1); tab._add_row(p2)
    g = tab._gather()
    check("tt_gather_ok", g is not None and len(g) == 4)
    # no ctx -> None
    host.ctx = None
    check("tt_gather_noctx", tab._gather() is None)
    host2 = _tt_host(); tab2 = gui.TransferTab(host2)
    tab2._add_row(_png("r1.png")); tab2._add_row(_png("r2.png"))
    # force both non-target to hit the "default last row as target" branch
    for r in tab2._rows:
        r["combo"].setCurrentIndex(1)   # auto-infer, no target
    g2 = tab2._gather()
    check("tt_gather_default_target", g2 is not None)
    # run in plan mode with a stubbed TransferWorker
    started = {"n": 0}
    class FakeWorker:
        def __init__(self, *a, **k): pass
        def __getattr__(self, n): return types.SimpleNamespace(connect=lambda *a, **k: None)
        def isRunning(self): return False
        def start(self): started["n"] += 1
    # TransferTab moved to gui_transfer_tab.py and resolves TransferWorker as a
    # global of THAT module, so patching gui's copy alone no longer intercepts
    # anything — the fake would go dead and this would run a real transfer.
    import gui_transfer_tab as _TT
    orig = _TT.TransferWorker
    _TT.TransferWorker = FakeWorker; gui.TransferWorker = FakeWorker
    try:
        tab2._run("plan"); check("tt_run_started", started["n"] == 1)
    finally:
        _TT.TransferWorker = orig; gui.TransferWorker = orig
    # callbacks
    tab2._on_preview({"asset": _png("as.png"), "mask": _png("mk.png")})
    tab2._on_done({"info": "plan text", "output": None})
    check("tt_on_done_plan", "plan text" in tab2.plan_view.toPlainText())
    outp = _png("outp.png")
    tab2._on_done({"info": "cosine 0.99", "output": outp})
    check("tt_on_done_output", tab2._viz["output"]._fullpath == outp)
    tab2._on_failed("bad"); check("tt_on_failed", "bad" in tab2.status.text())
    tab2.worker = FakeWorker(); tab2._on_finished(); check("tt_on_finished", tab2.worker is None)


def test_transfer_tab_draw_mask_and_masks():
    host = _tt_host()
    tab = gui.TransferTab(host)
    p1 = _png("dm1.png")
    tab._add_row(p1)
    # _draw_mask with a stubbed dialog that "accepts" with a mask
    maskfile = _png("transfer_mask_123.png")   # not in owned dir, but exercises path
    class FakeDlg:
        mask_path = maskfile; protect_face = False
        def __init__(self, *a, **k): pass
        def exec_(self): return QDialog.Accepted
    # Same relocation as TransferWorker: _draw_mask now resolves MaskDrawDialog
    # in gui_transfer_tab (imported there from gui_dialogs), so patching gui's
    # copy alone left the REAL dialog in place and exec_() blew up on the fake.
    import gui_transfer_tab as _TT2
    orig = _TT2.MaskDrawDialog
    _TT2.MaskDrawDialog = FakeDlg; gui.MaskDrawDialog = FakeDlg
    try:
        tab._draw_mask(tab._rows[0]["frame"])
        check("tt_draw_mask", tab._rows[0]["mask"] == maskfile)
        tab._draw_mask(QWidget())   # unknown frame -> no-op
    finally:
        _TT2.MaskDrawDialog = orig; gui.MaskDrawDialog = orig
    # _owns_mask: a file we wrote to the owned dir
    owned = str(gui.OUTPUT_DIR_GUI_MASK() / "transfer_mask_999.png")
    open(owned, "wb").write(b"\x89PNG")
    check("tt_owns_mask_true", tab._owns_mask(owned))
    check("tt_owns_mask_false", not tab._owns_mask(p1))
    check("tt_owns_mask_none", not tab._owns_mask(None))
    # _drop_mask_file removes an owned mask
    tab._rows[0]["mask"] = owned
    tab._drop_mask_file(tab._rows[0])
    check("tt_drop_mask", not os.path.exists(owned) and tab._rows[0]["mask"] is None)


# ------------------------------------------------------------------ TransferWorker

def _im_mod(**over):
    class Ref:
        def __init__(self, path, role):
            self.path = path; self.role = role; self.effective_role = role or "object_source"
            self.extracted_asset_path = None
        def is_extractable(self): return True
    base = {
        "ReferenceImage": Ref,
        "crop_to_mask": lambda p, m: p + ".crop",
        "infer_reference_roles": lambda instr, refs: None,
        "extract_reference_asset": lambda ctx, ref: setattr(ref, "extracted_asset_path", "asset.png"),
        "_ROLE_TARGET_REGION": {"object_source": "torso"},
        "_upload_image_to_comfy": lambda t, url: "uploaded",
        "_region_mask_file": lambda ctx, up, region, n, seed=1, timeout=600: "maskf.png",
        "COMFY_URL": "http://comfy",
        "build_edit_plan": lambda t, refs, instr: {"n_passes": 1, "steps": [
            {"roles": ["object_source"], "extract": True, "instruction": "do it"}]},
        "transfer_with_references": lambda ctx, t, refs, instr: _png("whole_out.png"),
        "plan_and_execute_transfer": lambda ctx, t, refs, instr, mask_override=None, protect_face=True: _png("contained_out.png"),
        "assert_deliverable": lambda out, where=None, source_path=None: out,
    }
    base.update(over)
    return base

def test_transfer_worker_modes():
    ctx = types.SimpleNamespace(is_cancelled=lambda: False)
    tgt = _png("wt.png")
    # plan mode
    w = gui.TransferWorker(ctx, tgt, [(_png("wr.png"), None)], "put jacket", "plan")
    rec = cap(w, "progress", "preview", "done", "failed")
    with fake_modules(image=_im_mod()):
        w.run()
    check("tw_plan_done", len(rec["done"]) == 1 and "Passes" in rec["done"][0]["info"])
    check("tw_plan_preview", len(rec["preview"]) == 1)
    # whole mode
    w2 = gui.TransferWorker(ctx, tgt, [(_png("wr2.png"), None)], "", "whole")
    rec2 = cap(w2, "done", "failed")
    with fake_modules(image=_im_mod(), identity_metrics={"identity_cosine": lambda a, b: 0.95}):
        w2.run()
    check("tw_whole_done", rec2["done"] and rec2["done"][0]["output"].endswith("whole_out.png"))
    check("tw_whole_cosine", "cosine" in rec2["done"][0]["info"])
    # contained mode with a source mask (crop branch) + drift warning
    w3 = gui.TransferWorker(ctx, tgt, [(_png("wr3.png"), "clothing_source", _png("srcm.png"))],
                            "x", "contained", target_mask=_png("tm.png"))
    rec3 = cap(w3, "progress", "done", "failed")
    with fake_modules(image=_im_mod(), identity_metrics={"identity_cosine": lambda a, b: 0.5}):
        w3.run()
    check("tw_contained_done", rec3["done"] and "DRIFT" in rec3["done"][0]["info"])
    # deliverable None -> failed
    w4 = gui.TransferWorker(ctx, tgt, [(_png("wr4.png"), None)], "x", "contained")
    rec4 = cap(w4, "done", "failed")
    with fake_modules(image=_im_mod(assert_deliverable=lambda out, where=None, source_path=None: None)):
        w4.run()
    check("tw_no_deliverable", rec4["failed"] and "no result" in rec4["failed"][0])
    # exception -> failed
    w5 = gui.TransferWorker(ctx, tgt, [(_png("wr5.png"), None)], "x", "contained")
    rec5 = cap(w5, "done", "failed")
    with fake_modules(image=_im_mod(plan_and_execute_transfer=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("gpu")))):
        w5.run()
    check("tw_exception", rec5["failed"] == ["gpu"])


# ------------------------------------------------------------------ MemoryCenterTab dialogs

def _mc_tab():
    import memory_center
    store = memory_center.MemoryStore(root=tempfile.mkdtemp(prefix="mcstore_"))
    store.create_profile("default")
    for i in range(3):
        store.add("default", f"session note {i}", etype="session", source="inferred")
    store.add("default", "a pinned fact", etype="fact", source="manual", importance=70)
    host = _Host(ctx=types.SimpleNamespace(active_memory_dir=store.profile_dir("default"),
                                           session_memory=[]),
                 _set_status=lambda m: None,
                 _memory_center_set_profile=lambda p: None)
    tab = gui.MemoryCenterTab(host)
    tab.store = store
    tab.refresh_all()
    return tab, store


def test_memory_center_dialogs():
    tab, store = _mc_tab()
    # select first row so _sel is set
    if tab.table.rowCount():
        tab.table.selectRow(0)
    # auto-accept every modal dialog
    orig_exec = QDialog.exec_
    QDialog.exec_ = lambda self: (self.reject() or QDialog.Rejected)
    orig_menu_exec = gui.QMenu.exec_; gui.QMenu.exec_ = lambda self, *a, **k: None
    orig_conf = tab._confirm; tab._confirm = lambda *a, **k: False   # decline destructive
    try:
        tab._add_dialog()
        tab._revisions_dialog() if tab._sel else None
        tab._trash_dialog()
        tab._move_menu() if tab._sel else None
        tab._delete_selected()      # declined -> no-op
        tab._delete_matching()      # declined -> no-op
        check("mc_dialogs_declined_safe", True)
        # now accept destructive
        tab._confirm = lambda *a, **k: True
        # dup + save + toggle pin
        tab._dup_entry() if tab._sel else None
        if tab._sel:
            tab.editor.setPlainText("edited"); tab.imp_slider.setValue(50)
            tab.tags_edit.setText("a, b"); tab._save_entry()
        # pin a session entry
        sess = [e for e in store.load("default") if e.type == "session"]
        if sess:
            tab._sel = sess[0]; tab._toggle_pin()
        check("mc_mutations", True)
        # diagnostics + rebuild + timeline
        tab.diag_query.setText("dark theme"); tab._run_diag()
        check("mc_diag", "Profile" in tab.diag_out.toPlainText())
        tab._rebuild_index()
        tab._refresh_timeline()
        tab._refresh_compaction()
        # compaction reject
        tab.compact_summary.setPlainText("x"); tab._reject_compaction()
        check("mc_compact_reject", tab.compact_summary.toPlainText() == "")
        tab._on_preview_ready("a summary"); check("mc_preview_ready", tab.approve_btn.isEnabled())
    finally:
        QDialog.exec_ = orig_exec; gui.QMenu.exec_ = orig_menu_exec; tab._confirm = orig_conf


def test_memory_center_profiles_and_export():
    tab, store = _mc_tab()
    # profile ops via stubbed input dialogs
    QInputDialog.getText = staticmethod(lambda *a, **k: ("newprof", True))
    tab._prof_create()
    check("mc_prof_create", "newprof" in store.profiles())
    tab.prof_list.setCurrentRow(0)
    tab._selected_profile_name()
    QInputDialog.getText = staticmethod(lambda *a, **k: ("renamed", True))
    # select a non-active profile to rename/dup
    for i in range(tab.prof_list.count()):
        name = tab._selected_profile_name()
        if name and name != tab._active_profile():
            break
        tab.prof_list.setCurrentRow(i)
    tab._prof_rename(); tab._prof_duplicate()
    tab._prof_suggested()
    check("mc_prof_suggested", len(store.profiles()) >= 1)
    tab._make_active()
    # export (stub save dialog) — json/md/zip
    for fmt, ext in [("json", ".json"), ("md", ".md"), ("zip", ".zip")]:
        dest = os.path.join(_TMP, f"exp{ext}")
        QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (dest, ""))
        tab._export(fmt)
        check(f"mc_export_{fmt}", os.path.exists(dest))
    # import preview declined
    src = os.path.join(_TMP, "exp.zip")
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (src, ""))
    orig_conf = tab._confirm; tab._confirm = lambda *a, **k: False
    try:
        tab._import()
        check("mc_import_declined", True)
    finally:
        tab._confirm = orig_conf
    # cancelled export (no path)
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: ("", ""))
    tab._export("json"); check("mc_export_cancel", True)


if __name__ == "__main__":
    _saved = {"getOpenFileName": QFileDialog.getOpenFileName,
              "getOpenFileNames": QFileDialog.getOpenFileNames,
              "getSaveFileName": QFileDialog.getSaveFileName,
              "getText": QInputDialog.getText}
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
    # restore statics
    QFileDialog.getOpenFileName = staticmethod(_saved["getOpenFileName"])
    QFileDialog.getOpenFileNames = staticmethod(_saved["getOpenFileNames"])
    QFileDialog.getSaveFileName = staticmethod(_saved["getSaveFileName"])
    QInputDialog.getText = staticmethod(_saved["getText"])
    shutil.rmtree(_TMP, ignore_errors=True)
    print(f"\n{len(fns)-failed}/{len(fns)} tab test functions passed "
          f"({sum(1 for _,c in RESULTS if c)}/{len(RESULTS)} checks)")
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)
