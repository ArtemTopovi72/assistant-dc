"""Interface language of the desktop app: English source, Russian on request.

Every screen string is written in English in the code. install("ru") wraps the
Qt calls that put text on screen (constructors, setText, tooltips, tabs, combo
items, message boxes, file dialogs), so a label is translated where it meets
Qt, not at hundreds of call sites. Combo boxes keep answering in English
(currentText/findText/setCurrentText), because code reads them as values.

Lookup order: exact phrase (gui_ru.RU), then a %s/%d template from the same
table, then the stage vocabulary the status line shares with Telegram
(stages.translate), then line by line. Anything left is logged once to
runtime/untranslated_gui.jsonl so a missed string is found by whoever meets it.
Chosen in Settings → Interface → Language and switched live: every wrapper keeps
the English source on the widget, and set_language() re-applies it everywhere.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading

logger = logging.getLogger("assistant.gui_i18n")

LANG = "en"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MISS_LOG = os.path.join(_ROOT, "runtime", "untranslated_gui.jsonl")
_LATIN = re.compile(r"[A-Za-z]{2,}")
_CYR = re.compile(r"[А-Яа-яЁё]")
_seen: set = set()
_lock = threading.Lock()
_exact: dict = {}
_templates: list = []
_back: dict = {}     # shown Russian -> English source, for combo values


def _compile(table: dict) -> None:
    _exact.clear(); _templates.clear(); _back.clear()
    for en, ru in table.items():
        if re.search(r"%(?:[-+ 0#]*\d*(?:\.\d+)?)[sdif]", en):
            parts = re.split(r"(%(?:[-+ 0#]*\d*(?:\.\d+)?)[sdif])", en)
            # %d / %f hold only numbers: '%ss' (seconds) as (.+?) turned «Facts» into «Fact с»
            rx = "".join(("([-+]?\\d+)" if p.endswith("d") else "([-+]?\\d+(?:[.,]\\d+)?)" if p.endswith("f")
                          else "(.+?)") if i % 2 else re.escape(p.replace("%%", "%")) for i, p in enumerate(parts))
            _templates.append((re.compile("^" + rx + "$", re.S),
                               re.sub(r"%(?:[-+ 0#]*\d*(?:\.\d+)?)[sdif]", "%s", ru)))
        else:
            _exact[en] = ru
            _back.setdefault(ru, en)
    _templates.sort(key=lambda t: -len(t[0].pattern))   # most specific first


def _one(s: str) -> str | None:
    hit = _exact.get(s)
    if hit is not None:
        return hit
    core = s.strip()
    if core != s and core in _exact:
        return s.replace(core, _exact[core], 1)
    for rx, ru in _templates:
        m = rx.match(s)
        if m:
            try:
                return ru % tuple(tr(g) for g in m.groups())
            except (TypeError, ValueError):
                return None
    try:
        import stages
        out = stages.translate(s, "ru")
        if out != s:
            return out
    except Exception:
        pass
    return None


def tr(s, log=True):
    """Text for the screen in the chosen language. Never raises.
    log=False for sinks that also carry model output and log lines (text panes)."""
    if LANG == "en" or not isinstance(s, str):
        return s
    if not _LATIN.search(s):
        # «24 h», «60s»: one Latin letter; the exact table first, then the numeric templates
        # («60s» only matches '%fs', and the song-length combo stayed English)
        hit = _exact.get(s)
        if hit is not None:
            return hit
        if not any(c.isalpha() for c in s):
            return s
        for rx, ru in _templates:
            m = rx.match(s)
            if m:
                try:
                    return ru % m.groups()
                except (TypeError, ValueError):
                    return s
        return s
    try:
        out = _one(s)
        if out is None and "\n" in s:
            lines = s.split("\n")
            outs = [(_one(l) if _LATIN.search(l) else l) for l in lines]
            if any(o is not None for o in outs):
                out = "\n".join(o if o is not None else l for o, l in zip(outs, lines))
        if out is None:
            if log:
                _missed(s)
            return s
        return out
    except Exception:
        return s


def back(s):
    """English source of a shown string (combo values)."""
    return _back.get(s, s) if LANG != "en" and isinstance(s, str) else s


def _missed(s: str) -> None:
    if _CYR.search(s) or s.startswith(("http", "/", "C:")) or len(s) > 400:
        return
    with _lock:
        if s in _seen:
            return
        _seen.add(s)
    try:
        os.makedirs(os.path.dirname(_MISS_LOG), exist_ok=True)
        with open(_MISS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _wrap_args(fn, idx=None):
    """fn with its string arguments (all, or those at positions `idx`) translated."""
    def w(*a, **k):
        a = tuple(tr(x) if (idx is None or i in idx) else x for i, x in enumerate(a))
        for key in ("text", "title", "label", "caption", "toolTip"):
            if key in k:
                k[key] = tr(k[key])
        return fn(*a, **k)
    w.__wrapped__ = fn
    return w


def _list_tr(fn):
    def w(self, *a):
        a = list(a)
        for i, x in enumerate(a):
            if isinstance(x, (list, tuple)) and all(isinstance(y, str) for y in x):
                a[i] = [tr(y) for y in x]
            elif isinstance(x, str):
                a[i] = tr(x)
        return fn(self, *a)
    w.__wrapped__ = fn
    return w


_ROLE = 0x0100 + 77          # item data role holding an item's English source
_PROP = "_i18n_"             # widget property prefix: English source per setter
_SETTERS = ("setText", "setToolTip", "setPlaceholderText", "setWindowTitle", "setTitle",
            "setStatusTip", "setSpecialValueText", "setSuffix", "setPrefix", "setFormat",
            "setInformativeText", "setLabelText")


def _keep(obj, prop, src):
    """Remember the English `src` of `prop` on a widget (property) or an item (data role)."""
    try:
        if hasattr(obj, "setProperty"):
            obj.setProperty(_PROP + prop, src)
        elif prop == "setText" and hasattr(obj, "setData"):
            obj.setData(_ROLE, src)
    except Exception:
        pass


def _shown(obj, src):
    out = tr(src)
    try:
        if hasattr(obj, "property") and obj.property("_i18n_upper"):
            out = out.upper()          # section headers are shown in capitals
    except Exception:
        pass
    return out


def _remember(fn, prop):
    """setter(text, ...): keep the English on the object, show the translation."""

    def w(self, *a, **k):
        if a and isinstance(a[0], str):
            _keep(self, prop, a[0])
            a = (_shown(self, a[0]),) + a[1:]
        return fn(self, *a, **k)
    w.__wrapped__ = fn
    return w


def _ctor(prop):
    """Constructor whose first argument is the text: remembered like setter `prop`."""
    def wrap(fn):
        def w(self, *a, **k):
            src = a[0] if a and isinstance(a[0], str) else None
            if src is not None:
                a = (tr(src),) + a[1:]
            fn(self, *a, **k)
            if src is not None:
                _keep(self, prop, src)
        w.__wrapped__ = fn
        return w
    return wrap


def install(lang: str) -> None:
    """Wrap Qt's text entry points once and pick the starting language. Must run
    before the first widget is built; later switches go through set_language()."""
    global LANG
    LANG = "ru" if lang == "ru" else "en"
    from gui_ru import RU
    _compile(RU)
    from PyQt5 import QtWidgets as W

    def patch(cls, name, wrapper):
        orig = getattr(cls, name)
        if getattr(orig, "__wrapped__", None) is None:
            setattr(cls, name, wrapper(orig))

    for cls, prop in ((W.QLabel, "setText"), (W.QPushButton, "setText"), (W.QCheckBox, "setText"),
                      (W.QRadioButton, "setText"), (W.QToolButton, "setText"),
                      (W.QCommandLinkButton, "setText"), (W.QAction, "setText"),
                      (W.QListWidgetItem, "setText"), (W.QTableWidgetItem, "setText"),
                      (W.QGroupBox, "setTitle"), (W.QMenu, "setTitle"),
                      (W.QDockWidget, "setWindowTitle")):
        patch(cls, "__init__", _ctor(prop))
    for cls, names in (
            (W.QLabel, ("setText",)), (W.QAbstractButton, ("setText",)),
            (W.QWidget, ("setToolTip", "setWindowTitle", "setStatusTip", "setWhatsThis")),
            (W.QLineEdit, ("setPlaceholderText",)), (W.QTextEdit, ("setPlaceholderText",)),
            (W.QPlainTextEdit, ("setPlaceholderText",)), (W.QGroupBox, ("setTitle",)),
            (W.QAction, ("setText", "setToolTip", "setStatusTip")), (W.QMenu, ("setTitle",)),
            (W.QListWidgetItem, ("setText", "setToolTip")), (W.QTableWidgetItem, ("setText", "setToolTip")),
            (W.QProgressBar, ("setFormat",)),
            (W.QAbstractSpinBox, ("setSpecialValueText",)), (W.QSpinBox, ("setSuffix", "setPrefix")),
            (W.QDoubleSpinBox, ("setSuffix", "setPrefix")), (W.QMessageBox, ("setText", "setInformativeText")),
            (W.QProgressDialog, ("setLabelText", "setCancelButtonText"))):
        for n in names:
            patch(cls, n, lambda f, n=n: _remember(f, n))
    patch(W.QStatusBar, "showMessage", _wrap_args)
    patch(W.QMenu, "addMenu", _wrap_args)

    # Tabs: the English title rides on the page widget, so it survives reordering.
    def tab_add(f, ti):
        def w(self, *a):
            a = list(a)
            if len(a) > ti and isinstance(a[ti], str):
                _keep(a[ti - 1], "tab", a[ti])
                a[ti] = tr(a[ti])
            return f(self, *a)
        w.__wrapped__ = f
        return w
    patch(W.QTabWidget, "addTab", lambda f: tab_add(f, 1))
    patch(W.QTabWidget, "insertTab", lambda f: tab_add(f, 2))

    def tab_set(f):
        def w(self, i, text):
            pg = self.widget(i)
            if pg is not None and isinstance(text, str):
                _keep(pg, "tab", text)
            return f(self, i, tr(text))
        w.__wrapped__ = f
        return w
    patch(W.QTabWidget, "setTabText", tab_set)
    patch(W.QTabWidget, "setTabToolTip", lambda f: _wrap_args(f, {2}))

    # A form row label passed as a string becomes a QLabel made here, so it is remembered.
    def form_row(f, at):
        def w(self, *a):
            a = list(a)
            if len(a) > at + 1 and isinstance(a[at], str):
                a[at] = W.QLabel(a[at])
            return f(self, *a)
        w.__wrapped__ = f
        return w
    patch(W.QFormLayout, "addRow", lambda f: form_row(f, 0))
    patch(W.QFormLayout, "insertRow", lambda f: form_row(f, 1))
    patch(W.QMenu, "addAction", _list_tr)
    patch(W.QToolBar, "addAction", _list_tr)

    def headers(f, prop):
        def w(self, labels):
            _keep(self, prop, list(labels))
            return f(self, [tr(x) for x in labels])
        w.__wrapped__ = f
        return w
    patch(W.QTableWidget, "setHorizontalHeaderLabels", lambda f: headers(f, "hh"))
    patch(W.QTableWidget, "setVerticalHeaderLabels", lambda f: headers(f, "vh"))
    patch(W.QTreeWidget, "setHeaderLabels", _list_tr)
    patch(W.QTreeWidget, "setHeaderLabel", _list_tr)
    for n in ("information", "warning", "critical", "question", "about"):
        patch(W.QMessageBox, n, lambda f: staticmethod(_wrap_args(f, {1, 2})))
    for n in ("getText", "getItem", "getInt", "getDouble", "getMultiLineText"):
        patch(W.QInputDialog, n, lambda f: staticmethod(_wrap_args(f, {1, 2})))
    for n in ("getOpenFileName", "getOpenFileNames", "getSaveFileName", "getExistingDirectory"):
        patch(W.QFileDialog, n, lambda f: staticmethod(_wrap_args(f, {1})))

    # Text panes also carry model answers and log lines: quiet lookup, history stays as written.
    def quiet(fn):
        def w(self, *a):
            return fn(self, *[tr(x, log=False) if isinstance(x, str) else x for x in a])
        w.__wrapped__ = fn
        return w
    patch(W.QTextEdit, "append", quiet)
    patch(W.QPlainTextEdit, "appendPlainText", quiet)

    def list_add(fn):
        def w(self, *a):
            a = list(a)
            if not (a and isinstance(a[-1], str)):
                return fn(self, *a)
            src = a[-1]
            a[-1] = tr(src, log=False)
            out = fn(self, *a)
            it = self.item(a[0] if len(a) == 2 else self.count() - 1)
            if it is not None:
                it.setData(_ROLE, src)
            return out
        w.__wrapped__ = fn
        return w
    patch(W.QListWidget, "addItem", list_add)
    patch(W.QListWidget, "insertItem", list_add)

    # Combo boxes: translated on screen, English to the code that reads them.
    C = W.QComboBox

    def combo_add(fn, many=False):
        def w(self, *a, **k):
            a = list(a)
            start = a[0] if a and isinstance(a[0], int) and a[0] >= 0 else self.count()
            srcs = []
            for i, x in enumerate(a):
                if many and isinstance(x, (list, tuple)):
                    srcs = list(x)
                    a[i] = [tr(y) for y in x]
                    break
                if not many and isinstance(x, str):
                    srcs = [x]
                    a[i] = tr(x)
                    break
            out = fn(self, *a, **k)
            for j, src in enumerate(srcs):
                if start + j < self.count():
                    self.setItemData(start + j, src, _ROLE)
            return out
        w.__wrapped__ = fn
        return w
    patch(C, "addItem", combo_add)
    patch(C, "insertItem", combo_add)
    patch(C, "addItems", lambda f: combo_add(f, True))
    patch(C, "insertItems", lambda f: combo_add(f, True))

    def combo_set(fn):
        def w(self, i, text):
            self.setItemData(i, text, _ROLE)
            return fn(self, i, tr(text))
        w.__wrapped__ = fn
        return w
    patch(C, "setItemText", combo_set)

    def combo_find(fn):
        def w(self, text, *a):
            for i in range(self.count()):
                if self.itemData(i, _ROLE) == text:
                    return i
            return fn(self, tr(text), *a)
        w.__wrapped__ = fn
        return w
    patch(C, "findText", combo_find)

    def combo_set_current(fn):
        def w(self, text):
            i = self.findText(text)
            if i >= 0:
                return self.setCurrentIndex(i)
            return fn(self, tr(text))
        w.__wrapped__ = fn
        return w
    patch(C, "setCurrentText", combo_set_current)

    def combo_text(fn, by_index):
        def w(self, *a):
            i = a[0] if by_index else self.currentIndex()
            src = self.itemData(i, _ROLE) if i >= 0 else None
            if isinstance(src, str):
                return src
            return back(fn(self, *a))
        w.__wrapped__ = fn
        return w
    patch(C, "currentText", lambda f: combo_text(f, False))
    patch(C, "itemText", lambda f: combo_text(f, True))


def set_language(lang: str) -> None:
    """Switch the running interface to `lang` in place, without a restart."""
    global LANG
    LANG = "ru" if lang == "ru" else "en"
    from PyQt5 import QtWidgets as W
    app = W.QApplication.instance()
    if app is None:
        return

    def raw(obj, name):
        f = getattr(type(obj), name)
        return getattr(f, "__wrapped__", f)

    def src(obj, prop):
        v = obj.property(_PROP + prop)
        return v if isinstance(v, str) else None

    for wdg in app.allWidgets():
        try:
            for prop in _SETTERS:
                s = src(wdg, prop)
                if s is not None and hasattr(wdg, prop):
                    raw(wdg, prop)(wdg, _shown(wdg, s))
            for act in wdg.actions():
                for prop in ("setText", "setToolTip"):
                    s = src(act, prop)
                    if s is not None:
                        raw(act, prop)(act, tr(s))
            if isinstance(wdg, W.QTabWidget):
                for i in range(wdg.count()):
                    pg = wdg.widget(i)
                    s = src(pg, "tab") if pg is not None else None
                    if s is not None:
                        raw(wdg, "setTabText")(wdg, i, tr(s))
            if isinstance(wdg, W.QComboBox):
                for i in range(wdg.count()):
                    s = wdg.itemData(i, _ROLE)
                    if isinstance(s, str):
                        raw(wdg, "setItemText")(wdg, i, tr(s))
            if isinstance(wdg, W.QListWidget):
                for i in range(wdg.count()):
                    it = wdg.item(i)
                    s = it.data(_ROLE)
                    if isinstance(s, str):
                        raw(it, "setText")(it, tr(s, log=False))
            if isinstance(wdg, W.QTableWidget):
                for prop, name in (("hh", "setHorizontalHeaderLabels"), ("vh", "setVerticalHeaderLabels")):
                    v = wdg.property(_PROP + prop)
                    if v:
                        raw(wdg, name)(wdg, [tr(x) for x in v])
                for r in range(wdg.rowCount()):
                    for c in range(wdg.columnCount()):
                        it = wdg.item(r, c)
                        s = it.data(_ROLE) if it is not None else None
                        if isinstance(s, str):
                            raw(it, "setText")(it, tr(s))
        except Exception:
            logger.debug("retranslate skipped a widget", exc_info=True)
