import ctypes
import types
from ctypes import wintypes

import pytest
from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QLabel

import _isolate_library
import gui
import gui_common
import gui_win_frame as frame
from gui_fakes import NoopLoader
from gui_layout_mixin import LayoutMixin
import ui_scale


@pytest.fixture
def window(tmp_path):
    app = QApplication.instance() or QApplication([])
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path))
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    ui_scale.force(effective=1, font_mult=1)
    app.setStyleSheet(gui_common.build_qss())
    win = gui.AssistantWindow("test", True, loader=NoopLoader)
    win.player = types.SimpleNamespace(stop=lambda: None)
    win.stack.setCurrentWidget(win.dashboard)
    yield win
    win.close()


def visible_keys(nav):
    return [nav.item(i).data(Qt.UserRole) for i in range(nav.count())
            if nav.item(i).data(Qt.UserRole) and not nav.item(i).isHidden()]


def test_commands_filter_both_languages_and_keep_disabled_state(window):
    window.pages.setCurrentIndex(0)
    for query in ("hands", "руки"):
        window.commands_search.setText(query)
        assert not window.fixhands_btn.isHidden()
        assert window.talk_btn.isHidden()
        assert not window.retryhands_btn.isEnabled()
    window.commands_search.clear()
    assert not window.talk_btn.isHidden()
    assert not window.retryhands_btn.isEnabled()


def test_commands_search_has_an_empty_state_and_never_executes_on_enter(window):
    window.pages.setCurrentIndex(0)
    window.commands_search.setText("no-such-command-8927")
    assert not window.commands_empty.isHidden()
    QTest.keyClick(window.commands_search, Qt.Key_Return)
    assert window.pages.currentIndex() == 0
    window.commands_search.clear()
    assert window.commands_empty.isHidden()


def test_workspace_search_preserves_page_until_enter_and_survives_rebuild(window):
    nav = window.workspace_nav
    before = window.tabs.currentWidget()
    window.workspace_search.setText("база")
    assert visible_keys(nav) == ["database"]
    assert window.tabs.currentWidget() is before
    QTest.keyClick(window.workspace_search, Qt.Key_Return)
    assert window._tab_key_for_widget(window.tabs.currentWidget()) == "database"
    window._show_tab("database", False)
    assert visible_keys(nav) == []
    assert not window.workspace_empty.isHidden()
    window._show_tab("database", True)
    assert visible_keys(nav) == ["database"]
    window.workspace_search.clear()
    assert len(visible_keys(nav)) == window.tabs.count()


def test_header_has_one_brand_and_long_model_does_not_widen_window(window):
    full_name = "gemma4-26b-a4b-uncensored-hauhaucs-balanced-" * 3
    window.model_chip.setText(full_name)
    window.resize(800, 600)
    window.show()
    QApplication.processEvents()
    assert window.width() <= 800
    brands = [label for label in window.findChildren(QLabel)
              if label.text() == "Assistant DC" and label.isVisible()]
    assert len(brands) == 1
    assert window.model_chip.toolTip() == full_name
    assert window.model_chip.text() == full_name
    assert QLabel.text(window.model_chip).endswith("…")
    assert window.layout_btn.geometry().right() <= window.dashboard.width()


def test_native_message_uses_the_actual_ctypes_handle_field():
    msg = wintypes.MSG()
    msg.hWnd = 123
    msg.message = frame.WM_NCCALCSIZE
    msg.wParam = 1
    parsed = frame.Msg(ctypes.addressof(msg))
    assert (parsed.hwnd, parsed.message, parsed.wParam) == (123, frame.WM_NCCALCSIZE, 1)


@pytest.mark.parametrize("failure", [False, True])
def test_frame_handler_is_active_during_synchronous_install(failure):
    owner = LayoutMixin()
    owner.native_frame = False
    owner.parts = {"platformName": lambda: "windows"}
    owner.winId = lambda: 7
    msg = wintypes.MSG()
    msg.hWnd, msg.message, msg.wParam = 7, frame.WM_NCCALCSIZE, 1
    replies = []
    def install(hwnd):
        replies.append(owner._win_frame_message(ctypes.addressof(msg)))
        if failure:
            raise OSError("frame install failed")
        return True
    owner.modules = {"gui_win_frame": types.SimpleNamespace(install=install, Msg=frame.Msg,
        remove_frame_area=lambda m: frame.remove_frame_area(m, user32=object()))}
    owner._install_native_frame()
    assert replies == [0]
    assert owner._win_native is (not failure)


@pytest.mark.parametrize(("hit_code", "expected"), [(0, None), (frame.HTCAPTION, frame.HTCAPTION),
                                                   (frame.HTRIGHT, frame.HTRIGHT)])
def test_client_hit_testing_leaves_buttons_to_qt(hit_code, expected):
    owner = LayoutMixin()
    owner._win_native = True
    owner.devicePixelRatioF = lambda: 1
    owner.modules = {"gui_win_frame": types.SimpleNamespace(
        Msg=lambda address: types.SimpleNamespace(message=frame.WM_NCHITTEST),
        WM_NCHITTEST=frame.WM_NCHITTEST, remove_frame_area=lambda msg: False,
        hit=lambda msg, caption, **kwargs: hit_code)}
    assert owner._win_frame_message(123) == expected


def test_caption_buttons_take_priority_over_the_resize_band():
    class Api:
        def GetWindowRect(self, hwnd, ref):
            rect = ctypes.cast(ref, ctypes.POINTER(wintypes.RECT)).contents
            rect.left, rect.top, rect.right, rect.bottom = 0, 0, 200, 100
            return 1
        def GetSystemMetrics(self, metric):
            return 9
    def hit(x, y):
        msg = types.SimpleNamespace(hwnd=7, lParam=(y << 16) | x)
        return frame.hit(msg, lambda x, y: y < 36, user32=Api(),
                         client=lambda x, y: 130 <= x <= 170 and 4 <= y <= 32)
    assert hit(150, 17) == frame.HTCLIENT
    assert hit(150, 1) == frame.HTTOP
    assert hit(50, 24) == frame.HTCAPTION


@pytest.mark.parametrize("source", ["Ready", "queries 7"])
def test_elided_label_can_switch_language_back_to_english(window, source):
    import gui_i18n
    from gui_chrome import ElidingLabel
    gui_i18n.install("en")
    label = ElidingLabel(source)
    try:
        gui_i18n.set_language("ru")
        assert label.text() == gui_i18n.tr(source)
        gui_i18n.set_language("en")
        assert label.text() == source
    finally:
        gui_i18n.set_language("en")
        label.close()


def test_workspace_finds_english_name_in_russian_ui(window):
    import gui_i18n
    from gui_nav import NavTabs, WorkspaceNav
    from PyQt5.QtWidgets import QWidget
    gui_i18n.install("ru")
    tabs = NavTabs()
    tabs.addTab(QWidget(), "Restyle Video")
    nav = WorkspaceNav(tabs, lambda widget: "restyle")
    try:
        assert tabs.tabText(0) == "Перерисовать видео"
        nav.filter("video")
        assert visible_keys(nav) == ["restyle"]
    finally:
        gui_i18n.set_language("en")
        nav.close()
        tabs.close()
