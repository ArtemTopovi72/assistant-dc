from types import SimpleNamespace
from PyQt5.QtWidgets import QApplication
from gui_telegram_tab import TelegramTab


def _tab():
    stages = []
    host = SimpleNamespace(stage=SimpleNamespace(set_stage=stages.append), worker=None)
    return SimpleNamespace(_active_chats={}, status_board=SimpleNamespace(setText=lambda _: None),
                           host=host, _log_append=lambda _: None), stages


def test_finishing_chat_keeps_the_latest_other_chat_stage():
    app = QApplication.instance() or QApplication([])
    tab, stages = _tab()
    TelegramTab._on_stage(tab, 1, 'Drawing a picture', False)
    TelegramTab._on_stage(tab, 2, 'Searching the web', False)
    TelegramTab._on_stage(tab, 1, 'Checking the layout', False)
    TelegramTab._on_stage(tab, 1, 'Done', True)
    assert stages[-1] == 'Searching the web'


def test_telegram_does_not_override_running_desktop_task():
    app = QApplication.instance() or QApplication([])
    tab, stages = _tab()
    tab.host.worker = object()
    TelegramTab._on_stage(tab, 1, 'Drawing a picture', False)
    TelegramTab._on_stage(tab, 1, 'Done', True)
    assert not stages
