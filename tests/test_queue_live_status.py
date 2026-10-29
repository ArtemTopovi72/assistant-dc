import threading
from types import SimpleNamespace

from tg_resolve import ResolveMixin


def test_waiting_position_is_refreshed_and_removed_when_task_starts():
    first = SimpleNamespace(task_id='first', chat_id=1)
    mine = SimpleNamespace(task_id='mine', chat_id=2)
    pending = [first, mine]
    edits = []
    class Bot(ResolveMixin):
        _queue_note_lock = threading.RLock()
        _task_lock = threading.Lock()
        _queue_note_tasks = {}
        _busy_notes = {}
        _running_task = {}
        _chat_interruptible = {}
        _backend = SimpleNamespace(service_order=lambda: list(pending),
                                   tasks_ahead=lambda tid: pending[:next(i for i, t in enumerate(pending) if t.task_id == tid)])
        def _card_busy(self): return False
        def _send_get_id(self, chat, text, **kw): edits.append(text); return 42
        def _edit_text(self, chat, mid, text, **kw): edits.append(text)
        def _delete(self, chat, mid): edits.append('deleted')
    bot = Bot()
    bot._queue_note(2, mine, 'ru')
    assert '#2' in edits[-1]
    pending.pop(0)
    bot._refresh_queue_notes()
    assert 'следующий' in edits[-1]
    bot._drop_busy_note(2, 'another-task')
    assert edits[-1] != 'deleted'
    pending.clear()
    bot._drop_busy_note(2, 'mine')
    assert edits[-1] == 'deleted'
