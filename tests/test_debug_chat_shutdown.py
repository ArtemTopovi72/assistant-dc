from types import SimpleNamespace
import gui_debug_chat_tab as debug


def test_poller_stops_without_emitting_after_shutdown(monkeypatch):
    poller = debug._Poller()
    emitted = []
    poller.state.connect(emitted.append)
    def close_during_request(*args, **kwargs):
        poller.stop()
        return {'messages': []}
    monkeypatch.setattr(debug, '_call', close_during_request)
    poller.run()
    assert emitted == []


def test_debug_tab_shutdown_stops_and_joins_poller():
    calls = []
    poller = SimpleNamespace(stop=lambda: calls.append('stop'),
                             wait=lambda ms: calls.append(('wait', ms)))
    debug.DebugChatTab.shutdown(SimpleNamespace(_poller=poller))
    assert calls[0] == 'stop'
    assert calls[1][0] == 'wait' and calls[1][1] >= 5000
