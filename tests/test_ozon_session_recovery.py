import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import ozon_client as oz


class FakeBrowser:
    def __init__(self):
        self.options = []

    def new_context(self, **kwargs):
        self.options.append(kwargs)
        return SimpleNamespace(new_page=lambda: object())


def test_corrupt_session_is_preserved_and_browser_can_start(tmp_path):
    state = tmp_path / 'session.json'
    broken = '{"cookies": []}{"cookies": []}'
    state.write_text(broken, encoding='utf-8')
    browser = FakeBrowser()
    slot = oz._Browser._context(browser, str(state))
    assert 'storage_state' not in browser.options[0]
    saved = list(tmp_path.glob('session.json.invalid-*'))
    assert len(saved) == 1 and saved[0].read_text(encoding='utf-8') == broken
    assert slot[3] == str(state)


def test_concurrent_saves_leave_one_complete_session(tmp_path):
    state = tmp_path / 'session.json'
    class Context:
        def __init__(self, n): self.n = n
        def storage_state(self, path):
            Path(path).write_text(json.dumps({'cookies': [{'name': str(self.n)}], 'origins': []}), encoding='utf-8')
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda n: oz._Browser._save([Context(n), None, False, str(state)]), range(32)))
    data = json.loads(state.read_text(encoding='utf-8'))
    assert len(data['cookies']) == 1 and data['origins'] == []


def test_failed_save_does_not_damage_existing_session(tmp_path):
    state = tmp_path / 'session.json'
    state.write_text('{"cookies": [], "origins": []}', encoding='utf-8')
    class Context:
        def storage_state(self, path):
            Path(path).write_text('{"cookies":', encoding='utf-8')
            raise RuntimeError('interrupted')
    oz._Browser._save([Context(), None, False, str(state)])
    assert json.loads(state.read_text(encoding='utf-8')) == {'cookies': [], 'origins': []}
