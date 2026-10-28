import hashlib
import json
from pathlib import Path

import context_v2 as C
from scripts.migrate_context_archives import migrate


def fixture(tmp_path):
    legacy = tmp_path / 'runtime' / 'ctx_archive'; legacy.mkdir(parents=True)
    text = 'The earlier conversation belongs to chat 101.'
    aid = 'a' + hashlib.sha1(text.encode()).hexdigest()[:7]
    entry = {'id': aid, 'text': text, 'kind': 'chat', 'label': 'turns'}
    (legacy / f'{aid}.json').write_text(json.dumps(entry), 'utf8')
    sessions = {'101': {'context_generation': 4, 'history': [{'role': 'user', 'content': 'current question'}]},
                '202': {'context_generation': 1, 'history': []}}
    (tmp_path / 'tg_sessions.json').write_text(json.dumps(sessions), 'utf8')
    return aid, entry, sessions


def test_preview_changes_neither_sessions_nor_archives(tmp_path):
    aid, _, sessions = fixture(tmp_path)
    result = migrate(tmp_path, {aid: {'chat_id': '101', 'generation': 4, 'active': True}})
    assert result['active'] == 1
    assert json.loads((tmp_path / 'tg_sessions.json').read_text('utf8')) == sessions
    assert not (tmp_path / 'tg_memory').exists()


def test_apply_preserves_history_and_originals_and_exposes_only_the_owners_copy(tmp_path):
    aid, entry, sessions = fixture(tmp_path)
    result = migrate(tmp_path, {aid: {'chat_id': '101', 'generation': 4, 'active': True}}, apply=True)
    migrated = json.loads((tmp_path / 'tg_sessions.json').read_text('utf8'))
    assert migrated['202'] == sessions['202']
    assert migrated['101']['history'][-1] == sessions['101']['history'][0]
    assert migrated['101']['context_generation'] == 4
    assert aid in migrated['101']['history'][0]['content']
    assert json.loads(Path(result['backup']).read_text('utf8')) == sessions
    assert json.loads((tmp_path / 'runtime/ctx_archive' / f'{aid}.json').read_text('utf8')) == entry
    with C.archive_scope(tmp_path / 'tg_memory/chat_101/archive/4'):
        assert C.archive_get(aid) == entry
    with C.archive_scope(tmp_path / 'tg_memory/chat_202/archive/1'):
        assert C.archive_get(aid) is None
    migrate(tmp_path, {aid: {'chat_id': '101', 'generation': 4, 'active': True}}, apply=True)
    assert json.loads((tmp_path / 'tg_sessions.json').read_text('utf8')) == migrated


def test_retired_records_are_bound_to_the_owner_but_not_reintroduced_after_clear(tmp_path):
    aid, entry, sessions = fixture(tmp_path)
    migrate(tmp_path, {aid: {'chat_id': '101', 'active': False}}, apply=True)
    assert json.loads((tmp_path / 'tg_sessions.json').read_text('utf8')) == sessions
    with C.archive_scope(tmp_path / 'tg_memory/chat_101/archive/4'):
        assert C.archive_get(aid) is None
    assert json.loads((tmp_path / 'tg_memory/chat_101/legacy_archive' / f'{aid}.json').read_text('utf8')) == entry


def test_conflicting_destination_aborts_without_overwriting_either_copy_or_history(tmp_path):
    import pytest
    aid, _, sessions = fixture(tmp_path)
    dest = tmp_path / 'tg_memory/chat_101/archive/4' / f'{aid}.json'
    dest.parent.mkdir(parents=True); dest.write_text('{"text":"different"}', 'utf8')
    with pytest.raises(ValueError, match='conflict'):
        migrate(tmp_path, {aid: {'chat_id': '101', 'generation': 4, 'active': True}}, apply=True)
    assert json.loads((tmp_path / 'tg_sessions.json').read_text('utf8')) == sessions
    assert json.loads(dest.read_text('utf8')) == {'text': 'different'}


def test_invalid_ownership_or_hash_never_writes(tmp_path):
    import pytest
    aid, _, sessions = fixture(tmp_path)
    for owner in ('../other', '303'):
        with pytest.raises(ValueError):
            migrate(tmp_path, {aid: {'chat_id': owner, 'active': True}}, apply=True)
    source = tmp_path / 'runtime/ctx_archive' / f'{aid}.json'
    source.write_text(json.dumps({'id': aid, 'text': 'tampered'}), 'utf8')
    with pytest.raises(ValueError, match='hash'):
        migrate(tmp_path, {aid: {'chat_id': '101', 'generation': 4, 'active': True}}, apply=True)
    assert json.loads((tmp_path / 'tg_sessions.json').read_text('utf8')) == sessions


def test_a_manifest_from_before_clear_cannot_activate_archives_in_the_new_generation(tmp_path):
    import pytest
    aid, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='generation changed'):
        migrate(tmp_path, {aid: {'chat_id': '101', 'generation': 3, 'active': True}}, apply=True)
    assert not (tmp_path / 'tg_memory').exists()
