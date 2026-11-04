import pytest


@pytest.fixture
def isolated_bot_storage(tmp_path, monkeypatch):
    import tg_bot as tg
    import nice_names, code_sandbox, dr_timing
    paths = ('_SESSION_FILE', '_USERS_DB', '_ACTIVITY_LOG_FILE', '_OFFSET_FILE',
             '_FEEDBACK_FILE', '_INFLIGHT_FILE', '_LIBRARY_DIR', '_IMAGE_DIR',
             '_MEMORY_DIR', '_MASHUP_DIR')
    for module, names in ((tg, paths),
                          (tg._userstore, ('_USERS_FILE_JSON', '_USERS_BACKUP_DIR')),
                          (tg._artifacts, ('_ARTIFACTS_DB',)),
                          (tg._queue_backends, ('_FORCE_INMEMORY',)),
                          (nice_names, ('_ROOT',)), (code_sandbox, ('SANDBOX_BASE',)),
                          (dr_timing, ('_TIMING_FILE',))):
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))
    tg.redirect_data_dir(str(tmp_path))


def test_menu_button_does_not_read_pending_forwarded_voice(isolated_bot_storage):
    import tg_bot as tg
    chat = 999941
    bot = tg.TelegramBot('123:TEST', lambda: object(), lambda: object(), lambda: {}, silent_mode=True)
    bot._send_text = lambda *args, **kwargs: 1
    bot._activity.log = lambda *args, **kwargs: None
    bot._user_store.put(tg._User(chat_id=chat, name='T', status='approved'))
    session = bot._get_session(chat)
    session.fwd_transcript = 'A forwarded voice'
    session.fwd_transcript_done = False
    bot._store.put(session)
    reads = []
    bot.fwdv_intent = lambda text: reads.append(text) or ''
    bot._resolve_and_push(chat, [{'type': 'text', 'text': tg._b('settings', 'ru')}])
    assert not reads


def test_admin_panel_button_ignores_pending_forwarded_voice(isolated_bot_storage):
    import tg_bot as tg
    chat = 999942
    bot = tg.TelegramBot('123:TEST', lambda: object(), lambda: object(), lambda: {}, silent_mode=True)
    bot._activity.log = lambda *args, **kwargs: None
    bot._user_store.put(tg._User(chat_id=chat, name='T', status='approved', is_admin=True))
    session = bot._get_session(chat)
    session.is_admin = True
    session.fwd_transcript = 'A forwarded voice'
    session.fwd_transcript_done = False
    bot._store.put(session)
    reads, panels = [], []
    bot.fwdv_intent = lambda text: reads.append(text) or ''
    bot._send_admin_panel = lambda cid: panels.append(cid)
    bot._resolve_and_push(chat, [{'type': 'text', 'text': tg._b('admin', 'ru')}])
    assert not reads and panels == [chat]
