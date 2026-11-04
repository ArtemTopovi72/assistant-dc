def test_menu_button_does_not_read_pending_forwarded_voice(tmp_path):
    import tg_bot as tg
    tg.redirect_data_dir(str(tmp_path))
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


def test_admin_panel_button_ignores_pending_forwarded_voice(tmp_path):
    import tg_bot as tg
    tg.redirect_data_dir(str(tmp_path))
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
