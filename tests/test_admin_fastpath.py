from test_tg_incoming_durability import make_bot
import tg_bot as T
import tg_strings as S


def test_admin_panel_opens_without_queue_and_checks_current_rights(tmp_path):
    bot = make_bot(tmp_path)
    opened, sent = [], []
    bot._send_admin_panel = lambda chat: opened.append(chat)
    bot._send_text = lambda chat, text, **kw: sent.append(text)
    bot._enqueue_item = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("panel queued"))
    bot._user_store.put(T._User(chat_id=101, name="Admin", status="approved", is_admin=True))
    def press(chat):
        bot._dispatch({"message": {"message_id": 1, "chat": {"id": chat, "type": "private"},
                                   "from": {"id": chat}, "text": S._BTN["admin"]["ru"]}})
    press(101)
    assert opened == [101]
    bot._user_store.put(T._User(chat_id=101, name="Admin", status="approved", is_admin=False))
    press(101)
    assert opened == [101]
    press(202)
    assert opened == [101] and sent
