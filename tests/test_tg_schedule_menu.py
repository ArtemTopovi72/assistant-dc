"""⏰ Tasks: the bot's section for reminders, scheduled tasks and Ozon price watches.

The user asked for one place to see, add, change and delete them with buttons
(2026-10-26) -- typing «отмени напоминание про маму» is the only other way. The
real TelegramBot is driven with the network stubbed: buttons are pressed as
Telegram delivers them (a label, or an inline callback), and what changes is read
back from reminders, not from the bot's words.
"""
import os, sys, tempfile, time, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_sched_"))
import reminders

CID, OTHER = 999601, 999602


class _Held:
    def __init__(self, *a):
        self.daemon = True

    def start(self):
        pass

    def cancel(self):
        pass


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(), lambda: {}, silent_mode=True)
    bot.sent, bot.edits, bot.pushed, bot.ran = [], [], [], []
    bot._send_text = lambda cid, text, **kw: (bot.sent.append((cid, text, kw.get("keyboard"))), 1)[1]
    bot._send_get_id = lambda cid, text, **kw: bot._send_text(cid, text, **kw)
    bot._edit_text = lambda cid, mid, text, parse_mode=None, keyboard=None: bot.edits.append((text, keyboard))
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._backend = types.SimpleNamespace(push=lambda task: bot.pushed.append(task), depth=lambda: 0,
                                         name=lambda: "stub", drop_chat=lambda cid: 0, close=lambda: None)
    bot._activity.log = lambda *a, **k: None
    bot._run_scheduled = lambda cid, text: bot.ran.append((cid, text))
    for cid in (CID, OTHER):
        bot._user_store.put(T._User(chat_id=cid, name=f"u{cid}", status="approved"))
        s = bot._get_session(cid)
        s.lang, s.menu, s.pending_prefix = "ru", "", ""
        bot._store.put(s)
    reminders.stop()
    reminders.register(lambda cid, t: None, os.path.join(tempfile.mkdtemp(), "r.json"), timer=_Held,
                       runner=lambda cid, t: None)
    return bot


def say(bot, text, cid=CID):
    """A typed message the way _dispatch hands it on: the gate first (a capture may take it)."""
    msg = {"chat": {"id": cid, "type": "private"}, "from": {"id": cid}, "message_id": 2, "text": text}
    if bot._user_gate(cid, msg):
        bot._resolve_and_push(cid, [{"type": "text", "text": text}])


def tap(bot, key, cid=CID):
    say(bot, T._b(key, "ru"), cid)


def press(bot, data, cid=CID):
    bot._dispatch({"callback_query": {"id": "1", "data": data, "from": {"id": cid},
                                      "message": {"chat": {"id": cid, "type": "private"}, "message_id": 5}}})


def buttons(kb):
    if not isinstance(kb, dict):
        return []
    return [b for row in kb.get("keyboard") or kb.get("inline_keyboard") or [] for b in row]


def datas(kb):
    return [b.get("callback_data") for b in buttons(kb) if isinstance(b, dict)]


def last_reply_kb(bot):
    return next(kb for _, _, kb in reversed(bot.sent) if isinstance(kb, dict) and "keyboard" in kb)


def last_inline(bot):
    for text, kb in reversed(bot.edits):
        return text, kb
    return next((t, kb) for _, t, kb in reversed(bot.sent) if not (isinstance(kb, dict) and "keyboard" in kb))


def test_the_main_menu_gains_tasks_without_crowding_a_row():
    main = T._main_kb(False, False, "ru")["keyboard"]
    assert T._b("tasks", "ru") in [b for row in main for b in row]
    assert all(len(row) <= 3 for row in main)
    tasks_row = next(row for row in main if T._b("tasks", "ru") in row)
    assert len(tasks_row) <= 2, "a long label shares its row with one other at most"


def test_opening_tasks_shows_its_keyboard_and_the_list():
    bot = make_bot()
    reminders.add(CID, 3600, "позвонить маме")
    tap(bot, "tasks")
    assert bot._get_session(CID).menu == "tasks"
    assert T._b("sch_price", "ru") in last_reply_kb(bot)["keyboard"][-3] + last_reply_kb(bot)["keyboard"][-2]
    text, kb = last_inline(bot)
    labels = [(b.get("text") or "") for b in buttons(kb)]
    assert any("позвонить маме" in l for l in labels), labels      # the button IS the list row
    assert any(d.startswith("sch:o:") for d in datas(kb)), (text, kb)
    tap(bot, "back")
    assert bot._get_session(CID).menu == "" and T._b("tasks", "ru") in buttons(last_reply_kb(bot))


def test_an_empty_list_says_how_to_add_and_has_a_way_back():
    bot = make_bot()
    tap(bot, "sch_list")
    text, kb = last_inline(bot)
    assert "нет" in text.lower(), text
    assert "nav:back" in datas(kb), datas(kb)


def test_a_reminder_card_moves_repeats_retimes_and_deletes():
    bot = make_bot()
    rid = reminders.add(CID, 3600, "позвонить маме")
    due = reminders.get(CID, rid)["due"]
    press(bot, f"sch:o:{rid}")
    text, kb = last_inline(bot)
    assert "позвонить маме" in text and f"sch:del:{rid}" in datas(kb), (text, datas(kb))
    press(bot, f"sch:sn:{rid}:60")
    assert abs(reminders.get(CID, rid)["due"] - (due + 3600)) < 5
    press(bot, f"sch:rep:{rid}:daily")
    assert reminders.get(CID, rid)["every"] == 86400
    press(bot, f"sch:rep:{rid}:none")
    assert not reminders.get(CID, rid).get("every")
    press(bot, f"sch:t:{rid}")
    say(bot, "18:30")
    lt = time.localtime(reminders.get(CID, rid)["due"])
    assert (lt.tm_hour, lt.tm_min) == (18, 30) and reminders.get(CID, rid)["due"] > time.time()
    assert not bot.pushed, "the typed time is an answer, not a message for the agent"
    press(bot, f"sch:x:{rid}")
    say(bot, "позвонить бабушке")
    assert reminders.get(CID, rid)["text"] == "позвонить бабушке"
    press(bot, f"sch:del:{rid}")
    assert reminders.get(CID, rid), "delete asks first"
    press(bot, f"sch:delok:{rid}")
    assert reminders.get(CID, rid) is None


def test_a_wrong_time_asks_again_and_a_button_leaves_the_capture():
    bot = make_bot()
    rid = reminders.add(CID, 3600, "позвонить маме")
    press(bot, f"sch:t:{rid}")
    say(bot, "когда-нибудь")
    assert bot._get_session(CID).reg_state.startswith("sch_"), "stays armed for a retry"
    tap(bot, "back")
    assert not bot._get_session(CID).reg_state


def test_a_price_watch_card_sets_frequency_target_and_extends():
    bot = make_bot()
    rid = reminders.add(CID, 6 * 3600, "следить за ценой: рюкзак", every_s=6 * 3600, kind="price",
                        extra={"product": "https://www.ozon.ru/product/x-1/", "name": "Рюкзак",
                               "price": 15475, "until": time.time() + 86400})
    press(bot, f"sch:o:{rid}")
    text, kb = last_inline(bot)
    assert "15 475" in text and f"sch:ev:{rid}:12" in datas(kb), (text, datas(kb))
    press(bot, f"sch:ev:{rid}:12")
    assert reminders.get(CID, rid)["every"] == 12 * 3600
    press(bot, f"sch:tg:{rid}")
    say(bot, "13 000")
    assert reminders.get(CID, rid)["target"] == 13000
    press(bot, f"sch:tg0:{rid}")
    assert not reminders.get(CID, rid).get("target")
    until = reminders.get(CID, rid)["until"]
    press(bot, f"sch:ext:{rid}")
    assert abs(reminders.get(CID, rid)["until"] - (until + 30 * 86400)) < 5


def test_a_task_can_be_run_now():
    bot = make_bot()
    rid = reminders.add(CID, 3600, "пришли погоду", kind="task")
    press(bot, f"sch:run:{rid}")
    assert bot.ran == [(CID, "пришли погоду")]


def test_morning_weather_is_one_tap():
    bot = make_bot()
    press(bot, "sch:wx:08:00")
    task = next(i for i in reminders.pending(CID) if i.get("kind") == "task")
    lt = time.localtime(task["due"])
    assert task["every"] == 86400 and (lt.tm_hour, lt.tm_min) == (8, 0) and "погод" in task["text"]


def test_someone_elses_task_cannot_be_touched():
    bot = make_bot()
    rid = reminders.add(OTHER, 3600, "чужое")
    press(bot, f"sch:delok:{rid}")
    press(bot, f"sch:rep:{rid}:daily")
    assert reminders.get(OTHER, rid) and not reminders.get(OTHER, rid).get("every")


def test_add_buttons_arm_the_agent_and_keep_the_tasks_keyboard():
    bot = make_bot()
    tap(bot, "tasks")
    tap(bot, "sch_price")
    s = bot._get_session(CID)
    assert "ozon_watch_price" in s.pending_prefix
    assert T._b("sch_list", "ru") in buttons(last_reply_kb(bot))


def test_every_screen_has_a_way_back():
    """No dead ends (2026-10-27): the picker, the confirmations and the capture
    prompts each carry nav:back, the way the other menus' pickers do."""
    bot = make_bot()
    rid = reminders.add(CID, 3600, "позвонить маме")
    tap(bot, "sch_list")
    assert "nav:back" in datas(last_inline(bot)[1])                 # the list
    press(bot, f"sch:o:{rid}")
    press(bot, f"sch:t:{rid}")                                      # a capture prompt
    assert "nav:back" in datas(bot.sent[-1][2]), bot.sent[-1]
    press(bot, "nav:back")                                          # it leaves the capture...
    assert not bot._get_session(CID).reg_state
    assert T._b("sch_list", "ru") in buttons(last_reply_kb(bot))    # ...and re-sends the tasks keyboard
    tap(bot, "sch_weather")
    assert "nav:back" in datas(bot.sent[-1][2])                     # the weather picker
    press(bot, "sch:wx:08:00")
    assert "nav:back" in datas(bot.sent[-1][2])                     # ...and its confirmation
    press(bot, f"sch:o:{rid}")
    press(bot, f"sch:del:{rid}")
    press(bot, f"sch:delok:{rid}")                                  # what remains after a delete
    assert "nav:back" in datas(last_inline(bot)[1])
