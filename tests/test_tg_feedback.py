"""👍/👎 under answers: a vote toggles, counts are per person, the store feeds the chart,
and a broken store never costs the user the answer."""
import os, sys, tempfile, types
os.environ.setdefault("F5_TEST_RUN", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_feedback as F


def fresh():
    return F._FeedbackStore(os.path.join(tempfile.mkdtemp(), "fb.db"))


def test_vote_toggles_and_switches():
    s = fresh()
    rid = s.record_reply(1, "q", "a")
    assert s.vote(rid, 7, 1) == {"up": 1, "down": 0, "mine": 1}
    assert s.vote(rid, 7, -1) == {"up": 0, "down": 1, "mine": -1}
    assert s.vote(rid, 7, -1) == {"up": 0, "down": 0, "mine": 0}


def test_votes_are_per_person_and_unknown_reply_is_ignored():
    s = fresh()
    rid = s.record_reply(1, "q", "a")
    s.vote(rid, 1, 1)
    assert s.vote(rid, 2, 1)["up"] == 2
    assert s.vote("nope", 1, 1) == {}


def test_aggregates():
    s = fresh()
    a, b = s.record_reply(1, "q1", "a1", kind="search"), s.record_reply(1, "q2", "a2")
    s.vote(a, 1, 1)
    s.vote(b, 1, -1)
    sm = s.summary(7)
    assert (sm["replies"], sm["up"], sm["down"], sm["rate"]) == (2, 1, 1, 0.5)
    d = s.daily(7)
    assert len(d) == 7 and d[-1]["up"] == 1 and d[-1]["down"] == 1 and d[0]["replies"] == 0
    assert {k["kind"] for k in s.by_kind(7)} == {"search", "chat"}
    assert [r["id"] for r in s.recent_down()] == [b]


def test_keyboard_marks_my_side_and_merges_under_inline_rows():
    kb = F.keyboard("abc", {"up": 2, "down": 0, "mine": 1})
    row = kb["inline_keyboard"][0]
    assert row[0]["text"] == "👍 2✓" and row[1]["text"] == "👎"
    assert row[0]["callback_data"] == "fb:abc:u"
    merged = F.with_feedback({"inline_keyboard": [[{"text": "x", "callback_data": "x"}]]}, "abc")
    assert len(merged["inline_keyboard"]) == 2
    assert len(F.with_feedback(None, "abc")["inline_keyboard"]) == 1


def test_press_updates_buttons_and_toasts():
    from tg_callbacks import CallbackMixin
    F.redirect_store(os.path.join(tempfile.mkdtemp(), "fb.db"))
    rid = F.store().record_reply(5, "q", "a")
    posts = []
    stub = types.SimpleNamespace(_api_post=lambda m, d: posts.append((m, d)))
    msg = {"message_id": 9, "chat": {"id": 5}, "reply_markup": F.keyboard(rid)}
    CallbackMixin._cb_feedback(stub, {"id": "c", "from": {"id": 5}}, msg, 5, f"fb:{rid}:d")
    edit = dict(posts)["editMessageReplyMarkup"]
    assert edit["reply_markup"]["inline_keyboard"][-1][1]["text"] == "👎 1✓"
    assert dict(posts)["answerCallbackQuery"]["text"] == "👎"


def test_broken_store_still_answers_the_press():
    from tg_callbacks import CallbackMixin
    posts = []
    stub = types.SimpleNamespace(_api_post=lambda m, d: posts.append(m))
    CallbackMixin._cb_feedback(stub, {"id": "c"}, {}, 5, "fb:garbage")
    assert posts == ["answerCallbackQuery"]


def test_failed_recording_keeps_the_original_keyboard():
    from tg_tasks import TaskRunnerMixin
    F.redirect_store(os.path.join(tempfile.mkdtemp(), "no_such_dir", "x", "fb.db"))
    reply_kb = {"keyboard": [[{"text": "a"}]]}
    out = TaskRunnerMixin._feedback_kb(types.SimpleNamespace(), reply_kb, 1, "q", "a", types.SimpleNamespace())
    assert out is reply_kb


def test_reply_keyboard_is_not_mixed_with_the_feedback_row():
    from tg_tasks import TaskRunnerMixin
    F.redirect_store(os.path.join(tempfile.mkdtemp(), "fb.db"))
    out = TaskRunnerMixin._feedback_kb(types.SimpleNamespace(), {"keyboard": [[{"text": "a"}]]},
                                       1, "q", "a", types.SimpleNamespace())
    assert "keyboard" not in out and len(out["inline_keyboard"]) == 1


def test_tab_renders_with_and_without_votes():
    from PyQt5.QtWidgets import QApplication
    from gui_feedback_tab import FeedbackTab
    app = QApplication.instance() or QApplication([])
    s = fresh()
    tab = FeedbackTab(store=lambda: s)
    tab.resize(700, 600)
    tab.grab()
    rid = s.record_reply(1, "q", "a")
    s.vote(rid, 1, -1)
    tab.refresh()
    tab.grab()
    assert tab._vals["down"].text() == "1" and "q" in tab._bad.toPlainText()
    assert app


def test_a_delivered_result_gets_the_row_under_its_own_buttons():
    """Only text answers could be rated: a voice reply, a picture, a clip or a song had
    no 👍/👎 at all. A result Telegram accepted is filed under its kind, with what was
    asked, and the row goes under the buttons the message already has."""
    import json
    import tg_transport
    F.redirect_store(os.path.join(tempfile.mkdtemp(), "fb.db"))
    posted = []
    bot = types.SimpleNamespace(_api_post=lambda m, d: posted.append((m, d)),
                                _get_session=lambda cid: types.SimpleNamespace(last_task_text="old"),
                                _restore_result_menu=lambda *args: None)
    resp = types.SimpleNamespace(json=lambda: {"result": {"message_id": 42}})
    own = {"inline_keyboard": [[{"text": "🔄", "callback_data": "again"}]]}
    with tg_transport.replying_to(7, 0, asked="спой про кота"):
        tg_transport.TransportMixin._attach_feedback(bot, 7, resp, "music", "song.mp3", own)
    method, data = posted[-1]
    rows = json.loads(data["reply_markup"])["inline_keyboard"]
    assert method == "editMessageReplyMarkup" and data["message_id"] == 42
    assert rows[0] == own["inline_keyboard"][0] and rows[1][0]["callback_data"].startswith("fb:")
    kinds = {k["kind"]: k for k in F.store().by_kind()}
    assert kinds["music"]["replies"] == 1
    rid = rows[1][0]["callback_data"].split(":")[1]
    F.store().vote(rid, 1, -1)
    assert F.store().recent_down()[0]["user"] == "спой про кота", "what was asked, not the last task"
    posted.clear()
    tg_transport.TransportMixin._attach_feedback(bot, 7, resp, "", "x")
    assert posted == [], "a send that opted out gets no row"


def test_kinds_chart_names_every_kind():
    from PyQt5.QtWidgets import QApplication
    from gui_feedback_tab import FeedbackTab, kind_label
    app = QApplication.instance() or QApplication([])
    s = fresh()
    for kind in ("chat", "voice", "video", "music", "research", "retell"):
        s.vote(s.record_reply(1, "q", "a", kind=kind), 1, 1 if kind != "video" else -1)
    tab = FeedbackTab(store=lambda: s)
    tab.resize(800, 700)
    tab.grab()
    assert len(tab._kinds._kinds) == 6 and kind_label("video") == "🎬 Video"
    assert "🎬" in tab._bad.toPlainText()
    assert app


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
