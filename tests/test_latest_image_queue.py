"""A follow-up must use the picture delivered before it starts, not an older one."""
import tg_bot as T
from test_tg_incoming_durability import make_bot


def test_context_picture_is_bound_when_claimed_not_when_queued(tmp_path):
    bot = make_bot(tmp_path)
    sess = bot._get_session(101)
    old = T._log_image(sess, str(tmp_path / "old.png"))
    new = T._log_image(sess, str(tmp_path / "new.png"))
    sess.turn_image = old
    producing = sess.begin_context_turn()
    task = T._Task(task_id="followup", chat_id=101, user_text="стиль — карандаш",
                   image_selection="context", context_generation=sess.context_generation)
    assert bot._persist_task(task)
    assert not task.image_id
    assert sess.record_delivered_image(producing, new)
    assert sess.turn_image == new
    from tg_tasks import _target_id_of
    assert _target_id_of(task, sess) == new


def test_picture_followup_waits_for_running_render(tmp_path):
    bot = make_bot(tmp_path)
    bot._running = True
    followup = T._Task(task_id="followup", chat_id=101, user_text="стиль — карандаш",
                       image_selection="context", intent_snapshot={"reads": [
                           [["стиль — карандаш", "", ""], {"wants": ["redraw_image"],
                                                              "about_picture": True}]]})
    bot._backend.push(followup)
    bot._chat_busy[101] = 1
    bot._chat_interruptible[101] = True
    bot._running_task[101] = [T._Task(task_id="render", chat_id=101,
                                      user_text="нарисуй лису")]
    assert bot._claim_task() is None
    assert bot._backend.chat_depth(101) == 1


def test_picture_followup_is_not_folded_into_running_render(tmp_path):
    bot = make_bot(tmp_path)
    running = T._Task(task_id="render", chat_id=101, user_text="нарисуй лису")
    bot._running_task[101] = [running]
    bot._steer_inboxes[running.task_id] = object()
    followup = T._Task(task_id="followup", chat_id=101, user_text="стиль — карандаш",
                       image_selection="context", intent_snapshot={"reads": [
                           [["стиль — карандаш", "", ""], {"wants": ["redraw_image"],
                                                              "about_picture": True}]]})
    assert bot._try_steer(101, followup) is False


def test_unicode_slash_command_does_not_join_debounce(tmp_path):
    bot = make_bot(tmp_path)
    seen = []
    bot._handle_command = lambda chat_id, text: seen.append((chat_id, text))
    bot._dispatch({'message': {'message_id': 1, 'chat': {'id': 101, 'type': 'private'},
                               'from': {'id': 101}, 'text': '/несуществующая_команда'}})
    assert seen == [(101, '/несуществующая_команда')]
    assert not bot._incoming.pending()
