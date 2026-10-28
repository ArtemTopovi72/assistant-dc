"""An unrelated answer must not advertise actions on a historical photo."""
from PIL import Image
import pytest
import tg_bot as T
from fakes import reads
from test_album_reaches_video import FakeCtx
from test_tg_incoming_durability import make_bot


@pytest.mark.parametrize('selection,about_picture,picture_question,upload,expect_actions', [
    ('context', False, False, False, False),
    ('context', False, True, False, False),
    ('context', True, True, False, True),
    ('explicit', False, False, False, True),
    ('upload', False, False, True, True),
])
def test_photo_actions_belong_to_the_current_reply(tmp_path, selection, about_picture, picture_question, upload, expect_actions):
    bot = make_bot(tmp_path)
    ctx = FakeCtx()
    vars(ctx).update(reads(read=lambda text: {'needs_tool': False, 'about_picture': about_picture,
                                            'is_question': True}, yes=lambda *a: False))
    bot._get_ctx = lambda: ctx; bot._get_graph = lambda: object()
    bot._get_base_state = lambda: {'messages': []}
    class Graph:
        def invoke(self, state):
            return {'final_answer': 'Answer', 'picture_question': picture_question,
                    'messages': state.get('messages', [])}
    bot.graph_builder = lambda ctx: Graph()
    path = tmp_path / 'photo.png'; Image.new('RGB', (20, 20), 'blue').save(path)
    sess = bot._get_session(101); sess.reply_mode = 'text'
    image = T._log_image(sess, str(path)); sess.turn_image = image
    sent = []; bot._send_text = lambda cid, text, **kw: sent.append(kw.get('keyboard'))
    bot._execute_task(T._Task(task_id='reply', chat_id=101, user_text='Find the official documentation',
        image_id=image, image_selection=selection, image_path=str(path) if upload else '', image_is_photo=upload))
    callbacks = [b.get('callback_data', '') for kb in sent if isinstance(kb, dict)
                 for row in kb.get('inline_keyboard', []) for b in row]
    assert any(x.startswith('ask:') for x in callbacks) == expect_actions
