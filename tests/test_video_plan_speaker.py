from types import SimpleNamespace

import tg_video_plan
import video


def test_condense_does_not_drop_explicit_speaker(monkeypatch):
    original = [
        {'text': 'The camera pulls back.', 'sec': 5},
        {'text': 'The woman on the right says "Hello".', 'sec': 5},
    ]
    sess = SimpleNamespace(video_plan=original[:], video_plan_request='The woman on the right says "Hello".')
    messages = []
    bot = SimpleNamespace(reads=object(), _send_text=lambda chat, text: messages.append(text))
    monkeypatch.setattr(video, 'condense_script', lambda *a, **kw: 'Camera moves. The woman says "Hello".')
    monkeypatch.setattr(video, 'split_script', lambda text: [text])
    monkeypatch.setattr(video, 'dictated_speaker', lambda ctx, script, request: 'right' if 'on the right' in script else '')
    tg_video_plan.VideoPlanMixin._squeeze_plan(bot, 1, sess, 'ru')
    assert sess.video_plan == original
    assert messages


def test_condense_keeps_plan_when_speaker_cannot_be_verified(monkeypatch):
    original = [
        {'text': 'The camera pulls back.', 'sec': 5},
        {'text': 'The woman on the right says "Hello".', 'sec': 5},
    ]
    sess = SimpleNamespace(video_plan=original[:], video_plan_request='The woman on the right says "Hello".')
    messages = []
    bot = SimpleNamespace(reads=object(), _send_text=lambda chat, text: messages.append(text))
    monkeypatch.setattr(video, 'dictated_speaker', lambda *a: '')
    monkeypatch.setattr(video, 'condense_script', lambda *a, **kw: 'The woman says "Hello".')
    monkeypatch.setattr(video, 'split_script', lambda text: [text])
    tg_video_plan.VideoPlanMixin._squeeze_plan(bot, 1, sess, 'ru')
    assert sess.video_plan == original
    assert messages
