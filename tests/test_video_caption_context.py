"""A forwarded video's caption is the setup of its joke. The retelling ignored it and read a
meme («Когда начальство внедрило ИИ…» over girls hauling sand) as a literal construction site."""
import os, sys, tempfile
from pathlib import Path
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_tasks
import tg_bot
import video_look


def test_the_retelling_is_told_a_mismatched_caption_is_a_joke():
    sys_ru = tg_tasks.retell_system("ru", True)
    assert "meme" in sys_ru and "literal" in sys_ru


def test_video_description_and_retelling_do_not_guess_gender_from_appearance():
    assert 'Do not infer a person\'s gender' in video_look._SHEET_SYSTEM
    assert 'Do not infer a person\'s gender' in tg_tasks.retell_system('ru', True)
    assert 'reliable context' in tg_tasks.retell_system('ru', True)


def test_only_plain_video_summary_skips_storyboard_zoom():
    assert tg_tasks._plain_video_summary('Кратко перескажи видео.\n<<<QUOTED MESSAGE -- context>>>')
    assert tg_tasks._plain_video_summary('Перескажи видео коротко')
    assert not tg_tasks._plain_video_summary('Перескажи видео и прочитай надпись на картинке')
    assert not tg_tasks._plain_video_summary('Как связаны видео и картинка?')


def test_video_generation_request_is_not_a_picture_question(monkeypatch):
    import graph
    read = {'is_question': False, 'about_picture': True, 'wants': ['generate_video']}
    monkeypatch.setattr(graph, '_picture_read', lambda ctx, state: read)
    assert not graph.is_picture_question(None, {})
    read['is_question'] = True
    assert graph.is_picture_question(None, {})


def test_storyboard_summary_keeps_overview_but_skips_extra_zoom(monkeypatch, tmp_path):
    import io
    from types import SimpleNamespace
    from PIL import Image
    import graph
    calls = []
    monkeypatch.setattr(graph, 'is_picture_question', lambda ctx, state: True)
    monkeypatch.setattr(graph, '_add_earlier_pictures', lambda ctx, state, summary: summary)
    monkeypatch.setattr(graph, '_close_look', lambda *args: calls.append('zoom') or 'zoomed')
    buf = io.BytesIO()
    Image.new('RGB', (20, 20)).save(buf, format='PNG')
    ctx = SimpleNamespace(video_summary_overview_only=True, deps=SimpleNamespace(
        downscale_image_bytes=lambda data: data, OUTPUT_DIR=tmp_path),
        vision_model=lambda **kwargs: 'overview', set_stage=lambda stage: None,
        remember=lambda *args: None, memory_text=lambda: '')
    state = {'image_data': buf.getvalue(), 'user_input': 'Кратко перескажи видео'}
    out = graph.vision_agent_node(ctx, state)
    assert out['vision_summary'] == 'overview' and not calls
    ctx.video_summary_overview_only = False
    out = graph.vision_agent_node(ctx, {'image_data': buf.getvalue(), 'user_input': 'Что за надпись?'})
    assert out['vision_summary'] == 'zoomed' and calls == ['zoom']


def test_video_comparison_drops_unsupported_viewer_guess_but_keeps_observations():
    reply = ('Визуальной связи между ними нет: на фото человек, а в видео монтаж. '
             'Как интерпретацию можно предположить, что это кадр человека, который смотрит такое видео во время отдыха.')
    result = tg_tasks._qualify_video_link_reply(reply, 'ru')
    assert 'на фото человек' in result
    assert 'смотрит такое видео' not in result


def test_video_comparison_does_not_repeat_existing_scope():
    reply = 'По доступным кадрам связи не видно.'
    assert tg_tasks._qualify_video_link_reply(reply, 'ru') == reply


def test_absence_of_video_image_connection_is_limited_to_sampled_frames():
    text = 'Прямой визуальной связи между ними нет: фото и видео разные.'
    assert tg_tasks._qualify_video_link_reply(text, 'ru').startswith('По доступным кадрам: ')
    assert tg_tasks._qualify_video_link_reply('Визуальной связи между ними не видно.', 'ru').startswith('По доступным кадрам: ')
    assert tg_tasks._qualify_video_link_reply('Возможно, это общий мотив.', 'ru') == 'Возможно, это общий мотив.'


def test_the_caption_line_exists_in_both_languages():
    assert "ИИ" in tg_bot._t("fwd_video_caption", "ru", caption="ИИ")
    assert "ИИ" in tg_bot._t("fwd_video_caption", "en", caption="ИИ")


def test_the_vision_pass_sees_the_caption(tmp_path):
    import subprocess
    clip = tmp_path / "c.mp4"
    r = subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=d=3:s=160x120:r=10", str(clip)],
                       capture_output=True)
    if r.returncode:
        import pytest; pytest.skip("no ffmpeg")
    asked = []
    video_look.look_at_video(None, str(clip), work_dir=tmp_path / "w",
                             vision=lambda frames, q, system: asked.append(q) or "x",
                             caption="Когда начальство внедрило ИИ")
    assert any("Когда начальство внедрило ИИ" in q for q in asked)


def test_the_speech_is_labelled_so_it_is_not_taken_for_the_caption():
    # A 5 s clip of a speech came back as «the sender's caption says…»: the bare transcript
    # beside a labelled caption was read as the caption (2026-10-15).
    heard = tg_bot._t("fwd_video_heard", "ru", heard="Разрешено всё")
    assert "Сказано в видео" in heard and "Разрешено всё" in heard
    sys_ru = tg_tasks.retell_system("ru", True)
    assert "Сказано в видео" in sys_ru and "never a caption" in sys_ru
