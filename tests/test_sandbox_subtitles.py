from types import SimpleNamespace

import sandbox_access
from code_sandbox import Sandbox
from tool_code_handlers import _handle_transcribe_media
from graph_personality import _forced_subtitles
from tool_code_handlers import _subtitle_cues


def test_subtitles_use_isolated_audio_and_existing_whisper(tmp_path):
    box = Sandbox(tmp_path)
    box.resolve('clip.mp4').write_bytes(b'video')
    class Whisper:
        def transcribe(self, path, **kwargs):
            return iter([SimpleNamespace(start=.5, end=1.8, text=' Привет, мир!')]), None
    import threading
    def run(box, code, **kwargs):
        assert kwargs['allow_host'] is False
        import re
        wav = re.search(r"'([^']+\.wav)'", code).group(1)
        box.resolve('.agent').mkdir(exist_ok=True)
        box.resolve(wav).write_bytes(b'isolated wav')
        return SimpleNamespace(ok=True, backend='docker', output='')
    ctx = SimpleNamespace(sandbox=box, sandbox_user=SimpleNamespace(prefs={'sandbox': sandbox_access.CODE}),
        models=SimpleNamespace(whisper=Whisper()), asr_lock=threading.Lock(),
        deps=SimpleNamespace(run_python=run))
    result = _handle_transcribe_media(ctx, {}, {'path':'clip.mp4'})
    files = list(tmp_path.glob('clip_*.srt'))
    assert len(files) == 1 and files[0].name in result
    assert '00:00:00,500 --> 00:00:01,800' in files[0].read_text(encoding='utf8')
    assert 'Привет, мир!' in files[0].read_text(encoding='utf8')


def test_automatic_subtitles_force_whisper_only_once():
    tools = [{'function': {'name': 'transcribe_media'}}]
    assert _forced_subtitles('Автоматические субтитры по речи', tools, set()) == 'transcribe_media'
    assert not _forced_subtitles('Автоматические субтитры по речи', tools, {'transcribe_media'})
    assert not _forced_subtitles('Наложи готовые субтитры из файла', tools, set())


def test_subtitle_cues_split_long_speech_with_word_timing():
    words = [SimpleNamespace(start=i * .5, end=(i + 1) * .5, word=' слово') for i in range(12)]
    cues = _subtitle_cues([SimpleNamespace(start=0, end=6, text=' '.join(w.word for w in words), words=words)])
    assert len(cues) >= 2
    assert all(end - start <= 4.1 and len(text) <= 48 for start, end, text in cues)


def test_subtitle_transcription_checks_cancellation_between_segments():
    import pytest
    segment = SimpleNamespace(start=0, end=1, text='hello', words=None)
    with pytest.raises(RuntimeError, match='cancelled'):
        _subtitle_cues([segment], lambda: True)


def test_already_transcribed_attachment_does_not_force_sandbox_tool():
    from graph_personality import _forced_intent
    def read(ctx, text, previous="", attached=""):
        return {"needs_tool": True, "must_call": "transcribe_media"}
    ctx = SimpleNamespace(last_image_path=None, media_transcript="hello from the clip", intent_read=read)
    tools = [{"function": {"name": "transcribe_media"}}]
    assert _forced_intent(ctx, {"messages": []}, "What was said in the recording?", False, tools, ()) == ""
    assert _forced_intent(ctx, {"messages": []}, "Create an SRT subtitle file", False, tools, ()) == "transcribe_media"
    ctx.media_transcript = ""
    assert _forced_intent(ctx, {"messages": []}, "What was said in the recording?", False, tools, ()) == "transcribe_media"


def test_missing_sandbox_path_uses_current_attachment_transcript(tmp_path):
    ctx = SimpleNamespace(sandbox=Sandbox(tmp_path), media_transcript="hello from the clip",
                          sandbox_user=SimpleNamespace(prefs={"sandbox": sandbox_access.CODE}))
    out = _handle_transcribe_media(ctx, {"user_input_original": "What was said?"}, {"path": "input_file_0.mp4"})
    assert "hello from the clip" in out and "TOOL ERROR" not in out


def test_transcript_is_preserved_in_the_durable_task():
    from tg_queue_backends import _Task
    task = _Task(task_id="1", chat_id=123, user_text="question", media_transcript="heard words")
    assert _Task.from_dict(task.to_dict()).media_transcript == "heard words"
    assert _Task.from_dict({"task_id": "2", "chat_id": 123, "user_text": "old"}).media_transcript == ""
