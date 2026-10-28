import threading
from types import SimpleNamespace

import pytest
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from gui_teacher_tab import TeacherTab, TeacherWorker


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def context(respond=None, speech=None):
    calls = []
    def answer(ctx, profile, history, text, target, native):
        calls.append((profile, history, text, target, native))
        return {'say': 'Guten Tag. Wie geht es dir?', 'translation': 'Добрый день. Как дела?',
                'correction': '<b>Исправление</b>', 'explanation': 'Объяснение',
                'profile': dict(profile, auto_level='A2'),
                'history': [*history, {'role': 'user', 'content': text}]}
    ctx = SimpleNamespace(custom_ref_wav='first.wav', cancel_event=threading.Event())
    ctx.modules = {'language_teacher': SimpleNamespace(respond=respond or answer),
        'translator': SimpleNamespace(voice_reference=lambda c: c.custom_ref_wav,
            lesson_speech=speech or (lambda *a, **kw: {'paths': ['de.wav', 'ru.wav'], 'combined_path': 'both.wav'}))}
    return ctx, calls


def wait(tab):
    for _ in range(200):
        QTest.qWait(10)
        if not tab.worker.isRunning():
            QApplication.processEvents()
            return
    pytest.fail('teacher worker did not finish')


def test_defaults_threaded_reply_and_plain_text_log(app):
    ctx, calls = context()
    tab = TeacherTab(SimpleNamespace(ctx=ctx))
    try:
        assert tab.target.count() == tab.native.count() == 10
        assert tab.target.currentData() == 'de' and tab.native.currentData() == 'ru'
        assert tab.level.currentData() == 'auto' and tab.voice_format.currentData() == 'combined'
        tab.text_in.setPlainText('Hallo'); tab.send_btn.click(); wait(tab)
        assert calls[0][2:] == ('Hallo', 'de', 'ru')
        assert '<b>Исправление</b>' in tab.result.toPlainText()
        assert 'Hallo' in tab.result.toPlainText() and tab.profile['auto_level'] == 'A2'
        assert tab.result.isReadOnly() and tab.play_btn.isEnabled() and tab.save_btn.isEnabled()
    finally:
        tab.shutdown(); tab.worker.wait(1000); tab.close()


def test_worker_snapshots_state_and_settings_voice(app):
    spoken = []
    ctx, calls = context(speech=lambda *a, **kw: spoken.append(kw['reference']) or {'paths': [], 'combined_path': ''})
    profile, history = {'level': 'B1'}, [{'role': 'user', 'content': 'old'}]
    worker = TeacherWorker(ctx, profile, history, 'Hallo', 'de', 'ru')
    profile['level'] = 'C1'; history[0]['content'] = 'changed'; ctx.custom_ref_wav = 'second.wav'
    worker.run()
    assert calls[0][0]['level'] == 'B1' and calls[0][1][0]['content'] == 'old'
    assert spoken == ['first.wav'] and not ctx.cancel_event.is_set()


def test_worker_speaks_native_rule_explanation_when_requested(app):
    spoken=[]
    def answer(ctx,profile,history,*args):
        return {'say':'Wie geht es dir?','translation':'Как дела?',
                'explanation':'Правило порядка слов.',
                'native_speech':'Как дела? Правило порядка слов.',
                'profile':profile,'history':history}
    def speech(ctx,say,target,native_speech,native,out_dir,**kwargs):
        spoken.append(native_speech)
        return {'paths':[],'combined_path':''}
    ctx,_=context(respond=answer,speech=speech)
    TeacherWorker(ctx,{},[],'Объясни порядок слов','de','ru').run()
    assert spoken==['Как дела? Правило порядка слов.']


def test_cancel_disables_controls_and_never_commits_or_cancels_host(app):
    entered = threading.Event()
    def answer(ctx, *args):
        entered.set(); ctx.cancel_event.wait(2)
        return {'say': 'late', 'translation': 'late', 'profile': {}, 'history': []}
    ctx, _ = context(respond=answer)
    tab = TeacherTab(SimpleNamespace(ctx=ctx))
    try:
        tab.text_in.setPlainText('Hallo'); tab.send_btn.click()
        assert entered.wait(1)
        assert not tab.target.isEnabled() and not tab.text_in.isEnabled() and tab.stop_btn.isEnabled()
        assert not tab.clear_btn.isEnabled() and not tab.voice_format.isEnabled()
        tab.stop_btn.click(); wait(tab)
        assert tab.history == [] and not tab.result.toPlainText()
        assert not ctx.cancel_event.is_set() and tab.send_btn.isEnabled()
    finally:
        tab.shutdown(); tab.worker.wait(2000); tab.close()


def test_language_and_topic_changes_reset_only_lesson_state(app):
    tab = TeacherTab()
    try:
        tab.history = [{'role': 'user', 'content': 'old'}]; tab.profile['auto_level'] = 'B2'
        tab.native.setCurrentIndex(tab.native.findData('en'))
        assert not tab.history and tab.profile['auto_level'] == 'B2'
        tab.history = [{'role': 'user', 'content': 'old'}]
        tab.topic.setText('Travel'); assert not tab.history
        tab.target.setCurrentIndex(tab.target.findData('fr'))
        assert tab.profile['auto_level'] == 'A1'
        tab.level.setCurrentIndex(tab.level.findData('B1'))
        assert tab.profile['level'] == 'B1'
    finally:
        tab.close()


def test_speech_failure_preserves_text_lesson(app):
    def fail(*args, **kwargs):
        raise ValueError('voice unavailable')
    ctx, _ = context(speech=fail)
    tab = TeacherTab(SimpleNamespace(ctx=ctx))
    try:
        tab.text_in.setPlainText('Hallo'); tab.send_btn.click(); wait(tab)
        assert 'Guten Tag' in tab.result.toPlainText() and tab.history
        assert 'voice unavailable' in tab.status.text() and not tab.play_btn.isEnabled()
    finally:
        tab.shutdown(); tab.worker.wait(1000); tab.close()


def test_model_failure_preserves_existing_lesson_and_does_not_speak(app):
    def fail(*args): raise ValueError('model unavailable')
    ctx, _ = context(respond=fail, speech=lambda *a, **kw: pytest.fail('must not synthesize'))
    tab = TeacherTab(SimpleNamespace(ctx=ctx))
    tab.history = [{'role': 'user', 'content': 'previous'}]
    tab.result.setPlainText('previous lesson')
    try:
        tab.text_in.setPlainText('Hallo'); tab.send_btn.click(); wait(tab)
        assert tab.history == [{'role': 'user', 'content': 'previous'}]
        assert tab.result.toPlainText() == 'previous lesson'
        assert 'model unavailable' in tab.status.text() and tab.text_in.toPlainText() == 'Hallo'
    finally:
        tab.shutdown(); tab.worker.wait(1000); tab.close()


def test_queued_done_after_cancel_is_ignored(app):
    tab = TeacherTab()
    token = threading.Event(); token.set()
    tab.worker = SimpleNamespace(cancel_event=token)
    try:
        tab._done({'profile': {}, 'history': ['late']})
        assert tab.history == [] and not tab.result.toPlainText()
    finally:
        tab.close()


def test_text_fallback_save_and_empty_input(app, tmp_path):
    ctx, calls = context()
    tab = TeacherTab(SimpleNamespace(ctx=ctx))
    try:
        tab.send_btn.click(); assert not calls and tab.worker is None
        tab.result.setPlainText('Hallo\nПривет')
        path = tmp_path / 'lesson.txt'
        tab.file_dialog = SimpleNamespace(getSaveFileName=lambda *a: (str(path), ''))
        tab._save()
        assert path.read_text('utf8') == 'Hallo\nПривет'
    finally:
        tab.close()


def test_separate_save_has_two_files_and_play_uses_combined(app, tmp_path):
    tab = TeacherTab()
    paths = [tmp_path / name for name in ('de.wav', 'ru.wav', 'both.wav')]
    for index, path in enumerate(paths): path.write_bytes(bytes([index]))
    tab.paths, tab.combined_path = [str(p) for p in paths[:2]], str(paths[2])
    tab.result.setPlainText('lesson'); tab._busy(False)
    destinations = iter([str(tmp_path / 'target.wav'), str(tmp_path / 'native.wav')])
    tab.file_dialog = SimpleNamespace(getSaveFileName=lambda *a: (next(destinations), ''))
    played = []
    tab.player = SimpleNamespace(play=played.append, stop=lambda: None)
    try:
        tab.voice_format.setCurrentIndex(tab.voice_format.findData('separate')); tab.save_btn.click()
        assert (tmp_path / 'target.wav').read_bytes() == b'\x00'
        assert (tmp_path / 'native.wav').read_bytes() == b'\x01'
        tab.play_btn.click(); assert played == [str(paths[2])]
        tab.clear_btn.click(); assert not tab.result.toPlainText() and not tab.history
    finally:
        tab.close()


def test_audio_input_uses_original_transcription_and_resets_after_reply(app, tmp_path):
    ctx, calls = context()
    source = str(tmp_path / 'lesson.ogg')
    heard = []
    ctx.modules['translator'].transcribe = lambda c, path: heard.append(path) or 'Ich gehen morgen.'
    tab = TeacherTab(SimpleNamespace(ctx=ctx))
    tab.file_dialog = SimpleNamespace(getOpenFileName=lambda *a: (source, ''))
    try:
        tab.audio_btn.click()
        assert tab.source == source and not tab.text_in.isEnabled()
        tab.send_btn.click(); wait(tab)
        assert heard == [source] and calls[0][2] == 'Ich gehen morgen.'
        assert 'Ich gehen morgen.' in tab.result.toPlainText()
        assert not tab.source and tab.text_in.isEnabled()
        tab.audio_btn.click(); tab.remove_audio_btn.click()
        assert not tab.source and tab.text_in.isEnabled()
    finally:
        tab.shutdown(); tab.worker.wait(1000); tab.close()


@pytest.mark.parametrize('control,status',[('pause_btn','paused'),('finish_btn','ended')])
def test_pause_or_finish_held_worker_suppresses_late_reply_then_resumes(app,control,status):
    entered, release = threading.Event(), threading.Event()
    def answer(ctx,*args):
        entered.set(); release.wait(2)
        return {'say':'late','translation':'late','profile':{},'history':['late']}
    ctx,_ = context(respond=answer)
    tab=TeacherTab(SimpleNamespace(ctx=ctx)); stopped=[]
    tab.player=SimpleNamespace(stop=lambda:stopped.append(True))
    tab.history=[{'role':'user','content':'earlier'}]
    tab.result.setPlainText('earlier lesson')
    try:
        tab.text_in.setPlainText('Hello');tab.send_btn.click()
        assert entered.wait(1) and tab.pause_btn.isEnabled() and tab.finish_btn.isEnabled()
        getattr(tab,control).click()
        assert tab.lesson_status==status and tab.worker.cancel_event.is_set()
        assert stopped and not ctx.cancel_event.is_set()
        release.set();wait(tab)
        assert tab.lesson_status==status and ('paused' if status=='paused' else 'finished') in tab.status.text()
        assert not tab.send_btn.isEnabled() and not tab.text_in.isEnabled() and not tab.audio_btn.isEnabled()
        assert tab.history==[{'role':'user','content':'earlier'}] and tab.result.toPlainText()=='earlier lesson'
        tab.pause_btn.click()
        assert tab.lesson_status=='active' and tab.send_btn.isEnabled() and tab.audio_btn.isEnabled()
        assert tab.history==[{'role':'user','content':'earlier'}]
    finally:
        release.set();tab.shutdown();tab.worker.wait(2000);tab.close()


def test_finish_returns_to_translator_and_clear_starts_fresh_with_preferences(app):
    from PyQt5.QtWidgets import QTabWidget, QWidget
    sections=QTabWidget();sections.addTab(QWidget(),'Translator');sections.addTab(QWidget(),'Teacher')
    sections.setCurrentIndex(1)
    tab=TeacherTab(SimpleNamespace(translator_tab=SimpleNamespace(sections=sections)))
    try:
        tab.level.setCurrentIndex(tab.level.findData('B1'));tab.topic.setText('Travel')
        tab.history=[{'role':'user','content':'earlier'}];tab.profile['auto_level']='B2'
        tab.result.setPlainText('earlier lesson')
        tab.finish_btn.click()
        assert tab.lesson_status=='ended' and sections.currentIndex()==0
        assert tab.history and tab.result.toPlainText()=='earlier lesson'
        assert not tab.send_btn.isEnabled() and not tab.audio_btn.isEnabled()
        tab.clear_btn.click()
        assert tab.lesson_status=='active' and tab.send_btn.isEnabled() and not tab.history
        assert tab.profile=={'level':'B1','auto_level':'A1','topic':'Travel','streak':0}
        assert tab.target.currentData()=='de' and tab.native.currentData()=='ru'
    finally:
        tab.close();sections.close()


def test_finish_standalone_and_late_progress_do_not_overwrite_status(app):
    tab=TeacherTab()
    try:
        tab.finish_btn.click()
        status=tab.status.text()
        tab._progress(1,2,'late');tab._failed('late error');tab._finished()
        tab._done({'profile':{},'history':['late']})
        assert tab.lesson_status=='ended' and tab.status.text()==status and not tab.history
        tab.pause_btn.click()
        assert tab.lesson_status=='active' and tab.text_in.isEnabled()
    finally:
        tab.close()
