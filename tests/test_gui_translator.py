import threading
from types import SimpleNamespace

import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest

from gui_translator_tab import TranslatorTab
from gui_busy_state import BusyStateMixin


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def test_text_translation_is_threaded_and_copy_and_clear_work(app):
    calls = []
    ctx = SimpleNamespace(text_model=lambda c, s, u, **kw: calls.append((s,u)) or '{"translation":"Bonjour"}')
    tab = TranslatorTab(SimpleNamespace(ctx=ctx))
    tab.target.setCurrentIndex(tab.target.findData('fr'))
    tab.text_in.setPlainText('Hello')
    tab.translate_btn.click()
    for _ in range(200):
        QTest.qWait(10)
        if tab.translate_btn.isEnabled(): break
    assert tab.result.toPlainText() == 'Bonjour' and 'French' in calls[0][0]
    tab.copy_btn.click(); assert QApplication.clipboard().text() == 'Bonjour'
    tab.clear_btn.click(); assert not tab.text_in.toPlainText() and not tab.result.toPlainText()
    tab.close()


def test_teacher_is_a_subsection_inside_translator(app):
    tab=TranslatorTab()
    try:
        assert tab.sections.count()==2
        assert tab.sections.widget(1) is tab.teacher_tab
        assert tab.teacher_tab.target.currentData()=='de'
        assert tab.teacher_tab.parentWidget() is not None
    finally:
        tab.shutdown();tab.close()


def test_nested_section_titles_switch_from_russian_back_to_english(app):
    import gui_i18n
    previous=gui_i18n.LANG
    gui_i18n.install('ru')
    tab=TranslatorTab()
    try:
        assert tab.sections.tabText(1)==gui_i18n.tr('Language teacher')
        gui_i18n.set_language('en')
        assert [tab.sections.tabText(i) for i in range(2)]==['Translator','Language teacher']
    finally:
        tab.shutdown();tab.close();tab.deleteLater();app.processEvents()
        gui_i18n.set_language(previous)


def test_translator_worker_is_cancelled_at_window_shutdown(app):
    entered, cancel = threading.Event(), threading.Event()
    def model(ctx, *a, **kw):
        entered.set(); ctx.cancel_event.wait(3)
        return 'Late result'
    class Host(BusyStateMixin): pass
    host = Host(); host.ctx = SimpleNamespace(text_model=model, cancel_event=cancel)
    host.translator_tab = tab = TranslatorTab(host)
    tab.text_in.setPlainText('Hello'); tab.translate_btn.click()
    try:
        assert entered.wait(1)
        assert tab.worker in host._live_threads()
        assert host._join_live_threads(1000) == []
        app.processEvents()
        assert not cancel.is_set() and not tab.worker.isRunning()
        assert tab.result.toPlainText() != 'Late result'
    finally:
        tab.worker.cancel(); tab.worker.wait(3000); tab.close()


def test_clear_and_cancel_stop_voice_playback(app):
    stopped = []
    tab = TranslatorTab()
    tab.player = SimpleNamespace(stop=lambda: stopped.append(True))
    tab._clear(); tab._cancel()
    assert len(stopped) == 2
    tab.close()


def test_media_mode_returns_separate_audio_and_subtitle_actions(app,tmp_path):
    calls=[]
    result={'text':'Translated dialogue','path':str(tmp_path/'out.mp4'),
            'audio_path':str(tmp_path/'out.wav'),'subtitle_path':str(tmp_path/'out.srt')}
    ctx=SimpleNamespace(modules={'translator':SimpleNamespace(dub=lambda *args:calls.append(args) or result)})
    tab=TranslatorTab(SimpleNamespace(ctx=ctx))
    assert tab.mode.findData('dub')>=0
    tab.mode.setCurrentIndex(tab.mode.findData('dub'));tab.source=str(tmp_path/'in.mp4')
    tab.translate_btn.click()
    for _ in range(200):
        QTest.qWait(10)
        if tab.translate_btn.isEnabled():break
    assert calls and tab.output_path==result['path']
    assert tab.audio_btn.isEnabled() and tab.subtitles_btn.isEnabled()
    tab.clear_btn.click()
    assert not tab.audio_btn.isEnabled() and not tab.subtitles_btn.isEnabled()
    tab.close()


def test_text_voice_mode_needs_no_input_file_and_uses_settings_voice(app,tmp_path):
    calls=[]
    ctx=SimpleNamespace(custom_ref_wav=str(tmp_path/'chosen.wav'),modules={
        'translator':SimpleNamespace(text_voice=lambda *args,**kw:calls.append((args,kw)) or
                        {'text':'Bonjour','path':str(tmp_path/'voice.wav')})})
    tab=TranslatorTab(SimpleNamespace(ctx=ctx))
    assert tab.mode.findData('tts')>=0
    tab.mode.setCurrentIndex(tab.mode.findData('tts'));tab.text_in.setPlainText('Hello')
    tab.translate_btn.click()
    for _ in range(200):
        QTest.qWait(10)
        if tab.translate_btn.isEnabled():break
    assert calls and calls[0][1]['reference']==str(tmp_path/'chosen.wav')
    assert tab.result.toPlainText()=='Bonjour' and tab.play_btn.isEnabled()
    tab.close()
