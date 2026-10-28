from types import SimpleNamespace

import pytest
import tg_bot as T


@pytest.fixture
def bot(tmp_path):
    T.redirect_data_dir(tmp_path)
    b = T.TelegramBot('123:TEST', lambda: None, lambda: None, lambda: {}, silent_mode=True)
    b.sent = []; b.jobs = []
    b._send_text = lambda cid, text, **kw: b.sent.append((text, kw.get('keyboard'))) or 1
    b._backend = SimpleNamespace(push=b.jobs.append, depth=lambda: 1, position_of=lambda tid: 1)
    b._api_post = lambda *a, **kw: {'ok': True, 'result': {'message_id': 1}}
    b._get_session(17).lang = 'ru'
    return b


def press(bot, key):
    bot._resolve_and_push(17, [{'type': 'text', 'text': T._b(key, 'ru')}])


def test_language_and_mode_persist_and_back_leaves_translator(bot):
    press(bot, 'translator')
    assert bot._get_session(17).menu == 'translator'
    press(bot, 'tr_lang_fr'); press(bot, 'tr_image')
    sess = bot._get_session(17)
    reloaded = T._Session(17, sess.to_dict())
    assert reloaded.translation_target == 'fr' and reloaded.translation_mode == 'image'
    press(bot, 'back')
    assert sess.menu == '' and not bot.jobs


def test_translation_uses_submitted_language_not_later_setting(bot):
    calls = []
    bot.modules = {'translator': SimpleNamespace(translate=lambda ctx, text, target: calls.append((text,target)) or 'Bonjour')}
    press(bot, 'translator'); press(bot, 'tr_lang_fr')
    bot._resolve_and_push(17, [{'type': 'text', 'text': 'Good morning'}])
    press(bot, 'tr_lang_de')
    assert len(bot.jobs) == 1
    from tg_translator import execute
    execute(bot, SimpleNamespace(is_cancelled=lambda: False), bot.jobs[0])
    assert calls == [('Good morning', 'fr')]
    assert any(text == '<b>🌐 🇫🇷 Français</b>\nBonjour' for text, _ in bot.sent)


def test_clear_prevents_a_late_translation_response(bot):
    bot.modules = {'translator': SimpleNamespace(translate=lambda *a: 'Bonjour')}
    press(bot, 'translator')
    bot._resolve_and_push(17, [{'type': 'text', 'text': 'Hello'}])
    bot._get_session(17).context_generation += 1
    from tg_translator import execute
    execute(bot, SimpleNamespace(is_cancelled=lambda: False), bot.jobs[0])
    assert not any(text == 'Bonjour' for text, _ in bot.sent)


def test_mode_button_and_navigation_never_become_translation_input(bot):
    press(bot, 'translator'); press(bot, 'tr_voice'); press(bot, 'tr_languages')
    assert not bot.jobs
    assert 'Français' in str(bot.sent[-1][1])


def test_translation_enters_the_durable_task_queue(bot):
    queued = []
    bot._backend = SimpleNamespace(push=queued.append, depth=lambda: 1, position_of=lambda tid: 1)
    press(bot, 'translator')
    bot._resolve_and_push(17, [{'type': 'text', 'text': 'Hello'}])
    assert len(queued) == 1
    restored = T._Task.from_dict(queued[0].to_dict())
    assert restored.translation['items'][0]['text'] == 'Hello'
    assert restored.task_id in bot._pending_journal


@pytest.mark.parametrize('item', [
    {'type':'fwd_voice','media':'video','seconds':10**6,'file_id':'clip'},
    {'type':'document','filename':'clip.mp4','file_id':'clip'},
    {'type':'document','filename':'speech.mp3','file_id':'clip'},
])
def test_media_mode_delivers_video_audio_and_subtitles(bot,tmp_path,item):
    from tg_translator import execute
    calls=[]; sent=[]
    paths={k:str(tmp_path/n) for k,n in [('path','dub.mp4'),('audio_path','dub.wav'),('subtitle_path','dub.srt')]}
    bot.modules={'translator':SimpleNamespace(dub=lambda ctx,src,target,out: calls.append(src) or
                    {'text':'Translated dialogue',**paths})}
    bot._dl_bytes=lambda fid:b'media'
    bot._send_video=lambda cid,p,**kw: sent.append(p) or True
    bot._send_document=lambda cid,p,**kw: sent.append(p) or True
    press(bot,'translator');press(bot,'tr_dub')
    bot._resolve_and_push(17,[item])
    execute(bot,SimpleNamespace(is_cancelled=lambda:False),bot.jobs[-1])
    assert calls and sent==list(paths.values())
    assert bot._get_session(17).translation_mode=='dub'


def test_translation_progress_is_visible_under_the_existing_status(bot):
    from tg_translator import execute
    edits=[]
    bot._edit_text=lambda cid,mid,text,**kw:edits.append(text)
    def translate(ctx,*args):
        ctx.progress_callback(1,2,'Step')
        return 'Bonjour'
    bot.modules={'translator':SimpleNamespace(translate=translate)}
    press(bot,'translator');bot._resolve_and_push(17,[{'type':'text','text':'Hello'}])
    execute(bot,SimpleNamespace(is_cancelled=lambda:False),bot.jobs[-1])
    assert any('50%' in text and 'Перевожу' in text for text in edits)


def test_text_voice_captures_main_settings_voice_before_queued_work(bot,tmp_path):
    from tg_translator import execute
    selected=SimpleNamespace(custom_ref_wav=str(tmp_path/'chosen.wav'))
    bot._get_ctx=lambda:selected
    calls=[];voices=[]
    bot.modules={'translator':SimpleNamespace(text_voice=lambda ctx,text,target,out,reference=None:
        calls.append((text,reference)) or {'text':'Bonjour','path':'voice.wav'})}
    bot._send_voice_from_wav=lambda cid,path,**kw:voices.append(path) or True
    press(bot,'translator');press(bot,'tr_tts')
    bot._resolve_and_push(17,[{'type':'text','text':'Hello'}])
    selected.custom_ref_wav=str(tmp_path/'later.wav')
    execute(bot,SimpleNamespace(is_cancelled=lambda:False),bot.jobs[-1])
    assert calls==[('Hello',str(tmp_path/'chosen.wav'))] and voices==['voice.wav']
