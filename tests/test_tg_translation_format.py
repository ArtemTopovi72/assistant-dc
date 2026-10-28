import html
from html.parser import HTMLParser
import re
from types import SimpleNamespace

import pytest

from tg_translation_format import format_sections, teacher_messages, translation_messages


class SafeHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.open = False

    def handle_starttag(self, tag, attrs):
        assert tag == 'b' and not attrs and not self.open
        self.open = True

    def handle_endtag(self, tag):
        assert tag == 'b' and self.open
        self.open = False


def test_short_lesson_is_one_message_with_localized_sections_and_escaped_model_text():
    result = dict(say='Hallo <script> & "friend"?',translation='Привет <b>мир</b>',
                  correction='<a href="bad">correction</a>',explanation='Почему & как?',lesson_intent='explain')
    chunks = teacher_messages(result,'de','ru','ru')
    assert len(chunks) == 1
    text = chunks[0]
    assert '<b>🇩🇪 Deutsch</b>\nHallo &lt;script&gt; &amp; "friend"?' in text
    assert '<b>🇷🇺 Русский</b>' in text
    assert '<b>✍️ Исправление</b>' in text and '<b>💡 Правило</b>' in text
    assert '<script>' not in text and '<a href=' not in text
    parser = SafeHTML();parser.feed(text);assert not parser.open


@pytest.mark.parametrize('body',['😀'*10000,'<&>"'*7000,'word\n'*3000],ids=['emoji','entities','lines'])
def test_long_sections_fit_raw_html_and_utf16_limits_without_losing_text(body):
    chunks = format_sections([('Title <&>',body)])
    assert len(chunks) > 1
    recovered = []
    for text in chunks:
        assert len(text) <= 3500 and len(text.encode('utf-16-le'))//2 <= 3500
        parser=SafeHTML();parser.feed(text);assert not parser.open
        plain=html.unescape(re.sub(r'</?b>','',text))
        recovered.append(plain.split('\n',1)[1])
    assert ''.join(recovered) == body


def test_empty_optional_sections_are_omitted_and_practice_explanation_is_why():
    result=dict(say='Hello?',translation='Привет?',correction='',explanation='word order',lesson_intent='practice')
    text=teacher_messages(result,'en','ru','en')[0]
    assert '<b>💡 Why</b>' in text and 'Correction' not in text
    assert '\n\n\n' not in text


def test_translator_has_target_flag_and_bold_header_without_model_markup():
    assert translation_messages('Bonjour <b>ami</b> & bienvenue','fr') == [
        '<b>🌐 🇫🇷 Français</b>\nBonjour &lt;b&gt;ami&lt;/b&gt; &amp; bienvenue']


@pytest.fixture
def queued_bot(tmp_path):
    import tg_bot as T
    T.redirect_data_dir(tmp_path)
    bot=T.TelegramBot('123:TEST',lambda:None,lambda:None,lambda:{},silent_mode=True)
    bot.posts=[];bot.jobs=[]
    bot._backend=SimpleNamespace(push=bot.jobs.append,depth=lambda:1,position_of=lambda tid:1)
    bot._api_post=lambda method,p:bot.posts.append(dict(p)) or {'ok':True,'result':{'message_id':1}}
    bot._send_voice_from_wav=lambda *a:True
    bot._get_session(17).lang='ru'
    return bot


def test_teacher_delivery_uses_html_but_speech_receives_only_raw_model_content(queued_bot):
    import tg_teacher
    bot=queued_bot;spoken=[]
    def respond(ctx,profile,history,*args):
        return dict(say='Hallo <b>Welt</b>?',translation='Привет & мир?',correction='',
                    explanation='Правило <tag>',native_speech='Привет & мир? Правило <tag>',
                    lesson_intent='explain',profile=profile,history=history)
    def speech(ctx,say,target,native_speech,*args,**kwargs):
        spoken.append((say,native_speech))
        return {'combined_path':'lesson.wav'}
    bot.modules={'language_teacher':SimpleNamespace(respond=respond),
                 'translator':SimpleNamespace(lesson_speech=speech)}
    sess=bot._get_session(17)
    tg_teacher.route(bot,17,sess,[{'type':'text','text':'Hello'}],'teacher')
    tg_teacher.execute(bot,SimpleNamespace(is_cancelled=lambda:False),bot.jobs[-1])
    result=[p for p in bot.posts if p.get('parse_mode')=='HTML' and 'Hallo' in p['text']]
    assert len(result)==1 and '&lt;b&gt;Welt&lt;/b&gt;' in result[0]['text']
    assert '<b>💡 Правило</b>' in result[0]['text']
    assert spoken==[('Hallo <b>Welt</b>?','Привет & мир? Правило <tag>')]
    assert not any('🇩🇪' in value for pair in spoken for value in pair)


def test_translator_delivery_formats_result_but_leaves_warning_outside_translation_block(queued_bot):
    import tg_translator
    bot=queued_bot
    bot.modules={'translator':SimpleNamespace(text_voice=lambda *a,**kw:
        {'text':'Hello <world> & friend','warning':'Voice unavailable'})}
    sess=bot._get_session(17);sess.translation_target='en';sess.translation_mode='tts'
    tg_translator.route(bot,17,sess,[{'type':'text','text':'Привет'}],'translator')
    tg_translator.execute(bot,SimpleNamespace(is_cancelled=lambda:False),bot.jobs[-1])
    result=[p for p in bot.posts if p.get('parse_mode')=='HTML' and '🌐' in p['text']]
    assert len(result)==1 and result[0]['text']=='<b>🌐 🇬🇧 English</b>\nHello &lt;world&gt; &amp; friend'
    warning=[p for p in bot.posts if p['text']=='Voice unavailable']
    assert len(warning)==1 and 'parse_mode' not in warning[0]
