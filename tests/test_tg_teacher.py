from types import SimpleNamespace as NS
from threading import Lock
import pytest
import tg_teacher as teacher


@pytest.fixture
def bot(tmp_path):
    sessions = {}
    def session(cid):
        if cid not in sessions:
            sessions[cid] = NS(teacher_target='de', teacher_native='ru', teacher_format='combined',
                teacher_profile={'level':'auto','auto_level':'A1','topic':'','streak':0},
                teacher_history=[],teacher_epoch=0,context_generation=0,menu='teacher',_lock=Lock(),drop_waiting=lambda:None,
                translation_target='en',translation_mode='text')
        return sessions[cid]
    b=NS(_get_session=session,_lang=lambda s:'en',_store=NS(put=lambda s:None), sent=[],jobs=[],voices=[],
         _get_ctx=lambda:NS(custom_ref_wav='chosen.wav'),_clone_dir=lambda cid:str(tmp_path/str(cid)/'clones'/'sample'),
         _user_store=NS(bump_usage=lambda *a:None))
    b._goto_menu=lambda s,m:setattr(s,'menu',m)
    b._state_kb=lambda s,l:teacher.keyboard(s,l)
    b._send_text=lambda cid,text,**kw:b.sent.append((cid,text)) or 1
    b._persist_task=lambda t:b.jobs.append(t) or True
    b._queue_note=lambda *a:None
    b._send_get_id=lambda cid,text,**kw:b.sent.append((cid,text)) or 1
    b._with_progress=lambda *a,**kw:(None,lambda *a:None)
    b._edit_text=lambda *a,**kw:None
    b._send_voice_from_wav=lambda cid,p:b.voices.append((cid,p)) or True
    def respond(ctx,profile,history,text,target,native):
        return dict(say='Hallo?',translation='Привет?',correction='Correction',explanation='',
                    profile=dict(profile),history=history+[{'role':'user','content':text}])
    b.modules={'language_teacher':NS(respond=respond),'translator':NS(lesson_speech=lambda *a,**kw:
        dict(paths=['target.wav','native.wav'],combined_path='combined.wav'))}
    return b


def queue(b,cid=1,text='Hello'):
    teacher.route(b,cid,b._get_session(cid),[{'type':'text','text':text}],'teacher')
    return b.jobs[-1]


def run(b,t):
    teacher.execute(b,NS(is_cancelled=lambda:False),t)


def test_teacher_entry_lives_inside_translator_not_main():
    from tg_keyboards import _main_kb, _translator_kb
    from tg_strings import _b
    for lang in ('ru','en'):
        label=_b('teacher',lang)
        assert not any(label in row for row in _main_kb(True,lang=lang)['keyboard'])
        assert any(label in row for row in _translator_kb(lang)['keyboard'])


def test_queued_turns_use_latest_history_and_are_isolated(bot):
    a=queue(bot,text='one');c=queue(bot,2,'other');b=queue(bot,text='two')
    run(bot,a);run(bot,c);run(bot,b)
    assert [m['content'] for m in bot._get_session(1).teacher_history]==['one','two']
    assert bot._get_session(2).teacher_history==[{'role':'user','content':'other'}]


@pytest.mark.parametrize('fmt,expected',[('combined',['combined.wav']),('separate',['target.wav','native.wav'])])
def test_voice_format_and_selected_reference(bot,fmt,expected):
    bot._get_session(1).teacher_format=fmt
    task=queue(bot)
    bot._get_ctx=lambda:NS(custom_ref_wav='later.wav')
    assert task.translation['voice_ref']=='chosen.wav'
    run(bot,task)
    assert [p for _,p in bot.voices]==expected
    assert any('Hallo?' in text and 'Привет?' in text and 'Correction' in text for _,text in bot.sent)


def test_settings_pickers_topic_and_back(bot):
    s=bot._get_session(1)
    teacher.action(bot,1,s,'en','teach_native');teacher.action(bot,1,s,'en','teach_lang_en')
    assert s.teacher_native=='en' and s.teacher_target=='de'
    teacher.action(bot,1,s,'en','teach_target');teacher.action(bot,1,s,'en','teach_lang_fr')
    assert s.teacher_target=='fr'
    teacher.action(bot,1,s,'en','teach_topic')
    assert teacher.route(bot,1,s,[{'type':'text','text':'Travel'}],'teacher_topic')
    assert s.teacher_profile['topic']=='Travel' and not bot.jobs
    teacher.action(bot,1,s,'en','teach_level');teacher.back(bot,1,s,'en')
    assert s.menu=='teacher'
    teacher.back(bot,1,s,'en');assert s.menu=='translator'


def test_stale_work_is_suppressed_before_and_during_model(bot):
    task=queue(bot)
    teacher.action(bot,1,bot._get_session(1),'en','teach_reset')
    run(bot,task)
    assert not bot.voices and not bot._get_session(1).teacher_history
    def respond(ctx,*args):
        teacher.action(bot,1,bot._get_session(1),'en','teach_reset')
        assert ctx.is_cancelled()
        return dict(say='STALE',translation='STALE',profile={},history=[])
    bot.modules['language_teacher'].respond=respond
    run(bot,queue(bot))
    assert not any('STALE' in text for _,text in bot.sent)


def test_voice_failure_preserves_text_history_and_warns(bot):
    bot._send_voice_from_wav=lambda *a:False
    run(bot,queue(bot))
    assert bot._get_session(1).teacher_history
    assert any('Could not deliver' in text for _,text in bot.sent)


def test_audio_is_transcribed_without_translation_and_video_rejected(bot):
    bot._dl_bytes=lambda fid:b'audio'
    bot.modules['translator'].transcribe=lambda ctx,path:'original student speech'
    s=bot._get_session(1)
    teacher.route(bot,1,s,[{'type':'voice','file_id':'a','seconds':4}],'teacher')
    run(bot,bot.jobs[-1])
    assert s.teacher_history[-1]['content']=='original student speech'
    teacher.route(bot,1,s,[{'type':'fwd_voice','file_id':'v','media':'video'}],'teacher')
    run(bot,bot.jobs[-1])
    assert len(s.teacher_history)==1
    assert any('Send 1–12000' in text for _,text in bot.sent)


def test_navigation_is_never_student_text(bot):
    from tg_strings import _b
    assert not teacher.route(bot,1,bot._get_session(1),[{'type':'text','text':_b('stop','en')}],'teacher')
    assert not bot.jobs


def test_forwarded_control_label_is_student_material(bot):
    text=teacher.BTN['teach_reset']['en']
    s=bot._get_session(1)
    teacher.route(bot,1,s,[{'type':'text','text':text,'forwarded':True}],'teacher')
    assert len(bot.jobs)==1 and s.teacher_epoch==0
    run(bot,bot.jobs[-1])
    assert s.teacher_history[-1]['content']==text


def test_stale_topic_capture_does_not_change_new_lesson(bot):
    s=bot._get_session(1)
    s.menu='teacher_topic'
    s.teacher_epoch=3
    teacher.route(bot,1,s,[{'type':'text','text':'old topic','_teacher_epoch':2}],'teacher_topic')
    assert s.teacher_profile['topic']=='' and s.teacher_epoch==3


def test_real_session_persistence_and_clear_preserve_lesson_settings(tmp_path):
    from tg_sessions import _Session, _Store
    s=_Session(51)
    s.teacher_target='fr';s.teacher_native='en';s.teacher_format='separate'
    s.teacher_profile.update(level='B1',topic='Travel',auto_level='A2',streak=2)
    s.teacher_history=[{'role':'user','content':'Bonjour'}]
    store=_Store(tmp_path/'sessions.json')
    store.put(s)
    loaded=store.get(51)
    assert loaded.teacher_history==s.teacher_history
    loaded.clear_context()
    store.put(loaded)
    restored=store.get(51)
    assert restored.teacher_history==[] and restored.teacher_epoch==1
    assert restored.teacher_profile=={'level':'B1','topic':'Travel','auto_level':'A1','streak':0}
    assert (restored.teacher_target,restored.teacher_native,restored.teacher_format)==('fr','en','separate')
