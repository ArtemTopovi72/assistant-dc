"""Durable teacher turns, with lesson state isolated from ordinary chat."""
from copy import deepcopy
from pathlib import Path
import time
import uuid

from seams import module
from tg_strings import _LABEL2KEY, _b, _t
from tg_teacher_strings import BTN, MSG
from tg_translation_format import teacher_messages
from translator import LANGUAGES, error_text, voice_reference


def _text(key, lang, **values):
    return MSG[key].get(lang, MSG[key]['en']).format(**values)


def _button(key, lang):
    return BTN[key].get(lang, BTN[key]['en'])


def keyboard(sess, lang):
    if sess.menu in ('teacher_target', 'teacher_native'):
        keys = ['teach_lang_' + code for code in LANGUAGES]
    elif sess.menu == 'teacher_level':
        keys = ['teach_lvl_' + level for level in ('auto','A1','A2','B1','B2','C1')]
    elif sess.menu == 'teacher_topic':
        keys = []
    else:
        status = sess.teacher_status
        keys = ['teach_target','teach_native','teach_level','teach_topic',
                'teach_combined','teach_separate',
                'teach_pause' if status == 'active' else 'teach_resume']
        if status != 'ended':
            keys.append('teach_finish')
        keys.append('teach_reset')
    rows = [[_button(k, lang) for k in keys[at:at+2]] for at in range(0,len(keys),2)]
    rows.append([_b('back',lang), _b('stop',lang)])
    return {'keyboard':rows,'resize_keyboard':True}


def _show(bot, cid, sess, lang):
    prompt = {'teacher_target':'teacher_pick_language','teacher_native':'teacher_pick_language',
              'teacher_level':'teacher_pick_level','teacher_topic':'teacher_pick_topic'}.get(sess.menu)
    if prompt:
        text = _text(prompt,lang)
    else:
        profile = sess.teacher_profile
        level = profile['level']
        if level == 'auto':
            level = ('Авто' if lang == 'ru' else 'Auto') + ' (' + profile['auto_level'] + ')'
        text = _text('teacher_title',lang,target=LANGUAGES[sess.teacher_target][1],
            native=LANGUAGES[sess.teacher_native][1],level=level,
            topic=profile['topic'] or _text('teacher_any_topic',lang),
            format=_button('teach_' + sess.teacher_format,lang))
        if sess.teacher_status != 'active':
            text += '\n\n' + _text('teacher_' + sess.teacher_status,lang)
    bot._send_text(cid,text,keyboard=keyboard(sess,lang))


def action(bot, chat_id, sess, lang, key):
    if key == 'teacher':
        sess.drop_waiting()
    menu = {'teach_target':'teacher_target','teach_native':'teacher_native',
            'teach_level':'teacher_level','teach_topic':'teacher_topic'}.get(key,'teacher')
    with sess._lock:
        changed = False
        if key.startswith('teach_lang_'):
            code = key.removeprefix('teach_lang_')
            if code not in LANGUAGES or sess.menu not in ('teacher_target','teacher_native'):
                return
            field = 'teacher_target' if sess.menu == 'teacher_target' else 'teacher_native'
            changed = getattr(sess,field) != code
            setattr(sess,field,code)
            if changed and field == 'teacher_target':
                sess.teacher_history = []
                sess.teacher_profile.update(auto_level='A1',streak=0)
        elif key.startswith('teach_lvl_'):
            level = key.removeprefix('teach_lvl_')
            if level not in ('auto','A1','A2','B1','B2','C1'):
                return
            changed = sess.teacher_profile['level'] != level
            if changed:
                sess.teacher_profile.update(level=level,streak=0)
        elif key in ('teach_combined','teach_separate'):
            fmt = key.removeprefix('teach_')
            changed = sess.teacher_format != fmt
            sess.teacher_format = fmt
        elif key == 'teach_reset':
            changed = True
            sess.teacher_history = []
            sess.teacher_profile.update(auto_level='A1',streak=0)
            sess.teacher_status = 'active'
        elif key in ('teach_pause','teach_resume','teach_start','teach_finish'):
            status = {'teach_pause':'paused','teach_finish':'ended'}.get(key,'active')
            changed = sess.teacher_status != status
            sess.teacher_status = status
        if changed:
            sess.teacher_epoch += 1
    if key == 'teach_finish':
        from tg_translator import action as translator_action
        translator_action(bot,chat_id,sess,lang,'translator')
        text = _text('teacher_ended',lang)
        bot._send_text(chat_id,text,keyboard=bot._state_kb(sess,lang))
        return
    bot._goto_menu(sess,menu)
    _show(bot,chat_id,sess,lang)


def back(bot, chat_id, sess, lang):
    if sess.menu == 'teacher':
        from tg_translator import action as translator_action
        translator_action(bot,chat_id,sess,lang,'translator')
        return
    bot._goto_menu(sess,'teacher')
    _show(bot,chat_id,sess,lang)


def route(bot, chat_id, sess, batch, input_menu):
    labels = {label:key for key,values in BTN.items() for label in values.values()}
    keys = ['' if it.get('forwarded') else labels.get((it.get('text') or '').strip(),
                      _LABEL2KEY.get((it.get('text') or '').strip(),'')) for it in batch]
    lang = bot._lang(sess)
    if len(batch) == 1 and (keys[0] == 'teacher' or keys[0].startswith('teach_')):
        action(bot,chat_id,sess,lang,keys[0])
        return True
    if not input_menu.startswith('teacher') or any(keys) or any(
            (it.get('text') or '').startswith('/') for it in batch):
        return False
    if input_menu != 'teacher':
        if input_menu == 'teacher_topic':
            text = '\n'.join(it.get('text','') for it in batch).strip()
            if all(it['type'] == 'text' for it in batch) and 0 < len(text) <= 300:
                with sess._lock:
                    if any(it.get('_teacher_epoch',sess.teacher_epoch) != sess.teacher_epoch for it in batch):
                        return True
                    sess.teacher_profile['topic'] = text
                    sess.teacher_history = []
                    sess.teacher_epoch += 1
                bot._goto_menu(sess,'teacher')
                _show(bot,chat_id,sess,lang)
            else:
                _show(bot,chat_id,sess,lang)
        else:
            _show(bot,chat_id,sess,lang)
        return True
    if sess.teacher_status != 'active':
        text = _text('teacher_' + sess.teacher_status,lang)
        bot._send_text(chat_id,text,keyboard=keyboard(sess,lang))
        return True
    import tg_bot
    items = deepcopy(batch)
    with sess._lock:
        if sess.teacher_status != 'active':
            return True
        data = {'kind':'teacher','items':items,'target':sess.teacher_target,'native':sess.teacher_native,
                'format':sess.teacher_format,'epoch':items[0].get('_teacher_epoch',sess.teacher_epoch)}
        generation = sess.context_generation
    data['voice_ref'] = voice_reference(bot._get_ctx())
    task = tg_bot._Task(task_id=str(uuid.uuid4()),chat_id=chat_id,
        user_text='\n'.join(it.get('text','') for it in items) or '[lesson audio]',
        reply_to=items[0].get('_message_id',0),context_generation=generation,
        enqueue_ts=min(it.get('_arrival_ts',time.time()) for it in items),translation=data)
    bot._user_store.bump_usage(chat_id,tg_bot.KIND_TASK)
    if bot._persist_task(task):
        bot._queue_note(chat_id,task,lang)
    return True


def execute(bot, ctx, task):
    cid, data = task.chat_id, task.translation
    sess = bot._get_session(cid)
    lang = bot._lang(sess)
    cancelled = ctx.is_cancelled
    def active():
        return not cancelled() and sess.teacher_status == 'active' and sess.teacher_epoch == data['epoch'] and sess.context_generation == task.context_generation
    task.delivered = False
    if not active():
        return
    ctx.is_cancelled = lambda: not active()
    text_sent=False
    def send(text='', *, messages=None):
        from tg_transport import verbatim_text
        formatted = messages is not None
        if messages is None:
            messages = [text[at:at+2000] for at in range(0,len(text),2000)]
        for message in messages:
            if not active():
                return False
            with verbatim_text():
                if not bot._send_get_id(cid,message,parse_mode='HTML' if formatted else None,
                                        keyboard=bot._state_kb(sess,lang)):
                    return False
        return True
    cancel = {'inline_keyboard':[[{'text':_t('cancel_btn',lang),'callback_data':f'cancel:{task.task_id}'}]]}
    working = _text('teacher_working',lang)
    mid = bot._send_get_id(cid,working,keyboard=cancel)
    _, ctx.progress_callback = bot._with_progress(cid,lang,
        lambda text:bot._edit_text(cid,mid,text,keyboard=cancel) if mid else None,
        shown=working)
    def stage(stage_name):
        if stage_name == 'Waiting for the graphics card':
            ctx.progress_callback(0, 0, 'In the GPU queue (1 ahead)')
    ctx.stage_callback = stage
    try:
        engine = module(bot,'translator')
        out = Path(bot._clone_dir(cid)).parent.parent / 'translations' / str(cid) / task.task_id
        out.mkdir(parents=True,exist_ok=True)
        items = data['items']
        files = [it for it in items if it['type'] != 'text']
        if len(files) > 1:
            raise ValueError(_text('teacher_input_error',lang))
        parts = [it.get('text','') for it in items if it['type'] == 'text']
        for it in files:
            suffix = Path(it.get('filename','')).suffix.lower()
            audio = it['type'] in ('voice','fwd_voice') or (it['type'] == 'document' and suffix in ('.wav','.mp3','.ogg','.m4a','.flac','.opus'))
            if not audio or it.get('media') == 'video' or int(it.get('file_size') or 0) > 20*1024*1024 or int(it.get('seconds') or 0) > 60:
                raise ValueError(_text('teacher_input_error',lang))
            raw = bot._dl_bytes(it.get('file_id',''))
            if not active():
                return
            if not raw or len(raw) > 20*1024*1024:
                raise ValueError(_text('teacher_input_error',lang))
            path = out / ('input' + (suffix or '.ogg'))
            path.write_bytes(raw)
            parts.append(engine.transcribe(ctx,str(path)))
        text = '\n'.join(parts).strip()
        if not text or len(text) > 12000:
            raise ValueError(_text('teacher_input_error',lang))
        with sess._lock:
            if not active():
                return
            profile, history = deepcopy(sess.teacher_profile), deepcopy(sess.teacher_history)
        result = module(bot,'language_teacher').respond(ctx,profile,history,text,data['target'],data['native'])
        if not active():
            return
        try:
            speech = engine.lesson_speech(ctx,result['say'],data['target'],result.get('native_speech',result['translation']),data['native'],str(out),reference=data.get('voice_ref'))
        except Exception:
            if not active():
                return
            speech = {}
        text_sent=send(messages=teacher_messages(result,data['target'],data['native'],lang))
        if not text_sent:
            return
        with sess._lock:
            if not active():
                return
            sess.teacher_profile, sess.teacher_history = result['profile'], result['history']
        bot._store.put(sess)
        try:
            if not active():
                return
            paths = [speech.get('combined_path')] if data['format'] == 'combined' else speech.get('paths',[])
            failed = not paths or any(not path for path in paths)
            for path in paths:
                if not active():
                    return
                keyboard = bot._state_kb(sess,lang)
                if 'keyboard' in keyboard:
                    keyboard = {**keyboard, 'is_persistent': True}
                if path and not bot._send_voice_from_wav(cid,path,keyboard=keyboard):
                    failed = True
            if failed or speech.get('warning'):
                send(_text('teacher_voice_warning',lang))
        except Exception:
            if active():
                send(_text('teacher_voice_warning',lang))
    except Exception as exc:
        if active():
            send(_text('teacher_failed',lang,detail=error_text(exc,lang)[:300]))
    finally:
        task.delivered = text_sent and active()
        from tg_transport import clear_working_status
        clear_working_status(bot, cid, mid, task.delivered)
        ctx.is_cancelled = cancelled
