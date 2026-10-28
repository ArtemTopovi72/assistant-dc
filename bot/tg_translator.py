"""The translator owns its inputs; no agent history or generation tools are involved."""
from pathlib import Path
import time
import uuid

from seams import module
from tg_keyboards import _translator_kb
from tg_strings import _t, _LABEL2KEY
from translator import LANGUAGES, error_text, voice_reference


def action(bot, chat_id, sess, lang, key):
    if key == 'translator':
        sess.drop_waiting()
    if key.startswith('tr_lang_'):
        code = key.removeprefix('tr_lang_')
        if code not in LANGUAGES:
            return
        sess.translation_target = code
    elif key in ('tr_text', 'tr_image', 'tr_voice', 'tr_dub','tr_tts'):
        sess.translation_mode = key.removeprefix('tr_')
    bot._goto_menu(sess, 'translator_languages' if key == 'tr_languages' else 'translator')
    target = LANGUAGES.get(sess.translation_target, LANGUAGES['en'])[1]
    text = _t('tr_pick_language', lang) if key == 'tr_languages' else _t('menu_translator_title', lang, target=target)
    if key in ('tr_text', 'tr_image', 'tr_voice', 'tr_dub','tr_tts'):
        text += '\n\n' + _t(key + '_prompt', lang)
    bot._send_text(chat_id, text, keyboard=_translator_kb(lang, key == 'tr_languages'))


def route(bot, chat_id, sess, batch, input_menu):
    keys = [_LABEL2KEY.get((it.get('text') or '').strip(), '') for it in batch]
    if len(batch) == 1 and (keys[0] == 'translator' or keys[0].startswith('tr_')):
        action(bot, chat_id, sess, bot._lang(sess), keys[0])
        return True
    if input_menu not in ('translator', 'translator_languages') or any(keys) or any(
            (it.get('text') or '').startswith('/') for it in batch):
        return False
    lang = bot._lang(sess)
    settings = batch[0].get('_translation') or (sess.translation_target, sess.translation_mode)
    target, mode = settings
    if target not in LANGUAGES:
        target = 'en'
    generation = sess.context_generation
    items = [dict(it) for it in batch]
    import tg_bot
    task = tg_bot._Task(task_id=str(uuid.uuid4()), chat_id=chat_id,
        user_text='\n'.join(it.get('text', '') for it in items) or '[translation media]',
        reply_to=items[0].get('_message_id', 0), context_generation=generation,
        enqueue_ts=min(it.get('_arrival_ts', time.time()) for it in items),
        translation={'items': items, 'target': target, 'mode': mode,
                     'voice_ref':voice_reference(bot._get_ctx()) if mode=='tts' else ''})
    bot._user_store.bump_usage(chat_id, tg_bot.KIND_TASK)
    if bot._persist_task(task):
        bot._queue_note(chat_id, task, lang)
    return True


def execute(bot, ctx, task):
    chat_id, generation = task.chat_id, task.context_generation
    sess = bot._get_session(chat_id); lang = bot._lang(sess)
    items = task.translation['items']
    target, mode = task.translation['target'], task.translation['mode']

    def work(ctx):
        def active():
            return not ctx.is_cancelled() and bot._get_session(chat_id).context_generation == generation
        def send(text):
            from tg_transport import verbatim_text
            # 2000 code points also fit Telegram's limit when every character is an emoji.
            for at in range(0, len(text), 2000):
                if not active():
                    return
                with verbatim_text():
                    bot._send_text(chat_id, text[at:at+2000],
                                   keyboard=bot._state_kb(bot._get_session(chat_id), lang))
        if not active():
            return
        engine = module(bot, 'translator')
        out_dir = Path(bot._clone_dir(chat_id)).parent.parent / 'translations' / str(chat_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        text_parts = [it.get('text', '') for it in items if it['type'] == 'text']
        try:
            if text_parts:
                source_text='\n'.join(text_parts)
                spoken=engine.text_voice(ctx,source_text,target,str(out_dir),
                    reference=task.translation.get('voice_ref')) if mode=='tts' else None
                text = spoken['text'] if spoken else engine.translate(ctx, source_text, target)
                if active():
                    send(text)
                    if spoken and spoken.get('path'):
                        if not bot._send_voice_from_wav(chat_id,spoken['path']):
                            raise ValueError(_t('tr_delivery_failed',lang))
                    if spoken and spoken.get('warning'):
                        warning=error_text(spoken['warning'],lang)
                        bot._send_text(chat_id,warning)
            files = [it for it in items if it['type'] != 'text']
            if sum(len(it.get('file_ids', [])) if it['type'] == 'album' else 1 for it in files) > 4:
                raise ValueError('Send up to four files at once.')
            for it in files:
                if not active():
                    return
                kind = it['type']
                suffix = Path(it.get('filename', '')).suffix.lower()
                video = it.get('media') == 'video' or suffix in ('.mp4','.mov','.mkv','.webm','.avi','.m4v')
                media_mode = video or (mode == 'dub' and (kind in ('voice','fwd_voice')
                    or suffix in ('.wav','.mp3','.ogg','.m4a','.flac','.opus')))
                if kind == 'document' and suffix not in ('.png', '.jpg', '.jpeg', '.webp', '.wav', '.mp3', '.ogg', '.m4a', '.flac', '.opus','.mp4','.mov','.mkv','.webm','.avi','.m4v'):
                    raise ValueError(_t('tr_input_error', lang))
                audio = kind in ('voice', 'fwd_voice') or suffix in ('.wav', '.mp3', '.ogg', '.m4a', '.flac', '.opus')
                if (not media_mode and int(it.get('seconds') or 0) > 60) or int(it.get('file_size') or 0) > 20*1024*1024:
                    raise ValueError(_t('tr_input_error', lang))
                fids = it.get('file_ids', []) if kind == 'album' else [it.get('file_id', '')]
                for fid in fids:
                    data = bot._dl_bytes(fid)
                    if not active():
                        return
                    if not data or len(data) > 20*1024*1024:
                        raise ValueError(_t('tr_input_error', lang))
                    src = out_dir / (uuid.uuid4().hex[:10] + (suffix or ('.mp4' if video else '.ogg' if audio else '.png')))
                    src.write_bytes(data)
                    if media_mode:
                        result = engine.dub(ctx, str(src), target, str(out_dir))
                    elif audio:
                        result = engine.voice(ctx, str(src), target, str(out_dir), speak=mode == 'voice')
                    else:
                        result = engine.image(ctx, str(src), target, str(out_dir))
                    if not active():
                        return
                    send(result['text'])
                    if result.get('path'):
                        if media_mode:
                            if Path(result['path']).suffix == '.mp4':
                                delivered = bot._send_video(chat_id, result['path'], ctx=ctx, feedback='')
                                if not delivered:
                                    delivered = bot._send_document(chat_id, result['path'], feedback='')
                            else:
                                delivered = bot._send_document(chat_id, result['path'], feedback='')
                            if not delivered:
                                raise ValueError(_t('tr_delivery_failed', lang))
                        elif audio:
                            bot._send_voice_from_wav(chat_id, result['path'])
                        else:
                            bot._send_photo(chat_id, result['path'])
                            bot._send_document(chat_id, result['path'], feedback='')
                    if media_mode:
                        for extra in ('audio_path','subtitle_path'):
                            path = result.get(extra)
                            if path and path != result.get('path'):
                                if not active(): return
                                if not bot._send_document(chat_id, path, feedback=''):
                                    raise ValueError(_t('tr_delivery_failed', lang))
                    if result.get('warning'):
                        warning = error_text(result['warning'], lang)
                        bot._send_text(chat_id, warning)
        except Exception as exc:
            if active():
                bot._send_text(chat_id, _t('tr_failed', lang, detail=error_text(exc, lang)[:300]),
                               keyboard=bot._state_kb(bot._get_session(chat_id), lang))
    keyboard = {'inline_keyboard': [[{'text': _t('cancel_btn', lang),
                                     'callback_data': f'cancel:{task.task_id}'}]]}
    mid = bot._send_get_id(chat_id, _t('tr_working', lang), keyboard=keyboard)
    _, ctx.progress_callback = bot._with_progress(chat_id, lang,
        lambda text: bot._edit_text(chat_id, mid, text, keyboard=keyboard) if mid else None,
        shown=_t('tr_working',lang))
    try:
        work(ctx)
        task.delivered = not ctx.is_cancelled()
    finally:
        if mid:
            bot._edit_text(chat_id, mid, '✅' if task.delivered else '⛔', keyboard={'inline_keyboard': []})
