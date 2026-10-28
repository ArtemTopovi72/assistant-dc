"""Immediate randomizer actions; nothing is sent to the model queue."""
from seams import module
from tg_keyboards import _randomizer_kb
from tg_strings import _t


def action(bot, chat_id, sess, lang, key):
    if key == 'rnd_prediction' or key.startswith('rnd_pred_'):
        bot._goto_menu(sess, 'randomizer_prediction')
        if key == 'rnd_prediction':
            bot._send_text(chat_id, _t('menu_randomizer_prediction_title', lang),
                           keyboard=_randomizer_kb(lang, prediction=True))
            return
        period = key.removeprefix('rnd_pred_')

        def work(ctx):
            try:
                text = module(bot, 'randomizer').prediction(ctx, period, lang)
            except Exception:
                if ctx.is_cancelled():
                    return
                text = _t('rnd_pred_fail', lang)
            if not ctx.is_cancelled():
                bot._send_text(chat_id, text, keyboard=bot._state_kb(sess, lang))

        bot._run_cancellable(chat_id, _t('rnd_pred_working', lang), work, lang=lang)
        return
    bot._goto_menu(sess, 'randomizer_range' if key == 'rnd_number' else 'randomizer')
    engine = module(bot, 'randomizer')
    if key == 'randomizer':
        text = _t('menu_randomizer_title', lang)
    elif key == 'rnd_number':
        text = _t('rnd_range_prompt', lang)
    elif key == 'rnd_slots':
        symbols = engine.spin()
        text = '🎰  ' + ' │ '.join(symbols) + '\n' + _t('rnd_win' if engine.jackpot(symbols) else 'rnd_miss', lang)
    elif key == 'rnd_coin':
        text = '🪙 ' + _t('rnd_heads' if engine.binary() else 'rnd_tails', lang)
    elif key == 'rnd_yesno':
        text = _t('rnd_yes' if engine.binary() else 'rnd_no', lang)
    else:
        text = '🎲 ' + str(engine.number(1, 6))
    bot._send_text(chat_id, text, keyboard=_randomizer_kb(lang, sess.menu == 'randomizer_range'))


def range_text(bot, chat_id, sess, lang, text):
    engine = module(bot, 'randomizer')
    try:
        low, high = engine.parse_range(text)
        answer = _t('rnd_number_result', lang, value=engine.number(low, high), low=low, high=high)
    except ValueError as exc:
        answer = _t('rnd_order_error' if str(exc).startswith('Lower bound') else 'rnd_range_prompt', lang)
    bot._send_text(chat_id, answer, keyboard=bot._state_kb(sess, lang))
