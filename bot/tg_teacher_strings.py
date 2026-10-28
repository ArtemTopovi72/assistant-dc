"""Teacher labels kept separate from translator controls."""
BTN = {
    'teacher': {'en':'🎓 Language teacher','ru':'🎓 Учитель языков'},
    'teach_start': {'en':'🎓 Start lesson','ru':'🎓 Начать урок'},
    'teach_target': {'en':'🎯 Learning language','ru':'🎯 Изучаемый язык'},
    'teach_native': {'en':'🏠 Native language','ru':'🏠 Родной язык'},
    'teach_level': {'en':'📈 Lesson level','ru':'📈 Уровень урока'},
    'teach_topic': {'en':'💬 Lesson topic','ru':'💬 Тема урока'},
    'teach_reset': {'en':'🧹 Clear lesson history','ru':'🧹 Очистить историю урока'},
    'teach_pause': {'en':'⏸ Pause lesson','ru':'⏸ Пауза урока'},
    'teach_resume': {'en':'▶ Resume lesson','ru':'▶ Продолжить урок'},
    'teach_finish': {'en':'⏹ Finish lesson','ru':'⏹ Завершить урок'},
    'teach_combined': {'en':'🎧 One voice message','ru':'🎧 Одно голосовое'},
    'teach_separate': {'en':'🎧 Two voice messages','ru':'🎧 Два голосовых'},
}
for code, label in {'ru':'Русский','en':'English','de':'Deutsch','fr':'Français','es':'Español',
                    'it':'Italiano','pt':'Português','zh':'中文','ja':'日本語','ko':'한국어'}.items():
    BTN['teach_lang_' + code] = {'en':'🎓 ' + label,'ru':'🎓 ' + label}
for level in ('auto','A1','A2','B1','B2','C1'):
    BTN['teach_lvl_' + level] = {'en':'🎓 ' + ('Auto' if level == 'auto' else level),
                                'ru':'🎓 ' + ('Авто' if level == 'auto' else level)}

MSG = {
    'teacher_title': {
        'en':'🎓 {target} · native: {native} · level: {level}\nTopic: {topic}\nVoice: {format}\nWrite in either language or send a voice message. Each reply includes speech, a translation and gentle corrections.',
        'ru':'🎓 {target} · родной: {native} · уровень: {level}\nТема: {topic}\nГолосовые: {format}\nПишите на любом из двух языков или пришлите голосовое. В ответе — озвучка, перевод и мягкое исправление ошибок.'},
    'teacher_pick_language': {'en':'Choose a lesson language.','ru':'Выберите язык урока.'},
    'teacher_pick_level': {'en':'Choose a level; Auto adapts to your practice.','ru':'Выберите уровень; Авто подстраивается под вашу практику.'},
    'teacher_pick_topic': {'en':'Write a lesson topic (up to 300 characters).','ru':'Напишите тему урока (до 300 символов).'},
    'teacher_any_topic': {'en':'Everyday conversation','ru':'Повседневное общение'},
    'teacher_working': {'en':'🎓 Preparing your reply and voice messages…','ru':'🎓 Готовлю ответ и голосовые…'},
    'teacher_input_error': {'en':'Send 1–12000 characters or one audio file up to 60 seconds and 20 MB.','ru':'Пришлите текст от 1 до 12000 символов или одну аудиозапись до 60 секунд и 20 МБ.'},
    'teacher_voice_warning': {'en':'Could not deliver the lesson audio. The text and lesson history are preserved.','ru':'Не удалось отправить озвучку урока. Текст и история урока сохранены.'},
    'teacher_failed': {'en':'Lesson failed: {detail}','ru':'Не удалось подготовить урок: {detail}'},
    'teacher_paused': {'en':'Lesson paused. Press ▶ Resume lesson to continue. Your history is saved.',
                       'ru':'Урок на паузе. Нажмите ▶ Продолжить урок. История сохранена.'},
    'teacher_ended': {'en':'Lesson finished. Your history is saved. Press ▶ Resume lesson to continue, or clear the lesson history to start again.',
                      'ru':'Урок завершён. История сохранена. Нажмите ▶ Продолжить урок или очистите историю урока, чтобы начать заново.'},
}
