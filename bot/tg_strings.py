"""Every user-visible string the Telegram bot can send, and the lookup for them.

Lifted out of tg_bot.py: 586 lines of message catalogue and button labels that
nothing but _t() and _b() ever read.

_DEFAULT_LANG is defined HERE and only here, and that is load-bearing rather
than tidy. A dozen functions across the tg_* modules take lang=_DEFAULT_LANG as
a DEFAULT ARGUMENT, which python binds once at def time — so a second copy of
this constant would not drift eventually, it would be wrong immediately, with
half the surface answering in one language and half in the other.

The house language is Russian. test_tg_localization globs tg_*.py, so this
module is inside the scan that keeps every string translated.
"""
import os

# ── localization ──────────────────────────────────────────────────────────────
# Buttons are matched by KEY, never by label: _LABEL2KEY maps EVERY language's
# label back to the same key, so a keyboard rendered before a language switch (or
# still on an old client) keeps working instead of falling through to the agent as
# a literal "🎨 Рисовать" prompt.
_LANGS = ("en", "ru")
# Russian is the house language: a user who never picks one, and a user whose
# Telegram client reports a locale we do not speak, both get Russian. English
# stays one tap away (/lang, ⚙️ Settings ▸ 🌐 Язык, 👤 Account ▸ Language) and an
# explicit choice is never overwritten — see the `if not sess0.lang` guard.
_DEFAULT_LANG = os.getenv("TG_DEFAULT_LANG", "ru").strip().lower()
if _DEFAULT_LANG not in _LANGS:
    _DEFAULT_LANG = "ru"

_BTN: dict[str, dict[str, str]] = {
    "draw":       {"en": "🎨 Draw",           "ru": "🎨 Рисовать"},
    "search":     {"en": "🔎 Search",         "ru": "🔎 Поиск"},
    "settings":   {"en": "⚙️ Settings",       "ru": "⚙️ Настройки"},
    "feedback":   {"en": "📝 Feedback",       "ru": "📝 Отзыв"},
    "account":    {"en": "👤 Account",        "ru": "👤 Аккаунт"},
    "help":       {"en": "❓ Help",            "ru": "❓ Помощь"},
    "stop":       {"en": "⛔ Stop",            "ru": "⛔ Стоп"},
    "admin":      {"en": "🔐 Admin Panel",    "ru": "🔐 Админ-панель"},
    "gen_image":  {"en": "🖼 Generate Image",  "ru": "🖼 Создать картинку"},
    "edit_image": {"en": "✏️ Edit Image",      "ru": "✏️ Изменить картинку"},
    "remove_obj": {"en": "🧽 Remove Object",   "ru": "🧽 Убрать объект"},
    "remove_text": {"en": "🔤 Remove Lettering", "ru": "🔤 Убрать надписи"},
    "regenerate": {"en": "🔄 Regenerate",      "ru": "🔄 Заново"},
    "analyze":    {"en": "📷 Analyze Photo",   "ru": "📷 Разбор фото"},
    "back":       {"en": "↩ Back",             "ru": "↩ Назад"},
    "web_search": {"en": "🔍 Web Search",      "ru": "🔍 Веб-поиск"},
    "deep":       {"en": "🔬 Deep Research",   "ru": "🔬 Глубокое исследование"},
    "deck":       {"en": "📊 Presentation",    "ru": "📊 Презентация"},
    "voice_off":  {"en": "🔇 Voice OFF",       "ru": "🔇 Голос ВЫКЛ"},
    "voice_on":   {"en": "🎙 Voice ON",        "ru": "🎙 Голос ВКЛ"},
    "reply_text":  {"en": "💬 Replies: text",         "ru": "💬 Ответ: текст"},
    "reply_both":  {"en": "💬🎙 Replies: text+voice", "ru": "💬🎙 Ответ: текст+голос"},
    "reply_voice": {"en": "🎙 Replies: voice",        "ru": "🎙 Ответ: голос"},
    "reply_fmt":  {"en": "🗣 Reply format",     "ru": "🗣 Формат ответа"},
    "notif":      {"en": "🔔 Notifications",   "ru": "🔔 Уведомления"},
    "think":      {"en": "🧠 Reasoning",       "ru": "🧠 Размышления"},
    "photo_file": {"en": "📎 Photo as file",   "ru": "📎 Фото файлом"},
    "my_voice":   {"en": "🗣 Assistant voice", "ru": "🗣 Голос ассистента"},
    "remember":   {"en": "📌 Remember",        "ru": "📌 Запомнить"},
    "clear":      {"en": "🗑 Clear Chat",      "ru": "🗑 Очистить чат"},
    "library":    {"en": "📚 Documents",       "ru": "📚 Документы"},
    "lib_list":   {"en": "📄 My documents",    "ru": "📄 Мои документы"},
    "lib_on":     {"en": "📚 Use docs: OFF",   "ru": "📚 Искать в документах: ВЫКЛ"},
    "lib_off":    {"en": "📚 Use docs: ON",    "ru": "📚 Искать в документах: ВКЛ"},
    "lib_clear":  {"en": "🗑 Clear documents", "ru": "🗑 Очистить документы"},
    "lang":       {"en": "🌐 Language",        "ru": "🌐 Язык"},
    "status":     {"en": "📊 Status",          "ru": "📊 Статус"},
    "facts":      {"en": "🧠 My facts",        "ru": "🧠 Мои факты"},
    "size":       {"en": "📐 Size",            "ru": "📐 Размер"},
    "depth":      {"en": "🎚 Depth",           "ru": "🎚 Глубина"},
    "wear":       {"en": "🌤 What to Wear",    "ru": "🌤 Что надеть"},
    "weather":    {"en": "🌤 Weather",         "ru": "🌤 Погода"},
    "creativity": {"en": "🎨 Creativity",      "ru": "🎨 Творчество"},
    "cr_images":  {"en": "🖼 Images",          "ru": "🖼 Картинки"},
    "cr_music":   {"en": "🎶 Music",           "ru": "🎶 Музыка"},
    "cr_video":   {"en": "🎬 Video",           "ru": "🎬 Видео"},
    "wtw_now":      {"en": "🌤 Now",       "ru": "🌤 Сейчас"},
    "wtw_24":       {"en": "🕓 24 h",      "ru": "🕓 24 ч"},
    "wtw_48":       {"en": "🕗 48 h",      "ru": "🕗 48 ч"},
    "wtw_date_btn": {"en": "📅 Date",      "ru": "📅 Дата"},
    "wtw_city_btn": {"en": "🏙 City",      "ru": "🏙 Город"},
    "songs":      {"en": "🎵 Songs",           "ru": "🎵 Песни"},
    "song_setup": {"en": "🎛 Song settings",   "ru": "🎛 Настройки песни"},
    "video_setup": {"en": "🎬 Long video recap", "ru": "🎬 Пересказ длинных видео"},
    "sandbox_btn": {"en": "📁 Sandbox",         "ru": "📁 Песочница"},
    "ozon":         {"en": "🛒 Ozon",           "ru": "🛒 Озон"},
    "ozon_find":    {"en": "🔍 Find",           "ru": "🔍 Найти товар"},
    "ozon_cheap":   {"en": "💸 Cheapest",       "ru": "💸 Подешевле"},
    "ozon_best":    {"en": "⭐ Best rated",     "ru": "⭐ С лучшим рейтингом"},
    "ozon_card":    {"en": "📦 Product card",   "ru": "📦 Карточка по ссылке"},
    "ozon_reviews": {"en": "💬 Reviews",        "ru": "💬 Что в отзывах"},
    "ozon_compare": {"en": "⚖️ Compare",        "ru": "⚖️ Сравнить"},
    "ozon_fast":    {"en": "🚚 Fast delivery",  "ru": "🚚 Доставка побыстрее"},
    "ozon_photo":   {"en": "📷 By photo",       "ru": "📷 Найти по фото"},
    "ozon_cart":    {"en": "🧺 Shopping list",  "ru": "🧺 Корзина"},
    "ozon_city":    {"en": "📍 City / pickup point", "ru": "📍 Город / пункт выдачи"},
    "ozon_basket":  {"en": "🛍 Build a set",     "ru": "🛍 Собрать набор"},
    "characters_btn": {"en": "🧑 Characters",    "ru": "🧑 Персонажи"},
    "animate_btn":    {"en": "🎞 Animate photo",  "ru": "🎞 Анимация фото"},
    "clone_btn":      {"en": "🎙 Clone voice",    "ru": "🎙 Клонировать голос"},
    "cover_btn": {"en": '🎚 Cover / Mashup', "ru": '🎚 Кавер / Мэшап'},
    "lyr_improve_btn": {"en": "✨ Improve lyrics", "ru": "✨ Улучшить текст"},
    "lyr_write_btn":   {"en": "✍️ Write lyrics",   "ru": "✍️ Сочинить текст"},
    "restyle_btn":    {"en": "🎨 Restyle video",  "ru": "🎨 Перерисовать видео"},
    "continue_btn":   {"en": "▶️ Continue video", "ru": "▶️ Продолжить видео"},
    "book_btn":       {"en": "📚 Audiobook",      "ru": "📚 Аудиокнига"},
    "style_menu_btn": {"en": "🎭 Change style",   "ru": "🎭 Сменить стиль"},
}
_LABEL2KEY: dict[str, str] = {
    label: key for key, forms in _BTN.items() for label in forms.values()
}


def _pick_form(forms: dict, lang: str) -> str:
    """Requested language → house default → any language we do have.

    The last step matters now that the default is Russian: before, a form that
    only had "en" still rendered, because English WAS the fallback. Falling back
    to the raw key would ship "acct_lang_btn" into a keyboard.
    """
    if not forms:
        return ""
    for cand in (lang, _DEFAULT_LANG, *_LANGS):
        val = forms.get(cand)
        if val:
            return val
    return ""


def _b(key: str, lang: str) -> str:
    return _pick_form(_BTN.get(key) or {}, lang) or key


def _nav_back_row(lang: str) -> list:
    """⬅ under an inline picker: back to the menu the user is in (nav:back).
    Without it a picker message was a dead end (live 10-02, «Размер»)."""
    return [{"text": _b("back", lang), "callback_data": "nav:back"}]


def _norm_lang(code: str) -> str:
    """Map a Telegram language_code (e.g. 'ru-RU') to a supported language."""
    base = (code or "").split("-")[0].lower()
    return base if base in _LANGS else _DEFAULT_LANG


# Message catalogue. Values are format strings; _t() applies **kw.
_MSG: dict[str, dict[str, str]] = {
    "welcome":       {"en": "👋 <b>Hi! I'm your AI assistant.</b>",
                      "ru": "👋 <b>Привет! Я твой ИИ-ассистент.</b>"},
    "main_menu":     {"en": "🏠 Main menu", "ru": "🏠 Главное меню"},
    # clear_context() also wipes tg_facts (deliberate — see clear_context's
    # docstring) but the reply used to say nothing about it, so a user who
    # read "Clear Chat" as "clear history" lost their saved facts with no
    # warning and no undo.
    "cleared":       {"en": "🗑 Conversation and saved facts cleared.",
                      "ru": "🗑 Диалог и сохранённые факты очищены."},
    "cancelled":     {"en": "❌ Cancelled.", "ru": "❌ Отменено."},
    "wait_cancel_btn": {"en": "✖️ Cancel", "ru": "✖️ Отмена"},
    "stopping":      {"en": "⛔ <b>Stopping…</b>{extra}",
                      "ru": "⛔ <b>Останавливаю…</b>{extra}"},
    "stop_dropped":  {"en": " Discarded {n} queued message(s).",
                      "ru": " Отброшено сообщений в очереди: {n}."},
    "stop_idle":     {"en": "⛔ Nothing was running.",
                      "ru": "⛔ Сейчас ничего не выполняется."},
    "cancel_done":   {"en": "⛔ Request cancelled.", "ru": "⛔ Запрос отменён."},
    # A whole-card job (a LoRA training run) holds the GPU. "Не смог нарисовать"
    # would send the user into a retry loop that cannot succeed for hours.
    "gpu_busy": {
        "en": "🛠 The GPU is busy with a training run right now, so nothing can "
              "be drawn until it finishes. Try again later.",
        "ru": "🛠 Видеокарта сейчас занята обучением — рисовать не получится, "
              "пока оно не закончится. Попробуй позже."},
    # The desktop app can be started with no LLM loaded (to leave the card free
    # for a training run). The bot keeps running so the owner can still control
    # it, but a user asking a question has to be told why nothing is happening
    # instead of waiting on a request that cannot be answered.
    "no_model_loaded": {
        "en": "⏸ The assistant is running without a language model right now "
              "(the GPU is busy with something else). Try again a bit later.",
        "ru": "⏸ Сейчас ассистент работает без языковой модели — видеокарта "
              "занята другой задачей. Попробуй чуть позже."},
    "cancel_gone":   {"en": "⌛ That request already finished.",
                      "ru": "⌛ Этот запрос уже завершён."},
    "cancel_pending": {"en": "⛔ Cancelling…", "ru": "⛔ Отменяю…"},
    # The graph came back with no text at all. "(no response)" was literally
    # what the user received -- a debug placeholder, in English, in a Russian
    # chat, with no hint that trying again is usually enough.
    "empty_reply": {
        "en": "🤔 The model returned nothing this time. Send it again — it "
              "usually works on the second try.",
        "ru": "🤔 Модель ничего не ответила. Повтори запрос — обычно со второго "
              "раза получается."},
    # Admin panel: a LoRA run owns the card for most of a day, and it is the
    # single most useful thing for the operator to see from a phone. Asked
    # repeatedly in chat as "как процесс?" while the answer sat in a log file.
    "adm_training": {
        "en": "🧠 <b>Training:</b> {slug} — step {step}/{total} ({pct}%)"
              "{eta}{rate}",
        "ru": "🧠 <b>Обучение:</b> {slug} — шаг {step}/{total} ({pct}%)"
              "{eta}{rate}"},
    "adm_training_idle": {
        "en": "🧠 <b>Training:</b> nothing running.",
        "ru": "🧠 <b>Обучение:</b> не идёт."},
    # "not running" hid the difference between a run that FINISHED and one that
    # was killed part-way. Which it was decides whether the next move is to
    # publish a checkpoint or to restart, so the panel says.
    "adm_training_stopped": {
        "en": "🧠 <b>Training:</b> {slug} — STOPPED at step {step}/{total}.",
        "ru": "🧠 <b>Обучение:</b> {slug} — ОСТАНОВЛЕНО на шаге {step}/{total}."},
    "adm_training_done": {
        "en": "🧠 <b>Training:</b> {slug} — finished ({total} steps).",
        "ru": "🧠 <b>Обучение:</b> {slug} — завершено ({total} шагов)."},
    "not_ready":     {"en": "⏳ Assistant still loading — try again.",
                      "ru": "⏳ Ассистент ещё загружается — попробуй ещё раз."},
    "layout_fixed": {"en": "⌨️ Looks like the wrong keyboard layout. You meant: «<i>{text}</i>»",
                     "ru": "⌨️ Похоже, ты ошибся раскладкой. Ты хотел сказать: «<i>{text}</i>»"},
    "restyle_ask_video": {"en": "🎨 Send a video (up to ~5 s is used). Its motion stays, I redraw the look.",
                          "ru": "🎨 Пришли видео (возьму первые ~5 с). Движение сохраню, а внешний вид перерисую."},
    "restyle_ask_text": {"en": "✅ Got the clip. Now write the new look: <i>anime</i>, <i>winter, snow</i>, <i>clay animation</i>…",
                         "ru": "✅ Видео есть. Теперь напиши, каким сделать: <i>аниме</i>, <i>зима, снег</i>, <i>пластилин</i>…"},
    "continue_ask_video": {"en": "▶️ Send the video to continue (the last sharp frame and the voices are taken from it).",
                           "ru": "▶️ Пришли видео, которое продолжить: возьму из него самый чёткий последний кадр и голоса."},
    "continue_ask_text": {"en": "✅ Got the clip. What should happen next? Describe the scene, the action, what they say.\n\n👤 Need a new person? Send their photo first (up to two), then the text.",
                          "ru": "✅ Видео есть. Что должно происходить дальше? Опиши сцену, действие, что говорят.\n\n👤 Нужен новый персонаж? Сначала пришли его фото (до двух), потом текст."},
    "continue_got_person": {"en": "👤 Got person {n}. Send another photo or describe what happens next -- who enters and what they do.",
                            "ru": "👤 Персонаж {n} есть. Пришли ещё фото или опиши, что дальше: кто входит в кадр и что делает."},
    "continue_people_full": {"en": "That's two new people already, the most a continuation holds. Now describe what happens.",
                             "ru": "Уже два новых персонажа, больше в продолжение не влезет. Теперь опиши, что происходит."},
    "book_ask_voice": {"en": "📚 <b>Audiobook.</b> First send a sample of the narrator's voice: a voice message, audio, video or a YouTube/TikTok/VK link with clear speech, 10+ seconds, one person, no music.",
                       "ru": "📚 <b>Аудиокнига.</b> Сначала пришли образец голоса диктора: голосовое, аудио, видео или ссылка на YouTube/TikTok/VK с чистой речью, от 10 секунд, один человек, без музыки."},
    "book_ask_book": {"en": "✅ Voice is ready. Now send the book: a .txt, .fb2 (or .fb2.zip), .epub, .pdf or .docx file (or paste a long text). Each chapter comes back as a voice message that starts with its title.",
                      "ru": "✅ Голос готов. Теперь пришли книгу: файл .txt, .fb2 (или .fb2.zip), .epub, .pdf или .docx (или вставь длинный текст). Каждая глава придёт голосовым сообщением, которое начинается с её названия."},
    "book_wrong_file": {"en": "📚 That is not a book file. Send .txt, .fb2, .epub, .pdf or .docx (the other buttons leave this mode).",
                        "ru": "📚 Это не файл книги. Пришли .txt, .fb2, .epub, .pdf или .docx (любая другая кнопка выходит из этого режима)."},
    "book_too_big": {"en": "📚 The file is {mb} MB; I can receive up to {limit} MB. Send it in parts.",
                     "ru": "📚 Файл {mb} МБ, принимаю до {limit} МБ. Пришли по частям."},
    "book_fail_read": {"en": "📚 Could not read the book. Check the file type (txt, fb2, epub, pdf with a text layer, docx) and send it again.",
                       "ru": "📚 Не смог прочитать книгу. Проверь формат (txt, fb2, epub, pdf с текстовым слоем, docx) и пришли ещё раз."},
    "book_working": {"en": "📚 Found {n} chapters, about {m} min of speech. Reading them one by one; each chapter arrives as it is ready.",
                     "ru": "📚 Нашёл глав: {n}, примерно {m} мин речи. Озвучиваю по порядку, каждая глава придёт, как будет готова."},
    "book_chapter_fail": {"en": "📚 Chapter {n} could not be voiced — skipped.", "ru": "📚 Главу {n} озвучить не удалось — пропускаю."},
    "book_unfinished": {"en": "📚 A book is half-read: {n} of {total} chapters left. Finish it?",
                        "ru": "📚 Есть недочитанная книга: осталось глав {n} из {total}. Дочитать?"},
    "book_resume_btn": {"en": "▶️ Finish it", "ru": "▶️ Дочитать"},
    "book_new_btn": {"en": "🆕 New book", "ru": "🆕 Новая книга"},
    "book_resuming": {"en": "📚 Continuing: {n} chapters left.", "ru": "📚 Продолжаю: осталось глав {n}."},
    "book_ocr": {"en": "📚 This PDF is scanned pages ({n}). Recognising the text first, about {m} min.",
                 "ru": "📚 Это PDF из картинок-сканов ({n} стр.). Сначала распознаю текст, примерно {m} мин."},
    "book_done": {"en": "📚 Done: {n} of {total} chapters.", "ru": "📚 Готово: глав {n} из {total}."},
    "book_fail_run": {"en": "📚 The audiobook stopped on an error. What was ready has been sent.",
                      "ru": "📚 Озвучка остановилась из-за ошибки. Что было готово — отправлено."},
    "continue_fail_dl": {"en": "▶️ Could not read the video — send it again.",
                         "ru": "▶️ Не смог прочитать видео — пришли ещё раз."},
    "restyle_fail_dl": {"en": "🎨 Could not get the video — send it again.",
                        "ru": "🎨 Не смог забрать видео — пришли ещё раз."},
    "restyle_fail_render": {"en": "🎨 The restyle failed. Try again later.",
                            "ru": "🎨 Не получилось перерисовать видео. Попробуй позже."},
    "restyle_done": {"en": "🎨 Restyled", "ru": "🎨 Перерисовал"},
    "anv_ask": {"en": "🎙 Will there be voice samples for the people in the clip? (up to 3)",
                "ru": "🎙 Будут образцы голосов для героев ролика? (до 3)"},
    "anv_yes": {"en": "✅ Yes", "ru": "✅ Да"},
    "anv_no": {"en": "✖️ No", "ru": "✖️ Нет"},
    "fwd_own_skipped": {"en": "↪️ A forwarded button is not a press — I'm skipping it. Tap the button itself.",
                        "ru": "↪️ Пересланная кнопка — не нажатие, пропускаю. Нажми саму кнопку."},
    "song_lyrics_choice": {"en": "🎵 This looks like ready lyrics. Sing them as they are, or polish them first (rhymes, rhythm, sense) and then sing?",
                           "ru": "🎵 Похоже на готовый текст. Спеть его как есть или сначала доработать (рифмы, ритм, смысл) и потом спеть?"},
    "song_keep_btn": {"en": "🎤 Sing as is", "ru": "🎤 Спеть как есть"},
    "song_polish_btn": {"en": "✨ Polish and sing", "ru": "✨ Доработать и спеть"},
    "song_new_btn": {"en": "✍️ Write new words", "ru": "✍️ Написать новый текст"},
    "song_polish_going": {"en": "🎵 Singing the polished lyric…", "ru": "🎵 Пою доработанный текст…"},
    "anv_offer": {"en": "🎙 Voices for the clip ({n} speaking): send your own samples — voice notes "
                        "or round videos, one per person, left to right — or go with the default voices.",
                  "ru": "🎙 Голоса для ролика (говорящих: {n}). Пришли свои — голосовые или кружки, "
                        "по одному на человека, слева направо — или поедем на стандартных."},
    "anv_own": {"en": "🎙 My voices", "ru": "🎙 Свои голоса"},
    "anv_default": {"en": "▶️ Default voices", "ru": "▶️ Стандартные"},
    "vl_recent": {"en": "recent voice", "ru": "недавний голос"},
    "vl_recent_at": {"en": "voice of {at}", "ru": "голос от {at}"},
    "vl_manage_btn": {"en": "📚 My voices", "ru": "📚 Мои голоса"},
    "vl_manage": {"en": "📚 <b>Your voices</b>: ▶️ listen · ✏️ name or rename (a named voice is kept for good) · 🗑 delete.",
                  "ru": "📚 <b>Твои голоса</b>: ▶️ послушать · ✏️ подписать или переименовать (подписанный голос хранится всегда) · 🗑 удалить."},
    "vl_empty": {"en": "📚 No voices yet: send a voice note while adding voices to a clip, and it lands here.",
                 "ru": "📚 Голосов пока нет: пришли голосовое, когда добавляешь голоса к видео, — оно появится здесь."},
    "vl_help": {"en": "📚 Your voices: tap one to make it the assistant's voice, ✏️ to name it "
                      "(a named voice is kept for good), 🗑 to delete it.",
                "ru": "📚 Твои голоса: нажми — станет голосом ассистента, ✏️ — подписать "
                      "(подписанный сохраняется навсегда), 🗑 — удалить."},
    "vl_save_btn": {"en": "💾 Name and keep this voice", "ru": "💾 Подписать и сохранить голос"},
    "vl_name_ask": {"en": "✏️ How should I call this voice? Send a name: «Stepan», «Grandma».",
                    "ru": "✏️ Как подписать этот голос? Пришли имя: «Степан», «Бабушка»."},
    "vl_saved": {"en": "💾 Saved the voice «{name}».", "ru": "💾 Сохранил голос «{name}»."},
    "vl_dropped": {"en": "🗑 «{name}» is deleted.", "ru": "🗑 «{name}» удалён."},
    "vl_gone": {"en": "That voice is no longer in the library.", "ru": "Этого голоса уже нет в библиотеке."},
    "vl_asst": {"en": "🗣 The assistant now speaks with «{name}».",
                "ru": "🗣 Теперь ассистент говорит голосом «{name}»."},
    "anv_btn": {"en": "🎙 Add voice samples", "ru": "🎙 Добавить свои голоса"},
    "anv_send": {"en": "🎙 Send voice {n} of {max}: a voice note, audio, round video or a YouTube/TikTok/VK link.",
                 "ru": "🎙 Пришли голос {n} из {max}: голосовое, аудио, кружок или ссылка на видео (YouTube/TikTok/VK)."},
    "anv_got": {"en": "🎙 Voice {n} accepted. Add another?", "ru": "🎙 Голос {n} принят. Добавим ещё?"},
    "anv_more": {"en": "➕ Another", "ru": "➕ Ещё"},
    "anv_done": {"en": "✅ Done", "ru": "✅ Готово"},
    "anv_full": {"en": "🎙 Voice {n} accepted — that's the maximum.", "ru": "🎙 Голос {n} принят — это максимум."},
    "anv_fail": {"en": "🎙 Couldn't get that file — send it again.", "ru": "🎙 Не смог забрать файл — пришли ещё раз."},
    "my_voice_off": {"en": "🗣 Back to the default assistant voice.",
                     "ru": "🗣 Вернул голос ассистента по умолчанию."},
    "my_voice_ask": {"en": "🗣 Send a voice note or clip (5-12 s, one speaker) — the assistant will answer in this voice from now on.",
                     "ru": "🗣 Пришли голосовое или запись (5-12 с, один голос) — дальше ассистент будет отвечать этим голосом."},
    "my_voice_ready": {"en": "🗣 Assistant voice changed.", "ru": "🗣 Голос ассистента изменён."},
    "my_voice_ready_off": {"en": "🔇 But voice replies are off — turn them on to hear it.",
                           "ru": "🔇 Но голосовые ответы выключены — включи, чтобы его услышать."},
    "my_voice_enable_btn": {"en": "🎙 Turn voice replies on", "ru": "🎙 Включить голосовые ответы"},
    "my_voice_reset_btn": {"en": "↩️ Default voice", "ru": "↩️ Голос по умолчанию"},
    "photo_file_state": {"en": "📎 Pictures also sent as a file, without Telegram's compression: <b>{state}</b>",
                         "ru": "📎 Картинки дублируются файлом, без сжатия Телеграма: <b>{state}</b>"},
    "think_state":   {"en": "🧠 Model reasoning: <b>{state}</b>\nOn: a little more careful on hard tasks, but several times slower. Off by default.",
                      "ru": "🧠 Размышления модели: <b>{state}</b>\nВключены — чуть аккуратнее на сложных задачах, но в разы медленнее. По умолчанию выключены."},
    "img_offer":     {"en": "Want to do something with the picture?",
                      "ru": "Сделать что-нибудь с картинкой?"},
    "voice_state":   {"en": "🎙 Voice notes: <b>{state}</b>",
                      "ru": "🎙 Голосовые ответы: <b>{state}</b>"},
    "on":            {"en": "ON ✅",  "ru": "ВКЛ ✅"},
    "off":           {"en": "OFF ❌", "ru": "ВЫКЛ ❌"},
    "queue_pos":     {"en": "⏳ <b>You're #{pos} in queue</b> — about {eta} ahead of you.",
                      "ru": "⏳ <b>Ты #{pos} в очереди</b> — впереди примерно {eta}."},
    "too_many":      {"en": "🚧 You already have {n} request(s) waiting. "
                            "Let them finish, or press ⛔ Stop to clear them.",
                      "ru": "🚧 У тебя уже {n} запрос(ов) в очереди. "
                            "Дождись их или нажми ⛔ Стоп, чтобы сбросить."},
    "quota_hit":     {"en": "🚦 Daily limit reached for <b>{kind}</b> "
                            "({used}/{limit}). It resets at midnight UTC.",
                      "ru": "🚦 Дневной лимит исчерпан: <b>{kind}</b> "
                            "({used}/{limit}). Сбросится в полночь UTC."},
    "prompt_hint":   {"en": "✏️ <b>{hint}</b> — type your query:",
                      "ru": "✏️ <b>{hint}</b> — напиши запрос:"},
    # What the user is being asked for, in their own language.
    #
    # These used to be filled with the raw _PROMPT_KB routing prefix, so a
    # Russian user tapping 🖼 Создать картинку was answered with
    # "✏️ generate an image of — напиши запрос:" — an internal English token,
    # mid-sentence, in a Russian message. The prefix is a pipeline instruction
    # that happens to be English because the agent reasons in English; it was
    # never meant to be read by anyone.
    "hint_gen_image":  {"en": "Image description", "ru": "Описание картинки"},
    "hint_edit_image": {"en": "What to change",    "ru": "Что изменить"},
    "hint_remove_obj": {"en": "What to remove",    "ru": "Что убрать"},
    "hint_web_search": {"en": "Search query",      "ru": "Поисковый запрос"},
    "hint_ozon_find":    {"en": "What to find on Ozon (you can add a budget)",
                          "ru": "Что найти на Озоне (можно с бюджетом)"},
    "hint_ozon_cheap":   {"en": "What to find cheapest on Ozon",
                          "ru": "Что найти подешевле на Озоне"},
    "hint_ozon_best":    {"en": "What to find with the best rating",
                          "ru": "Что найти с лучшим рейтингом"},
    "hint_ozon_card":    {"en": "Ozon product link or SKU",
                          "ru": "Ссылка на товар Озона или артикул"},
    "hint_ozon_reviews": {"en": "Ozon product link or SKU — I will sum up the reviews",
                          "ru": "Ссылка или артикул — перескажу отзывы"},
    "hint_ozon_fast":    {"en": "What to find with delivery in 1-2 days",
                          "ru": "Что найти с доставкой за 1–2 дня"},
    "hint_ozon_photo":   {"en": "Send a photo of the thing and what matters: cheaper, faster, best reviews",
                          "ru": "Пришли фото товара и что важно: подешевле, побыстрее, с лучшими отзывами"},
    "hint_ozon_cart":    {"en": "Add a link/SKU, 'remove …', 'clear' — or just send 'show'",
                          "ru": "Ссылка/артикул — добавлю; «убери …», «очисти» или «покажи»"},
    "hint_ozon_basket":  {"en": "An occasion or a list: 'picnic for 4 under 3000', 'plates, cups, napkins'",
                          "ru": "Повод или список: «пикник на 4 до 3000 ₽», «тарелки, стаканы, салфетки»"},
    "hint_ozon_city":    {"en": "Your city or address — I will pick the nearest Ozon pickup point",
                          "ru": "Город или адрес — выберу ближайший пункт выдачи Озона"},
    "hint_ozon_compare": {"en": "Two or more Ozon links or product names",
                          "ru": "Две и больше ссылок или названий товаров"},
    "hint_deep":       {"en": "Research topic",    "ru": "Тема исследования"},
    "hint_remember":   {"en": "Fact to remember",  "ru": "Что запомнить"},
    "hint_deck":       {"en": "Presentation topic", "ru": "Тема презентации"},
    "ask_name":      {"en": "👋 <b>Welcome!</b> This is a private AI assistant.\n\n"
                            "To get started, please tell me your <b>name</b>:",
                      "ru": "👋 <b>Добро пожаловать!</b> Это частный ИИ-ассистент.\n\n"
                            "Для начала напиши своё <b>имя</b>:"},
    "name_again":    {"en": "❓ Please enter your <b>name</b> (at least 2 characters):",
                      "ru": "❓ Напиши своё <b>имя</b> (минимум 2 символа):"},
    "not_a_menu_tap": {"en": "❓ That looks like a menu button, not text — please "
                             "<b>type it directly</b>:",
                       "ru": "❓ Это похоже на нажатие кнопки меню, а не текст — "
                             "<b>напиши это напрямую</b>:"},
    "ask_password":  {"en": "Nice to meet you, <b>{name}</b>! 👋\n\n"
                            "Now create a <b>password</b> for your account (min 4 characters).\n"
                            "<i>It is stored hashed, never in plain text.</i>",
                      "ru": "Приятно познакомиться, <b>{name}</b>! 👋\n\n"
                            "Теперь придумай <b>пароль</b> (минимум 4 символа).\n"
                            "<i>Он хранится в виде хеша, не в открытом виде.</i>"},
    "pwd_short":     {"en": "❌ Password must be at least <b>4 characters</b>. Try again:",
                      "ru": "❌ Пароль должен быть не короче <b>4 символов</b>. Ещё раз:"},
    "reg_pending":   {"en": "✅ <b>Registration complete!</b>\n\n"
                            "⏳ Your account is <b>awaiting admin approval</b>.\n"
                            "You will get a notification once reviewed.",
                      "ru": "✅ <b>Регистрация завершена!</b>\n\n"
                            "⏳ Аккаунт <b>ждёт одобрения администратора</b>.\n"
                            "Пришлю уведомление, когда рассмотрят."},
    # The session outlived the account record (a purge, a restored backup, a
    # deleted row). Pressing 👤 Аккаунт then did nothing at all -- and doing
    # nothing is exactly what a broken bot looks like.
    "acct_gone": {
        "en": "👤 I no longer have an account for this chat — send /start to "
              "register again.",
        "ru": "👤 Аккаунта для этого чата у меня больше нет — отправь /start, "
              "чтобы зарегистрироваться заново."},
    "reg_cancelled": {"en": "❌ Registration cancelled. Send /start to begin again.",
                      "ru": "❌ Регистрация отменена. Отправь /start, чтобы начать заново."},
    "resume_name":   {"en": "👋 Please tell me your <b>name</b> to continue:",
                      "ru": "👋 Напиши своё <b>имя</b>, чтобы продолжить:"},
    "resume_pwd":    {"en": "👋 Welcome back, <b>{name}</b>!\n\nRegistration is not "
                            "finished — create a <b>password</b> (min 4 characters).\n"
                            "<i>/cancel to start over.</i>",
                      "ru": "👋 С возвращением, <b>{name}</b>!\n\nРегистрация не "
                            "закончена — придумай <b>пароль</b> (минимум 4 символа).\n"
                            "<i>/cancel — начать заново.</i>"},
    "locked_out":    {"en": "🔒 This chat is logged out. Send your <b>password</b> to log in.",
                      "ru": "🔒 Чат разлогинен. Отправь <b>пароль</b>, чтобы войти."},
    "wrong_pwd":     {"en": "❌ Wrong password. Try again ({left} attempt(s) left):",
                      "ru": "❌ Неверный пароль. Ещё раз (осталось попыток: {left}):"},
    "login_locked":  {"en": "⏱ Too many failed attempts. Try again in {sec}s.",
                      "ru": "⏱ Слишком много неудачных попыток. Повтори через {sec} с."},
    "welcome_back":  {"en": "✅ Welcome back, <b>{name}</b>!",
                      "ru": "✅ С возвращением, <b>{name}</b>!"},
    "pending":       {"en": "⏳ Your account is <b>awaiting approval</b>.\n"
                            "You'll be notified when access is granted.",
                      "ru": "⏳ Аккаунт <b>ждёт одобрения</b>.\n"
                            "Сообщу, когда доступ откроют."},
    "revoked":       {"en": "⛔ Your access has been revoked. Contact the administrator.",
                      "ru": "⛔ Доступ отозван. Свяжись с администратором."},
    "group_chat":    {"en": "👥 I only work in private chats. Please write to me directly.",
                      "ru": "👥 Я работаю только в личных чатах. Напиши мне напрямую."},
    "interrupted":   {"en": "⚠️ The assistant restarted while your request was running, "
                            "so it did not finish. Tap 🔄 Retry to run it again.",
                      "ru": "⚠️ Ассистент перезапустился во время твоего запроса, "
                            "он не завершился. Нажми 🔄 Повторить, чтобы запустить снова."},
    "retry_btn":     {"en": "🔄 Retry", "ru": "🔄 Повторить"},
    "cancel_btn":    {"en": "⛔ Cancel this request", "ru": "⛔ Отменить запрос"},
    "retry_gone":    {"en": "🤷 Nothing to retry.", "ru": "🤷 Нечего повторять."},
    "await_photo":   {"en": "📷 Send me the photo and I'll describe it.",
                      "ru": "📷 Пришли фото — я его опишу."},
    "no_regenerate": {"en": "🔄 Nothing to regenerate yet — describe a picture first.",
                      "ru": "🔄 Пока нечего перерисовывать — сначала опиши картинку."},
    "no_account":    {"en": "No account is linked to this chat — send /start.",
                      "ru": "К этому чату не привязан аккаунт — отправь /start."},
    "going_offline": {"en": "🔴 <b>Going offline.</b> I'll notify you when I'm back.",
                      "ru": "🔴 <b>Ухожу офлайн.</b> Сообщу, когда вернусь."},
    "back_online":   {"en": "🟢 <b>Hello, I'm back online!</b> Ready to assist.",
                      "ru": "🟢 <b>Привет, я снова онлайн!</b> Готов помогать."},
    "lang_choose":   {"en": "🌐 Choose your language:", "ru": "🌐 Выбери язык:"},
    "lang_set":      {"en": "✅ Language set to <b>English</b>.",
                      "ru": "✅ Язык переключён на <b>русский</b>."},
    "lib_empty":     {"en": "📚 You have no documents yet. Send me a PDF, EPUB, TXT, MD "
                            "or DOCX file and I'll index it.",
                      "ru": "📚 Документов пока нет. Пришли PDF, EPUB, TXT, MD или DOCX — "
                            "я его проиндексирую."},
    "lib_header":    {"en": "📚 <b>Your documents</b> ({n}, {chunks} passages)\n",
                      "ru": "📚 <b>Твои документы</b> ({n}, фрагментов: {chunks})\n"},
    "lib_truncated": {"en": "\n… showing {shown} of {n}.",
                      "ru": "\n… показано {shown} из {n}."},
    "lib_indexing":  {"en": "📚 Indexing <b>{name}</b> — this may take a minute…",
                      "ru": "📚 Индексирую <b>{name}</b> — это займёт минуту…"},
    "lib_indexed":   {"en": "✅ <b>{name}</b> indexed ({chunks} passages). Document "
                            "search is now <b>ON</b> — ask me anything about it.",
                      "ru": "✅ <b>{name}</b> проиндексирован (фрагментов: {chunks}). "
                            "Поиск по документам <b>ВКЛЮЧЁН</b> — спрашивай."},
    "lib_failed":    {"en": "❌ Could not index <b>{name}</b>: {err}",
                      "ru": "❌ Не удалось проиндексировать <b>{name}</b>: {err}"},
    "lib_unsupported": {"en": "📎 <b>{name}</b> isn't a document I can add to your "
                             "library (supported: {exts}). If you meant to attach "
                             "it to a message, send it along with your question.",
                       "ru": "📎 <b>{name}</b> — не документ, который можно добавить "
                             "в библиотеку (поддерживаются: {exts}). Если хотел "
                             "приложить его к сообщению, отправь вместе с вопросом."},
    "lib_toggled":   {"en": "📚 Document search: <b>{state}</b>",
                      "ru": "📚 Поиск по документам: <b>{state}</b>"},
    # -- the coding sandbox. Its whole point is being understood, and it was
    # shipped as Russian string literals in tg_commands, so an English user got
    # Russian. Same rule as everything else: the surface says nothing the
    # dictionary does not.
    "sbx_denied":   {"en": "🔐 The working folder is not enabled for this account "
                           "— an administrator grants it.",
                     "ru": "🔐 Рабочая папка для этого аккаунта не включена — "
                           "её выдаёт администратор."},
    "sbx_how":      {"en": "\n\nHow to use it: send a file — an archive, a "
                           "jar, code, a photo, anything — and say in words what "
                           "to do with it. I unpack it, look inside, fix it and "
                           "send it back.\n"
                           "/files — what is in the folder (/files name to look "
                           "inside) · /sandbox — status · /reset_sandbox — wipe it",
                     "ru": "\n\nКак пользоваться: пришли файл — архив, jar, "
                           "код, фото, что угодно — и напиши словами, что с ним "
                           "сделать. Я распакую, посмотрю внутрь, поправлю и "
                           "отправлю обратно.\n"
                           "/files — что лежит в папке (/files папка — заглянуть "
                           "внутрь) · /sandbox — статус · /reset_sandbox — очистить всё"},
    "sbx_reset":    {"en": "🧹 Working folder cleared, {n} object(s) removed.",
                     "ru": "🧹 Рабочая папка очищена, удалено объектов: {n}."},
    "sbx_status":   {"en": "🧰 Sandbox\n\n• Access: {access}\n• Files: {files}, size: {kb} KB\n"
                           "• Running code: {backend}",
                     "ru": "🧰 Песочница\n\n• Доступ: {access}\n• Файлов: {files}, объём: {kb} КБ\n"
                           "• Выполнение кода: {backend}"},
    "sbx_title":    {"en": "<b>Sandbox</b>", "ru": "<b>Песочница</b>"},
    "set_queue":    {"en": "Queue backend", "ru": "Очередь"},
    "set_notif":    {"en": "Notifications", "ru": "Уведомления"},
    "set_notif_none": {"en": "none", "ru": "нет"},
    "set_notif_startup": {"en": "bot start/stop", "ru": "запуск/остановка бота"},
    "sbx_empty":    {"en": "<b>Sandbox</b> — your own working folder. It is "
                           "empty right now.",
                     "ru": "<b>Песочница</b> — твоя личная рабочая папка. "
                           "Сейчас она пуста."},
    "sbx_more":     {"en": "\n… and {n} more",
                     "ru": "\n… и ещё {n}"},
    "sbx_tap":      {"en": "\nTap a file to download it, a folder to open it.",
                     "ru": "\nНажми на файл — пришлю его, на папку — открою её."},
    "sbx_up":       {"en": "⬆ Up", "ru": "⬆ Назад"},
    "sbx_stale":    {"en": "⌛ That list is out of date — here is the current one.",
                     "ru": "⌛ Этот список устарел — вот актуальный."},
    "sbx_send_fail": {"en": "⚠️ Could not send {name}.",
                      "ru": "⚠️ Не получилось отправить {name}."},
    "lib_cleared":   {"en": "🗑 Removed {n} document(s) from your library.",
                      "ru": "🗑 Удалено документов: {n}."},
    "lib_menu":      {"en": "📚 <b>Documents</b> — send a file to add it, then ask "
                            "questions about it.",
                      "ru": "📚 <b>Документы</b> — пришли файл, чтобы добавить, потом "
                            "задавай вопросы по нему."},
    "facts_none":    {"en": "🧠 I haven't saved any facts about you yet.",
                      "ru": "🧠 Я пока ничего о тебе не запомнил."},
    "facts_header":  {"en": "🧠 <b>What I remember about you</b>\n",
                      "ru": "🧠 <b>Что я о тебе помню</b>\n"},
    "facts_forget":  {"en": "🗑 Forget all", "ru": "🗑 Забыть всё"},
    "facts_cleared": {"en": "🗑 Forgot {n} fact(s).", "ru": "🗑 Забыто фактов: {n}."},
    "facts_clear_confirm": {
        "en": "🗑 Forget all {n} saved fact(s)? This cannot be undone.",
        "ru": "🗑 Забыть все сохранённые факты ({n})? Это необратимо."},
    "facts_clear_yes": {"en": "🗑 Yes, forget all", "ru": "🗑 Да, забыть всё"},
    "facts_clear_no":  {"en": "↩️ Cancel", "ru": "↩️ Отмена"},
    "facts_clear_cancelled": {"en": "↩️ Kept your saved facts.",
                              "ru": "↩️ Сохранённые факты оставлены."},
    "doc_too_big":   {"en": "📦 <code>{name}</code> is {mb} MB — this bot can receive files up to {limit} MB only. Split the archive into smaller parts, or send the files themselves.",
                      "ru": "📦 <code>{name}</code> — {mb} МБ, а этот бот принимает файлы только до {limit} МБ. Разбей архив на части поменьше или пришли сами файлы."},
    "doc_in_sandbox": {"en": "📁 Put <code>{name}</code> in your working folder — ask me anything about it.",
                       "ru": "📁 Положил <code>{name}</code> в песочницу — спрашивай, что в нём."},
    "doc_unreadable": {"en": "❓ Could not read <code>{name}</code>.",
                      "ru": "❓ Не удалось прочитать <code>{name}</code>."},
    "no_transcribe": {"en": "🎙 Could not transcribe — try again.",
                      "ru": "🎙 Не удалось распознать — попробуй ещё раз."},
    "err_generic":   {"en": "❌ <b>Error:</b> {err}", "ru": "❌ <b>Ошибка:</b> {err}"},
    "err_timeout":   {"en": "⏱ This request took too long and was abandoned. "
                            "Please try again.",
                      "ru": "⏱ Запрос выполнялся слишком долго и был отменён. "
                            "Попробуй ещё раз."},
    "dr_failed":     {"en": "❌ <b>Deep research failed:</b> {err}",
                      "ru": "❌ <b>Глубокое исследование не удалось:</b> {err}"},
    "dr_no_topic":   {"en": "❓ Tell me what to research.",
                      "ru": "❓ Укажи тему исследования."},
    "dr_nothing":    {"en": "⚠️ Could not gather material on this topic.",
                      "ru": "⚠️ Не удалось собрать материал по этой теме."},
    "dr_header":     {"en": "🔬 <b>Deep Research</b>{note}\n📊 {sources} sources · "
                            "{pages} pages · {findings} findings · {elapsed:.0f}s\n\n",
                      "ru": "🔬 <b>Глубокое исследование</b>{note}\n📊 {sources} источников · "
                            "{pages} страниц · {findings} фактов · {elapsed:.0f}с\n\n"},
    "dr_partial":    {"en": " — stopped early, incomplete",
                      "ru": " — остановлено досрочно, неполный"},
    "dr_file_cap":   {"en": "📄 Full report ({n} characters) — attached as a file.",
                      "ru": "📄 Полный отчёт ({n} символов) — во вложении."},
    "extremism":     {"en": "⚠️ <b>Request refused.</b>\n\nThis query cannot be processed "
                            "because it may contain content that risks violating "
                            "applicable extremism or terrorism laws.\nPlease reformulate.",
                      "ru": "⚠️ <b>Запрос отклонён.</b>\n\nОн не может быть обработан: "
                            "возможно нарушение законодательства об экстремизме и "
                            "терроризме.\nПереформулируй, пожалуйста."},
    "feedback_ask":  {"en": "📝 <b>Send feedback</b>\n\nDescribe your bug or feature "
                            "request. /cancel to abort.",
                      "ru": "📝 <b>Отправить отзыв</b>\n\nОпиши баг или пожелание. "
                            "/cancel — отмена."},
    "feedback_ok":   {"en": "✅ <b>Feedback received.</b> Thank you!",
                      "ru": "✅ <b>Отзыв получен.</b> Спасибо!"},

    # ── account lifecycle ─────────────────────────────────────────────────────
    # These were the last hardcoded-English messages in a bilingual bot, and they
    # sat on the most important path there is: the first thing a new user ever
    # reads. `approve_user` even computed `lang` and passed it to _help_text(),
    # so a Russian user got an English paragraph followed by Russian help.
    "acct_approved": {"en": "✅ <b>Your account has been approved!</b>\n\n"
                            "You now have full access to the assistant.\n",
                      "ru": "✅ <b>Твой аккаунт одобрен!</b>\n\n"
                            "Теперь у тебя полный доступ к ассистенту.\n"},
    "acct_rejected": {"en": "❌ <b>Your account request was declined.</b>\n"
                            "Please contact the administrator if you think this "
                            "is a mistake.",
                      "ru": "❌ <b>Заявка на аккаунт отклонена.</b>\n"
                            "Если это ошибка, свяжись с администратором."},
    "acct_banned":   {"en": "⛔ Your account has been suspended. "
                            "Contact the administrator.",
                      "ru": "⛔ Твой аккаунт заблокирован. "
                            "Свяжись с администратором."},
    "acct_admin_ok": {"en": "👑 You have been granted <b>admin access</b>.",
                      "ru": "👑 Тебе выданы <b>права администратора</b>."},
    "acct_admin_revoked": {"en": "👤 Your <b>admin access</b> has been revoked.",
                      "ru": "👤 Твои <b>права администратора</b> отозваны."},
    "admin_welcome": {"en": "✅ <b>Welcome, Admin!</b> You have full access.\n",
                      "ru": "✅ <b>Добро пожаловать, администратор!</b> "
                            "У тебя полный доступ.\n"},
    "admins_only":   {"en": "⛔ Admins only.", "ru": "⛔ Только для админов."},
    "unknown_cmd":   {"en": "❓ I don't know the command <code>{cmd}</code>.\n",
                      "ru": "❓ Не знаю команду <code>{cmd}</code>.\n"},

    # ── name / password / profile ─────────────────────────────────────────────
    "ask_new_name":  {"en": "✏️ Current name: <b>{name}</b>\n\n"
                            "Send your new name (or /cancel to abort):",
                      "ru": "✏️ Текущее имя: <b>{name}</b>\n\n"
                            "Отправь новое имя (или /cancel для отмены):"},
    "ask_new_pass":  {"en": "🔑 Send your <b>new password</b> "
                            "(min 4 characters, or /cancel):",
                      "ru": "🔑 Отправь <b>новый пароль</b> "
                            "(минимум 4 символа, или /cancel):"},
    "name_too_short": {"en": "❓ Name must be at least 2 characters. "
                             "Try again (or /cancel):",
                       "ru": "❓ Имя должно быть не короче 2 символов. "
                             "Попробуй ещё раз (или /cancel):"},
    "name_updated":  {"en": "✅ Name updated to <b>{name}</b>.",
                      "ru": "✅ Имя изменено на <b>{name}</b>."},
    "pass_updated":  {"en": "✅ Password updated.", "ru": "✅ Пароль обновлён."},
    "profile_replace": {"en": "⚠️ <b>Replace the profile «{name}»?</b>\n\n"
                              "This deletes the account for this Telegram chat and "
                              "starts a fresh registration. It cannot be undone.",
                        "ru": "⚠️ <b>Заменить профиль «{name}»?</b>\n\n"
                              "Аккаунт для этого чата будет удалён и начнётся "
                              "новая регистрация. Отменить это нельзя."},
    "profile_yes_btn": {"en": "🗑 Yes, replace it", "ru": "🗑 Да, заменить"},
    "profile_no_btn":  {"en": "↩ Keep my profile", "ru": "↩ Оставить профиль"},
    "feedback_on":   {"en": "📬 You will now receive user feedback reports.",
                      "ru": "📬 Теперь ты будешь получать отзывы пользователей."},
    "feedback_off":  {"en": "🔕 Muted — you will no longer receive user feedback "
                            "reports. Other admins are unaffected.",
                      "ru": "🔕 Отключено — отзывы пользователей больше приходить "
                            "не будут. На других админов это не влияет."},
    "user_approved_by": {"en": "✅ User {id} approved.",
                         "ru": "✅ Пользователь {id} одобрен."},
    "user_rejected_by": {"en": "❌ User {id} rejected.",
                         "ru": "❌ Пользователь {id} отклонён."},
    "user_not_found":  {"en": "⚠️ User {id} no longer exists — nothing changed.",
                        "ru": "⚠️ Пользователь {id} больше не существует — ничего не изменено."},
    "user_not_pending": {"en": "⚠️ User {id} is currently '{status}', not pending — "
                               "approving from this button would silently change their "
                               "status. Use the desktop GUI to change it explicitly.",
                         "ru": "⚠️ Пользователь {id} сейчас в статусе «{status}», а не "
                               "«ожидает» — одобрение этой кнопкой незаметно изменило бы "
                               "статус. Используй GUI на компьютере, чтобы изменить его явно."},
    "profile_kept":  {"en": "👍 Kept. Send your password to log back in.",
                      "ru": "👍 Оставили. Отправь пароль, чтобы войти снова."},
    "profile_gone":  {"en": "🗑 Profile deleted.\n\n👋 Let's start over — "
                            "what is your name?",
                      "ru": "🗑 Профиль удалён.\n\n👋 Начнём заново — как тебя зовут?"},
    "logged_out":    {"en": "🚪 Logged out of «<b>{name}</b>»\n\n"
                            "Conversation history and session memory for this chat "
                            "were cleared. Send your password to log back in.",
                      "ru": "🚪 Выход из «<b>{name}</b>»\n\n"
                            "История переписки и память сессии для этого чата "
                            "очищены. Отправь пароль, чтобы войти снова."},

    # ── misc flow ─────────────────────────────────────────────────────────────
    "subscribed":    {"en": "🔔 <b>Subscribed!</b> You'll be told when the bot goes "
                            "offline and when it comes back.",
                      "ru": "🔔 <b>Подписка включена!</b> Сообщу, когда бот уйдёт "
                            "офлайн и когда вернётся."},
    "unsubscribed":  {"en": "🔕 <b>Unsubscribed</b> from all notifications.",
                      "ru": "🔕 <b>Подписка отключена</b> — уведомлений не будет."},
    "notif_title":   {"en": "🔔 <b>Notifications</b>\n\n",
                      "ru": "🔔 <b>Уведомления</b>\n\n"},
    "notif_state":   {"en": "• Online / offline notices: <b>{state}</b>\n\n"
                            "/subscribe · /unsubscribe",
                      "ru": "• Уведомления об онлайне/офлайне: <b>{state}</b>\n\n"
                            "/subscribe · /unsubscribe"},
    "feedback_only_admin": {"en": "🔐 Only admins receive feedback reports.",
                            "ru": "🔐 Отчёты с отзывами получают только админы."},

    # ── admin badge (customizable, not a fixed set — free emoji entry too) ────
    "badge_only_admin": {"en": "🔐 Only admins can set a badge.",
                         "ru": "🔐 Значок могут выбрать только админы."},
    "badge_menu_title": {"en": "🏷 <b>Pick a badge</b>\nShows next to your name "
                               "in the admin panel and the desktop user list. "
                               "Pick one below, or send any emoji of your own.",
                         "ru": "🏷 <b>Выбери значок</b>\nОн будет показываться "
                               "рядом с твоим именем в админ-панели и в списке "
                               "пользователей на компьютере. Выбери один ниже "
                               "или пришли свой эмодзи."},
    "badge_custom_btn": {"en": "✏️ Custom…", "ru": "✏️ Свой…"},
    "badge_clear_btn":  {"en": "🚫 Clear badge", "ru": "🚫 Убрать значок"},
    "badge_custom_prompt": {"en": "✍️ Send one emoji to use as your badge, or /cancel.",
                            "ru": "✍️ Пришли один эмодзи для значка, или /cancel."},
    "badge_invalid": {"en": "⚠️ That doesn't look like a single emoji — try again, "
                            "or /cancel.",
                      "ru": "⚠️ Это не похоже на один эмодзи — попробуй ещё раз, "
                            "или /cancel."},
    "badge_set": {"en": "🏷 Badge set to {badge}.",
                 "ru": "🏷 Значок установлен: {badge}."},
    "badge_cleared": {"en": "🚫 Badge cleared.", "ru": "🚫 Значок убран."},

    # ── weather: "what to wear" ────────────────────────────────────────────
    "wtw_prompt": {"en": "🌤 Which city? Send a name, or use your default "
                        "(<b>{city}</b>) below.",
                  "ru": "🌤 Какой город? Напиши название, или используй "
                        "город по умолчанию (<b>{city}</b>) ниже."},
    "wtw_default_btn": {"en": "📍 Use {city}", "ru": "📍 Использовать {city}"},
    "wtw_checking": {"en": "🌤 Checking…", "ru": "🌤 Проверяю…"},
    "wtw_48h_btn": {"en": "📅 Next 48 hours", "ru": "📅 На 48 часов"},
    "wtw_pickdate_btn": {"en": "🗓 Pick a date", "ru": "🗓 Выбрать дату"},
    "wtw_today":    {"en": "📅 Today", "ru": "📅 Сегодня"},
    "wtw_tomorrow": {"en": "🗓 Tomorrow", "ru": "🗓 Завтра"},
    "wtw_date_prompt": {"en": "🗓 Which date? Send it as DD.MM or DD.MM.YYYY "
                             "(e.g. 20.08), or type 'today'/'tomorrow'. "
                             "/cancel to go back.",
                       "ru": "🗓 На какую дату? Напиши в формате ДД.ММ или "
                             "ДД.ММ.ГГГГ (например, 20.08), либо «сегодня»/"
                             "«завтра». /cancel — отмена."},
    "wtw_date_invalid": {"en": "📅 Couldn't parse that date — try DD.MM, "
                              "DD.MM.YYYY, 'today' or 'tomorrow'.",
                        "ru": "📅 Не разобрал дату — попробуй ДД.ММ, "
                              "ДД.ММ.ГГГГ, «сегодня» или «завтра»."},
    "describe_remove": {"en": "🧽 What should I remove? E.g. \"the cup on the table\", "
                              "\"the man on the left\" or \"all the lettering\":",
                        "ru": "🧽 Что убрать? Например: «чашку на столе», "
                              "«человека слева» или «все надписи»:"},
    "describe_ask": {"en": "❓ What do you want to ask or discuss about this picture?",
                     "ru": "❓ Что спросить или обсудить по этой картинке?"},
    "describe_edit": {"en": "✏️ Describe what to change:",
                      "ru": "✏️ Опиши, что изменить:"},
    "describe_clothes": {"en": "👗 What should the outfit become? E.g. \"the hat "
                                "into a cap\" or \"a white summer dress\". Or send a photo "
                                "of the clothes (a caption is optional):",
                         "ru": "👗 Во что переодеть? Например: «шапку на шляпу» "
                               "или «белое летнее платье». Или пришли фото одежды "
                               "(можно с подписью):"},
    "menu_draw_title":   {"en": "🎨 <b>Draw &amp; Edit</b>",
                          "ru": "🎨 <b>Рисование и редактирование</b>"},
    "menu_search_title": {"en": "🔎 <b>Search</b>", "ru": "🔎 <b>Поиск</b>"},
    "menu_ozon_title": {"en": "🛒 <b>Ozon</b>\nJust type what you want — I will search Ozon, "
                              "compare prices and ratings and give links.",
                        "ru": "🛒 <b>Озон</b>\nПросто напиши, что ищешь, — найду на Озоне, "
                              "сравню цены и рейтинги и дам ссылки."},
    "menu_weather_title": {"en": "🌤 <b>Weather</b>", "ru": "🌤 <b>Погода</b>"},
    "menu_creativity_title": {"en": "🎨 <b>Creativity</b>", "ru": "🎨 <b>Творчество</b>"},
    "menu_cr_images_title": {"en": "🖼 <b>Images</b>", "ru": "🖼 <b>Картинки</b>"},
    "menu_cr_music_title":  {"en": "🎶 <b>Music</b>",  "ru": "🎶 <b>Музыка</b>"},
    "menu_cr_video_title":  {"en": "🎬 <b>Video</b>",  "ru": "🎬 <b>Видео</b>"},

    # ── characters (per-person LoRA adapters) ────────────────────────────────
    "menu_characters_title": {
        "en": "🧑 <b>Characters</b>\nPick who to draw.",
        "ru": "🧑 <b>Персонажи</b>\nВыбери, кого рисуем."},
    "char_none": {
        "en": "🧑‍🎨 No characters are available yet. They are trained in the "
              "desktop app (Characters tab) and appear here once an adapter "
              "exists.",
        "ru": "🧑‍🎨 Пока никого нет. Персонажи обучаются в приложении (вкладка "
              "«Персонажи») и появляются здесь, когда готов адаптер."},
    "char_prompt": {
        "en": "🧑 <b>{name}</b> — what should the picture show? Just describe "
              "the scene; the character is added automatically. /cancel to go "
              "back.",
        "ru": "🧑 <b>{name}</b> — что рисуем? Просто опиши сцену, персонаж "
              "подставится сам. /cancel — отмена."},
    "char_drawing": {"en": "🎨 Drawing {name}…", "ru": "🎨 Рисую {name}…"},
    "char_gone": {
        "en": "⌛ That character is not available any more — its adapter file is "
              "missing.",
        "ru": "⌛ Этот персонаж больше недоступен — файл адаптера пропал."},
    "char_failed": {
        "en": "⚠️ Could not draw it. Details are in the app log.",
        "ru": "⚠️ Не получилось нарисовать. Подробности в логе приложения."},
    "characters_collect_btn": {
        "en": "📸 Collect style references", "ru": "📸 Собрать референсы стиля"},
    "admin_only": {
        "en": "🔐 This is an admin-only tool.",
        "ru": "🔐 Это доступно только администратору."},
    "lora_collect_armed": {
        "en": "📸 Send photos with a caption describing each one -- they go "
              "straight into the style dataset (currently {n} images). An "
              "album shares one caption across all its photos, which is fine "
              "for a multi-angle set of the same subject. Press ⛔ Stop or "
              "/cancel when done.",
        "ru": "📸 Присылай фото с подписью к каждому — они уйдут прямо в "
              "датасет стиля (сейчас там {n} кадров). Альбом получает одну "
              "подпись на все фото — это нормально для нескольких ракурсов "
              "одной сцены. Когда закончишь — ⛔ Стоп или /cancel."},
    "lora_collect_saved": {
        "en": "✅ Saved {n} (total {total}): {caption}",
        "ru": "✅ Сохранено {n} (всего {total}): {caption}"},
    "lora_collect_no_caption": {
        "en": "no caption -- add one next time so the LoRA learns something "
              "from this frame",
        "ru": "без подписи — в следующий раз подпиши, иначе лора не поймёт, "
              "что в кадре"},
    "lora_collect_failed": {
        "en": "⚠️ Couldn't download that -- try sending it again.",
        "ru": "⚠️ Не смог скачать — пришли ещё раз."},
    "lora_collect_gone": {
        "en": "⌛ The style dataset entry is missing; collection mode is off.",
        "ru": "⌛ Запись датасета стиля пропала — режим сбора выключен."},

    # ── songs ────────────────────────────────────────────────────────────────
    "song_topic_prompt": {"en": "🎵 What should the song be about? Send a short "
                                "topic or wish (e.g. 'a birthday for my sister'). "
                                "/cancel to go back.",
                          "ru": "🎵 О чём песня? Напиши короткую тему или "
                                "пожелание (например, «день рождения для сестры»). "
                                "/cancel — отмена."},
    "song_generating": {"en": "🎵 Writing the lyrics and generating the song…",
                        "ru": "🎵 Пишу текст и генерирую песню…"},
    "song_unavailable": {"en": "🎵 Song generation isn't set up yet — sorry!",
                         "ru": "🎵 Генерация песен пока не настроена — извини!"},
    "song_error": {"en": "🎵 Something went wrong generating the song. Try again later.",
                  "ru": "🎵 Не получилось сгенерировать песню. Попробуй позже."},
    "song_lyrics_failed": {"en": "🎵 I couldn't get the lyrics written this time. "
                                 "Try again, or word the topic a bit differently.",
                           "ru": "🎵 Не получилось написать текст песни с этой попытки. "
                                 "Попробуй ещё раз или сформулируй тему немного иначе."},

    # ── song settings (genre / tempo / vocal / duration) ─────────────────────
    "ms_title": {"en": "🎛 <b>Song settings</b>",
                 "ru": "🎛 <b>Настройки песни</b>"},
    "ms_hint": {"en": "These apply to every song you generate from now on. "
                      "Auto lets the songwriter choose what suits the topic.",
                "ru": "Применяются ко всем песням, которые ты будешь "
                      "генерировать дальше. «Авто» — пусть подбирает под тему."},
    "ms_field_title": {"en": "🎛 <b>{field}</b>", "ru": "🎛 <b>{field}</b>"},
    "ms_genre":    {"en": "Genre",    "ru": "Жанр"},
    "ms_tempo":    {"en": "Tempo",    "ru": "Темп"},
    "ms_vocal":    {"en": "Vocal",    "ru": "Вокал"},
    "ms_duration": {"en": "Length",   "ru": "Длина"},
    "ms_pick_genre":    {"en": "Pick a genre.", "ru": "Выбери жанр."},
    "ms_pick_tempo":    {"en": "Pick a tempo. The BPM shown is approximate — "
                               "the songwriter keeps some room to be musical.",
                         "ru": "Выбери темп. BPM примерный — у автора остаётся "
                               "запас, чтобы получилось музыкально."},
    "ms_pick_vocal":    {"en": "Pick a voice. Naming one is the best guard "
                               "against ending up with an instrumental by accident.",
                         "ru": "Выбери голос. Явно названный голос — лучшая "
                               "защита от случайной инструменталки."},
    "ms_pick_duration": {"en": "Pick a length, type your own, or let the bot size the "
                               "song to its topic. The song is written to this length "
                               "and ends on its own outro.",
                         "ru": "Выбери длину, впиши свою или отдай на усмотрение бота — "
                               "он подберёт под тему. Песня пишется под эту длину и "
                               "заканчивается своим аутро."},
    "ms_quality": {"en": "Quality", "ru": "Качество"},
    "ms_pick_quality": {"en": "Pick a quality mode. Higher quality means bigger "
                              "models and a longer wait. Steps are the sampler's "
                              "passes: Auto uses a speed LoRA at 8 steps; any "
                              "explicit step count turns that LoRA off and runs "
                              "the plain model instead (20 is enough for most "
                              "songs, 50 is the finest). Every button shows the "
                              "measured wait for a 180-second song.",
                        "ru": "Выбери режим качества. Выше качество — больше "
                              "модели и дольше ожидание. Шаги — проходы сэмплера: "
                              "«Авто» использует ускоряющую лору за 8 шагов; любое "
                              "явное число шагов отключает эту лору и запускает "
                              "обычную модель (20 хватает почти всегда, 50 — самая "
                              "тонкая проработка). На каждой кнопке — замеренное "
                              "ожидание для песни в 180 с."},
    "m_sec_short": {"en": "s", "ru": "с"},
    "mq_fast":    {"en": "⚡ Fast",     "ru": "⚡ Быстро"},
    "mq_quality": {"en": "💎 Quality",  "ru": "💎 Качество"},
    "mq_max":     {"en": "🐘 Maximum",  "ru": "🐘 Максимум"},
    "ms_reset": {"en": "♻️ Reset to Auto", "ru": "♻️ Сбросить на «Авто»"},

    # ── long videos (Creativity ▸ 🎬 Video) ──────────────────────────────────
    "vs_title": {"en": "🎬 <b>Long video recap</b>", "ru": "🎬 <b>Пересказ длинных видео</b>"},
    "vs_hint": {"en": "A short clip or a round note is watched and listened to in one go. A LONG "
                      "video is taken apart portion by portion: transcript + key frames + a "
                      "retelling with timecodes, each portion delivered as soon as it is ready; "
                      "⛔ Stop ends it between portions. Just send the video to the chat.",
                "ru": "Короткий ролик или кружок смотрю и слушаю целиком. ДЛИННОЕ видео разбираю по "
                      "частям: расшифровка + ключевые кадры + пересказ с таймкодами, каждая часть "
                      "приходит, как только готова; ⛔ Стоп прерывает между частями. Просто кинь видео "
                      "в чат."},
    "vs_field_title": {"en": "🎬 <b>{field}</b>", "ru": "🎬 <b>{field}</b>"},
    "vs_chunk": {"en": "Portion", "ru": "Порция"},
    "vs_frames": {"en": "Frames per portion", "ru": "Кадров на порцию"},
    "vs_out": {"en": "Deliver", "ru": "Отдавать"},
    "vs_long": {"en": "Long from", "ru": "Длинное — от"},
    "vs_pick_chunk": {"en": "How many minutes of video go into one portion.",
                      "ru": "Сколько минут видео в одной порции."},
    "vs_pick_frames": {"en": "Key frames picked per portion (where the picture changed, the sharpest one).",
                       "ru": "Сколько ключевых кадров на порцию (где картинка изменилась, берётся самый резкий)."},
    "vs_pick_out": {"en": "What each portion delivers.", "ru": "Что приходит на каждую порцию."},
    "vs_pick_long": {"en": "A video at least this long is taken apart by portions; shorter ones are "
                           "watched in one go.",
                     "ru": "Видео от этой длины разбирается по частям; короче — смотрится целиком."},
    "vs_min": {"en": "{n} min", "ru": "{n} мин"},
    "vs_frames_n": {"en": "{n}", "ru": "{n}"},
    "vs_out_both": {"en": "storyboard + retelling", "ru": "раскадровка + пересказ"},
    "vs_out_board": {"en": "storyboard", "ru": "раскадровка"},
    "vs_out_retell": {"en": "retelling", "ru": "пересказ"},
    "vs_reset": {"en": "♻️ Defaults", "ru": "♻️ По умолчанию"},
    "vs_back": {"en": "⬅ Back", "ru": "⬅ Назад"},
    "vs_was_reset": {"en": "↩️ Video settings are back to defaults.", "ru": "↩️ Настройки видео сброшены."},
    "lv_offer": {"en": "🎬 <b>Long video</b> · {mins} min\nI will take it apart portion by portion "
                       "({setup}). Start?",
                 "ru": "🎬 <b>Длинное видео</b> · {mins} мин\nРазберу по частям ({setup}). Поехали?"},
    "lv_go": {"en": "▶️ Go", "ru": "▶️ Поехали"},
    "lv_settings": {"en": "⚙️ Settings", "ru": "⚙️ Настройки"},
    "lv_skip": {"en": "✖ Not now", "ru": "✖ Не надо"},
    "lv_gone": {"en": "⌛ That video is no longer pending — send it again.",
                "ru": "⌛ Это видео уже не в работе — пришли его ещё раз."},
    "lv_skipped": {"en": "👌 OK, leaving it.", "ru": "👌 Хорошо, оставляю."},
    "lv_busy": {"en": "⏳ A long video is already in progress here — ⛔ Stop it first.",
                "ru": "⏳ Здесь уже разбирается длинное видео — сначала ⛔ Стоп."},
    "lv_downloading": {"en": "⬇️ Downloading the video…", "ru": "⬇️ Скачиваю видео…"},
    "lv_no_file": {"en": "⚠️ Could not download the video (files over 20 MB need the local Bot API server).",
                   "ru": "⚠️ Не смог скачать видео (файлы больше 20 МБ требуют локальный Bot API сервер)."},
    "lv_start": {"en": "🎬 {total} → {parts} portion(s) of {mins} min. Delivering each as it is ready; ⛔ Stop ends it.",
                 "ru": "🎬 {total} → {parts} ч. по {mins} мин. Отдаю каждую, как готова; ⛔ Стоп прерывает."},
    "lv_part_head": {"en": "🎬 <b>Part {n}/{of}</b> · {a}–{b}", "ru": "🎬 <b>Часть {n}/{of}</b> · {a}–{b}"},
    "lv_board": {"en": "Storyboard", "ru": "Раскадровка"},
    "lv_retell": {"en": "Retelling", "ru": "Пересказ"},
    "lv_part_failed": {"en": "⚠️ Part {n}: could not cut it, skipping.", "ru": "⚠️ Часть {n}: не смог вырезать, пропускаю."},
    "lv_stopped": {"en": "⛔ Stopped after {done}/{of} portions.", "ru": "⛔ Остановлено после {done}/{of} частей."},
    "lv_done": {"en": "✅ Done: {parts} portion(s), {total}.", "ru": "✅ Готово: {parts} ч., {total}."},
    "lv_failed": {"en": "⚠️ The long-video job failed — see the log.", "ru": "⚠️ Разбор видео сорвался — смотри лог."},
    "ms_back":  {"en": "⬅ Back",           "ru": "⬅ Назад"},
    "ms_was_reset": {"en": "♻️ Song settings reset to Auto.",
                     "ru": "♻️ Настройки песни сброшены на «Авто»."},
    "m_auto": {"en": "✨ Auto", "ru": "✨ Авто"},
    "m_auto_bot": {"en": "🎲 Bot decides", "ru": "🎲 На усмотрение бота"},
    "ms_custom_duration": {"en": "✏️ Custom length", "ru": "✏️ Своя длина"},
    "ms_custom_tempo": {"en": "✏️ Custom BPM", "ru": "✏️ Свой BPM"},
    "ms_custom_steps": {"en": "✏️ Other step count", "ru": "✏️ Другое число шагов"},
    "ms_steps_head": {"en": "🔬 Steps: {n} (20 = fast, 50 = finest)",
                      "ru": "🔬 Шагов: {n} (20 — быстро, 50 — тщательнее)"},
    "ms_steps_short": {"en": "{n} steps", "ru": "{n} шагов"},
    "ms_turbo_on": {"en": "⚡ speed LoRA active", "ru": "⚡ ускоряющая лора включена"},
    "ms_steps_auto": {"en": "⚡ Auto (turbo, {n})", "ru": "⚡ Авто (турбо, {n})"},
    "ms_ask_steps": {"en": "✏️ How many sampler steps? A number from {lo} to {hi}. "
                           "More steps — finer detail, slower.",
                     "ru": "✏️ Сколько шагов сэмплера? Число от {lo} до {hi}. "
                           "Больше шагов — тоньше детали, дольше ждать."},
    "ms_ask_duration": {"en": "✏️ How many seconds? A number from {lo} to {hi}.",
                        "ru": "✏️ Сколько секунд? Число от {lo} до {hi}."},
    "ms_ask_tempo": {"en": "✏️ What tempo, in BPM? A number from {lo} to {hi}.",
                     "ru": "✏️ Какой темп в BPM? Число от {lo} до {hi}."},
    "ms_bad_number": {"en": "🔢 I need a number from {lo} to {hi} — or /cancel.",
                      "ru": "🔢 Нужно число от {lo} до {hi} — или /cancel."},
    "ms_custom_set": {"en": "✅ Set: {what}.", "ru": "✅ Установлено: {what}."},
    "song_bot_chose": {"en": "🎲 <b>The bot chose:</b> {items}\n<i>{why}</i>",
                       "ru": "🎲 <b>Бот выбрал сам:</b> {items}\n<i>{why}</i>"},
    "song_bot_chose_why": {"en": "It seemed the best fit for the topic.",
                           "ru": "Показалось, что так лучше всего подходит теме."},
    "m_secs": {"en": "{n}s",    "ru": "{n} с"},
    # genres
    "mg_pop":       {"en": "🎤 Pop",         "ru": "🎤 Поп"},
    "mg_rock":      {"en": "🎸 Rock",        "ru": "🎸 Рок"},
    "mg_edm":       {"en": "🔊 EDM",         "ru": "🔊 EDM"},
    "mg_synth":     {"en": "🌆 Synth-pop",   "ru": "🌆 Синти-поп"},
    "mg_hiphop":    {"en": "🎧 Hip-hop",     "ru": "🎧 Хип-хоп"},
    "mg_rnb":       {"en": "💜 R&B",         "ru": "💜 R&B"},
    "mg_soul":      {"en": "🎺 Soul",        "ru": "🎺 Соул"},
    "mg_jazz":      {"en": "🎷 Jazz",        "ru": "🎷 Джаз"},
    "mg_folk":      {"en": "🪕 Folk",        "ru": "🪕 Фолк"},
    "mg_country":   {"en": "🤠 Country",     "ru": "🤠 Кантри"},
    "mg_metal":     {"en": "🤘 Metal",       "ru": "🤘 Метал"},
    "mg_cinematic": {"en": "🎬 Cinematic",   "ru": "🎬 Кинематографично"},
    "mg_lofi":      {"en": "🌙 Lo-fi",       "ru": "🌙 Lo-fi"},
    # tempos
    "mt_slow":     {"en": "🐢 Slow",      "ru": "🐢 Медленно"},
    "mt_medium":   {"en": "🚶 Medium",    "ru": "🚶 Средне"},
    "mt_fast":     {"en": "🏃 Fast",      "ru": "🏃 Быстро"},
    "mt_veryfast": {"en": "⚡ Very fast", "ru": "⚡ Очень быстро"},
    # vocals
    "mv_female":       {"en": "👩 Female",       "ru": "👩 Женский"},
    "mv_male":         {"en": "👨 Male",         "ru": "👨 Мужской"},
    "mv_duet":         {"en": "👫 Duet",         "ru": "👫 Дуэт"},
    "mv_instrumental": {"en": "🎻 No words (may hum)", "ru": "🎻 Без слов (возможен вокализ)"},

    # ── image size picker ─────────────────────────────────────────────────────
    "size_title":   {"en": "📐 <b>Image size</b>\nNow: <b>{w}×{h}</b> ({mp} MP)",
                     "ru": "📐 <b>Размер картинки</b>\nСейчас: <b>{w}×{h}</b> ({mp} Мп)"},
    "size_auto":    {"en": "✨ Auto (from the prompt)",
                     "ru": "✨ Авто (по запросу)"},
    "size_hint":    {"en": "Auto picks portrait or landscape from what you ask for. "
                           "Higher quality takes longer to draw.",
                     "ru": "Авто выбирает вертикаль или горизонталь по твоему запросу. "
                           "Чем выше качество, тем дольше рисуется."},
    # ── 🎤 cover (tg_cover) ──
    "lyr_ask_improve": {"en": "✨ <b>Improve lyrics</b> · send the lyric. I'll check every rhyme (no lazy ones like light/tonight), the rhythm, the line lengths and the sense, and polish it in a few passes.",
                        "ru": "✨ <b>Улучшить текст</b> · пришли текст песни. Проверю каждую рифму (без банальных вроде свет/рассвет), ритм, длину строк и смысл, и доведу за несколько проходов."},
    "lyr_ask_write":   {"en": "✍️ <b>Write lyrics</b> · send the theme or a description: what the song is about, the mood, who sings it.",
                        "ru": "✍️ <b>Сочинить текст</b> · пришли тему или описание: о чём песня, настроение, от чьего лица."},
    "lyr_working_improve": {"en": "✨ Working on the lyric: checking rhymes, rhythm and sense…",
                            "ru": "✨ Работаю над текстом: проверяю рифмы, ритм и смысл…"},
    "lyr_working_write":   {"en": "✍️ Writing: a draft, then checks and polish…",
                            "ru": "✍️ Сочиняю: черновик, потом проверки и шлифовка…"},
    "lyr_head_improve": {"en": "✨ <b>Your improved lyric</b> — score {score}/10 after {rounds} pass(es). What changed:",
                         "ru": "✨ <b>Ваш улучшенный текст</b> — оценка {score}/10, проходов: {rounds}. Что поправил:"},
    "lyr_head_write":   {"en": "✍️ <b>Lyric written</b> — score {score}/10 after {rounds} pass(es) of checks. The text is in the next message.",
                         "ru": "✍️ <b>Текст готов</b> — оценка {score}/10, проходов проверки: {rounds}. Сам текст — следующим сообщением."},
    "lyr_already_good": {"en": "Nothing to fix: rhymes, rhythm and sense hold up.",
                         "ru": "Править нечего: рифмы, ритм и смысл держатся."},
    "lyr_not_better": {"en": "No rewrite came out better than this one, so it is unchanged.",
                       "ru": "Ни одна переделка не вышла лучше — текст оставил как есть."},
    "lyr_left":   {"en": "Still not perfect:", "ru": "Что ещё можно доработать:"},
    "lyr_sing_btn":  {"en": "🎵 Sing it", "ru": "🎵 Спеть"},
    "lyr_again_btn": {"en": "✨ One more pass", "ru": "✨ Ещё улучшить"},
    "lyr_gone": {"en": "That lyric is gone — send it again.", "ru": "Этого текста уже нет — пришли его снова."},
    "lyr_fail": {"en": "⚠️ Couldn't work on the lyric right now — try again.",
                 "ru": "⚠️ Не получилось обработать текст — попробуй ещё раз."},
    "cover_ask_audio": {"en": '🎚 <b>Cover / Mashup</b> · send the song: a YouTube/VK link, an audio file, a voice message or a video. Its voice and melody stay.', "ru": '🎚 <b>Кавер / Мэшап</b> · пришли песню: ссылку на YouTube/VK, аудиофайл, голосовое или видео. Её голос и мелодия останутся.'},
    "cover_ask_text": {"en": "✅ Got the song. Now send either:\n• <b>text</b> — the song will sing it in its own voice and melody;\n• a <b>second song</b> (file or link) — a quick mashup, then the first song sings the <b>second one's words</b> on its own melody, in its own singer's voice.", "ru": '✅ Песню получил. Теперь пришли одно из двух:\n• <b>текст</b> — песня споёт его своим голосом и на своей мелодии;\n• <b>вторую песню</b> (файл или ссылку) — сначала быстрый мэшап, потом первая песня споёт <b>слова второй</b> на своей мелодии голосом своего певца.'},
    "cover_working": {"en": '🎚 Working on it — a few minutes…', "ru": '🎚 Делаю — несколько минут…'},
    "cover_done": {"en": '🎚 Remix', "ru": '🎚 Готово'},
    "cover_rvc_training": {"en": "🎙 Now learning the singer's own voice — once per artist, about 15 min; then the same cover comes again in it.",
                           "ru": "🎙 Теперь учу голос самого певца — один раз на исполнителя, минут 15; потом пришлю этот же кавер его голосом."},
    "cover_two_songs_words": {"en": "📝 Taking the second song's words — the first will sing them on its melody…",
                              "ru": "📝 Беру слова второй песни — первая споёт их на своей мелодии…"},
    "cover_voice_pick": {"en": "🎤 Whose voice sings it? (kept for later covers)", "ru": "🎤 Чьим голосом петь? (запомню для следующих каверов)"},
    "cover_voice_go": {"en": "▶️ Sing this song as is in {who}'s voice", "ru": "▶️ Спеть эту песню как есть голосом: {who}"},
    "cover_voice_auto": {"en": "🎤 The original singer's", "ru": "🎤 Голосом исполнителя оригинала"},
    "cover_voice_none": {"en": "🗣 No voice swap (clearest words)", "ru": "🗣 Без замены голоса (слова чище)"},
    "cover_rvc_done": {"en": "🎚 In the singer's own voice", "ru": "🎚 Голосом певца"},
    "mashup_done": {"en": '🎛 Mashup: the first vocal over the second song', "ru": '🎛 Мэшап: вокал первой на музыке второй'},
    "cover_fail_off":      {"en": "⚠️ The cover engine is not installed on this machine.",
                            "ru": "⚠️ Движок каверов на этой машине не установлен."},
    "cover_link_fetch": {"en": "🎤 Fetching the song from the link…", "ru": "🎤 Скачиваю песню по ссылке…"},
    "cover_link_long":  {"en": "⚠️ That video is {mins} min long; a cover takes up to {limit} min. Send a shorter one or the song itself.",
                         "ru": "⚠️ Это видео на {mins} мин, а для кавера — до {limit} мин. Пришли покороче или саму песню."},
    "cover_link_fail":  {"en": "⚠️ Couldn't download that link. Send the song as a file or another link.",
                         "ru": "⚠️ Не смог скачать по ссылке. Пришли песню файлом или другую ссылку."},
    "cover_fail_too_big":  {"en": "⚠️ That file is {mb} MB; I can receive up to {limit} MB. Send it as an mp3 or a voice message, or cut it shorter.",
                            "ru": "⚠️ Файл весит {mb} МБ, а я принимаю до {limit} МБ. Пришли его в mp3 или голосовым, либо обрежь покороче."},
    "cover_fail_no_audio": {"en": "⚠️ I could not get a sound track out of that. Send an audio file or a voice message.",
                            "ru": "⚠️ Не удалось достать звук из этого файла. Пришли аудиофайл или голосовое."},
    "cover_fail_no_words": {"en": "No words to sing: send the lyrics as text.",
                            "ru": "Нечего петь: пришли текст песни сообщением."},
    "cover_fail_render":   {"en": "⚠️ The cover did not work out — try again or another song.",
                            "ru": "⚠️ Кавер не получился — попробуй ещё раз или другую песню."},
    # ── 🎙 voice clone (tg_voice_clone) ──
    "clone_ask_audio": {"en": "🎙 Send the voice to clone: a voice message, an audio file, a video or a round video, or a YouTube/TikTok/VK link — "
                              "anything with speech. <b>6–10 seconds of clear speech</b> is enough; I will cut the best part myself.",
                        "ru": "🎙 Пришли голос для клонирования: голосовое, аудиофайл, видео, кружок или ссылку на YouTube/TikTok/VK — "
                              "что угодно, где говорят. Хватит <b>6–10 секунд чистой речи</b>, лучший кусок вырежу сам."},
    "clone_working":  {"en": "⏳ Listening to the voice…", "ru": "⏳ Слушаю голос…"},
    "clone_ready":    {"en": "✅ Voice captured. Now send the <b>text</b> to speak in it — every message you send "
                             "becomes a voice note. A new clip replaces the voice; any menu button stops.",
                       "ru": "✅ Голос готов. Теперь пришли <b>текст</b>, который озвучить, — каждое твоё сообщение "
                             "станет голосовым. Новый файл заменит голос, любая кнопка меню — выход."},
    "clone_fail_no_audio":        {"en": "⚠️ I could not get a sound track out of that. Try a voice message or an audio file.",
                                   "ru": "⚠️ Не удалось достать звук из этого файла. Попробуй голосовое или аудиофайл."},
    "clone_fail_too_little_speech": {"en": "There is too little speech in it (need at least ~3 seconds). Send a longer clip.",
                                     "ru": "Слишком мало речи (нужно хотя бы ~3 секунды). Пришли запись подлиннее."},
    "clone_fail_no_words":        {"en": "I hear sound but no words — music or noise. Send a clip where someone speaks.",
                                   "ru": "Слышу звук, но не слова — музыка или шум. Пришли запись, где говорят."},
    "clone_fail_synth":           {"en": "⚠️ Could not voice that text — try again or send a shorter one.",
                                   "ru": "⚠️ Не получилось озвучить этот текст — попробуй ещё раз или покороче."},
    "size_set":     {"en": "📐 Size: <b>{w}×{h}</b> ({mp} MP)",
                     "ru": "📐 Размер: <b>{w}×{h}</b> ({mp} Мп)"},
    # ── deep-research depth picker ───────────────────────────────────────────
    "depth_title":  {"en": "🎚 <b>Research depth</b>\nNow: <b>{name}</b> · about <b>{eta}</b>",
                     "ru": "🎚 <b>Глубина исследования</b>\nСейчас: <b>{name}</b> · примерно <b>{eta}</b>"},
    "depth_set":    {"en": "🎚 Depth: <b>{name}</b> · about <b>{eta}</b>",
                     "ru": "🎚 Глубина: <b>{name}</b> · примерно <b>{eta}</b>"},
    "d_quick":      {"en": "⚡ Quick", "ru": "⚡ Быстро"},
    "d_standard":   {"en": "⚖️ Standard", "ru": "⚖️ Стандарт"},
    "d_deep":       {"en": "🔬 Deep", "ru": "🔬 Глубоко"},
    "depth_hint":   {"en": "Quick reads fewer sources and answers sooner; Deep crawls "
                           "further and writes a longer paper. You can press ⛔ Stop at "
                           "any point.",
                     "ru": "Быстро — меньше источников и ответ раньше; Глубоко — больше "
                           "обхода и длиннее работа. В любой момент можно нажать ⛔ Стоп."},
    "depth_eta_measured": {"en": "Times are the median of your last {n} runs.",
                           "ru": "Время — медиана твоих последних {n} запусков."},
    "depth_eta_guess":    {"en": "Times are rough estimates until you have run one.",
                           "ru": "Время примерное, пока не было ни одного запуска."},
    "dr_started":   {"en": "🔬 <b>Deep research started</b> · {name} · about <b>{eta}</b>\n"
                           "I will send the report when it is ready.",
                     "ru": "🔬 <b>Глубокое исследование запущено</b> · {name} · примерно "
                           "<b>{eta}</b>\nПришлю отчёт, когда будет готов."},
    "doc_send_failed": {"en": "📎 I built the file but could not send it. Ask me to "
                              "try again.",
                        "ru": "📎 Файл собран, но отправить не получилось. Попроси "
                              "меня попробовать ещё раз."},
    "doc_missing":    {"en": "⚠️ <b>No file was actually produced.</b> Whatever I "
                             "said above, there is no presentation to open — ask "
                             "me to build it again.",
                       "ru": "⚠️ <b>Файл на самом деле не получился.</b> Что бы я "
                             "ни написал выше, открывать нечего — попроси собрать "
                             "презентацию заново."},

    # ── video ────────────────────────────────────────────────────────────────
    "video_send_failed": {"en": "⚠️ <b>The clip was rendered but could not be sent.</b> "
                                "It is probably too large for Telegram — ask me for a "
                                "shorter one.",
                          "ru": "⚠️ <b>Видео отрисовалось, но отправить его не вышло.</b> "
                                "Скорее всего, оно слишком большое для Telegram — "
                                "попроси покороче."},

    # ── a forwarded voice note ───────────────────────────────────────────────
    "fwd_voice_ask": {"en": "🎧 <b>Forwarded voice note</b> · {mins}\nWhat should I "
                            "do with it?",
                      "ru": "🎧 <b>Пересланное голосовое</b> · {mins}\nЧто с ним сделать?"},
    # The storyboard is behind its own button (👁), not dumped into the ask.
    "fwd_video_ask": {"en": "🎥 <b>Forwarded video</b> · {mins}\nWhat should I do with it?",
                      "ru": "🎥 <b>Пересланное видео</b> · {mins}\nЧто с ним сделать?"},
    "fwd_batch_ask": {"en": "📨 <b>Forwarded conversation</b> · {parts}{who}\nWhat should I do with it?",
                      "ru": "📨 <b>Пересланная переписка</b> · {parts}{who}\nЧто с ней сделать?"},
    "fwd_batch_who": {"en": "\n👥 {names}", "ru": "\n👥 {names}"},
    "fwd_lbl_voice": {"en": "🎤 voice {n}", "ru": "🎤 ГС {n}"},
    "fwd_lbl_video": {"en": "🎥 video {n}", "ru": "🎥 видео {n}"},
    "fwd_lbl_text":  {"en": "💬 {n}", "ru": "💬 {n}"},
    "fwd_continue": {"en": "▶️ Continue the video", "ru": "▶️ Продолжить видео"},
    "fwd_pick":     {"en": "🎞 Pick a picture or video", "ru": "🎞 Выбрать фото или видео"},
    "fwd_pick_ask": {"en": "Which one?", "ru": "Какое из них?"},
    "fwd_picked":   {"en": "🖼 {label} is chosen. Say what to do with it.", "ru": "🖼 {label} выбрано. Напиши, что с ним сделать."},
    "fwd_lbl_photo": {"en": "🖼 picture {n}", "ru": "🖼 фото {n}"},
    "fwd_no_speech": {"en": "(no speech)", "ru": "(без слов)"},
    "fwd_voice_board": {"en": "👁 Storyboard", "ru": "👁 Раскадровка"},
    "fwd_voice_own":   {"en": "✍️ My own request", "ru": "✍️ Свой вариант"},
    "fwd_board_head":  {"en": "👁 <b>Storyboard</b>", "ru": "👁 <b>Раскадровка</b>"},
    "fwd_board_none":  {"en": "👁 There is no picture in this one — only sound.",
                        "ru": "👁 Здесь нет картинки — только звук."},
    "fwd_own_ask": {"en": "✍️ Write what to do with it — e.g. «what deadline did he give?» or «draft a reply».",
                    "ru": "✍️ Напиши, что сделать с пересланным — например «какой срок он назвал?» или «составь ответ»."},
    "fwd_video_seen": {"en": "👁 Storyboard of the video:\n{seen}", "ru": "👁 Раскадровка видео:\n{seen}"},
    "fwd_video_work": {"en": "🎥 Watching and listening…", "ru": "🎥 Смотрю и слушаю…"},
    "fwd_voice_text": {"en": "📝 Transcript", "ru": "📝 Расшифровка"},
    "fwd_voice_sum":  {"en": "📋 Summary", "ru": "📋 Краткое содержание"},
    "retell_title_voice": {"en": "Summary of your voice message", "ru": "Краткое содержание твоего голосового сообщения"},
    "retell_title_video": {"en": "Summary of your video", "ru": "Краткое содержание твоего видео"},
    "fwd_voice_both": {"en": "📝+📋 Both", "ru": "📝+📋 И то, и другое"},
    "fwd_voice_none": {"en": "🎧 I could not make out any speech in that voice note.",
                       "ru": "🎧 Не удалось разобрать речь в этом голосовом."},
    "fwd_voice_gone": {"en": "🎧 That voice note is no longer in my hands — forward it again.",
                       "ru": "🎧 Этого голосового у меня больше нет — перешли его заново."},
    "fwd_voice_head": {"en": "📝 <b>Transcript</b>", "ru": "📝 <b>Расшифровка</b>"},
    "fwd_voice_work": {"en": "🎧 Listening to it…", "ru": "🎧 Слушаю…"},

    # ── pointing at a specific image ─────────────────────────────────────────
    "img_gone":     {"en": "🖼 That picture is no longer available — send it again, "
                           "or pick another one.",
                     "ru": "🖼 Этой картинки больше нет — пришли её заново или выбери другую."},
    # Same honest-refusal shape as img_gone, generalised to the other artifact
    # kinds the durable store (tg_artifacts_store.py) now tracks. Added
    # 2026-09-19 alongside that store, for callers that resolve "that video" /
    # "that song" / "that file" and find the row's path no longer on disk.
    "vid_gone":     {"en": "🎬 That video is no longer available on the server — "
                           "ask me to make it again.",
                     "ru": "🎬 Этого видео больше нет на сервере — попроси сделать заново."},
    "music_gone":   {"en": "🎵 That track is no longer available on the server — "
                           "ask me to make it again.",
                     "ru": "🎵 Этого трека больше нет на сервере — попроси сделать заново."},
    "doc_gone":     {"en": "📄 That file is no longer available on the server — "
                           "ask me to make it again.",
                     "ru": "📄 Этого файла больше нет на сервере — попроси сделать заново."},
    "img_which":    {"en": "🖼 <b>Which picture?</b>\nI have {n} in this chat. Pick one, "
                           "or reply directly to the picture you mean.",
                     "ru": "🖼 <b>Какую картинку?</b>\nВ этом чате их {n}. Выбери нужную "
                           "или ответь (reply) прямо на неё."},
    "img_latest":   {"en": "🆕 The most recent one", "ru": "🆕 Самая последняя"},
    "img_picked":   {"en": "🖼 Working on picture {i}. Say what to do with it.",
                     "ru": "🖼 Работаю с картинкой {i}. Скажи, что с ней сделать."},
    "img_unlabelled": {"en": "picture", "ru": "картинка"},
    "to_animate_label": {"en": "to animate", "ru": "для анимации"},
    "to_restyle_label": {"en": "to restyle", "ru": "для смены стиля"},
    "q_draft":      {"en": "Draft", "ru": "Черновик"},
    "q_standard":   {"en": "Standard", "ru": "Стандарт"},
    "q_high":       {"en": "High", "ru": "Высокое"},
    "q_ultra":      {"en": "Ultra", "ru": "Максимум"},
    "other_profile_btn": {"en": "🆕 Use a different profile",
                          "ru": "🆕 Войти в другой профиль"},

    # ── admin tools ───────────────────────────────────────────────────────────
    "bcast_usage":   {"en": "📢 Usage: <code>/broadcast your message</code>",
                      "ru": "📢 Использование: <code>/broadcast текст</code>"},
    "bcast_confirm": {"en": "📢 Send this to <b>{n}</b> approved user(s)?\n\n{body}",
                      "ru": "📢 Отправить это <b>{n}</b> одобренным пользователям?"
                            "\n\n{body}"},
    "bcast_go_btn":  {"en": "📤 Send", "ru": "📤 Отправить"},
    "bcast_no_btn":  {"en": "↩ Cancel", "ru": "↩ Отмена"},
    "bcast_cancelled": {"en": "↩ Broadcast cancelled.", "ru": "↩ Рассылка отменена."},
    "bcast_announce":  {"en": "📢 <b>Announcement</b>\n\n{body}",
                        "ru": "📢 <b>Объявление</b>\n\n{body}"},
    "bcast_sent":    {"en": "📢 Sent to {sent} user(s), {failed} failed.",
                      "ru": "📢 Отправлено {sent} пользователям, ошибок: {failed}."},
    "stats_none":    {"en": "📊 No usage recorded in the last 7 days.",
                      "ru": "📊 За последние 7 дней активности не было."},
    "feedback_from": {"en": "📬 <b>Feedback from {name}</b>\n\n{body}",
                      "ru": "📬 <b>Отзыв от {name}</b>\n\n{body}"},
    "new_reg":       {"en": "🔔 <b>New registration:</b> {name}{handle}\n"
                            "<code>chat_id: {id}</code>",
                      "ru": "🔔 <b>Новая регистрация:</b> {name}{handle}\n"
                            "<code>chat_id: {id}</code>"},
    "approve_btn":   {"en": "✅ Approve {name}", "ru": "✅ Одобрить {name}"},
    "reject_btn":    {"en": "❌ Reject", "ru": "❌ Отклонить"},

    # ── image action buttons (under every generated picture) ──────────────────
    # Kept short on purpose: Telegram lays out 3 inline buttons per row at a fixed
    # width, so anything past ~11-12 Cyrillic characters (incl. the emoji) gets
    # cut off with "…" -- live, 2026-09-19, the user saw a grid of unreadable
    # "Улучшить ка…" / "Сменить оде…" / "Перерисоват…" buttons. The fuller
    # phrasing this used to carry (which action, what it does) belongs in the
    # bot's own reply text, not the button.
    "img_change_clothes": {"en": "👗 Outfit",   "ru": "👗 Одежда"},
    "img_remove_object": {"en": "🧽 Remove",   "ru": "🧽 Убрать"},
    "img_remove_text": {"en": "🔤 No lettering", "ru": "🔤 Без надписей"},
    "img_regen":     {"en": "🔄 Redraw",        "ru": "🔄 Ещё раз"},
    "img_edit":      {"en": "🎨 Edit",          "ru": "🎨 Правка"},
    "img_ask":       {"en": "❓ Ask",           "ru": "❓ Спросить"},
    "img_style":     {"en": "🎭 Style",         "ru": "🎭 Стиль"},
    "img_animate":   {"en": "🎬 Animate",       "ru": "🎬 Анимация"},
    "style_ask_ref": {"en": "🎨 Pick a style below, describe your own, or just send "
                            "me a reference photo — the picture whose style "
                            "(lighting, colours, mood) I should copy onto this one.",
                      "ru": "🎨 Выбери стиль из списка, опиши свой, или пришли "
                            "фото-референс — снимок, чей стиль (свет, цвета, "
                            "настроение) перенести на эту картинку."},
    "style_custom_btn": {"en": "✏️ Describe my own style",
                          "ru": "✏️ Свой вариант стиля"},
    "style_ask_custom": {"en": "✍️ Describe the style you want (e.g. \"1980s "
                                "polaroid photo\", \"stained glass\").",
                          "ru": "✍️ Опиши, какой стиль нужен (например «полароид "
                                "80-х», «витраж»)."},

    # ── change style (Creativity ▸ 🎭 Change style) ────────────────────────────
    "style_menu_ask_photo": {"en": "📷 Send the photo whose style you want to "
                                    "change.",
                              "ru": "📷 Пришли фото, стиль которого нужно "
                                    "поменять."},

    # ── animate photo (Creativity ▸ 🎞 Animate photo) ──────────────────────────
    "animate_ask_photo": {"en": "📷 Send the photo to animate.",
                           "ru": "📷 Пришли фото, которое нужно анимировать."},
    "animate_ask_preset": {"en": "🎬 Pick what should happen in the clip, or "
                                  "describe your own.",
                            "ru": "🎬 Выбери, что должно происходить в ролике, "
                                  "или опиши своё."},
    "animate_custom_btn": {"en": "✏️ Describe my own motion",
                            "ru": "✏️ Свой вариант"},
    "animate_ask_custom": {"en": "✍️ Describe what should happen in the clip "
                                  "(e.g. \"turns head and smiles\").",
                            "ru": "✍️ Опиши, что должно происходить в ролике "
                                  "(например «поворачивает голову и улыбается»)."},

    # ── status page ───────────────────────────────────────────────────────────
    "status_title":  {"en": "📊 <b>Status</b>\n", "ru": "📊 <b>Состояние</b>\n"},
    "status_llm":    {"en": "• LLM (LM Studio): {v}",
                      "ru": "• Модель (LM Studio): {v}"},
    "status_comfy":  {"en": "• Images (ComfyUI): {v}",
                      "ru": "• Картинки (ComfyUI): {v}"},
    "status_queue":  {"en": "• Queue: <b>{waiting}</b> waiting · {running} running",
                      "ru": "• Очередь: <b>{waiting}</b> в ожидании · "
                            "{running} в работе"},
    "status_uptime": {"en": "• Uptime: <b>{h} h</b>",
                      "ru": "• Аптайм: <b>{h} ч</b>"},
    "status_llm_busy": {"en": "busy with a render", "ru": "занята рендером"},
    "status_backend": {"en": "• Backend: <code>{name}</code>",
                       "ru": "• Бэкенд: <code>{name}</code>"},
    "status_usage":  {"en": "📈 <b>Your usage today</b>",
                      "ru": "📈 <b>Твоё использование сегодня</b>"},
    "status_docs":   {"en": "• 📚 documents: <b>{n}</b> ({state})",
                      "ru": "• 📚 документов: <b>{n}</b> ({state})"},
    "kind_task":     {"en": "tasks",         "ru": "задачи"},
    "kind_image":    {"en": "images",        "ru": "картинки"},
    "kind_research": {"en": "deep research", "ru": "исследования"},

    # ── account page ──────────────────────────────────────────────────────────
    "acct_title":    {"en": "👤 <b>Your Account</b>\n", "ru": "👤 <b>Твой аккаунт</b>\n"},
    "acct_name":     {"en": "• Name: <b>{name}</b>", "ru": "• Имя: <b>{name}</b>"},
    "acct_handle":   {"en": "• Telegram: <b>{handle}</b>",
                      "ru": "• Телеграм: <b>{handle}</b>"},
    "acct_notif":    {"en": "• Online / offline notices: <b>{state}</b>",
                      "ru": "• Уведомления об онлайне/офлайне: <b>{state}</b>"},
    "acct_feedback": {"en": "• Feedback reports: <b>{state}</b>",
                      "ru": "• Отчёты с отзывами: <b>{state}</b>"},
    "acct_badge":    {"en": "• Badge: <b>{badge}</b>", "ru": "• Значок: <b>{badge}</b>"},
    "badge_none":    {"en": "not set", "ru": "не задан"},
    "acct_today":    {"en": "• Today: <b>{quotas}</b>", "ru": "• Сегодня: <b>{quotas}</b>"},
    "acct_ask":      {"en": "\nWhat would you like to change?",
                      "ru": "\nЧто изменить?"},
    "acct_lang_btn": {"en": "🌐 Language / Язык", "ru": "🌐 Язык / Language"},
    "acct_name_btn": {"en": "✏️ Change name", "ru": "✏️ Сменить имя"},
    "acct_pass_btn": {"en": "🔑 Change password", "ru": "🔑 Сменить пароль"},
    "acct_notif_btn": {"en": "🔔 Toggle online/offline notices",
                       "ru": "🔔 Уведомления об онлайне/офлайне"},
    "acct_mute_btn": {"en": "🔕 Mute feedback reports",
                      "ru": "🔕 Отключить отчёты с отзывами"},
    "acct_unmute_btn": {"en": "📬 Receive feedback reports",
                        "ru": "📬 Получать отчёты с отзывами"},
    "acct_badge_btn": {"en": "🏷 Set badge", "ru": "🏷 Выбрать значок"},
    "acct_logout_btn": {"en": "🚪 Log out / switch profile",
                        "ru": "🚪 Выйти / сменить профиль"},

    # ── admin panel ───────────────────────────────────────────────────────────
    'ax_user_cancelled': {"en": '⛔ An admin cancelled your request: “{text}”', "ru": '⛔ Администратор отменил запрос: «{text}»'},
    'ax_user_stopped': {"en": '⛔ An admin stopped your current requests.', "ru": '⛔ Администратор остановил твои текущие запросы.'},
    'ax_user_say': {"en": '📢 <b>From the admins:</b>\n{text}', "ru": '📢 <b>Администрация:</b>\n{text}'},
    'ax_running': {"en": '⚡ <b>Running</b> ({n})', "ru": '⚡ <b>Выполняется</b> ({n})'},
    'ax_queue': {"en": '📋 <b>Queue</b> ({n})', "ru": '📋 <b>Очередь</b> ({n})'},
    'ax_quiet': {"en": '  <i>idle</i>', "ru": '  <i>тишина</i>'},
    'ax_empty': {"en": '  <i>empty</i>', "ru": '  <i>пусто</i>'},
    'ax_more': {"en": '  … and {n} more', "ru": '  … и ещё {n}'},
    'ax_waits': {"en": 'waiting {t}', "ru": 'ждёт {t}'},
    'ax_users_line': {"en": '👥 approved {a} · pending {p} · banned {b} · backend {be}', "ru": '👥 одобрено {a} · ждут {p} · бан {b} · бэкенд {be}'},
    'ax_btn_cancel': {"en": '⛔ Cancel #{i} ({u})', "ru": '⛔ Отменить #{i} ({u})'},
    'ax_btn_first': {"en": '⬆️ #{i} next', "ru": '⬆️ #{i} первым'},
    'ax_btn_users': {"en": '👥 Users', "ru": '👥 Пользователи'},
    'ax_btn_stats': {"en": '📊 Stats', "ru": '📊 Статистика'},
    'ax_btn_close': {"en": '✖ Close', "ru": '✖ Закрыть'},
    'ax_btn_back': {"en": '⬅️ Back', "ru": '⬅️ Назад'},
    'ax_btn_say': {"en": '✉️ Message as admin', "ru": '✉️ Написать от администрации'},
    'ax_btn_stopall': {"en": '⛔ Stop everything', "ru": '⛔ Остановить всё'},
    'ax_btn_unban': {"en": '✅ Unban', "ru": '✅ Разбанить'},
    'ax_btn_ban': {"en": '🚫 Ban', "ru": '🚫 Забанить'},
    'ax_btn_home': {"en": '🏠 Panel', "ru": '🏠 Панель'},
    'ax_users_title': {"en": '👥 <b>Users</b>', "ru": '👥 <b>Пользователи</b>'},
    'ax_today': {"en": 'today {n}', "ru": 'сегодня {n}'},
    'ax_not_found': {"en": 'User not found.', "ru": 'Пользователь не найден.'},
    'ax_card_meta': {"en": 'id <code>{id}</code> · {st}{adm} · requests today: {n}', "ru": 'id <code>{id}</code> · {st}{adm} · запросов сегодня: {n}'},
    'ax_admin_mark': {"en": ' · 👑 admin', "ru": ' · 👑 админ'},
    'ax_idle_user': {"en": '<i>nothing running</i>', "ru": '<i>сейчас ничего не делает</i>'},
    'ax_paused': {"en": '\n\n⏸ <i>live updates paused — press 🔄</i>', "ru": '\n\n⏸ <i>обновление на паузе — нажми 🔄</i>'},
    'ax_only_admins': {"en": 'Admins only', "ru": 'Только для админов'},
    'ax_t_stopping': {"en": '⛔ Stopping', "ru": '⛔ Останавливаю'},
    'ax_t_dropped': {"en": '✖ Removed from the queue', "ru": '✖ Убрал из очереди'},
    'ax_t_gone': {"en": 'Already finished', "ru": 'Уже завершён'},
    'ax_t_next': {"en": '⬆️ Next up', "ru": '⬆️ Следующий'},
    'ax_t_notq': {"en": 'No longer queued', "ru": 'Уже не в очереди'},
    'ax_t_stopped': {"en": '⛔ Stopped, removed {n}', "ru": '⛔ Остановлено, убрано {n}'},
    'ax_t_banned': {"en": '🚫 Banned', "ru": '🚫 Забанен'},
    'ax_t_unbanned': {"en": '✅ Unbanned', "ru": '✅ Разбанен'},
    'ax_say_prompt': {"en": '✉️ Type the message for <b>{name}</b> — it goes out from the admins.\n/cancel to drop it.', "ru": '✉️ Напиши сообщение для <b>{name}</b> — оно уйдёт от имени администрации.\n/cancel — передумал.'},
    'ax_say_cancel': {"en": 'Cancelled.', "ru": 'Отменено.'},
    'ax_say_sent': {"en": '✅ Sent.', "ru": '✅ Отправлено.'},
    "adm_title":     {"en": "🔐 <b>Admin Panel</b>\n", "ru": "🔐 <b>Панель админа</b>\n"},
    "adm_queue":     {"en": "📋 <b>Queue depth:</b> {n}",
                      "ru": "📋 <b>Длина очереди:</b> {n}"},
    "adm_active":    {"en": "\n⚡ <b>Active dialogs:</b>",
                      "ru": "\n⚡ <b>Активные диалоги:</b>"},
    "adm_pending":   {"en": "\n⏳ <b>Pending approval:</b>",
                      "ru": "\n⏳ <b>Ждут одобрения:</b>"},
    "adm_pending_more": {"en": "\n… and {n} more waiting — only the first {shown} are "
                               "actionable here; approve/reject the rest from the "
                               "desktop GUI.",
                         "ru": "\n… и ещё {n} в ожидании — здесь можно одобрить/отклонить "
                               "только первые {shown}; остальных — через GUI на компьютере."},
    "adm_bcast_tip": {"en": "\n📢 <code>/broadcast &lt;message&gt;</code> to announce.",
                      "ru": "\n📢 <code>/broadcast &lt;текст&gt;</code> — объявление."},
    "adm_stats_title": {"en": "📊 <b>Usage — last 7 days</b>\n",
                        "ru": "📊 <b>Использование — за 7 дней</b>\n"},

    # ── live status while a task runs ─────────────────────────────────────────
    "research_failed": {"en": "❌ Research failed: {err}",
                        "ru": "❌ Исследование не удалось: {err}"},
    "report_caption": {"en": "[deep research report on '{topic}']",
                       "ru": "[отчёт по исследованию: «{topic}»]"},

    # ── keyboard placeholders ─────────────────────────────────────────────────
    "ph_main":       {"en": "Type or tap a button…", "ru": "Напиши или нажми кнопку…"},
    "ph_draw":       {"en": "Draw or edit…",  "ru": "Рисовать…"},
    "ph_search":     {"en": "Search…",        "ru": "Поиск…"},
    "ph_ozon":       {"en": "What to find on Ozon…", "ru": "Что найти на Озоне…"},
    "ph_settings":   {"en": "Settings…",      "ru": "Настройки…"},
    "reply_pick":    {"en": "How should I reply?", "ru": "Как отвечать?"},
    "ph_library":    {"en": "Send a file or ask…", "ru": "Пришли файл или спроси…"},
    "ph_weather":    {"en": "Weather…",        "ru": "Погода…"},
    "ph_creativity": {"en": "Creativity…",     "ru": "Творчество…"},
    "acct_status":   {"en": "• Status: {icon} <b>{state}</b>",
                      "ru": "• Статус: {icon} <b>{state}</b>"},
    "st_approved":   {"en": "approved", "ru": "одобрен"},
    "st_pending":    {"en": "pending",  "ru": "ожидает"},
    "st_rejected":   {"en": "rejected", "ru": "отклонён"},
    "st_banned":     {"en": "banned",   "ru": "заблокирован"},
    "img_missing":   {"en": "⚠️ <b>That didn't actually work.</b> No new image was "
                            "produced, so ignore what I said above. Try again, or "
                            "reword the request.",
                      "ru": "⚠️ <b>На самом деле не получилось.</b> Новое "
                            "изображение не создано — не обращай внимания на "
                            "сообщение выше. Попробуй ещё раз или переформулируй."},
    "already_running": {"en": "⏳ I'm already doing exactly that — hang on.",
                        "ru": "⏳ Уже делаю ровно это — подожди немного."},
    "on_mark":       {"en": "ON ✅",  "ru": "ВКЛ ✅"},
    "off_mark":      {"en": "OFF ❌", "ru": "ВЫКЛ ❌"},
}


def _t(key: str, lang: str = _DEFAULT_LANG, **kw) -> str:
    forms = _MSG.get(key) or {}
    text = _pick_form(forms, lang) or key
    try:
        return text.format(**kw) if kw else text
    except Exception:      # a stray brace in a translation must not kill the reply
        return text
