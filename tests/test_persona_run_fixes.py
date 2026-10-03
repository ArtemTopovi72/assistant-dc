"""Fixes from the persona run 2026-09-27 (bench/persona_users.py)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stages
import tg_bot
import tg_dispatch


def test_looking_closer_is_russian():
    assert stages.translate("Looking closer (2/4)", "ru") == "Рассматриваю поближе (2/4)"


def test_as_file_request():
    # the model's read is stubbed; the phrases run live in bench/intent_sweep3_live.py
    import intent
    yes = ("пришли файлом", "скинь её файлом", "без сжатия", "as a file", "document", "uncompressed")
    intent.YES_STUB = lambda q, t: "as a FILE" in q and t in yes
    for t in yes:
        assert tg_dispatch._asks_as_file(t), t
    for t in ("нарисуй кота", "вот документ по налогам", "покажи оригинал фото"):
        assert not tg_dispatch._asks_as_file(t), t


def test_help_explains_mic_and_file():
    ru = tg_bot._help_text("ru")
    assert "микрофона" in ru and "файлом" in ru


def test_long_reply_goes_as_text_beside_voice():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(tg_dispatch.__file__))), "bot/tg_tasks.py"), encoding="utf-8").read()
    assert "Every reply goes out as text too" in src


def test_self_forwarded_media_is_material():
    m = {"from": {"id": 7}, "forward_origin": {"type": "user", "sender_user": {"id": 7}}}
    assert not tg_bot._is_forwarded(m)                    # own text: an instruction
    assert tg_bot._is_forwarded(m, self_is_own=False)     # own round video: offer retelling
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(tg_dispatch.__file__))), "bot/tg_dispatch.py"), encoding="utf-8").read()
    assert src.count("_is_forwarded(msg, self_is_own=False)") >= 3


def test_forwarded_bot_messages_and_labels_are_not_acted_on():
    bot = {"forward_origin": {"type": "user", "sender_user": {"id": 1, "is_bot": True}}}
    assert tg_dispatch._from_bot(bot) and not tg_dispatch._from_bot({"forward_origin": {"type": "hidden_user"}})
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(tg_dispatch.__file__))), "bot/tg_dispatch.py"), encoding="utf-8").read()
    assert "_from_bot(msg) or (text and _is_button(text))" in src


def test_song_topic_disarms_and_ready_words_are_asked_about():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(tg_dispatch.__file__))), "bot/tg_registration.py"), encoding="utf-8").read()
    i = src.index('sess.reg_state == "song_topic"')
    body = src[i:i + 2500]
    assert 'sess.reg_state = ""' in body and "song_lyr:keep" in body and "song_lyr:new" in body
    import tg_songs
    assert tg_songs.looks_like_lyrics("раз\nдва три\nчетыре пять")


def test_forward_skip_runs_before_the_prompt_gate():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(tg_dispatch.__file__))), "bot/tg_dispatch.py"), encoding="utf-8").read()
    body = src[src.index("def _dispatch(self"):]
    assert body.index("_from_bot(msg) or") < body.index("self._user_gate(chat_id, msg)")
    assert "< 3600" in body          # «пришли файлом» never digs up an old picture


def test_own_lyrics_layout_is_verbatim_or_nothing():
    import tg_songs, llm, music
    g = "(Интро: тяжелый рифф)\nраз два три\nчетыре пять\nраз два три\nчетыре пять"
    assert "тяжелый рифф" in tg_songs.author_notes(g)
    orig = llm.call_llm_simple
    try:
        llm.call_llm_simple = lambda *a, **k: "[verse]\nраз два три\nчетыре пять\n[chorus]\nраз два три\nчетыре пять"
        assert "[chorus]" in tg_songs.structure_lyrics(object(), g)
        llm.call_llm_simple = lambda *a, **k: "[verse]\nраз два ТРИ\nчетыре шесть"
        assert "четыре шесть" not in tg_songs.structure_lyrics(object(), g)
    finally:
        llm.call_llm_simple = orig
    labelled = "Куплет 1:\nа б\nБридж (резко):\nв г\nVerse 2:\nд е"
    n = music.normalize_lyrics(tg_songs.tag_lyrics(labelled))
    assert n.split("\n") == ["[verse 1]", "а б", "[bridge]", "в г", "[verse 2]", "д е"]


def test_typed_russian_switches_an_unchosen_language_and_ozon_stages_are_russian():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(tg_dispatch.__file__))), "bot/tg_dispatch.py"), encoding="utf-8").read()
    assert "not sess0.lang_chosen" in src and 'sess0.lang = "ru"' in src
    import tg_sessions, stages
    assert "lang_chosen" in open(tg_sessions.__file__, encoding="utf-8").read()
    assert stages.translate("Ozon: choosing among 52 for снегоуборщик", "ru") == "Озон: выбираю из 52: снегоуборщик"


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
