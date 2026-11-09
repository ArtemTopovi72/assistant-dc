import tg_songs


def test_explicit_quoted_lyrics_are_preserved():
    request = 'Спой эти строки как есть: «За окном рассвет. Город уже проснулся». Без музыки.'
    assert tg_songs.quoted_lyrics(request) == 'За окном рассвет. Город уже проснулся'
    assert tg_songs.quoted_lyrics('Спой этот текст слово в слово: "Hello world"') == 'Hello world'
    assert tg_songs.quoted_lyrics('Эти строки: «Тихо светает» — спой как есть') == 'Тихо светает'


def test_song_title_or_ambiguous_quotes_are_not_treated_as_lyrics():
    assert not tg_songs.quoted_lyrics('Сочини песню про «Лето» как есть')
    assert not tg_songs.quoted_lyrics('Спой эти строки как есть: «один» и «два»')
    assert not tg_songs.quoted_lyrics('Спой эти строки, придумай продолжение: «Привет»')
