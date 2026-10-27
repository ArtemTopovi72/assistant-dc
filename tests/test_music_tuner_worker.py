from types import SimpleNamespace

import pytest
from gui_music_tab import MusicWorker


@pytest.mark.parametrize('instrumental', [False, True])
def test_desktop_song_uses_the_same_tuner_except_for_instrumentals(instrumental):
    rendered, written = [], []
    engine = SimpleNamespace(
        build_structured_caption=lambda *a, **k: {'lyrics': '[verse]\nraw words', 'style': 'pop'},
        generate_music=lambda ctx, lyrics, style, **k: rendered.append(lyrics) or 'song.wav')
    tuner = SimpleNamespace(write=lambda *a, **k: written.append((a[1], k['max_lines'])) or {'text': '[verse]\ntuned words'})
    worker = MusicWorker(SimpleNamespace(), 'a song about a boot', 'ru', prefs={'instrumental': instrumental}, duration_s=60)
    worker.modules = {'music': engine, 'lyrics_craft': tuner}
    worker.run()
    assert rendered == ['[verse]\nraw words' if instrumental else '[verse]\ntuned words']
    from music import line_budget
    assert written == ([] if instrumental else [('a song about a boot', line_budget(60)[2])])


def test_short_render_keeps_budgeted_draft_when_tuner_returns_too_many_words():
    rendered = []
    engine = SimpleNamespace(
        build_structured_caption=lambda *a, **k: {'lyrics': '[verse]\nshort words', 'style': 'pop'},
        generate_music=lambda ctx, lyrics, style, **k: rendered.append(lyrics) or 'song.wav')
    tuner = SimpleNamespace(write=lambda *a, **k: {'text': '[verse]\n' + 'long line\n' * 60})
    worker = MusicWorker(SimpleNamespace(), 'short song', 'ru', duration_s=60)
    worker.modules = {'music': engine, 'lyrics_craft': tuner}
    worker.run()
    assert rendered == ['[verse]\nshort words']
