"""A per-request duration outranks the session's Auto duration setting."""
import json
from types import SimpleNamespace

import pytest
import music as M
import tg_bot as T
import tg_songs as S
from seams import Over


@pytest.mark.parametrize('asked,session_duration,expected', [(30, '', 30), (0, '30', 30), (0, '', 90)])
def test_song_delivery_resolves_auto_from_the_effective_duration(asked, session_duration, expected):
    long = '[verse]\n' + '\n'.join(f'Story line {n}' for n in range(14))
    compact = '[verse]\nCat found the key.\nOpened the door with glee.\n[outro]\nHome to his owner he ran.\nShe hugged the little man.'
    choices, rendered, written = [], [], []
    def choose(ctx, topic, lang, **kw):
        choices.append(kw)
        # A noisy model may return seconds even when only sound fields were requested.
        return {'seconds': 90, 'why': 'relaxed pop'}
    def writer(role, system, user):
        if role == 'draft':
            written.append(user)
        return ''  # real fallback polishing retains the draft, then the shared cap fits it
    ctx = SimpleNamespace(intent_read=lambda *a, **k: {'reply_language': 'en'},
                          yes_no=lambda *a, **k: False,
                          text_model=lambda *a, **k: json.dumps({'lyrics': compact}),
                          deps=SimpleNamespace(lyrics_llm=writer, DRAFTS=1, accent=lambda *a: []))
    class Bot(S.SongsMixin):
        def __init__(self):
            self.sess = T._Session(7)
            self.sess.music_duration = session_duration
            self._store = SimpleNamespace(put=lambda *a: None)
            self.music = Over(M, engine_available=lambda *a: (True, ''), choose_auto_params=choose,
                build_structured_caption=lambda *a, **k: {'lyrics': long, 'style': 'pop'},
                generate_music=lambda ctx, lyrics, style, **kw: rendered.append((lyrics, kw)) or 'song.mp3')
        def _get_session(self, chat): return self.sess
        def _send_audio(self, *a, **k): return True
        def _send_text(self, *a, **k): pass
    bot = Bot()
    assert bot._generate_song(7, 'Cat finds key, opens door and returns home to owner', 'en', asked, ctx=ctx)
    assert rendered[0][1]['duration_s'] == expected
    assert choices[0]['duration'] == (expected == 90)
    assert M.count_sung_lines(rendered[0][0]) <= M.line_budget(expected)[2]
    assert S.LAST_SONG[7] == rendered[0][0]
    if expected == 30:
        assert all(event in rendered[0][0] for event in ('found the key', 'Opened the door', 'Home to his owner'))
        assert 'AT MOST 6 sung lines' in written[0]
    else:
        assert rendered[0][0] == long
