"""A cover with new words is RE-SUNG on the original's full score (10-07: «3 сентября» with
the Govnovoz words -- every attempt to squeeze the words into the original's syllables was
rejected; YuE2 on the full score got «МОЛОДЕЦ!!!»)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "core", "agent", "media", "voice", "bot"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("F5_TEST_RUN", "1")

import cover  # noqa: E402
import llm  # noqa: E402
import music  # noqa: E402
import remix as R  # noqa: E402

ABC = """X:1
Q:1/4=68
K:Fm
% intro
"Fm" z4 |
V: Vocal
% verse
"Fm" C D E F C D E F |
% chorus
"Db" A G F E A G F E A G |
% inst
z4 |
% verse
"Fm" C D E F C D E F |
% chorus
"Db" A G F E A G F E A G |
V: Ins
% outro
A B c d e f g a b c d |
"""


def test_sections_follow_the_score():
    assert R._score_sections(ABC) == ["Intro", "Verse", "Chorus", "Interlude", "Verse", "Chorus", "Outro"]


def test_section_notes_count_only_the_melody():
    assert R._score_parts(ABC) == [("Intro", 0), ("Verse", 8), ("Chorus", 10), ("Interlude", 0),
                                   ("Verse", 8), ("Chorus", 10), ("Outro", 0)]


def test_tempo_and_key_read_from_the_score():
    assert R._score_tempo_key(ABC) == "68 bpm, F minor"
    assert R._score_tempo_key("Q:1/4=120\nK:Bb\n") == "120 bpm, B flat major"


def test_layout_places_every_line_on_the_sung_sections(monkeypatch):
    monkeypatch.setattr(llm, "call_llm_simple", lambda *a, **k: '{"sections": [[1, 2], [5, 6], [3, 4], [5, 6]]}')
    text = R._layout(None, "ла ла\nли ли\nло ло\nлу лу\nле ле\nлы лы", R._score_parts(ABC))
    assert text.split("\n\n") == ["[Intro]", "[Verse]\nла ла\nли ли", "[Chorus]\nле ле\nлы лы", "[Interlude]",
                                  "[Verse]\nло ло\nлу лу", "[Chorus]\nле ле\nлы лы", "[Outro]"]


def test_layout_keeps_a_section_within_its_notes(monkeypatch):
    # 10-07: Cheri Cheri Lady -- 144 syllables on a 44-note chorus came out as mumbling.
    monkeypatch.setattr(llm, "call_llm_simple", lambda *a, **k: '{"sections": [[1, 2, 3]]}')
    text = R._layout(None, "ла ла ла ла ла\nли ли ли ли ли\nло ло ло ло ло", [("Chorus", 10)])
    assert text == "[Chorus]\nла ла ла ла ла\nли ли ли ли ли"


def test_a_verse_never_resings_another_verse(monkeypatch):
    # 10-07: the model gave verse 2 the lines of verse 1; the third verse's lines were never sung.
    monkeypatch.setattr(llm, "call_llm_simple", lambda *a, **k: '{"sections": [[1, 2], [5], [1, 2]]}')
    text = R._layout(None, "ла ла\nли ли\nло ло\nлу лу\nле ле", [("Verse", 8), ("Chorus", 10), ("Verse", 8)])
    assert text == "[Verse]\nла ла\nли ли\n\n[Chorus]\nле ле\n\n[Verse]\nло ло\nлу лу"


def test_a_silent_section_is_instrumental(monkeypatch):
    monkeypatch.setattr(llm, "call_llm_simple", lambda *a, **k: '{"sections": [[1]]}')
    assert R._layout(None, "ла ла", [("Verse", 2), ("Chorus", 20)]) == "[Interlude]\n\n[Chorus]\nла ла"


def test_layout_survives_a_broken_plan(monkeypatch):
    monkeypatch.setattr(llm, "call_llm_simple", lambda *a, **k: '{"sections": [[99]]}')
    text = R._layout(None, "ла\nли\nло\nлу", [("Verse", 20), ("Chorus", 20)])
    assert text == "[Verse]\nла\nли\n\n[Chorus]\nло\nлу"


class _Whisper:
    def transcribe(self, path, **k):
        return [type("S", (), {"words": [type("W", (), {"word": " я", "start": 0.0, "end": 0.3})()],
                               "text": "я"})()], None


def test_resing_renders_the_full_score_with_new_words(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(R, "_stems", lambda song, work, name: (np.zeros(10), np.zeros(10)))
    monkeypatch.setattr(R, "_score", lambda ctx, wav, work: ABC)
    monkeypatch.setattr(R, "_sung_words", lambda segs: [{"w": "я", "s": 0, "e": 1}])
    monkeypatch.setattr(llm, "call_llm_simple", lambda ctx, sys_, user, **k:
                        "russian chanson, male baritone" if "style" in sys_ else '{"sections": [[1], [2], [1], [2]]}')

    def render(ctx, cpp, job, seed):
        seen.update(job)
        raise cover.CoverFailed("stop")
    monkeypatch.setattr(music, "_render_yue2_once", render)
    ctx = type("C", (), {"models": type("M", (), {"whisper": _Whisper()})()})()
    with pytest.raises(cover.CoverFailed):
        R.resing(ctx, "song.mp3", "строка раз\nстрока два")
    assert seen["abc"] == ABC
    assert "[Chorus]" in seen["lyrics"] and "строка два" in seen["lyrics"]
    assert "68 bpm, F minor" in seen["style"] and "chanson" in seen["style"]


def test_resing_needs_words():
    with pytest.raises(cover.CoverFailed):
        R.resing(None, "song.mp3", "[Verse]\n\n")


def test_yue2_renders_with_lyric_guidance(monkeypatch):
    # 10-07: at the protocol's cfg 1.0 (no guidance) Whisper heard 1 of 21 lines; at 3.0, 20 of 21.
    seen = {}
    monkeypatch.setattr(music, "run_gpu_worker", lambda ctx, py, script, job, *a, **k: (seen.update(job), (True, ""))[1])
    music._render_yue2_once(None, True, {"out": "x.mp3", "lyrics": "", "style": ""}, 7)
    assert seen["cfg_scale"] >= 2.0 and seen["cot"] == "full" and seen["lm_seed"] == 7
