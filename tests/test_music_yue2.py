"""YuE2 is the default engine; its lyric and style inputs follow the model card."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import music as M


def test_tags_follow_the_card():
    got = M.yue2_lyrics("[intro]\n[verse 2]\nраз\nдва\n[chorus]\nлети\n[instrumental]")
    assert got == "[Intro]\n\n[Verse]\nраз\nдва\n\n[Chorus]\nлети\n\n[Interlude]"


def test_style_is_a_short_tag_list():
    got = M.yue2_style("Global Metadata: Russian pop, 120 BPM. The track builds from a hush.\n"
                       "Vocal Details: Female lead vocal, clear and warm.\n"
                       "Arrangement: Acoustic guitar with bright drums.")
    assert got.startswith("Russian pop, 120 BPM") and "Female lead vocal" in got
    assert "Global Metadata" not in got and len(got) <= 300


def test_default_engine_is_yue2(monkeypatch):
    monkeypatch.setattr(M, "yue2_available", lambda: True)
    assert M.MUSIC_ENGINE == "yue2" and M.engine_available() == (True, "")
    calls = []
    monkeypatch.setattr(M, "_generate_yue2", lambda ctx, l, s, seed, prefs=None: calls.append(s) or __file__)
    monkeypatch.setattr(M, "apply_fade", lambda p: False)
    assert M.generate_music(None, "[verse]\nраз", "pop") == __file__ and calls == ["pop"]


def test_cpp_render_uses_16_ode_steps(monkeypatch, tmp_path):
    jobs = []
    monkeypatch.setattr(M, "yue2_cpp_available", lambda: True)
    monkeypatch.setattr(M, "run_gpu_worker", lambda ctx, py, sc, job, *a, **k: jobs.append(job) or (True, ""))
    monkeypatch.setattr(M, "_valid_audio_file", lambda p: True)
    M._generate_yue2(None, "[verse]\nраз", "pop", 5)
    assert jobs[0]["steps"] == 16


_SCENIC = "75 BPM, Lo-fi hip hop, male vocal, minor key, lonely, dimly lit apartment at dusk, study music"


def test_style_cleanup_drops_scenery_keeps_tempo_and_vocal(monkeypatch):
    import lyrics_craft
    monkeypatch.setattr(lyrics_craft, "LLM_STUB",
                        lambda r, s, u: '{"tags": ["75 BPM", "Lo-fi hip hop", "male vocal", "minor key", "lonely"]}')
    assert M.yue2_tags(None, _SCENIC) == "75 BPM, Lo-fi hip hop, male vocal, minor key, lonely"


def test_style_cleanup_refuses_an_answer_that_lost_bpm_or_vocal(monkeypatch):
    import lyrics_craft
    monkeypatch.setattr(lyrics_craft, "LLM_STUB", lambda r, s, u: '{"tags": ["Lo-fi hip hop", "lonely", "ambient"]}')
    assert M.yue2_tags(None, _SCENIC) == _SCENIC
    monkeypatch.setattr(lyrics_craft, "LLM_STUB", lambda r, s, u: "nonsense")
    assert M.yue2_tags(None, _SCENIC) == _SCENIC


def test_russian_lyrics_name_the_language_in_the_style():
    assert M.yue2_language("male vocal, pop", "Мы идём по дороге") == "male vocal, Russian vocal, pop"
    assert M.yue2_language("male vocal, pop", "We walk") == "male vocal, pop"


def test_the_vocal_button_reaches_yue2_as_a_positive_tag():
    # 10-09 «мужской не доезжал»: «no female vocals» put the word «female» in the tags
    p = M.prefs_from(genre="punk", vocal="male")
    got = M.yue2_pin("female vocal, Pop, Bright female soprano lead vocal, 120 BPM, synth", p)
    assert got.split(", ")[0] == "male vocal" and "female" not in got
    assert "Punk rock" in got and "Pop" not in got.split(", ")
    assert "female" not in M.enforce_vocal("Vocal Details: female soprano.", p["vocal"]).replace("male vocal", "")
    got = M.yue2_pin("male vocal, man's voice, Pop, 120 BPM", M.prefs_from(vocal="female"))
    assert got.split(", ")[0] == "female vocal" and "man" not in got.replace("woman", "")


def test_an_instrumental_has_no_vocal_tags_and_no_words(monkeypatch, tmp_path):
    # 10-09 «инструментал со словами»
    p = M.prefs_from(vocal="instrumental")
    assert M.yue2_pin("Pop, Russian vocal, airy female vocal, synth", p) == "instrumental, Pop, synth"
    seen = {}
    monkeypatch.setattr(M, "yue2_cpp_available", lambda: True)
    monkeypatch.setattr(M, "yue2_tags", lambda ctx, s: s)
    monkeypatch.setattr(M, "_render_yue2_once", lambda ctx, cpp, job, seed: (seen.update(job), "")[1])
    monkeypatch.setattr(M, "_valid_audio_file", lambda p: True)
    monkeypatch.setattr(M, "_master", lambda *a, **k: None)
    M._generate_yue2(None, "[verse]\nслова песни", "Pop, synth", 1, prefs=p)
    assert seen["lyrics"] == "[instrumental]" and seen["style"].startswith("instrumental")


def test_an_instrumental_keeps_its_pinned_genre():
    # «Vocal jazz / jazz ballad / swing»: the vocal filter took the only genre tag with it
    p = M.prefs_from(genre="jazz", vocal="instrumental")
    got = M.yue2_pin("Vocal jazz, piano, 70 BPM", p)
    assert got.startswith("instrumental, ") and "jazz ballad" in got and "vocal" not in got.lower()
    assert M.yue2_preview(M.prefs_from(genre="jazz", vocal="male")).startswith("male vocal, Vocal jazz")
