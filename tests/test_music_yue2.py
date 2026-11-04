"""YuE2 is the default engine; its lyric and style inputs follow the model card."""
import os, sys
import types
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


def _deps(**kw):
    return types.SimpleNamespace(deps=types.SimpleNamespace(**kw))


def test_default_engine_is_yue2():
    calls = []
    ctx = _deps(yue2_available=lambda: True, yue2_cpp_available=lambda: False,
                _generate_yue2=lambda ctx, l, s, seed, prefs=None: calls.append(s) or __file__,
                apply_fade=lambda p: False)
    assert M.MUSIC_ENGINE == "yue2" and M.engine_available(ctx) == (True, "")
    assert M.generate_music(ctx, "[verse]\nраз", "pop") == __file__ and calls == ["pop"]


def test_cpp_render_uses_16_ode_steps():
    jobs = []
    ctx = _deps(yue2_cpp_available=lambda: True,
                run_gpu_worker=lambda ctx, py, sc, job, *a, **k: jobs.append(job) or (True, ""),
                _valid_audio_file=lambda p: True, _master=lambda *a, **k: None)
    M._generate_yue2(ctx, "[verse]\nраз", "pop", 5)
    assert jobs[0]["steps"] == 16


_SCENIC = "75 BPM, Lo-fi hip hop, male vocal, minor key, lonely, dimly lit apartment at dusk, study music"


def test_style_cleanup_drops_scenery_keeps_tempo_and_vocal():
    ctx = types.SimpleNamespace(deps=types.SimpleNamespace(
        lyrics_llm=lambda r, s, u: '{"tags": ["75 BPM", "Lo-fi hip hop", "male vocal", "minor key", "lonely"]}'))
    assert M.yue2_tags(ctx, _SCENIC) == "75 BPM, Lo-fi hip hop, male vocal, minor key, lonely"


def test_style_cleanup_puts_back_a_lost_bpm_or_vocal_and_keeps_the_rest():
    """Throwing the whole cleanup away for one lost tag sent the emotional-arc prose to YuE2
    (2026-10-22): the BPM and vocal tags go back in front, the scenery stays out."""
    def said(answer):
        return types.SimpleNamespace(deps=types.SimpleNamespace(lyrics_llm=lambda r, s, u: answer))
    got = M.yue2_tags(said('{"tags": ["Lo-fi hip hop", "lonely", "ambient"]}'), _SCENIC)
    assert "75 BPM" in got and "male vocal" in got and "dimly lit" not in got, got
    assert M.yue2_tags(said("nonsense"), _SCENIC) == _SCENIC


def test_russian_lyrics_name_the_language_in_the_style():
    assert M.yue2_language("male vocal, pop", "Мы идём по дороге") == "male vocal, Russian vocal, pop"
    assert M.yue2_language("male vocal, pop", "We walk") == "male vocal, pop"


def test_the_vocal_button_reaches_yue2_as_a_positive_tag():
    # «no female vocals» would put the word «female» in the tags
    p = M.prefs_from(genre="punk", vocal="male")
    got = M.yue2_pin("female vocal, Pop, Bright female soprano lead vocal, 120 BPM, synth", p)
    assert got.split(", ")[0] == "male vocal" and "female" not in got
    assert "Punk rock" in got and "Pop" not in got.split(", ")
    assert "female" not in M.enforce_vocal("Vocal Details: female soprano.", p["vocal"]).replace("male vocal", "")
    got = M.yue2_pin("male vocal, man's voice, Pop, 120 BPM", M.prefs_from(vocal="female"))
    assert got.split(", ")[0] == "female vocal" and "man" not in got.replace("woman", "")


def test_an_instrumental_has_no_vocal_tags_and_no_words():
    p = M.prefs_from(vocal="instrumental")
    assert M.yue2_pin("Pop, Russian vocal, airy female vocal, synth", p) == "instrumental, Pop, synth"
    seen = {}
    ctx = _deps(yue2_cpp_available=lambda: True, yue2_tags=lambda ctx, s: s,
                _render_yue2_once=lambda ctx, cpp, job, seed: (seen.update(job), "")[1],
                _valid_audio_file=lambda p: True, _master=lambda *a, **k: None)
    M._generate_yue2(ctx, "[verse]\nслова песни", "Pop, synth", 1, prefs=p)
    assert seen["lyrics"] == "[instrumental]" and seen["style"].startswith("instrumental")


def test_an_instrumental_keeps_its_pinned_genre():
    # «Vocal jazz / jazz ballad / swing»: the vocal filter took the only genre tag with it
    p = M.prefs_from(genre="jazz", vocal="instrumental")
    got = M.yue2_pin("Vocal jazz, piano, 70 BPM", p)
    assert got.startswith("instrumental, ") and "jazz ballad" in got and "vocal" not in got.lower()
    assert M.yue2_preview(M.prefs_from(genre="jazz", vocal="male")).startswith("male vocal, Vocal jazz")


def test_pin_abc_tempo_changes_only_header():
    abc = "X:1\nM:4/4\nQ:1/4=104\nK:C\nC D E F |\n"
    out = M._pin_yue2_abc_tempo(abc, 140)
    assert out == abc.replace("Q:1/4=104", "Q:1/4=140")
    assert M._pin_yue2_abc_tempo(abc + "Q:1/4=90\n", 140).count("Q:1/4=140") == 2
    assert M._pin_yue2_abc_tempo("not an ABC score", 140) == ""


def test_yue2_plans_then_pins_before_synthesis(tmp_path):
    import contextlib
    seen = []
    def worker(ctx, py, script, job, label, timeout, cmd=None, env=None):
        seen.append((label, dict(job)))
        if label == "YuE2 plan":
            from pathlib import Path
            Path(cmd[cmd.index("--out") + 1]).write_text("X:1\nQ:1/4=104\nK:C\nC D |\n", encoding="utf8")
        return True, "ok"
    ctx = _deps(run_gpu_worker=worker, _gpu_slot=lambda **kw: contextlib.nullcontext())
    M._render_yue2_once(ctx, True, {"out": str(tmp_path / "song.mp3"), "target_bpm": 140}, 7)
    assert [x[0] for x in seen] == ["YuE2 plan", "YuE2"]
    assert "Q:1/4=140" in seen[1][1]["abc"]
    assert "target_bpm" not in seen[1][1]


def test_yue2_invalid_plan_does_not_claim_controlled_tempo(tmp_path):
    import contextlib, pytest
    seen = []
    def worker(ctx, py, script, job, label, timeout, cmd=None, env=None):
        seen.append(label)
        return False, "plan failed"
    ctx = _deps(run_gpu_worker=worker, _gpu_slot=lambda **kw: contextlib.nullcontext())
    with pytest.raises(M.MusicUnavailable, match="tempo-controlled"):
        M._render_yue2_once(ctx, True, {"out": str(tmp_path / "song.mp3"), "target_bpm": 140}, 7)
    assert seen == ["YuE2 plan"]


def test_yue2_cancel_during_plan_releases_gpu_slot(tmp_path):
    import contextlib, pytest
    phases = []
    @contextlib.contextmanager
    def slot(**kw):
        phases.append("entered")
        try:
            yield
        finally:
            phases.append("released")
    def worker(ctx, py, script, job, label, timeout, cmd=None, env=None):
        phases.append(label)
        raise M.MusicUnavailable("cancelled")
    ctx = _deps(run_gpu_worker=worker, _gpu_slot=slot)
    with pytest.raises(M.MusicUnavailable, match="cancelled"):
        M._render_yue2_once(ctx, True, {"out": str(tmp_path / "song.mp3"), "target_bpm": 140}, 7)
    assert phases == ["entered", "YuE2 plan", "released"]
