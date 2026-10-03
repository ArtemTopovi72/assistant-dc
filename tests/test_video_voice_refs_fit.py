"""Live 2026-10-01: «два деда дерутся» with two voice samples spoke with neither
voice. The samples were 43-51 s (H3 takes 2-15 s, 15 s for all), the turn sent
9 old chat pictures as references, and a redo reused the last clip, whose own
soundtrack rides along with it."""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")
from pydub import AudioSegment
from pydub.generators import Sine
import video as V


def _speechy(path, seconds):
    tone, gap = Sine(220).to_audio_segment(duration=900), AudioSegment.silent(duration=500)
    seg = AudioSegment.empty()
    while len(seg) < seconds * 1000:
        seg += tone + gap
    seg.export(path, format="wav")
    return path


def test_long_samples_fit_the_budget(tmp_path):
    a = _speechy(str(tmp_path / "a.wav"), 43)
    b = _speechy(str(tmp_path / "b.wav"), 51)
    out = V.fit_audio_refs([a, b])
    total = sum(len(AudioSegment.from_file(p)) for p in out)
    assert total <= V.AUDIO_REF_TOTAL_MS and all(len(AudioSegment.from_file(p)) >= 2000 for p in out)


def test_short_sample_is_untouched(tmp_path):
    a = _speechy(str(tmp_path / "a.wav"), 5)
    assert V.fit_audio_refs([a]) == [a]


def test_voices_drop_the_previous_clip(tmp_path):
    import tool_image_handlers as H, tools as T
    V.engine_available = lambda ctx: (True, "")
    T._render_budget_exhausted = lambda *a, **k: None
    seen = []
    orig = V.generate_video
    V.generate_video = lambda ctx, d, **k: seen.append(k) or {"status": "cancelled"}
    try:
        clip, voice = tmp_path / "old.mp4", tmp_path / "v.wav"
        clip.write_bytes(b"x"); voice.write_bytes(b"x")
        ctx = types.SimpleNamespace(voice_choice="own", anim_voices=[str(voice)], set_stage=lambda s: None,
                                    last_image_path=None, last_video_path=str(clip))
        H._handle_generate_video(ctx, {}, {"description": "two men talk", "speakers": 2,
                                           "use_current_images": False, "use_current_video": True})
        assert seen[-1]["videos"] == [] and seen[-1]["audios"] == [str(voice)]
    finally:
        V.generate_video = orig
