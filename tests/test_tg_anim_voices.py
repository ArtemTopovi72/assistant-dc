"""🎙 Voices for 🎬 Animate + «видео моим голосом», on the REAL bot / tool handler."""
import os, sys, tempfile, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_anv_"))
import tool_image_handlers as H
import video as V
from tests.fakes import Over

CID = 999099


def _bot():
    bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(), lambda: {}, silent_mode=True)
    bot.sent = []
    bot._send_text = lambda cid, text, **kw: bot.sent.append((text, kw.get("keyboard")))
    bot._api_post = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._dl_bytes = lambda fid: b"OggS" + b"0" * 500
    s = bot._get_session(CID); s.clear_context(); s.lang = "ru"; bot._store.put(s)
    return bot


def _cbs(kb):
    return [b["callback_data"] for row in (kb or {}).get("inline_keyboard", []) for b in row]


def test_voices_are_slots_left_to_right():
    """Voices in the video are configurable slots, filled left to right."""
    bot = _bot()
    bot._animate_ask_voices(CID, bot._get_session(CID), "ru")
    assert "animate_preset:wave" in _cbs(bot.sent[-1][1]) and "anv:yes" in _cbs(bot.sent[-1][1])
    bot._cb_anim_voices(CID, "anv:yes")
    text, kb = bot.sent[-1]
    assert "1. ▫️ стандартный" in text and {"anv:slot:0", "anv:slot:2", "anv:done"} <= set(_cbs(kb))
    bot._cb_anim_voices(CID, "anv:slot:1")                       # the middle person first
    assert "человека 2" in bot.sent[-1][0]
    assert bot._anim_voice_take_media(CID, bot._get_session(CID), "ru", {"voice": {"file_id": "a"}})
    s = bot._get_session(CID)
    assert not s.anim_voices[0] and s.anim_voices[1] and not s.anim_voices[2]
    assert "anv:clr:1" in _cbs(bot.sent[-1][1]) and s.anim_voice_state == ""
    assert not bot._anim_voice_take_media(CID, s, "ru", {"voice": {"file_id": "x"}})   # not armed
    old = s.anim_voices[1]
    bot._cb_anim_voices(CID, "anv:slot:1")                       # re-record replaces that slot
    bot._anim_voice_take_media(CID, bot._get_session(CID), "ru", {"voice": {"file_id": "b"}})
    s2 = bot._get_session(CID)
    assert s2.anim_voices[1] != old and not s2.anim_voices[0]
    bot._cb_anim_voices(CID, "anv:clr:1")
    assert not any(bot._get_session(CID).anim_voices)
    bot._cb_anim_voices(CID, "anv:done")
    assert any(c.startswith("animate_preset:") for c in _cbs(bot.sent[-1][1]))


def test_no_skips_to_presets():
    bot = _bot()
    bot._animate_ask_voices(CID, bot._get_session(CID), "ru")
    bot._cb_anim_voices(CID, "anv:no")
    assert any(c.startswith("animate_preset:") for c in _cbs(bot.sent[-1][1]))
    assert not bot._anim_voice_take_media(CID, bot._get_session(CID), "ru", {"voice": {"file_id": "x"}})


def _engine(seen):
    return Over(V, engine_available=lambda ctx: (True, ""),
                generate_video=lambda ctx, d, **kw: seen.update(desc=d, **kw) or {"path": None, "status": "fail"})


def _run_tool(ctx, args):
    seen = {}
    ctx.video_engine = _engine(seen)
    H._handle_generate_video(ctx, {}, args)
    return seen


def test_tool_uses_collected_voices_once(tmp_path):
    a, b = tmp_path / "v1.ogg", tmp_path / "v2.ogg"
    a.write_bytes(b"x"); b.write_bytes(b"x")
    ctx = types.SimpleNamespace(anim_voices=[str(a), str(b)], voice_ref="", set_stage=lambda s: None)
    seen = _run_tool(ctx, {"description": "two people talk", "use_current_images": False})
    assert seen["audios"] == [str(a), str(b)] and "<Audio 2>" in seen["desc"]
    assert ctx.anim_voices == []


def test_empty_slot_keeps_its_person_default(tmp_path):
    b = tmp_path / "v2.ogg"
    b.write_bytes(b"x")
    ctx = types.SimpleNamespace(anim_voices=["", str(b)], voice_ref="", set_stage=lambda s: None)
    seen = _run_tool(ctx, {"description": "two people talk", "use_current_images": False})
    assert seen["audios"] == [str(b)] and "second from the left person speaks with the voice of <Audio 1>" in seen["desc"]


def test_my_voice_needs_a_clone(tmp_path):
    ctx = types.SimpleNamespace(anim_voices=[], voice_ref="", set_stage=lambda s: None)
    ctx.video_engine = _engine({})
    out = H._handle_generate_video(ctx, {}, {"description": "я говорю привет", "use_my_voice": True,
                                             "use_current_images": False})
    assert "Клон голоса" in out
    ref = tmp_path / "me.wav"; ref.write_bytes(b"x")
    ctx.voice_ref = str(ref)
    seen = _run_tool(ctx, {"description": "я говорю привет", "use_my_voice": True,
                                        "use_current_images": False})
    assert seen["audios"] == [str(ref)] and "<Audio 1>" in seen["desc"]


def test_menu_button_ends_collection():
    import tg_strings as S
    bot = _bot()
    bot._animate_ask_voices(CID, bot._get_session(CID), "ru")
    bot._cb_anim_voices(CID, "anv:slot:0")
    bot._resolve_and_push(CID, [{"type": "text", "text": S._BTN["cover_btn"]["ru"]}])
    assert bot._get_session(CID).anim_voice_state == ""


def test_video_link_is_a_voice_sample():
    import tg_links
    bot = _bot()
    bot._animate_ask_voices(CID, bot._get_session(CID), "ru")
    bot._cb_anim_voices(CID, "anv:yes")
    bot._cb_anim_voices(CID, "anv:slot:0")
    bot._run_busy = lambda chat_id, fn, *a, **k: fn(*a)           # the job inline
    bot.links = Over(tg_links, fetch_video=lambda url, *a, **k: {"data": b"\x00" * 600})
    s = bot._get_session(CID)
    assert bot._anim_voice_take_link(CID, s, "ru", "https://www.youtube.com/watch?v=UhPKSA2UhnU")
    assert bot._get_session(CID).anim_voices[0]
    s = bot._get_session(CID)
    s.anim_voice_state = "collect"
    assert not bot._anim_voice_take_link(CID, s, "ru", "просто текст")       # plain text is not ours
    bot._cb_anim_voices(CID, "anv:done")
    assert not bot._anim_voice_take_link(CID, bot._get_session(CID), "ru", "https://youtu.be/x")   # not collecting


def _link_bot(is_voice):
    import tg_links
    bot = _bot()
    bot._run_busy = lambda chat_id, fn, *a, **k: fn(*a)
    bot.links = Over(tg_links, fetch_video=lambda url, *a, **k: {"data": b"\x00" * 600})
    bot.reads = types.SimpleNamespace(yes_no=lambda prompt, text: is_voice)
    bot.queued = []
    bot._enqueue_item = lambda cid, item: bot.queued.append(item)
    return bot


def test_link_named_as_voice_skips_the_card():
    """«Голос возьми отсюда: <ссылка>» in the request itself: the link's sound is slot 1
    and the clip reruns, no «send voices» card."""
    req = "Анимируй. Голос возьми отсюда: https://youtu.be/JCjqy3R4yQw"
    bot = _link_bot(True)
    bot._offer_voices(CID, bot._get_session(CID), "ru", req, 1)
    s = bot._get_session(CID)
    assert s.anim_voices[0] and s.voice_choice == "own"
    assert bot.queued == [{"type": "text", "text": req}]
    assert not any("anv:yes" in _cbs(kb) for _, kb in bot.sent)


def test_link_not_about_voice_still_asks():
    bot = _link_bot(False)
    bot._offer_voices(CID, bot._get_session(CID), "ru", "Анимируй, видео как здесь https://youtu.be/x", 1)
    assert "anv:yes" in _cbs(bot.sent[-1][1]) and not bot.queued


def test_plan_request_ignores_the_rewritten_link_notes():
    """▶ on a plan compares the request with the rerun's text; the summary of a watched video
    link is rewritten on every watch and made the rerun look like a new request."""
    import tg_links
    a = "Анимируй. Голос отсюда: https://youtu.be/x\n[The user's message links a 90-second video «V» (u). Speech in it: A.]"
    b = "Анимируй. Голос отсюда: https://youtu.be/x\n[The user's message links a 90-second video «V» (u). Speech in it: B.]"
    assert tg_links.without_link_notes(a) == tg_links.without_link_notes(b) == "Анимируй. Голос отсюда: https://youtu.be/x"
    assert tg_links.without_link_notes("просто текст") == "просто текст"


def test_every_continuation_part_keeps_the_voice_and_the_first_picture(tmp_path):
    """Parts 2+ copied the sound of the part before: the voice at the end was another man's
    (live 2026-10-19). The voice sample and the original picture ride every part."""
    seen = []
    clip = tmp_path / "c.mp4"; clip.write_bytes(b"x")

    def gen(ctx, desc, **kw):
        seen.append(kw)
        return {"path": str(clip)}
    eng = Over(V, generate_video=gen, motion_context_on=lambda: False, join_continuation=lambda a, b: str(clip),
               _adopt_output=lambda p, d: p, OUTPUT_DIR=str(tmp_path))
    tools = types.SimpleNamespace(seed_frame=lambda a, b: True, cut_tail=lambda a, b: True,
                                  first_frame=lambda a, b: True)
    ctx = types.SimpleNamespace(video_engine=eng, clip_tools=tools, set_stage=lambda s: None,
                                is_cancelled=lambda: False, vision_model=lambda *a, **k: "NONE")
    H._render_remaining_parts(ctx, str(clip), ["p1", "p2", "p3"], {}, people=["orig.jpg"], voices=["v.wav"])
    assert len(seen) == 2
    assert all(k["audios"] == ["v.wav"] and k["images"][1:] == ["orig.jpg"] for k in seen)


def test_a_part_that_skipped_its_opening_lines_is_made_again(tmp_path, monkeypatch):
    """The continuation path had no check that a part said its lines: H3 began mid-speech and
    the clip went out that way (live 2026-10-19)."""
    n, clip = [], tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    eng = Over(V, generate_video=lambda ctx, d, **kw: n.append(d) or {"path": str(clip)},
               motion_context_on=lambda: False, join_continuation=lambda a, b: str(clip),
               _adopt_output=lambda p, d: p, OUTPUT_DIR=str(tmp_path))
    tools = types.SimpleNamespace(seed_frame=lambda a, b: True, cut_tail=lambda a, b: True,
                                  first_frame=lambda a, b: True)
    ctx = types.SimpleNamespace(video_engine=eng, clip_tools=tools, set_stage=lambda s: None,
                                is_cancelled=lambda: False, vision_model=lambda *a, **k: "NONE")
    heard = iter([False, True])
    ctx.speech_ok = lambda c, p, lines: next(heard)
    H._render_remaining_parts(ctx, str(clip), ["p1", 'He says: "Привет, мир."'], {}, voices=["v.wav"])
    assert len(n) == 2


def test_a_talking_head_part_starts_from_the_still_not_from_the_clip_so_far(tmp_path):
    """Each continuation inherited the drift of the one before: the face was a horror mask by the
    end (owner, 2026-10-19). With one still and a voice every part starts from the still."""
    seen, clip = [], tmp_path / "c.mp4"
    clip.write_bytes(b"x")

    def gen(ctx, desc, **kw):
        seen.append(kw)
        return {"path": str(clip)}
    eng = Over(V, generate_video=gen, motion_context_on=lambda: True,
               join_continuation=lambda a, b, **k: str(clip), _adopt_output=lambda p, d: p, OUTPUT_DIR=str(tmp_path))
    tools = types.SimpleNamespace(seed_frame=lambda a, b: True, cut_tail=lambda a, b: True,
                                  first_frame=lambda a, b: True)
    ctx = types.SimpleNamespace(video_engine=eng, clip_tools=tools, set_stage=lambda s: None,
                                is_cancelled=lambda: False, vision_model=lambda *a, **k: "NONE")
    H._render_remaining_parts(ctx, str(clip), ["p1", "p2", "p3"], {}, voices=["v.wav"], fresh_from="orig.jpg")
    assert len(seen) == 2
    # the still is frame 0 (FL2VA) and no voice sample rides: Ref2VA+voice recast the face and the
    # eyes (white-of-eye 0.01-0.04 vs 0.00-0.003); the voice is put on the finished clip (revoice)
    assert all(k["images"] == ["orig.jpg"] and not k.get("videos") and k["keyframe"] and not k.get("audios") for k in seen)


def test_a_keyframe_clip_is_fl2va_a_voice_sample_alone_is_ref2va():
    assert V.pick_mode(["a.jpg"], (), (), keyframe=True) == "t2va"
    assert V.pick_mode(["a.jpg"], (), ["v.wav"]) == "ref2va"
    assert V.pick_mode((), (), (), keyframe=True) == "t2va"
    assert V.pick_mode(["a.jpg"], ["c.mp4"], (), keyframe=True) == "ref2va"



def test_a_talking_head_part_ends_on_its_still_as_well(tmp_path):
    """Without an end frame the eyes and the head wandered off over a part (live 2026-10-20:
    eye match 0.58, the head turned down); the still as the last frame brings the part back."""
    from PIL import Image
    still = tmp_path / "s.jpg"
    Image.new("RGB", (64, 48)).save(still)
    got = {}

    def build(desc, **kw):
        got.update(kw)
        raise V.VideoUnavailable("stop here")
    eng = Over(V, engine_available=lambda ctx: (True, ""), build_workflow=build)
    ctx = types.SimpleNamespace(video_engine=eng, set_stage=lambda s: None)
    V.generate_video(ctx, "He says hello.", images=[str(still)], keyframe=True, seconds=6, prepared=True)
    assert got["images"] == [str(still), str(still)]
    V.generate_video(ctx, "He says hello.", images=[str(still)], seconds=6, prepared=True)
    assert got["images"] == [str(still)]


def test_a_one_frame_join_keeps_every_part_when_the_sound_runs_past_the_picture(tmp_path):
    """The sound ran 0.04 s past the last frame, the one-frame fade reached past the end of the
    clip, and five 10 s parts came out as 20 s of picture under 45 s of sound (live 2026-10-20)."""
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        return
    clips = []
    for k in range(3):
        c = tmp_path / f"p{k}.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=24:duration=2",
                        "-f", "lavfi", "-i", "sine=duration=2.05", "-c:v", "libx264", "-c:a", "aac", str(c)], check=True)
        clips.append(str(c))
    path = clips[0]
    for c in clips[1:]:
        path = V.join_continuation(path, c, fade=1 / 24, drop_frames=1)
    assert V.probe(path)["video_seconds"] > 5.8

def test_every_front_end_takes_a_voice_link_through_one_engine(tmp_path):
    import voice_clone
    got = voice_clone.fetch_sample("u", str(tmp_path), lambda url: {"data": b"x" * 9})
    assert got and os.path.getsize(got) == 9
    assert voice_clone.fetch_sample("u", str(tmp_path), lambda url: {}) == ""
    link = "https://youtu.be/JCjqy3R4yQw"
    assert voice_clone.voice_link_in(lambda *a: True, "голос возьми отсюда: " + link) == link
    assert voice_clone.voice_link_in(lambda *a: False, "смотри " + link) == ""
    assert voice_clone.voice_link_in(lambda *a: True, "без ссылки") == ""


def _voice_gate_run(tmp_path, monkeypatch, sims):
    """One continuation part, its voice similarities handed out in order; returns (takes made, the clip joined)."""
    takes, joined = [], []
    clips = [tmp_path / f"t{k}.mp4" for k in range(3)]
    for c in clips:
        c.write_bytes(b"x")

    def gen(ctx, desc, **kw):
        takes.append(desc)
        return {"path": str(clips[len(takes) - 1])}
    eng = Over(V, generate_video=gen, motion_context_on=lambda: False,
               join_continuation=lambda a, b, **k: joined.append(b) or str(clips[2]),
               _adopt_output=lambda p, d: p, OUTPUT_DIR=str(tmp_path))
    tools = types.SimpleNamespace(seed_frame=lambda a, b: True, cut_tail=lambda a, b: True,
                                  first_frame=lambda a, b: True)
    ctx = types.SimpleNamespace(video_engine=eng, clip_tools=tools, set_stage=lambda s: None,
                                is_cancelled=lambda: False, vision_model=lambda *a, **k: "NONE")
    it = iter(sims)
    ctx.voice_sim = lambda clip, voices: next(it)
    ctx.speech_ok = lambda c, p, lines: True
    H._render_remaining_parts(ctx, str(clips[2]), ["p1", 'He says: "Привет, мир."'], {}, voices=["v.wav"])
    return takes, [os.path.basename(j) for j in joined]


def test_a_part_whose_voice_drifted_is_made_again_and_the_closer_take_stays(tmp_path, monkeypatch):
    """H3 invents the timbre of each part: 0.46..0.95 against the same reference (night 2026-10-20),
    and the voice «degraded with the length» (owner). Below the bar the part is drawn again; the
    take nearer to the voice goes into the clip."""
    takes, joined = _voice_gate_run(tmp_path, monkeypatch, [0.5, 0.4])
    assert len(takes) == 2 and joined == ["t0.mp4"]
    takes, joined = _voice_gate_run(tmp_path, monkeypatch, [0.5, 0.9])
    assert len(takes) == 2 and joined == ["t1.mp4"]


def test_a_part_in_the_voice_is_made_once(tmp_path, monkeypatch):
    takes, _ = _voice_gate_run(tmp_path, monkeypatch, [0.9])
    assert len(takes) == 1
    takes, _ = _voice_gate_run(tmp_path, monkeypatch, [None])     # a check that could not run never blocks
    assert len(takes) == 1
