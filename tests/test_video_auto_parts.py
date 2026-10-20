"""A script the bot wrote itself is one or two episodes: 5-8 parts took ~25 min to render
and the longer it ran the more nonsense it held (owner, 2026-10-15). A simple action stays
one part; a length the user named is still followed."""
import os
import types
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "core", "agent", "media"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("F5_TEST_RUN", "1")
os.environ.pop("VIDEO_AUTO_MAX_PARTS", None)

import pytest  # noqa: E402
import tg_continue  # noqa: E402
import tool_image_handlers as H  # noqa: E402
import video as V  # noqa: E402
from tests.fakes import Over  # noqa: E402

LONG = " ".join(f"Then the cat does trick number {i} and the dog answers it with a bark." for i in range(14))


class _Ctx:
    reference_images = []
    voice_choice = None
    continue_tail = continue_src = ""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def set_stage(self, *_): pass
    def is_cancelled(self): return False
    def remember(self, *a): pass
    def memory_text(self): return ""


def _run(tmp_path, args, condensed="", state=None, ctx_kw=None, **eng):
    calls = []

    def fake_gen(ctx, description, **kw):
        out = tmp_path / f"part{len(calls) + 1}.mp4"
        out.write_bytes(b"v")
        calls.append(description)
        return {"path": str(out), "status": "success", "seconds": 5.2, "width": 768, "height": 1344}

    fakes = dict(AUTO_MAX_PARTS=2, generate_video=fake_gen, engine_available=lambda ctx: (True, ""),
                 herrgott_on=lambda: False, motion_context_on=lambda: False,
                 mark_inserts=lambda ctx, ps: ps, validate_script=lambda ctx, ps, **k: ps,
                 condense_script=lambda *a, **k: condensed,
                 join_continuation=lambda a, b, **k: str(tmp_path / "joined.mp4"),
                 _adopt_output=lambda p, *a: p, probe=lambda p: {"seconds": 10.4})
    ctx = _Ctx(**(ctx_kw or {}),
        video_engine=Over(V, **{**fakes, **eng}),
        clip_tools=Over(tg_continue, seed_frame=lambda src, out: open(out, "wb").write(b"j") > 0,
                        cut_tail=lambda src, out: open(out, "wb").write(b"t") > 0,
                        first_frame=lambda src, out: False))
    H._handle_generate_video(ctx, state or {}, {"use_current_images": False, **args})
    return calls


def test_a_long_script_of_its_own_is_at_most_two_parts(tmp_path):
    assert len(V.split_script(LONG)) > 2
    assert len(_run(tmp_path, {"description": LONG})) == 2


def test_a_simple_action_is_one_part(tmp_path):
    assert len(_run(tmp_path, {"description": "The cat jumps onto the sofa."})) == 1


def test_a_named_length_is_followed(tmp_path):
    assert len(_run(tmp_path, {"description": LONG, "seconds": 30})) > 2


def test_a_where_it_continues_from_line_is_not_a_part_of_its_own():
    # «The video continues from the first frame…» was a part with no action, rewritten
    # into a copy of the next part, which then shrank to «the cat walks» (2026-10-17).
    d = ("The video continues from the first frame where the cat is on the chair. The cat "
         "jumps down from the chair onto the floor, lands gracefully, and begins walking "
         "towards the dumbbells. The camera follows the cat with a low tracking shot.")
    assert V.split_script(d) == [d]


def test_long_monologue_is_cut_into_parts_at_sentence_ends():
    """85 words in one pair of quotes rode a single 10 s part: the clip said the first two
    sentences and the rest of the user's speech was lost (live 2026-10-19)."""
    speech = ("Ах, вот и ты. Хорошо. Ну, единственное хорошее — что ты здесь. У нас проблема. Большая. "
              "Емкость для хранения заготовок переполнена. Нет места — нет хруща. Серьезно, это критично "
              "для нашего выживания. Тебе нужно освободить нам место в холодильнике и забрать банки. "
              "Нам нужен твой морозильник. Слушай, просто будь с рюкзачком, хорошо?")
    parts = V.split_script(f'The caretaker speaks to the camera: "{speech}"')
    assert len(parts) >= 3
    said = " ".join(w for p in parts for q in V._QUOTED.finditer(p) for w in next(g for g in q.groups() if g).split())
    assert said == speech.replace(" — ", " — ")
    assert all(V.estimate_seconds(p) <= 10.2 for p in parts)


def test_users_dictated_speech_is_not_the_bots_script_to_condense():
    said = "Анимируй. Пусть он скажет: Ах, вот и ты. Хорошо! Слушай, просто будь с рюкзачком."
    mine = 'He looks at the camera and says: "Ах, вот и ты. Хорошо! Слушай, просто будь с рюкзачком."'
    assert V.speech_is_users(mine, said)
    assert not V.speech_is_users('He says: "Привет, дружище, как дела?"', said)      # the bot's own line
    assert not V.speech_is_users("He waves.", said)



def test_a_dictated_monologue_stays_whole_when_the_agent_retold_it_in_english(tmp_path):
    """The agent quoted the speech in English and cut short; the user's 106 words were added
    beside it, the script looked like the bot's own and went out as one 10 s part (live 2026-10-20)."""
    speech = " ".join(f"Слушай внимательно, это правило номер {i}, его нельзя забывать." for i in range(12))
    calls = _run(tmp_path, {"description": 'An old man says: "Listen carefully, rule number one."'},
                 condensed="Он говорит: «Коротко.»", state={"user_input_original": "Сделай видео, пусть он скажет: " + speech},
                 dictated_speech=lambda said, raw, ctx: speech)
    assert len(calls) > 2
    assert not any("Listen carefully" in c for c in calls)


def test_no_sentence_of_a_dictated_monologue_is_lost_on_the_way_to_the_parts(tmp_path):
    """Two of the user's sentences never reached the plan, whatever step dropped them (live 2026-10-20)."""
    speech = " ".join(f"Слушай внимательно, это правило номер {i}, его нельзя забывать." for i in range(8))
    desc = 'An old man says: "' + speech + '"'
    eaten = lambda ctx, ps, **k: [ps[0]] + ps[2:]            # an editor that drops a part
    calls = _run(tmp_path, {"description": desc}, state={"user_input_original": "Сделай видео, пусть он скажет: " + speech},
                 dictated_speech=lambda said, raw, ctx: speech, validate_script=eaten)
    assert not V.lost_sentences(speech, calls)


def test_a_talking_heads_parts_get_no_invented_actions(tmp_path):
    """«points to a map», «gestures toward the refrigerator»: the head turned from the camera (live 2026-10-20)."""
    speech = " ".join(f"Слушай внимательно, это правило номер {i}, его нельзя забывать." for i in range(8))
    still = tmp_path / "s.jpg"
    from PIL import Image
    Image.new("RGB", (64, 48)).save(still)
    seen, looked = [], []
    calls = _run(tmp_path, {"description": 'An old man says: "' + speech + '"', "use_current_images": True},
                 state={"user_input_original": "Сделай видео, пусть он скажет: " + speech},
                 dictated_speech=lambda said, raw, ctx: speech, is_talking_head=lambda ctx, d, s="", **k: True,
                 validate_script=lambda ctx, ps, **k: seen.append(ps) or [p + " He points to a map." for p in ps],
                 ctx_kw={"deps": types.SimpleNamespace(_current_image_paths=lambda ctx, st: [str(still)]),
                         "vision_model": lambda ctx, **kw: looked.append(kw.get("user_text", "")) or "NONE",
                         "modules": {"video_look": types.SimpleNamespace(
                             duration_seconds=lambda p: 6.0, sample_times=lambda d, **k: [0.0, 3.0],
                             extract_frames=lambda p, ts, d: [(t, str(still)) for t in ts])}})
    assert not seen and not any("map" in c for c in calls)
    # nor is the clip checked for actions: against «the camera zooms in» the note said one was
    # missing, and the reply told the user to «pick a voice variant» (live 2026-10-20)
    assert not any("ACTIONS of the script" in t for t in looked)


@pytest.mark.parametrize("framed", [True, False])
def test_a_talking_heads_part_whose_light_drifted_goes_in_pulled_back(tmp_path, framed):
    """Pulled only on the still's framing: run 18's push-in came out with a blue beard (2026-10-20)."""
    speech = " ".join(f"Слушай внимательно, это правило номер {i}, его нельзя забывать." for i in range(8))
    still = tmp_path / "s.jpg"
    from PIL import Image
    Image.new("RGB", (64, 48)).save(still)
    joined = []
    pulled = str(tmp_path / "pulled.mp4")
    open(pulled, "wb").write(b"p")
    _run(tmp_path, {"description": 'An old man says: "' + speech + '"', "use_current_images": True},
         state={"user_input_original": "Сделай видео, пусть он скажет: " + speech},
         dictated_speech=lambda said, raw, ctx: speech, is_talking_head=lambda ctx, d, s="", **k: True,
         eye_drift=lambda s, c, **k: 0.0, look_drift=lambda s, c: -1.0 if c == pulled else 0.4,
         match_look=lambda s, c: pulled, framing_kept=lambda s, c: framed,
         join_continuation=lambda a, b, **k: joined.append(b) or str(tmp_path / "joined.mp4"),
         _adopt_output=lambda p, *a: joined.append(p) or p,
         ctx_kw={"deps": types.SimpleNamespace(_current_image_paths=lambda ctx, st: [str(still)]),
                 "vision_model": lambda ctx, **kw: "NONE"})
    assert (pulled in joined) == framed, joined

def test_a_line_cut_off_by_the_clips_end_is_not_heard():
    """Runs 17/18: «…заготовок переполнена» ended «перепол…» at the clip's end (2026-10-20)."""
    line = "Ёмкость для хранения заготовок переполнена."
    assert not V.heard_lines([line], "Ёмкость для хранения заготовок перепол.")
    assert V.heard_lines([line], "Емкость для хранения заготовок переполнена.")
    assert V.SPEECH_WORDS_PER_PART * 1 / 2.3 + 1.0 + 1.3 <= V.frames_to_seconds(V.VIDEO_MAX_FRAMES_AUTO)


def test_a_talking_heads_part_one_that_did_not_say_its_lines_is_taken_again(tmp_path):
    speech = " ".join(f"Слушай внимательно, это правило номер {i}, его нельзя забывать." for i in range(8))
    still = tmp_path / "s.jpg"
    from PIL import Image
    Image.new("RGB", (64, 48)).save(still)
    heard = iter([False, True] + [True] * 20)
    calls = _run(tmp_path, {"description": 'An old man says: "' + speech + '"', "use_current_images": True},
                 state={"user_input_original": "Сделай видео, пусть он скажет: " + speech},
                 dictated_speech=lambda said, raw, ctx: speech, is_talking_head=lambda ctx, d, s="", **k: True,
                 eye_drift=lambda s, c, **k: 0.0,
                 ctx_kw={"deps": types.SimpleNamespace(_current_image_paths=lambda ctx, st: [str(still)]),
                         "vision_model": lambda ctx, **kw: "NONE",
                         "speech_ok": lambda ctx, clip, lines: next(heard)})
    n_parts = len(V.fit_speech([f'An old man says: "{speech}"']))
    assert len(calls) == n_parts + 1, (len(calls), n_parts)      # part 1 twice, the rest once


def test_dictated_speech_is_copied_only_when_the_user_wrote_it():
    said = "Анимируй голову. Пусть он скажет: Ах, вот и ты. Хорошо. Слушай, просто будь с рюкзачком, хорошо?"
    ctx = types.SimpleNamespace(text_model=lambda *a, **k: "Ах, вот и ты. Хорошо. Слушай, просто будь с рюкзачком, хорошо?")
    assert V.dictated_speech(said, "He talks in sync with the provided text.", ctx).startswith("Ах, вот и ты")
    assert V.dictated_speech(said, 'He says "Ах, вот и ты. Хорошо. Слушай, просто будь с рюкзачком, хорошо?"', ctx) == ""
    invented = types.SimpleNamespace(text_model=lambda *a, **k: "Привет, дружище, как твои дела сегодня?")
    assert V.dictated_speech(said, "He talks.", invented) == ""            # not in the user's message
    assert V.dictated_speech("Оживи", "He talks.", ctx) == ""


def test_a_part_without_speech_joins_the_next_when_voices_are_given():
    """The first part «looks seriously at the camera» had the voice reference and no line: it
    mumbled (live 2026-10-19)."""
    parts = ["He looks seriously at the camera.", 'He says: "Ах, вот и ты."', 'He goes on: "Хорошо."']
    out = V.merge_silent_parts(parts)
    assert out == ['He looks seriously at the camera. He says: "Ах, вот и ты."', 'He goes on: "Хорошо."']
    assert V.merge_silent_parts(['He says: "A."', "He nods."]) == ['He says: "A." He nods.']
    assert V.merge_silent_parts(["He waves.", "He sits."]) == ["He waves. He sits."]
    # each part carries the scene's look line: merged, it was said three times (2026-10-20)
    look = "The lighting remains dim."
    out = V.merge_silent_parts([f"He looks up. {look}", f"He blinks. {look}", f'He says: "Ах." {look}'])
    assert out == [f'He looks up. He blinks. He says: "Ах." {look}'], out


def test_a_part_never_carries_more_speech_than_one_clip_says():
    """Two quotes of ~22 words rode one 10 s part and the voice began mid-speech (live 2026-10-19)."""
    a = " ".join(f"слово{i}" for i in range(16))
    b = " ".join(f"фраза{i}" for i in range(16))
    part = f'He frowns and says: "{a}." He pauses, then goes on: "{b}."'
    out = V.fit_speech([part, "He waves."])
    assert len(out) == 3 and out[-1] == "He waves."
    assert all(V._spoken_words(p) <= V.SPEECH_WORDS_PER_PART for p in out)
    assert a in out[0] and b in out[1]
    assert V.fit_speech(["He waves."]) == ["He waves."]


def test_a_word_spoken_with_a_wrong_middle_fails_the_speech_check():
    """«системы» came out «систешимы» and passed the 0.6 share check (live 2026-10-19)."""
    line = "Мы не можем сделать новую, а процесс слишком сложен для обходной системы."
    ok = "Мы не можем сделать новую, а процесс слишком сложен для обходной системы."
    bad = "Мы не можем сделать новую, а процесс слишком сложен для обходной систешимы."
    assert V.heard_lines([line], ok)
    assert not V.heard_lines([line], bad)
    assert V.heard_lines(["Это очень хорошее место."], "Это очень хорошо место.")     # another ending is not a garble


def test_yo_is_restored_in_spoken_lines_but_not_where_the_word_is_real_with_e():
    got = V._marks_only("Емкость переполнена, все спят", "ёмкость переполнена, всё спят")
    assert got.startswith("Ёмкость")
    assert "все спят" in got


def test_continuation_prefix_does_not_count_as_a_camera_request():
    """The template says the camera «keeps moving»; judged with it no part got the static lock
    and the camera drove down the corridor (owner, 2026-10-19)."""
    seen = []
    ctx = types.SimpleNamespace(yes_no=lambda *a, **k: seen.append(a[1]) or False)
    out = V._lock_framing_unless_requested(V.CONTINUE_PREFIX + 'He says: "Привет."', ctx)
    assert out.endswith(V._FRAMING_LOCK_SUFFIX) and seen == ['He says: "Привет."']


def test_garble_with_a_changed_first_vowel_is_caught_too():
    """Live 2026-10-19 second take: «системы» became «сестешимы»."""
    line = "для обходной системы."
    assert not V.heard_lines([line], "для обходной сестешимы.")
    assert V.heard_lines([line], "для обходной системы.")


def test_a_word_heard_garbled_is_spoken_without_a_stress_mark_next_time():
    V.UNSTRESSED.clear()
    assert not V.heard_lines(["для обходной системы."], "для обходной систешимы.")
    assert "системы" in V.UNSTRESSED
    got = V._marks_only("для системы", "для систе́мы")
    assert "́" not in got
    V.UNSTRESSED.clear()
    assert "́" in V._marks_only("для системы", "для систе́мы")


def test_talking_head_mode_needs_a_spoken_line_and_the_models_yes():
    yes = types.SimpleNamespace(yes_no=lambda *a, **k: True)
    no = types.SimpleNamespace(yes_no=lambda *a, **k: False)
    said = 'He frowns and says: "Ах, вот и ты."'
    assert V.is_talking_head(yes, said) and not V.is_talking_head(no, said)
    assert not V.is_talking_head(yes, "He walks away down the corridor.")      # nothing spoken
    # the user dictated the words: the agent's 'single quotes' on an approved plan's rerun
    # found no quoted line, and six parts went on from each other instead of the still
    assert V.is_talking_head(yes, "He looks at the camera and says: 'Ах, вот и ты.'", spoken=True)


def test_the_talking_head_button_decides_without_asking_the_model():
    boom = types.SimpleNamespace(yes_no=lambda *a, **k: (_ for _ in ()).throw(AssertionError("asked")))
    assert V.is_talking_head(boom, "He waits.", "Talking head video: Ах, вот и ты")
    assert not V.is_talking_head(types.SimpleNamespace(yes_no=lambda *a, **k: False), 'He says: "Ах."',
                                 "video with several characters: they talk")


def test_a_light_under_the_retake_bar_is_still_pulled_back():
    """Run 19's part 2 sat at 40/50 of the still's 29/42, under the retake bar (2026-10-20)."""
    def fake(light):
        return types.SimpleNamespace(look_drift=lambda s, c: light, framing_kept=lambda s, c: True,
                                     match_look=lambda s, c: "pulled.mp4", LOOK_MAX_SHIFT=V.LOOK_MAX_SHIFT,
                                     LOOK_PULL_SHIFT=V.LOOK_PULL_SHIFT)
    assert H._pull_light(fake(-0.01), "s.jpg", "take.mp4", 2) == "pulled.mp4"
    assert H._pull_light(fake(-0.2), "s.jpg", "take.mp4", 2) == "take.mp4"


def test_words_nobody_wrote_fail_the_speech_check():
    """Run 19: «…и забрать банки. Васмуре басмутэс амут» -- H3 filled the rest of the part (2026-10-20)."""
    line = "Тебе нужно освободить нам место в холодильнике и забрать банки."
    assert not V.heard_lines([line], "Тебе нужно освободить нам место в холодильнике и забрать банки! Васмуре басмутэс амут")
    assert V.heard_lines([line], "Тебе нужно освободить нам место в холодильнике и забрать банки!")
    assert V.heard_lines(["У нас есть примерно четыре-пять недель."], "У нас есть примерно 4-5 недель.")
