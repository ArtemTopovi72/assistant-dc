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


def _run(tmp_path, args, condensed=""):
    calls = []

    def fake_gen(ctx, description, **kw):
        out = tmp_path / f"part{len(calls) + 1}.mp4"
        out.write_bytes(b"v")
        calls.append(description)
        return {"path": str(out), "status": "success", "seconds": 5.2, "width": 768, "height": 1344}

    ctx = _Ctx(
        video_engine=Over(V, AUTO_MAX_PARTS=2, generate_video=fake_gen, engine_available=lambda ctx: (True, ""),
                          herrgott_on=lambda: False, motion_context_on=lambda: False,
                          mark_inserts=lambda ctx, ps: ps, validate_script=lambda ctx, ps, **k: ps,
                          condense_script=lambda *a, **k: condensed,
                          join_continuation=lambda a, b: str(tmp_path / "joined.mp4"),
                          _adopt_output=lambda p, *a: p, probe=lambda p: {"seconds": 10.4}),
        clip_tools=Over(tg_continue, seed_frame=lambda src, out: open(out, "wb").write(b"j") > 0,
                        cut_tail=lambda src, out: open(out, "wb").write(b"t") > 0,
                        first_frame=lambda src, out: False))
    H._handle_generate_video(ctx, {}, {"use_current_images": False, **args})
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


def test_a_part_never_carries_more_speech_than_one_clip_says():
    """Two quotes of ~22 words rode one 10 s part and the voice began mid-speech (live 2026-10-19)."""
    a = " ".join(f"слово{i}" for i in range(22))
    b = " ".join(f"фраза{i}" for i in range(22))
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
