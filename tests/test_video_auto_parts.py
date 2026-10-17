"""A script the bot wrote itself is one or two episodes: 5-8 parts took ~25 min to render
and the longer it ran the more nonsense it held (owner, 2026-10-15). A simple action stays
one part; a length the user named is still followed."""
import os
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
        video_engine=Over(V, generate_video=fake_gen, engine_available=lambda ctx: (True, ""),
                          herrgott_on=lambda: False, motion_context_on=lambda: False,
                          mark_inserts=lambda ctx, ps: ps, validate_script=lambda ctx, ps: ps,
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
    assert len(_run(tmp_path, {"description": LONG})) == V.AUTO_MAX_PARTS == 2


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
