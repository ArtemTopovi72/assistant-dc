"""image_style.restyle: how a style request is rendered.

One-word styles on Lightning 8 steps at cfg 1 came back alike (anime == comic,
oil a warm filter) and a cartoon «в ультрареализм» stayed a cartoon
(owner, 2026-10-11; bench/style_ab.py). A preset renders its own spelled-out
text; free words are spelled out by the chat model first; every restyle samples
with real guidance and a negative against the look being left behind; towards a
photo the realism LoRA loads and who is in the picture rides the instruction.
"""
import contextlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import image as I
import image_style as IS
import style_presets as SP
from fakes import Over, deps_ctx


def _run(monkeypatch, instructions, *, expand=None, who="a man with a black mustache"):
    seen = {}

    @contextlib.contextmanager
    def sampler(steps, cfg, *, lightning=True, negative="", upsize=False):
        seen["sampler"] = (steps, cfg, lightning, negative)
        seen["upsize"] = upsize
        yield

    @contextlib.contextmanager
    def lora(name, strength=1.0):
        seen["lora"] = name
        yield

    calls = []
    ctx = deps_ctx(
        image=Over(I, firered_sampler=sampler, firered_extra_lora=lora,
                   edit_image_with_firered=lambda ctx, path, instr, **kw: seen.setdefault("instr", instr) and "out.png"),
        _who=lambda ctx, path: (who, False),
        _expand=lambda ctx, words: calls.append(words) or expand)
    out = IS.restyle(ctx, "src.png", instructions)
    return out, seen, calls


def test_a_preset_renders_its_own_text_with_real_guidance(monkeypatch):
    out, seen, calls = _run(monkeypatch, SP.preset_prompt("anime"))
    assert out == "out.png" and not calls
    assert seen["instr"].startswith(SP.preset_prompt("anime"))
    steps, cfg, lightning, neg = seen["sampler"]
    assert steps >= 12 and cfg > 1 and lightning is False and "photo" in neg
    assert seen["lora"] is None and seen["upsize"]


def test_anime_and_comic_are_different_instructions(monkeypatch):
    a = _run(monkeypatch, SP.preset_prompt("anime"))[1]["instr"]
    c = _run(monkeypatch, SP.preset_prompt("comic"))[1]["instr"]
    assert "anime face" in a and "Ben-Day" in c and "not anime" in c


def test_towards_a_photo_loads_the_realism_lora_and_names_who_is_there(monkeypatch):
    _, seen, _ = _run(monkeypatch, SP.preset_prompt("photo"))
    assert seen["lora"] == IS.PHOTO_LORA
    assert "a man with a black mustache" in seen["instr"]
    assert "cartoon" in seen["sampler"][3]       # the negative is the drawn look


def test_free_words_are_spelled_out_first(monkeypatch):
    _, seen, calls = _run(monkeypatch, "сделай ультрареализм",
                          expand=("photo", "Change picture 1 into a real photo."))
    assert calls and seen["lora"] == IS.PHOTO_LORA
    assert "Change picture 1 into a real photo." in seen["instr"]
    _, seen, _ = _run(monkeypatch, "в стиле гравюры",
                      expand=("drawn", "Turn picture 1 into a woodcut engraving."))
    assert seen["lora"] is None and "woodcut" in seen["instr"]


def test_a_preset_wrapped_by_the_agent_is_still_the_preset():
    wrapped = "Please: " + SP.preset_prompt("watercolor") + " Thanks."
    assert SP.preset_for(wrapped) == "watercolor"
    assert SP.preset_for("make it moody") is None


def test_a_redraw_that_is_not_a_style_is_edited_as_written(monkeypatch):
    out, seen, calls = _run(monkeypatch, "make it night time",
                            expand=("other", "Turn picture 1 into a night scene."))
    assert calls and seen["instr"] == "make it night time" and "sampler" not in seen


def test_a_small_picture_is_restyled_at_working_size_not_shrunk_back(monkeypatch):
    """A 736x495 picture restyled at 1 MP and shrunk back to 736x495 came out soft."""
    seen = {}
    ctx = deps_ctx(_enforce_output_size=lambda wf, w, h: seen.update(w=w, h=h),
                   _upload_image_to_comfy=lambda p, u: "x.png", _source_dims=lambda p: (736, 495),
                   _submit_and_poll=lambda *a, **k: "out.png", exists=lambda p: True)
    with I.firered_sampler(12, 3.0, lightning=False, upsize=True):
        I._edit_image_with_firered_once(ctx, "src.png", "x", seed=1)
    assert seen["w"] > 1000 and abs(seen["w"] / seen["h"] - 736 / 495) < 0.02, seen
    I._edit_image_with_firered_once(ctx, "src.png", "x", seed=1)
    assert (seen["w"], seen["h"]) == (736, 495)          # a plain edit keeps the source size


def test_realism_keeps_materials_props_and_the_background():
    """A white character sheet came back in a forest, her plate armour a cloth top,
    her scythes and boots gone (owner, 2026-10-12): the photo text must not invite
    fabric or a new place, and the description names materials, props, background."""
    p = SP.preset_prompt("photo")
    assert "fabric weave" not in p and "depth of field" not in p
    assert "armor stays rigid armor" in p and "white photo-studio backdrop" in p
    for word in ("material", "footwear", "weapons", "background"):
        assert word in IS._WHO_PROMPT


def test_realism_neither_undresses_nor_dresses_and_keeps_the_skin_clean(monkeypatch):
    """Motoko's skin-coloured suit was described as «bare skin on the chest, abdomen» and came
    back undressed; the next wording put two takes of three in black leather; «skin texture
    with pores» over cel shadows read as dirt on the face (owner, 2026-10-21;
    bench/photo_restyle_live.py: 4 of 4 suited, clean, the building kept)."""
    assert "collar edge" in IS._WHO_PROMPT and "never as bare skin" in IS._WHO_PROMPT
    assert "side locks" in IS._WHO_PROMPT                         # the sleek bob came back as clumps
    instr = _run(monkeypatch, SP.preset_prompt("photo"))[1]["instr"]
    assert "Nobody is undressed or dressed" in instr and "skin-tight suit" in instr
    # a describer pass that left the glove's colour out let it come back black (1 take of 22)
    assert "not named above keeps exactly its colour" in instr
    p = SP.preset_prompt("photo")
    assert "pores" not in p and "no dirt" in p and "the same place" in p


def test_a_new_scene_is_not_a_style_of_the_old_picture():
    """«A cinematic, highly detailed scene of Lenin in a cybernetic suit fighting demons in lava»
    read as a drawn style, got «keep the same composition», and the Wikipedia screenshot came back
    as itself tinted with lava (live 2026-10-21). A change of what is shown is edited as written."""
    import types
    reply = lambda *a, **k: '{"target": "drawn", "style": "Turn picture 1 into a digital painting."}'
    scene = types.SimpleNamespace(text_model=reply, yes_no=lambda q, t, *a, **k: "WHAT the picture shows" in q)
    assert IS._expand(scene, "Lenin in a cybernetic suit fights demons in lava")[0] == "other"
    style = types.SimpleNamespace(text_model=reply, yes_no=lambda *a, **k: False)
    assert IS._expand(style, "make it a digital painting")[0] == "drawn"
