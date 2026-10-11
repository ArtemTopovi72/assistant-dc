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


def _run(monkeypatch, instructions, *, expand=None, who="a man with a black mustache"):
    seen = {}

    @contextlib.contextmanager
    def sampler(steps, cfg, *, lightning=True, negative=""):
        seen["sampler"] = (steps, cfg, lightning, negative)
        yield

    @contextlib.contextmanager
    def lora(name, strength=1.0):
        seen["lora"] = name
        yield

    monkeypatch.setattr(I, "firered_sampler", sampler)
    monkeypatch.setattr(I, "firered_extra_lora", lora)
    monkeypatch.setattr(I, "edit_image_with_firered",
                        lambda ctx, path, instr, **kw: seen.setdefault("instr", instr) and "out.png")
    monkeypatch.setattr(IS, "_who", lambda ctx, path: who)
    calls = []
    monkeypatch.setattr(IS, "_expand", lambda ctx, words: calls.append(words) or expand)
    out = IS.restyle(None, "src.png", instructions)
    return out, seen, calls


def test_a_preset_renders_its_own_text_with_real_guidance(monkeypatch):
    out, seen, calls = _run(monkeypatch, SP.preset_prompt("anime"))
    assert out == "out.png" and not calls
    assert seen["instr"].startswith(SP.preset_prompt("anime"))
    steps, cfg, lightning, neg = seen["sampler"]
    assert steps >= 12 and cfg > 1 and lightning is False and "photo" in neg
    assert seen["lora"] is None


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
