"""Style presets: the preset table itself, and the [style]/[style_preset]
button-tool hard-force (see graph_personality._BUTTON_TOOL)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import style_presets as SP
import graph_personality as P
import image_style as IS


def _built(key):
    IS._who = lambda ctx, path: ("", False)
    return IS.build(None, SP.preset_prompt(key))[:2]


def test_every_preset_has_a_prompt_and_labels():
    assert len(SP.STYLE_PRESET_ORDER) >= 5
    for key in SP.STYLE_PRESET_ORDER:
        assert key in SP.STYLE_PRESETS
        prompt = SP.preset_prompt(key)
        assert prompt and len(prompt) > 20
        instr, _ = _built(key)          # the keep-clause rides every preset (image_style)
        assert "pose" in instr and "setting" in instr and "entire image" in instr
        assert SP.preset_label(key, "ru")
        assert SP.preset_label(key, "en")


def test_unknown_preset_key_returns_empty():
    assert SP.preset_prompt("nonexistent") == ""
    assert SP.preset_label("nonexistent", "ru") == "nonexistent"


def test_style_preset_button_forces_redraw_image():
    offered = [{"function": {"name": "redraw_image"}},
               {"function": {"name": "transfer_image"}}]
    text = ("[style_preset] call redraw_image with mode=\"redraw\" "
            "instructions=\"convert to anime art style\" on the current image.")
    assert P._forced_button(text, offered, set()) == "redraw_image"


def test_style_reference_button_forces_transfer_image():
    offered = [{"function": {"name": "redraw_image"}},
               {"function": {"name": "transfer_image"}}]
    text = ("[style] call transfer_image with instructions=\"Change the "
            "visual style of the target image to match the reference image\"")
    assert P._forced_button(text, offered, set()) == "transfer_image"


def test_style_preset_prefix_does_not_collide_with_style_prefix():
    # "[style_preset]" must never be caught by the shorter "[style]" entry --
    # they name different tools (transfer_image needs a second image,
    # redraw_image does not).
    offered = [{"function": {"name": "redraw_image"}},
               {"function": {"name": "transfer_image"}}]
    text = "[style_preset] call redraw_image with mode=\"redraw\""
    assert P._forced_button(text, offered, set()) == "redraw_image"


def test_cyberpunk_adapts_the_outfit_but_keeps_the_person():
    """The cyberpunk button must turn the current clothes into their cyberpunk
    version, not just tint the photo."""
    p = SP.preset_prompt("cyberpunk")
    assert "Re-imagine the clothes" in p and "same kind of garment" in p
    instr, _ = _built("cyberpunk")
    assert "same pose" in instr and "face and identity" in instr
    assert "same clothing" not in instr          # the keep-the-outfit clause would undo it
    assert "same clothing" in _built("anime")[0]


if __name__ == "__main__":
    test_every_preset_has_a_prompt_and_labels()
    test_unknown_preset_key_returns_empty()
    test_style_preset_button_forces_redraw_image()
    test_style_reference_button_forces_transfer_image()
    test_style_preset_prefix_does_not_collide_with_style_prefix()
    test_cyberpunk_adapts_the_outfit_but_keeps_the_person()
    print("PASS 6  FAIL 0")
