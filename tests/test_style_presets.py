"""Style presets: the preset table itself, and the [style]/[style_preset]
button-tool hard-force added alongside it (previously a gap -- see
graph_personality._BUTTON_TOOL's comment)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import style_presets as SP
import graph_personality as P


def test_every_preset_has_a_prompt_and_labels():
    assert len(SP.STYLE_PRESET_ORDER) >= 5
    for key in SP.STYLE_PRESET_ORDER:
        assert key in SP.STYLE_PRESETS
        prompt = SP.preset_prompt(key)
        assert prompt and len(prompt) > 20
        assert "pose" in prompt and "setting" in prompt
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


if __name__ == "__main__":
    test_every_preset_has_a_prompt_and_labels()
    test_unknown_preset_key_returns_empty()
    test_style_preset_button_forces_redraw_image()
    test_style_reference_button_forces_transfer_image()
    test_style_preset_prefix_does_not_collide_with_style_prefix()
    print("PASS 5  FAIL 0")


def test_cyberpunk_adapts_the_outfit_but_keeps_the_person():
    """Owner 10-03: the cyberpunk button must turn the current clothes into
    their cyberpunk version, not just tint the photo."""
    import style_presets as SP
    p = SP.preset_prompt("cyberpunk")
    assert "Re-imagine the clothes" in p and "same kind of garment" in p
    assert "keep the exact same pose" in p.lower() and "face and identity" in p
    assert "outfit and background" not in p      # the old keep-the-outfit clause
