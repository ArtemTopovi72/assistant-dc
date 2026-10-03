"""Animate-photo presets: the preset table itself, and the
[animate]/"animate this photo:" button-tool hard-force added alongside it
(same gap pattern as style_presets -- see graph_personality._BUTTON_TOOL's
comment)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import animate_presets as AP
import graph_personality as P


def test_every_preset_has_a_prompt_and_labels():
    assert len(AP.ANIMATE_PRESET_ORDER) >= 5
    for key in AP.ANIMATE_PRESET_ORDER:
        assert key in AP.ANIMATE_PRESETS
        prompt = AP.preset_prompt(key)
        assert prompt and len(prompt) > 20
        assert "photo" in prompt
        assert AP.preset_label(key, "ru")
        assert AP.preset_label(key, "en")


def test_unknown_preset_key_returns_empty():
    assert AP.preset_prompt("nonexistent") == ""
    assert AP.preset_label("nonexistent", "ru") == "nonexistent"


def test_animate_preset_button_forces_generate_video():
    offered = [{"function": {"name": "generate_video"}},
               {"function": {"name": "redraw_image"}}]
    text = ("[animate] call generate_video with description=\"slow cinematic "
            "push-in on the subject\" using the current image.")
    assert P._forced_button(text, offered, set()) == "generate_video"


def test_animate_custom_prefix_forces_generate_video():
    offered = [{"function": {"name": "generate_video"}},
               {"function": {"name": "redraw_image"}}]
    text = "animate this photo: make the subject wave at the camera"
    assert P._forced_button(text, offered, set()) == "generate_video"


if __name__ == "__main__":
    test_every_preset_has_a_prompt_and_labels()
    test_unknown_preset_key_returns_empty()
    test_animate_preset_button_forces_generate_video()
    test_animate_custom_prefix_forces_generate_video()
    print("PASS 4  FAIL 0")
