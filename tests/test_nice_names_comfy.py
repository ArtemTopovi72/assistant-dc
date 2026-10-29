"""A ComfyUI counter name gets a human name like any other machine name
(«ideogram_00115_.png» next to «Картинка — Дед.png»)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
import nice_names as N


def test_comfy_counter_is_renamed():
    assert N.display_name("x/ideogram_00115_.png", "нарисуй деда").startswith("Картинка")
    assert N.display_name("x/Мой кот.png", "нарисуй деда") == "Мой кот.png"


if __name__ == "__main__":
    test_comfy_counter_is_renamed()
    print("ok")


def test_style_preset_payload_names_the_preset_words_not_the_call():
    from nice_names import display_name
    payload = ('[style_preset] call redraw_image with mode="redraw" '
               'instructions="Turn picture 1 into a Japanese anime illustration '
               'with clean lines" on the current image. Do not generate a new image.')
    name = display_name("x/ideogram_00115_.png", payload)
    assert "redraw_image" not in name and "mode=" not in name, name
    assert "anime" in name.lower(), name
