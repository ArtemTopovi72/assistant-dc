"""A ComfyUI counter name gets a human name like any other machine name
(live 10-03: «ideogram_00115_.png» next to «Картинка — Дед.png»)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
import nice_names as N


def test_comfy_counter_is_renamed():
    assert N.display_name("x/ideogram_00115_.png", "нарисуй деда").startswith("Картинка")
    assert N.display_name("x/Мой кот.png", "нарисуй деда") == "Мой кот.png"


if __name__ == "__main__":
    test_comfy_counter_is_renamed()
    print("ok")
