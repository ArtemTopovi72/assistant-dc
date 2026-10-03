"""Numbered and bulleted lists are voiced as «во-первых … в-третьих», not as digits."""
import os, sys
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import audio


def test_numbered_list_becomes_ordinals():
    out = audio.clean_text("Нужно:\n1. Купить хлеб\n2. Позвонить маме\n3. Выгулять собаку")
    assert "Во первых, Купить хлеб." in out and "Во вторых, Позвонить маме." in out, out
    assert "И в заключение, Выгулять собаку" in out and "1" not in out, out


def test_bullets_and_english():
    assert "Во первых, дёшево. Во вторых, быстро" in audio.clean_text("Плюсы:\n- дёшево\n- быстро")
    assert audio.clean_text("1) open\n2) run\n3) close") == "First, open. Second, run. Finally, close"


def test_the_graph_voice_path_sees_the_list_before_digits_become_words():
    out = audio.post_process_answer("Итак:\n1. Купить хлеб\n2. Позвонить маме")
    assert "Во-первых" in out and "один" not in out, out


def test_a_single_number_is_not_a_list():
    assert audio.clean_text("Шаг 1. просто текст") == "Шаг 1. просто текст"


if __name__ == "__main__":
    test_numbered_list_becomes_ordinals(); test_bullets_and_english(); test_the_graph_voice_path_sees_the_list_before_digits_become_words(); test_a_single_number_is_not_a_list()
    print("ok")
