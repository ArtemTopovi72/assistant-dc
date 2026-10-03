"""🧽 Remove object: under a picture and in the Draw menu, both ask WHAT to remove,
the armed prefix keeps the Draw keyboard, and the agent gets a narrow tool list."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")
import tg_bot, tg_strings, tg_keyboards, graph_fastpath


def test_under_picture():
    verbs = [b["callback_data"].split(":")[0] for row in tg_bot._image_kb("ru")["inline_keyboard"] for b in row]
    assert "remove_object" in verbs


def test_draw_menu_button_and_prefix():
    labels = [b["text"] if isinstance(b, dict) else b for row in tg_keyboards._draw_kb("ru")["keyboard"] for b in row]
    assert tg_strings._BTN["remove_obj"]["ru"] in labels
    assert tg_bot._PROMPT_KB["remove_obj"] == "remove from the image: "
    assert tg_bot.TelegramBot._PREFIX_MENU["remove from the image: "] == "draw"


def test_narrow_tools():
    hit = [t for p, t in graph_fastpath._INTENT_TOOL_MAP if p == "remove from the image:"]
    assert hit and "inpaint_image" in hit[0] and "generate_image" not in hit[0]


def test_strings_localised():
    assert "Что убрать" in tg_strings._MSG["describe_remove"]["ru"]


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
