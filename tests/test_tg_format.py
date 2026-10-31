import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "bot"))
from tg_markup import _format_telegram_text


def test_queue_status_keeps_count_on_its_own_line():
    old = "⚙️ <b>Жду своей очереди к модели (впереди: 1)…</b>\n⏳ 0:09"
    assert _format_telegram_text(old) == "⚙️ <b>Очередь к модели</b>\n👥 Впереди: 1\n⏳ 0:09"


def test_general_text_normalizes_blank_lines_but_preserves_code():
    assert _format_telegram_text("Заголовок\n\n\nТекст") == "Заголовок\n\nТекст"
    code = "<pre>строка 1\n\n\nстрока 4</pre>"
    assert _format_telegram_text(code) == code
