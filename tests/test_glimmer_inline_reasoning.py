"""Muse Glimmer via LM Studio 2.44: reasoning arrives inline in content as
`to=self<|message|> ... <|eom|>answer`. It reached a Telegram user as
"to=selfUser says "привет". Need respond in Russian…Привет!" (2026-09-24)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import utils  # noqa: E402

CASES = [
    (" to=self<|message|>привет\n\nWe need respond in Russian.<|eom|>Привет! Как дела?",
     "Привет! Как дела?"),
    (" to=self<|message|>погода?\n\nUse search.<|eom|>", ""),                 # tool turn
    (" to=self<|message|>ran out of budget while thinking", ""),              # unterminated
    ("Обычный ответ без разметки", "Обычный ответ без разметки"),
    ("Слово to=self без разметки остаётся", "Слово to=self без разметки остаётся"),
]


def test_inline_reasoning_is_removed():
    for raw, want in CASES:
        got = utils.strip_control_tokens(raw).strip()
        assert got == want, (raw, got)
        assert "Need respond" not in got and "Use search" not in got


if __name__ == "__main__":
    test_inline_reasoning_is_removed()
    print("ok")
