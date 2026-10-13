"""The model's read of agent/ask_read.py, stubbed for the suites: what each
fixture phrase asks. The phrases themselves run live in bench/compose_ask_live.py.
A suite hands `DEPS` (as a context's `deps`) or `CTX` to the code under test."""
import types

_READS = {
    "сколько букв о в слове 'обороноспособность'?": {"op": "letter_count", "letter": "о", "word": "обороноспособность"},
    "сколько букв о в слове обороноспособность?": {"op": "letter_count", "letter": "о", "word": "обороноспособность"},
    "напиши слово 'программирование' задом наперёд": {"op": "reverse", "target": "программирование"},
    "сколько слов в предложении: 'Мама мыла раму, а папа читал газету у окна'?":
        {"op": "word_count", "target": "Мама мыла раму, а папа читал газету у окна"},
    "how many r's in 'strawberry'?": {"op": "letter_count", "letter": "r", "word": "strawberry"},
    "отсортируй по алфавиту: яблоко, груша, абрикос, ёжевика, вишня, ежевика":
        {"op": "sort", "items": ["яблоко", "груша", "абрикос", "ёжевика", "вишня", "ежевика"]},
    "вес 14 кг, сколько парацетамола дать?": {"dose": True, "weight_kg": 14},
    "сколько будет 15% от 2340 плюс НДС 20%?": {"op": "percent", "percents": [{"pct": 15, "of": 2340}]},
    "о каком городе мы говорили?": {"earlier_talk": True},
    "what did i ask three messages ago?": {"earlier_talk": True},
    "инструкцию на 1200 слов": {"words_asked": 1200},
    "расскажи подробно": {"detailed": True},
    "напиши скрипт, который считает строки в csv": {"code": True},
    "fix the bug in main.py": {"code": True},
    "refactor this module": {"code": True},
}


def _read(text):
    if text.endswith("Сколько раз я написал слово молоко?"):
        return {"op": "word_occurrences", "word": "молоко"}
    return _READS.get(text.strip())


DEPS = types.SimpleNamespace(ask_read=_read)
CTX = types.SimpleNamespace(deps=DEPS)
