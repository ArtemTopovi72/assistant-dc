"""Letters, reversals and word counts are computed, not guessed by the model."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _ask_stub  # noqa: F401  the model's read, stubbed
import graph_compose as g

assert "7 times" in g._word_count_note("сколько букв о в слове 'обороноспособность'?")
assert "7 times" in g._word_count_note("сколько букв о в слове обороноспособность?")
assert "еинавориммаргорп" in g._word_count_note("напиши слово 'программирование' задом наперёд")
assert "9 words" in g._word_count_note("сколько слов в предложении: 'Мама мыла раму, а папа читал газету у окна'?")
assert "3 times" in g._word_count_note("how many r's in 'strawberry'?")
assert "120 times" in g._word_count_note(" ".join(["купил молоко."] * 120) + " Сколько раз я написал слово молоко?")
assert g._word_count_note("сколько букв в алфавите?") == ""
assert g._word_count_note("переведи «hello» на русский") == ""
assert "абрикос, вишня, груша, ежевика, ёжевика, яблоко" in g._word_count_note(
    "отсортируй по алфавиту: яблоко, груша, абрикос, ёжевика, вишня, ежевика")
# Pediatric dose: 14 kg -> 140-210 mg paracetamol (live answer said 100-200)
assert "paracetamol 140-210 mg" in g._word_count_note("вес 14 кг, сколько парацетамола дать?")
assert g._word_count_note("как дела?") == ""
print("PASS text ops")
assert "15% of 2340 = 351" in g._word_count_note("сколько будет 15% от 2340 плюс НДС 20%?")
import graph_personality as gp
print("PASS percent")
from graph_compose import no_history_note
assert no_history_note("о каком городе мы говорили?", [])
assert not no_history_note("о каком городе мы говорили?", [{"role": "user", "content": "Казань"}])
assert not no_history_note("какая столица Канады?", [])
print("PASS no invented past after /clear")
