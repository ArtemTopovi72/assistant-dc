"""A name in a pasted chat («DCbot:») must not be 'fixed' as shifted Russian (it became «ВСищеЖ»)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import keyboard_layout

pasted = "Нашёл штуку\nDCbot:\n🎬 Выбери, что должно происходить\n\nArtem:\n1. 3.2 с — Молодой человек\n\nDCbot:\n🎨 Генерирую видео…"
assert keyboard_layout.fix(pasted) == pasted
assert keyboard_layout.fix("iPhone YouTube") == "iPhone YouTube"
assert keyboard_layout.fix("ghbdtn rfr ltkf") == "привет как дела"      # real wrong-layout still fixed
print("PASS layout keeps names")
