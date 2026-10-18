"""Klein boxes: thin ones are widened (BFL: a ~4% box often draws nothing), text gets margin."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import klein

b, w = klein._fit_box([500, 500, 520, 530], False)
assert w and b[2] - b[0] >= 60 and b[3] - b[1] >= 60, b
b, w = klein._fit_box([990, 10, 1000, 20], False)
assert 0 <= b[0] and b[2] <= 1000 and b[2] - b[0] == 60, b
b, w = klein._fit_box([100, 100, 300, 900], True)
assert b == [92, 92, 308, 908] and not w, b
cap = {"high_level_description": "x", "compositional_deconstruction": {"elements": [
    {"desc": "a red cup", "bbox": [500, 500, 520, 520]}]}}
assert '"desc":"Small a red cup"' in klein.caption_to_prompt(cap)
print("ok")
assert klein.with_keep_clause("Remove the van") == "Remove the van. Everything else stays identical to the source image."
assert klein.with_keep_clause("Remove the van; rest unchanged.") == "Remove the van; rest unchanged."
print("ok2")
cap = {"high_level_description": "x", "compositional_deconstruction": {"elements": [
    {"desc": "gold serif sign lettering", "text": "ХЛЕБ И СОЛЬ", "type": "text", "bbox": [80, 200, 220, 800]},
    {"desc": "chalk menu", "text": "Кофе 150", "type": "text", "bbox": [300, 200, 420, 800]}]}}
out = klein.caption_to_prompt(cap)
assert "gold serif sign lettering in capital letters. Text: \\\"ХЛЕБ И СОЛЬ\\\"" in out, out
assert "chalk menu. Text" in out
print("ok3")
