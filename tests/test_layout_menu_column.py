"""A menu of 8 lines planned as 8 lettering boxes (+ board) is past MAX_ELEMENTS: one multi-line box."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import draw_agent as D
items = ["Эспрессо 150", "Американо 180", "Капучино 220", "Латте 240", "Раф 260", "Какао 200", "Чай 120", "Круассан 190"]
els = [{"x": .1, "y": .05, "w": .8, "h": .9, "desc": "chalkboard"}] + [
    {"x": .2, "y": .15 + .1 * i, "w": .4, "h": .08, "desc": "menu item in chalk", "text": t} for i, t in enumerate(items)]
out = D.merge_text_column({"elements": els})["elements"]
assert len(out) == 2 and out[1]["text"].split("\n") == items, out
assert abs(out[1]["y"] - .15) < 1e-9 and abs(out[1]["y"] + out[1]["h"] - .93) < 1e-9, out[1]
assert D.merge_text_column({"elements": els[:5]})["elements"] == els[:5]
print("PASS")
