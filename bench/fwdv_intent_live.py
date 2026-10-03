"""Live check of tg_bot._fwdv_intent on the house model (needs LM Studio up).

    venv/Scripts/python bench/fwdv_intent_live.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "bot"), os.path.join(ROOT, "agent")]
sys.stdout.reconfigure(encoding="utf-8")
import tg_bot as T

CASES = [
    ("перескажи", "sum"), ("перескажи кратко", "sum"), ("о чём там", "sum"),
    ("сделай саммари", "sum"), ("summarize this", "sum"), ("tl;dr", "sum"),
    ("в двух словах суть", "sum"),
    ("расшифруй", "text"), ("дословно", "text"), ("переведи в текст", "text"),
    ("transcribe it", "text"), ("что он там говорит слово в слово?", "text"),
    ("и то, и другое", "both"), ("расшифруй и перескажи", "both"),
    ("покажи раскадровку", "board"), ("что в кадре по секундам?", "board"),
    ("нарисуй кота", ""), ("спасибо", ""), ("что думаешь?", ""),
    ("А если бы тебя добавили? что бы сказали твои друзья?", ""),
    ("ответь ему вежливо", ""), ("какой срок он назвал?", ""),
]
bad = 0
for text, want in CASES:
    got = T._fwdv_intent(text)
    ok = got == want
    bad += not ok
    print(("PASS " if ok else "FAIL ") + f"{text!r} -> {got!r} (want {want!r})")
print(f"{len(CASES) - bad}/{len(CASES)}")
sys.exit(1 if bad else 0)
