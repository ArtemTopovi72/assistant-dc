"""A wrong letter is a failed label whatever the similarity ratio says.

Live 2026-09-12 (journey 25): "только до пятницы" was painted as "ТОЛЬКО ДО
ПЯТНИЦЯ"; ratio 0.93 passed the 0.8 bar and the misspelled poster went out.
Reader confusables (О/0, Й/И, Latin look-alikes) stay tolerated.
"""
import os, sys
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import draw_text as D
from text_layout import TEXT_MATCH_OK

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

sim = D._similarity
check("пятниця is under the bar", sim("только до пятницы", "ТОЛЬКО ДО ПЯТНИЦЯ") < TEXT_MATCH_OK)
check("ХЛЕП for ХЛЕБ is under the bar", sim("ХЛЕБ", "ХЛЕП") < TEXT_MATCH_OK)
check("the exact word is 1.0", sim("только до пятницы", "ТОЛЬКО ДО ПЯТНИЦЫ") == 1.0)
check("a zero read for О is a reader confusable, still a match", sim("РАСПРОДАЖА", "РАСПР0ДАЖА") >= TEXT_MATCH_OK)
check("Й read as И is tolerated", sim("СЕГОДНЯ ЧАЙ", "СЕГОДНЯ ЧАИ") >= TEXT_MATCH_OK)
check("Л read as Х (OCR shape confusion) is tolerated", sim("У ОЛЬГИ", "УОХЬГИ") >= TEXT_MATCH_OK)
check("C read as G is tolerated", sim("POLICE", "POLIGE") >= TEXT_MATCH_OK)
check("three wrong letters is simply a low ratio (no special case needed)", sim("ПЯТНИЦА", "ПЯТЛИЧО") < TEXT_MATCH_OK)
check("a short word is left to the ratio", D._letter_swapped("абв", "абг") is False)
check("different lengths are left to the ratio", D._letter_swapped("пятница", "пятниц") is False)
check("a split reading joined still matches", D._match_lines("У ОЛЬГИ", ["У", "ОЛЬГИ"])[1] >= TEXT_MATCH_OK)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
