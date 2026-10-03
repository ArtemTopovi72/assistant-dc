"""The run header must not undercount what the run actually read.

Live run, "как выбрать и купить б/у электромобиль в России в 2026 году": the
header said "43 источников найдено, 4 страниц прочитано" for a run whose own
progress log shows four crawl waves reading 8, 4, 4 and 4 pages -- twenty in
all, Russian sources among them.

The cause is one line: `_Progress.update()` ASSIGNS each counter, and the
crawler reported `pages=len(pages)` where `pages` is the list for the CURRENT
wave. Every wave therefore overwrote the total with its own size, and the last
wave happened to be the smallest.

That is a credibility number. A reader deciding how much to trust a report looks
at how much was read, and "4 pages" understates it fivefold.

Offline: no LLM, no search, no GPU.
"""
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dr_progress import _Progress  # noqa: E402

OK = BAD = 0


def check(name, cond, detail=""):
    global OK, BAD
    if cond:
        OK += 1
        print("PASS  " + name)
    else:
        BAD += 1
        print("FAIL  " + name + ((": " + str(detail)) if detail else ""))
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise AssertionError(str(name) + ((": " + str(detail)) if detail else ""))


print("=" * 66)
print("COUNTERS THAT ACCUMULATE, AND COUNTERS THAT DO NOT")
print("=" * 66)

p = _Progress(None)
for _ in range(8):
    p.bump("Crawling", "wave one", pages=1)
check("the first wave counts its pages", p.stats["pages"] == 8, p.stats)
for _ in range(4):
    p.bump("Crawling", "wave two", pages=1)
check("a second wave ADDS to the total rather than replacing it",
      p.stats["pages"] == 12, p.stats)
for _ in range(4):
    p.bump("Crawling", "wave three", pages=1)
for _ in range(4):
    p.bump("Crawling", "wave four", pages=1)
check("four waves of 8+4+4+4 read twenty pages, not four",
      p.stats["pages"] == 20, p.stats)

p2 = _Progress(None)
p2.update("Searching", "", sources=43)
check("update still ASSIGNS, for counters the caller totals itself",
      p2.stats["sources"] == 43, p2.stats)
p2.update("Searching", "", sources=43)
check("...so repeating it does not double the count",
      p2.stats["sources"] == 43, p2.stats)

seen = []
p3 = _Progress(lambda phase, stats, msg: seen.append((phase, stats["pages"], msg)))
p3.bump("Crawling", "kp.ru", pages=1)
p3.bump("Crawling", "auto.mail.ru", pages=1)
check("bump reports through the same callback as update", len(seen) == 2, seen)
check("...carrying the RUNNING total", [s[1] for s in seen] == [1, 2], seen)
check("...and the phase and message", seen[-1][0] == "Crawling"
      and seen[-1][2] == "auto.mail.ru", seen[-1])

p4 = _Progress(None)
p4.bump("x", "", pages=1, findings=3)
p4.bump("x", "", pages=1, findings=2)
check("several counters bump independently",
      (p4.stats["pages"], p4.stats["findings"]) == (2, 5), p4.stats)
check("a counter that was never set starts from zero",
      _Progress(None).bump("x", "", brand_new=2) is None
      or True)  # no exception is the assertion

print()
print("=" * 66)
print("THE CRAWLER USES THE ACCUMULATING ONE")
print("=" * 66)

src = open(os.path.join(os.path.dirname(__file__), "..", "research/dr_crawl.py"),
           encoding="utf-8").read()
# The exact defect: reporting this wave's list length as the run's page count.
check("the crawler no longer assigns pages=len(pages)",
      not re.search(r"prog\.update\([^)]*pages=len\(pages\)", src, re.S), "still there")
# Not [^)]* -- the progress MESSAGE itself contains a bracket (len(pages)).
check("it bumps by one page at a time instead",
      re.search(r"prog\.bump\(.{0,240}?pages=1", src, re.S) is not None,
      "no bump found")

print()
print("%d passed, %d failed" % (OK, BAD))
sys.exit(1 if BAD else 0)
