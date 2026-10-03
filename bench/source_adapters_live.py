"""Do the source adapters still work against the LIVE services?

Split out of tests/test_source_adapters.py, which keeps the parsing checks
against captured payloads. The live call belongs here: tests/ must be
deterministic, and this one went red on a day when arXiv simply did not answer
— a network outage reported as a code failure, which is the opposite of useful.

Here an outage IS the finding: run it when a fetch looks broken in production.

Run: venv/Scripts/python.exe bench/source_adapters_live.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import source_adapters as S  # noqa: E402

CASES = [
    ("arxiv", "https://arxiv.org/abs/1706.03762", S.fetch_arxiv),
    ("wikipedia", "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
     S.fetch_wikipedia),
]

bad = 0
for name, url, fn in CASES:
    try:
        out = fn(url)
    except Exception as exc:
        out, err = None, "%s: %s" % (type(exc).__name__, exc)
    else:
        err = ""
    if out and out.get("text"):
        print("ok   %-10s %s  (%d chars via %s)"
              % (name, out.get("title", "")[:48], len(out["text"]), out.get("via")))
    else:
        bad += 1
        print("FAIL %-10s returned nothing  %s" % (name, err))

print()
print("%d/%d adapters answered" % (len(CASES) - bad, len(CASES)))
sys.exit(1 if bad else 0)
