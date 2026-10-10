"""The «you promised file work» correction fires only for a freshly touched folder.

Live 2026-10-09: an Arduino chat whose working folder held old files lost two
corrective rounds per reply — «проверим», «напишем» in plain advice matched the
file-promise words. The check exists for files the user just handed over.
"""
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
for d in ("agent", "core", "bot", "media", "voice", "imaging", "knowledge", "research", "services", "sandbox", ""):
    sys.path.insert(0, str(ROOT / d))
os.environ.setdefault("F5_TEST_RUN", "1")

import graph_personality as GP  # noqa: E402

OK = BAD = 0


def check(name, cond, detail=""):
    global OK, BAD
    if cond:
        OK += 1
        print("PASS ", name)
    else:
        BAD += 1
        print("FAIL ", name, detail)


with tempfile.TemporaryDirectory() as d:
    box = SimpleNamespace(root=Path(d))
    check("an empty folder is not fresh", not GP._sandbox_touched_recently(box))
    f = Path(d) / "DCIM.zip"
    f.write_bytes(b"x")
    check("a file dropped just now is fresh", GP._sandbox_touched_recently(box))
    old = time.time() - 3 * 3600
    os.utime(f, (old, old))
    check("a file from hours ago is not — advice in that chat is not a file promise",
          not GP._sandbox_touched_recently(box))
check("no sandbox is not fresh", not GP._sandbox_touched_recently(None))

print(f"\n{OK}/{OK + BAD} checks passed")
if __name__ == "__main__":
    sys.exit(0 if BAD == 0 else 1)
