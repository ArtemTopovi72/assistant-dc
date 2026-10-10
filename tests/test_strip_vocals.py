"""An instrumental that came out sung loses its vocal; a quiet vocal stem is left alone.

_strip_vocals used `np` without importing it, so every call died in its own
except and logged «could not strip the vocal» — the feature never ran once.
"""
import os
import sys
import tempfile
import types

os.environ.setdefault("F5_TEST_RUN", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import music  # noqa: E402

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond)
    BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")


calls = []
fake = types.ModuleType("mashup_stems")
fake.SAMPLE_RATE = 44100
fake.release_separator = lambda: None
fake.backing_of = lambda st: st["back"]
sys.modules["mashup_stems"] = fake
import subprocess  # noqa: E402
_run = subprocess.run
subprocess.run = lambda cmd, **k: (calls.append(cmd), open(cmd[-1], "wb").close())[0]
warned = []
music.logger.warning = lambda *a, **k: warned.append(a[0])

d = tempfile.mkdtemp(prefix="stripvox_")
path = os.path.join(d, "song.mp3")
open(path, "wb").close()
back = np.full((1000, 2), 0.2, dtype=np.float32)

fake.separate = lambda p: {"vocals": np.full((1000, 2), 0.3, dtype=np.float32), "back": back}
music._strip_vocals(path)
check("a loud vocal stem on an instrumental is removed (ffmpeg re-encodes the backing)", len(calls) == 1, calls)
check("...without the «could not strip» failure", not warned, warned)

calls.clear()
warned.clear()
fake.separate = lambda p: {"vocals": np.full((1000, 2), 0.001, dtype=np.float32), "back": back}
music._strip_vocals(path)
check("a near-silent vocal stem leaves the take as rendered", not calls and not warned, (calls, warned))

subprocess.run = _run
print(f"\n{OK}/{OK + BAD} checks passed")
if __name__ == "__main__":
    sys.exit(0 if BAD == 0 else 1)
