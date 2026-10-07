"""Auto mashup: B's backing is laid bar by bar onto A's bar grid -- every B downbeat
must come out on an A downbeat (rubberband time map), keys are moved the short way."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import numpy as np
import mashup_auto as M

ok = []


def check(c, m):
    ok.append(bool(c)); print(("ok   " if c else "FAIL ") + m)


check(M.semitones((9, True), (0, False)) == 0, "A minor -> C major: relative keys, no shift")
check(M.semitones((0, False), (7, False)) == -5, "C -> G goes down 5, not up 7")
check(M.semitones((0, False), (2, False)) == 2, "C -> D up 2")
a = np.arange(0, 40, 1.7)
check(len(M.match_bars(a, np.arange(0, 40, 3.4))) > 20, "half-time B grid is split to A's bar length")
check(len(M.match_bars(a, np.arange(0, 40, 0.85))) < 30, "double-time B grid is paired")

if os.path.isfile(M.RUBBERBAND):
    SR, rng = M.SR, np.random.default_rng(0)
    bars_b = np.cumsum(np.r_[0.3, rng.uniform(1.5, 1.7, 20)])
    bars_a = np.cumsum(np.r_[0.5, rng.uniform(1.65, 1.8, 30)])
    y = np.zeros(int((bars_b[-1] + 1) * SR), np.float32)
    for t in bars_b:
        i = int(t * SR); y[i:i + 4410] = rng.uniform(-.5, .5, 4410)
    bed = M.build_bed(y, bars_b, bars_a, 0, int((bars_a[-1] + 1) * SR), tempfile.mkdtemp())
    e, off = np.convolve(np.abs(bed), np.ones(220) / 220, "same"), []
    for t in bars_a[:-1]:
        lo = int((t - .3) * SR); seg = e[lo:lo + int(.6 * SR)]
        off.append(abs((lo + np.argmax(seg > .5 * seg.max())) / SR - t))
    check(max(off) < 0.02, f"every B downbeat lands on A's (worst {max(off) * 1000:.0f} ms)")
    check(abs(len(bed) - int((bars_a[-1] + 1) * SR)) == 0, "bed is exactly the vocal's length")
else:
    print("skip: rubberband not installed")

print(f"\n{sum(ok)}/{len(ok)}")
sys.exit(0 if all(ok) else 1)
