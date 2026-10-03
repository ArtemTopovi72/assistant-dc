"""A YuE2 song cut off by the token budget is rendered again one section shorter
(the same words repeat the same cut); a whole song renders once."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import music as M

calls = []
M.yue2_cpp_available = lambda: True
M._valid_audio_file = lambda p: True
tails = []
M.run_gpu_worker = lambda ctx, py, sc, job, label, timeout, cmd=None, env=None: (
    calls.append(job["lyrics"]) or (True, tails.pop(0)))

LYR = "[verse]\nодин\n\n[chorus]\nдва\n\n[verse]\nтри\n\n[outro]\nчетыре"
tails[:] = ["[AR] plan song 0: 24576 tokens (truncated)", "[AR] plan song 0: 20000 tokens"]
M._generate_yue2(None, LYR, "pop", 1)
assert len(calls) == 2 and "четыре" in calls[0] and "четыре" not in calls[1] and "три" in calls[1], calls
print("ok a truncated song is rendered once more, one section shorter")

calls.clear(); tails[:] = ["[AR] plan song 0: 9000 tokens"]
M._generate_yue2(None, LYR, "pop", 1)
assert len(calls) == 1, calls
print("ok a whole song renders once")

calls.clear(); tails[:] = ["(truncated)", "(truncated)"]
M._generate_yue2(None, LYR, "pop", 1)
assert len(calls) == 2, calls
print("ok at most one retry")

y = M.yue2_lyrics("[intro]\nэй-эй\n\n[verse]\nстрока")
assert y == "[Intro]\n\n[Verse]\nстрока", repr(y)
print("ok [Intro] is left empty")
