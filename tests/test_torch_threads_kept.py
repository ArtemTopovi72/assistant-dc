"""Loading silero (VAD, stress) must not leave the whole process on one CPU thread.

silero_vad and silero_stress call torch.set_num_threads(1) at import. After the first voice
message every CPU model in the app ran single-threaded: the web-search injection guard took
10-13 s in the app and 1.25 s outside it (torch threads 1 vs 20, 2026-10-25).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import torch
import utils
import audio, stress

before = torch.get_num_threads()
with utils.torch_threads_kept():
    torch.set_num_threads(1)                    # what the silero modules do on import
assert torch.get_num_threads() == before, (torch.get_num_threads(), before)

calls = []
audio._load_silero_vad_model(loader=lambda: (torch.set_num_threads(1), calls.append(1), "vad")[2], store={})
assert calls and torch.get_num_threads() == before, torch.get_num_threads()

import inspect
assert inspect.getsource(stress).count("torch_threads_kept()") >= 2, "silero_stress and ruaccent keep the thread count"
import voice_clone
assert "torch_threads_kept()" in inspect.getsource(voice_clone), "silero_vad import keeps the thread count"
print("ok silero loads keep the process thread count")
