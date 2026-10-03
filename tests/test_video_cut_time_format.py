import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm, video

_reply = ("integrated_multimodal_description: [Shot 1] a cat. [Shot 2] At 04.00, it jumps. "
          "[Shot 3] At 00:05.500, it lands.\n\noverall_soundscape: beeps\n\nnon_diegetic_music: N/A")
llm.call_llm_simple = lambda *a, **k: _reply
out = video.to_context_ir(None, "кот", mode="t2va", seconds=6)
assert "At 00:04.000," in out and "At 00:05.500," in out, out
print("PASS cut times in the guide's format")
