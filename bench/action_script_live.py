"""Live: video.action_script on agent scripts that only described the frame (2026-10-13).
Prints each rewrite and how split_script cuts it; needs LM Studio up."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, d) for d in ("core", "agent", "media", "imaging", "voice")]
import types  # noqa: E402

import video  # noqa: E402

SCRIPTS = [
    "A cinematic battle scene of warrior cats. In the foreground, three epic cats wearing capes labeled "
    "'Claude Sonnet', 'Claude Opus', and 'Claude 3.5' stand ready, their fur blowing in a dramatic wind. "
    "A black flag with an orange starburst and 'Claude' text waves intensely. The camera pans across the "
    "army. Sound: thunder, wind.",
    "A cinematic shot of the warrior cats army. The camera slowly pans across the front line, showing the "
    "intense eyes and flowing capes labeled 'Claude Sonnet', 'Claude Opus', and 'Claude 3.5'. In the "
    "background, the black 'Claude' flag ripples in a heavy wind, and the 'RDAP' flag flutters violently. "
    "Dust and embers swirl through the air as the cats let out low, guttural growls, preparing for the "
    "charge. Sound: deep cinematic percussion, howling wind.",
    "A cozy cafe. A waiter in a white apron holds a tray with three cups of coffee. Sunlight falls "
    "through the window. Sound: cafe chatter.",
]
ctx = types.SimpleNamespace(model_name="house", no_think=True, is_cancelled=lambda: False)
for s in SCRIPTS:
    for run in range(2):
        out = video.action_script(s, ctx)
        parts = video.split_script(out) if out else []
        print(f"--- run {run}: {len(parts)} parts\n{out}\n")
