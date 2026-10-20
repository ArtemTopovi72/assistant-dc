"""The reply said «Готово! швыряет шапку» over a clip where the hat stayed on (live 2026-10-18).
The finished clip is looked at once against the script; actions the frames do not show go into
the tool result, so the reply cannot claim them.

Run: venv/Scripts/python.exe tests/test_video_action_check.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import logging; logging.basicConfig(level=logging.CRITICAL)

import tool_image_handlers as H

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {str(extra)[:200]}")


seen = {}
look = types.SimpleNamespace(
    duration_seconds=lambda p: 6.0,
    sample_times=lambda d, every=5, max_frames=12: [0.0, 1.5, 3.0],
    extract_frames=lambda p, ts, d: [(t, os.path.join(str(d), f"f{t}.jpg")) for t in ts])


def ctx_for(answer):
    def vision(ctx, **kw):
        seen.update(kw)
        if isinstance(answer, Exception):
            raise answer
        return answer
    return types.SimpleNamespace(vision_model=vision, modules={"video_look": look})


script = ["He takes off his hat and says \"не ваше дело\", then throws the hat away."]
r = H._check_actions(ctx_for("the hat is never thrown"), "x.mp4", script)
check("a missing action comes back", r == "the hat is never thrown", r)
check("the script and the frames reach the vision call",
      "throws the hat" in seen["user_text"] and len(seen["image_paths"]) == 3, seen.get("user_text"))
check("NONE means all shown", H._check_actions(ctx_for("NONE"), "x.mp4", script) == "")
# reasoning first, the verdict last: the reply was told an action was missing (live 2026-10-20)
r = H._check_actions(ctx_for("The frames show his face and lips. Speech cannot be seen. **NONE**."), "x.mp4", script)
check("a NONE at the end means all shown too", r == "", r)
check("a failing model never blocks the clip", H._check_actions(ctx_for(RuntimeError("x")), "x.mp4", script) == "")
check("no script, no call", H._check_actions(ctx_for("anything"), "x.mp4", ["ok"]) == "")

print(f"\n{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
