"""Words about a picture the user pointed at are answered by looking at it.

«обсудим цитату» after pressing ❓ under a quote photo is not phrased as a
question. The press makes the picture the target (like a reply to it) and one
rule in graph.needs_relook covers both: the user's own words about a
pointed-at picture -> look again, unless they order an edit. A button's own
task (task.image_id) is an action, not words.
"""
import os, sys, inspect
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, d) for d in ("bot", "agent", "core")]
import graph as G
sys.path.insert(0, os.path.join(ROOT, "tests"))
from fakes import reads
# the model's read of the words (agent/intent.py)
_READ = reads(read={"убери надпись": {"needs_tool": True, "wants": ["inpaint_image"]}}.get)
import tg_tasks, tg_callbacks, tg_bot


class Cx:
    last_image_path = __file__          # any existing file
    intent_read = staticmethod(_READ["intent_read"])
    image_pointed_at = True


st = lambda t: {"user_input": t, "user_input_original": t}  # noqa: E731
assert G.needs_relook(Cx(), st("обсудим цитату"))
assert G.needs_relook(Cx(), st("классная"))
assert not G.needs_relook(Cx(), st("убери надпись"))
Cx.image_pointed_at = False
assert not G.needs_relook(Cx(), st("обсудим цитату")) or G.is_video_sheet(__file__)
assert "image_pointed_at = not getattr(task, \"image_id\", \"\")" in inspect.getsource(tg_tasks)
assert "pending_prefix" not in inspect.getsource(tg_callbacks.CallbackMixin._cb_ask_image)
assert "about this picture" not in inspect.getsource(tg_callbacks) + inspect.getsource(G) + inspect.getsource(tg_bot)
print("ok")
