"""The bot's per-task ctx built over the GUI's _ScopedCtx view must own its
cancel token and fields. Otherwise Cancel says «Cancelling…» while the task
reads the wrapper's token and keeps going, and pinned facts / session memory
are written onto the app's shared Context."""
import os, sys, threading, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gui_common import _ScopedCtx
import tg_bot

real = types.SimpleNamespace(cancel_event=threading.Event(), session_memory=None, pinned_facts=["REAL"])
ev = threading.Event()
t = tg_bot._scoped_ctx(_ScopedCtx(real, threading.Event()), cancel_event=ev)
ev.set()
assert t.cancel_event.is_set(), "task must see its own Cancel"
assert real.pinned_facts == ["REAL"], real.pinned_facts
assert real.cancel_event is not ev and not real.cancel_event.is_set()
t2 = tg_bot._scoped_ctx(real, cancel_event=threading.Event())   # plain ctx still works
assert not t2.cancel_event.is_set() and real.pinned_facts == ["REAL"]
print("PASS scoped ctx unwraps the GUI view")
