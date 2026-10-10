"""Load a model exactly as the app does (REST: ctx 40960, parallel 1, batch 2048), unloading the rest."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
for d in ("", "core", "agent"):
    sys.path.insert(0, os.path.join(ROOT, d))
import lmstudio  # noqa: E402

ok, msg = lmstudio.load_model_exclusive("http://127.0.0.1:1234", sys.argv[1], context_length=40960)
print(ok, msg)
sys.exit(0 if ok else 1)
