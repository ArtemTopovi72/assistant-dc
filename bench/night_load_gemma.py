"""Load the house chat model the way the app does (full context, not a JIT 8192 copy)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, lmstudio
print(lmstudio.ensure_exclusive("http://127.0.0.1:1234", config.MODEL_NAME), flush=True)
