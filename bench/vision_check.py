"""One vision self-test against a given served model id: python bench/vision_check.py MODEL_ID"""
import sys, types
sys.stdout.reconfigure(encoding="utf-8")
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vision_selftest as v
ok = v._run_once(types.SimpleNamespace(model_name=sys.argv[1]))
print("VISION", sys.argv[1], ok, repr(v.last().get("answer", ""))[:200])
