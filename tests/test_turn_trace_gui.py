"""A GUI turn that ends in an error is traced as failed, not done.
Run: venv/Scripts/python.exe tests/test_turn_trace_gui.py
"""
import json
import os
import sys
import tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "agent"), os.path.join(ROOT, "core"), os.path.join(ROOT, "gui")]
os.environ["TURNS_DIR"] = tempfile.mkdtemp()
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtCore import QCoreApplication  # noqa: E402
app = QCoreApplication([])
import gui_workers  # noqa: E402


class Ctx:
    def set_stage(self, s): pass


w = gui_workers.RequestWorker(Ctx(), None, {"messages": []}, text="")   # empty -> failed
w.run()
line = open(os.path.join(os.environ["TURNS_DIR"], os.listdir(os.environ["TURNS_DIR"])[0]),
            encoding="utf-8").read().splitlines()[-1]
out = json.loads(line)["meta"]["outcome"]
print(("PASS" if out.startswith("failed") else "FAIL") + "  outcome = " + out)
sys.exit(0 if out.startswith("failed") else 1)
