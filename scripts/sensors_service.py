"""Elevated sensor reader: the CPU's temperature and power need LibreHardwareMonitor's PawnIO driver
and admin rights, which the app does not run with. Started at logon by the task that
scripts/install_sensors.ps1 registers; writes runtime/sensors.json every 2 s for core/hw_sensors.py."""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))
import hw_sensors  # noqa: E402

out = hw_sensors.SERVICE_FILE
os.makedirs(os.path.dirname(out), exist_ok=True)
while True:
    try:
        v = hw_sensors.read_lhm()
        tmp = out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(v, fh)
        os.replace(tmp, out)
    except Exception:
        pass
    time.sleep(2.0)
