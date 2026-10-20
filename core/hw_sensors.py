"""CPU and GPU temperatures, power and memory for the admin panel.

LibreHardwareMonitor (the HardwareMonitor package) reads the GPU here without admin rights: core,
hot spot and memory junction temperatures, package power, VRAM. The CPU's temperatures and power need
its PawnIO driver and an elevated process, so scripts/sensors_service.py runs elevated (installed by
scripts/install_sensors.ps1) and writes runtime/sensors.json; the CPU values are taken from there
while it is fresh. nvidia-smi stands in when LibreHardwareMonitor cannot load.

    read() -> {"gpu_name", "gpu_temp", "gpu_hotspot", "gpu_mem_temp", "gpu_power", "gpu_load",
               "vram_used_mb", "vram_total_mb", "cpu_name", "cpu_temp", "cpu_power", "cpu_load"}
Any value it cannot read is None; it never raises.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time

logger = logging.getLogger("assistant.hw_sensors")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVICE_FILE = os.path.join(ROOT, "runtime", "sensors.json")
SERVICE_FRESH_S = 15.0
CACHE_S = 3.0

_lock = threading.Lock()
_state = {"computer": None, "failed": False, "t": 0.0, "v": None}


def _computer():
    if _state["computer"] is None and not _state["failed"]:
        try:
            from HardwareMonitor.Hardware import Computer
            c = Computer()
            c.IsCpuEnabled = True
            c.IsGpuEnabled = True
            c.Open()
            _state["computer"] = c
        except Exception:
            _state["failed"] = True
            logger.warning("LibreHardwareMonitor did not load -- nvidia-smi only", exc_info=True)
    return _state["computer"]


def _blank() -> dict:
    return dict.fromkeys(("gpu_name", "gpu_temp", "gpu_hotspot", "gpu_mem_temp", "gpu_power", "gpu_load",
                          "vram_used_mb", "vram_total_mb", "cpu_name", "cpu_temp", "cpu_power", "cpu_load"))


def read_lhm() -> dict:
    """What LibreHardwareMonitor sees from this process."""
    v = _blank()
    c = _computer()
    if c is None:
        return v
    for hw in c.Hardware:
        hw.Update()
        kind = str(hw.HardwareType)
        sensors = {(str(s.SensorType), str(s.Name)): s.Value for s in hw.Sensors}
        get = lambda t, n: (float(sensors[(t, n)]) if sensors.get((t, n)) is not None else None)
        if kind.startswith("Gpu"):
            v.update(gpu_name=str(hw.Name), gpu_temp=get("Temperature", "GPU Core"),
                     gpu_hotspot=get("Temperature", "GPU Hot Spot"),
                     gpu_mem_temp=get("Temperature", "GPU Memory Junction"),
                     gpu_power=get("Power", "GPU Package"), gpu_load=get("Load", "GPU Core"),
                     vram_used_mb=get("SmallData", "GPU Memory Used"),
                     vram_total_mb=get("SmallData", "GPU Memory Total"))
        elif kind == "Cpu":
            temp, power = get("Temperature", "CPU Package"), get("Power", "CPU Package")
            # without the PawnIO driver the package power reads 0.0, not None: no reading at all
            v.update(cpu_name=str(hw.Name), cpu_temp=temp,
                     cpu_power=None if temp is None and not power else power,
                     cpu_load=get("Load", "CPU Total"))
    return v


def _nvidia_smi(v: dict) -> None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip().splitlines()[0]
        name, used, total, util, temp, power = [x.strip() for x in out.split(",")]
        num = lambda x: float(x) if x.replace(".", "", 1).isdigit() else None
        v.update(gpu_name=v["gpu_name"] or name, vram_used_mb=v["vram_used_mb"] or num(used),
                 vram_total_mb=v["vram_total_mb"] or num(total), gpu_load=v["gpu_load"] or num(util),
                 gpu_temp=v["gpu_temp"] or num(temp), gpu_power=v["gpu_power"] or num(power))
    except Exception:
        pass


def _service(v: dict) -> None:
    """The elevated service's CPU values, when it has written lately."""
    try:
        if time.time() - os.path.getmtime(SERVICE_FILE) > SERVICE_FRESH_S:
            return
        with open(SERVICE_FILE, encoding="utf-8") as fh:
            s = json.load(fh)
    except Exception:
        return
    for k in ("cpu_name", "cpu_temp", "cpu_power", "cpu_load", "gpu_hotspot", "gpu_mem_temp"):
        if v.get(k) is None and s.get(k) is not None:
            v[k] = s[k]


def read() -> dict:
    with _lock:
        if _state["v"] is not None and time.monotonic() - _state["t"] < CACHE_S:
            return dict(_state["v"])
        try:
            v = read_lhm()
        except Exception:
            logger.debug("sensor read failed", exc_info=True)
            v = _blank()
        if v["gpu_temp"] is None or v["vram_used_mb"] is None:
            _nvidia_smi(v)
        _service(v)
        _state.update(t=time.monotonic(), v=v)
        return dict(v)


def lines(v: dict = None) -> list:
    """«🎮 RTX 3090 · 21.1/24.0 ГБ · 38% · 39° (хотспот 50°, память 50°) · 28 Вт» and the CPU's."""
    v = read() if v is None else v
    out = []
    deg = lambda x: f"{x:.0f}°" if x is not None else "?"
    if v.get("gpu_name") or v.get("vram_used_mb") is not None:
        name = (v.get("gpu_name") or "GPU").replace("NVIDIA GeForce ", "")
        bits = [f"🎮 {name}"]
        if v.get("vram_used_mb") is not None and v.get("vram_total_mb"):
            bits.append(f"{v['vram_used_mb'] / 1024:.1f}/{v['vram_total_mb'] / 1024:.1f} ГБ")
        if v.get("gpu_load") is not None:
            bits.append(f"{v['gpu_load']:.0f}%")
        if v.get("gpu_temp") is not None:
            extra = ", ".join(f"{k} {deg(v[f])}" for k, f in (("хотспот", "gpu_hotspot"), ("память", "gpu_mem_temp"))
                              if v.get(f) is not None)
            bits.append(deg(v["gpu_temp"]) + (f" ({extra})" if extra else ""))
        if v.get("gpu_power") is not None:
            bits.append(f"{v['gpu_power']:.0f} Вт")
        out.append(" · ".join(bits))
    if v.get("cpu_name") or v.get("cpu_load") is not None:
        name = (v.get("cpu_name") or "CPU").replace("Intel Core ", "").replace("AMD Ryzen ", "Ryzen ")
        bits = [f"🧠 {name}"]
        if v.get("cpu_load") is not None:
            bits.append(f"{v['cpu_load']:.0f}%")
        if v.get("cpu_temp") is not None or v.get("cpu_power") is not None:
            if v.get("cpu_temp") is not None:
                bits.append(deg(v["cpu_temp"]))
            if v.get("cpu_power") is not None:
                bits.append(f"{v['cpu_power']:.0f} Вт")
        else:
            bits.append("t° и Вт: нужен сервис датчиков (scripts/install_sensors.ps1)")
        out.append(" · ".join(bits))
    return out
