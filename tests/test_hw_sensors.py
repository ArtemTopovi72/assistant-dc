"""The admin panel's hardware lines: GPU memory, temperatures (core, hot spot, memory) and power, the
CPU's load, temperature and power -- and an honest note when the CPU cannot be read (owner,
2026-10-21)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

import hw_sensors  # noqa: E402


def _v(**kw):
    v = hw_sensors._blank()
    v.update(kw)
    return v


def test_every_reading_reaches_the_panel():
    gpu, cpu = hw_sensors.lines(_v(gpu_name="NVIDIA GeForce RTX 3090", gpu_temp=39.0, gpu_hotspot=50.2,
                                   gpu_mem_temp=50.0, gpu_power=28.2, gpu_load=38.0, vram_used_mb=21635.0,
                                   vram_total_mb=24576.0, cpu_name="Intel Core i7-14700KF", cpu_temp=62.0,
                                   cpu_power=95.4, cpu_load=8.0))
    assert gpu == "🎮 RTX 3090 · 21.1/24.0 ГБ · 38% · 39° (хотспот 50°, память 50°) · 28 Вт"
    assert cpu == "🧠 i7-14700KF · 8% · 62° · 95 Вт"


def test_a_cpu_read_without_the_driver_says_what_is_missing():
    _gpu, cpu = hw_sensors.lines(_v(gpu_name="RTX 3090", vram_used_mb=1.0, vram_total_mb=2.0,
                                    cpu_name="Intel Core i7-14700KF", cpu_load=7.0))
    assert "нужен сервис датчиков" in cpu and "0 Вт" not in cpu


def test_nothing_read_is_no_line():
    assert hw_sensors.lines(_v()) == []
