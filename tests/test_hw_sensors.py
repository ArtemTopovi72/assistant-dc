"""The admin panel's hardware block: GPU load, memory, power and temperatures (core, hot spot, memory),
the CPU's load, temperature and power -- a device per head line, a heat mark on every temperature, and
an honest note when the CPU cannot be read (owner, 2026-10-21)."""
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
    out = hw_sensors.lines(_v(gpu_name="NVIDIA GeForce RTX 3090", gpu_temp=39.0, gpu_hotspot=88.2,
                              gpu_mem_temp=106.0, gpu_power=28.2, gpu_load=38.0, vram_used_mb=21635.0,
                              vram_total_mb=24576.0, cpu_name="Intel Core i7-14700KF", cpu_temp=62.0,
                              cpu_power=95.4, cpu_load=8.0))
    assert out == ["🎮 <b>RTX 3090</b>  ▰▰▰▰▱▱▱▱▱▱ 38%",
                   "      💾 21.1 / 24.0 ГБ · ⚡ 28 Вт",
                   "      🌡 ядро 🟢 39°",
                   "      🌡 хотспот 🟡 88°",
                   "      🌡 память 🔴 106°",
                   "🧠 <b>i7-14700KF</b>  ▰▱▱▱▱▱▱▱▱▱ 8%",
                   "      ⚡ 95 Вт",
                   "      🌡 ядро 🟢 62°"]


def test_a_cpu_read_without_the_driver_says_what_is_missing():
    out = hw_sensors.lines(_v(gpu_name="RTX 3090", vram_used_mb=1.0, vram_total_mb=2.0,
                              cpu_name="Intel Core i7-14700KF", cpu_load=7.0))
    assert "нужен сервис датчиков" in out[-1] and "0 Вт" not in out[-1]


def test_nothing_read_is_no_line():
    assert hw_sensors.lines(_v()) == []
