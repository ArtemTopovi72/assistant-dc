"""The app keeps exactly one ComfyUI alive: a killed server is restarted, a
listening or still-loading one is never doubled (2026-10-12: a killed server
stayed down and every video request failed «not running»)."""
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Stop(Exception):
    pass


def _run(up, alive, ticks=3):
    spec = importlib.util.spec_from_file_location(
        "launch_all_wd", os.path.join(ROOT, "scripts", "launch_all.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    spawned, n = [], [0]

    def sleep(_):
        n[0] += 1
        if n[0] > ticks:
            raise _Stop
    m.time = type("T", (), {"sleep": staticmethod(sleep)})
    m._up = lambda *a, **k: up
    m._comfy_process_alive = lambda: alive
    m._spawn_comfy_source = lambda: spawned.append(1) or object()
    try:
        m._comfy_watchdog(0)
    except _Stop:
        pass
    return len(spawned)


def test_dead_server_is_restarted():
    assert _run(up=False, alive=False) == 3


def test_listening_server_is_left_alone():
    assert _run(up=True, alive=True) == 0


def test_loading_server_is_not_doubled():
    assert _run(up=False, alive=True) == 0
