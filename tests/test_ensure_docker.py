"""The app starts its own Docker engine, like it starts LM Studio and ComfyUI.

A healthy but not launched Docker Desktop must not read as "Docker не запущен".
"""
import os, sys
from types import SimpleNamespace as NS
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import code_runner as R

_FLAG = 0x08000000


def _machine(spawned, up, binary=None, clock=None):
    """A context whose Docker, process spawner and clock are the test's."""
    sp = NS(Popen=lambda args, **k: spawned.append((args, k)), DEVNULL=-3,
            DETACHED_PROCESS=0x8, CREATE_NO_WINDOW=_FLAG)
    return NS(deps=NS(docker_available=up, desktop_binary=lambda: binary, subprocess=sp,
                      time=clock or NS(sleep=lambda s: None, monotonic=lambda: 0.0)))


def test_already_up_spawns_nothing():
    spawned = []
    assert R.ensure_docker(ctx=_machine(spawned, lambda timeout=12: True)) is True and spawned == []


def test_not_installed_is_false_without_spawning():
    spawned = []
    assert R.ensure_docker(ctx=_machine(spawned, lambda timeout=12: False)) is False and spawned == []


def test_spawns_desktop_hidden_and_waits_for_the_engine():
    answers = iter([False, False, True])
    spawned = []
    ctx = _machine(spawned, lambda timeout=12: next(answers), binary=r"C:\x\Docker Desktop.exe")
    assert R.ensure_docker(wait_s=30, ctx=ctx) is True
    (args, kw), = spawned
    assert args == [r"C:\x\Docker Desktop.exe"]
    assert kw["creationflags"] & _FLAG == _FLAG


def test_gives_up_after_the_deadline():
    t = [0.0]

    def mono():
        t[0] += 5.0
        return t[0]
    ctx = _machine([], lambda timeout=12: False, binary=r"C:\x\Docker Desktop.exe",
                   clock=NS(sleep=lambda s: None, monotonic=mono))
    assert R.ensure_docker(wait_s=20, ctx=ctx) is False


def test_run_python_waits_for_a_booting_engine():
    import inspect
    src = inspect.getsource(R.run_python)
    assert "ensure_docker(" in src, "run_code must wait for the engine, not fail on the first script"
    assert "ensure_docker(" in inspect.getsource(R.install)


def test_launch_all_starts_docker():
    src = open(os.path.join(os.path.dirname(__file__), "..", "scripts", "launch_all.py"), encoding="utf-8").read()
    assert "ensure_docker(" in src
