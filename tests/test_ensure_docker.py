"""The app starts its own Docker engine, like it starts LM Studio and ComfyUI.

Live 2026-09-14: the bot found every bridge and then said "Docker не
запущен" -- Docker Desktop was installed and healthy, just not launched.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import code_runner as R


def test_already_up_spawns_nothing(monkeypatch):
    monkeypatch.setattr(R, "docker_available", lambda timeout=12: True)
    spawned = []
    monkeypatch.setattr(R.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    assert R.ensure_docker() is True and spawned == []


def test_not_installed_is_false_without_spawning(monkeypatch):
    monkeypatch.setattr(R, "docker_available", lambda timeout=12: False)
    monkeypatch.setattr(R, "desktop_binary", lambda: None)
    spawned = []
    monkeypatch.setattr(R.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    assert R.ensure_docker() is False and spawned == []


def test_spawns_desktop_hidden_and_waits_for_the_engine(monkeypatch):
    answers = iter([False, False, True])
    monkeypatch.setattr(R, "docker_available", lambda timeout=12: next(answers))
    monkeypatch.setattr(R, "desktop_binary", lambda: r"C:\x\Docker Desktop.exe")
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    spawned = []
    monkeypatch.setattr(R.subprocess, "Popen", lambda args, **k: spawned.append((args, k)))
    assert R.ensure_docker(wait_s=30) is True
    (args, kw), = spawned
    assert args == [r"C:\x\Docker Desktop.exe"]
    assert kw["creationflags"] & getattr(R.subprocess, "CREATE_NO_WINDOW", 0) == getattr(R.subprocess, "CREATE_NO_WINDOW", 0)


def test_gives_up_after_the_deadline(monkeypatch):
    monkeypatch.setattr(R, "docker_available", lambda timeout=12: False)
    monkeypatch.setattr(R, "desktop_binary", lambda: r"C:\x\Docker Desktop.exe")
    monkeypatch.setattr(R.subprocess, "Popen", lambda *a, **k: None)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    t = [0.0]
    def mono():
        t[0] += 5.0; return t[0]
    monkeypatch.setattr(R.time, "monotonic", mono)
    assert R.ensure_docker(wait_s=20) is False


def test_run_python_waits_for_a_booting_engine():
    import inspect
    src = inspect.getsource(R.run_python)
    assert "ensure_docker(" in src, "run_code must wait for the engine, not fail on the first script"
    assert "ensure_docker(" in inspect.getsource(R.install)


def test_launch_all_starts_docker():
    src = open(os.path.join(os.path.dirname(__file__), "..", "scripts", "launch_all.py"), encoding="utf-8").read()
    assert "ensure_docker(" in src
