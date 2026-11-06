import importlib.util
from pathlib import Path
from types import SimpleNamespace


def test_restart_uses_project_environment(monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'restart_assistant.py'
    spec = importlib.util.spec_from_file_location('restart_assistant', script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root = tmp_path / 'project'
    (root / 'scripts').mkdir(parents=True)
    python = root / 'venv' / 'Scripts' / 'pythonw.exe'
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(mod, '__file__', str(root / 'scripts' / 'restart_assistant.py'))
    monkeypatch.setattr(mod.sys, 'argv', ['restart_assistant.py', '123'])
    monkeypatch.setattr(mod.os, 'name', 'nt')
    kernel = SimpleNamespace(OpenProcess=lambda *args: 0,
                             WaitForSingleObject=lambda *args: 0,
                             CloseHandle=lambda *args: None)
    monkeypatch.setattr(mod.ctypes, 'windll', SimpleNamespace(kernel32=kernel), raising=False)
    called = []
    monkeypatch.setattr(mod.subprocess, 'Popen', lambda args, **kw: called.append((args, kw)))
    mod.main()
    assert called[0][0] == [str(python), str(root / 'scripts' / 'launch_all.py')]
    assert called[0][1]['env']['STARTUP_DIALOG_TIMEOUT_S'] == '1'
