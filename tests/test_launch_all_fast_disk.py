import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize('enabled, expected', [('1', '--fast-disk'), ('0', '--disable-fast-disk')])
def test_comfy_disk_mode(monkeypatch, tmp_path, enabled, expected):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv('F5_TEST_RUN', '1')
    monkeypatch.setenv('COMFY_FAST_DISK', enabled)
    spec = importlib.util.spec_from_file_location('launch_all_disk_test', root / 'scripts' / 'launch_all.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / 'main.py').write_text('', encoding='utf8')
    python = tmp_path / 'python.exe'
    python.touch()
    monkeypatch.setattr(module, 'COMFY_SRC', str(tmp_path))
    monkeypatch.setattr(module, 'COMFY_PY', str(python))
    monkeypatch.setattr(module, 'COMFY_LOG', str(tmp_path / 'comfy.log'))
    calls = []
    monkeypatch.setattr(module.subprocess, 'Popen', lambda args, **kwargs: calls.append(args) or object())
    assert module._spawn_comfy_source()
    assert expected in calls[0]
    assert ('--fast-disk' in calls[0]) != ('--disable-fast-disk' in calls[0])
