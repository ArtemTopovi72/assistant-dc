import subprocess
import os

import pytest

import case_cooling


@pytest.mark.skipif(os.name != 'nt', reason='Windows scheduled task')
def test_installed_cooling_starts_scheduled_task(tmp_path):
    env = {'LOCALAPPDATA': str(tmp_path)}
    marker = tmp_path / 'AssistantDC' / 'Cooling' / 'installed.json'
    marker.parent.mkdir(parents=True)
    marker.write_text('{}')
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)
    assert case_cooling.ensure_started(run=run, environ=env)
    assert calls == [['schtasks.exe', '/Run', '/TN', 'AssistantDC-CaseCooling']]


def test_other_machines_and_tests_do_not_start_hardware_control(tmp_path):
    env = {'LOCALAPPDATA': str(tmp_path)}
    def forbidden(*a, **kw):
        raise AssertionError('must not control hardware')
    assert not case_cooling.ensure_started(run=forbidden, environ=env)
    marker = tmp_path / 'AssistantDC' / 'Cooling' / 'installed.json'
    marker.parent.mkdir(parents=True)
    marker.write_text('{}')
    env['F5_TEST_RUN'] = '1'
    assert not case_cooling.ensure_started(run=forbidden, environ=env)


@pytest.mark.skipif(os.name != 'nt', reason='Windows scheduled task')
def test_task_failure_is_reported(tmp_path, caplog):
    env = {'LOCALAPPDATA': str(tmp_path)}
    marker = tmp_path / 'AssistantDC' / 'Cooling' / 'installed.json'
    marker.parent.mkdir(parents=True)
    marker.write_text('{}')
    assert not case_cooling.ensure_started(run=lambda a, **k: subprocess.CompletedProcess(a, 1, '', 'denied'), environ=env)
    assert 'cooling' in caplog.text.lower()
