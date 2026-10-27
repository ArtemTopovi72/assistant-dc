"""Start the locally installed case-fan profile without elevating the bot."""
import logging
import os
from pathlib import Path
import subprocess


def ensure_started(*, run=subprocess.run, environ=None):
    environ = os.environ if environ is None else environ
    if os.name != 'nt' or environ.get('F5_TEST_RUN'):
        return False
    installed = Path(environ.get('LOCALAPPDATA', '')) / 'AssistantDC' / 'Cooling' / 'installed.json'
    if not installed.is_file():
        return False
    try:
        result = run(['schtasks.exe', '/Run', '/TN', 'AssistantDC-CaseCooling'],
                     capture_output=True, text=True, timeout=10,
                     creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode == 0:
            return True
        logging.getLogger(__name__).warning('Case cooling did not start: %s', result.stderr)
    except (OSError, subprocess.SubprocessError):
        logging.getLogger(__name__).exception('Case cooling did not start')
    return False
