import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.name != 'nt', reason='Windows OEM console output')
def test_hidden_windows_process_discovery_has_no_reader_thread_decode_errors():
    code = '''
import no_console_windows
no_console_windows.install(force=True)
import llama_backend
assert isinstance(llama_backend._pids(), list)
'''
    result = subprocess.run([sys.executable, '-X', 'utf8', '-c', code],
                            capture_output=True, timeout=25)
    assert result.returncode == 0, result.stderr.decode('utf-8', errors='replace')
    assert b'UnicodeDecodeError' not in result.stderr
