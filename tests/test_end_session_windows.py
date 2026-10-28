"""A session request to a secondary Qt window must not shut down the bot."""
import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows session messages')
def test_restart_manager_cannot_close_windows_through_another_qt_window():
    code = textwrap.dedent('''
        import ctypes, os
        from ctypes import wintypes
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication, QWidget
        import close_trace as C
        app = QApplication([])
        C.install_end_session_guard(app)
        closed = []
        quits = []
        app.aboutToQuit.connect(lambda: quits.append(True))
        class Main(QWidget):
            def nativeEvent(self, kind, message):
                if C.refuse_end_session(message):
                    return True, 0
                return super().nativeEvent(kind, message)
            def closeEvent(self, event):
                closed.append(True)
                event.accept()
        main, other = Main(), QWidget()
        for window in (main, other):
            window.setAttribute(Qt.WA_ShowWithoutActivating)
            window.move(-30000, -30000)
            window.show()
        app.processEvents()
        user32 = ctypes.windll.user32
        user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.SendMessageW.restype = wintypes.LPARAM
        target = int(other.winId())
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(wintypes.HWND(target), ctypes.byref(owner))
        assert owner.value == os.getpid()
        for _ in range(3):
            answer = user32.SendMessageW(target, C.WM_QUERYENDSESSION, 0, 1)
            app.processEvents()
            assert answer == 0 and main.isVisible() and not closed, (answer, closed)
            user32.SendMessageW(target, C.WM_ENDSESSION, 0, 1)
            user32.SendMessageW(target, C.WM_ENDSESSION, 1, 1)
            app.processEvents()
            assert main.isVisible() and not closed and not quits
        assert main.close() and closed == [True]
        other.close()
        print('PASS')
    ''')
    result = subprocess.run([sys.executable, '-X', 'utf8', '-c', code],
                            env=dict(os.environ, QT_QPA_PLATFORM='windows'),
                            capture_output=True, timeout=25)
    assert result.returncode == 0, result.stderr.decode('utf-8', errors='replace')
    assert b'PASS' in result.stdout
