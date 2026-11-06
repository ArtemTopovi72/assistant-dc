"""Small allowlist of desktop controls callable without showing the window."""
import threading

from PyQt5.QtCore import QObject, QThread, pyqtSignal


class GuiControl(QObject):
    requested = pyqtSignal(object)

    def __init__(self, buttons):
        super().__init__()
        self.buttons = buttons
        self.requested.connect(self._press)

    def _press(self, request):
        if request['cancelled'].is_set():
            return
        button = self.buttons[request['action']]
        try:
            if button.isEnabled():
                button.click()
                request['result'] = {'ok': True}
            else:
                request['result'] = {'error': 'action disabled'}
        except RuntimeError:
            request['result'] = {'error': 'control unavailable'}
        finally:
            request['done'].set()

    def press(self, action):
        if action not in self.buttons:
            return {'error': 'unknown action'}
        request = {'action': action, 'done': threading.Event(), 'cancelled': threading.Event()}
        if QThread.currentThread() == self.thread():
            self._press(request)
        else:
            self.requested.emit(request)
            if not request['done'].wait(5):
                request['cancelled'].set()
                return {'error': 'GUI did not respond; action may have executed'}
        return request['result']
