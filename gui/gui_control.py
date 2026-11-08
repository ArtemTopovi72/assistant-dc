"""Small allowlist of desktop controls callable without showing the window."""
import logging
import threading

from PyQt5.QtCore import QObject, QThread, pyqtSignal


class GuiControl(QObject):
    requested = pyqtSignal(object)

    def __init__(self, buttons, model_state=None, model_select=None, model_lock=None):
        super().__init__()
        self.buttons = buttons
        self._model_state = model_state
        self._model_select = model_select
        self._model_lock = model_lock
        self.requested.connect(self._press)

    def _press(self, request):
        if request['cancelled'].is_set():
            return
        try:
            if request['action'] == '_model_state':
                request['result'] = self._model_state()
                return
            if request['action'] == '_model_select':
                request['result'] = self._model_select(*request['args'])
                return
            button = self.buttons[request['action']]
            if button.isEnabled():
                if request['action'] == 'restart':
                    # Acknowledge acceptance before shutdown blocks the GUI thread.
                    request['result'] = {'ok': True}
                    request['done'].set()
                button.click()
                request['result'] = {'ok': True}
            else:
                request['result'] = {'error': 'action disabled'}
        except Exception:
            logging.getLogger('assistant.gui_control').exception('GUI control failed')
            request['result'] = {'error': 'control unavailable'}
        finally:
            request['done'].set()

    def press(self, action):
        """For restart, ok means accepted; callers check health for readiness."""
        if action not in self.buttons:
            return {'error': 'unknown action'}
        return self._request(action)

    def model_state(self):
        return self._request('_model_state') if self._model_state else {'error': 'model control unavailable'}

    def model_select(self, model, thinking):
        if not self._model_select:
            return {'error': 'model control unavailable'}
        lock = self._model_lock() if self._model_lock else None
        timeout = 0 if QThread.currentThread() == self.thread() else 7
        if lock is not None and not lock.acquire(timeout=timeout):
            return {'error': 'bot is busy'}
        try:
            return self._request('_model_select', model, thinking, lock)
        finally:
            if lock is not None:
                lock.release()

    def _request(self, action, *args):
        request = {'action': action, 'args': args, 'done': threading.Event(), 'cancelled': threading.Event()}
        if QThread.currentThread() == self.thread():
            self._press(request)
        else:
            self.requested.emit(request)
            if not request['done'].wait(5):
                request['cancelled'].set()
                return {'error': 'GUI did not respond; action may have executed'}
        return request['result']
