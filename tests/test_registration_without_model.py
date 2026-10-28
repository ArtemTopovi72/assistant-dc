"""Display names need no semantic model; registration controls stay controls."""
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT] + [os.path.join(ROOT, d) for d in ("agent", "bot", "core", "imaging")]

import tg_bot
from tg_registration import RegistrationMixin
from tg_strings import _b


class Gate(RegistrationMixin):
    def __init__(self, state="awaiting_name", user=None):
        self.session = SimpleNamespace(reg_state=state, reg_name="", is_admin=False)
        self.users = {} if user is None else {101: user}
        self._store = SimpleNamespace(put=lambda sess: None)
        self._user_store = SimpleNamespace(get=self.users.get,
                                           put=lambda user: self.users.update({101: user}))
        self._activity = SimpleNamespace(log=lambda *args: None)
        self.reads = SimpleNamespace(yes_no=self._unexpected_read)
        self.sent = []

    @staticmethod
    def _unexpected_read(*args, **kwargs):
        raise AssertionError("Registration must not wait for an LLM")

    def _get_session(self, chat_id):
        return self.session

    def _lang(self, sess):
        return "ru"

    def _send_text(self, chat_id, text, **kwargs):
        self.sent.append(text)

    def _main_menu_kb(self, sess, lang):
        return {}

    def _scrub_secret(self, chat_id, msg):
        msg["_secret"] = True


def main():
    for name in ("Artem", "Иван Иванов", "Синий кот", "a " * 30):
        gate = Gate()
        assert gate._user_gate(101, {"text": name}) is False
        assert gate.session.reg_name == name.strip()[:50]
        assert gate.session.reg_state == "awaiting_password"

    for state in ("awaiting_name", "awaiting_password"):
        for text in ("/start", "/clear", _b("library", "ru")):
            gate = Gate(state)
            msg = {"text": text}
            assert gate._user_gate(101, msg) is False
            assert gate.session.reg_state == state
            assert not gate.session.reg_name and not gate.users
            assert not msg.get("_secret")
        gate = Gate(state)
        assert gate._user_gate(101, {"text": "/cancel"}) is False
        assert gate.session.reg_state == ""

    for name in ("", "x"):
        gate = Gate()
        assert gate._user_gate(101, {"text": name}) is False
        assert gate.session.reg_state == "awaiting_name"

    gate = Gate("awaiting_password")
    msg = {"text": "abc"}
    assert gate._user_gate(101, msg) is False
    assert msg["_secret"] and not gate.users

    user = SimpleNamespace(status="approved", name="Old", is_admin=False)
    gate = Gate("change_name", user)
    assert gate._user_gate(101, {"text": "Синий кот"}) is False
    assert user.name == "Синий кот" and gate.session.reg_state == ""
    print("PASS registration accepts display names without a model and preserves controls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
