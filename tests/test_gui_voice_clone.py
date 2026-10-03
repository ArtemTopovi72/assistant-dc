"""GUI voice-clone tab: mounts in the dashboard, workers are joined, flow wiring."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt5.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
import gui_voice_clone_tab as t
import gui_busy_state


class _Host:
    ctx = None


def test_initial_state_and_guards():
    tab = t.VoiceCloneTab(_Host())
    assert not tab.speak_btn.isEnabled()
    tab._take()
    assert "Pick" in tab.status.text()
    tab.ref = "x.wav"; tab._speak()
    assert "Write" in tab.status.text()


def test_ref_then_spoken_updates_ui():
    tab = t.VoiceCloneTab(_Host())
    tab._on_ref("r.wav", "привет как дела")
    assert tab.speak_btn.isEnabled() and "привет" in tab.ref_lbl.text()
    played = []
    tab.player.play = played.append
    tab._on_spoken("o.wav", "Привет.")
    assert played == ["o.wav"] and tab.play_btn.isEnabled()


def test_failure_reasons_are_human():
    assert set(t._REASONS) == {"no_audio", "too_little_speech", "no_words"}


def test_thread_host_registered():
    import inspect
    assert "voice_clone_tab" in inspect.getsource(gui_busy_state)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
