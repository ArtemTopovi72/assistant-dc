import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import threading, time
import stress


def test_warm_and_first_call_load_once():
    a = stress.BilingualAccentor(enable_english=False)
    loads = []

    def fake_load():
        if a._ru is None:
            loads.append(1); time.sleep(0.3); a._ru = object()
        return a._ru
    a._ensure_ru_locked = fake_load
    a.warm()
    time.sleep(0.05)
    assert a._ensure_ru() is a._ru
    assert loads == [1]
