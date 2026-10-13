"""llm_gate: our queue in front of LM Studio -- slots across processes,
priority (a person before a bench), cancellation while waiting."""
import os, sys, time, threading, subprocess, textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import llm_gate as G


class _Ctx:
    def __init__(self, prio=None):
        self.llm_priority = prio
        self.cancel = False
        self.stages = []
    def is_cancelled(self): return self.cancel
    def set_stage(self, s): self.stages.append(s)


_GATE = {}


@pytest.fixture(autouse=True)
def gate(tmp_path):
    _GATE["dir"] = tmp_path
    return tmp_path


def _slot(ctx):
    """One slot in this test's own folder, one at a time, gate on."""
    return G.slot(ctx, gate_dir=_GATE["dir"], n_slots=1, on=True)


def test_one_slot_serialises_calls():
    inside, peak = [0], [0]
    def work():
        with _slot(_Ctx(0)):
            inside[0] += 1; peak[0] = max(peak[0], inside[0])
            time.sleep(0.1); inside[0] -= 1
    ts = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert peak[0] == 1


def test_a_person_goes_before_a_bench():
    order = []
    holder_in = threading.Event(); release = threading.Event()
    def holder():
        with _slot(_Ctx(0)):
            holder_in.set(); release.wait(5)
    def waiter(name, prio, delay):
        time.sleep(delay)
        with _slot(_Ctx(prio)):
            order.append(name)
    h = threading.Thread(target=holder); h.start(); holder_in.wait(5)
    b = threading.Thread(target=waiter, args=("bench", 2, 0.0))
    p = threading.Thread(target=waiter, args=("person", 0, 0.2))
    b.start(); p.start(); time.sleep(0.5); release.set()
    for t in (h, b, p): t.join(5)
    assert order == ["person", "bench"]


def test_cancel_stops_waiting_and_shows_the_place():
    holder_in = threading.Event(); release = threading.Event()
    def holder():
        with _slot(_Ctx(0)):
            holder_in.set(); release.wait(5)
    h = threading.Thread(target=holder); h.start(); holder_in.wait(5)
    c = _Ctx(0)
    threading.Timer(0.6, lambda: setattr(c, "cancel", True)).start()
    with pytest.raises(G.Cancelled):
        with _slot(c):
            pass
    release.set(); h.join(5)
    assert any(s.startswith("Waiting for the model") for s in c.stages)
    assert not list(_GATE["dir"].glob("t_*")), "ticket left behind"


def test_the_slot_is_shared_across_processes(gate):
    code = textwrap.dedent(f"""
        import sys, time; sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})
        import llm_gate as G
        with G.slot(None):
            print("in", flush=True); time.sleep(1.5)
    """)
    env = dict(os.environ, LLM_SLOTS="1", LLM_PRIORITY="0", LLM_GATE="1", LLM_GATE_DIR=str(gate))
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=env)
    assert p.stdout.readline().strip() == "in"
    t0 = time.monotonic()
    with _slot(_Ctx(0)):
        waited = time.monotonic() - t0
    p.wait(10)
    assert waited > 0.8, waited


def test_it_can_be_switched_off():
    with G.slot(_Ctx(), gate_dir=_GATE["dir"], on=False):
        pass


def test_the_russian_status_line():
    import stages
    assert stages.translate("Waiting for the model (2 ahead)", "ru").startswith("Жду своей очереди")
