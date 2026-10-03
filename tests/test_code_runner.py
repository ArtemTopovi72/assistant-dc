"""Brakes and walls for model-written code.

The suite is mostly about what must NOT happen. Running code a local 26B wrote,
on request from a Telegram chat, is the single most dangerous thing in this
project, and the two ways it goes wrong are: it never stops, or it escapes.

The container flags are asserted literally. A mock cannot tell us whether the
server accepts a payload (that lesson is recorded in test_forced_calculate),
but it can absolutely tell us whether `--network none` is still in the argv —
and a hardening flag silently dropped in a refactor is invisible in every other
way, because the code keeps working perfectly right up until it matters.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import code_runner as R
from code_sandbox import Sandbox


@pytest.fixture
def box(tmp_path):
    return Sandbox(tmp_path / "proj")


@pytest.fixture
def no_docker(monkeypatch):
    monkeypatch.setattr(R, "docker_available", lambda *a, **k: False)
    # ...and no engine to start either (ensure_docker would otherwise really
    # launch Docker Desktop and wait a minute for it).
    monkeypatch.setattr(R, "desktop_binary", lambda: None)


# --- the rule that matters: never silently degrade -------------------------

def test_without_isolation_code_does_not_run(box, no_docker):
    res = R.run_python(box, "print('hi')")
    assert not res.ok
    assert "unavailable" in res.output.lower()


def test_the_refusal_does_not_execute_anything(box, no_docker, monkeypatch):
    """The script is written down for inspection, but nothing spawns."""
    called = []

    def fake(*a, **k):
        called.append(a)
        return R.RunResult(True, 0, "", 0.0)

    monkeypatch.setattr(R, "_spawn", fake)
    R.run_python(box, "import os; os.remove('x')")
    assert not called, "a subprocess was started with no isolated backend"


def test_host_execution_needs_an_explicit_decision(box, no_docker):
    res = R.run_python(box, "print('allowed')", allow_host=True)
    assert res.ok, res.output
    assert "allowed" in res.output
    assert "UNISOLATED" in res.backend, "an unisolated run must say so"


# --- the brakes -------------------------------------------------------------

def test_an_endless_loop_is_killed(box, no_docker):
    t0 = time.monotonic()
    res = R.run_python(box, "while True:\n    pass\n", timeout=3, allow_host=True)
    assert not res.ok
    assert "killed after" in res.output
    assert time.monotonic() - t0 < 30, "the deadline did not hold"


def test_waiting_on_input_fails_instead_of_hanging(box, no_docker):
    """stdin is /dev/null, so input() raises immediately rather than blocking
    the bot until the timeout."""
    res = R.run_python(box, "input()", timeout=20, allow_host=True)
    assert not res.ok
    assert "killed after" not in res.output, "it blocked instead of failing fast"


def test_runaway_output_is_clipped(box, no_docker):
    res = R.run_python(box, "print('x' * 200000)", timeout=30, allow_host=True)
    assert len(res.output) < R.MAX_OUTPUT_CHARS + 500, len(res.output)
    assert "chars cut" in res.output


def test_clipping_keeps_both_ends():
    """The traceback is at the end; what caused it is at the beginning."""
    text = "START" + ("m" * 50_000) + "END"
    out = R._clip(text)
    assert out.startswith("START") and out.endswith("END")


def test_the_timeout_is_capped(box, no_docker, monkeypatch):
    seen = {}

    def fake(cmd, cwd, timeout):
        seen["t"] = timeout
        return R.RunResult(True, 0, "", 0.0)

    monkeypatch.setattr(R, "_spawn", fake)
    R.run_python(box, "print(1)", timeout=99999, allow_host=True)
    assert seen["t"] == R.MAX_TIMEOUT


# --- installing -------------------------------------------------------------

def test_option_shaped_packages_are_refused(box, no_docker):
    """`--target /elsewhere` and `--index-url http://…` are how a package list
    turns into an arbitrary pip invocation."""
    res = R.install(box, ["requests", "--target", "C:/Windows"])
    assert not res.ok and "option-shaped" in res.output


def test_installing_never_touches_the_project_environment(box, no_docker, monkeypatch):
    """The project pins a CUDA torch build; a pip run that resolves torch into
    the project venv replaces it with the CPU wheel."""
    seen = {}

    def fake(cmd, cwd, timeout):
        seen["cmd"] = cmd
        return R.RunResult(True, 0, "", 0.0)

    monkeypatch.setattr(R, "_spawn", fake)
    R.install(box, ["requests"], allow_host=True)
    cmd = seen["cmd"]
    assert "--target" in cmd
    target = cmd[cmd.index("--target") + 1]
    assert str(box.root) in str(target), target
    assert "-m" in cmd and "pip" in cmd


# --- the container hardening ------------------------------------------------

def _argv_for(box, monkeypatch, network=False):
    seen = {}

    def fake(cmd, cwd, timeout):
        seen["cmd"] = cmd
        return R.RunResult(True, 0, "", 0.0)

    monkeypatch.setattr(R, "docker_binary", lambda: "docker")
    monkeypatch.setattr(R, "_spawn", fake)
    R._run_in_container(box, ["python", "-u", "x.py"], 30, network=network)
    return seen["cmd"]


@pytest.mark.parametrize("flag,value", [
    ("--network", "none"),
    ("--cap-drop", "ALL"),
    ("--security-opt", "no-new-privileges"),
    ("--memory", R.CONTAINER_MEMORY),
    ("--pids-limit", R.CONTAINER_PIDS),
    ("--cpus", R.CONTAINER_CPUS),
])
def test_every_hardening_flag_is_present(box, monkeypatch, flag, value):
    cmd = _argv_for(box, monkeypatch)
    assert flag in cmd, f"{flag} is gone"
    assert cmd[cmd.index(flag) + 1] == value


def test_the_container_filesystem_is_read_only(box, monkeypatch):
    assert "--read-only" in _argv_for(box, monkeypatch)


def test_only_this_sandbox_is_mounted(box, monkeypatch):
    cmd = _argv_for(box, monkeypatch)
    mounts = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-v"]
    assert len(mounts) == 1, mounts
    assert mounts[0].startswith(str(box.root)), mounts
    assert mounts[0].endswith(":/work")


def test_the_container_is_thrown_away(box, monkeypatch):
    """--rm: nothing a script does survives its own run."""
    assert "--rm" in _argv_for(box, monkeypatch)


def test_installing_is_the_only_thing_that_gets_the_network(box, monkeypatch):
    net = _argv_for(box, monkeypatch, network=True)
    assert net[net.index("--network") + 1] == "bridge"
    plain = _argv_for(box, monkeypatch, network=False)
    assert plain[plain.index("--network") + 1] == "none"


def test_a_stopped_engine_is_not_mistaken_for_a_missing_one(monkeypatch):
    """Docker Desktop leaves the binary on disk with the daemon down; in that
    state `--version` succeeds and every run fails."""
    monkeypatch.setattr(R, "docker_binary", lambda: "docker")
    monkeypatch.setattr(R, "docker_available", lambda *a, **k: False)
    backend, why = R.backend_status()
    assert backend == ""
    assert "not running" in why
