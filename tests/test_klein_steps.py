"""Klein's step count: KLEIN_STEPS (32) for users, KLEIN_DEBUG_STEPS (16) for a debug-chat task --
the debug chat is where edits are tried over and over and speed beats detail (owner, 2026-10-21).

Run: venv/Scripts/python.exe -m pytest tests/test_klein_steps.py -q
"""
import inspect, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import klein


def _steps(wf):
    return wf[klein.N_SAMPLER]["inputs"]["steps"]


def test_users_get_the_full_steps_and_a_debug_task_the_fast_ones():
    assert _steps(klein.t2i_workflow("a cat", 512, 512, 1)) == config.KLEIN_STEPS == 32
    with klein.steps_for_task(config.KLEIN_DEBUG_STEPS):
        assert _steps(klein.t2i_workflow("a cat", 512, 512, 1)) == 16
        assert _steps(klein.edit_workflow("x.png", "make it red", 1)) == 16
        assert _steps(klein.t2i_workflow("a cat", 512, 512, 1, steps=8)) == 8   # an explicit count wins
    assert _steps(klein.edit_workflow("x.png", "make it red", 1)) == 32


def test_the_bot_sets_the_debug_steps_for_a_debug_chat_task():
    import tg_tasks
    src = inspect.getsource(tg_tasks)
    assert "steps_for_task(_cfg.KLEIN_DEBUG_STEPS if _dbg.is_debug(chat_id) else None)" in src
