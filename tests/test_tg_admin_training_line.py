"""The admin panel answers "как процесс?" without opening a log file.

A LoRA run owns the card for most of a day. The operator asked how it was
going repeatedly, in chat, while the answer sat in a log file on a machine they
were not sitting at. The panel now carries one line: run, step, percent,
remaining time and rate.

Two things it must not do. It must not report a FINISHED run as live -- a dead
run advertising "осталось ~16ч" is a lie the operator would wait on. And it
must not take the panel down with it: everything about training is read off the
disk, and a half-written log or a deleted run folder is normal.

Run: venv/Scripts/python.exe tests/test_tg_admin_training_line.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import time
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_admtrain_"))

import characters as C
C.redirect_store(os.path.join(tempfile.mkdtemp(prefix="tgtest_admstore_"),
                              "characters.json"))
import lora_training as LT

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name
          + ("" if cond or not detail else "  -- " + str(detail)[:300]))
    if not cond and os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


def _bot():
    return T.TelegramBot("123:TEST", lambda: None, lambda: None,
                         lambda: {}, silent_mode=True)


def _progress(running=True, **kw):
    base = {"slug": "neurostepan_ideo32", "total": 8000, "step": 1071,
            "checkpoints": 4, "samples": 0, "last_sample": "", "tail": "",
            "eta": "16:40:38", "rate": "8.66s/it", "running": running}
    base.update(kw)
    return base


class _Patch:
    def __init__(self, prog, chars=(("neurostepan", ), )):
        self.prog = prog

    def __enter__(self):
        self.orig_p, self.orig_c = LT.progress, C.list_characters
        LT.progress = lambda slug: self.prog
        C.list_characters = lambda: [{"slug": "neurostepan", "name": "Степан"}]
        return self

    def __exit__(self, *a):
        LT.progress, C.list_characters = self.orig_p, self.orig_c


def test_a_live_run_is_reported_with_everything_worth_knowing():
    with _Patch(_progress()):
        line = _bot()._training_line("ru")
    check("the run is named", "neurostepan_ideo32" in line, line)
    check("the step and total are there", "1071" in line and "8000" in line, line)
    check("a percentage is worked out", "13%" in line, line)
    check("the remaining time is human, not hh:mm:ss",
          "16ч 40м" in line and "16:40:38" not in line, line)
    check("the rate is there", "8.66s/it" in line, line)


def test_it_speaks_english_too():
    with _Patch(_progress()):
        line = _bot()._training_line("en")
    check("English gets an English line", "Training" in line and "step" in line, line)


def test_a_run_that_is_over_says_HOW_it_ended():
    """"не идёт" hid the difference between a run that FINISHED and one that
    was killed part-way -- and that decides whether the next move is to publish
    a checkpoint or to restart."""
    with _Patch(_progress(running=False, step=1071)):
        line = _bot()._training_line("ru")
    check("a killed run says it stopped", "ОСТАНОВЛЕНО" in line, line)
    check("and where it stopped", "1071/8000" in line, line)
    check("and offers no remaining time", "осталось" not in line, line)

    with _Patch(_progress(running=False, step=8000)):
        line = _bot()._training_line("ru")
    check("a finished run says so", "завершено" in line, line)
    check("and is not called stopped", "ОСТАНОВЛЕНО" not in line, line)

    # A run stopped weeks ago is history, not news: it sat in the desktop
    # admin header as «Обучение: neurostepan…» with the GPU at 0%.
    old_log = os.path.join(tempfile.mkdtemp(prefix="tgtest_oldlog_"), "train.log")
    open(old_log, "w").close()
    os.utime(old_log, (time.time() - 20 * 86400,) * 2)
    orig_lp = LT.log_path
    LT.log_path = lambda slug: __import__("pathlib").Path(old_log)
    try:
        with _Patch(_progress(running=False, step=1071)):
            line = _bot()._training_line("ru")
    finally:
        LT.log_path = orig_lp
    check("a run stopped weeks ago is not reported", "ОСТАНОВЛЕНО" not in line, line)

    with _Patch(_progress(running=False, step=0)):
        line = _bot()._training_line("ru")
    check("a character that never trained is not reported at all",
          "не идёт" in line, line)

    # A run with no recorded step count cannot be judged. run_state has to
    # call it stopped, and "ОСТАНОВЛЕНО на шаге 1147/0" reads as a bug.
    with _Patch(_progress(running=False, step=1147, total=0)):
        line = _bot()._training_line("ru")
    check("an unmeasurable run is not reported as stopped at N of zero",
          "не идёт" in line, line)


def test_a_run_with_no_estimate_yet_still_reports():
    with _Patch(_progress(eta="", rate="", step=0)):
        line = _bot()._training_line("ru")
    check("it still names the run", "neurostepan_ideo32" in line, line)
    check("with no dangling separator", "· ·" not in line
          and not line.rstrip().endswith("·"), line)


def test_a_broken_read_cannot_take_the_panel_down():
    """Everything here is read off the disk while a run is writing to it."""
    orig = LT.progress
    LT.progress = lambda slug: (_ for _ in ()).throw(OSError("half-written log"))
    try:
        line = _bot()._training_line("ru")
    finally:
        LT.progress = orig
    check("it degrades to the idle line rather than raising",
          isinstance(line, str) and line.strip(), line)


def _main():
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        try:
            fn()
        except Exception as exc:
            check(fn.__name__ + " raised", False, exc)
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d/%d passed" % (len(RESULTS) - len(bad), len(RESULTS)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_main())
