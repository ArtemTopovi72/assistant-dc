"""The Персонажи tab must answer the only question a day-long run raises.

An 8000-step Ideogram run is ~20 hours. The tab showed "шаг 787 из 8000" and
the raw tqdm bar, which contains the remaining time but buried in
"[1:52:05<17:35:31,  8.78s/it]" -- a string most people read as elapsed, or as
a clock time.

The figure is LIFTED FROM THE TRAINER'S OWN BAR rather than computed here from
step counts and wall clock. The trainer's estimate accounts for its own warm-up
and its own stalls, and it survives a GUI restart, which a locally accumulated
average would not.

Run: venv/Scripts/python.exe tests/test_training_eta.py
"""
import io
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name
          + ("" if cond or not detail else "  -- " + str(detail)[:300]))
    if not cond and os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


import lora_training as LT

_TMP = Path(tempfile.mkdtemp(prefix="eta_"))


_REAL_TRAIN_ROOT = LT.TRAIN_ROOT


def _run_with_log(text, slug="fake_run"):
    """A run folder holding nothing but a log — progress() reads the disk.

    TRAIN_ROOT is restored afterwards: these suites share one process under
    run_all.py, and a module global left pointing at a temp dir is how a later
    suite reads an empty machine.
    """
    LT.TRAIN_ROOT = _TMP
    d = _TMP / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.log").write_text(text, encoding="utf-8")
    orig = LT.is_training
    LT.is_training = lambda s: True
    try:
        return LT.progress(slug)
    finally:
        LT.is_training = orig
        LT.TRAIN_ROOT = _REAL_TRAIN_ROOT


BAR = ("fake_run:  10%|9         | 783/8000 [1:52:05<17:35:31,  8.78s/it, "
       "lr: 1.0e-04 loss: 6.785e-01]")


def test_the_bar_gives_up_step_eta_and_rate():
    p = _run_with_log(BAR)
    check("the live step is read", p["step"] == 783, p)
    check("the total is the run's own, not a spin box", p["total"] == 8000, p)
    check("the remaining time is lifted out", p["eta"] == "17:35:31", p)
    check("and so is the rate", p["rate"] == "8.78s/it", p)


def test_a_bar_with_no_timing_yet_still_yields_the_step():
    """The first bar of a run has no estimate at all. One regex for step and
    timing together would have dropped the step with it."""
    p = _run_with_log("fake_run:   0%|          | 0/8000 [00:00<?, ?it/s]", "no_eta")
    check("the step survives", p["step"] == 0 and p["total"] == 8000, p)
    check("and the eta is simply empty, not garbage", p["eta"] in ("", "?"), p)


def test_a_finished_bar_does_not_invent_time_left():
    p = _run_with_log("fake_run: 100%|##########| 8000/8000 [19:31:12<00:00,  8.78s/it]",
                      "done_run")
    check("the step is the last one", p["step"] == 8000, p)
    check("nothing is left", p["eta"] == "00:00", p)


def test_human_eta_is_readable_at_every_scale():
    src = (ROOT / "gui/gui_characters_tab.py").read_text(encoding="utf-8")
    ns = {}
    exec(src[src.index("def _human_eta"):src.index("def _fingerprint")], ns)
    f = ns["_human_eta"]
    check("hours and minutes", f("17:23:40") == "17ч 23м", f("17:23:40"))
    check("under an hour drops the hour", f("00:45:12") == "45м", f("00:45:12"))
    check("mm:ss is minutes", f("02:00") == "2м", f("02:00"))
    check("a nonsense value is passed through, not crashed on", f("?") == "?")


def test_the_tab_actually_shows_it():
    src = (ROOT / "gui/gui_characters_tab.py").read_text(encoding="utf-8")
    body = src[src.index("def _refresh_progress"):src.index("def _selected")]
    check("the tab renders the eta", "_human_eta(" in body, body)
    check("only while the run is live -- a stale eta from a dead run is a lie",
          'p["running"]' in body[body.index('p.get("eta")'):
                                 body.index('p.get("eta")') + 120], body)
    check("a percentage is shown too", '100.0 * p["step"]' in body, body)


def test_a_run_is_not_started_onto_a_full_disk():
    """A trainer that runs out of DISK does not fail loudly: it dies inside a
    checkpoint write, hours in, and leaves a partial .safetensors that the
    resume path picks up as the newest save. One stat call before starting."""
    orig = LT.MIN_FREE_GB
    try:
        LT.MIN_FREE_GB = 0.0
        check("plenty of room says nothing", LT.disk_warning() == "",
              LT.disk_warning())
        LT.MIN_FREE_GB = 10.0 ** 9
        warn = LT.disk_warning()
        check("no room says so", bool(warn), warn)
        check("and it names the actual figure", "ГБ" in warn, warn)
    finally:
        LT.MIN_FREE_GB = orig
    check("an unreadable path is not reported as a full disk",
          LT.free_gb("Q:/definitely/not/here") == 0.0)
    check("and an unknown figure does not trigger the warning",
          LT.disk_warning("Q:/definitely/not/here") == "")

    tab = (ROOT / "gui/gui_characters_tab.py").read_text(encoding="utf-8")
    body = tab[tab.index("def _train(self):"):tab.index("def _train_done")]
    check("the tab asks before starting on a full disk",
          "disk_warning()" in body, body[-600:])
    check("and it is a question, not a refusal -- the run may be deliberate",
          "QMessageBox.question(" in body[body.index("disk_warning()"):], body[-600:])


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
