"""The Персонажи tab must report on the run that is ACTUALLY training.

Found live: the tab showed "обучение: не активно · шаг 2250 из 2000" while a
rank-32 Ideogram run was mid-flight. Two separate bugs behind one line:

  * a character can own several runs -- the old model under its bare slug, Ideogram
    under <slug>_ideo and <slug>_ideo32 -- and progress() only ever looked at
    the bare slug, so it described a run that had finished hours earlier;
  * the total came from the spin box on the form rather than from the run, so
    a run extended past that number read "2250 из 2000".

Run: venv/Scripts/python.exe tests/test_lora_progress_run_resolution.py
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import lora_training as LT

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name
          + ("" if cond or not detail else "  -- " + str(detail)[:300]))
    if not cond and os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


class _Root:
    """Redirect TRAIN_ROOT at a scratch tree — these tests must never read the
    real runs, whose contents change while a training job is live."""
    def __enter__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="loraprog_"))
        self.orig = LT.TRAIN_ROOT
        LT.TRAIN_ROOT = self.tmp
        return self.tmp

    def __exit__(self, *a):
        LT.TRAIN_ROOT = self.orig


def _make_run(root, slug, steps, ckpt_steps=()):
    d = root / slug
    (d / slug).mkdir(parents=True, exist_ok=True)
    (d / ("%s.yaml" % slug)).write_text("        steps: %d\n" % steps,
                                        encoding="utf-8")
    for st in ckpt_steps:
        (d / slug / ("%s_%09d.safetensors" % (slug, st))).write_bytes(b"x")
    return d


def test_variants_are_found_and_ordered():
    with _Root() as root:
        _make_run(root, "hero", 2000, (250, 500))
        _make_run(root, "hero_ideo", 4250, (250,))
        _make_run(root, "hero_ideo32", 8000)
        _make_run(root, "someone_else", 100)
        names = LT.run_slugs("hero")
        check("all three of this character's runs are found",
              set(names) == {"hero", "hero_ideo", "hero_ideo32"}, names)
        check("another character's run is not swept in",
              "someone_else" not in names, names)
        check("an unknown character has no runs", LT.run_slugs("nobody") == [])


def test_the_live_run_wins_over_the_newest():
    with _Root() as root:
        _make_run(root, "hero", 2000, (250, 500, 2250))
        _make_run(root, "hero_ideo32", 8000)
        orig = LT.is_training
        LT.is_training = lambda s: s == "hero_ideo32"
        try:
            p = LT.progress("hero")
        finally:
            LT.is_training = orig
        check("progress resolves to the run that is training",
              p["slug"] == "hero_ideo32", p)
        check("and reports it as running", p["running"] is True, p)
        check("the total comes from that run's config, not the other's",
              p["total"] == 8000, p)


def test_the_total_comes_from_the_run():
    with _Root() as root:
        _make_run(root, "hero", 4250, (250, 4000))
        orig = LT.is_training
        LT.is_training = lambda s: False
        try:
            p = LT.progress("hero")
        finally:
            LT.is_training = orig
        check("the configured step count is read from the yaml",
              p["total"] == 4250, p)
        check("the step is the newest checkpoint", p["step"] == 4000, p)
        check("a step never exceeds its own total", p["step"] <= p["total"], p)


def test_the_live_step_is_read_from_the_progress_bar():
    """Checkpoints land every 250, so between them the step read off filenames
    is stale -- and reads 0 for the first quarter-hour of a run, which looks
    exactly like nothing happening."""
    with _Root() as root:
        d = _make_run(root, "hero", 8000)
        bar = ("hero:  18%|#8    | 1466/8000 [3:30:04<2:28:19,  8.61s/it]\r"
               "hero:  18%|#8    | 1467/8000 [3:30:12<2:28:11,  8.61s/it]\r")
        (d / "train.log").write_text("loading\n" + bar, encoding="utf-8")
        orig = LT.is_training
        LT.is_training = lambda s: True
        try:
            p = LT.progress("hero")
        finally:
            LT.is_training = orig
        check("the current step comes from the trainer's own bar",
              p["step"] == 1467, p)
        check("so does the total", p["total"] == 8000, p)


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
