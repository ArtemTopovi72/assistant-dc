"""The Characters tab, checked where it could lie to the user.

Constructing the real widget is the point: the defects worth catching here are
"a button is enabled that cannot work" and "the tab keeps its own idea of the
Telegram flag", and neither is visible in a mock.
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication      # noqa: E402

import characters as C                        # noqa: E402
import qt_teardown as QT                      # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    yield a


@pytest.fixture()
def tab(app, tmp_path):
    C.redirect_store(tmp_path / "characters.json")
    from gui_characters_tab import CharactersTab

    class Host:
        ctx = None
        images_panel = None
    w = CharactersTab(Host())
    yield w
    QT.release(w)


def _row_texts(t, row):
    return [t.table.item(row, c).text() for c in range(4)]


def test_a_character_without_an_adapter_cannot_be_drawn_or_shared(tab, tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    for i in range(3):
        (ds / ("a%d.jpg" % i)).write_bytes(b"x")
    C.upsert("hero", name="Hero", trigger="hero", dataset_dir=str(ds))
    tab.refresh()
    assert tab.table.rowCount() == 1
    assert _row_texts(tab, 0)[2] == "3"          # frame count is measured, not stored
    tab.table.selectRow(0)
    assert tab.gen_btn.isEnabled() is False, "offered to draw with no adapter"
    holder = tab.table.cellWidget(0, 4)
    box = holder.findChildren(type(tab.free_chk))[0]
    assert box.isEnabled() is False, "let the user promise Telegram a broken persona"


def test_the_telegram_checkbox_writes_to_the_shared_store(tab, tmp_path):
    lora = tmp_path / "hero.safetensors"
    lora.write_bytes(b"x")
    C.upsert("hero", name="Hero", status="ready", lora_file=str(lora))
    tab.refresh()
    holder = tab.table.cellWidget(0, 4)
    box = holder.findChildren(type(tab.free_chk))[0]
    assert box.isEnabled() is True
    box.setChecked(True)
    # The store, not the widget, is what the bot will read.
    assert [c["slug"] for c in C.telegram_characters()] == ["hero"]
    box.setChecked(False)
    assert C.telegram_characters() == []


def test_status_line_names_what_is_missing(tab):
    """A tab that silently does nothing when Train is pressed is worse than one
    that says the trainer is not installed."""
    import lora_training as LT
    if LT.toolkit_status()["ready"]:
        pytest.skip("trainer is installed on this machine")
    assert "trainer not ready" in tab.status.text()


def test_the_evicted_chat_model_comes_back_on_every_exit(tab, monkeypatch):
    """Restoring only on success leaves a failed run with no model loaded and
    nothing on screen saying why the whole app went mute."""
    import lora_training as LT
    restored = []
    monkeypatch.setattr(LT, "reload_llm", lambda mid, log=None: restored.append(mid))
    for exit_path in ("ok", "bad_code", "raised"):
        tab._evicted_llm = "some-chat-model"
        if exit_path == "ok":
            monkeypatch.setattr(LT, "latest_adapter", lambda slug: None)
            tab._train_done("hero", 0)
        elif exit_path == "bad_code":
            tab._train_done("hero", 1)
        else:
            tab._train_failed("hero", "boom")
    # One per exit path: clean finish, non-zero exit code, and a raise.
    assert restored == ["some-chat-model"] * 3, restored


def test_nothing_is_reloaded_when_nothing_was_evicted(tab, monkeypatch):
    import lora_training as LT
    called = []
    monkeypatch.setattr(LT, "reload_llm", lambda mid, log=None: called.append(mid))
    tab._evicted_llm = ""
    tab._restore_llm()
    assert called == []


def test_a_render_that_cannot_work_says_why_rather_than_starting(tab, tmp_path,
                                                                 monkeypatch):
    """Four ways this render is impossible, and each used to be a silent
    `return`: no character selected, no adapter file, no LLM (the Ideogram
    caption is PLANNED by the model), and the GPU held by a training run --
    which is the likeliest of the four here, since this is the tab the run is
    started from."""
    said = []
    monkeypatch.setattr(tab, "_say", lambda m: said.append(m))

    tab.table.clearSelection()
    tab._generate()
    assert said and "pick a character" in said[-1].lower(), said
    assert tab.worker is None

    ds = tmp_path / "ds2"; ds.mkdir()
    C.upsert("noadapter", name="Без файла", trigger="x", dataset_dir=str(ds))
    tab.refresh(); tab.table.selectRow(0)
    tab._generate()
    assert "adapter" in said[-1], said[-1]
    assert tab.worker is None

    lora = tmp_path / "hero2.safetensors"; lora.write_bytes(b"x")
    C.delete("noadapter")
    C.upsert("hero2", name="Герой", status="ready", trigger="hero2",
             lora_file=str(lora))
    tab.refresh(); tab.table.selectRow(0)
    tab.prompt.setText("на крыше")

    # No model: the caption planner has nothing to plan with.
    tab.host.ctx = None
    tab._generate()
    assert "Settings" in said[-1], said[-1]
    assert tab.worker is None

    import types
    tab.host.ctx = types.SimpleNamespace(model_name="")
    tab._generate()
    assert "Settings" in said[-1], said[-1]
    assert tab.worker is None

    # A model is loaded, but a whole-card job owns the GPU.
    tab.host.ctx = types.SimpleNamespace(model_name="house-model")
    import comfy_client
    monkeypatch.setattr(comfy_client, "gpu_holder", lambda: "Ideogram 4 training")
    tab._generate()
    assert "Ideogram 4 training" in said[-1], said[-1]
    assert tab.worker is None, "started a render on a card held by a trainer"


def test_the_tab_distinguishes_finished_from_killed(tab, monkeypatch):
    """"не активно" covered two very different endings. A run that reached its
    configured step count FINISHED; one that stopped at 1147 of 8000 was killed
    or crashed, and the difference decides whether the next move is to publish
    a checkpoint or to restart the run. The trainer writes no verdict, so it is
    read off the numbers."""
    import lora_training as LT
    C.upsert("hero4", name="Герой", trigger="hero4")
    tab.refresh(); tab.table.selectRow(0)

    def _p(**kw):
        base = {"slug": "hero4", "total": 8000, "step": 0, "checkpoints": 0,
                "samples": 0, "last_sample": "", "tail": "", "eta": "",
                "rate": "", "running": False}
        base.update(kw)
        return base

    def _text(**kw):
        monkeypatch.setattr(LT, "progress", lambda slug: _p(**kw))
        tab._refresh_progress()
        return tab.prog_lbl.text()

    check_live = _text(running=True, step=1147, eta="16:31:52")
    assert "running" in check_live, check_live
    assert "left" in check_live, check_live

    done = _text(step=8000, checkpoints=8)
    assert "finished" in done, done
    assert "stopped" not in done, done

    killed = _text(step=1147, checkpoints=4)
    assert "stopped at 1147 of 8000" in killed, killed
    assert "left" not in killed, killed

    never = _text(step=0)
    assert "never started" in never, never


def test_a_finished_run_is_not_reported_as_live(tab, monkeypatch):
    """The remaining-time figure is the trainer's own, lifted out of its
    progress bar -- and it must not survive the run that produced it. A dead
    run advertising "осталось ~17ч" is a lie the user would wait on."""
    import lora_training as LT
    C.upsert("hero3", name="Герой", trigger="hero3")
    tab.refresh(); tab.table.selectRow(0)
    monkeypatch.setattr(LT, "progress", lambda slug: {
        "slug": slug, "total": 8000, "step": 783, "checkpoints": 3,
        "samples": 0, "last_sample": "", "tail": "", "eta": "17:35:31",
        "rate": "8.78s/it", "running": False})
    tab._refresh_progress()
    text = tab.prog_lbl.text()
    # Not "не активно" any more -- a run stopped at 783 of 8000 was killed, and
    # the label says which ending it was (see the test above). What is pinned
    # here is only that no remaining time is offered for a run that is over.
    assert "stopped" in text, text
    assert "left" not in text, text


def test_the_publish_picker_can_reach_every_run_of_a_character(tab, tmp_path,
                                                               monkeypatch):
    """A character has several runs: the old model one under the bare slug and
    the Ideogram ones under <slug>_ideo / <slug>_ideo32. The picker consulted
    only the bare slug, so the checkpoint actually worth publishing was
    unreachable from the app -- the last publish had to be done by hand from a
    terminal."""
    import lora_training as LT
    from PyQt5.QtWidgets import QInputDialog

    C.upsert("hero", name="Hero", trigger="hero")
    tab.refresh(); tab.table.selectRow(0)

    made = {}
    for run in ("hero_ideo32", "hero_ideo", "hero"):
        d = tmp_path / run
        d.mkdir()
        made[run] = []
        for step in (250, 500):
            f = d / ("%s_%09d.safetensors" % (run, step))
            f.write_bytes(b"x" * 16)
            made[run].append((step, f))

    monkeypatch.setattr(LT, "run_slugs", lambda slug: list(made))
    monkeypatch.setattr(LT, "list_checkpoints",
                        lambda run, limit=10: list(reversed(made.get(run, []))))

    shown = {}
    monkeypatch.setattr(QInputDialog, "getItem",
                        staticmethod(lambda *a, **k: (shown.setdefault("items", a[3])
                                                      and None, ("", False))[1]))
    tab._publish()
    items = shown.get("items") or []
    assert items, "the picker offered nothing"
    assert any("hero_ideo32" in i for i in items), items
    assert any("hero_ideo ·" in i for i in items), items
    # Grouped by run, newest run first -- a global sort by step interleaves
    # them, and since every run counts from zero the list then reads as one
    # sequence with repeats.
    assert items[0].startswith("hero_ideo32"), items[:3]
    # And within a run, newest step first.
    first_run = [i for i in items if i.startswith("hero_ideo32")]
    assert "500" in first_run[0] and "250" in first_run[1], first_run


def test_publish_with_nothing_selected_says_so(tab, monkeypatch):
    said = []
    monkeypatch.setattr(tab, "_say", lambda m: said.append(m))
    tab.table.clearSelection()
    tab._publish()
    assert said and "pick a character" in said[-1].lower(), said
