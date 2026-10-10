"""Characters tab: build a LoRA for a person, and draw with it.

Modeled on MashupTab -- a self-contained QWidget(host) plus QThreads for the
slow parts. Three of them here, because they fail differently: preparing a
dataset is CPU work on the user's photos, training is a child process that
runs for an hour, and generation is a ComfyUI round trip.

Two things this surface deliberately does NOT own:

  * WHETHER A CHARACTER IS OFFERED IN TELEGRAM. The checkbox writes to
    characters.py and nothing else; the bot reads the same store. Keeping a
    copy here is how the two would come to disagree.
  * WHAT "READY" MEANS. The table shows the status characters.py settles
    against the adapter file on disk, so a character whose .safetensors was
    moved shows up as broken here rather than failing at render time.

The training button is honest about a machine where the trainer was never
installed: it says what is missing instead of starting something that cannot
run.
"""
import os

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QDoubleSpinBox,
                             QFileDialog, QHBoxLayout, QHeaderView,
                             QInputDialog, QLabel, QLineEdit, QMessageBox,
                             QPushButton, QSpinBox, QTableWidget,
                             QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget)

import characters as C
import lora_training as LT
from ui_scale import px
from gui_common import MUTED, _section

import logging
logger = logging.getLogger("assistant.gui")   # same channel as gui.py

_COLS = ('Character', 'Trigger', 'Frames', 'Status', 'In Telegram')
_STATUS_RU = {"dataset": 'dataset ready', "training": 'training',
              "ready": 'ready', "failed": 'no adapter'}


class DatasetWorker(QThread):
    """Crop and caption a folder of photos. CPU-bound, minutes on 50 images."""
    line = pyqtSignal(str)
    done = pyqtSignal(int, int, str)      # kept, dropped, out_dir
    failed = pyqtSignal(str)

    def __init__(self, src, out, trigger):
        super().__init__()
        # Snapshotted at construction, never read off the widgets in run():
        # run() is on the worker thread, and touching Qt widgets from there is
        # the unhandled-slot-exception path that aborts the process with no
        # traceback.
        self.src, self.out, self.trigger = str(src), str(out), str(trigger)

    def run(self):
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
            import prep_lora_dataset as P
            kept, dropped, reasons = P.build_dataset(
                self.src, self.out, self.trigger,
                progress=lambda n, total, name: self.line.emit(
                    "  [%d/%d] %s" % (n, total, name)))
            for why, names in reasons.items():
                self.line.emit('dropped (%s): %s' % (why, ", ".join(names)))
            self.done.emit(kept, dropped, self.out)
        except Exception as exc:
            logger.exception("Dataset preparation failed")
            self.failed.emit(str(exc))


class TrainWorker(QThread):
    """Run the trainer as a child process and stream its output.

    Stopping means killing the child. A QThread cannot be interrupted mid-call
    and the work is not in this process anyway, so cancel() terminates the
    process and lets run() fall out of the read loop.
    """
    line = pyqtSignal(str)
    done = pyqtSignal(int)                # exit code
    failed = pyqtSignal(str)

    def __init__(self, cfg_path, slug=""):
        super().__init__()
        self.cfg_path = str(cfg_path)
        self.slug = slug
        self.proc = None
        self._stopped = False

    def run(self):
        try:
            self.proc = LT.spawn(self.cfg_path)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        # Tee to the canonical log as well as the UI: the tab is not the only
        # reader (a run outlives a GUI restart), and progress() follows this
        # file whoever started the training.
        try:
            log = open(LT.log_path(self.slug), "a", encoding="utf-8",
                       errors="replace") if self.slug else None
        except OSError:
            log = None
        try:
            for raw in self.proc.stdout:
                self.line.emit(raw.rstrip())
                if log:
                    log.write(raw)
                    log.flush()
            code = self.proc.wait()
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        finally:
            if log:
                log.close()
        self.done.emit(-1 if self._stopped else code)

    def cancel(self):
        self._stopped = True
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass


class GenerateWorker(QThread):
    """One ComfyUI render with the character's adapter attached."""
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, ctx, prompt, lora_name, strength, trigger=None):
        super().__init__()
        self.ctx, self.prompt = ctx, prompt
        self.lora_name, self.strength = lora_name, float(strength)
        self.trigger = trigger

    def run(self):
        try:
            import image_generate as G
            path = G.generate_image_with_comfy(
                self.ctx, self.prompt, lora_name=self.lora_name,
                lora_strength=self.strength, trigger=self.trigger)
            if not path:
                # "ComfyUI не вернул картинку" is true and useless when the
                # reason is that this tab's OWN training run holds the card --
                # which is exactly when a user reaches for a test render.
                busy = None
                try:
                    import comfy_client as _cc
                    busy = _cc.recent_gpu_refusal()
                except Exception:
                    pass
                self.failed.emit(
                    ('The GPU is busy with «%s» — no drawing until it finishes.' % busy) if busy else
                    'ComfyUI returned no picture — see Log')
                return
            self.done.emit(path)
        except Exception as exc:
            logger.exception("Character render failed")
            self.failed.emit(str(exc))


def _human_eta(raw):
    """Thin alias. The formatter lives in lora_training because the Telegram
    admin panel shows the same figure and two of them would drift."""
    import lora_training as _LT
    return _LT.human_eta(raw)


def _fingerprint(path, chunk: int = 1 << 20):
    """Cheap content identity for an adapter file.

    Same-rank checkpoints are byte-identical in SIZE, so size cannot tell them
    apart; hashing 85 MB ten times to fill a dialog would be worse. The head
    and tail plus the length separate them at 2 MB of reads per file.
    """
    import hashlib
    try:
        size = path.stat().st_size
        h = hashlib.sha256(str(size).encode())
        with open(path, "rb") as fh:
            h.update(fh.read(chunk))
            if size > chunk:
                fh.seek(max(0, size - chunk))
                h.update(fh.read(chunk))
        return h.hexdigest()
    except OSError:
        return None


class CharactersTab(QWidget):
    """The character list, the buttons that change it, and one log."""

    def __init__(self, host):
        super().__init__()
        self.host = host                  # AssistantWindow (for ctx)
        self.worker = None

        root = QVBoxLayout(self)
        root.setContentsMargins(px(6), px(6), px(6), px(6))
        root.setSpacing(px(8))

        root.addWidget(_section('Characters'))
        help_lbl = QLabel(
            "A LoRA adapter for one specific person: photo folder → dataset → training → generation by trigger word. The «In Telegram» box shows the character in the bot's Creativity menu; it clears itself if the adapter file disappears.")
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet("color:%s;" % MUTED)
        root.addWidget(help_lbl)

        self.table = QTableWidget(0, len(_COLS))
        self.table.setHorizontalHeaderLabels(_COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setStretchLastSection(False)
        for c in range(len(_COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)   # the status line is the long one
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_select)
        root.addWidget(self.table, 3)

        row = QHBoxLayout()
        self.new_btn = QPushButton('➕  New character')
        self.new_btn.clicked.connect(self._new_character)
        self.data_btn = QPushButton('🖼  Rebuild dataset')
        self.data_btn.clicked.connect(self._rebuild_dataset)
        self.train_btn = QPushButton('🎓  Train')
        self.train_btn.clicked.connect(self._train)
        self.stop_btn = QPushButton('⏹  Stop')
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        self.publish_btn = QPushButton('📦  Publish adapter')
        self.publish_btn.clicked.connect(self._publish)
        self.del_btn = QPushButton('🗑  Delete')
        self.del_btn.clicked.connect(self._delete)
        for b in (self.new_btn, self.data_btn, self.train_btn, self.stop_btn,
                  self.publish_btn, self.del_btn):
            row.addWidget(b)
        row.addStretch(1)
        root.addLayout(row)

        opts = QHBoxLayout()
        opts.addWidget(QLabel('Steps:'))
        self.steps = QSpinBox(); self.steps.setRange(200, 6000)
        self.steps.setSingleStep(250); self.steps.setValue(LT.DEFAULTS["steps"])
        opts.addWidget(self.steps)
        opts.addWidget(QLabel('Rank:'))
        self.rank = QSpinBox(); self.rank.setRange(4, 128)
        self.rank.setValue(LT.DEFAULTS["rank"])
        opts.addWidget(self.rank)
        self.free_chk = QCheckBox('Unload LM Studio and ComfyUI before training')
        self.free_chk.setChecked(True)
        self.free_chk.setToolTip(
            '24 GB is not enough for everything at once: training next to a loaded model runs out of memory forty minutes in, not right away.')
        opts.addWidget(self.free_chk)
        opts.addStretch(1)
        root.addLayout(opts)

        gen = QHBoxLayout()
        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText(
            'what to draw (the trigger is added for you): in a spacesuit…')
        self.prompt.returnPressed.connect(self._generate)
        gen.addWidget(self.prompt, 1)
        gen.addWidget(QLabel('Strength:'))
        self.strength = QDoubleSpinBox()
        self.strength.setRange(0.1, 1.5); self.strength.setSingleStep(0.05)
        self.strength.setValue(0.9)
        gen.addWidget(self.strength)
        self.gen_btn = QPushButton('🎨  Draw')
        self.gen_btn.clicked.connect(self._generate)
        gen.addWidget(self.gen_btn)
        root.addLayout(gen)

        # Progress is read off the DISK, not from the worker, so a training
        # started outside the app (or before the app was restarted) is still
        # visible here. A tab that goes blank while the GPU is plainly busy is
        # worse than no tab.
        prog = QHBoxLayout()
        self.prog_lbl = QLabel("")
        self.prog_lbl.setWordWrap(True)
        prog.addWidget(self.prog_lbl, 1)
        self.sample_lbl = QLabel()
        self.sample_lbl.setFixedSize(px(96), px(96))
        self.sample_lbl.setScaledContents(True)
        prog.addWidget(self.sample_lbl)
        root.addLayout(prog)

        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setMinimumHeight(px(120))
        root.addWidget(self.log, 1)

        self._poll = QTimer(self)
        self._poll.setInterval(5000)
        self._poll.timeout.connect(self._refresh_progress)
        self._poll.start()

        self.status = QLabel("")
        self.status.setStyleSheet("color:%s;" % MUTED)
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        self.refresh()

    # ---- data -----------------------------------------------------------
    def refresh(self):
        rows = C.list_characters()
        self._rows = rows
        self.table.setRowCount(len(rows))
        for i, rec in enumerate(rows):
            n = LT.dataset_size(rec.get("dataset_dir") or "")
            vals = [rec.get("name", ""), rec.get("trigger", ""), str(n),
                    _STATUS_RU.get(rec.get("status"), rec.get("status", ""))]
            for j, v in enumerate(vals):
                self.table.setItem(i, j, QTableWidgetItem(v))
            box = QCheckBox()
            box.setChecked(bool(rec.get("tg_enabled")))
            # The checkbox is only meaningful once an adapter exists; leaving
            # it clickable would let someone tick a character the bot will
            # then refuse to draw, which reads as a bot bug.
            box.setEnabled(bool(rec.get("lora_exists")))
            box.stateChanged.connect(
                lambda st, slug=rec["slug"]: self._toggle_tg(slug, st))
            holder = QWidget(); h = QHBoxLayout(holder)
            h.setContentsMargins(0, 0, 0, 0); h.setAlignment(Qt.AlignCenter)
            h.addWidget(box)
            self.table.setCellWidget(i, 4, holder)
        st = LT.toolkit_status()
        free = LT.gpu_free_mb()
        bits = []
        if not st["ready"]:
            bits.append('trainer not ready: ' + "; ".join(st["missing"]))
        if free is not None:
            bits.append('free VRAM: %d MB' % free)
        self.status.setText(" · ".join(bits))
        self._on_select()

    def _refresh_progress(self):
        rec = self._selected()
        if not rec:
            self.prog_lbl.setText("")
            self.sample_lbl.clear()
            return
        p = LT.progress(rec["slug"])
        # The run's OWN step count, not the spin box: the widget still holds
        # whatever was last typed, which is how a run extended to 4250 reported
        # "шаг 2250 из 2000".
        total = p.get("total") or self.steps.value()
        # "не активно" covered two very different endings. A run that reached
        # its configured step count FINISHED; one that stopped at 1147 of 8000
        # was killed or crashed, and the difference decides whether the right
        # next move is to publish a checkpoint or to restart the run. The
        # trainer writes no verdict, so it is read off the numbers.
        kind = LT.run_state(dict(p, total=total))
        state = {LT.RUN_RUNNING: 'running',
                 LT.RUN_NEVER: 'never started',
                 LT.RUN_DONE: 'finished'}.get(
                     kind, 'stopped at %d of %d' % (p["step"], total))
        bits = ['training: %s' % state]
        # A character can have several runs (the old model under the bare slug, the
        # Ideogram ones under <slug>_ideo*). Name the one these numbers describe
        # -- silently reporting on a different run than the one training is how
        # a live rank-32 job showed as "не активно".
        if p.get("slug") and p["slug"] != rec["slug"]:
            bits.append('run: %s' % p["slug"])
        if p["step"] and kind != LT.RUN_STOPPED:
            bits.append('step %d of %d' % (p["step"], total))
            if total:
                bits.append("%.0f%%" % (100.0 * p["step"] / total))
        # The trainer's own remaining-time figure, lifted out of the raw bar.
        # An 8000-step run is most of a day, and "шаг 787 из 8000" alone does
        # not answer the only question anyone asks about it.
        if p.get("eta") and p["running"]:
            bits.append('~%s left' % _human_eta(p["eta"]))
        if p.get("rate"):
            bits.append(p["rate"])
        bits.append('checkpoints: %d' % p["checkpoints"])
        if p["samples"]:
            bits.append('samples: %d' % p["samples"])
        if p["tail"]:
            bits.append(p["tail"])
        self.prog_lbl.setText(" · ".join(bits))
        if p["last_sample"] and p["last_sample"] != getattr(self, "_shown_sample", ""):
            pix = QPixmap(p["last_sample"])
            if not pix.isNull():
                self.sample_lbl.setPixmap(pix)
                self._shown_sample = p["last_sample"]

    def _selected(self):
        i = self.table.currentRow()
        if 0 <= i < len(getattr(self, "_rows", [])):
            return self._rows[i]
        return None

    def _on_select(self):
        rec = self._selected()
        busy = self.worker is not None and self.worker.isRunning()
        for b in (self.data_btn, self.train_btn, self.del_btn, self.publish_btn):
            b.setEnabled(rec is not None and not busy)
        self.gen_btn.setEnabled(
            rec is not None and bool(rec.get("lora_exists")) and not busy)
        self.new_btn.setEnabled(not busy)

    def _toggle_tg(self, slug, state):
        """Write the Telegram flag straight to the shared store.

        No local copy is kept: the bot reads characters.py, and a second copy
        here is exactly how the checkbox and the bot would come to disagree.
        """
        C.set_tg_enabled(slug, state == Qt.Checked)
        rec = C.get(slug) or {}
        self._say('%s %s in Telegram' % (rec.get("name", slug),
                                        'available' if rec.get("tg_available")
                                        else 'hidden'))

    def _say(self, text):
        self.log.append(text)

    # ---- actions --------------------------------------------------------
    def _new_character(self):
        name, ok = QInputDialog.getText(self, 'New character', 'Name:')
        if not ok or not name.strip():
            return
        slug = C.slugify(name)
        if C.get(slug):
            QMessageBox.warning(self, 'Characters',
                                'This character already exists: ' + slug)
            return
        folder = QFileDialog.getExistingDirectory(self, 'Photo folder')
        if not folder:
            return
        out = LT.DATASET_ROOT / slug
        C.upsert(slug, name=name.strip(), trigger=slug, status="dataset",
                 dataset_dir=str(out))
        self.refresh()
        self._start_dataset(folder, out, slug)

    def _rebuild_dataset(self):
        rec = self._selected()
        if not rec:
            return
        folder = QFileDialog.getExistingDirectory(self, 'Photo folder')
        if not folder:
            return
        self._start_dataset(folder, rec.get("dataset_dir")
                            or str(LT.DATASET_ROOT / rec["slug"]),
                            rec.get("trigger") or rec["slug"])

    def _start_dataset(self, folder, out, trigger):
        self._say('Preparing the dataset from %s' % folder)
        w = DatasetWorker(folder, out, trigger)
        w.line.connect(self._say)
        w.done.connect(self._dataset_done)
        w.failed.connect(lambda m: (self._say('Error: ' + m), self._finish()))
        self.worker = w
        self._on_select()
        w.start()

    def _dataset_done(self, kept, dropped, out):
        self._say('Done: kept %d, dropped %d → %s' % (kept, dropped, out))
        self._say('Every frame has the same caption — edit the .txt next to each photo, or the trigger learns the average of the set.')
        self._finish()

    def _train(self):
        rec = self._selected()
        if not rec:
            self._say('Pick a character in the list first.')
            return
        st = LT.toolkit_status()
        if not st["ready"]:
            QMessageBox.warning(self, 'Training',
                                'Cannot start:\n\n• ' + "\n• ".join(st["missing"]))
            return
        n = LT.dataset_size(rec.get("dataset_dir") or "")
        if n < 10:
            QMessageBox.warning(self, 'Training',
                                'The dataset has %d frames — too few. Collect at least 20.' % n)
            return
        if self.free_chk.isChecked():
            # Ask WHICH model is loaded before evicting it: afterwards nothing
            # is, and "restore what was running" would degrade into "load the
            # config default", which is a different model than the user chose.
            self._evicted_llm = LT.loaded_llm_id()
            if self._evicted_llm:
                self._say('Taking the GPU from %s — it comes back after training'
                          % self._evicted_llm)
            LT.free_gpu(self._say)
        free = LT.gpu_free_mb()
        if free is not None and free < 18000:
            QMessageBox.warning(
                self, 'Training',
                'Only %d MB of VRAM is free. Something else holds the GPU — training will run out of memory tens of minutes in, not right away.' % free)
            return
        # Checked beside the VRAM check and for the same reason: a run that runs
        # out of DISK does not fail loudly either. It dies inside a checkpoint
        # write, hours in, leaving a partial .safetensors that the resume path
        # will pick up as the newest save.
        low_disk = LT.disk_warning()
        if low_disk and QMessageBox.question(
                self, 'Training', low_disk + chr(10) + chr(10) + 'Start anyway?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No) != QMessageBox.Yes:
            return
        cfg = LT.build_config(rec["slug"], rec.get("trigger") or rec["slug"],
                              rec["dataset_dir"], steps=self.steps.value(),
                              rank=self.rank.value())
        C.upsert(rec["slug"], status="training")
        self.refresh()
        self._say('Starting training: %s' % cfg)
        w = TrainWorker(cfg, rec["slug"])
        w.line.connect(self._say)
        w.done.connect(lambda code, slug=rec["slug"]: self._train_done(slug, code))
        w.failed.connect(lambda m, slug=rec["slug"]: self._train_failed(slug, m))
        self.worker = w
        self.stop_btn.setEnabled(True)
        self._on_select()
        w.start()

    def _train_done(self, slug, code):
        adapter = LT.latest_adapter(slug)
        if code == 0 and adapter:
            C.upsert(slug, status="ready", lora_file=str(adapter))
            self._say('Training finished. Adapter: %s' % adapter)
            self._say('Checkpoints at 1500-1750 steps often look more alike than the final one — pick one with «Publish».')
            self._restore_llm()
        else:
            C.upsert(slug, status="failed",
                     note='exit code %s' % code)
            self._say('Training produced no adapter (code %s)' % code)
        self._restore_llm()
        self._finish()

    def _train_failed(self, slug, msg):
        C.upsert(slug, status="failed", note=msg)
        self._say('Training error: ' + msg)
        self._restore_llm()
        self._finish()

    def _stop(self):
        if isinstance(self.worker, TrainWorker):
            self._say('Stopping training…')
            self.worker.cancel()

    def _publish(self):
        """Pick which checkpoint becomes the character's adapter.

        Used to be a bare file dialog opened on the run folder, which asked the
        user to recognise `neurostepan_000002250.safetensors` among ten
        near-identical names. The step number is the ONLY thing you know after
        comparing renders, so the list offers steps, newest first, and says
        which one is currently published.
        """
        rec = self._selected()
        if not rec:
            self._say('Pick a character in the list first.')
            return
        slug = rec["slug"]
        # EVERY run of this character, not just the one named after them. The
        # Ideogram runs live under <slug>_ideo / <slug>_ideo32, and with only
        # the bare slug consulted this picker could not reach the checkpoint
        # that was actually worth publishing -- which is why the last publish
        # had to be done by hand from a terminal.
        # run_slugs is newest-touched first. Keep that order and sort by step
        # only WITHIN a run: a global sort by step interleaves the runs, and
        # since every run counts from zero the list then reads as one sequence
        # with repeats, which is worse than no grouping at all.
        runs = [r for r in (LT.run_slugs(slug) or [slug])][:3]
        rows, run_of = [], {}
        for run in runs:
            got = [(st, p) for st, p in LT.list_checkpoints(run, limit=10)
                   if st is not None]
            got.sort(key=lambda r: r[0], reverse=True)
            for step, path in got:
                rows.append((step, path))
                run_of[str(path)] = run
        if not rows:
            QMessageBox.information(
                self, 'Characters',
                '%s has no saved checkpoints — train it first.' % slug)
            return
        multi = len(set(run_of.values())) > 1

        # Which one is live. NOT by file size: every checkpoint of a run has
        # the same rank and therefore the same size to the byte, so a size test
        # marked seven of ten as "currently published". The source path is
        # recorded at publish time; for adapters published before that existed,
        # fall back to a content fingerprint.
        published = LT.LORA_OUT_DIR / (slug + ".safetensors")
        known = rec.get("lora_source") or ""
        live_fp = None if known else _fingerprint(published)
        labels, paths = [], []
        for step, path in rows:
            try:
                mb = path.stat().st_size / 1024 ** 2
            except OSError:
                continue
            same = (os.path.normcase(str(path)) == os.path.normcase(known)
                    if known else
                    live_fp is not None and _fingerprint(path) == live_fp)
            labels.append('%sstep %-5d   %.0f MB%s'
                          % (("%s · " % run_of.get(str(path), "")) if multi else "",
                             step or 0, mb,
                             '   ← published now' if same else ""))
            paths.append(path)
        labels.append('Choose a file manually…')

        choice, ok = QInputDialog.getItem(
            self, 'Which checkpoint to publish',
            'Adapter for «%s» (last %d saves):'
            % (rec.get("name") or slug, len(paths)),
            labels, 0, False)
        if not ok:
            return
        if choice == labels[-1]:
            picked, _ = QFileDialog.getOpenFileName(
                self, 'Which checkpoint to publish',
                str(LT.TRAIN_ROOT), "LoRA (*.safetensors)")
            if not picked:
                return
        else:
            picked = str(paths[labels.index(choice)])

        try:
            dest = LT.publish(slug, picked)
        except Exception as exc:
            logger.exception('publishing the adapter failed')
            QMessageBox.warning(self, 'Characters',
                                'Could not publish: %s' % exc)
            return
        C.upsert(slug, status="ready", lora_file=str(dest),
                 lora_source=str(picked))
        self._say('Published: %s → %s' % (os.path.basename(picked), dest))
        self.refresh()

    def _delete(self):
        rec = self._selected()
        if not rec:
            return
        ans = QMessageBox.question(
            self, 'Delete character',
            'Remove «%s» from the list?\n\nThe photos and the adapter file stay on disk — only the entry is removed.' % rec.get("name"),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ans == QMessageBox.Yes:
            C.delete(rec["slug"])
            self.refresh()

    def _generate(self):
        rec = self._selected()
        if not rec:
            self._say('Pick a character in the list first.')
            return
        if not rec.get("lora_exists"):
            # Silent before: the row is there, the button is live, and nothing
            # happens. The adapter file is usually what is missing.
            self._say('«%s» has no adapter file — training is not finished or the file was deleted.' % (rec.get("name") or rec["slug"]))
            return
        # The caption for an Ideogram render is PLANNED by the LLM, so a render
        # started without one spends its time and comes back with nothing.
        ctx = getattr(self.host, "ctx", None)
        if ctx is None or not (getattr(ctx, "model_name", "") or "").strip():
            self._say('No model is loaded — pick one in ⚙ Settings → Model Config → Apply. Without it nothing can write the picture caption.')
            return
        busy = None
        try:
            import comfy_client as _cc
            busy = _cc.gpu_holder()
        except Exception:
            pass
        if busy:
            # Said BEFORE the render rather than after it fails: this tab is
            # where a training run is started, so its own run is the most
            # likely holder and the user is a click away from waiting for
            # nothing.
            self._say('The GPU is busy with «%s» — no drawing until it finishes.' % busy)
            return
        import image_lora
        text = image_lora.prompt_with_trigger(self.prompt.text(),
                                              rec.get("trigger") or rec["slug"])
        if not text.strip():
            self._say('Write what to draw.')
            return
        lora_file = os.path.basename(rec.get("lora_file") or "")
        self._say('Drawing: ' + text)
        w = GenerateWorker(ctx, text, lora_file,
                           self.strength.value(),
                           rec.get("trigger") or rec["slug"])
        w.done.connect(self._render_done)
        w.failed.connect(lambda m: (self._say('Error: ' + m), self._finish()))
        self.worker = w
        self._on_select()
        w.start()

    def _render_done(self, path):
        self._say('Done: ' + path)
        panel = getattr(self.host, "images_panel", None)
        if panel is not None and hasattr(panel, "add_image"):
            try:
                panel.add_image(path)
            except Exception:
                logger.exception("could not show the render in the Images panel")
        self._finish()

    def _restore_llm(self):
        """Give the card back to the chat model the training took it from.

        Runs on EVERY exit from a training run -- success, failure and Stop
        alike. Restoring only on success is how a failed run leaves the whole
        app mute with no model loaded and nothing saying why.
        """
        mid = getattr(self, "_evicted_llm", "")
        if not mid:
            return
        self._evicted_llm = ""
        self._say('Bringing the model back %s…' % mid)
        LT.reload_llm(mid, self._say)

    def _finish(self):
        self.worker = None
        self.stop_btn.setEnabled(False)
        self.refresh()
