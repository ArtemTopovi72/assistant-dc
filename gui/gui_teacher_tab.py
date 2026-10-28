"""A private language lesson, with cancellable speech and no shared chat history."""
import copy
import shutil
import threading
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

import gui_i18n
import translator
from gui_common import _ScopedCtx, _section
from seams import file_dialog, module
from ui_scale import px


class TeacherWorker(QThread):
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)

    def __init__(self, ctx, profile, history, text, target, native, source=''):
        super().__init__()
        self.cancel_event = threading.Event()
        self.ctx = _ScopedCtx(ctx, self.cancel_event)
        object.__setattr__(self.ctx, 'set_progress', self.progress.emit)
        self.profile, self.history = copy.deepcopy(profile), copy.deepcopy(history)
        self.text, self.target, self.native = text, target, native
        self.source = source
        self.voice_ref = module(ctx, 'translator').voice_reference(ctx)

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        from config import OUTPUT_DIR
        try:
            if self.cancel_event.is_set(): return
            if self.source:
                self.text = module(self.ctx, 'translator').transcribe(self.ctx, self.source)
            if self.cancel_event.is_set(): return
            result = module(self.ctx, 'language_teacher').respond(self.ctx, self.profile,
                self.history, self.text, self.target, self.native)
            if self.cancel_event.is_set(): return
            try:
                speech = module(self.ctx, 'translator').lesson_speech(self.ctx, result['say'],
                    self.target, result['translation'], self.native,
                    str(Path(OUTPUT_DIR) / 'lessons' / 'desktop'), reference=self.voice_ref)
                result.update(speech)
            except Exception as exc:
                result.update(paths=[], combined_path='', warning=str(exc))
            if not self.cancel_event.is_set():
                result['learner_text'] = self.text
                self.done.emit(result)
        except Exception as exc:
            if not self.cancel_event.is_set(): self.failed.emit(str(exc))


class TeacherTab(QWidget):
    def __init__(self, host=None):
        super().__init__()
        self.host, self.worker, self.player = host, None, None
        self.profile = {'level': 'auto', 'auto_level': 'A1', 'topic': '', 'streak': 0}
        self.history, self.paths, self.combined_path = [], [], ''
        self.source = ''
        root = QVBoxLayout(self)
        root.setContentsMargins(px(12), px(12), px(12), px(12)); root.setSpacing(px(10))
        root.addWidget(_section('Language teacher'))
        hint = QLabel('Write in your native language or practise the language you are learning. Replies include an exact translation and gentle corrections. Voice uses the main settings voice.')
        hint.setWordWrap(True); root.addWidget(hint)
        row = QHBoxLayout()
        self.target, self.native, self.level = QComboBox(), QComboBox(), QComboBox()
        for widget, label in ((self.target, 'Learning language'), (self.native, 'Native language')):
            widget.setAccessibleName(label)
            for code, (english, native) in translator.LANGUAGES.items():
                widget.addItem(f'{native} · {english}', code)
            row.addWidget(QLabel(label)); row.addWidget(widget)
        self.target.setCurrentIndex(self.target.findData('de'))
        self.native.setCurrentIndex(self.native.findData('ru'))
        self.level.setAccessibleName('Lesson level')
        for value in ('auto', 'A1', 'A2', 'B1', 'B2', 'C1'):
            self.level.addItem('Auto' if value == 'auto' else value, value)
        row.addWidget(QLabel('Lesson level')); row.addWidget(self.level); root.addLayout(row)
        row = QHBoxLayout()
        self.topic = QLineEdit(); self.topic.setMaxLength(300)
        self.topic.setAccessibleName('Lesson topic'); self.topic.setPlaceholderText('For example: travel or daily life')
        self.voice_format = QComboBox(); self.voice_format.setAccessibleName('Voice format')
        self.voice_format.addItem('Combined audio', 'combined')
        self.voice_format.addItem('Separate audio', 'separate')
        row.addWidget(QLabel('Lesson topic')); row.addWidget(self.topic, 1)
        row.addWidget(self.voice_format); root.addLayout(row)
        self.level_status = QLabel(); root.addWidget(self.level_status)
        self.text_in = QPlainTextEdit(); self.text_in.setMaximumHeight(px(140))
        self.text_in.setPlaceholderText('Write your lesson message…'); root.addWidget(self.text_in)
        row = QHBoxLayout()
        self.audio_btn = QPushButton('Choose lesson audio…')
        self.remove_audio_btn = QPushButton('Remove audio')
        self.audio_label = QLabel('No file selected'); self.audio_label.setWordWrap(True)
        self.audio_btn.clicked.connect(self._choose_audio)
        self.remove_audio_btn.clicked.connect(self._remove_audio)
        for widget in (self.audio_btn, self.remove_audio_btn, self.audio_label): row.addWidget(widget)
        root.addLayout(row)
        row = QHBoxLayout()
        self.send_btn, self.stop_btn, self.clear_btn = (QPushButton('Send lesson message'),
            QPushButton('Cancel'), QPushButton('Reset lesson'))
        self.send_btn.clicked.connect(self._send); self.stop_btn.clicked.connect(self._cancel)
        self.clear_btn.clicked.connect(self._clear)
        for button in (self.send_btn, self.stop_btn, self.clear_btn): row.addWidget(button)
        root.addLayout(row)
        self.status = QLabel(); self.status.setWordWrap(True); root.addWidget(self.status)
        self.result = QPlainTextEdit(); self.result.setReadOnly(True)
        self.result.setPlaceholderText('Lesson dialogue'); root.addWidget(self.result)
        row = QHBoxLayout()
        self.play_btn, self.save_btn = QPushButton('▶ Play voice'), QPushButton('Save result…')
        self.play_btn.clicked.connect(self._play); self.save_btn.clicked.connect(self._save)
        row.addWidget(self.play_btn); row.addWidget(self.save_btn); root.addLayout(row)
        self.target.currentIndexChanged.connect(lambda: self._settings_changed(True))
        self.native.currentIndexChanged.connect(self._settings_changed)
        self.topic.textChanged.connect(self._settings_changed)
        self.level.currentIndexChanged.connect(self._level_changed)
        self._show_level(); self._busy(False)

    def _show_level(self):
        self.level_status.setText(gui_i18n.tr('Current level: ') + (
            self.profile['auto_level'] if self.profile['level'] == 'auto' else self.profile['level']))

    def _level_changed(self):
        self.profile['level'], self.profile['streak'] = self.level.currentData(), 0
        self._show_level()

    def _settings_changed(self, reset_level=False):
        # Qt sends index/text values; only the explicit target callback resets proficiency.
        if reset_level is True: self.profile['auto_level'] = 'A1'
        self._clear(reset_level=False)
        self.profile['topic'] = self.topic.text()

    def _busy(self, busy):
        for widget in (self.target, self.native, self.level, self.topic, self.voice_format,
                       self.text_in, self.send_btn, self.clear_btn):
            widget.setEnabled(not busy)
        self.text_in.setEnabled(not busy and not self.source)
        self.audio_btn.setEnabled(not busy)
        self.remove_audio_btn.setEnabled(not busy and bool(self.source))
        self.stop_btn.setEnabled(busy)
        self.play_btn.setEnabled(not busy and bool(self.combined_path))
        self.save_btn.setEnabled(not busy and bool(self.result.toPlainText()))

    def _choose_audio(self):
        path, _ = file_dialog(self).getOpenFileName(self, 'Choose lesson audio…', '',
            'Audio (*.wav *.mp3 *.ogg *.m4a *.flac *.opus)')
        if path:
            self.source = path
            self.audio_label.setText(Path(path).name)
            self._busy(False)

    def _remove_audio(self):
        self.source = ''
        self.audio_label.setText(gui_i18n.tr('No file selected'))
        self._busy(False)

    def _send(self):
        if self.worker and self.worker.isRunning(): return
        ctx = getattr(self.host, 'ctx', None)
        if ctx is None:
            self.status.setText('The model is not ready yet.'); return
        text = self.text_in.toPlainText()
        if not self.source and (not text.strip() or len(text) > 12000):
            self.status.setText('Send 1–12000 characters of lesson text.'); return
        if self.player: self.player.stop()
        try:
            self.worker = TeacherWorker(ctx, self.profile, self.history, text,
                self.target.currentData(), self.native.currentData(), self.source)
        except Exception as exc:
            self.status.setText(str(exc)); return
        self._busy(True); self.status.setText('Preparing lesson…')
        self.worker.done.connect(self._done)
        self.worker.failed.connect(lambda error: self.status.setText(translator.error_text(error, gui_i18n.LANG)))
        self.worker.progress.connect(self._progress)
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def _progress(self, done, total, what):
        import progress
        self.status.setText(progress.line(done, total, what, gui_i18n.LANG))

    def _done(self, result):
        if self.worker and self.worker.cancel_event.is_set(): return
        self.profile, self.history = result['profile'], result['history']
        self.paths, self.combined_path = result.get('paths', []), result.get('combined_path', '')
        self.result.appendPlainText('\n'.join(part for part in (result['learner_text'], result['say'],
            result['translation'], result.get('correction', ''), result.get('explanation', '')) if part))
        self.result.appendPlainText('')
        self.text_in.clear(); self._show_level()
        self.source = ''
        self.audio_label.setText(gui_i18n.tr('No file selected'))
        self.status.setText(translator.error_text(result['warning'], gui_i18n.LANG)
            if result.get('warning') else gui_i18n.tr('Done.'))

    def _finished(self):
        self._busy(False)

    def _cancel(self):
        if self.worker: self.worker.cancel()
        if self.player: self.player.stop()
        self.status.setText('Cancelled')

    def _clear(self, checked=False, *, reset_level=True):
        if self.worker and self.worker.isRunning(): return
        if self.player: self.player.stop()
        self.history, self.paths, self.combined_path = [], [], ''
        self.source = ''
        self.audio_label.setText(gui_i18n.tr('No file selected'))
        self.profile['streak'] = 0
        if reset_level: self.profile['auto_level'] = 'A1'
        self.text_in.clear(); self.result.clear(); self.status.clear()
        self._show_level(); self._busy(False)

    def _play(self):
        if not self.combined_path or not self.play_btn.isEnabled(): return
        try:
            if self.player is None: self.player = module(self, 'audio').AudioPlayer()
            self.player.play(self.combined_path)
        except Exception as exc:
            self.status.setText(str(exc))

    def _save(self):
        if self.worker and self.worker.isRunning(): return
        sources = self.paths if self.voice_format.currentData() == 'separate' else [self.combined_path]
        sources = [path for path in sources if path]
        if not sources:
            if not self.result.toPlainText(): return
            path, _ = file_dialog(self).getSaveFileName(self, 'Save lesson text', 'lesson.txt', 'Text (*.txt)')
            if path:
                try: Path(path).write_text(self.result.toPlainText(), 'utf8')
                except OSError as exc: self.status.setText(str(exc))
            return
        for source in sources:
            path, _ = file_dialog(self).getSaveFileName(self, 'Save lesson audio', Path(source).name, 'Audio (*.wav)')
            if not path: break
            try:
                if Path(path).resolve() != Path(source).resolve(): shutil.copyfile(source, path)
            except OSError as exc:
                self.status.setText(str(exc)); break

    def shutdown(self):
        self._cancel()
