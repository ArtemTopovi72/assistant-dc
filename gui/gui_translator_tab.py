"""Translation utility with its own cancellable worker and explicit inputs."""
import shutil
import threading
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

import gui_i18n
import translator
from gui_common import _ScopedCtx, _section
from seams import file_dialog, module
from ui_scale import px


class TranslationWorker(QThread):
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, ctx, mode, target, text, source):
        super().__init__()
        self.cancel_event = threading.Event()
        self.ctx = _ScopedCtx(ctx, self.cancel_event)
        self.mode, self.target, self.text, self.source = mode, target, text, source

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        from config import OUTPUT_DIR
        try:
            engine = module(self.ctx, 'translator')
            out = str(Path(OUTPUT_DIR) / 'translations' / 'desktop')
            if self.mode == 'image':
                result = engine.image(self.ctx, self.source, self.target, out)
            elif self.mode == 'voice':
                result = engine.voice(self.ctx, self.source, self.target, out)
            else:
                result = {'text': engine.translate(self.ctx, self.text, self.target), 'path': ''}
            if not self.cancel_event.is_set():
                self.done.emit(result)
        except Exception as exc:
            if not self.cancel_event.is_set():
                self.failed.emit(str(exc))


class TranslatorTab(QWidget):
    def __init__(self, host=None):
        super().__init__()
        self.host, self.worker, self.player = host, None, None
        self.source, self.output_path = '', ''
        root = QVBoxLayout(self)
        root.setContentsMargins(px(12), px(12), px(12), px(12))
        root.setSpacing(px(10))
        root.addWidget(_section('Translator'))
        hint = QLabel('Source language is detected automatically. Picture lettering is overlaid without image generation.')
        hint.setWordWrap(True); root.addWidget(hint)
        options = QHBoxLayout()
        self.target, self.mode = QComboBox(), QComboBox()
        self.target.setAccessibleName('Target language')
        self.mode.setAccessibleName('Translation mode')
        for code, (en, native) in translator.LANGUAGES.items():
            self.target.addItem(f'{native} · {en}', code)
        self.target.setCurrentIndex(self.target.findData('en'))
        for label, mode in (('Text', 'text'), ('Picture lettering', 'image'), ('Voice → voice', 'voice')):
            self.mode.addItem(label, mode)
        options.addWidget(QLabel('Translate into')); options.addWidget(self.target)
        options.addWidget(self.mode); root.addLayout(options)
        self.text_in = QPlainTextEdit()
        self.text_in.setPlaceholderText('Paste or type text to translate…')
        self.text_in.setMaximumHeight(px(180)); root.addWidget(self.text_in)
        files = QHBoxLayout()
        self.file_btn = QPushButton('📎 Choose picture or audio')
        self.file_btn.clicked.connect(self._choose_file); files.addWidget(self.file_btn)
        self.file_label = QLabel('No file selected')
        self.file_label.setWordWrap(True); files.addWidget(self.file_label, 1)
        root.addLayout(files)
        hint = QLabel('Pictures: a local background patch covers the old letters. Voice: send 3–60 seconds of clear speech; the reply uses this recording as the voice sample.')
        hint.setWordWrap(True); root.addWidget(hint)
        buttons = QHBoxLayout()
        self.translate_btn = QPushButton('🌐 Translate')
        self.translate_btn.clicked.connect(self._translate)
        self.stop_btn = QPushButton('Cancel'); self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._cancel)
        self.clear_btn = QPushButton('Clear')
        self.clear_btn.clicked.connect(self._clear)
        for b in (self.translate_btn, self.stop_btn, self.clear_btn): buttons.addWidget(b)
        root.addLayout(buttons)
        self.status = QLabel(''); self.status.setWordWrap(True); root.addWidget(self.status)
        self.result = QPlainTextEdit(); self.result.setReadOnly(True)
        self.result.setPlaceholderText('Translation'); root.addWidget(self.result)
        self.preview = QLabel(); self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(0); self.preview.hide(); root.addWidget(self.preview)
        actions = QHBoxLayout()
        self.copy_btn = QPushButton('Copy translation'); self.copy_btn.clicked.connect(self._copy)
        self.save_btn = QPushButton('Save result…'); self.save_btn.clicked.connect(self._save)
        self.play_btn = QPushButton('▶ Play voice'); self.play_btn.clicked.connect(self._play)
        self.play_btn.setEnabled(False)
        for b in (self.copy_btn, self.save_btn, self.play_btn): actions.addWidget(b)
        root.addLayout(actions)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self._mode_changed()

    def _mode_changed(self):
        self.text_in.setVisible(self.mode.currentData() == 'text')

    def _choose_file(self):
        path, _ = file_dialog(self).getOpenFileName(self, 'Choose picture or audio', '',
            'Pictures and audio (*.png *.jpg *.jpeg *.webp *.wav *.mp3 *.ogg *.m4a *.flac *.opus);;All files (*)')
        if path:
            self.source = path; self.file_label.setText(Path(path).name)
            mode = 'image' if Path(path).suffix.lower() in ('.png','.jpg','.jpeg','.webp') else 'voice'
            self.mode.setCurrentIndex(self.mode.findData(mode))

    def _busy(self, busy):
        for w in (self.translate_btn, self.target, self.mode, self.file_btn, self.clear_btn):
            w.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)

    def _translate(self):
        if self.worker and self.worker.isRunning(): return
        if self.player: self.player.stop()
        ctx = getattr(self.host, 'ctx', None)
        if ctx is None:
            self.status.setText('The model is not ready yet.'); return
        mode = self.mode.currentData()
        if mode != 'text' and not self.source:
            self.status.setText('Choose a picture or audio file first.'); return
        if mode == 'image' and Path(self.source).suffix.lower() not in ('.png','.jpg','.jpeg','.webp'):
            self.status.setText('Choose a picture or audio file first.'); return
        self.output_path = ''; self.preview.clear(); self.preview.hide()
        self.play_btn.setEnabled(False); self.result.clear()
        self._busy(True); self.status.setText('Translating…')
        self.worker = TranslationWorker(ctx, mode, self.target.currentData(), self.text_in.toPlainText(), self.source)
        self.worker.done.connect(self._done)
        self.worker.failed.connect(lambda error: self.status.setText(translator.error_text(error, gui_i18n.LANG)))
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def _done(self, result):
        self.result.setPlainText(result['text'])
        self.output_path = result.get('path', '')
        self.status.setText(translator.error_text(result['warning'], gui_i18n.LANG) if result.get('warning') else gui_i18n.tr('Done.'))
        if self.output_path and Path(self.output_path).suffix == '.png':
            self.preview.setPixmap(QPixmap(self.output_path).scaled(px(720), px(360), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.preview.show()
        self.play_btn.setEnabled(bool(self.output_path) and Path(self.output_path).suffix == '.wav')

    def _finished(self):
        self._busy(False)

    def _cancel(self):
        if self.worker: self.worker.cancel()
        if self.player: self.player.stop()
        self.status.setText('Cancelled')

    def _clear(self):
        if self.player: self.player.stop()
        self.source = self.output_path = ''
        self.text_in.clear(); self.result.clear(); self.preview.clear(); self.preview.hide()
        self.file_label.setText('No file selected'); self.status.clear(); self.play_btn.setEnabled(False)

    def _copy(self):
        QApplication.clipboard().setText(self.result.toPlainText())

    def shutdown(self):
        self._cancel()

    def _save(self):
        if not self.result.toPlainText(): return
        ext = Path(self.output_path).suffix if self.output_path else '.txt'
        path, _ = file_dialog(self).getSaveFileName(self, 'Save result', 'translation' + ext, f'Result (*{ext})')
        if path:
            try:
                if self.output_path:
                    if Path(path).resolve() != Path(self.output_path).resolve():
                        shutil.copyfile(self.output_path, path)
                else:
                    Path(path).write_text(self.result.toPlainText(), 'utf8')
            except OSError as exc:
                self.status.setText(str(exc))

    def _play(self):
        if self.output_path and self.play_btn.isEnabled():
            try:
                if self.player is None:
                    from audio import AudioPlayer
                    self.player = AudioPlayer()
                self.player.play(self.output_path)
            except Exception as exc:
                self.status.setText(str(exc))
