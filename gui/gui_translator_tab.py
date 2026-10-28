"""Translation utility with its own cancellable worker and explicit inputs."""
import shutil
import threading
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
    QPlainTextEdit, QPushButton, QTabWidget, QVBoxLayout, QWidget)

import gui_i18n
import translator
from gui_common import _ScopedCtx, _section
from seams import file_dialog, module
from ui_scale import px


class TranslationWorker(QThread):
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int,int,str)

    def __init__(self, ctx, mode, target, text, source):
        super().__init__()
        self.cancel_event = threading.Event()
        self.ctx = _ScopedCtx(ctx, self.cancel_event)
        object.__setattr__(self.ctx, 'set_progress', self.progress.emit)
        self.mode, self.target, self.text, self.source = mode, target, text, source
        self.voice_ref=translator.voice_reference(ctx) if mode=='tts' else ''

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        from config import OUTPUT_DIR
        try:
            engine = module(self.ctx, 'translator')
            out = str(Path(OUTPUT_DIR) / 'translations' / 'desktop')
            if self.mode == 'image':
                result = engine.image(self.ctx, self.source, self.target, out)
            elif self.mode == 'dub':
                result = engine.dub(self.ctx, self.source, self.target, out)
            elif self.mode == 'voice':
                result = engine.voice(self.ctx, self.source, self.target, out)
            elif self.mode == 'tts':
                result = engine.text_voice(self.ctx,self.text,self.target,out,reference=self.voice_ref)
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
        self.audio_path = self.subtitle_path = ''
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0,0,0,0)
        self.sections = QTabWidget()
        outer.addWidget(self.sections)
        translation_page = QWidget()
        root = QVBoxLayout(translation_page)
        self.sections.addTab(translation_page,'Translator')
        from gui_teacher_tab import TeacherTab
        self.teacher_tab = TeacherTab(host)
        self.sections.addTab(self.teacher_tab,'Language teacher')
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
        for label, mode in (('Text', 'text'), ('Picture lettering', 'image'), ('Voice → voice', 'voice'), ('Video / audio', 'dub'), ('Text → voice (settings voice)', 'tts')):
            self.mode.addItem(label, mode)
        options.addWidget(QLabel('Translate into')); options.addWidget(self.target)
        options.addWidget(self.mode); root.addLayout(options)
        self.text_in = QPlainTextEdit()
        self.text_in.setPlaceholderText('Paste or type text to translate…')
        self.text_in.setMaximumHeight(px(180)); root.addWidget(self.text_in)
        files = QHBoxLayout()
        self.file_btn = QPushButton('📎 Choose media')
        self.file_btn.clicked.connect(self._choose_file); files.addWidget(self.file_btn)
        self.file_label = QLabel('No file selected')
        self.file_label.setWordWrap(True); files.addWidget(self.file_label, 1)
        root.addLayout(files)
        hint = QLabel('Pictures: a local background patch covers the old letters. Voice: send 3–60 seconds of clear speech; the reply uses this recording as the voice sample.')
        hint.setWordWrap(True); root.addWidget(hint)
        hint = QLabel('Video / audio: up to 10 minutes / 100 MB. Original speaker voices and timing; WAV and SRT included. Replaces the whole sound track, without preserving music or effects. No lip-sync changes.')
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
        extras = QHBoxLayout()
        self.audio_btn = QPushButton('Save audio track…')
        self.audio_btn.clicked.connect(lambda: self._save_path(self.audio_path))
        self.subtitles_btn = QPushButton('Save subtitles…')
        self.subtitles_btn.clicked.connect(lambda: self._save_path(self.subtitle_path))
        for b in (self.audio_btn, self.subtitles_btn):
            b.setEnabled(False); extras.addWidget(b)
        root.addLayout(extras)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self._mode_changed()

    def _mode_changed(self):
        self.text_in.setVisible(self.mode.currentData() in ('text','tts'))

    def _choose_file(self):
        path, _ = file_dialog(self).getOpenFileName(self, 'Choose media', '',
            'Pictures, audio and video (*.png *.jpg *.jpeg *.webp *.wav *.mp3 *.ogg *.m4a *.flac *.opus *.mp4 *.mov *.mkv *.webm *.avi *.m4v);;All files (*)')
        if path:
            self.source = path; self.file_label.setText(Path(path).name)
            suffix=Path(path).suffix.lower()
            mode = ('image' if suffix in ('.png','.jpg','.jpeg','.webp') else 'dub'
                    if suffix in ('.mp4','.mov','.mkv','.webm','.avi','.m4v') or self.mode.currentData()=='dub' else 'voice')
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
        if mode not in ('text','tts') and not self.source:
            self.status.setText('Choose a picture or audio file first.'); return
        if mode == 'image' and Path(self.source).suffix.lower() not in ('.png','.jpg','.jpeg','.webp'):
            self.status.setText('Choose a picture or audio file first.'); return
        self.output_path = ''; self.preview.clear(); self.preview.hide()
        self.audio_path = self.subtitle_path = ''
        self.audio_btn.setEnabled(False); self.subtitles_btn.setEnabled(False)
        self.play_btn.setEnabled(False); self.result.clear()
        self._busy(True); self.status.setText('Translating…')
        self.worker = TranslationWorker(ctx, mode, self.target.currentData(), self.text_in.toPlainText(), self.source)
        self.worker.done.connect(self._done)
        self.worker.progress.connect(self._progress)
        self.worker.failed.connect(lambda error: self.status.setText(translator.error_text(error, gui_i18n.LANG)))
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def _progress(self,done,total,what):
        import progress
        self.status.setText(progress.line(done,total,what,gui_i18n.LANG))

    def _done(self, result):
        self.result.setPlainText(result['text'])
        self.output_path = result.get('path', '')
        self.audio_path = result.get('audio_path') or (self.output_path if Path(self.output_path).suffix == '.wav' else '')
        self.subtitle_path = result.get('subtitle_path', '')
        self.audio_btn.setEnabled(bool(self.audio_path)); self.subtitles_btn.setEnabled(bool(self.subtitle_path))
        self.status.setText(translator.error_text(result['warning'], gui_i18n.LANG) if result.get('warning') else gui_i18n.tr('Done.'))
        if self.output_path and Path(self.output_path).suffix == '.png':
            self.preview.setPixmap(QPixmap(self.output_path).scaled(px(720), px(360), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.preview.show()
        self.play_btn.setEnabled(bool(self.audio_path))

    def _finished(self):
        self._busy(False)

    def _cancel(self):
        if self.worker: self.worker.cancel()
        if self.player: self.player.stop()
        self.status.setText('Cancelled')

    def _clear(self):
        if self.player: self.player.stop()
        self.source = self.output_path = ''
        self.audio_path = self.subtitle_path = ''
        self.audio_btn.setEnabled(False); self.subtitles_btn.setEnabled(False)
        self.text_in.clear(); self.result.clear(); self.preview.clear(); self.preview.hide()
        self.file_label.setText('No file selected'); self.status.clear(); self.play_btn.setEnabled(False)

    def _copy(self):
        QApplication.clipboard().setText(self.result.toPlainText())

    def shutdown(self):
        self._cancel()
        self.teacher_tab.shutdown()

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

    def _save_path(self, source):
        if not source: return
        ext=Path(source).suffix
        path,_=file_dialog(self).getSaveFileName(self,'Save result',Path(source).name,f'Result (*{ext})')
        if path and Path(path).resolve()!=Path(source).resolve():
            try:
                shutil.copyfile(source,path)
            except OSError as exc:
                self.status.setText(str(exc))

    def _play(self):
        if self.audio_path and self.play_btn.isEnabled():
            try:
                if self.player is None:
                    from audio import AudioPlayer
                    self.player = AudioPlayer()
                self.player.play(self.audio_path)
            except Exception as exc:
                self.status.setText(str(exc))
