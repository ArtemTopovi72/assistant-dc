"""Local random draws and a separately cancellable LLM mini prediction."""
import threading
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

import gui_i18n
import randomizer
from gui_common import _section, _ScopedCtx
from ui_scale import px


class PredictionWorker(QThread):
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, ctx, period, lang):
        super().__init__()
        self._cancel = threading.Event()
        self.ctx, self.period, self.lang = _ScopedCtx(ctx, self._cancel), period, lang

    def cancel(self):
        self._cancel.set()

    def run(self):
        try:
            text = randomizer.prediction(self.ctx, self.period, self.lang)
            if not self._cancel.is_set():
                self.done.emit(text)
        except Exception as exc:
            if not self._cancel.is_set():
                self.failed.emit(str(exc))


class RandomizerTab(QWidget):
    def __init__(self, host=None):
        super().__init__()
        self.host, self.worker = host, None
        self.rng = None
        root = QVBoxLayout(self)
        root.setContentsMargins(px(12), px(12), px(12), px(12))
        root.setSpacing(px(12))
        root.addWidget(_section('Randomizer'))
        root.addWidget(QLabel('Three matching symbols win.'))
        self.reels = QLabel('❔  ❔  ❔')
        self.reels.setAlignment(Qt.AlignCenter)
        self.reels.setStyleSheet(f'font-size: {px(40)}px; padding: {px(14)}px;')
        root.addWidget(self.reels)
        self.spin_button = QPushButton('🎰 Spin')
        self.spin_button.clicked.connect(self._spin)
        root.addWidget(self.spin_button)
        self.spin_result = QLabel('')
        self.spin_result.setAlignment(Qt.AlignCenter)
        root.addWidget(self.spin_result)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._ticks = 0

        root.addWidget(_section('Random integer'))
        bounds = QHBoxLayout()
        self.low, self.high = QLineEdit('1'), QLineEdit('100')
        for label, edit in (('From', self.low), ('To', self.high)):
            edit.setMaxLength(65)
            edit.setAccessibleName(label)
            edit.returnPressed.connect(self._number)
            bounds.addWidget(QLabel(label))
            bounds.addWidget(edit)
        root.addLayout(bounds)
        self.number_button = QPushButton('Generate number')
        self.number_button.clicked.connect(self._number)
        root.addWidget(self.number_button)
        self.number_result = QLabel('')
        self.number_result.setWordWrap(True)
        self.number_result.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.number_result)

        root.addWidget(_section('Quick draws'))
        buttons = QHBoxLayout()
        for label, action in (('✅ Yes / No', 'yesno'), ('💰 Flip a coin', 'coin'), ('🎲 Roll a die', 'die')):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, a=action: self._quick(a))
            buttons.addWidget(button)
        root.addLayout(buttons)
        self.quick_result = QLabel('')
        self.quick_result.setAlignment(Qt.AlignCenter)
        root.addWidget(self.quick_result)
        root.addWidget(_section('Mini prediction'))
        prediction_row = QHBoxLayout()
        self.period = QComboBox()
        for label, value in (('For a day', 'day'), ('For a week', 'week'), ('For a month', 'month')):
            self.period.addItem(label, value)
        prediction_row.addWidget(self.period)
        self.predict_button = QPushButton('🔮 Predict')
        self.predict_button.clicked.connect(self._predict)
        prediction_row.addWidget(self.predict_button)
        root.addLayout(prediction_row)
        self.prediction_result = QLabel('')
        self.prediction_result.setWordWrap(True)
        self.prediction_result.setTextFormat(Qt.PlainText)
        self.prediction_result.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.prediction_result)
        root.addStretch()

    def _spin(self):
        self._ticks = 8
        self.spin_button.setEnabled(False)
        self.spin_result.clear()
        self.timer.start(80)

    def _tick(self):
        if not self._ticks:
            return
        symbols = randomizer.spin(self.rng)
        self.reels.setText('  '.join(symbols))
        self._ticks -= 1
        if not self._ticks:
            self.timer.stop()
            self.spin_button.setEnabled(True)
            self.spin_result.setText('🎉 Jackpot!' if randomizer.jackpot(symbols) else 'Try again!')

    def _number(self):
        try:
            low, high = int(self.low.text()), int(self.high.text())
        except ValueError:
            self.number_result.setText('Enter two integers, for example: -10 100.')
            return
        try:
            self.number_result.setText(str(randomizer.number(low, high, self.rng)))
        except ValueError as exc:
            self.number_result.setText(str(exc))

    def _quick(self, action):
        if action == 'die':
            result = '🎲 ' + str(randomizer.number(1, 6, self.rng))
        elif action == 'coin':
            result = gui_i18n.tr('Heads' if randomizer.binary(self.rng) else 'Tails')
        else:
            result = gui_i18n.tr('✅ Yes' if randomizer.binary(self.rng) else '❌ No')
        self.quick_result.setText(result)

    def _predict(self):
        if self.worker and self.worker.isRunning():
            return
        ctx = getattr(self.host, 'ctx', None)
        if ctx is None:
            self.prediction_result.setText('The model is not ready yet.')
            return
        self.predict_button.setEnabled(False)
        self.period.setEnabled(False)
        self.prediction_result.setText('Generating a mini prediction…')
        self.worker = PredictionWorker(ctx, self.period.currentData(), gui_i18n.LANG)
        self.worker.done.connect(self.prediction_result.setText)
        self.worker.failed.connect(self.prediction_result.setText)
        self.worker.finished.connect(self._prediction_finished)
        self.worker.start()

    def _prediction_finished(self):
        self.predict_button.setEnabled(True)
        self.period.setEnabled(True)
