"""Microphone, hands-free VAD, TTS toggle and the webcam frame grabber.

Everything that turns hardware into a turn. The VAD state machine is the part
worth having in one file: push-to-talk and hands-free share the recorder but
NOT the lifecycle — hands-free must survive between utterances, push-to-talk
must not — and that distinction was previously spread across 200 lines of an
otherwise unrelated class.

gui_workers.RequestWorker and config.OUTPUT_DIR are read through their defining modules
(gui_workers, config) rather than imported by value, because gui.py reads them
too: a by-value import here would give the two halves separate copies, and
patching one would silently move only half the behaviour.

Needs from the host: _add_system, _add_user, _add_user_image, _busy,
_on_failed, _set_busy, _set_status, _start_deep_research, _start_worker.
"""
import logging
import cv2
import numpy as np
import time
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from audio import MicRecorder, VadListener
import config
from config import MIC_GAIN_DB, SAMPLE_RATE
from gui_common import ACCENT, MUTED, PANEL2, TranscribeWorker, build_qss
import gui_workers

logger = logging.getLogger("assistant")


class VoiceMixin:
    # ---- capture ----
    def _capture_frame(self):
        enc = self._encode_current_frame()
        if enc is None:
            self._set_status("No camera frame to capture.")
            return
        self.captured_image = enc
        path = str(config.OUTPUT_DIR / f"_capture_{int(time.time() * 1000)}.jpg")
        try:
            with open(path, "wb") as f:
                f.write(enc)
            self._add_user_image(path)
        except Exception:
            pass
        self._set_status("📸 Frame captured — now ask your question (type or Talk).")

    def _take_captured_image(self):
        img = self.captured_image
        self.captured_image = None
        return img

    def _toggle_recording(self):
        # Allow stopping an in-progress recording even if busy; block starting a new one.
        if self.recorder is None and (self.graph is None or self._busy()):
            return
        if self.recorder is None and self.ctx is not None and self.ctx.mic_disabled:
            return
        if self.recorder is None:
            # prefer an explicitly captured frame; else grab the live frame if the camera is on
            self.pending_image = self.captured_image or self._encode_current_frame()
            self.captured_image = None
            # Opening the input stream fails routinely on Windows — no capture
            # device, or another app holding it. Unguarded, the half-constructed
            # recorder stayed on self.recorder: the Talk button then read as "not
            # recording" but took the stop branch on the next click, and _vad_tick
            # saw a live recorder and kept VAD permanently deaf.
            try:
                rec = MicRecorder(fs=SAMPLE_RATE, gain_db=MIC_GAIN_DB)
                rec.start()
            except Exception as exc:
                logger.exception("Microphone start failed")
                self.recorder = None
                self.pending_image = None
                self._add_system(f"Could not turn on the microphone: {exc}")
                self._set_status("Microphone unavailable.")
                return
            self.recorder = rec
            self._mic_viz_timer.start()  # make the waterfall shake to my voice
            self.talk_btn.setText("■  Stop"); self.talk_btn.setObjectName("rec"); self.talk_btn.setStyleSheet(build_qss())
            self._set_status("Recording… click Stop to send")
        else:
            self._mic_viz_timer.stop()
            try:
                audio = self.recorder.stop()
            except Exception as exc:
                # Same reasoning in reverse: a failed stop must still release the
                # button, or Talk is stuck on "■ Stop" for the rest of the session.
                logger.exception("Microphone stop failed")
                audio = np.zeros(0, dtype=np.float32)
                self._add_system(f"The microphone disconnected while recording: {exc}")
            finally:
                self.recorder = None
            self.viz.stop()
            self.talk_btn.setText("🎤  Talk"); self.talk_btn.setObjectName("ghost"); self.talk_btn.setStyleSheet(build_qss())
            if len(audio) < SAMPLE_RATE * 0.5:
                self.pending_image = None; self._set_status("Too short — try again."); return
            image, self.pending_image = self.pending_image, None
            self._start_worker(gui_workers.RequestWorker(self.ctx, self.graph, self.base_state,
                                             audio=audio, image=image, use_db=self.usedb_on,
                                             db_k=self._db_k()))

    def _pump_mic_level(self):
        """Feed the live microphone loudness into the audio visualizer so the
        waterfall shakes while the user is speaking (mirrors play_file for TTS).
        Reads whichever mic source is live — the Talk recorder or the VAD listener.
        Yields to TTS playback, which drives the visualizer itself."""
        source = self.recorder or self.vad_listener
        if source is None or self.player.is_active:
            if self.recorder is None and self.vad_listener is None:
                self._mic_viz_timer.stop()
            return
        self.viz.push_level(source.level())
    # ---- hands-free VAD mode ----

    def _toggle_vad(self):
        if self.vad_listener is not None:
            self._stop_vad("VAD mode OFF.")
            return
        if self.graph is None or self.ctx is None:
            return
        if self.ctx.mic_disabled:
            self._set_status("Microphone is disabled in Settings — enable it to use VAD mode.")
            return
        try:
            listener = VadListener(on_utterance=self.vad_utterance.emit, gain_db=MIC_GAIN_DB)
            listener.start()
        except Exception as exc:
            logger.exception("VAD start failed")
            self._add_system(f"Could not turn on VAD mode: {exc}")
            return
        self.vad_listener = listener
        self.vad_btn.setText("🎙  VAD: ON")
        self.vad_btn.setStyleSheet(f"background:{ACCENT}; color:#fff;")
        self._mic_viz_timer.start()  # waterfall shakes while VAD hears speech
        self._vad_tick()  # sync the paused state immediately
        self._set_status("VAD mode ON — just speak, I'm listening.")

    def _stop_vad(self, status_msg=""):
        listener, self.vad_listener = self.vad_listener, None
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                logger.exception("VAD stop failed")
        if self.recorder is None:
            self._mic_viz_timer.stop()
            self.viz.stop()
        self.vad_btn.setText("🎙  VAD: OFF")
        self.vad_btn.setStyleSheet("")
        if status_msg:
            self._set_status(status_msg)

    def _vad_tick(self):
        if self.vad_listener is None:
            return
        if self.ctx is not None and self.ctx.mic_disabled:
            self._stop_vad("VAD mode OFF (microphone was disabled in Settings).")
            return
        if not self.vad_listener.is_healthy():
            # Mic unplugged / audio device switched: the stream died silently —
            # without this, VAD stays "ON" but hears nothing.
            self._stop_vad('VAD is off: the microphone disappeared (device unplugged?).')
            self._add_system('VAD mode stopped — the microphone stopped responding. Check the device and turn VAD on again.')
            return
        # Deaf while a request runs, while a manual recording is in progress, and
        # while the assistant's own voice plays — it must never hear itself.
        self.vad_listener.set_paused(self._busy() or self.recorder is not None
                                     or self.player.is_active)

    def _on_vad_utterance(self, audio):
        if self.vad_listener is None or self.graph is None:
            return
        if self._busy() or self.recorder is not None or self.player.is_active:
            return  # the utterance raced a state change — drop it (likely TTS echo)
        if audio is None or len(audio) < SAMPLE_RATE * 0.3:
            return
        audio = np.asarray(audio, dtype=np.float32)
        # Attach only an EXPLICITLY captured frame. Unlike the Talk button, VAD must
        # not auto-grab the live camera image: that would turn every spoken phrase
        # into a vision turn and overwrite the working image (ctx.last_image_path)
        # with a webcam frame — silently replacing the picture being edited.
        image = self.captured_image
        self.captured_image = None
        if self.ultra_search_on:
            # Ultra Search expects a research TOPIC, so transcribe first and route
            # to deep research — exactly what typed text does in _send_text. The
            # worker ref makes _busy() true, which keeps the VAD listener deaf;
            # the topic is only acted on after `finished`, when the thread is done
            # and the busy gate is released.
            w = TranscribeWorker(self.ctx, audio)
            w.recognized.connect(self._set_pending_topic)
            w.failed.connect(self._on_failed)
            w.finished.connect(self._on_transcribe_finished)
            self.transcribe_worker = w
            self._pending_topic = ""
            self._set_busy(True, "Transcribing…")
            self.stage.set_stage("Transcribing")
            w.start()
            return
        self._start_worker(gui_workers.RequestWorker(self.ctx, self.graph, self.base_state,
                                         audio=audio, image=image, use_db=self.usedb_on,
                                         db_k=self._db_k()))

    def _set_pending_topic(self, text):
        self._pending_topic = text

    def _on_transcribe_finished(self):
        self.transcribe_worker = None
        topic, self._pending_topic = self._pending_topic, ""
        self.stage.set_stage("Ready")
        self._set_status("")
        self._set_busy(False)   # always reset visually; _start_deep_research re-sets if needed
        if topic:
            self._add_user(topic)
            self._start_deep_research(topic)

    def _toggle_voice(self):
        if self.ctx is None:
            return
        self.ctx.tts_disabled = not self.ctx.tts_disabled
        if self.ctx.tts_disabled:
            self.player.stop()
            self.viz.stop()
            self.pause_btn.setText("⏸  Pause speech")
        self._update_voice_btn()

    def _update_voice_btn(self):
        muted = self.ctx is not None and self.ctx.tts_disabled
        if muted:
            self.voice_btn.setText("🔇  Voice: OFF")
            self.voice_btn.setStyleSheet(f"background:{PANEL2}; color:{MUTED};")
        else:
            self.voice_btn.setText("🔊  Voice: ON")
            self.voice_btn.setStyleSheet("")
    # ---- camera ----

    def _toggle_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = None; self._set_status("Camera not available."); return
            self.camera_label.show(); self.cam_btn.setText("📷  Camera: on"); self.cam_timer.start(30)
            self.capture_btn.setEnabled(not self._busy())
            self._set_status("Camera on — click Capture frame, then ask about it.")
        else:
            self.cam_timer.stop(); self.cap.release(); self.cap = None
            self.current_frame = None; self.camera_label.hide(); self.cam_btn.setText("📷  Camera: off")
            self.capture_btn.setEnabled(False)

    def _update_camera(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        self.current_frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.camera_label.width(), self.camera_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _encode_current_frame(self):
        if self.current_frame is None:
            return None
        ok, enc = cv2.imencode(".jpg", self.current_frame)
        return enc.tobytes() if ok else None
