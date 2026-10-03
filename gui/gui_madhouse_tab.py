"""The Madhouse: several AI characters talking to each other, and to you.

Second tab out of gui.py — 1,300 lines including its grid renderer, its two
background workers, the reply generator and the speaker-selection rule.

It moved as ONE unit because that is what it is: MadhouseGrid draws the room,
MadhouseReplyWorker and MadhouseSpeakWorker keep generation and speech off the
UI thread, generate_madhouse_reply produces a turn, and choose_next_speaker
decides whose turn it is. Splitting those across modules would have made four
files that only ever call each other.

Cancellation is the subtle part and it lives in _ScopedCtx (gui_common): the
global ctx.cancel_event is set by the main chat's Stop button and cleared only
when the main chat takes its next turn, so one Stop press used to mute the whole
room permanently — every reply came back None and the room printed "the model
returned no text" forever. Each long-running tab therefore takes a scoped view
with its own cancel token.

_MADHOUSE_TTS_LOCK serialises speech: the characters generate concurrently but
must not talk over one another.
"""
import json
import os
import random
import threading
import time
from pathlib import Path

from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QLabel, QLineEdit, QListWidget, QMessageBox,
    QPushButton, QWidget,
)

from audio import AudioPlayer, MicRecorder
from config import MIC_GAIN_DB, OUTPUT_DIR
from ui_scale import px
from gui_common import (MUTED, TranscribeWorker,
                        _FlowWidget, _ScopedCtx, _flow, _section)

import logging
logger = logging.getLogger("assistant.gui")   # same channel as gui.py

# The brain (what to say, who says it next) and the room renderer moved out.
# Re-exported by value so that gui.py's import, `gui_madhouse_tab.<name>` call
# sites, and MadhouseReplyWorker's own bare global lookups all keep resolving --
# the last of those is the seam a suite patches to stub a reply.
import gui_madhouse_brain as _brain
import gui_madhouse_grid as _grid
from gui_madhouse_ui import MadhouseUIMixin
from gui_madhouse_cast import MadhouseCastMixin

generate_madhouse_reply = _brain.generate_madhouse_reply
_name_is_addressed      = _brain._name_is_addressed
choose_next_speaker     = _brain.choose_next_speaker
MadhouseGrid            = _grid.MadhouseGrid


MADHOUSE_MIN_DELAY_MS = 2000
MADHOUSE_MAX_DELAY_MS = 5000

_MADHOUSE_TTS_LOCK = threading.Lock()   # serializes the ctx.custom_ref_wav swap below


class MadhouseReplyWorker(QThread):
    """One generated line, off the UI thread."""
    done = pyqtSignal(str, str)       # character_id, text
    failed = pyqtSignal(str, str)     # character_id, error

    def __init__(self, ctx, characters, history, last_id, use_router, human_name):
        super().__init__()
        self.ctx, self.characters, self.history = ctx, characters, history
        self.last_id, self.use_router, self.human_name = last_id, use_router, human_name

    def run(self):
        # Routing runs here too: it's a model call, so it must stay off the UI thread.
        character = None
        try:
            character = choose_next_speaker(self.ctx, self.characters, self.history,
                                            self.last_id, self.use_router, self.human_name)
            if character is None:
                self.failed.emit("", "no character available to speak")
                return
            speaking = {**character, "_human": self.human_name}   # who's in the room
            self.done.emit(character["id"],
                           generate_madhouse_reply(self.ctx, speaking, self.history))
        except Exception as exc:
            logger.exception("Madhouse reply failed")
            self.failed.emit((character or {}).get("id", ""), str(exc))


class MadhouseSpeakWorker(QThread):
    """Synthesize one line in the character's cloned voice. Emits '' on failure."""
    ready = pyqtSignal(str)

    def __init__(self, ctx, voice_wav, text, idx):
        super().__init__()
        self.ctx, self.voice_wav, self.text, self.idx = ctx, voice_wav, text, idx

    def run(self):
        path = ""
        try:
            from audio import synth_single_segment
            from config import ASSISTANT_ACTOR
            # ctx.custom_ref_wav is process-wide state, so hold the lock across the
            # whole synth and always restore it — otherwise a concurrent assistant
            # reply would be spoken in a Madhouse character's voice.
            with _MADHOUSE_TTS_LOCK:
                prev = getattr(self.ctx, "custom_ref_wav", None)
                self.ctx.custom_ref_wav = self.voice_wav or None
                logger.info("Madhouse line %d voice: %s", self.idx,
                            os.path.basename(self.voice_wav) if self.voice_wav else "default")
                try:
                    path = synth_single_segment(
                        self.ctx, self.idx, ASSISTANT_ACTOR, self.text,
                        out_stem=str(OUTPUT_DIR / "madhouse"),
                    ) or ""
                finally:
                    self.ctx.custom_ref_wav = prev
        except Exception:
            logger.exception("Madhouse TTS failed")
        self.ready.emit(path)


class MadhouseTab(MadhouseUIMixin, MadhouseCastMixin, QWidget):
    """Interactive multi-character chat room.

    Load a cast from JSON, let it run itself (random speaker who did NOT speak
    last, random 2-5 s gap), cut in manually as any character, pause/resume, and
    clear the room. Every line is spoken in that character's cloned voice through
    the tab's own AudioPlayer, so it never fights the main chat's playback.
    """

    USER_ID = "__you__"

    def __init__(self, host):
        super().__init__()
        self.host = host                  # AssistantWindow (for ctx)
        self.characters = []              # [{id, name, voice, prompt, _voice_wav}]
        self.messages = []                # [{character_id, name, text, ts}]
        self.player = AudioPlayer()
        # You, the human, are a permanent participant: always in the "speak as" list,
        # never picked by the auto-dialogue, and never synthesized (you have a voice).
        self.user = {"id": self.USER_ID, "name": "You", "voice": "", "personality": "",
                     "prompt": "", "_voice_wav": None, "_is_user": True}
        self.reply_worker = None
        self.speak_worker = None
        self.recorder = None
        self.transcribe_worker = None
        # The room's own cancel flag (see _ScopedCtx) — set by Stop, cleared by Start.
        self._room_cancel = threading.Event()
        self._speech_q = []               # [(voice_wav, text)] pending synthesis
        self._auto = False
        self._paused = False
        self._run_id = 0                  # invalidates timers from a previous run
        self._seq = 0                     # unique stem per synthesized segment
        # Poll for playback-end so the next synthesis starts only after the current
        # line has actually finished playing (not just finished synthesizing).
        self._play_poll = QTimer(self)
        self._play_poll.setInterval(150)
        self._play_poll.timeout.connect(self._on_play_poll)

        self._build_ui()

    def _character(self, cid):
        if cid == self.USER_ID:
            return self.user
        for c in self.characters:
            if c["id"] == cid:
                return c
        return None

    # ----- transcript --------------------------------------------------------
    def _post(self, character, text):
        ts = time.time()
        self.messages.append({"character_id": character["id"], "name": character["name"],
                              "text": text, "ts": ts})
        # Show speech bubble on the minimap
        self.grid.show_speech(character["id"], text)
        # Always mirror into the main centre pane (the minimap shows actions,
        # the main chat is the readable transcript).
        fn = getattr(self.host, "_add_room_line", None)
        if fn is not None:
            try:
                fn(character["name"], text, is_user=bool(character.get("_is_user")))
            except Exception:
                logger.exception("mirroring a room line into the main chat failed")
        if character.get("_is_user"):
            return                        # you already said it out loud — don't echo it
        self._speech_q.append((character["_voice_wav"], text))
        self._pump_speech()

    def _note(self, text):
        self.grid.show_note(text)

    def _clear_history(self):
        self._stop_speech()
        self.messages.clear()
        self.grid.clear_all()

    # ----- speech ------------------------------------------------------------
    def _on_voice_mode(self):
        if self.voice_mode.currentData() == "off":
            self._stop_speech()                        # silence takes effect at once
        self._sync_controls()

    def _pump_speech(self):
        if self.speak_worker is not None or not self._speech_q:
            return
        # Don't start the next synthesis while the player is still playing the
        # previous line — that would interrupt it. Let _on_play_poll() re-trigger us
        # when the player goes idle.
        if self.player.is_active:
            self._play_poll.start()
            return
        mode = self.voice_mode.currentData()
        ctx = getattr(self.host, "ctx", None)
        # Silent mode, no model loaded, or the app-wide TTS mute: drop the backlog and
        # let the room keep running as text.
        if mode == "off" or ctx is None or getattr(ctx, "tts_disabled", False):
            self._speech_q.clear()
            return
        voice, text = self._speech_q.pop(0)
        if mode == "default":
            voice = None                               # everyone in the assistant's voice
        self._seq += 1
        self.speak_worker = MadhouseSpeakWorker(ctx, voice, text, self._seq)
        self.speak_worker.ready.connect(self._on_speech_ready)
        self.speak_worker.finished.connect(self._on_speak_worker_finished)
        self.speak_worker.start()
        self._sync_controls()

    def _on_play_poll(self):
        """Drain the speech queue once the current playback finishes."""
        if not self.player.is_active:
            self._play_poll.stop()
            self._pump_speech()

    def _on_speech_ready(self, path):
        if path and os.path.exists(path):
            try:
                self.player.play(path)
                # Start polling so we know when this line finishes playing.
                self._play_poll.start()
            except Exception:
                logger.exception("Madhouse playback failed")

    def _on_speak_worker_finished(self):
        self.speak_worker = None
        self._pump_speech()
        self._sync_controls()

    def _stop_speech(self):
        self._speech_q.clear()
        self._play_poll.stop()
        try:
            self.player.stop()
        except Exception:
            logger.exception("Madhouse stop failed")

    # ----- auto-dialogue -----------------------------------------------------
    def _start_auto(self):
        if len(self.characters) < 2:
            return
        self._stop_auto()                 # never leave a second timer chain running
        self._room_cancel.clear()         # a previous Stop must not mute this run
        self._auto = True
        self._paused = False
        self._schedule(400)               # first line comes quickly
        self._sync_controls()

    def _stop_auto(self):
        self._auto = False
        self._paused = False
        self._run_id += 1                 # any in-flight singleShot becomes a no-op
        self._room_cancel.set()           # abort a reply that is mid-stream
        self._stop_speech()
        self._sync_controls()
        ctx = getattr(self.host, "ctx", None)
        if ctx is not None:
            stage = getattr(self.host, "stage", None)
            if stage is not None:
                stage.set_stage("Ready")

    def _toggle_pause(self):
        if not self._auto:
            return
        self._paused = not self._paused   # current line finishes; no new ones queued
        self._sync_controls()

    def _schedule(self, delay_ms=None):
        if not self._auto:
            return
        if delay_ms is None:
            delay_ms = random.randint(MADHOUSE_MIN_DELAY_MS, MADHOUSE_MAX_DELAY_MS)
        rid = self._run_id
        QTimer.singleShot(delay_ms, lambda: self._tick(rid))

    def _tick(self, rid):
        if rid != self._run_id or not self._auto:      # stale timer from a stopped run
            return
        self._room_cancel.clear()                      # clear any stale cancel from a previous Stop
        # The next line waits until the previous ones have been SPOKEN, not merely
        # synthesized: generating while lines queue for the speaker let the text run
        # ahead of the voice without end.
        if (self._paused or self.reply_worker is not None or self.speak_worker is not None
                or self._speech_q or self.player.is_active):
            self._schedule(600)                        # busy/paused/speaking: try again shortly
            return
        if not self.characters:
            self._schedule()
            return
        ctx = getattr(self.host, "ctx", None)
        if ctx is not None:
            ctx = _ScopedCtx(ctx, self._room_cancel)   # immune to the main chat's Stop
        last_id = self.messages[-1]["character_id"] if self.messages else None
        self.reply_worker = MadhouseReplyWorker(
            ctx, list(self.characters), list(self.messages), last_id,
            self.router_box.isChecked(), self.user["name"])
        self.reply_worker.done.connect(self._on_reply)
        self.reply_worker.failed.connect(self._on_reply_failed)
        self.reply_worker.finished.connect(self._on_reply_worker_finished)
        self.reply_worker.start()
        self._sync_controls()

    def _on_reply(self, cid, text):
        char = self._character(cid)
        if char is not None and text:
            self._post(char, text)
        self._schedule()

    def _on_reply_failed(self, cid, err):
        # Cancellation is expected during Stop/load — don't surface it as an error.
        if "cancelled" not in str(err).lower():
            self._note(f"reply failed: {err}")
        # Back off 3 s on failure to avoid hammering LM Studio when it is under load
        self._schedule(3000)

    def _on_reply_worker_finished(self):
        self.reply_worker = None
        self._sync_controls()

    # ----- manual turn -------------------------------------------------------
    def _on_my_name_changed(self, text):
        # Past lines keep the name they were said under; only new ones change.
        self.user["name"] = (text or "").strip() or "You"
        idx = self.who.findData(self.USER_ID)
        if idx >= 0:
            self.who.setItemText(idx, f"🙋  {self.user['name']}")

    def _speaker_choice(self):
        return self._character(self.who.currentData()) or self.user

    def _send_manual(self):
        text = self.say_in.text().strip()
        if not text:
            return
        self.say_in.clear()
        # Works mid-run; the auto loop simply waits its turn.
        self._post(self._speaker_choice(), text)

    # ----- joining by voice --------------------------------------------------
    def _toggle_talk(self):
        """Click to record, click again to transcribe and post. Same Whisper path
        the main chat uses, so language handling matches."""
        ctx = getattr(self.host, "ctx", None)
        if ctx is None:
            self._note("no model loaded yet — the microphone needs Whisper")
            return
        if getattr(ctx, "mic_disabled", False):
            self._note("the microphone is disabled in Settings ▸ Input / Output")
            return
        if self.recorder is None:
            try:
                self.recorder = MicRecorder(gain_db=MIC_GAIN_DB)
                self.recorder.start()
            except Exception as exc:
                self.recorder = None
                self._note(f"microphone unavailable: {exc}")
                return
            self.talk_btn.setText("⏹")
            return
        try:
            audio = self.recorder.stop()
        except Exception as exc:
            audio = None
            self._note(f"recording failed: {exc}")
        finally:
            self.recorder = None
            self.talk_btn.setText("🎤")
        if audio is None or not len(audio):
            return
        self.talk_btn.setEnabled(False)
        self.transcribe_worker = TranscribeWorker(ctx, audio)
        self.transcribe_worker.recognized.connect(self._on_transcribed)
        self.transcribe_worker.failed.connect(lambda m: self._note(m))
        self.transcribe_worker.finished.connect(self._on_transcribe_finished)
        self.transcribe_worker.start()

    def _on_transcribed(self, text):
        text = (text or "").strip()
        if text:
            self._post(self._speaker_choice(), text)

    def _on_transcribe_finished(self):
        self.transcribe_worker = None
        self.talk_btn.setEnabled(True)

    # ----- state -------------------------------------------------------------
    def _sync_controls(self):
        enough = len(self.characters) >= 2
        self.start_btn.setEnabled(enough and not self._auto)
        self.stop_btn.setEnabled(self._auto)
        self.pause_btn.setEnabled(self._auto)
        self.pause_btn.setText("▶  Resume" if self._paused else "⏸  Pause")
        self.say_in.setEnabled(True)      # you can talk to an empty room if you like
        self.who.setEnabled(True)
        self.hush_btn.setEnabled(self.voice_mode.currentData() != "off")
        if not self.characters:
            state = "load a cast to begin"
        elif not enough:
            state = "need at least 2 characters for auto-dialogue"
        elif self.reply_worker is not None:
            state = "thinking…"
        elif self.speak_worker is not None:
            state = "speaking…"
        elif self._paused:
            state = "paused"
        elif self._auto:
            state = "running"
        else:
            state = "idle"
        self.status_lbl.setText(state)

    def shutdown(self):
        """Called from the main window's close handler — kill timers and audio."""
        self._stop_auto()
        if self.recorder is not None:
            try:
                self.recorder.stop()
            except Exception:
                logger.exception("Madhouse recorder stop failed")
            self.recorder = None
        for w in (self.reply_worker, self.speak_worker, self.transcribe_worker):
            if w is not None:
                w.wait(3000)
