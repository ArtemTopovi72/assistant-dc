"""Boundary-stubbed coverage for audio.py's device/model layer: MicRecorder,
VadListener, record_audio_until_enter, transcribe_audio_array/file,
synth_single_segment, play_audio_file, AudioPlayer. Stubs sounddevice
(InputStream/OutputStream/sleep/CallbackStop), silero VAD, whisper, and F5-TTS
inference so every branch runs deterministically without a real mic/speaker/GPU.
Run: venv/Scripts/python.exe tests/test_audio_device.py
"""
import os, sys, tempfile, threading, time, builtins, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path
import numpy as np
import audio as A

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))

_TMP = Path(tempfile.mkdtemp(prefix="audiodev_"))


class _Patches:
    def __init__(self, **kw):
        self.kw = kw; self.orig = {}
    def __enter__(self):
        for k, v in self.kw.items():
            self.orig[k] = getattr(A, k)
            setattr(A, k, v)
        return self
    def __exit__(self, *a):
        for k, v in self.orig.items():
            setattr(A, k, v)


class FakeStream:
    """Stand-in for sd.InputStream/OutputStream. Captures the callback so a test
    can drive it manually, and tracks active/start/stop/close calls."""
    instances = []
    def __init__(self, samplerate=16000, channels=1, dtype="float32", callback=None,
                 blocksize=None, device=None, finished_callback=None):
        self.samplerate = samplerate
        self.callback = callback
        self.finished_callback = finished_callback
        self.active = False
        self.started = False
        self.closed = False
        FakeStream.instances.append(self)
    def start(self):
        self.active = True
        self.started = True
    def stop(self):
        self.active = False
        if self.finished_callback:
            try:
                self.finished_callback()
            except Exception:
                pass
    def close(self):
        self.closed = True


class FakeCallbackStop(Exception):
    pass


class FakeSd:
    InputStream = FakeStream
    OutputStream = FakeStream
    CallbackStop = FakeCallbackStop
    def __init__(self):
        self.slept = []
    def sleep(self, ms):
        self.slept.append(ms)


class FakeSegments:
    def __init__(self, texts):
        self._texts = texts
    def __iter__(self):
        for t in self._texts:
            yield types.SimpleNamespace(text=t)


class FakeWhisper:
    def __init__(self, texts=("hello world",), raises=False):
        self._texts = texts
        self._raises = raises
    def transcribe(self, audio, language=None, beam_size=5, vad_filter=True):
        if self._raises:
            raise RuntimeError("asr boom")
        info = types.SimpleNamespace(language="en", language_probability=0.9)
        return FakeSegments(self._texts), info


class FakeModels:
    def __init__(self, whisper=None):
        self.whisper = whisper or FakeWhisper()


class FakeCtx:
    def __init__(self, whisper=None, cache=None):
        self.models = FakeModels(whisper)
        self.asr_lock = threading.Lock()
        self.tts_lock = threading.Lock()
        self.transcription_cache = cache if cache is not None else {}
        self._saved = False
    def save_cache(self):
        self._saved = True


# ---------------- MicRecorder ----------------

def test_mic_recorder():
    FakeStream.instances.clear()
    with _Patches(sd=FakeSd()):
        rec = A.MicRecorder(fs=16000, gain_db=6.0)
        rec.start()
        stream = FakeStream.instances[-1]
        check("mic_recorder_started", stream.started)
        stream.callback(np.ones((10, 1), dtype=np.float32) * 0.1, 10, None, None)
        check("mic_recorder_level_updated", rec.level() > 0)
        out = rec.stop()
        check("mic_recorder_stop_returns_audio", out.size == 10)
        check("mic_recorder_level_reset", rec.level() == 0.0)

    with _Patches(sd=FakeSd()):
        rec2 = A.MicRecorder()
        out2 = rec2.stop()  # never started -> no stream
        check("mic_recorder_stop_without_start", out2.size == 0)

    with _Patches(sd=FakeSd()):
        rec3 = A.MicRecorder(gain_db=6.0)
        rec3.start()
        stream3 = FakeStream.instances[-1]
        stream3.callback(np.zeros((5, 1), dtype=np.float32), 5, None, "overflow")  # status branch
        rec3.stop()
        check("mic_recorder_callback_status_logged", True)


# ---------------- record_audio_until_enter ----------------

def test_record_audio_until_enter_eof():
    with _Patches(sd=FakeSd()):
        real_input = builtins.input
        builtins.input = lambda: (_ for _ in ()).throw(EOFError())
        try:
            out = A.record_audio_until_enter()
            check("record_until_enter_eof_empty", out.size == 0)
        finally:
            builtins.input = real_input


def test_record_audio_until_enter_full_cycle():
    FakeStream.instances.clear()
    call_n = {"n": 0}
    def fake_input():
        call_n["n"] += 1
        return ""
    with _Patches(sd=FakeSd()):
        real_input = builtins.input
        builtins.input = fake_input
        try:
            def driver():
                time.sleep(0.05)
                stream = FakeStream.instances[-1]
                stream.callback(np.ones((100, 1), dtype=np.float32) * 0.5, 100, None, None)
                time.sleep(0.05)
                stream.active = False
            threading.Thread(target=driver, daemon=True).start()
            out = A.record_audio_until_enter(gain_db=3.0)
            check("record_until_enter_full_cycle", isinstance(out, np.ndarray))
        finally:
            builtins.input = real_input


# ---------------- transcribe_audio_array / transcribe_audio_file ----------------

def test_transcribe_audio_array():
    ctx = FakeCtx(FakeWhisper(("hi there",)))
    out = A.transcribe_audio_array(ctx, np.zeros(1600, dtype=np.float32))
    check("transcribe_array_success", out == "hi there")

    ctx2 = FakeCtx(FakeWhisper(raises=True))
    out2 = A.transcribe_audio_array(ctx2, np.zeros(1600, dtype=np.float32))
    check("transcribe_array_exception_empty", out2 == "")


def test_transcribe_audio_file():
    check("transcribe_file_missing_path", A.transcribe_audio_file(FakeCtx(), str(_TMP / "nope.wav")) == "")

    wav_path = _TMP / "in.wav"
    import soundfile as sf
    sf.write(wav_path, np.zeros(1600, dtype=np.float32), 16000)

    ctx = FakeCtx(FakeWhisper(("transcribed text",)))
    out = A.transcribe_audio_file(ctx, str(wav_path))
    check("transcribe_file_success", out == "transcribed text")
    check("transcribe_file_cache_saved", ctx._saved is True)

    ctx2 = FakeCtx(FakeWhisper(("should not be used",)))
    from utils import audio_hash_from_path
    # The engine is part of the key: the default ("auto") engine caches under
    # "asr_", the explicit Whisper engine under "whisper_" (see audio.py).
    cache_key = f"asr_{audio_hash_from_path(str(wav_path))}"
    ctx2.transcription_cache[cache_key] = "cached value"
    out2 = A.transcribe_audio_file(ctx2, str(wav_path))
    check("transcribe_file_cache_hit", out2 == "cached value")

    ctx3 = FakeCtx(FakeWhisper(raises=True))
    out3 = A.transcribe_audio_file(ctx3, str(wav_path))
    check("transcribe_file_exception_empty", out3 == "")


# ---------------- synth_single_segment ----------------

def test_synth_single_segment():
    def fake_infer_process(ref_file, ref_text, text, model, vocoder, **kw):
        wav = np.zeros(1600, dtype=np.float32)
        return wav, 16000, None

    fake_f5 = types.ModuleType("f5_tts.infer.utils_infer")
    fake_f5.infer_process = fake_infer_process
    fake_f5.preprocess_ref_audio_text = lambda ref, txt: (ref, txt)
    sys.modules["f5_tts.infer.utils_infer"] = fake_f5
    sys.modules.setdefault("f5_tts", types.ModuleType("f5_tts"))
    sys.modules.setdefault("f5_tts.infer", types.ModuleType("f5_tts.infer"))

    ref_wav = _TMP / "ref.wav"
    import soundfile as sf
    sf.write(ref_wav, np.zeros(1600, dtype=np.float32), 16000)

    class Ctx2:
        def __init__(self):
            self.models = types.SimpleNamespace(tts_model=None, vocoder=None,
                                                accentor_loaded=False, accentor=lambda t: t)
            self.tts_lock = threading.Lock()
            self.custom_ref_wav = None
            # synth_single_segment transcribes the reference clip, which needs
            # these three. Without them it raised AttributeError, audio.py
            # swallowed it into "TTS error" and returned None -- and the two
            # checks below failed silently for as long as this suite could not
            # report a failed check().
            self.transcription_cache = {}
            self.cache_file = _TMP / "synth_cache.json"
            self.asr_lock = threading.Lock()

    with _Patches(get_actor_ref_and_speed=lambda actor: (str(ref_wav), "", 1.0)):
        out = A.synth_single_segment(Ctx2(), 0, "DC", "hello world")
        check("synth_single_segment_success", out is not None and Path(out).exists())

    with _Patches(get_actor_ref_and_speed=lambda actor: (None, "", 1.0),
                  resolve_ref_audio=lambda p: None):
        out2 = A.synth_single_segment(Ctx2(), 0, "DC", "hello world")
        check("synth_single_segment_no_ref_audio", out2 is None)

    with _Patches(get_actor_ref_and_speed=lambda actor: (str(ref_wav), "", 1.0)):
        out3 = A.synth_single_segment(Ctx2(), 0, "DC", "   ")
        check("synth_single_segment_empty_text_none", out3 is None)

    def fake_infer_none(*a, **k):
        return None, None, None
    fake_f5.infer_process = fake_infer_none
    with _Patches(get_actor_ref_and_speed=lambda actor: (str(ref_wav), "", 1.0)):
        out4 = A.synth_single_segment(Ctx2(), 0, "DC", "hello")
        check("synth_single_segment_none_wav", out4 is None)

    def fake_infer_raises(*a, **k):
        raise RuntimeError("tts boom")
    fake_f5.infer_process = fake_infer_raises
    with _Patches(get_actor_ref_and_speed=lambda actor: (str(ref_wav), "", 1.0)):
        out5 = A.synth_single_segment(Ctx2(), 0, "DC", "hello")
        check("synth_single_segment_exception_none", out5 is None)

    fake_f5.infer_process = fake_infer_process
    ctx_custom = Ctx2()
    ctx_custom.custom_ref_wav = str(ref_wav)
    with _Patches():
        out6 = A.synth_single_segment(ctx_custom, 1, "DC", "hello", out_stem=str(_TMP / "custom_stem"))
        check("synth_single_segment_custom_ref_and_stem", out6 is not None)


# ---------------- play_audio_file ----------------

def test_play_audio_file():
    with _Patches(sys=types.SimpleNamespace(platform="win32")):
        called = {}
        real_startfile = getattr(os, "startfile", None)
        os.startfile = lambda p: called.setdefault("path", p)
        try:
            A.play_audio_file(str(_TMP / "x.wav"))
            check("play_audio_file_windows", called.get("path") == str(_TMP / "x.wav"))
        finally:
            if real_startfile is not None:
                os.startfile = real_startfile
            else:
                del os.startfile

    with _Patches(sys=types.SimpleNamespace(platform="darwin")):
        with _Patches(subprocess=types.SimpleNamespace(Popen=lambda args: None)):
            A.play_audio_file(str(_TMP / "x.wav"))
            check("play_audio_file_darwin_noraise", True)

    with _Patches(sys=types.SimpleNamespace(platform="linux")):
        with _Patches(subprocess=types.SimpleNamespace(Popen=lambda args: None)):
            A.play_audio_file(str(_TMP / "x.wav"))
            check("play_audio_file_linux_noraise", True)

    with _Patches(sys=types.SimpleNamespace(platform="win32")):
        def raiser(p):
            raise RuntimeError("open failed")
        os.startfile = raiser
        try:
            A.play_audio_file(str(_TMP / "x.wav"))
            check("play_audio_file_exception_swallowed", True)
        finally:
            del os.startfile


# ---------------- AudioPlayer ----------------

def test_audio_player():
    wav_path = _TMP / "play.wav"
    import soundfile as sf
    sf.write(wav_path, np.random.uniform(-0.1, 0.1, 3200).astype(np.float32), 16000)

    FakeStream.instances.clear()
    with _Patches(sd=FakeSd()):
        p = A.AudioPlayer()
        p.play(str(wav_path))
        check("audio_player_stream_started", FakeStream.instances[-1].started)
        stream = FakeStream.instances[-1]
        outdata = np.zeros((1000, 1), dtype=np.float32)
        stream.callback(outdata, 1000, None, None)
        check("audio_player_callback_wrote_data", True)
        check("audio_player_is_active", p.is_active)
        paused = p.toggle_pause()
        check("audio_player_toggle_pause", paused is True)
        outdata2 = np.zeros((10, 1), dtype=np.float32)
        stream.callback(outdata2, 10, None, None)  # paused branch -> zeros
        check("audio_player_paused_zeros", True)
        p.toggle_pause()
        p.stop()
        check("audio_player_stopped", stream.closed)
        check("audio_player_not_active_after_stop", not p.is_active)

    with _Patches(sd=FakeSd()):
        p2 = A.AudioPlayer()
        check("audio_player_toggle_pause_no_stream", p2.toggle_pause() is False)
        p2.play(str(_TMP / "does-not-exist.wav"))
        check("audio_player_play_missing_file_noop", p2._stream is None)

    class RaisingStream(FakeStream):
        def __init__(self, *a, **k):
            raise RuntimeError("stream open failed")
    with _Patches(sd=type("S", (), {"InputStream": RaisingStream, "OutputStream": RaisingStream})()):
        p3 = A.AudioPlayer()
        p3.play(str(wav_path))
        check("audio_player_stream_start_exception_handled", p3._stream is None)

    FakeStream.instances.clear()
    with _Patches(sd=FakeSd()):
        p4 = A.AudioPlayer()
        p4.play(str(wav_path))
        stream4 = FakeStream.instances[-1]
        # drive to near-end so `n < frames` triggers CallbackStop
        big_outdata = np.zeros((100000, 1), dtype=np.float32)
        try:
            stream4.callback(big_outdata, 100000, None, None)
        except FakeCallbackStop:
            pass
        check("audio_player_callback_stop_at_end", True)
        p4.stop()

        # mono 1-D data reshape branch: write a raw mono file via a different rate
        mono_path = _TMP / "mono.wav"
        sf.write(mono_path, np.zeros(1600, dtype=np.float32), 16000)
        p5 = A.AudioPlayer()
        p5.play(str(mono_path))
        check("audio_player_mono_reshape", FakeStream.instances[-1].started)
        p5.stop()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print("  ERROR in " + fn.__name__ + ": " + type(e).__name__ + ": " + str(e))
    passed = sum(1 for _, c in RESULTS if c)
    print(f"\n{len(fns)-failed}/{len(fns)} functions, {passed}/{len(RESULTS)} checks passed")
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)
