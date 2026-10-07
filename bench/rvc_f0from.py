"""RVC with the pitch taken from ANOTHER recording: the words come from `src` (our lines spoken
and laid syllable for syllable on the song), the melody from `f0src` (the original singer's
vocal stem, same length). No WORLD resynthesis, no SoulX in between -- both blurred the words.
    venv_applio/Scripts/python.exe bench/rvc_f0from.py src.wav f0src.wav out.wav model.pth index [pitch]"""
import os, sys
src, f0src, out, pth, index = [os.path.abspath(a) for a in sys.argv[1:6]]
shift = int(sys.argv[6]) if len(sys.argv) > 6 else 0
A = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models_ext", "applio")
os.chdir(A); sys.path.insert(0, A)
import numpy as np, librosa
from scipy import signal
from rvc.infer.pipeline import Pipeline
from rvc.infer.infer import VoiceConverter

ref, _ = librosa.load(f0src, sr=16000)
inp, _ = librosa.load(src, sr=16000)
n = max(len(ref), len(inp))
ref, inp = np.pad(ref, (0, n - len(ref))), np.pad(inp, (0, n - len(inp)))
_orig = Pipeline.get_f0


def get_f0(self, x, p_len, *a, **kw):
    """The melody's F0, read off the original vocal padded exactly like `x`; silent where
    our input is silent (a breath stays a breath, not a hum)."""
    bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)
    r = np.pad(signal.filtfilt(bh, ah, ref), (self.t_pad, self.t_pad), mode="reflect")[: len(x)]
    coarse, f0 = _orig(self, r, p_len, *a, **kw)
    hop = self.window
    rms = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(len(f0))])
    quiet = rms < 0.01 * max(1e-6, np.percentile(rms, 95))
    f0[quiet] = 0
    coarse[quiet] = 1
    return coarse, f0


Pipeline.get_f0 = get_f0
VoiceConverter().convert_audio(audio_input_path=src, audio_output_path=out, model_path=pth, index_path=index,
                               pitch=shift, f0_method="rmvpe", index_rate=0.4, volume_envelope=1.0,
                               protect=0.5, hop_length=128, split_audio=False, f0_autotune=False,
                               clean_audio=False, export_format="WAV", embedder_model="contentvec", sid=0)
print("saved", out, flush=True)
