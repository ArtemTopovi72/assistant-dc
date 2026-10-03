"""Russian lines spoken in a clip carry stress marks (H3 guessed stress wrong)."""
import os, sys, types
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(R, d) for d in ("media", "agent", "services", "core", "voice", "")]
import video as V

class Acc:   # RUAccent's contract: '+' before the stressed vowel
    def __call__(self, t):
        return t.replace("иди", "ид+и").replace("замок", "з+амок").replace("ты", "т+ы")
ctx = types.SimpleNamespace(models=types.SimpleNamespace(accentor=Acc(), accentor_loaded=True))
out = V.mark_speech_stress(ctx, 'He yells "иди ты" over the замок, «замок»')
print(out)
assert '"иди́ ты"' in out, out                   # stressed, one-vowel word left clean
assert "«за́мок»" in out, out
assert "over the замок" in out, out                     # only spoken lines, not the scene
assert "+" not in out
print("ok")

# the Context-IR rewrite writes spoken lines as <d>[Russian] ...</d>
out2 = V.mark_speech_stress(ctx, "shouts, <d>[Russian] иди ты</d> and runs")
assert "<d>[Russian] иди́ ты</d>" in out2, out2
print("d-tag ok")

# the real RUAccent drops "<", "/" and quotes; the tags must survive anyway (live 10-02: "хуй!d>")
class Lossy(Acc):
    def __call__(self, t):
        return super().__call__(t).replace("<", "").replace("/", "").replace("«", "").replace("»", "")
ctx3 = types.SimpleNamespace(models=types.SimpleNamespace(accentor=Lossy(), accentor_loaded=True))
out3 = V.mark_speech_stress(ctx3, "shouts, <d>[Russian] иди ты!</d> then «замок»")
assert "<d>[Russian] иди́ ты!</d>" in out3, out3
assert "«за́мок»" in out3, out3
print("lossy accentor ok")
