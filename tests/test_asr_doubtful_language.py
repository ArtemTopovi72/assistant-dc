"""A short mumbled clip is re-read in the user's language when Whisper's
language guess is a coin toss.

Live 2026-09-12 (journey 22): a one-word "Угу." voice note was detected as
French (69%), transcribed as a goodbye, and the bot said goodbye back. With
auto-detection on, a guess under ASR_LANG_CONFIDENCE that disagrees with the
session language triggers one forced re-read; a confident guess, an agreeing
guess, or a forced WHISPER_LANGUAGE never does.
"""
import os, sys, threading, types
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import audio as A

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

calls = []
class FakeWhisper:
    def __init__(self, lang, prob): self.lang, self.prob = lang, prob
    def transcribe(self, src, language=None, **kw):
        calls.append(language)
        info = types.SimpleNamespace(language=self.lang, language_probability=self.prob)
        text = "Au revoir" if language is None else "Угу"
        return [types.SimpleNamespace(text=text)], info

class Ctx:
    def __init__(self, w):
        self.models = types.SimpleNamespace(whisper=w); self.asr_lock = threading.Lock()

A.WHISPER_LANGUAGE = None
A._use_gigaam = lambda: False

calls.clear()
t = A._whisper_transcribe(Ctx(FakeWhisper("fr", 0.69)), "x.wav", lang_hint="ru")
check("a doubtful foreign guess is re-read in the hinted language", calls == [None, "ru"] and t == "Угу", (calls, t))

calls.clear()
t = A._whisper_transcribe(Ctx(FakeWhisper("fr", 0.97)), "x.wav", lang_hint="ru")
check("a confident guess stands (a Russian user may speak French)", calls == [None] and t == "Au revoir", calls)

calls.clear()
t = A._whisper_transcribe(Ctx(FakeWhisper("ru", 0.55)), "x.wav", lang_hint="ru")
check("a doubtful guess that AGREES with the hint is not re-read", calls == [None], calls)

calls.clear()
t = A._whisper_transcribe(Ctx(FakeWhisper("fr", 0.3)), "x.wav")
check("no hint → no re-read", calls == [None], calls)

calls.clear()
A.WHISPER_LANGUAGE = "en"
t = A._whisper_transcribe(Ctx(FakeWhisper("fr", 0.3)), "x.wav", lang_hint="ru")
check("a forced WHISPER_LANGUAGE is never second-guessed", calls == ["en"], calls)
A.WHISPER_LANGUAGE = None

check("the threshold is a real bar, not a token", 0.7 <= A.ASR_LANG_CONFIDENCE <= 0.95, A.ASR_LANG_CONFIDENCE)

src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(A.__file__))), "bot/tg_resolve.py"), encoding="utf-8").read()
check("the Telegram voice path passes the session language", src.count('lang_hint=(sess.lang or "ru")') >= 2)
src2 = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(A.__file__))), "bot/tg_tasks.py"), encoding="utf-8").read()
check("and defaults to the house language", 'lang_hint=lang_hint or "ru"' in src2)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
