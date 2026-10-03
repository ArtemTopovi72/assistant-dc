"""Russian typed on the English layout: «NEN ;T ERFPFYJ» -> «ТУТ ЖЕ УКАЗАНО».

Live 2026-09-26: a schoolkid argued with the bot in the wrong layout and the
bot had to guess. The whole message or only some words can be wrong
(«привет rfr дела»). A run of Latin words is converted only when it reads as
Russian and not as English: few English vowels before, enough Russian ones
after, and no common English word in it.
"""
import re

_EN = "`qwertyuiop[]asdfghjkl;'zxcvbnm,.~QWERTYUIOP{}ASDFGHJKL:\"ZXCVBNM<>"
_RU = "ёйцукенгшщзхъфывапролджэячсмитьбюЁЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ"
_MAP = str.maketrans(_EN, _RU)
_EN_WORDS = re.compile(r"(?i)\b(the|and|is|are|you|to|of|in|it|what|how|this|that|for|with|on|my|me|can|do|ok|lol)\b")
_RU_VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
# A word: letters plus the punctuation keys that are letters on the Russian layout.
_WORD = r"[;'\[\]`{}:\"<>]*[A-Za-z](?:[A-Za-z;'\[\]`{}:\"<>,.]*[A-Za-z;'\[\]`{}:\"<>])?"
_RUN = re.compile(r"(?:%s)(?:[ \t]+(?:%s))*" % (_WORD, _WORD))


def _word_votes(run: str):
    """(english, russian) summed Zipf frequency of the run's words, or None when
    neither dictionary knows any of them (typos, slang) -- then the letter
    heuristic decides. «songs» 4.8 vs «ыщтпы» 0; «ghbdtn» 0 vs «привет» 5.1."""
    try:
        from wordfreq import zipf_frequency as z
    except ImportError:
        return None
    words = re.findall(r"[a-z]+", run.lower())
    en = sum(z(w, "en") for w in words)
    ru = sum(z(w.translate(_MAP), "ru") for w in words)
    return (en, ru) if en or ru else None


def _reads_russian(run: str) -> bool:
    latin = [c for c in run if c.isalpha()]
    if len(latin) < 3:
        return False        # «48 h», «x2»: too short to judge
    if " " not in run.strip() and run.isupper() and len(latin) <= 5:
        return False        # RTX, GPU, NASA: an acronym, not a word
    votes = _word_votes(run)
    if votes:
        return votes[1] > votes[0]
    if _EN_WORDS.search(run):
        return False

    # English runs ~35-45% vowels; Russian typed on the English keys ~0-20%.
    if sum(c in "aeiouyAEIOUY" for c in latin) / len(latin) > 0.22:
        return False
    ru = [c for c in run.translate(_MAP) if c.isalpha()]
    return bool(ru) and sum(c in _RU_VOWELS for c in ru) / len(ru) >= 0.33


def fix(text: str) -> str:
    """The text with every wrong-layout run converted; unchanged if none."""
    t = text or ""
    if re.search(r"https?://|www\.|@\w|/\w", t):
        return text
    return _RUN.sub(lambda m: m.group(0).translate(_MAP) if _reads_russian(m.group(0)) else m.group(0), t)


if __name__ == "__main__":
    assert fix("NEN ;T ERFPFYJ XNJ DFHBFYN CVSCKJDJT TLBYCNDJ YT GHFDBKMYS").lower().startswith("тут же указано что вариант")
    assert fix("ghbdtn rfr ltkf") == "привет как дела"
    assert fix("привет rfr дела?") == "привет как дела?"
    assert fix("hello world, how are you") == "hello world, how are you"
    assert fix("Stable Diffusion prompt") == "Stable Diffusion prompt"
    assert fix("открой Photoshop и RTX 3090") == "открой Photoshop и RTX 3090"
    assert fix("привет") == "привет"
    assert fix("lol kek") == "lol kek"
    assert fix("🕓 24 h") == "🕓 24 h"
    assert fix("Songs") == "Songs" and fix("Depth") == "Depth"
    assert fix("Help me please") == "Help me please"
    print("ok")
