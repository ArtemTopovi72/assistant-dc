"""📚 Audiobook: chapters from the book's own headings, whole-sentence pieces, the title spoken first,
the same voice for every piece. A fake synthesiser (a tone per character) stands in for F5; real pydub/ffmpeg."""
import os
import sys
import tempfile
import zipfile

os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "media"))
import audiobook as A

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

D = tempfile.mkdtemp(prefix="abook_")
SENT = "Он шёл по улице. Было холодно, и снег падал медленно! Она ждала? Нет… "
BOOK = ("Предисловие\n\nКороткий текст автора.\n\nГлава 1\nНачало\n\n" + SENT * 30 + "\n\nГлава 2\n\n" + "Утро пришло. " * 80
        + "\n\n3\n\nМало.")

ch = A.split_chapters(BOOK)
titles = [t for t, _ in ch]
check("chapters follow the book's headings; «Глава 1» takes its own name from the next line",
      "Глава 1. Начало" in titles and "Глава 2" in titles, titles)
check("a stray short «3» is glued to the previous chapter, not read as a chapter",
      not any(t == "3" for t in titles), titles)
plain = A.split_chapters("Просто текст. " * 1500)
check("a book with no headings is cut into «Часть N»", len(plain) >= 3 and plain[0][0] == "Часть 1", [t for t, _ in plain])

pieces = A.chunk_text(ch[1][1] if ch[0][0] else ch[0][1])
check("pieces are whole sentences within the limit", all(len(p) <= A.MAX_CHUNK for p, _ in pieces)
      and all(p.rstrip()[-1] in ".!?…" for p, _ in pieces), [p for p, _ in pieces][:2])
long_sentence = "слово, " * 100 + "конец."
check("one endless sentence is cut at a comma, never mid-word",
      all(len(p) <= A.MAX_CHUNK and not p.endswith("сло") for p, _ in A.chunk_text(long_sentence)))
check("the title is said first; a book without one gets «Глава N»",
      A.spoken_title("Глава 4. Дом", 4) == "Глава 4. Дом" and A.spoken_title("", 7) == "Глава 7")

# fake voice: 60 ms per letter of 440 Hz tone, the same call every time
import subprocess
SEEN = []
def fake_synth(text):
    SEEN.append(text)
    wav = os.path.join(D, f"p{len(SEEN)}.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={max(0.5, len(text) * 0.06):.2f}", wav], check=True)
    return wav

out = os.path.join(D, "ch.wav")
okr = A.render_chapter(fake_synth, "Глава 1. Начало", "Первое предложение. Второе предложение.\nНовый абзац.", 1, out)
from pydub import AudioSegment
check("a chapter renders into one file", okr and os.path.exists(out) and len(AudioSegment.from_wav(out)) > 3000)
check("its title is spoken first", SEEN[0] == "Глава 1. Начало", SEEN[:2])

cancel = {"n": 0}
def stop_after_two():
    cancel["n"] += 1
    return cancel["n"] > 2
check("cancel stops the chapter and returns False",
      A.render_chapter(fake_synth, "", "Раз. " * 50, 1, os.path.join(D, "c.wav"), cancelled=stop_after_two) is False)
check("a voice that returns nothing fails the chapter instead of delivering silence",
      A.render_chapter(lambda t: None, "", "Раз. Два.", 1, os.path.join(D, "n.wav")) is False)

# books of each kind
fb2 = os.path.join(D, "b.fb2")
open(fb2, "w", encoding="utf-8").write(
    '<?xml version="1.0" encoding="utf-8"?><FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">'
    "<description/><body><section><title><p>Глава 1</p></title><p>Первый абзац.</p></section></body>"
    '<body name="notes"><section><p>Сноска, её не читать.</p></section></body></FictionBook>')
txt = A.read_book(fb2)
check("fb2: the story is read, the notes body is not", "Первый абзац" in txt and "Сноска" not in txt, txt)
z = os.path.join(D, "b.fb2.zip")
with zipfile.ZipFile(z, "w") as zf:
    zf.write(fb2, "b.fb2")
check("fb2.zip is read", "Первый абзац" in A.read_book(z))
t1251 = os.path.join(D, "w.txt")
open(t1251, "wb").write("Глава 1\n\nПривет, мир.".encode("cp1251"))
check("a cp1251 txt is read", "Привет" in A.read_book(t1251))
try:
    A.read_book(os.path.join(D, "x.exe")); check("an unknown type is refused", False)
except ValueError:
    check("an unknown type is refused", True)

long_wav = os.path.join(D, "long.wav")
AudioSegment.silent(duration=70_000).export(long_wav, format="wav")
parts = A.split_long(long_wav, seconds=30)
check("a long chapter is cut into parts under the limit", len(parts) >= 2 and all(len(AudioSegment.from_wav(p)) <= 36_000 for p in parts), parts)

check("citation marks [12] and superscripts are not read", "[12]" not in A.clean("Слово[12] и ещё² текст.") and "²" not in A.clean("ещё² текст"))
tone = AudioSegment.silent(duration=400) + AudioSegment.from_file(fake_synth("тест тест")) + AudioSegment.silent(duration=500)
check("the engine's own edge silence is cut (a hair stays)", len(A._trim(tone)) < len(tone) - 600, (len(A._trim(tone)), len(tone)))

print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
