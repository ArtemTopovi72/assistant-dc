"""A URL in a voiced reply is spoken as 'ссылка', never spelled out.

Live 2026-09-13 (mega journey, step 85): 'дай ссылку на источник' with voice
on was read as 'Вплюс отс плюс ссылка на официальный сайт...' — the scheme,
slashes and Latin host are outside what the Russian voice can say. The text
reply keeps the link.
"""
import os, sys
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import audio as A

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

out = A.post_process_answer("Вот ссылка на сайт ЦБ: https://www.cbr.ru/about_br/ — там всё есть.")
check("no scheme in the spoken text", "http" not in out and "://" not in out, out)
check("no Latin host in the spoken text", "cbr" not in out, out)
check("'ссылка' already said -> the URL is simply dropped", out.count("ссылк") == 1, out)

out = A.post_process_answer("Источник: https://cbr.ru/press/ (официальный сайт).")
check("a bare URL is voiced as 'ссылка'", "ссылка" in out and "cbr" not in out, out)

out = A.post_process_answer("Подробнее на www.cbr.ru и https://t.me/cbr_official")
check("www. and t.me forms too", "www" not in out and "t.me" not in out and out.count("ссылка") == 2, out)

out = A.post_process_answer("Инфляция составила 7% в 2026 году.")
check("text without links is untouched by the URL pass", "семь процентов" in out and "ссылк" not in out, out)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
