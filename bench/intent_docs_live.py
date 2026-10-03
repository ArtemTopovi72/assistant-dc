"""Live check of intent.doc_question: with the user's documents indexed, which
messages go to retrieval. The phrases of tests/test_rag_skips_tool_requests.py
(the word list it replaced: tg_tasks._NOT_A_DOC_QUESTION_RE).

    venv/Scripts/python bench/intent_docs_live.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [ROOT] + [os.path.join(ROOT, d) for d in ("agent", "core", "bot", "imaging", "media", "knowledge")] + [p for p in sys.path if os.path.basename(p) != "bench"]
sys.stdout.reconfigure(encoding="utf-8")
os.environ.pop("F5_TEST_RUN", None)
import tg_tasks as T

BYPASS = [
    "нарисуй витрину пекарни с вывеской «ХЛЕБ У ДОМА»", "замени надпись на «ХЛЕБ И СОЛЬ»",
    "сделай эту картинку ночной, с включённой подсветкой вывески", "а теперь сделай это фото в стиле аниме",
    "draw a lighthouse on a cliff during a storm", "upscale", "outpaint", "[upscale]",
    "какая погода в Казани завтра?", "найди, какая самая высокогорная дорога в Европе",
    "напиши текст песни про осень, четыре строки", "сочини ещё одну песню про зиму на минуту",
    "сделай презентацию на 4 слайда про эту дорогу", "запомни: меня зовут Марат, я вегетарианец",
    "забудь всё, что я про себя рассказывал", "https://ru.wikipedia.org/wiki/Омлет", "👍",
    "и то же самое по-английски", "переведи этот список на английский", "короче",
]
RETRIEVE = [
    "можно ли с собакой?", "за сколько дней надо предупредить о выезде?", "а залог возвращают?",
    "прочитай и скажи главное в трёх пунктах", "сколько стоит аренда в месяц?",
    "что написано в пункте 4.2?", "какой товар самый дорогой?",
    "составь из этого список покупок на следующую неделю",
    "кто подписал договор со стороны арендодателя?",
]
bad = 0
for t in BYPASS:
    ok = T._plainly_not_a_doc_question(t); bad += not ok
    print(("PASS " if ok else "FAIL ") + "bypass " + t)
for t in RETRIEVE:
    ok = not T._plainly_not_a_doc_question(t); bad += not ok
    print(("PASS " if ok else "FAIL ") + "retrieve " + t)
print(f"{len(BYPASS) + len(RETRIEVE) - bad}/{len(BYPASS) + len(RETRIEVE)}")
sys.exit(1 if bad else 0)
