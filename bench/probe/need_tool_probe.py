"""Probe&Prefill analogue: does this message need a tool? A logistic probe on
BGE-M3 embeddings of the user text, leave-one-out on the bench cases plus
extra no-tool chat. Compared with the current keyword gate (fast path)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from sklearn.linear_model import LogisticRegression
from bench.tc_cases import cases_for
from bench.tc_para import PARA
import knowledge

EXTRA_NO_TOOL = """спасибо, ты лучший
как ты думаешь, в чём смысл жизни?
объясни, что такое рекурсия, простыми словами
расскажи анекдот про программистов
напиши стих про осень
что лучше: чай или кофе по утрам?
почему небо голубое?
придумай имя для котёнка
как мне перестать прокрастинировать?
переведи на английский: я люблю тебя
что такое фотосинтез?
чем отличается класс от объекта?
дай совет, как подготовиться к собеседованию
напиши короткое поздравление с днём рождения для мамы
что ты умеешь?
мне грустно сегодня
объясни разницу между TCP и UDP
как правильно заваривать зелёный чай?
придумай слоган для пекарни
кто написал «Войну и мир»?
сколько будет 2+2?
расскажи сказку на ночь
какие плюсы у удалённой работы?
что такое чёрная дыра?
помоги сформулировать письмо начальнику об отпуске
как по-английски будет «вдохновение»?
что посоветуешь почитать из фантастики?
объясни теорему Пифагора
почему кошки мурлычут?
напиши хокку про дождь
ты человек или бот?
как успокоиться перед экзаменом?
что такое инфляция простыми словами?
придумай загадку для ребёнка
какой твой любимый цвет?
как научиться играть на гитаре с нуля?
перескажи сюжет «Ромео и Джульетты»
что значит слово «эмпатия»?
давай поболтаем
как говорить «нет» и не чувствовать вины?
в чём разница между погодой и климатом?
напиши тост на свадьбу друга
что такое ООП?
расскажи интересный факт о космосе
как составить резюме?
почему люди боятся темноты?
что такое метафора? приведи пример
окей, понял
а если подробнее?
какие бывают виды облаков?""".splitlines()

rows = [(c["text"], 0 if c["cat"] == "gate" else 1, c["cat"]) for c in cases_for(
    ["gate", "route", "args", "chain", "recover", "safety"])]
rows += [(p["text"], 1, "para") for p in PARA]
rows += [(t, 0, "chat") for t in EXTRA_NO_TOOL]
E = knowledge.Embedder(timeout=30.0)
X = np.array(E.embed([r[0] for r in rows]), dtype=float)
X /= np.linalg.norm(X, axis=1, keepdims=True)
y = np.array([r[1] for r in rows])
cats = [r[2] for r in rows]
p = np.zeros(len(rows))
for i in range(len(rows)):
    m = np.ones(len(rows), bool); m[i] = False
    clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced").fit(X[m], y[m])
    p[i] = clf.predict_proba(X[i:i + 1])[0, 1]
import graph, graph_fastpath as F
def kw_tool(t):   # current gate: a trigger word or image intent => full loop
    return bool(graph._TOOL_TRIGGER_RE.search(t) or F._IMAGE_INTENT_RE.search(t) or F._hard_arithmetic(t))
for thr in (0.3, 0.4, 0.5, 0.6):
    pred = p >= thr
    out = {}
    for c in sorted(set(cats)):
        idx = [i for i, cc in enumerate(cats) if cc == c]
        out[c] = f"{sum(pred[i] == y[i] for i in idx)}/{len(idx)}"
    print("probe thr", thr, out)
kw = np.array([kw_tool(r[0]) for r in rows])
out = {c: f"{sum(kw[i] == y[i] for i in [j for j, cc in enumerate(cats) if cc == c])}/{cats.count(c)}" for c in sorted(set(cats))}
print("keyword gate    ", out)
both = kw | (p >= 0.5)
out = {c: f"{sum(both[i] == y[i] for i in [j for j, cc in enumerate(cats) if cc == c])}/{cats.count(c)}" for c in sorted(set(cats))}
print("keyword OR probe", out)
print("para misses:", [rows[i][0] for i in range(len(rows)) if cats[i] == "para" and p[i] < 0.5])
print("chat false alarms:", [(rows[i][0], round(p[i], 2)) for i in range(len(rows)) if y[i] == 0 and p[i] >= 0.5])

if "--save" in sys.argv:
    clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced").fit(X, y)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "need_tool_probe.json")
    json.dump({"w": clf.coef_[0].tolist(), "b": float(clf.intercept_[0]), "threshold": 0.5,
               "embed_model": "text-embedding-bge-m3", "n_train": int(len(y)),
               "note": "bench/probe/need_tool_probe.py --save; LOO: para 15/15 route 27/27 "
                       "chat 48/50 gate 17/20 at 0.5 (probe alone)"},
              open(path, "w", encoding="utf-8"))
    print("saved", path)
