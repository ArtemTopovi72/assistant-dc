"""Live: agent/ask_read.py -- exact text operations, length, earlier talk, doses, code."""
import os,sys
os.environ["INTENT_LIVE"]="1"
R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("","agent","core","research","imaging","bot","media","voice","gui","knowledge"): sys.path.insert(0,os.path.join(R,d))
import graph_compose as g
from graph_compose import no_history_note
C=[("сколько букв о в слове 'обороноспособность'?","7 times"),
("сколько букв о в слове обороноспособность?","7 times"),
("напиши слово 'программирование' задом наперёд","еинавориммаргорп"),
("сколько слов в предложении: 'Мама мыла раму, а папа читал газету вечером дома'?","words"),
("how many r's in 'strawberry'?","3 times"),
(" ".join(["купил молоко."]*120)+" Сколько раз я написал слово молоко?","120 times"),
("сколько букв в алфавите?",""),("переведи «hello» на русский",""),
("отсортируй по алфавиту: яблоко, груша, ёжевика, ежевика, абрикос, вишня","абрикос, вишня, груша, ежевика, ёжевика, яблоко"),
("вес 14 кг, сколько парацетамола дать?","paracetamol 140-210 mg"),("как дела?",""),
("сколько будет 15% от 2340 плюс НДС 20%?","15% of 2340 = 351")]
bad=0
for t,w in C:
    got=g._word_count_note(t); ok=(w in got) if w else got==""
    bad+=not ok; print("ok " if ok else "BAD", t[-60:], "->", got[:110])
for t,w in [("инструкцию на 1200 слов",3600),("расскажи подробно",2600),("привет, 2025 год",0)]:
    v=g._asked_tokens(t); ok=(v>=w if w else v==0); bad+=not ok; print("ok " if ok else "BAD",t,v)
for t,w in [("о каком городе мы говорили?",True),("какая столица Канады?",False),("что я спросил первым?",True)]:
    v=bool(no_history_note(t,[])); ok=v==w; bad+=not ok; print("ok " if ok else "BAD",t,v)
class B: sandbox=object()
for t,w in [("напиши скрипт, который считает строки в csv",True),("fix the bug in main.py",True),("какая погода?",False),("нарисуй кота",False)]:
    v=bool(g._ponytail_block(B(),t)); ok=v==w; bad+=not ok; print("ok " if ok else "BAD",t,v)
print("ALL OK" if not bad else f"{bad} wrong")
