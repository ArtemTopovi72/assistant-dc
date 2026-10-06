"""Live: a photo of a person, 👗 under it, then a photo of the clothes -> the person re-dressed.
    venv/Scripts/python.exe bench/outfit_drive.py person.jpg clothes.jpg ["caption"]"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live_tg_drive as D

person, clothes = sys.argv[1], sys.argv[2]
cap = sys.argv[3] if len(sys.argv) > 3 else ""
bot, ctx = D.build()
u = D.Chat(bot, 910077, "Outfit")


def kb_of(ev):
    raw = ev["payload"].get("reply_markup")
    kb = json.loads(raw) if isinstance(raw, str) else (raw or {})
    return {b["callback_data"].split(":")[0]: b["callback_data"]
            for row in kb.get("inline_keyboard") or [] for b in row if b.get("callback_data")}


u.photo(person)
evs = u.wait(until=lambda e: "change_clothes" in kb_of(e), timeout=600, quiet=15)
hit = [e for e in evs if "change_clothes" in kb_of(e)]
print("after photo:", D.final_text(evs)[:300].replace("\n", " | "))
if not hit:
    print("NO 👗 BUTTON"); os._exit(1)
u.press(kb_of(hit[-1])["change_clothes"], message_id=hit[-1]["message_id"])
evs = u.wait(timeout=60, quiet=4)
print("after 👗:", D.final_text(evs)[:300].replace("\n", " | "))
t0 = time.time()
u.photo(clothes, caption=cap)
evs = u.wait(until=lambda e: e["method"] == "sendPhoto", timeout=1500, quiet=30)
print(f"after clothes ({time.time() - t0:.0f}s):", D.final_text(evs)[:500].replace("\n", " | "))
for f in D.files_of(evs, "sendPhoto"):
    print("RESULT", f)
os._exit(0)
