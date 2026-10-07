"""Live: a two-part chef script through the real bot (Herrgott chain + stitch), then
▶️ Continue on the delivered clip (sent back as Telegram would) goes on from its latent.
Stop the desktop bot first; ComfyUI stays up.
    venv/Scripts/python.exe -u bench/chain_drive.py"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live_tg_drive as D

SCRIPT = ("Сделай видео: седой мужчина в синем фартуке стоит у деревянного стола, берёт большой нож и "
          "быстро режет морковь, говоря «Ну что, начнём готовить!». Потом он сбрасывает нарезанную "
          "морковь в кипящую кастрюлю, мешает деревянной ложкой, пробует, морщится, досаливает и "
          "говорит «Соли маловато, сейчас исправим».")
bot, ctx = D.build()
u = D.Chat(bot, 910099, "Chain")
s = bot._get_session(u.id); s.voice_choice = "default"; bot._store.put(s)
t0 = time.time()
u.say(SCRIPT)
evs = u.wait(until=lambda e: e["method"] == "sendVideo", timeout=3600, quiet=120)
vids = D.files_of(evs, "sendVideo")
print(f"after script ({time.time() - t0:.0f}s):", D.final_text(evs)[:400].replace("\n", " | "))
print("VIDEO", vids)
if not vids:
    os._exit(1)
s = bot._get_session(u.id); s.continue_state = "want_video"; bot._store.put(s)
fid = f"vid_{u.id}_{int(time.time() * 1000)}.mp4"
D.WIRE.files[fid] = Path(vids[-1])
D.WIRE.push({"message": u._msg(video={"file_id": fid, "duration": 15, "mime_type": "video/mp4",
                                     "width": 768, "height": 1024})})
evs = u.wait(timeout=120, quiet=6)
print("after clip:", D.final_text(evs)[:300].replace("\n", " | "))
t0 = time.time()
u.say("Он улыбается, поворачивается к камере и говорит «Вот теперь отлично!»")
evs = u.wait(until=lambda e: e["method"] == "sendVideo", timeout=2400, quiet=120)
print(f"after continue ({time.time() - t0:.0f}s):", D.final_text(evs)[:400].replace("\n", " | "))
print("VIDEO", D.files_of(evs, "sendVideo"))
os._exit(0)
