"""Five fresh multi-part stories through the real generate_video handler (~6 s parts,
each continuing the last). Writes runtime/story_probe/<n>.mp4 and prints the part count."""
import os, sys, threading, time, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
for d in ("", "agent", "imaging", "media", "bot", "core"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.chdir(ROOT)
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
import models, video, tool_image_handlers as H

STORIES = [
    ("Сделай видео: мальчик во дворе лепит снежок. Он бросает снежок в друга. Снежок попадает другу в плечо, тот смеётся и бросает снежок в ответ.",
     "In a snowy yard a boy packs a snowball in his gloves. He throws the snowball at his friend. The snowball hits the friend on the shoulder; the friend laughs, scoops up snow and throws a snowball back."),
    ("Повар на кухне подбрасывает блин на сковороде. Блин падает на пол. Кот подбегает и утаскивает блин.",
     "A chef in a small kitchen flips a pancake high from a frying pan. The pancake misses the pan and falls onto the floor. A ginger cat runs in, grabs the pancake in its mouth and drags it away. The chef chases the cat around the kitchen table, slips on flour and sits down on the floor. The cat sits on the windowsill, eating the pancake, and looks at the chef."),
    ("Девушка в парке пинает футбольный мяч. Мяч летит и сбивает с дерева яблоко. Яблоко падает на голову мужчине на скамейке, он удивлённо поднимает глаза.",
     "A girl in a park kicks a football hard. The ball flies through the air and knocks an apple off a tree branch. The apple drops onto the head of a man reading on a bench below; he looks up, surprised, rubbing his head."),
    ("Старик кормит голубей хлебом. Собака вбегает и разгоняет голубей. Голуби взлетают, старик ругается на собаку.",
     "An old man on a city square tosses bread crumbs from his hand to pigeons. A shaggy dog runs into the flock barking. The pigeons burst into the air. The old man shakes his fist at the dog, shouting \"Пошла вон!\". The dog sits down in front of him and wags its tail. The old man sighs, smiles and tosses the last piece of bread to the dog, which catches it in the air."),
    ("Мужчина открывает бутылку шампанского. Пробка вылетает и попадает в люстру. Люстра качается, все за столом смеются и чокаются.",
     "At a festive dinner table a man twists the cork of a champagne bottle. The cork shoots out with a pop and hits the crystal chandelier. The chandelier swings and jingles, and a crystal drop falls into a salad bowl. A woman fishes the crystal out of the salad with a fork and holds it up. Everyone at the table laughs and clinks their glasses. The man pours champagne into the glasses, foam spilling over the rims onto the tablecloth."),
]

out_dir = os.path.join(ROOT, "runtime", "story_probe"); os.makedirs(out_dir, exist_ok=True)
H._current_image_paths = lambda c, s: []
for i, (ru, en) in enumerate(STORIES, 1):
    print(f"=== story {i}: parts={len(video.split_script(en))}", flush=True)
    ctx = models.Context(models=None, transcription_cache={}, cache_file=None,
                         asr_lock=threading.Lock(), tts_lock=threading.Lock())
    ctx.video_plan_ask, ctx.video_plan_parts = False, []
    state = {"user_input_original": ru, "user_input": en}
    t0 = time.time()
    try:
        out = H._handle_generate_video(ctx, state, {"description": en})
    except Exception as e:
        out = f"EXC {e!r}"
    p = state.get("video_path") or ""
    if p and os.path.exists(p):
        shutil.copyfile(p, os.path.join(out_dir, f"{i}.mp4"))
    print(f"RESULT {i} {str(out)[:160]} {p} {time.time()-t0:.0f}s", flush=True)
