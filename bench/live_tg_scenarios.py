"""Scenarios for bench/live_tg_drive.py -- fake users, real everything else.

    venv/Scripts/python.exe bench/live_tg_scenarios.py chat music picture voice two_users

Each scenario prints the turn-by-turn wire and a timing line, and leaves a
`report.json` next to the transcript with what was checked and what looked
wrong, so the findings can be turned into fixes and then into suites.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# bench/ holds scripts named like app modules (knowledge.py): keep it off the path
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
sys.path.insert(0, ROOT)
import bench.live_tg_drive as D  # noqa: E402  (patches requests by URL)

FINDINGS: list[dict] = []
TIMES: list[dict] = []


def note(scenario: str, what: str, **extra):
    FINDINGS.append({"scenario": scenario, "what": what, **extra})
    print(f"  !! {scenario}: {what} {extra if extra else ''}")


def timed(scenario: str, label: str, t0: float, evs=None, **extra):
    """Wall time from the user's message to the LAST thing the bot sent."""
    end = max([e["t"] for e in (evs or [])] or [time.time()])
    dt = round(end - t0, 1)
    TIMES.append({"scenario": scenario, "label": label, "s": dt, **extra})
    print(f"  == {scenario}/{label}: {dt}s")
    return dt


_CTX = None
_HEARD: dict[str, str] = {}


def texts(evs):
    """What the user reads: messages, captions, and voice notes as Whisper hears them."""
    out = []
    for e in evs:
        t = e["payload"].get("text") or e["payload"].get("caption") or ""
        if e["method"] == "sendVoice" and e["file"] and _CTX is not None:
            if e["file"] not in _HEARD:
                _HEARD[e["file"]] = D.transcribe(_CTX, e["file"])
            t = "🔊 " + _HEARD[e["file"]]
        if e["method"] in ("sendMessage", "sendVoice") or e["payload"].get("caption"):
            out.append(t)
    return out


HALLUCINATION_MARKERS = (
    "я задумался", "(no response)", "не выдал ответ", "[TOOL ERROR]",
    "<tool_call", "```json", "final_answer", "analysis", "<|", "tool_call>",
)


def sanity(scenario: str, evs, *, expect_file: str = ""):
    """Everything a user would notice: control tokens, empty turns, missing media."""
    for t in texts(evs):
        low = t.lower()
        for m in HALLUCINATION_MARKERS:
            if m.lower() in low:
                note(scenario, "reply carries a marker", marker=m, text=t[:200])
        if not t.strip():
            note(scenario, "empty text message sent")
    if expect_file and not D.files_of(evs, expect_file):
        note(scenario, f"no {expect_file} delivered", replies=texts(evs)[-3:])
    if not evs:
        note(scenario, "no reply at all")


# ---------------------------------------------------------------------------
def sc_chat(bot, ctx):
    u = D.Chat(bot, 910001, "Lena")
    for q in ("привет! ты кто?",
              "сколько будет 17 умножить на 23? просто число",
              "а теперь напомни, о чём я спрашивала до этого"):
        t0 = time.time(); u.say(q)
        evs = u.wait(quiet=6)
        timed("chat", q[:30], t0, evs)
        sanity("chat", evs)
        reply = [t for t in texts(evs) if not t.startswith("⚙️")]
        print("     reply:", (reply[-1] if reply else "")[:300])
        if "391" in q or "17" in q:
            if not any("391" in r for r in reply):
                note("chat", "arithmetic wrong", reply=reply[-1:] )
    return u


def sc_music(bot, ctx):
    u = D.Chat(bot, 910002, "Dima")
    t0 = time.time()
    u.say("сочини короткую весёлую песню про кота, который ждёт ужина. секунд 30, не длиннее")
    # The song flow runs on its own thread, outside the task queue, so the
    # chat never looks busy: wait on the audio itself.
    evs = u.wait(until=lambda e: e["method"] in ("sendAudio", "sendDocument"),
                 timeout=900, quiet=400)
    dt = timed("music", "song", t0, evs)
    sanity("music", evs, expect_file="sendAudio")
    aud = D.files_of(evs, "sendAudio")
    if aud:
        try:
            import soundfile as sf
            info = sf.info(aud[-1]); print(f"     audio: {info.duration:.1f}s {info.samplerate}Hz")
            if info.duration > 75:
                note("music", "song much longer than asked", duration=info.duration)
        except Exception as exc:
            print("     (could not read audio:", exc, ")")
    # every intermediate status line the user saw
    for t in texts(evs):
        print("     msg:", t[:200].replace("\n", " | "))
    return u


def sc_render(bot, ctx):
    """One plain scene, no lettering: the render must be accepted first time."""
    u = D.Chat(bot, 910007, "Vika")
    t0 = time.time()
    u.say("нарисуй рыжего кота, который спит на подоконнике, за окном зима")
    evs = u.wait(until=lambda e: e["method"] == "sendPhoto", timeout=900, quiet=12)
    dt = timed("render", "render", t0, evs)
    sanity("render", evs, expect_file="sendPhoto")
    if dt > 150:
        note("render", "a plain scene took more than one render's worth of time", s=dt)
    for t in texts(evs):
        print("     msg:", t[:200].replace("\n", " | "))
    return u


def sc_picture(bot, ctx):
    u = D.Chat(bot, 910003, "Olya")
    t0 = time.time()
    u.say("нарисуй рыжего кота, который спит на подоконнике, за окном зима")
    evs = u.wait(until=lambda e: e["method"] == "sendPhoto", timeout=900, quiet=12)
    timed("picture", "render", t0, evs)
    sanity("picture", evs, expect_file="sendPhoto")
    for t in texts(evs):
        print("     msg:", t[:200].replace("\n", " | "))
    # a follow-up edit on the picture the bot just sent
    t0 = time.time()
    u.say("сделай кота чёрным, остальное не трогай")
    evs2 = u.wait(until=lambda e: e["method"] == "sendPhoto", timeout=900, quiet=12)
    timed("picture", "edit", t0, evs2)
    sanity("picture", evs2, expect_file="sendPhoto")
    for t in texts(evs2):
        print("     msg:", t[:200].replace("\n", " | "))
    return u


def sc_voice(bot, ctx):
    """A voice note in, a voice note back; both transcribed with the bot's Whisper."""
    u = D.Chat(bot, 910004, "Sasha")
    from audio import synth_single_segment  # the bot's own voice asks the question
    from graph import ASSISTANT_ACTOR
    q = "Привет. Расскажи в двух предложениях, почему небо голубое."
    wav = synth_single_segment(ctx, -1, ASSISTANT_ACTOR, q,
                               out_stem=str(D.OUT / "question"), use_censoring=False)
    assert wav, "TTS made no question"
    ogg = str(D.OUT / "question.ogg")
    os.system(f'ffmpeg -y -loglevel error -i "{wav}" -c:a libopus "{ogg}"')
    heard = D.transcribe(ctx, wav)
    print("     the question as Whisper hears it:", heard)
    t0 = time.time()
    u.voice(ogg)
    evs = u.wait(until=lambda e: e["method"] == "sendVoice", timeout=600, quiet=10)
    timed("voice", "roundtrip", t0, evs)
    sanity("voice", evs)
    vs = D.files_of(evs, "sendVoice")
    if vs:
        back = D.transcribe(ctx, vs[-1])
        print("     the answer as Whisper hears it:", back[:400])
        if len(back.split()) < 6:
            note("voice", "answer voice note is nearly empty", heard=back)
        if not any(w in back.lower() for w in ("рассе", "свет", "синий", "голуб", "атмосф", "рэле", "рэл")):
            note("voice", "voice answer does not talk about the sky", heard=back[:200])
    else:
        note("voice", "no voice reply to a voice note", replies=texts(evs)[-3:])
    for t in texts(evs):
        print("     msg:", t[:200].replace("\n", " | "))
    return u


def sc_two_users(bot, ctx):
    """One user renders, another chats mid-render: the second must be told
    honestly that the card is busy and then get an answer."""
    a = D.Chat(bot, 910005, "Kostya")
    b = D.Chat(bot, 910006, "Masha")
    t0 = time.time()
    a.say("нарисуй маяк на скале в шторм")
    time.sleep(20)   # let the render claim the card
    t1 = time.time()
    b.say("привет, как тебя зовут?")
    evs_b = b.wait(timeout=900, quiet=6)
    timed("two_users", "chat during render", t1, evs_b)
    sanity("two_users", evs_b)
    for t in texts(evs_b):
        print("     B msg:", t[:200].replace("\n", " | "))
    evs_a = a.wait(until=lambda e: e["method"] == "sendPhoto", timeout=900, quiet=12)
    timed("two_users", "render", t0, evs_a)
    sanity("two_users", evs_a, expect_file="sendPhoto")
    for t in texts(evs_a):
        print("     A msg:", t[:200].replace("\n", " | "))
    return a, b


def sc_photo(bot, ctx):
    """A photo with a caption question, then a follow-up about the same photo."""
    u = D.Chat(bot, 910008, "Igor")
    src = os.path.join(D.ROOT, "runtime", "bottle_v3_524263226.png")
    t0 = time.time()
    u.photo(src, caption="что написано на этикетке и какого цвета бутылка?")
    evs = u.wait(timeout=600, quiet=8)
    timed("photo", "caption question", t0, evs)
    sanity("photo", evs)
    reply = " ".join(texts(evs)).lower()
    if not any(w in reply for w in ("пролетар", "театр", "гост")):
        note("photo", "the label text was not read back", reply=reply[:200])
    if not any(w in reply for w in ("зелен", "зелён", "тёмн", "темн", "чёрн", "черн")):
        note("photo", "the bottle colour was not named", reply=reply[:200])
    for t in texts(evs):
        print("     msg:", t[:200].replace("\n", " | "))
    t0 = time.time()
    u.say("а сколько на ней строк текста?")
    evs = u.wait(timeout=600, quiet=8)
    timed("photo", "follow-up", t0, evs)
    sanity("photo", evs)
    for t in texts(evs):
        print("     msg:", t[:200].replace("\n", " | "))
    return u


def sc_search(bot, ctx):
    u = D.Chat(bot, 910009, "Nina")
    t0 = time.time()
    u.say("найди в интернете, какой сейчас курс доллара к рублю по ЦБ, коротко")
    evs = u.wait(timeout=900, quiet=10)
    timed("search", "web", t0, evs)
    sanity("search", evs)
    reply = " ".join(texts(evs))
    import re as _re
    if not _re.search(r"\d{2,3}[.,]?\d*", reply):
        note("search", "no number in a rate answer", reply=reply[:200])
    for t in texts(evs):
        print("     msg:", t[:300].replace("\n", " | "))
    return u


def sc_memory(bot, ctx):
    u = D.Chat(bot, 910010, "Petya")
    t0 = time.time()
    u.say("запомни: моего кота зовут Барсик, ему 4 года")
    evs = u.wait(timeout=300, quiet=6)
    timed("memory", "remember", t0, evs)
    sanity("memory", evs)
    t0 = time.time()
    u.say("как зовут моего кота и сколько ему лет?")
    evs = u.wait(timeout=300, quiet=6)
    timed("memory", "recall", t0, evs)
    sanity("memory", evs)
    reply = " ".join(texts(evs)).lower()
    if "барсик" not in reply or not any(w in reply for w in ("4", "четыре", "четыр")):
        note("memory", "the fact did not come back", reply=reply[:200])
    for t in texts(evs):
        print("     msg:", t[:200].replace("\n", " | "))
    return u


def sc_weather(bot, ctx):
    u = D.Chat(bot, 910011, "Anya")
    t0 = time.time()
    u.say("какая погода в Казани завтра?")
    evs = u.wait(timeout=600, quiet=8)
    timed("weather", "typed", t0, evs)
    sanity("weather", evs)
    reply = " ".join(texts(evs)).lower()
    if not any(w in reply for w in ("градус", "°", "температур", "дожд", "ясно", "облач", "солнеч", "снег")):
        note("weather", "no weather in the answer", reply=reply[:200])
    for t in texts(evs):
        print("     msg:", t[:300].replace(chr(10), " | "))
    return u


def sc_stop(bot, ctx):
    """Stop pressed mid-render: nothing must be delivered, and the bot must
    say so once, not claim work it is not doing."""
    u = D.Chat(bot, 910012, "Gleb")
    t0 = time.time()
    u.say("нарисуй космонавта на лошади на Марсе")
    time.sleep(25)
    import tg_bot as T
    u.say(T._BTN["stop"]["ru"] if isinstance(T._BTN.get("stop"), dict) else "⛔ Стоп")
    evs = u.wait(timeout=600, quiet=20)
    timed("stop", "cancel", t0, evs)
    sanity("stop", evs)
    if D.files_of(evs, "sendPhoto"):
        note("stop", "a picture was delivered after Stop")
    for t in texts(evs):
        print("     msg:", t[:200].replace(chr(10), " | "))
    return u


def sc_sign(bot, ctx):
    """Lettering in the picture: the words must come back as asked."""
    u = D.Chat(bot, 910013, "Roma")
    t0 = time.time()
    u.say("нарисуй уютное кафе с вывеской «У ОЛЬГИ» над дверью")
    evs = u.wait(until=lambda e: e["method"] == "sendPhoto", timeout=1200, quiet=12)
    timed("sign", "render", t0, evs)
    sanity("sign", evs, expect_file="sendPhoto")
    for t in texts(evs):
        print("     msg:", t[:200].replace(chr(10), " | "))
    return u


SIX = ("нарисуй: рыжий кот спит на зелёном диване, рядом торшер, на столике перед диваном "
       "чашка кофе и раскрытая книга, на стене картина с морем, в углу большой фикус")


def _buttons_of(ev) -> dict:
    """verb -> callback_data from the inline keyboard under a delivered photo."""
    raw = ev["payload"].get("reply_markup")
    if not raw:
        return {}
    kb = json.loads(raw) if isinstance(raw, str) else raw
    out = {}
    for row in kb.get("inline_keyboard") or []:
        for b in row:
            out[b["callback_data"].split(":")[0]] = b["callback_data"]
    return out


def sc_buttons(bot, ctx):
    """A six-object scene, then every inline button under the picture."""
    u = D.Chat(bot, 910014, "Zhenya")
    t0 = time.time()
    u.say(SIX)
    evs = u.wait(until=lambda e: e["method"] == "sendPhoto", timeout=1200, quiet=12)
    timed("buttons", "six objects", t0, evs)
    sanity("buttons", evs, expect_file="sendPhoto")
    photos = [e for e in evs if e["method"] == "sendPhoto"]
    if not photos:
        return u
    for t in texts(evs):
        print("     msg:", t[:200].replace(chr(10), " | "))
    kb = _buttons_of(photos[-1])
    print("     buttons:", kb)
    # BUTTONS=upscale,edit presses those; BUTTONS= (empty) renders the scene only.
    _env = os.environ.get("BUTTONS")
    verbs = tuple(v for v in (_env if _env is not None
                              else "upscale,enhance_faces,regenerate,outpaint,restore,edit").split(",") if v)
    for verb in verbs:
        data = kb.get(verb)
        if not data:
            note("buttons", "button missing under the picture", verb=verb); continue
        t0 = time.time()
        u.press(data, message_id=photos[-1]["message_id"])
        if verb == "edit":
            time.sleep(3)
            u.say("сделай кота чёрным, остальное не трогай")
        evs = u.wait(until=lambda e: e["method"] == "sendPhoto", timeout=1200, quiet=15)
        timed("buttons", verb, t0, evs)
        sanity("buttons", evs, expect_file="sendPhoto")
        for t in texts(evs):
            print(f"     [{verb}] msg:", t[:200].replace(chr(10), " | "))
        ph = [e for e in evs if e["method"] == "sendPhoto"]
        if ph:
            try:
                from PIL import Image
                im = Image.open(ph[-1]["file"]); print(f"     [{verb}] image {im.size} {ph[-1]['file']}")
            except Exception: pass
    return u


def sc_bare_photo_buttons(bot, ctx):
    """New feature (2026-09-19): a bare photo with no caption should get a
    description AND the full picture-action keyboard right under that reply,
    keyed to that exact photo -- no trip to the Creativity menu needed. A
    video note (кружок) right after must NOT pick up that keyboard: it is a
    completely separate code path and must keep behaving exactly as before."""
    u = D.Chat(bot, 910016, "Oksana")
    src = os.path.join(D.ROOT, "runtime", "bottle_v3_524263226.png")
    t0 = time.time()
    m = u.photo(src)
    evs = u.wait(timeout=600, quiet=8)
    timed("bare_photo_buttons", "bare photo", t0, evs)
    sanity("bare_photo_buttons", evs)
    reply = " ".join(texts(evs))
    if not reply.strip():
        note("bare_photo_buttons", "no description came back for the bare photo")
    for t in texts(evs):
        print("     msg:", t[:200].replace(chr(10), " | "))
    kb_evs = [e for e in evs if e["payload"].get("reply_markup")]
    if not kb_evs:
        note("bare_photo_buttons", "no keyboard was attached to the description reply")
    else:
        kb = _buttons_of(kb_evs[-1])
        want = {"upscale", "enhance_faces", "change_clothes", "regenerate", "edit",
                "outpaint", "restore", "style", "animate"}
        missing = want - set(kb)
        if missing:
            note("bare_photo_buttons", "picture-action buttons missing", missing=sorted(missing))
        ids = {v.split(":", 1)[1] for v in kb.values() if ":" in v}
        if len(ids) != 1:
            note("bare_photo_buttons", "buttons don't all key to the SAME photo", ids=ids)
        print("     buttons:", kb)

    # Now a video note -- must get the ordinary keyboard, never this one.
    t0 = time.time()
    vpath = os.path.join(D.ROOT, "runtime", "assistant_h3_00001_.mp4")
    u.video_note(vpath, seconds=6)
    evs2 = u.wait(timeout=600, quiet=8)
    timed("bare_photo_buttons", "video note after", t0, evs2)
    sanity("bare_photo_buttons", evs2)
    for t in texts(evs2):
        print("     [vnote] msg:", t[:200].replace(chr(10), " | "))
    for e in evs2:
        raw = e["payload"].get("reply_markup")
        if not raw:
            continue
        kb2 = json.loads(raw) if isinstance(raw, str) else raw
        verbs2 = {b["callback_data"].split(":")[0]
                  for row in kb2.get("inline_keyboard") or [] for b in row}
        if verbs2 & {"upscale", "enhance_faces", "change_clothes", "regenerate",
                     "outpaint", "restore", "style", "animate"}:
            note("bare_photo_buttons", "the video-note reply got the PICTURE keyboard", verbs=sorted(verbs2))
    return u


def sc_document(bot, ctx):
    """A text file with a caption question, then a question that needs the file."""
    u = D.Chat(bot, 910015, "Lida")
    doc = os.path.join(D.OUT, "recipe.txt")
    open(doc, "w", encoding="utf-8").write(chr(10).join([
        "Рецепт бабушкиного пирога", "",
        "Тесто: 300 г муки, 150 г масла, 2 яйца, щепотка соли.",
        "Начинка: 5 яблок сорта антоновка, 100 г сахара, корица.",
        "Выпекать 45 минут при 180 градусах. Подавать тёплым со сметаной.",
        "Секрет: яблоки нарезать тонко и сбрызнуть лимонным соком, чтобы не потемнели.", ""]))
    t0 = time.time()
    u.document(doc, caption="сколько яблок нужно и какого сорта?")
    evs = u.wait(timeout=600, quiet=8)
    timed("document", "caption question", t0, evs)
    sanity("document", evs)
    reply = " ".join(texts(evs)).lower()
    if not ("антонов" in reply and any(w in reply for w in ("5", "пять"))):
        note("document", "the answer did not come from the file", reply=reply[:200])
    for t in texts(evs):
        print("     msg:", t[:200].replace(chr(10), " | "))
    t0 = time.time()
    u.say("а в чём секрет, чтобы яблоки не потемнели?")
    evs = u.wait(timeout=600, quiet=8)
    timed("document", "follow-up", t0, evs)
    sanity("document", evs)
    reply = " ".join(texts(evs)).lower()
    if "лимон" not in reply:
        note("document", "the follow-up lost the file", reply=reply[:200])
    for t in texts(evs):
        print("     msg:", t[:200].replace(chr(10), " | "))
    return u


def sc_fwd_voice(bot, ctx):
    """A forwarded voice note: the bot must ask what to do, then do it."""
    u = D.Chat(bot, 910016, "Mark")
    from audio import synth_single_segment
    from graph import ASSISTANT_ACTOR
    q = ("Привет, это Серёжа. Встречаемся завтра в семь у метро Пушкинская, "
         "возьми с собой документы на машину и не опаздывай.")
    wav = synth_single_segment(ctx, -1, ASSISTANT_ACTOR, q,
                               out_stem=str(D.OUT / "fwd"), use_censoring=False)
    ogg = str(D.OUT / "fwd.ogg")
    os.system(f'ffmpeg -y -loglevel error -i "{wav}" -c:a libopus "{ogg}"')
    t0 = time.time()
    u.voice(ogg, forwarded=True)
    evs = u.wait(timeout=300, quiet=6)
    timed("fwd_voice", "ask", t0, evs)
    sanity("fwd_voice", evs)
    for t in texts(evs):
        print("     msg:", t[:200].replace(chr(10), " | "))
    kb = {}
    for e in evs:
        kb.update(_buttons_of(e))
    print("     buttons:", kb)
    t0 = time.time()
    u.say("перескажи коротко")
    evs = u.wait(timeout=300, quiet=6)
    timed("fwd_voice", "summary", t0, evs)
    sanity("fwd_voice", evs)
    reply = " ".join(texts(evs)).lower()
    if not any(w in reply for w in ("пушкин", "семь", "7", "документ")):
        note("fwd_voice", "the summary lost the content", reply=reply[:200])
    for t in texts(evs):
        print("     msg:", t[:200].replace(chr(10), " | "))
    return u


def sc_two_draws(bot, ctx):
    """Two users draw at once: the second must be told its queue position and
    both must get their own picture."""
    a = D.Chat(bot, 910017, "Oleg")
    b = D.Chat(bot, 910018, "Ira")
    t0 = time.time()
    a.say("нарисуй красный велосипед у кирпичной стены")
    time.sleep(4)
    b.say("нарисуй синюю лодку на озере на рассвете")
    evs_b = b.wait(until=lambda e: e["method"] == "sendPhoto", timeout=1500, quiet=15)
    timed("two_draws", "second user", t0, evs_b)
    sanity("two_draws", evs_b, expect_file="sendPhoto")
    for t in texts(evs_b):
        print("     B msg:", t[:200].replace(chr(10), " | "))
    evs_a = a.wait(until=lambda e: e["method"] == "sendPhoto", timeout=1500, quiet=15)
    timed("two_draws", "first user", t0, evs_a)
    sanity("two_draws", evs_a, expect_file="sendPhoto")
    for t in texts(evs_a):
        print("     A msg:", t[:200].replace(chr(10), " | "))
    return a, b


def sc_stop_song(bot, ctx):
    """Stop pressed while a song renders."""
    u = D.Chat(bot, 910019, "Fedya")
    t0 = time.time()
    u.say("сочини песню про дождь, 30 секунд")
    time.sleep(20)
    u.say("⛔ Стоп")
    evs = u.wait(until=lambda e: e["method"] == "sendAudio", timeout=600, quiet=120)
    timed("stop_song", "stop", t0, evs)
    sanity("stop_song", evs)
    if D.files_of(evs, "sendAudio"):
        note("stop_song", "the song was delivered after Stop")
    for t in texts(evs):
        print("     msg:", t[:200].replace(chr(10), " | "))
    return u


def _presets_of(ev, prefix: str) -> list[str]:
    """Every `<prefix>:<key>` callback_data under a message's inline keyboard,
    in on-screen order. _buttons_of collapses same-prefix buttons to one
    (it keys by the FIRST ":"-segment), which hides all but the last preset
    -- useless for a keyboard that is nothing but a row of presets."""
    raw = ev["payload"].get("reply_markup")
    if not raw:
        return []
    kb = json.loads(raw) if isinstance(raw, str) else raw
    out = []
    for row in kb.get("inline_keyboard") or []:
        for b in row:
            if b["callback_data"].startswith(prefix + ":"):
                out.append(b["callback_data"])
    return out


def sc_character_animate(bot, ctx):
    """Real user path end to end: pick a trained character (LoRA), draw a
    scene with it, ASK about the picture that came back, then feed that same
    picture into the new 🎞 Animate photo flow and check a clip comes back.

    Cross-tests three things at once, the way a real user actually chains
    them: character rendering, the vision/analyze path on a BOT-drawn (not
    user-uploaded) picture, and the animate-preset flow reusing a picture
    that never came from the user's camera roll.
    """
    import tg_bot as T
    import characters as _chars
    u = D.Chat(bot, 910021, "Vera")
    avail = _chars.telegram_characters()
    if not avail:
        note("character_animate", "no trained character available to test with")
        return u
    slug = avail[0]["slug"]

    u.say(T._b("creativity", "ru")); u.wait(quiet=3)
    u.say(T._b("characters_btn", "ru"))
    evs = u.wait(quiet=5)
    kb_msgs = [e for e in evs if e["payload"].get("reply_markup")]
    if not kb_msgs or not any(f"char:{slug}" in json.dumps(e["payload"]["reply_markup"])
                              for e in kb_msgs):
        note("character_animate", "character list did not offer the trained slug", slug=slug)
        return u
    t0 = time.time()
    u.press(f"char:{slug}")
    u.wait(quiet=3)
    u.say("на пляже строит замок из песка, солнечный день")
    evs = u.wait(until=lambda e: e["method"] == "sendPhoto", timeout=900, quiet=12)
    timed("character_animate", "character render", t0, evs)
    sanity("character_animate", evs, expect_file="sendPhoto")
    photos = [e for e in evs if e["method"] == "sendPhoto"]
    if not photos:
        note("character_animate", "no picture came back for the character render")
        return u
    render_path = photos[-1]["file"]
    for t in texts(evs):
        print("     msg:", t[:200].replace(chr(10), " | "))
    # the item-15 regression check: this delivery MUST carry the image
    # keyboard and a resolvable message_id, or a reply below means nothing
    kb = _buttons_of(photos[-1])
    if not kb:
        note("character_animate", "character render was delivered with NO button row",
             file=render_path)

    # ASK about the picture the bot itself just drew -- not a user upload.
    t0 = time.time()
    u.say("что тут нарисовано, подробно опиши")
    evs = u.wait(timeout=300, quiet=8)
    timed("character_animate", "ask about own render", t0, evs)
    sanity("character_animate", evs)
    reply = " ".join(texts(evs)).lower()
    if not any(w in reply for w in ("пляж", "песок", "замок", "солнеч")):
        note("character_animate", "the analysis did not describe the character render",
             reply=reply[:200])
    print("     reply:", reply[:300].replace(chr(10), " | "))

    # Feed the SAME picture into 🎞 Animate photo -- a real user re-sends a
    # picture the bot drew (forward, or download-then-upload) rather than
    # having any other way to hand it back in.
    t0 = time.time()
    u.say(T._b("animate_btn", "ru"))
    u.wait(quiet=3)
    u.photo(render_path)
    evs = u.wait(quiet=5)
    preset_msgs = [e for e in evs if _presets_of(e, "animate_preset")]
    if not preset_msgs:
        note("character_animate", "no animate-preset keyboard after sending the photo",
             replies=texts(evs)[-3:])
        return u
    presets = _presets_of(preset_msgs[-1], "animate_preset")
    u.press(presets[0], message_id=preset_msgs[-1]["message_id"])
    evs = u.wait(until=lambda e: e["method"] in ("sendVideo", "sendAnimation", "sendDocument"),
                 timeout=1200, quiet=20)
    timed("character_animate", "animate", t0, evs)
    sanity("character_animate", evs)
    clip = (D.files_of(evs, "sendVideo") or D.files_of(evs, "sendAnimation")
            or D.files_of(evs, "sendDocument"))
    if not clip:
        note("character_animate", "no clip delivered for the animate preset",
             replies=texts(evs)[-3:])
        return u
    frame = str(D.OUT / "character_animate_frame.jpg")
    os.system(f'ffmpeg -y -loglevel error -ss 0.3 -i "{clip[-1]}" -frames:v 1 "{frame}"')
    if os.path.exists(frame) and os.path.getsize(frame) > 0:
        print("     animate clip:", clip[-1], "-> frame saved:", frame)
    else:
        note("character_animate", "clip delivered but a frame could not be pulled from it",
             clip=clip[-1])
    for t in texts(evs):
        print("     msg:", t[:200].replace(chr(10), " | "))
    return u


def sc_presentation_regression(bot, ctx):
    """The exact historical failure (live 2026-09-18): a topic that once made
    the model call generate_video instead of create_presentation, or call no
    tool at all on retry. Presses the REAL 📊 Presentation button -- forced-
    tool coverage in the unit suites checks the button TEXT, never whether an
    actual .pptx lands in the chat and NOT a video."""
    import tg_bot as T
    u = D.Chat(bot, 910022, "Grisha")
    u.say(T._b("creativity", "ru")); u.wait(quiet=3)
    t0 = time.time()
    u.say(T._b("deck", "ru"))
    evs = u.wait(quiet=4)
    sanity("presentation", evs)
    u.say("гайд как правильно бездельничать на работе и не спалиться")
    evs = u.wait(until=lambda e: e["method"] in ("sendDocument", "sendVideo", "sendAnimation"),
                 timeout=900, quiet=15)
    timed("presentation", "deck", t0, evs)
    sanity("presentation", evs)
    if D.files_of(evs, "sendVideo") or D.files_of(evs, "sendAnimation"):
        note("presentation", "the deck button produced a VIDEO instead of a presentation",
             replies=texts(evs)[-3:])
    docs = D.files_of(evs, "sendDocument")
    if not docs:
        note("presentation", "no document delivered for a presentation request",
             replies=texts(evs)[-3:])
    elif not docs[-1].lower().endswith((".pptx", ".ppt", ".pdf")):
        note("presentation", "delivered document is not a presentation file", file=docs[-1])
    for t in texts(evs):
        print("     msg:", t[:200].replace(chr(10), " | "))
    return u


def sc_style_menu_standalone(bot, ctx):
    """Item 7: 🎭 Change style from the Creativity menu, with NO picture
    already in the chat -- must ask for a photo first, then offer presets."""
    import tg_bot as T
    u = D.Chat(bot, 910023, "Oleg")
    u.say(T._b("creativity", "ru")); u.wait(quiet=3)
    t0 = time.time()
    u.say(T._b("style_menu_btn", "ru"))
    evs = u.wait(quiet=4)
    sanity("style_menu", evs)
    asked_for_photo = any("фото" in t.lower() for t in texts(evs))
    if not asked_for_photo:
        note("style_menu", "pressing Change style did not ask for a photo",
             replies=texts(evs)[-3:])
    u.photo(os.path.join(D.ROOT, "runtime", "bottle_v3_524263226.png"))
    evs = u.wait(quiet=5)
    preset_msgs = [e for e in evs if _presets_of(e, "style_preset")]
    if not preset_msgs:
        note("style_menu", "no style-preset keyboard after sending a fresh photo",
             replies=texts(evs)[-3:])
        return u
    presets = _presets_of(preset_msgs[-1], "style_preset")
    u.press(presets[0], message_id=preset_msgs[-1]["message_id"])
    evs = u.wait(until=lambda e: e["method"] == "sendPhoto", timeout=900, quiet=15)
    timed("style_menu", "preset", t0, evs)
    sanity("style_menu", evs, expect_file="sendPhoto")
    for t in texts(evs):
        print("     msg:", t[:200].replace(chr(10), " | "))
    return u


def sc_video_direct(bot, ctx):
    """A video asked for in plain words, no button -- and the clip is
    actually LOOKED at (a pulled frame), not just "a file arrived"."""
    u = D.Chat(bot, 910024, "Katya")
    t0 = time.time()
    u.say("сделай короткое видео: бумажный кораблик плывёт по ручью")
    evs = u.wait(until=lambda e: e["method"] in ("sendVideo", "sendAnimation", "sendDocument"),
                 timeout=1200, quiet=20)
    timed("video", "t2v", t0, evs)
    sanity("video", evs)
    clip = (D.files_of(evs, "sendVideo") or D.files_of(evs, "sendAnimation")
            or D.files_of(evs, "sendDocument"))
    if not clip:
        note("video", "no clip delivered", replies=texts(evs)[-3:])
        return u
    frame = str(D.OUT / "video_direct_frame.jpg")
    os.system(f'ffmpeg -y -loglevel error -ss 0.3 -i "{clip[-1]}" -frames:v 1 "{frame}"')
    if not (os.path.exists(frame) and os.path.getsize(frame) > 0):
        note("video", "clip delivered but no frame could be pulled from it", clip=clip[-1])
    else:
        print("     video clip:", clip[-1], "-> frame saved:", frame)
    for t in texts(evs):
        print("     msg:", t[:200].replace(chr(10), " | "))
    return u


def sc_misc(bot, ctx):
    """Small everyday asks: translate, a joke, a list, a reply-to-picture edit."""
    u = D.Chat(bot, 910020, "Tanya")
    import tg_bot as T
    u.say(T._b("voice_off", "ru") if hasattr(T, "_b") else "🔇 Выключить голос")
    u.wait(quiet=3)
    for q, want in (("переведи на английский: у меня сегодня хорошее настроение", ("mood", "good")),
                    ("расскажи короткий анекдот про программиста", ("програм", "код", "баг")),
                    ("составь список покупок для борща, только список", ("свекл", "свёкл", "капуст"))):
        t0 = time.time(); u.say(q)
        evs = u.wait(timeout=300, quiet=6)
        timed("misc", q[:30], t0, evs)
        sanity("misc", evs)
        reply = " ".join(t for t in texts(evs) if not t.startswith("⚙️")).lower()
        if not any(w in reply for w in want):
            note("misc", "reply misses the point", q=q[:40], reply=reply[:200])
        print("     reply:", reply[:200].replace(chr(10), " | "))
    # analyze-photo button first, THEN the photo (the armed order)
    t0 = time.time()
    u.say(T._b("analyze", "ru"))
    u.wait(quiet=3)
    u.photo(os.path.join(D.ROOT, "runtime", "bottle_v3_524263226.png"))
    evs = u.wait(timeout=600, quiet=8)
    timed("misc", "analyze armed then photo", t0, evs)
    sanity("misc", evs)
    reply = " ".join(t for t in texts(evs) if not t.startswith("⚙️")).lower()
    if "бутыл" not in reply and "шампан" not in reply:
        note("misc", "the armed analysis did not describe the photo", reply=reply[:200])
    print("     reply:", reply[:300].replace(chr(10), " | "))
    return u


def sc_cover(bot, ctx):
    """🎤 Кавер: the button, a song as an audio FILE, new words -> a cover comes back."""
    u = D.Chat(bot, 910031, "Kostya")
    src = sorted(Path(ROOT, "runtime").glob("song_yue2_*.flac"))[-1]
    u.say("🎤 Кавер")
    evs = u.wait(timeout=60, quiet=4)
    print("     msg:", " | ".join(t[:120] for t in texts(evs)))
    u.document(str(src))
    evs = u.wait(timeout=120, quiet=6)
    print("     msg:", " | ".join(t[:120] for t in texts(evs)))
    if not any("cover:keep" in json.dumps(e["payload"], ensure_ascii=False) for e in evs):
        note("cover", "no «keep the words» button after the song was sent")
    t0 = time.time()
    u.say("Стиль: рок, мужской вокал\nКот сидит у окна\nЖдёт хозяйку до темна\nМиска пуста, но он не спит\nИ тихо песенку мурчит")
    evs = u.wait(until=lambda e: e["method"] in ("sendAudio", "sendDocument"), timeout=1200, quiet=900)
    timed("cover", "cover", t0, evs)
    sanity("cover", evs, expect_file="sendAudio")
    for t in texts(evs):
        print("     msg:", t[:200].replace("\n", " | "))
    return u


def sc_sandbox_links(bot, ctx):
    """File names in a sandbox reply are t.me deep links; tapping one sends the file."""
    import re as _re
    import code_sandbox as _cs
    import sandbox_access as _sa
    u = D.Chat(bot, 910032, "Lena")
    usr = bot._user_store.get(u.id); usr.prefs = dict(usr.prefs or {}, **{_sa.PREFS_KEY: _sa.FILES})
    bot._user_store.put(usr)
    box = _cs.sandbox_for(u.id)
    (Path(box.root) / "notes.txt").write_text("купить молоко\n", encoding="utf-8")
    (Path(box.root) / "report.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    u.say("📁 Песочница")
    u.wait(timeout=60, quiet=4)
    t0 = time.time()
    u.say("какие файлы лежат у меня в песочнице? перечисли их имена")
    evs = u.wait(timeout=300, quiet=10)
    timed("sandbox_links", "list", t0, evs)
    sanity("sandbox_links", evs)
    html = "\n".join(e["payload"].get("text") or "" for e in evs if e["method"] == "sendMessage")
    print("     reply:", html[:400].replace("\n", " | "))
    toks = _re.findall(r'href="https://t\.me/[^?"]+\?start=(f_[0-9a-f]{16})"', html)
    if not toks:
        note("sandbox_links", "no file link in the reply", reply=html[:500])
        return u
    u.say("/start " + toks[0])
    evs = u.wait(until=lambda e: e["method"] == "sendDocument", timeout=60, quiet=5)
    got = D.files_of(evs, "sendDocument") + [e["payload"]["document"] for e in evs
             if e["method"] == "sendDocument" and str(e["payload"].get("document", "")).startswith("file://")]
    print("     tapped ->", got, [t[:80] for t in texts(evs)])
    if not got:
        note("sandbox_links", "tapping the link sent no file", evs=[e["method"] for e in evs])
    return u


def sc_silent_video(bot, ctx):
    """A forwarded кружок with no speech, then 📋: retold from what is seen
    (live 16:11 it asked «пришлите ТРАНСКРИПТ»). The clip is a real one kept
    out of the repo: runtime/live_media/silent_kruzhok.mp4."""
    clip = Path(ROOT) / "runtime" / "live_media" / "silent_kruzhok.mp4"
    if not clip.exists():
        note("silent_video", "no clip at " + str(clip)); return
    u = D.Chat(bot, 910021, "Влад")
    u.video_note(str(clip), forwarded=True, seconds=7)
    evs = u.wait(timeout=300, quiet=60, until=lambda e: "fwdv:sum:" in json.dumps(e["payload"]))
    sum_cb = next((m for e in evs for m in re.findall(r"fwdv:sum:\w+", json.dumps(e["payload"]))), "")
    if not sum_cb:
        note("silent_video", "no 📋 button", texts=texts(evs)); return
    u.press(sum_cb)
    evs = u.wait(timeout=300, quiet=20, until=lambda e: "📋" in json.dumps(e["payload"], ensure_ascii=False))
    reply = " ".join(texts(evs))
    print("     reply:", reply[:400].replace(chr(10), " | "))
    if re.search(r"транскрипт|пришлите|пришли текст", reply, re.I) or len(reply) < 80:
        note("silent_video", "asked for a transcript instead of retelling", reply=reply[:300])


def sc_deds_voices(bot, ctx):
    """Live 2026-10-01: a fight of two grandfathers voiced with the user's own two
    samples (a voice note and a round video) -- the agent asks, the samples come
    in through the buttons, the clip renders. Nothing here is scripted for the model."""
    u = D.Chat(bot, 910077, "Artem")
    pic = os.path.join(ROOT, "runtime", "generated", "Artem", "Изображения", "ideogram_00055_.png")
    u.photo(pic)
    u.wait(timeout=180, quiet=8)
    t0 = time.time()
    u.say("Анимируй: два деда дерутся, рыжий дед говорит \"Да иди ты на хуй!\", а второй дед говорит "
          "\"Сам иди нахуй! Ты вообще старый пидр\". Деды это говорят не переставая драться. Играет на фоне "
          "эпичный баян как в боевике и звуки ударов очень сочные.")
    evs = u.wait(until=lambda e: "anv:yes" in json.dumps(e["payload"]), timeout=600, quiet=20)
    print("     bot:", texts(evs))
    if not any("anv:yes" in json.dumps(e["payload"]) for e in evs):
        note("deds_voices", "no voice question with buttons"); return
    u.press("anv:yes"); u.wait(timeout=60, quiet=4)
    u.voice(os.path.join(ROOT, "anim_voices", os.environ.get("ANIM_VOICES_CHAT", ""), "voice1.ogg")); u.wait(timeout=120, quiet=5)
    u.video_note(os.path.join(ROOT, "anim_voices", os.environ.get("ANIM_VOICES_CHAT", ""), "voice2.mp4")); u.wait(timeout=120, quiet=5)
    u.press("anv:done")
    evs = u.wait(until=lambda e: e["method"] == "sendVideo", timeout=1800, quiet=60)
    timed("deds_voices", "clip", t0, evs)
    print("     bot:", texts(evs))
    vids = [e["file"] for e in evs if e["method"] == "sendVideo" and e["file"]]
    print("     VIDEO:", vids)
    if not vids:
        note("deds_voices", "no clip delivered"); return
    sims = voice_similarity(vids[-1], [os.path.join(ROOT, "anim_voices", os.environ.get("ANIM_VOICES_CHAT", ""), f)
                                       for f in ("voice1.ogg", "voice2.mp4")])
    print("     voice similarity to the samples:", sims)
    if max(sims) < VOICE_SIM_MIN:
        note("deds_voices", "the clip does not sound like the user's samples", sims=sims)


VOICE_SIM_MIN = 0.65   # measured 10-01: right voices 0.68, lost voices 0.60 (with music on top)


def voice_similarity(clip: str, samples: list) -> list:
    """Speaker-embedding cosine of the clip's sound vs each sample (Resemblyzer).
    Crude with music and two speakers on one track, but it separates a clip
    that took the samples from one that did not -- no listening by hand."""
    import subprocess
    from resemblyzer import VoiceEncoder, preprocess_wav
    enc = VoiceEncoder("cpu")

    def emb(p):
        w = str(D.OUT / "_sim.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p, "-vn", "-ac", "1", "-ar", "16000", w])
        return enc.embed_utterance(preprocess_wav(w))
    c = emb(clip)
    return [round(float(c @ emb(s)), 3) for s in samples]


def sc_video_link(bot, ctx):
    """A YouTube Shorts link is watched (live 10-03: «не умею смотреть видео по ссылкам»)."""
    u = D.Chat(bot, 910077, "Artem")
    # the first one FORWARDED from Stepan: live 10-03 a forwarded link was not watched
    for q, fw in (("https://youtube.com/shorts/5uI-Dae7-qs", "Stepan"), ("а что там говорят?", "")):
        t0 = time.time(); u.say(q, forward_from=fw)
        evs = u.wait(quiet=8)
        timed("video_link", q[:30], t0, evs)
        reply = [t for t in texts(evs) if not t.startswith("⚙️")]
        print("     reply:", (reply[-1] if reply else "")[:500])
        if any("не умею" in r or "не могу посмотреть" in r for r in reply):
            note("video_link", "still refuses to watch", reply=reply[-1:])
    return u


def sc_clone_link(bot, ctx):
    """🎙 Clone voice from a YouTube link (owner 10-03)."""
    import tg_strings
    u = D.Chat(bot, 910078, "Artem")
    u.say(tg_strings._BTN["clone_btn"]["ru"]); u.wait(quiet=4)
    t0 = time.time(); u.say("https://youtube.com/shorts/5uI-Dae7-qs")
    evs = u.wait(quiet=10); timed("clone_link", "link", t0, evs)
    print("     link:", texts(evs)[-2:])
    t0 = time.time(); u.say("Привет, это проверка клонированного голоса.")
    evs = u.wait(quiet=10); timed("clone_link", "speak", t0, evs)
    voices = D.files_of(evs, "voice")
    print("     voices:", voices, "heard:", [D.transcribe(ctx, v) for v in voices][:1])
    return u


def sc_remove_cat(bot, ctx):
    """live 10-03: «убери кота» left a cat-shaped stain, then a pasted rectangle.
    Through the agent; the delivered picture is copied to runtime/remove_cat_live.png."""
    u = D.Chat(bot, 910021, "Kot")
    src = os.path.join(D.ROOT, "runtime", "comfy", "ideogram_00115_.png")
    t0 = time.time()
    u.photo(src, caption="убери кота")
    evs = u.wait(until=lambda e: e["method"] == "sendPhoto", timeout=900, quiet=15)
    timed("remove_cat", "remove", t0, evs)
    sanity("remove_cat", evs, expect_file="sendPhoto")
    for t in texts(evs):
        print("     msg:", t[:200].replace("\n", " | "))
    import shutil
    got = [f for k, f in D.WIRE.files.items() if k.startswith("botphoto_")]
    if got:
        shutil.copy(got[-1], os.path.join(D.ROOT, "runtime", "remove_cat_live.png"))
        print("     file:", got[-1])
    return u


def sc_audiobook(bot, ctx):
    """📚 Аудиокнига: button -> a voice sample -> a 2-chapter book as a .txt -> a voice message per chapter,
    each opening with its title, each with a caption. Heard back with the bot's own Whisper."""
    u = D.Chat(bot, 910090, "Book")
    ref = os.path.join(ROOT, "DC_short_ref12.wav")
    book = os.path.join(D.OUT, "book.txt")
    s1 = ("Старый дом стоял на краю деревни. Окна его давно потемнели, но по вечерам в них мелькал свет. "
          "Никто не знал, кто там живёт. Мальчишки обходили забор стороной и шептались о привидениях. "
          "Только почтальон каждую пятницу приносил туда письмо. Он оставлял его на крыльце и быстро уходил.")
    s2 = ("Утром пришёл дождь. Он стучал по крыше и смывал пыль с дороги. Река поднялась и подошла к самому порогу. "
          "В доме открылась дверь, и на крыльцо вышла старая женщина с лампой. Она посмотрела на небо и улыбнулась. "
          "Потом взяла письмо, села на ступеньку и стала читать его вслух самой себе.")
    NL = chr(10)
    open(book, "w", encoding="utf-8").write("Глава 1" + NL + "Старый дом" + NL + NL + s1 + NL + NL
                                           + "Глава 2" + NL + "Дождь" + NL + NL + s2 + NL)
    u.say("📚 Аудиокнига"); evs = u.wait(timeout=60, quiet=4)
    print("     msg:", " | ".join(t[:100] for t in texts(evs)))
    t0 = time.time(); u.voice(ref); evs = u.wait(timeout=240, quiet=8)
    timed("audiobook", "voice sample", t0, evs)
    print("     msg:", " | ".join(t[:100] for t in texts(evs)))
    t0 = time.time(); u.document(book)
    evs = u.wait(until=lambda e, _n=[0]: (_n.__setitem__(0, _n[0] + (e["method"] == "sendVoice")), _n[0] >= 2)[1], timeout=1500, quiet=120)
    timed("audiobook", "two chapters", t0, evs)
    voices = [e for e in evs if e["method"] == "sendVoice"]
    if len(voices) < 2:
        note("audiobook", "fewer than two chapter voices came back", got=len(voices), msgs=texts(evs)[-4:])
    for i, e in enumerate(voices, 1):
        heard = D.transcribe(ctx, e["file"]) if e["file"] else ""
        cap = e["payload"].get("caption", "")
        print(f"     ch{i}: caption={cap!r} heard={heard[:140]!r}")
        if "глава" not in heard.lower()[:40]:
            note("audiobook", f"chapter {i} does not open with its title", heard=heard[:80])
        if not cap:
            note("audiobook", f"chapter {i} has no caption")
    sanity("audiobook", evs)
    return u


SCENARIOS = {"audiobook": sc_audiobook, "remove_cat": sc_remove_cat, "video_link": sc_video_link, "clone_link": sc_clone_link, "deds_voices": sc_deds_voices, "cover": sc_cover, "sandbox_links": sc_sandbox_links, "photo": sc_photo, "misc": sc_misc, "document": sc_document, "fwd_voice": sc_fwd_voice,
             "two_draws": sc_two_draws, "stop_song": sc_stop_song, "buttons": sc_buttons, "weather": sc_weather, "stop": sc_stop, "sign": sc_sign, "search": sc_search, "memory": sc_memory,
             "chat": sc_chat, "music": sc_music, "picture": sc_picture, "render": sc_render,
             "voice": sc_voice, "two_users": sc_two_users,
             "character_animate": sc_character_animate,
             "presentation_regression": sc_presentation_regression,
             "style_menu": sc_style_menu_standalone,
             "video_direct": sc_video_direct,
             "bare_photo_buttons": sc_bare_photo_buttons,
             "silent_video": sc_silent_video}


def main(names):
    global _CTX
    bot, ctx = D.build()
    _CTX = ctx
    print("bot up; output ->", D.OUT)
    try:
        for n in names:
            print("\n" + "=" * 70 + "\n" + n + "\n" + "=" * 70)
            try:
                SCENARIOS[n](bot, ctx)
            except Exception:
                traceback.print_exc()
                note(n, "scenario crashed", tb=traceback.format_exc()[-800:])
    finally:
        try: bot.stop()
        except Exception: pass
        (D.OUT / "report.json").write_text(
            json.dumps({"times": TIMES, "findings": FINDINGS}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print("\nTIMES"); [print("  ", t) for t in TIMES]
        print("FINDINGS"); [print("  ", f) for f in FINDINGS]
        print("->", D.OUT)


if __name__ == "__main__":
    main(sys.argv[1:] or ["chat"])
