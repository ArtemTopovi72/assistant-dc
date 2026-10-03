"""Unscripted users: the model PLAYS a person and uses the real bot.

The scripted journeys (live_journeys.py) type what a developer expects a user
to type. Real users do not: they tap whatever button is there, type in the
wrong layout, change their mind, cancel, spam, write «stop», expect a button to
do what its label says. 2026-09-27: a night of green suites, and the first
real user got «🎵 Songs» -> «Ыщтпы» -> a redrawn photo -> a Cancel that did
nothing. This harness sits a PERSONA in front of the bot:

  * it sees only what a Telegram user sees: the bot's texts (HTML stripped),
    [photo]/[voice]/[file] marks, the reply keyboard, inline buttons;
  * each step it picks ONE action (type / tap a keyboard button / tap an inline
    button / send a photo / wait) and says how it feels and why;
  * a persona "types" like its person: the kid slips into the English layout,
    drops punctuation, makes typos;
  * at the end it writes a review in its own voice, and scripts/ux_audit.py
    runs over the transcript.

Findings = every step where the persona was confused/annoyed, the review, and
the auditor's ERRORs. Nothing here decides pass/fail; a person reads the report.

Stop the desktop app first (it holds Whisper/F5 and the real bot token), then:
    venv/Scripts/python.exe -u bench/persona_users.py [--only kid,granny] [--steps 14]
Report: runtime/live_drive/<stamp>/persona_report.md
"""
from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live_tg_drive as D  # noqa: E402  (fakes api.telegram.org on import)

ROOT = D.ROOT
ASSETS = os.path.join(ROOT, "bench", "assets")
PHOTOS = {"desk": os.path.join(ASSETS, "man_at_desk.jpg"),
          "receipt": os.path.join(ASSETS, "receipt.jpg"),
          "carpet": os.path.join(ASSETS, "red_carpet.jpg")}

PERSONAS = {
    "kid": dict(
        lang="ru", chat=930001, name="Stepa",
        who="Школьник 13 лет. Ленивый, нетерпеливый, пишет коротко, без заглавных и "
            "знаков, со сленгом («го», «чё», «норм»). Кнопки жмёт наугад, чтобы посмотреть "
            "что будет. Если долго — жмёт ещё раз или пишет «ну чё».",
        goal="Хочет песню-прикол про друга Тёму, потом смешную картинку, потом "
             "переделать эту картинку. Если что-то не работает — пробует по-другому.",
        typing=dict(layout=0.15, typo=0.1, lower=True)),
    "granny": dict(
        lang="ru", chat=930002, name="Valentina",
        who="Пенсионерка 70 лет. Пишет длинно, вежливо, с «здравствуйте» и «спасибо», "
            "не понимает что такое кнопки и инлайн-кнопки, иногда пишет названия кнопок "
            "словами. Путается, если ответ длинный или с непонятными значками.",
        goal="Узнать погоду на завтра в Твери, потом спросить, что написано на фото чека "
             "(сколько потратила), потом попросить говорить с ней голосом.",
        typing=dict(layout=0.0, typo=0.05, lower=False)),
    "busy_en": dict(
        lang="en", chat=930003, name="Mark",
        who="An impatient adult who set the English interface. Short messages. If "
            "something takes long he presses Cancel or types stop and tries again "
            "differently. Hates vague answers.",
        goal="Get a picture of a cozy coffee shop for his flyer, then fix one thing in "
             "it, then find out how to download it as a file without compression.",
        typing=dict(layout=0.0, typo=0.05, lower=False)),
    "troll": dict(
        lang="ru", chat=930004, name="Kolyan",
        who="Проверяльщик-тролль 17 лет. Пытается сломать бота: спамит одно и то же, "
            "шлёт бессмыслицу и эмодзи, спрашивает «ты чатгпт?», просит невозможное, "
            "резко меняет тему, жмёт Стоп посреди работы.",
        goal="Найти, где бот отвечает глупо, врёт, зависает или ломается.",
        typing=dict(layout=0.1, typo=0.1, lower=True)),
    "student": dict(
        lang="ru", chat=930005, name="Alina",
        who="Студентка 20 лет. Пишет нормально, по делу. Ждёт, что бот сделает то, "
            "что обещают кнопки меню.",
        goal="Сделать презентацию на 5 слайдов про экологию города, потом поправить "
             "один слайд, потом спросить что-то по учёбе с поиском в интернете.",
        typing=dict(layout=0.0, typo=0.03, lower=False)),
}

_EN = "`qwertyuiop[]asdfghjkl;'zxcvbnm,."
_RU = "ёйцукенгшщзхъфывапролджэячсмитьбю"
_TO_EN = str.maketrans(_RU + _RU.upper(), _EN + _EN.upper())


def humanize(text: str, typing: dict, rng: random.Random) -> str:
    """Type it the way this person types: wrong layout, lower case, typos."""
    t = text
    if typing.get("lower"):
        t = t.lower().rstrip(".!")
    if rng.random() < typing.get("typo", 0) and len(t) > 6:
        i = rng.randrange(1, len(t) - 1)
        t = t[:i] + t[i + 1] + t[i] + t[i + 2:]
    if rng.random() < typing.get("layout", 0) and re.search("[а-яё]", t, re.I):
        t = t.translate(_TO_EN)       # forgot to switch the layout
    return t


def _plain(t: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", t or ""))


class Screen:
    """What the person sees in their Telegram client."""

    def __init__(self):
        self.lines: list[str] = []
        self.keyboard: list[str] = []
        self.inline: list[tuple[str, str, int]] = []   # (label, data, message_id)
        self.bubbles: dict[int, int] = {}               # message_id -> line index (edited status)

    def feed(self, evs: list[dict]) -> None:
        for e in evs:
            p, m = e["payload"], e["method"]
            mk = p.get("reply_markup")
            if isinstance(mk, str):
                try: mk = json.loads(mk)
                except ValueError: mk = None
            if isinstance(mk, dict) and mk.get("keyboard"):
                self.keyboard = [b if isinstance(b, str) else b.get("text", "")
                                 for row in mk["keyboard"] for b in row]
            if isinstance(mk, dict) and "inline_keyboard" in mk:
                mid = e.get("message_id") or p.get("message_id") or 0
                self.inline = [x for x in self.inline if x[2] != mid]
                self.inline += [(b.get("text", ""), b.get("callback_data", ""), mid)
                                for row in mk["inline_keyboard"] for b in row
                                if b.get("callback_data")]
            text = _plain(p.get("text") or p.get("caption") or "")
            kind = {"sendPhoto": "[фото]", "sendVoice": "[голосовое]", "sendAudio": "[аудио]",
                    "sendDocument": "[файл " + os.path.basename(e.get("file") or "") + "]",
                    "sendVideo": "[видео]"}.get(m, "")
            if m == "editMessageText" and p.get("message_id") in self.bubbles:
                self.lines[self.bubbles[p["message_id"]]] = "БОТ (статус): " + text
                continue
            if m == "deleteMessage":            # gone from the screen, buttons and all
                idx = self.bubbles.pop(p.get("message_id"), None)
                if idx is not None:
                    self.lines[idx] = ""
                self.inline = [x for x in self.inline if x[2] != p.get("message_id")]
                continue
            if text or kind:
                self.lines.append("БОТ: " + " ".join(x for x in (kind, text) if x)[:900])
                if e.get("message_id"):
                    self.bubbles[e["message_id"]] = len(self.lines) - 1

    def user(self, what: str) -> None:
        self.lines.append("Я: " + what)

    def render(self) -> str:
        kb = " | ".join(self.keyboard) or "(нет)"
        il = " | ".join(f"[{i}] {l}" for i, (l, _d, _m) in enumerate(self.inline[-8:])) or "(нет)"
        return ("ЧАТ (последнее):\n" + "\n".join([l for l in self.lines if l][-18:]) +
                f"\n\nКНОПКИ ВНИЗУ ЭКРАНА: {kb}\nКНОПКИ ПОД СООБЩЕНИЯМИ: {il}")


_ACT = """Ты играешь человека, который пользуется Telegram-ботом. Не помогай боту и не
будь вежливее своего персонажа. Действуй как этот человек на самом деле.

КТО ТЫ: {who}
ЧЕГО ХОЧЕШЬ: {goal}
Прошло шагов: {step} из {steps}.

{screen}

Выбери ОДНО действие. Ответь ТОЛЬКО JSON:
{{"action": "type" | "key" | "inline" | "photo" | "wait" | "quit",
  "text": "что пишешь (для type) или точная надпись кнопки внизу (для key)",
  "index": номер кнопки под сообщением (для inline),
  "photo": "desk" | "receipt" | "carpet" (для photo, text = подпись),
  "feeling": "ok" | "confused" | "annoyed" | "happy",
  "why": "одна фраза от первого лица: что ты видишь и почему так поступаешь"}}
wait = подождать, бот ещё работает. quit = ты всё получил или сдался."""

_REVIEW = """Ты — этот человек: {who}
Ты хотел(а): {goal}
Вот вся переписка с ботом:
{chat}

Напиши отзыв от первого лица, как написал бы этот человек другу (3-6 предложений):
что получилось, где запутался, что бесило, что было непонятно. Потом строкой
«ОЦЕНКА: N/5». Потом «ГЛАВНЫЕ ПРОБЛЕМЫ:» и 1-5 пунктов, каждый — что именно
произошло на экране (цитата), без советов программисту."""


def _ask(ctx, prompt: str, max_tokens=500, temperature=0.8) -> str:
    import llm
    r = llm.send_to_lm_studio(ctx, [{"role": "user", "content": prompt}],
                              temperature=temperature, max_tokens=max_tokens)
    return (r or {}).get("content") or ""


def _json(s: str) -> dict:
    try:
        from utils import safe_json_from_llm
        v = safe_json_from_llm(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        try: return json.loads(m.group(0)) if m else {}
        except ValueError: return {}


def run_persona(bot, ctx, key: str, steps: int, rng: random.Random) -> dict:
    P = PERSONAS[key]
    chat = D.Chat(bot, P["chat"], P["name"], P["lang"])
    screen, log = Screen(), []
    chat.say("/start"); screen.user("/start"); screen.feed(chat.wait(timeout=60))
    for step in range(1, steps + 1):
        raw = _ask(ctx, _ACT.format(who=P["who"], goal=P["goal"], step=step, steps=steps,
                                    screen=screen.render()))
        a = _json(raw)
        act, text = a.get("action", "wait"), str(a.get("text") or "")
        entry = {"step": step, "action": act, "text": text, "feeling": a.get("feeling", ""),
                 "why": a.get("why", ""), "t": time.strftime("%H:%M:%S")}
        print(f"[{key} {step}] {act} {text!r} ({entry['feeling']}) {entry['why'][:100]}")
        if act == "quit":
            log.append(entry); break
        if act == "type" and text:
            typed = humanize(text, P["typing"], rng)
            entry["typed"] = typed
            chat.say(typed); screen.user(typed)
        elif act == "key" and text:
            # a person taps what is on the screen; a label not there is typed
            chat.say(text); screen.user(f"(нажал кнопку) {text}")
        elif act == "inline":
            vis = screen.inline[-8:]
            try:
                label, data, mid = vis[int(a.get("index", -1))]
                entry["text"], entry["data"] = label, data
                chat.press(data, mid); screen.user(f"(нажал кнопку под сообщением) {label}")
            except (ValueError, IndexError, TypeError):
                entry["bad_action"] = True
        elif act == "photo":
            path = PHOTOS.get(a.get("photo") or "", PHOTOS["desk"])
            chat.photo(path, caption=text); screen.user(f"[отправил фото] {text}")
        # «wait» and an impatient person both just look again after a while
        t0 = time.time()
        screen.feed(chat.wait(timeout=240 if act != "wait" else 120, quiet=8))
        entry["waited_s"] = round(time.time() - t0)
        log.append(entry)
    screen.feed(chat.wait(timeout=300, quiet=10))
    review = _ask(ctx, _REVIEW.format(who=P["who"], goal=P["goal"],
                                      chat="\n".join(screen.lines)[-6000:]),
                  max_tokens=700, temperature=0.4)
    return {"persona": key, "chat_id": P["chat"], "steps": log, "screen": screen.lines,
            "review": review}


def audit(results: list[dict]) -> list:
    """Run the transcript auditor over what the fake wire recorded."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import ux_audit
    labels = ux_audit._labels()
    found = []
    for r in results:
        rows = []
        for e in D.WIRE.sent:
            if e["chat_id"] == r["chat_id"]:
                rows.append({"ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e["t"])),
                             "kind": "out", "data": dict(e["payload"], method=e["method"])})
        for s in r["steps"]:
            ts = time.strftime("%Y-%m-%d ") + s["t"]
            if s["action"] in ("type", "key") and s.get("text"):
                rows.append({"ts": ts, "kind": "in", "data": {"text": s.get("typed") or s["text"]}})
            elif s["action"] == "inline" and s.get("data"):
                rows.append({"ts": ts, "kind": "in", "data": {"button": s["data"]}})
        rows.sort(key=lambda x: x["ts"])
        found += [(r["persona"],) + f for f in ux_audit.audit_chat(rows, labels)]
    return found


def report(results: list[dict], found: list) -> str:
    md = ["# Persona run " + D.STAMP, ""]
    for r in results:
        bad = [s for s in r["steps"] if s["feeling"] in ("confused", "annoyed")]
        md += [f"## {r['persona']} — {len(r['steps'])} steps, {len(bad)} confused/annoyed", "",
               r["review"].strip(), "", "### Where it hurt"]
        md += [f"- step {s['step']} {s['action']} «{s.get('typed') or s['text']}» — "
               f"{s['feeling']}: {s['why']}" for s in bad] or ["- (nowhere)"]
        md += ["", "<details><summary>screen</summary>", "", "```",
               *r["screen"], "```", "</details>", ""]
    md += ["## ux_audit", ""] + [f"- {p} {lvl} {ts} {rule}: {det}" for p, lvl, ts, rule, det in found]
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--steps", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    keys = [k for k in (a.only.split(",") if a.only else PERSONAS) if k in PERSONAS]
    rng = random.Random(a.seed or time.time())
    bot, ctx = D.build()
    results = []
    for k in keys:
        try:
            results.append(run_persona(bot, ctx, k, a.steps, rng))
        except Exception as exc:
            import traceback; traceback.print_exc()
            results.append({"persona": k, "chat_id": PERSONAS[k]["chat"], "steps": [],
                            "screen": [], "review": f"CRASHED: {exc!r}"})
        (D.OUT / "persona_results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    found = audit(results)
    md = report(results, found)
    (D.OUT / "persona_report.md").write_text(md, encoding="utf-8")
    print("\nREPORT", D.OUT / "persona_report.md")
    os._exit(0)


if __name__ == "__main__":
    main()
