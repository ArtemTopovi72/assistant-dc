"""Every reply-keyboard button and every inline button one level under it, pressed through
a real TelegramBot with the network stubbed; what the bot sends is drawn as a Telegram-like
chat page for a visual review of how the bot LOOKS (texts, markup, keyboards).

Run: venv/Scripts/python tests/shot_tg_screens.py [ru|en]  ->  runtime/tg_screens/<lang>.html
"""
import html, os, re, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
os.environ.setdefault("F5_TEST_RUN", "1")
import logging; logging.basicConfig(level=logging.CRITICAL)
import tg_bot as T
import tg_strings

T.redirect_data_dir(tempfile.mkdtemp(prefix="tg_screens_"))
LANG = sys.argv[1] if len(sys.argv) > 1 else "ru"
CID = 999872
OUT = Path("runtime/tg_screens"); OUT.mkdir(parents=True, exist_ok=True)

events = []          # (kind, text, parse_mode, keyboard)


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)

    def send(cid, text, parse_mode=None, keyboard=None, **k):
        events.append(("msg", text, parse_mode, keyboard)); return 1
    bot._send_text = send
    bot._send_get_id = send
    bot._edit_text = lambda cid, mid, text, parse_mode=None, keyboard=None, **k: events.append(
        ("edit", text, parse_mode, keyboard))
    bot._send_photo = lambda cid, *a, **k: events.append(("photo", "[photo]", None, k.get("keyboard")))
    bot._activity.log = lambda *a, **k: None
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._backend.push = lambda task: events.append(("queued", task.user_text, None, None))
    # no debounce thread: a label is resolved right away, so its reply lands in `events`
    bot._enqueue_item = lambda cid, item: bot._resolve_and_push(cid, [item])
    bot._user_store.put(T._User(chat_id=CID, name="Артём", status="approved"))
    sess = bot._get_session(CID)
    sess.lang = LANG
    bot._store.put(sess)
    return bot


def upd_text(text):
    return {"update_id": 1, "message": {"message_id": 5, "chat": {"id": CID}, "from": {"id": CID}, "text": text}}


def upd_cb(data):
    return {"update_id": 1, "callback_query": {"id": "q", "data": data, "from": {"id": CID},
                                               "message": {"message_id": 5, "chat": {"id": CID}}}}


SAFE_CB = re.compile(r"^(music|video|menu|nav|size|ms|mg|cvv|song_set|vs|set|lang|help|wtw_menu|creat|kb|cat)", re.I)
SKIP_CB = re.compile(r"(go|reset|delete|del|ban|approve|reject|kill|cancel|stop|skip|logout|clear)", re.I)

sections = []
labels = sorted({forms.get(LANG) for forms in tg_strings._BTN.values() if forms.get(LANG)})
for label in ["/start", "/help"] + labels:
    bot = make_bot()
    events.clear()
    try:
        bot._dispatch_logged(upd_text(label))
    except Exception as exc:
        events.append(("error", repr(exc)[:300], None, None))
    shots = [("▶ " + label, list(events))]
    inline = []
    for _k, _t, _p, kb in list(events):
        for row in (kb or {}).get("inline_keyboard", []) if isinstance(kb, dict) else []:
            for b in row:
                d = b.get("callback_data") or ""
                if d and SAFE_CB.match(d) and not SKIP_CB.search(d) and d not in inline:
                    inline.append(d)
    for d in inline[:12]:
        events.clear()
        try:
            bot._dispatch_logged(upd_cb(d))
        except Exception as exc:
            events.append(("error", repr(exc)[:300], None, None))
        if events:
            shots.append(("  ↳ " + d, list(events)))
    sections.append((label, shots))


def render_text(text, mode):
    if mode == "HTML":
        allowed = re.sub(r"<(?!/?(b|i|u|s|code|pre|a|blockquote|tg-spoiler)\b)", "&lt;", text or "")
        return allowed.replace("\n", "<br>")
    return html.escape(text or "").replace("\n", "<br>")


def render_kb(kb):
    if not isinstance(kb, dict):
        return ""
    out = ""
    if kb.get("inline_keyboard"):
        out += '<div class="ikb">' + "".join(
            '<div class="row">' + "".join(f'<span class="ib" title="{html.escape(b.get("callback_data") or b.get("url") or "")}">'
                                         f'{html.escape(b.get("text", ""))}</span>' for b in r) + "</div>"
            for r in kb["inline_keyboard"]) + "</div>"
    if kb.get("keyboard"):
        out += '<div class="rkb">' + "".join(
            '<div class="row">' + "".join(f'<span class="rb">{html.escape(b if isinstance(b, str) else b.get("text", ""))}</span>'
                                         for b in r) + "</div>" for r in kb["keyboard"]) + "</div>"
    return out


parts = []
for label, shots in sections:
    parts.append(f'<section><h2>{html.escape(label)}</h2>')
    for title, evs in shots:
        parts.append(f'<div class="me">{html.escape(title)}</div>')
        for kind, text, mode, kb in evs:
            cls = {"edit": "bot edit", "error": "bot err", "queued": "bot q"}.get(kind, "bot")
            tag = {"edit": "✎ edited", "queued": "→ to the model", "error": "ERROR"}.get(kind, "")
            parts.append(f'<div class="{cls}">' + (f'<small>{tag}</small><br>' if tag else "")
                         + render_text(text, mode) + render_kb(kb) + "</div>")
    parts.append("</section>")

page = f"""<!doctype html><html lang="{LANG}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>TG screens {LANG}</title>
<style>
:root {{ --bg:#0e1621; --bubble:#182533; --me:#2b5278; --text:#f5f5f5; --muted:#8a9aa9; --btn:#2b394a; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.4 -apple-system,"Segoe UI",Roboto,sans-serif; }}
main {{ max-width:460px; margin:0 auto; padding:16px; }}
section {{ border-bottom:1px solid #22303f; padding-bottom:16px; margin-bottom:16px; }}
h2 {{ font-size:13px; color:var(--muted); margin:8px 0; }}
.me {{ background:var(--me); margin:8px 0 8px auto; width:fit-content; max-width:85%; padding:6px 10px; border-radius:12px 12px 4px 12px; }}
.bot {{ background:var(--bubble); margin:6px 0; max-width:92%; padding:8px 10px; border-radius:12px 12px 12px 4px; overflow-wrap:anywhere; }}
.bot.err {{ background:#5a1e1e; }} .bot.q {{ background:#3d3517; }}
.bot small {{ color:var(--muted); }}
code, pre {{ font-family:Consolas,monospace; background:#0b1118; padding:0 3px; border-radius:3px; }}
pre {{ display:block; padding:6px; white-space:pre-wrap; }}
.ikb, .rkb {{ margin-top:6px; }}
.row {{ display:flex; gap:4px; margin-top:4px; }}
.ib, .rb {{ flex:1; text-align:center; background:var(--btn); border-radius:6px; padding:6px 4px; font-size:13px;
  overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:0; }}
.rkb {{ border-top:1px dashed #33475c; padding-top:4px; }} .rb {{ background:#1f2c3a; }}
</style></head><body><main>{''.join(parts)}</main></body></html>"""
(OUT / f"{LANG}.html").write_text(page, encoding="utf-8")
print("sections:", len(sections), "->", OUT / f"{LANG}.html")
