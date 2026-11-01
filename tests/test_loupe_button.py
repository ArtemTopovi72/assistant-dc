"""🔍 The loupe: find something on a picture and show it highlighted.

Two halves:
  the pipeline (imaging/image_highlight) — mask->boxes, OCR word matching,
  VLM box parsing, the draw, and highlight()'s fallback ladder, all offline;
  the surfaces — the agent tool (highlight_region), the fastpath prefix
  route, the Telegram 🔍 button (asks WHAT to find, same free-text capture as
  🧽/👗, arms "find in the image: ") and the GUI mirrors (dashboard button,
  viewer 🔍, RU labels).

Run: venv/Scripts/python.exe tests/test_loupe_button.py
"""
RUN_AS_SCRIPT = True
import os, sys, tempfile, time, types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
os.environ.setdefault("F5_TEST_RUN", "1")

from PIL import Image

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {str(extra)[:200]}")

TMP = tempfile.mkdtemp(prefix="loupe_test_")
def _png(name, size=(640, 480), color=(40, 90, 40)):
    p = os.path.join(TMP, name)
    Image.new("RGB", size, color).save(p)
    return p


# ── the pipeline ─────────────────────────────────────────────────────────────
import image_highlight as LH

print("=" * 70)
print("draw(): the highlight lands on a COPY, source untouched")
print("=" * 70)
src = _png("src.png")
source_bytes = open(src, "rb").read()
out = LH.draw(src, [(100, 80, 300, 260), (380, 200, 560, 400)],
              os.path.join(TMP, "marked.jpg"))
check("draw returns the new file", out and os.path.exists(out), out)
with Image.open(src) as a, Image.open(out) as b:
    check("source bytes unchanged", open(src, "rb").read() == source_bytes)
    check("output same dimensions as source", a.size == b.size, (a.size, b.size))

print("=" * 70)
print("mask -> boxes: one per blob, specks dropped")
print("=" * 70)
mask = os.path.join(TMP, "mask.png")
im = Image.new("L", (1000, 800), 0)
from PIL import ImageDraw as _ID
d = _ID.Draw(im)
d.rectangle((100, 100, 300, 260), fill=255)     # blob 1
d.rectangle((600, 400, 900, 700), fill=255)     # blob 2
d.rectangle((500, 10, 505, 14), fill=255)       # speck
im.save(mask)
boxes = LH._mask_boxes(mask)
check("two blobs found (speck dropped)", len(boxes) == 2, boxes)
if len(boxes) == 2:
    check("blob boxes are sorted largest-first",
          (boxes[0][2]-boxes[0][0])*(boxes[0][3]-boxes[0][1]) >=
          (boxes[1][2]-boxes[1][0])*(boxes[1][3]-boxes[1][1]), boxes)
    check("largest box hugs blob 2", abs(boxes[0][0]-600) < 3 and abs(boxes[0][3]-700) < 3, boxes[0])
    check("smaller box hugs blob 1", abs(boxes[1][0]-100) < 3 and abs(boxes[1][3]-260) < 3, boxes[1])
flat = _png("flat.png", size=(600, 240), color=(255, 255, 255))
with Image.open(flat) as _flat:
    _flat_draw = _ID.Draw(_flat)
    for x in (50, 250, 450):
        _flat_draw.ellipse((x, 70, x + 80, 150), fill=(230, 20, 20))
    _flat.save(flat)
flat_mask = os.path.join(TMP, "flat-mask.png")
_fm = Image.new("L", (600, 240), 0)
_ID.Draw(_fm).ellipse((250, 70, 330, 150), fill=255)
_fm.save(flat_mask)
check("flat-colour repeats are found at all three actual positions",
      len(LH._same_colour_instances(flat, flat_mask)) == 3,
      LH._same_colour_instances(flat, flat_mask))

print("=" * 70)
print("OCR word matching + the lettering/thing read stays off regexes")
print("=" * 70)
check("query terms drop locating verbs",
      LH._query_terms("где написано Привет мир") == ["привет", "мир"],
      LH._query_terms("где написано Привет мир"))
check("english query terms too",
      LH._query_terms("where it says SALE now") == ["sale", "now"],
      LH._query_terms("where it says SALE now"))
check("piece matches its term", LH._match_piece("Привет мир", ["привет"], False))
check("fuzzy piece still matches", LH._match_piece("Прпвет", ["привет"], False))
check("unrelated piece does not", not LH._match_piece("Скидка 50%", ["привет"], False))
check("all_ok highlights every piece", LH._match_piece("что угодно", [], True))
import ocr_reader as _OCR
_orig_boxes = _OCR.read_boxes
_OCR.read_boxes = lambda path: [([(10, 20), (90, 20), (90, 40), (10, 40)], "SALE", 0.95)]
_test_flag = os.environ.pop("F5_TEST_RUN", None)
try:
    check("OCR coordinates highlight the matching word",
          LH._ocr_boxes(None, src, ["sale"], False) == [(10, 20, 90, 40)])
finally:
    if _test_flag is not None:
        os.environ["F5_TEST_RUN"] = _test_flag
    _OCR.read_boxes = _orig_boxes

print("=" * 70)
print("VLM boxes: 0-1000 grid JSON -> pixel boxes, junk dropped")
print("=" * 70)
_called = {}
def _fake_vision(ctx):
    def call(**kw):
        _called["prompt"] = kw.get("user_text")
        return ('{"found": true, "boxes": [[100, 100, 400, 300], '
                '[500, 200, 900, 600], [10, 10, 11, 11], "junk"]}')
    return call
_orig_vision = LH.vision_model
LH.vision_model = lambda ctx: _fake_vision(ctx)
img640 = _png("vlm.png")
vb = LH._vlm_boxes(None, img640, "squirrel")
check("two sane boxes parsed, junk/speck dropped", len(vb) == 2, vb)
if len(vb) == 2:
    # 640x480 image: 100/1000*640=64, 300/1000*480=144
    check("grid scaled to pixels", abs(vb[0][0] - 64) < 1 and abs(vb[0][3] - 144) < 1, vb[0])
LH.vision_model = _orig_vision

print("=" * 70)
print("highlight(): the ladder falls through and never touches the source")
print("=" * 70)
_src_real = _png("real.png")
_orig_wants = LH._wants_lettering
_orig_ocr, LH._ocr_boxes = LH._ocr_boxes, lambda *a, **k: []
LH._wants_lettering = lambda ctx, q: False
import image as _IMG
_orig_up = _IMG._upload_image_to_comfy
_IMG._upload_image_to_comfy = lambda *a, **k: None          # SAM3 unreachable
LH._vlm_boxes = lambda ctx, p, ph: []                        # nothing anywhere
res = LH.highlight(None, _src_real, "белка")
check("total failure -> honest empty result, no exception",
      res["path"] == "" and res["count"] == 0, res)
LH._vlm_boxes = lambda ctx, p, ph: [(120.0, 90.0, 420.0, 330.0)]
res = LH.highlight(None, _src_real, "белка")
check("VLM fallback produces the highlighted copy",
      res["path"] and os.path.exists(res["path"]) and res["via"] == "vlm", res)
check("count and boxes carried", res["count"] == 1 and len(res["boxes"]) == 1, res)
_IMG._upload_image_to_comfy = lambda *a, **k: "uploaded.jpg"
_old_region_mask = _IMG._region_mask_file
_IMG._region_mask_file = lambda *a, **k: mask
_old_mask_boxes = LH._mask_boxes
LH._mask_boxes = lambda p: [(250.0, 120.0, 380.0, 260.0)]
_old_colour = LH._same_colour_instances
LH._same_colour_instances = lambda p, m: [(50.0, 120.0, 180.0, 260.0),
                                          (250.0, 120.0, 380.0, 260.0),
                                          (450.0, 120.0, 580.0, 260.0)]
res = LH.highlight(None, _src_real, "all three circles")
check("flat-colour instances recovered after one SAM3 hit",
      res["count"] == 3 and res["via"] == "colour", res)
LH._same_colour_instances = lambda p, m: []
_old_yes_no = LH.yes_no
LH.yes_no = lambda ctx: lambda *a, **k: True
res = LH.highlight(None, _src_real, "all three squirrels")
check("one mask cannot claim all three objects", not res["path"], res)
LH.yes_no = _old_yes_no
LH._same_colour_instances = _old_colour
LH._mask_boxes = _old_mask_boxes
_IMG._region_mask_file = _old_region_mask
LH._wants_lettering = _orig_wants
LH._ocr_boxes = _orig_ocr
LH._vlm_boxes = lambda ctx, p, ph: []
_IMG._upload_image_to_comfy = _orig_up

# ── the agent tool ───────────────────────────────────────────────────────────
import tools as TOOLS
import tool_image_handlers as TIH
import graph_fastpath as GF
import graph_personality as GP

print("=" * 70)
print("highlight_region: registration + the narrowed route")
print("=" * 70)
spec = next((t for t in TOOLS.TOOLS if t.name == "highlight_region"), None)
check("tool registered", spec is not None)
check("args class wired", spec and spec.args_model.__name__ == "HighlightRegionArgs")
narrowed = GF._intent_tools("find in the image: белка")
check("the prefix narrows to the loupe",
      narrowed == frozenset({"highlight_region"}), narrowed)
check("the loupe button forces its tool on the first round",
      GP._forced_button("find in the image: где написано SALE",
                        [{"function": {"name": "highlight_region"}}], set()) == "highlight_region")
check("no prefix -> full set (None)", GF._intent_tools("нарисуй кота") is None)
check("loupe counted as an image tool (omitted with no image context)",
      "highlight_region" in GF._IMAGE_TOOL_NAMES)

print("=" * 70)
print("highlight_region handler: found / not found / no image")
print("=" * 70)
class _Ctx:
    def __init__(self):
        self.last_image_path = _src_real
        self.image_undo = []
        self.stage = ""
        self.mem = []
    def set_stage(self, s): self.stage = s
    def remember(self, kind, text, meta): self.mem.append((kind, text))
    def memory_text(self): return " ".join(t for _, t in self.mem)

src_png = _png("handle.png")
HL_OUT = LH.draw(src_png, [(50, 50, 200, 200)], os.path.join(TMP, "hl1.jpg"))
_orig_hl = TIH.__dict__.get("highlight")  # handler imports the module, not the name
import image_highlight as _LH2
_LH2.highlight = lambda ctx, p, q: {"path": HL_OUT, "boxes": [(50, 50, 200, 200)],
                                    "via": "sam3", "count": 1}
ctx = _Ctx(); ctx.last_image_path = src_png
state = {"image_path": src_png}
r = TIH._handle_highlight_region(ctx, state, {"query": "squirrel"})
check("found -> [done] summary", r.startswith("[done] Found it"), r[:80])
check("the highlighted copy becomes the current image",
      state.get("image_path") == HL_OUT and ctx.last_image_path == HL_OUT)
check("registered as a version of the original",
      state.get("image_derived_from") == src_png, state.get("image_derived_from"))
check("memory entry logged", any(k == "loupe" for k, _ in ctx.mem))
check("the agent is told to stop after delivering",
      "no more tool calls" in r.lower() or "make NO more tool calls" in r, r[-120:])
check("the answer says WHERE", "top" in r or "bottom" in r or "middle" in r or "centre" in r, r)

_LH2.highlight = lambda ctx, p, q: {"path": "", "boxes": [], "via": "", "count": 0}
ctx2 = _Ctx(); ctx2.last_image_path = src_png
state2 = {"image_path": src_png}
r2 = TIH._handle_highlight_region(ctx2, state2, {"query": "unicorn"})
check("not found -> honest tool error", r2.startswith("[TOOL ERROR]") and "NOT found" in r2, r2[:100])
check("not found -> agent must not invent or edit",
      "do NOT invent" in r2 and "do NOT edit" in r2, r2[:160])
check("not found -> current image unchanged",
      state2.get("image_path") == src_png and ctx2.last_image_path == src_png)

_LH2.highlight = lambda ctx, p, q: {"path": "", "boxes": [], "via": "", "count": 0}
ctx3, state3 = _Ctx(), {}
ctx3.last_image_path = ""
r3 = TIH._handle_highlight_region(ctx3, state3, {"query": "squirrel"})
check("no image -> clear tool error", r3.startswith("[TOOL ERROR]") and "No image" in r3, r3[:100])

# ── Telegram: the 🔍 button and its ask-flow ────────────────────────────────
_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_loupe_")
import tg_bot as T
T.redirect_data_dir(_DATA_DIR)

print("=" * 70)
print("TG keyboard: the button in both languages, verb auto-accepted")
print("=" * 70)
kb_ru = T._image_kb("ru")["inline_keyboard"]
kb_en = T._image_kb("en")["inline_keyboard"]
flat_ru = [b["text"] for row in kb_ru for b in row]
flat_en = [b["text"] for row in kb_en for b in row]
check("ru keyboard carries 🔍 Лупа", "🔍 Лупа" in flat_ru, flat_ru)
check("en keyboard carries 🔍 Find", "🔍 Find" in flat_en, flat_en)
kbi = T._image_kb("ru", "ab12cd")["inline_keyboard"]
find_btn = next(b for row in kbi for b in row if b["callback_data"].startswith("find:"))
check("button carries the picture id", find_btn["callback_data"] == "find:ab12cd", find_btn)
check("verb auto-accepted by the id validator", "find" in T._IMAGE_VERBS)
check("the armed prefix keeps the Draw submenu",
      T.TelegramBot._PREFIX_MENU.get("find in the image: ") == "draw")
check("old buttons all still present (nothing dropped)",
      all(x in flat_ru for x in ("❓ Спросить", "👗 Одежда", "🧽 Убрать", "🔤 Без надписей",
                                 "🔄 Ещё раз", "🎨 Правка", "🎭 Стиль", "🎬 Анимация", "🔁 Замена")),
      flat_ru)
check("10 buttons -> rows of 3 plus one (no broken grid)",
      len(flat_ru) == 10 and max(len(r) for r in kb_ru) == 3, [len(r) for r in kb_ru])

CID = 999905
IMG_DIR = tempfile.mkdtemp(prefix="tgtest_loupe_imgs_")
def make_png(name):
    p = os.path.join(IMG_DIR, name)
    with open(p, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + os.urandom(64))
    return p

def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(),
                        lambda: {}, silent_mode=True)
    bot.sent = []; bot.pushed = []
    def _send(cid, text, **kw):
        bot.sent.append((text, kw.get("keyboard"))); return 1
    bot._send_text = _send
    bot._send_get_id = lambda cid, text, **kw: (_send(cid, text, **kw), 1)[1]
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._backend = types.SimpleNamespace(
        push=lambda t: bot.pushed.append(t), depth=lambda: 0, name=lambda: "stub",
        chat_depth=lambda cid: 0, drop_chat=lambda cid: 0, close=lambda: None,
        tasks_ahead=lambda tid: [])
    bot._activity.log = lambda *a, **k: None
    u = T._User(chat_id=CID, name="Tester", status="approved")
    bot._user_store.put(u)
    return bot

def press(bot, data, msg_id=900):
    bot._dispatch({"callback_query": {"id": "1", "data": data, "from": {"id": CID},
                                      "message": {"chat": {"id": CID, "type": "private"},
                                                  "message_id": msg_id}}})
    deadline = time.monotonic() + 5.0
    base = len(bot.sent)
    while time.monotonic() < deadline:
        if len(bot.sent) != base:
            time.sleep(0.05); return
        time.sleep(0.03)

def send_text(bot, text, msg_id=902):
    bot._dispatch({"message": {"chat": {"id": CID, "type": "private"},
                               "message_id": msg_id, "date": int(time.time()),
                               "text": text}})
    deadline = time.monotonic() + 5.0
    base = len(bot.pushed)
    while time.monotonic() < deadline:
        if len(bot.pushed) != base:
            time.sleep(0.1); return
        time.sleep(0.03)

print("=" * 70)
print("TG flow: press 🔍 -> asks what -> free text becomes the prefixed task")
print("=" * 70)
bot = make_bot()
s = bot._get_session(CID); s.clear_context(); s.lang = "ru"
a = make_png("target.png")
id_a = T._log_image(s, a, msg_id=101, label="target", src="bot")
bot._store.put(s)

press(bot, f"find:{id_a}")
sess = bot._get_session(CID)
check("pending_prefix armed to the loupe prefix",
      sess.pending_prefix == "find in the image: ", sess.pending_prefix)
check("target_image resolved to the pressed picture",
      sess.target_image == id_a, sess.target_image)
check("asked what to find, nothing enqueued", bool(bot.sent) and not bot.pushed,
      (bot.sent, bot.pushed))
check("the ask is the RU describe_find string",
      bot.sent and "подсветить" in bot.sent[-1][0] and "белка" in bot.sent[-1][0],
      bot.sent[-1][0] if bot.sent else None)

send_text(bot, "белка")
check("exactly one task pushed", len(bot.pushed) == 1, bot.pushed)
if bot.pushed:
    check("task text = prefix + the user's own words",
          bot.pushed[0].user_text == "find in the image: белка", bot.pushed[0].user_text)
sess2 = bot._get_session(CID)
check("prefix consumed after the free text", sess2.pending_prefix == "", sess2.pending_prefix)

# EN locale
bot2 = make_bot()
s2 = bot2._get_session(CID); s2.clear_context(); s2.lang = "en"
b = make_png("en.png")
id_b = T._log_image(s2, b, msg_id=111, label="en", src="bot")
bot2._store.put(s2)
press(bot2, f"find:{id_b}")
check("EN ask is the describe_find string",
      bot2.sent and "highlight" in bot2.sent[-1][0], bot2.sent[-1][0] if bot2.sent else None)

# ── GUI mirrors ──────────────────────────────────────────────────────────────
print("=" * 70)
print("GUI: dashboard button, mixin route, viewer 🔍, RU labels, worker")
print("=" * 70)
import gui_image_fix as FIX
check("mixin carries the same prefix as the bot",
      FIX.ImageFixMixin.FIND_PREFIX == "find in the image: ")
check("mixin method present", callable(getattr(FIX.ImageFixMixin, "_find_on_last_image", None)))

import gui_dashboard as DASH
import inspect as _insp
_src_dash = _insp.getsource(DASH)
check("dashboard Images group includes the loupe", "self.find_btn" in _src_dash)
check("dashboard button wired to the mixin handler",
      "_find_on_last_image" in _src_dash and "Find on image" in _src_dash)

import gui_ru as RU
check("RU label for the dashboard button", RU.RU.get("🔍  Find on image") == "🔍  Найти на фото")
check("RU prompt for the dialog", "подсветить" in RU.RU.get(
    "What to find and highlight? (e.g. «the squirrel», «the man on the left», «where it says SALE»)", ""))

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
from PyQt5.QtWidgets import QApplication
_app = QApplication.instance() or QApplication([])
from gui_dialogs import ImageViewerDialog
p1 = _png("view1.png")
dlg = ImageViewerDialog([p1], 0, None)
check("viewer toolbar has the 🔍 button", dlg._find_btn is not None)
dlg._on_find_done(p1, 2)          # simulate a worker result
check("highlighted copy appended right after the original",
      len(dlg._paths) == 2 and dlg._idx == 1 and dlg._paths[0] == p1, dlg._paths)
dlg._on_find_done("", 0)
check("nothing-found keeps the gallery and says so",
      "Nothing" in dlg._info.text(), dlg._info.text())
dlg.deleteLater()

import gui_workers as W
check("HighlightWorker exists with done(int)/failed signals",
      hasattr(W, "HighlightWorker"))
wk = W.HighlightWorker(None, p1, "test")     # constructed only, never started
check("worker takes ctx/path/query", wk.source_path == p1 and wk.query == "test")
wk.deleteLater()

print(f"\n{OK}/{OK+BAD} checks passed")
if __name__ == "__main__":
    sys.exit(0 if BAD == 0 else 1)
