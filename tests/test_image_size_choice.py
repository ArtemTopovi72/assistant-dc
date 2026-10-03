# -*- coding: utf-8 -*-
"""User-selectable output size: resolver, pipeline plumbing, Telegram picker.

Covers the three things that have to hold for "let me pick the resolution and
aspect ratio" to actually reach ComfyUI:

  1. config.resolve_image_size turns (aspect, quality) into legal pixels;
  2. image.py honours a session's choice instead of the module default, and
     normalize_resolution no longer crushes every size to 960x544;
  3. tg_bot renders the picker, records the tap, and swaps the choice onto ctx
     for the duration of one chat's turn.
"""
import os
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sweep_stub  # noqa: F401  the model's narrow reads, stubbed

import config as C          # noqa: E402
import image as I           # noqa: E402

PASS = FAIL = 0
FAILURES = []


def check(cond, label):
    """Count in script mode, RAISE under pytest.

    This file is both: `python tests/test_image_size_choice.py` reports a
    tally, and pytest collects the test_ functions directly. Counting alone
    made the pytest half decorative -- every function returned normally no
    matter what check() saw, so the suite reported green while a hardcoded
    960x544 sat in the size path. Verified by mutation: with the count-only
    version, replacing session_image_size with a constant still passed.
    """
    global PASS, FAIL
    if cond:
        PASS += 1
        return
    FAIL += 1
    FAILURES.append(label)
    print("  FAIL: " + label)
    if os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(label)


# ── 1. the resolver ───────────────────────────────────────────────────────────
def test_resolver():
    print("[resolver]")
    for aspect, (rw, rh) in C.IMAGE_ASPECTS.items():
        for quality, mp in C.IMAGE_QUALITIES.items():
            w, h = C.resolve_image_size(aspect, quality)
            tag = f"{aspect}/{quality}"
            check(w % 16 == 0 and h % 16 == 0, f"{tag}: not a multiple of 16 ({w}x{h})")
            check(max(w, h) <= C.IMAGE_MAX_SIDE, f"{tag}: exceeds max side ({w}x{h})")
            check(min(w, h) >= C.IMAGE_MIN_SIDE, f"{tag}: below min side ({w}x{h})")
            # Orientation must follow the ratio, not the other way round.
            if rw > rh:
                check(w > h, f"{tag}: landscape ratio produced {w}x{h}")
            elif rh > rw:
                check(h > w, f"{tag}: portrait ratio produced {w}x{h}")
            else:
                check(w == h, f"{tag}: square ratio produced {w}x{h}")
            # Within the snap/clamp tolerance the ratio survives.
            want, got = rw / rh, w / h
            check(abs(want - got) / want < 0.05, f"{tag}: aspect drifted {got:.3f} vs {want:.3f}")

    # A tier is meaningfully bigger than the one below it, or the picker is a lie.
    order = ("draft", "standard", "high", "ultra")
    for lo, hi in zip(order, order[1:]):
        a = C.resolve_image_size("16:9", lo)
        b = C.resolve_image_size("16:9", hi)
        check(b[0] * b[1] > a[0] * a[1] * 1.15,
              f"{hi} is not meaningfully larger than {lo}")

    # The old hard-coded pair is still reachable, as the draft tier.
    check(C.resolve_image_size("16:9", "draft") == (960, 544),
          "draft 16:9 should reproduce the historic 960x544")

    # The default tier is now DELIBERATELY the cheapest one, and users opt up.
    # One 24 GB card is shared by the LLM, Whisper, F5-TTS and ComfyUI, so
    # rendering every casual picture at 1.6 MP spends the shared queue on people
    # who never asked for it.
    #
    # This used to assert `DEFAULT_WIDTH * DEFAULT_HEIGHT > 1_200_000`, pinning
    # the fix for "why is everything blurry". That bug was NOT the resolution —
    # it was that 960x544 was HARD-CODED in the resolver and in the prompt, so a
    # user who picked "ultra" still got 960x544. A low default that the picker
    # can actually move is a different thing, so the checks below pin the part
    # that mattered: the number comes from the tier table, and choosing a bigger
    # tier really does produce a bigger picture.
    check(C.DEFAULT_IMAGE_QUALITY in C.IMAGE_QUALITIES,
          f"the default tier {C.DEFAULT_IMAGE_QUALITY!r} is not one of the tiers")
    check((C.DEFAULT_WIDTH, C.DEFAULT_HEIGHT)
          == C.resolve_image_size(C.DEFAULT_IMAGE_ASPECT, C.DEFAULT_IMAGE_QUALITY),
          f"the default pair {C.DEFAULT_WIDTH}x{C.DEFAULT_HEIGHT} is not what the "
          f"resolver returns for the default tier — it is hard-coded somewhere")
    # The escape hatch has to be real: asking for more must give more.
    _d = C.resolve_image_size("16:9", C.DEFAULT_IMAGE_QUALITY)
    _u = C.resolve_image_size("16:9", "ultra")
    check(_u[0] * _u[1] > _d[0] * _d[1] * 2,
          f"opting up is not worth it: default {_d} vs ultra {_u}")
    # And the default must be the CHEAPEST tier, not merely a valid one.
    check(C.IMAGE_QUALITIES[C.DEFAULT_IMAGE_QUALITY] == min(C.IMAGE_QUALITIES.values()),
          f"default tier {C.DEFAULT_IMAGE_QUALITY!r} is not the minimum")

    # The default must be the DEFAULT_IMAGE_ASPECT/QUALITY pair, stated
    # independently. Comparing against DEFAULT_WIDTH alone is a tautology: that
    # constant is itself produced by this function, so a broken fallback moves
    # both sides of the assertion together.
    check(C.resolve_image_size(None, None)
          == C.resolve_image_size(C.DEFAULT_IMAGE_ASPECT, C.DEFAULT_IMAGE_QUALITY),
          "the no-argument default is not the configured aspect/quality pair")
    drw, drh = C.IMAGE_ASPECTS[C.DEFAULT_IMAGE_ASPECT]
    dw, dh = C.resolve_image_size(None, None)
    check(abs((dw / dh) - (drw / drh)) / (drw / drh) < 0.05,
          f"the default size {dw}x{dh} is not {C.DEFAULT_IMAGE_ASPECT}")

    # The max-side rail has to hold for a tier bigger than any we ship today,
    # or the clamp is only "tested" by never being reached.
    saved = dict(C.IMAGE_QUALITIES)
    try:
        C.IMAGE_QUALITIES["_huge"] = 60.0
        for aspect in C.IMAGE_ASPECTS:
            w, h = C.resolve_image_size(aspect, "_huge")
            check(max(w, h) <= C.IMAGE_MAX_SIDE,
                  f"{aspect}: an oversized tier escaped the max-side clamp ({w}x{h})")
            rw, rh = C.IMAGE_ASPECTS[aspect]
            check(abs((w / h) - (rw / rh)) / (rw / rh) < 0.05,
                  f"{aspect}: clamping an oversized tier broke the ratio ({w}x{h})")
    finally:
        C.IMAGE_QUALITIES.clear()
        C.IMAGE_QUALITIES.update(saved)

    # Garbage in -> defaults out, never an exception.
    for bad in (None, "", "nope", "16/9", 5, object()):
        try:
            w, h = C.resolve_image_size(bad, bad)
            check((w, h) == (C.DEFAULT_WIDTH, C.DEFAULT_HEIGHT),
                  f"bad input {bad!r} should fall back to the default, got {w}x{h}")
        except Exception as exc:                                   # noqa: BLE001
            check(False, f"bad input {bad!r} raised {exc!r}")


# ── 2. image.py ───────────────────────────────────────────────────────────────
class _Ctx:
    def __init__(self, aspect="", quality=""):
        self.image_aspect = aspect
        self.image_quality = quality


def test_normalize():
    print("[normalize_resolution]")
    # The regression this whole feature hangs on: a big size must SURVIVE.
    w, h = I.normalize_resolution(1920, 1080)
    check(w * h > 1_500_000, f"1920x1080 was crushed to {w}x{h}")
    check(abs((w / h) - (16 / 9)) < 0.05, f"1920x1080 lost its aspect: {w}x{h}")
    check((w, h) != (960, 544), "1920x1080 still collapses to the old constant")

    w, h = I.normalize_resolution(1080, 1920)
    check(h > w, "portrait input must stay portrait")
    check((w, h) != (544, 960), "portrait still collapses to the old constant")

    # Rails.
    w, h = I.normalize_resolution(99999, 99999)
    check(max(w, h) <= C.IMAGE_MAX_SIDE, f"no max-side clamp: {w}x{h}")
    w, h = I.normalize_resolution(32, 32)
    check(min(w, h) >= C.IMAGE_MIN_SIDE, f"no min-side floor: {w}x{h}")
    # An extreme-but-legal ratio: flooring the short side by itself (which the
    # snap already does) would leave the long side alone and destroy the ratio.
    w, h = I.normalize_resolution(200, 1200)
    check(min(w, h) >= C.IMAGE_MIN_SIDE, f"short side not floored: {w}x{h}")
    check(abs((h / w) - 6.0) / 6.0 < 0.05,
          f"flooring the short side broke the ratio: {w}x{h}")

    w, h = I.normalize_resolution(4000, 2000)
    check(abs((w / h) - 2.0) < 0.05, f"clamping broke the ratio: {w}x{h}")

    # Junk -> default, no crash.
    for bad in ((0, 0), (-5, 10), ("x", "y"), (None, None)):
        try:
            check(I.normalize_resolution(*bad) == (C.DEFAULT_WIDTH, C.DEFAULT_HEIGHT),
                  f"normalize{bad} should return the default")
        except Exception as exc:                                   # noqa: BLE001
            check(False, f"normalize{bad} raised {exc!r}")

    # Multiple of 16 always (ComfyUI latents).
    for pair in ((1000, 667), (1234, 999), (777, 333)):
        w, h = I.normalize_resolution(*pair)
        check(w % 16 == 0 and h % 16 == 0, f"normalize{pair} -> {w}x{h} not snapped")


def test_fix_image_params():
    print("[fix_image_params]")
    land_w, land_h = C.resolve_image_size("16:9", "high")
    port_w, port_h = C.resolve_image_size("9:16", "high")

    # Unpinned: the text still decides orientation (existing behaviour kept).
    w, h = I.fix_image_params("a wide city panorama", "", port_w, port_h)
    check(w > h, f"unpinned landscape text should flip to landscape, got {w}x{h}")
    w, h = I.fix_image_params("портрет девушки в полный рост", "", land_w, land_h)
    check(h > w, f"unpinned portrait text should flip to portrait, got {w}x{h}")

    # Pinned: the user's explicit choice outranks the text heuristic.
    w, h = I.fix_image_params("a wide city panorama", "", port_w, port_h, pinned=True)
    check(h > w, f"pinned portrait must survive landscape wording, got {w}x{h}")
    check((w, h) == (port_w, port_h), f"pinned size was altered: {w}x{h}")
    w, h = I.fix_image_params("портрет девушки", "", land_w, land_h, pinned=True)
    check(w > h, f"pinned landscape must survive portrait wording, got {w}x{h}")

    # Either way the result stays on the rails and keeps the pixel budget.
    for pinned in (True, False):
        w, h = I.fix_image_params("city", "", 1900, 1060, pinned=pinned)
        check(w * h > 1_200_000, f"pinned={pinned}: budget lost ({w}x{h})")


def test_session_size():
    print("[session size on ctx]")
    check(I.session_image_size(_Ctx()) == (C.DEFAULT_WIDTH, C.DEFAULT_HEIGHT),
          "an unset ctx should give the configured default")
    check(I.session_image_size(_Ctx("9:16", "ultra")) == C.resolve_image_size("9:16", "ultra"),
          "ctx choice is not honoured")
    check(I.session_image_size(object()) == (C.DEFAULT_WIDTH, C.DEFAULT_HEIGHT),
          "a ctx without the attributes must not crash")
    check(I.session_size_pinned(_Ctx("1:1", "high")) is True, "explicit aspect is pinned")
    check(I.session_size_pinned(_Ctx("", "ultra")) is False,
          "quality alone must NOT pin orientation")
    check(I.session_size_pinned(_Ctx()) is False, "unset ctx is not pinned")


def test_parse_defaults():
    print("[parse_generation_params]")
    # No size in the tag -> the caller's default wins.
    _, _, _, _, w, h = I.parse_generation_params("a red car", defaults=(1024, 768))
    check((w, h) == (1024, 768), f"tag with no size ignored the caller default: {w}x{h}")
    # No default given -> module default.
    _, _, _, _, w, h = I.parse_generation_params("a red car")
    check((w, h) == (C.DEFAULT_WIDTH, C.DEFAULT_HEIGHT),
          f"no-default call should use the config default, got {w}x{h}")
    # An explicit size in the tag still wins over the caller default.
    _, _, _, _, w, h = I.parse_generation_params(
        "a red car | width=1280 | height=720", defaults=(1024, 768))
    check((w, h) == (1280, 720), f"explicit width=/height= was overridden: {w}x{h}")
    # Resolution keyword in the prompt also wins.
    _, _, _, _, w, h = I.parse_generation_params("a red car in 4k", defaults=(1024, 768))
    check((w, h) == (3840, 2160), f"4k keyword ignored: {w}x{h}")
    # WxH in the prompt wins.
    _, _, _, _, w, h = I.parse_generation_params("a poster 1200x1600", defaults=(1024, 768))
    check((w, h) == (1200, 1600), f"WxH in prompt ignored: {w}x{h}")


# The shapes the model actually returns. The prefill ends inside "[generate]",
# so the usual reply is a bare BODY plus the closer; the fully tagged form and
# the no-tag fallback both happen too, and all three have to end up at the
# session's size rather than at a hardcoded default.
_RESPONSES = [
    ("[generate]a cat, studio light[/generate]", "fully tagged"),
    ("a cat, studio light[/generate]",           "prefill continuation"),
    ("Thought process: I will draw a cat.",      "no tag at all"),
    ("",                                         "empty response"),
]


def test_tool_schema():
    print("[tool schema]")
    import tools
    args = tools.GenerateImageArgs(description="x", width=99999, height=99999)
    check(args.width <= C.IMAGE_MAX_SIDE and args.height <= C.IMAGE_MAX_SIDE,
          f"schema clamp ignores IMAGE_MAX_SIDE ({args.width}x{args.height})")
    check(args.width >= C.IMAGE_MIN_SIDE, "schema clamp ignores IMAGE_MIN_SIDE")
    big = tools.GenerateImageArgs(description="x", width=2048)
    check(big.width == 2048, "2048 must be reachable now (old ceiling was 1920)")
    unset = tools.GenerateImageArgs(description="x")
    check(unset.width is None and unset.height is None,
          "unset size must stay None so the session choice applies")


# ── 3. the Telegram picker ────────────────────────────────────────────────────
def test_tg_picker():
    print("[telegram picker]")
    import tg_bot as T

    class S:
        image_aspect = ""
        image_quality = ""

    # every aspect the config knows is offered, exactly once
    sess = S()
    kb = T._size_menu_kb(sess, "en")
    datas = [b["callback_data"] for row in kb["inline_keyboard"] for b in row]
    for aspect in C.IMAGE_ASPECTS:
        want = "size:ar:" + aspect.replace(":", "x")
        check(datas.count(want) == 1, f"aspect {aspect} missing/duplicated in the picker")
    for quality in C.IMAGE_QUALITIES:
        check(datas.count("size:q:" + quality) == 1,
              f"quality {quality} missing/duplicated in the picker")
    check("size:ar:auto" in datas, "no Auto option")
    check(len(datas) == len(set(datas)), "duplicate callback_data in the picker")

    # the tick marks the ACTIVE choice, in both languages, for every option
    for lang in ("en", "ru"):
        sess = S()
        kb = T._size_menu_kb(sess, lang)
        ticked = [b["text"] for row in kb["inline_keyboard"] for b in row if "✅" in b["text"]]
        # one tick per picker row group: aspect, quality (the engine rows went
        # with Qwen 2.1, 2026-09-25)
        check(len(ticked) == 2, f"{lang}: unset session should tick Auto + default quality "
                                f"(got {ticked})")
        check(not any(b["callback_data"][:8] in ("size:ge:", "size:ee:")
                      for row in kb["inline_keyboard"] for b in row),
              f"{lang}: engine rows are back in the picker")
        check(any("size:ar:auto" == b["callback_data"]
                  for row in kb["inline_keyboard"] for b in row if "✅" in b["text"]),
              f"{lang}: Auto is not ticked when no aspect is chosen")

        for aspect in C.IMAGE_ASPECTS:
            s2 = S(); s2.image_aspect = aspect
            kb2 = T._size_menu_kb(s2, lang)
            tick = [b["callback_data"] for row in kb2["inline_keyboard"]
                    for b in row if "✅" in b["text"] and b["callback_data"].startswith("size:ar:")]
            check(tick == ["size:ar:" + aspect.replace(":", "x")],
                  f"{lang}: tick not on {aspect} (got {tick})")
        for quality in C.IMAGE_QUALITIES:
            s2 = S(); s2.image_quality = quality
            kb2 = T._size_menu_kb(s2, lang)
            tick = [b["callback_data"] for row in kb2["inline_keyboard"]
                    for b in row if "✅" in b["text"] and b["callback_data"].startswith("size:q:")]
            check(tick == ["size:q:" + quality], f"{lang}: quality tick not on {quality}")

    # the header states the resolved pixels, translated
    for lang, marker in (("en", "MP"), ("ru", "Мп")):
        s2 = S(); s2.image_aspect = "9:16"; s2.image_quality = "ultra"
        w, h = C.resolve_image_size("9:16", "ultra")
        text = T._size_menu_text(s2, lang)
        check(f"{w}×{h}" in text, f"{lang}: header does not show the resolved size")
        check(marker in text, f"{lang}: header not localized ({marker!r} missing)")
        check("size_title" not in text and "{w}" not in text,
              f"{lang}: unformatted placeholder leaked into the header")
    check(T._size_menu_text(S(), "en") != T._size_menu_text(S(), "ru"),
          "the picker header is identical in both languages")

    # the Draw menu exposes it, in both languages, and the label maps back
    for lang in ("en", "ru"):
        labels = [b for row in T._draw_kb(lang)["keyboard"] for b in row]
        check(T._b("size", lang) in labels, f"{lang}: no Size button in the Draw menu")
        check(T._LABEL2KEY.get(T._b("size", lang)) == "size",
              f"{lang}: Size label does not map back to its key")
    check(T._DIRECT_KB.get("size") == "__image_size__", "Size button has no action")
    check(T._b("size", "en") != T._b("size", "ru"), "Size button is not translated")

    # /size is advertised in both languages
    cmds = {c[0]: c for c in T.TelegramBot._COMMANDS}
    check("size" in cmds, "/size is not in the command menu")
    if "size" in cmds:
        check(cmds["size"][1] != cmds["size"][2], "/size description is not translated")


def test_tg_callback(tmpdir):
    print("[telegram callback]")
    import tg_bot as T
    T.redirect_data_dir(tmpdir)

    sent, edited, logged = [], [], []

    class Bot(T.TelegramBot):
        def _api_post(self, method, payload=None, **kw):
            return {"ok": True, "result": {"message_id": 1}}

        def _edit_text(self, chat_id, msg_id, text, parse_mode=None, keyboard=None):
            edited.append((chat_id, msg_id, text, keyboard))

        def _send_text(self, chat_id, text, parse_mode=None, keyboard=None):
            sent.append((chat_id, text, keyboard))

    bot = Bot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    bot._activity.log = lambda *a, **k: logged.append(a)
    chat = 990501
    bot._user_store.put(T._User(chat_id=chat, name="U", status="approved"))

    def press(data, with_msg=True):
        upd = {"callback_query": {"id": "1", "data": data,
                                  "from": {"id": chat},
                                  "message": ({"chat": {"id": chat}, "message_id": 7}
                                              if with_msg else {"chat": {"id": chat}})}}
        bot._dispatch(upd)
        return bot._get_session(chat)

    # a tap is recorded and persisted
    s = press("size:ar:9x16")
    check(s.image_aspect == "9:16", f"aspect tap not recorded (got {s.image_aspect!r})")
    check(bot._store.get(chat).image_aspect == "9:16", "aspect not persisted to the store")
    s = press("size:q:ultra")
    check(s.image_quality == "ultra", f"quality tap not recorded (got {s.image_quality!r})")
    check(bot._store.get(chat).image_quality == "ultra", "quality not persisted")

    # redrawn in place, not appended
    check(len(edited) >= 2, "the picker is not redrawn in place")
    if edited:
        check(edited[-1][1] == 7, "redraw targeted the wrong message")
        last_kb = edited[-1][3]
        ticked = [b["callback_data"] for row in last_kb["inline_keyboard"]
                  for b in row if "✅" in b["text"]]
        check("size:q:ultra" in ticked, "the redrawn keyboard does not show the new choice")
        check("size:ar:9x16" in ticked, "the redraw forgot the earlier choice")

    # Auto clears the pin
    s = press("size:ar:auto")
    check(s.image_aspect == "", f"Auto did not clear the aspect (got {s.image_aspect!r})")
    check(s.image_quality == "ultra", "Auto must not reset the quality tier")

    # garbage never changes state and never raises
    s = press("size:q:enormous")
    check(s.image_quality == "ultra", "unknown quality was accepted")
    s = press("size:ar:42x1")
    check(s.image_aspect == "", "unknown aspect was accepted")
    for junk in ("size:", "size:ar", "size:zz:1", "size"):
        try:
            press(junk)
        except Exception as exc:                                   # noqa: BLE001
            check(False, f"malformed callback {junk!r} raised {exc!r}")
    check(True, "malformed callbacks survived")

    # the picker has a way back to the Draw menu (live 10-02: a dead end)
    datas = [b["callback_data"] for row in T._size_menu_kb(s, "ru")["inline_keyboard"] for b in row]
    check("nav:back" in datas, "the size picker has no Back button")
    before = len(sent)
    press("nav:back")
    kb = sent[-1][2] if len(sent) > before else {}
    check(isinstance(kb, dict) and "keyboard" in kb, "Back did not bring the Draw menu keyboard")
    import tg_transport as TT
    check(TT._fit_markup({"keyboard": [["a"]]}).get("is_persistent") is True,
          "reply menus can be collapsed by the client")

    # no message_id (e.g. an old/inaccessible message) -> still answers
    before = len(sent)
    press("size:q:high", with_msg=False)
    check(len(sent) > before, "with no message to edit the user gets no confirmation")

    # the choice round-trips through serialization
    s = bot._get_session(chat)
    s.image_aspect, s.image_quality = "3:2", "standard"
    d = s.to_dict()
    check(d.get("image_aspect") == "3:2" and d.get("image_quality") == "standard",
          "size choice is missing from to_dict — it would not survive a restart")
    s2 = T._Session(chat, d)
    check((s2.image_aspect, s2.image_quality) == ("3:2", "standard"),
          "size choice does not survive a reload")
    check(T._Session(chat, {}).image_aspect == "",
          "a fresh session must start unpinned")

    # the Draw ▸ Size button opens the picker
    before = len(sent)
    # _resolve_and_push is where button keys are acted on (after the debounce
    # window), so drive that directly rather than racing the timer.
    bot._resolve_and_push(chat, [{"type": "text", "text": T._b("size", "en")}])
    opened = [x for x in sent[before:] if x[2] and "inline_keyboard" in x[2]]
    check(len(opened) == 1, "the Size button did not open the picker")


def test_orientation_not_pixels():
    """The MODEL chooses the shape; the USER's quality setting chooses the pixels.

    Live regression: IMAGE_GENERATOR_PROMPT dictated literal sizes (960x544 /
    544x960 / 768x768), so the model wrote width=960 into every [generate] tag —
    and an explicit width always beat the caller's default. Every picture came
    back at 0.5 MP no matter what 📐 Size said (measured: 960x544 while the
    session resolved 1680x944). That is the "blurry" complaint, and the size
    picker could not fix it because the prompt was overriding the picker.
    """
    print("\n[orientation, not pixels]")
    # (The prose prompt that dictated sizes, IMAGE_GENERATOR_PROMPT, was removed
    # with the old model; the parser contract below is what still matters.)
    SESSION = (1680, 944)     # what resolve_image_size gives at the default quality
    for shape, want in (("landscape", (1680, 944)),
                        ("portrait", (944, 1680))):
        got = I.parse_generation_params(
            f"A scene | steps=8 | cfg=1.0 | orientation={shape} | seed=5",
            defaults=SESSION)[4:]
        check(got == want, f"orientation={shape} -> {want} (got {got})")
    sq = I.parse_generation_params(
        "A logo | steps=8 | orientation=square | seed=5", defaults=SESSION)[4:]
    check(sq[0] == sq[1], f"orientation=square is square (got {sq})")
    check(abs(sq[0] * sq[1] - SESSION[0] * SESSION[1]) < SESSION[0] * SESSION[1] * 0.05,
          f"a square keeps the session's megapixel budget (got {sq})")

    # the escape hatches must still work
    px = I.parse_generation_params(
        "A scene | steps=8 | width=1024 | height=1024 | seed=5", defaults=SESSION)[4:]
    check(px == (1024, 1024), f"an explicit pixel size still wins (got {px})")
    k4 = I.parse_generation_params(
        "A scene in 4k | steps=8 | seed=5", defaults=SESSION)[4:]
    check(k4 == (3840, 2160), f"a resolution keyword still wins (got {k4})")
    none = I.parse_generation_params(
        "A scene | steps=8 | seed=5", defaults=SESSION)[4:]
    check(none == SESSION, f"saying nothing keeps the session size (got {none})")
    junk = I.parse_generation_params(
        "A scene | steps=8 | orientation=diagonal | seed=5", defaults=SESSION)[4:]
    check(junk == SESSION, f"an unknown shape changes nothing (got {junk})")

    # A CHOICE outranks the model's reading of the scene. Live regression: a chat
    # that had picked 9:16 asked for "a cat on a windowsill", the model said
    # orientation=landscape, and the picture came back 1680x944 — the picker
    # silently overruled. Same rule fix_image_params already applies to the text
    # heuristic.
    PICKED = C.resolve_image_size("9:16", "high")
    for shape in ("landscape", "square"):
        got = I.parse_generation_params(
            f"A cat on a windowsill | steps=8 | orientation={shape} | seed=5",
            defaults=PICKED, pinned=True)[4:]
        check(got == PICKED,
              f"a pinned aspect beats orientation={shape} (got {got}, want {PICKED})")
    unpinned = I.parse_generation_params(
        "A cat on a windowsill | steps=8 | orientation=landscape | seed=5",
        defaults=PICKED, pinned=False)[4:]
    check(unpinned[0] > unpinned[1],
          f"…but with nothing pinned the model still chooses (got {unpinned})")
    # width=/height= inside a [generate] TAG is the MODEL's own generated text,
    # not the end user typing pixels — indistinguishable in trustworthiness from
    # a resolution keyword in prose ("in stunning 4k"). A 12-family real-chat
    # audit (2026-08-03) proved this exact path let a model-written width=1920/
    # height=1080 silently override a 9:16 pin, the same "prompt carries a value
    # the resolver owns" class as the original 960x544 bug. So a pin must beat
    # this too — reversing what this check used to assert.
    px = I.parse_generation_params(
        "A scene | steps=8 | width=1024 | height=512 | seed=5",
        defaults=PICKED, pinned=True)[4:]
    check(px == PICKED, f"a pinned aspect beats an in-tag width=/height= too "
                        f"(got {px}, want {PICKED})")
    unpinned_px = I.parse_generation_params(
        "A scene | steps=8 | width=1024 | height=512 | seed=5",
        defaults=PICKED, pinned=False)[4:]
    check(unpinned_px == (1024, 512),
          f"…but with nothing pinned, width=/height= still applies (got {unpinned_px})")

    # The rule above is only worth anything if the REAL caller passes the pin.
    import inspect as _insp
    _src = _insp.getsource(I.generate_image_with_refinement)
    check("pinned=_image.session_size_pinned(ctx)" in _src,
          "generate_image_with_refinement must pass the session's pin through")


def test_tg_ctx_swap():
    print("[telegram ctx swap]")
    import ast
    # _run_task_inner moved to tg_tasks.py when TelegramBot was split into
    # mixins; scan the whole family so this can never silently find nothing.
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fn = None
    src = ""
    for _f in ("bot/tg_bot.py", "bot/tg_tasks.py", "bot/tg_dispatch.py",
               "bot/tg_commands.py", "bot/tg_resolve.py"):
        _text = open(os.path.join(_root, _f), encoding="utf-8").read()
        for node in ast.walk(ast.parse(_text)):
            if isinstance(node, ast.FunctionDef) and node.name == "_run_task_inner":
                fn, src = node, _text   # keep the source that OWNS the function
    check(fn is not None, "_run_task_inner not found")
    if not fn:
        return
    def _assigns_to_ctx(scope, attr):
        """Assignments of the form ctx.<attr> = <expr>, as (expr_source, node)."""
        out = []
        for node in ast.walk(scope):
            if not isinstance(node, ast.Assign):
                continue
            for tgt in node.targets:
                if (isinstance(tgt, ast.Attribute) and tgt.attr == attr
                        and isinstance(tgt.value, ast.Name) and tgt.value.id == "ctx"):
                    out.append((ast.dump(node.value), node))
        return out

    finally_nodes = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Try):
            for stmt in node.finalbody:
                finally_nodes.extend(ast.walk(stmt))

    for attr in ("image_aspect", "image_quality"):
        assigns = _assigns_to_ctx(fn, attr)
        # Naming the attribute is not enough — it must be ASSIGNED FROM the
        # session, or the user's choice never reaches the generator.
        check(any("sess" in src for src, _ in assigns),
              f"ctx.{attr} is never assigned from the session — the choice cannot reach ComfyUI")
        # ...and it must not leak to another chat. That used to be done by saving
        # the old value and restoring it in a `finally`, which only works while
        # exactly ONE task runs at a time. The bot now runs several concurrently,
        # so the guarantee comes from each task holding its OWN scoped Context
        # (tg_bot._scoped_ctx) instead — nothing global is written, so there is
        # nothing to restore. Assert the real invariant, not the old mechanism.
        check("_scoped_ctx(" in src,
              f"ctx.{attr} can leak: the task does not run on a scoped context")

    # The isolation itself, behaviourally: two concurrent tasks must not see
    # each other's size, and neither may write it onto the shared context.
    import types as _types
    import tg_bot as T
    _shared = _types.SimpleNamespace(image_aspect="", image_quality="")
    _a = T._scoped_ctx(_shared, cancel_event=None)
    _b = T._scoped_ctx(_shared, cancel_event=None)
    _a.image_aspect, _a.image_quality = "1:1", "high"
    _b.image_aspect, _b.image_quality = "16:9", "low"
    check(_a.image_aspect == "1:1" and _b.image_aspect == "16:9",
          "two concurrent tasks share ctx.image_aspect")
    check(_shared.image_aspect == "",
          "a task wrote its size onto the SHARED context")


def main():
    import tempfile
    print("=" * 60)
    print("IMAGE SIZE / ASPECT SELECTION")
    print("=" * 60)
    test_resolver()
    test_normalize()
    test_fix_image_params()
    test_session_size()
    test_parse_defaults()
    test_tool_schema()
    test_orientation_not_pixels()
    test_tg_picker()
    # mkdtemp, not TemporaryDirectory: sqlite keeps the user store open for the
    # life of the process, so Windows refuses to unlink it on the way out.
    test_tg_callback(tempfile.mkdtemp(prefix="tgtest_size_"))
    test_tg_ctx_swap()

    print("=" * 60)
    print(f"PASS {PASS}  FAIL {FAIL}")
    if FAILURES:
        for f in FAILURES:
            print("  - " + f)
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
