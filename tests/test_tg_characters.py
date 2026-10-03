"""Creativity → 🧑 Персонажи, driven through the REAL bot.

The interesting failures in this flow are all about a keyboard outliving the
thing it points at: Telegram buttons never expire, so between the tap and the
prompt a character can be unticked or its adapter file deleted. Rendering
anyway would draw a stranger and put the character's name on it.

Nothing here reaches ComfyUI: the render entry point is replaced, and what is
checked is WHAT it would have been asked for.

Run: venv/Scripts/python.exe tests/test_tg_characters.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T

# MUST come before any bot or store is built: without it the suite writes into
# the operator's live tg_users.db.
_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_chars_")
T.redirect_data_dir(_DATA_DIR)

import characters as C
# Same reasoning: the registry is a module-level path, and an unredirected run
# would add test characters to the operator's real list.
_STORE_DIR = tempfile.mkdtemp(prefix="tgtest_charstore_")
C.redirect_store(os.path.join(_STORE_DIR, "characters.json"))

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name
          + ("" if cond or not detail else "  -- " + str(detail)[:300]))
    if not cond and os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


CID = 999878


def _adapter(name="hero.safetensors"):
    p = os.path.join(_STORE_DIR, name)
    with open(p, "wb") as fh:
        fh.write(b"x")
    return p


def _bot():
    for slug in [c["slug"] for c in C.list_characters()]:
        C.delete(slug)
    # A ctx with a model NAME, not None: a character render plans its Ideogram
    # caption with the LLM, so the bot refuses one when no model is loaded --
    # which is the desktop app's "start without a model" state, not this one.
    # Handing it None here would test that refusal instead of the flow.
    import types as _types
    _ctx = _types.SimpleNamespace(model_name="house-model", set_stage=lambda s: None)
    bot = T.TelegramBot("123:TEST", lambda: _ctx, lambda: None,
                        lambda: {}, silent_mode=True)
    bot.sent, bot.kbs, bot.photos, bot.renders = [], [], [], []
    def _send(cid, text, **k):
        bot.sent.append(text)
        bot.kbs.append(k.get("keyboard"))
        return 1
    bot._send_text = _send
    bot._send_get_id = _send
    bot._send_photo = lambda cid, path, **k: bot.photos.append(path) or True
    bot._activity.log = lambda *a, **k: None
    bot._api_post = lambda *a, **k: {}
    bot._user_store.put(T._User(chat_id=CID, name="U", status="approved"))
    # The render is the only slow, external part; capture the request instead.
    # task/cancel/status_id arrived when the character render gained the same
    # ⛔ button and status line every other render has; they are captured too,
    # because a render that is started WITHOUT them is one the user cannot stop.
    def _fake_render(chat_id, rec, text, lang, task=None, cancel=None,
                     status_id=None, ref_file_ids=None):
        bot.renders.append((rec["slug"], text, rec.get("lora_file")))
        bot.render_refs.append(list(ref_file_ids or []))
        bot.render_cancels.append((task, cancel, status_id))
        sess = bot._get_session(chat_id)
        sess.reg_state = ""
        bot._store.put(sess)
    bot.render_cancels = []
    bot.render_refs = []
    bot._render_character = _fake_render
    # Sessions live in a file that outlives one bot object, so without this a
    # test inherits the previous one's armed capture mode -- and then passes or
    # fails for a reason that has nothing to do with what it claims to check.
    bot._sessions.pop(CID, None)
    sess = bot._get_session(CID)
    sess.reg_state = ""
    sess.char_slug = ""
    sess.pending_prefix = ""
    bot._store.put(sess)
    return bot


def _say(bot, text):
    """One ordinary text message, routed the way a real update is."""
    return bot._user_gate(CID, {"chat": {"id": CID}, "text": text,
                                "from": {"id": CID}})


def _texts(bot):
    return " || ".join(bot.sent)


def _sess(bot):
    return bot._get_session(CID)


def test_no_model_means_an_honest_refusal_not_a_stranger():
    """The desktop app can be started with no LLM loaded, to leave the card
    free for a training run. A character render is not LLM-free -- its Ideogram
    caption is PLANNED by the model -- so without one it would spend minutes and
    deliver nothing. The user has to be told, not left watching a status line."""
    bot = _bot()
    C.upsert("hero", name="Герой", status="ready",
             lora_file=_adapter(), tg_enabled=True)
    bot._pick_character(CID, _sess(bot), "ru", "hero")
    bot._get_ctx = lambda: None                      # runtime not up at all
    bot.sent.clear()
    _say(bot, "на крыше")
    check("nothing was rendered without a model", bot.renders == [], bot.renders)
    check("and the user was told why", "модел" in _texts(bot).lower()
          or "видеокарт" in _texts(bot).lower(), _texts(bot))

    import types as _types
    bot2 = _bot()
    C.upsert("hero", name="Герой", status="ready",
             lora_file=_adapter(), tg_enabled=True)
    bot2._pick_character(CID, _sess(bot2), "ru", "hero")
    bot2._get_ctx = lambda: _types.SimpleNamespace(model_name="")   # unloaded on purpose
    bot2.sent.clear()
    _say(bot2, "на крыше")
    check("an empty model name refuses the same way", bot2.renders == [],
          bot2.renders)


def test_only_ticked_and_trained_characters_are_listed():
    bot = _bot()
    C.upsert("nodata", name="Без адаптера", tg_enabled=True)          # no file
    C.upsert("offtg", name="Не в тг", status="ready",
             lora_file=_adapter("offtg.safetensors"), tg_enabled=False)
    C.upsert("hero", name="Герой", status="ready",
             lora_file=_adapter(), tg_enabled=True)
    bot._send_characters_menu(CID, "ru")
    kb = [k for k in bot.kbs if k][-1]
    labels = [b[0]["text"] for b in kb["inline_keyboard"]]
    check("only the ready+ticked character is offered", labels == ["Герой"], labels)


def test_an_empty_list_says_where_characters_come_from():
    bot = _bot()
    bot._send_characters_menu(CID, "ru")
    body = _texts(bot)
    check("empty list explains itself", "Персонажи" in body or "приложении" in body,
          body)
    check("no inline keyboard when there is nobody",
          not any(k and "inline_keyboard" in k for k in bot.kbs), bot.kbs)


def test_picking_arms_the_next_message_as_a_scene():
    bot = _bot()
    C.upsert("hero", name="Герой", trigger="hero", status="ready",
             lora_file=_adapter(), tg_enabled=True)
    bot._cb_pick_character(CID, "char:hero")
    check("prompt mode is armed", _sess(bot).reg_state == "char_prompt",
          _sess(bot).reg_state)
    check("the chosen character is remembered", _sess(bot).char_slug == "hero")
    _say(bot, "на фоне гор")
    check("the scene reaches the renderer", len(bot.renders) == 1, bot.renders)
    if bot.renders:
        slug, text, lora = bot.renders[0]
        check("with the right character and text",
              slug == "hero" and text == "на фоне гор", bot.renders[0])
    check("the mode is cleared after the render", _sess(bot).reg_state == "",
          _sess(bot).reg_state)


def test_a_character_untricked_after_the_tap_is_not_drawn():
    """The keyboard outlives the permission: this is the defect that matters."""
    bot = _bot()
    C.upsert("hero", name="Герой", status="ready",
             lora_file=_adapter(), tg_enabled=True)
    bot._cb_pick_character(CID, "char:hero")
    C.set_tg_enabled("hero", False)              # unticked in the desktop app
    _say(bot, "на фоне гор")
    check("nothing was rendered", bot.renders == [], bot.renders)
    check("and the user is told why", "недоступен" in _texts(bot), _texts(bot))


def test_a_deleted_adapter_after_the_tap_is_not_drawn():
    bot = _bot()
    path = _adapter("gone.safetensors")
    C.upsert("hero", name="Герой", status="ready", lora_file=path,
             tg_enabled=True)
    bot._cb_pick_character(CID, "char:hero")
    os.unlink(path)                              # the file vanishes
    _say(bot, "на фоне гор")
    check("a missing adapter stops the render", bot.renders == [], bot.renders)


def test_tapping_a_stale_button_for_an_unknown_character():
    bot = _bot()
    bot._cb_pick_character(CID, "char:ghost")
    check("an unknown slug does not arm anything",
          _sess(bot).reg_state != "char_prompt", _sess(bot).reg_state)
    check("and it answers instead of going silent", bool(_texts(bot)))


def test_an_unrelated_button_press_disarms_the_scene_capture():
    """Same trap the feedback mode had: a tap is not an answer to 'type the
    scene', and leaving it armed files the user's next ordinary message as a
    picture request."""
    bot = _bot()
    C.upsert("hero", name="Герой", status="ready",
             lora_file=_adapter(), tg_enabled=True)
    bot._cb_pick_character(CID, "char:hero")
    bot._cb_abandon_capture(CID)
    check("scene capture is dropped on an unrelated press",
          _sess(bot).reg_state == "", _sess(bot).reg_state)


def test_photos_in_character_mode_are_references_not_silence():
    """Live 2026-09-17 16:10: two stills + «Нарисуй Степана так, чтобы он ...
    в этом стиле» sent while НейроСтепан was armed got NO reaction -- the
    gate consumed a photo message as an empty scene. Now every page of the
    album is collected, and one render starts after the quiet period with
    the caption as the scene and the photos as references."""
    import time as _t
    import tg_characters as TC
    TC._REF_QUIET_S = 0.3
    bot = _bot()
    C.upsert("hero", name="Герой", status="ready",
             lora_file=_adapter(), tg_enabled=True)
    bot._pick_character(CID, _sess(bot), "ru", "hero")
    bot.sent.clear()
    photo = lambda fid, cap: {"chat": {"id": CID}, "from": {"id": CID},
                              "photo": [{"file_id": fid, "width": 640, "height": 480}],
                              "caption": cap, "media_group_id": "g1"}
    r1 = bot._user_gate(CID, photo("ref1", "Нарисуй героя в этом стиле"))
    r2 = bot._user_gate(CID, photo("ref2", ""))
    check("photo pages are consumed by the character mode", r1 is False and r2 is False, (r1, r2))
    check("no render before the album settles", bot.renders == [], bot.renders)
    _t.sleep(1.0)
    check("ONE render after the quiet period", len(bot.renders) == 1, bot.renders)
    check("the caption is the scene", bot.renders and bot.renders[0][1] == "Нарисуй героя в этом стиле", bot.renders)
    check("both photos ride along as references", bot.render_refs == [["ref1", "ref2"]], bot.render_refs)
    check("the collected references are cleared afterwards",
          _sess(bot).char_ref_ids == [] and _sess(bot).char_ref_caption == "", _sess(bot).char_ref_ids)


def test_reference_description_folds_into_the_scene():
    """The vision fragment lends pose/outfit/style; identity stays with the LoRA."""
    import types as _types, tg_characters as TC, llm as _llm
    bot = _bot()
    bot._dl_bytes = lambda fid: b"img" + fid.encode()
    seen = []
    _llm.analyze_image_with_llm = lambda ctx, image_bytes=None, user_text="", system_prompt="", **k: (
        seen.append(system_prompt), "shirtless, flexing one arm, white wall, 80s film look")[-1]
    ctx = _types.SimpleNamespace(set_stage=lambda s: None)
    look = TC.CharactersMixin._describe_character_references(bot, ctx, ["a", "b"])
    check("every reference is read", len(seen) == 2 and "Do NOT describe who the person is" in seen[0], seen)
    check("the fragments are joined", look.count("flexing") == 2, look)


def test_delivered_render_carries_the_image_keyboard_and_msg_id():
    """Regression (live report, item 15): a character render was sent with
    _send_photo(chat_id, path) and NOTHING else -- no keyboard, no msg_id
    capture -- unlike every other picture path in the bot. It LOOKED like it
    never landed in the chat: no button under it worked, and replying to it
    resolved nothing. Drives the REAL _render_character (only the GPU call
    is stubbed), unlike the other tests here which replace _render_character
    itself and so never touch this delivery code at all."""
    import image_generate as G
    bot = _bot()
    C.upsert("hero", name="Герой", trigger="hero", status="ready",
             lora_file=_adapter(), tg_enabled=True)
    rec = C.get("hero")
    out_path = _adapter("rendered.png")   # any real file on disk

    calls = []
    def _fake_send_photo(cid, path, **k):
        calls.append((cid, path, k.get("keyboard")))
        bot._last_photo_msg_id = 4242
        return True
    bot._send_photo = _fake_send_photo
    orig_render = G.generate_image_with_comfy
    G.generate_image_with_comfy = lambda ctx, prompt, **k: out_path
    import tg_characters as TC
    try:
        # _bot() replaces the INSTANCE attribute _render_character with a
        # fake for every other test in this file; go around it to reach the
        # real method under test, unbound, exactly as tg_bot dispatches it.
        TC.CharactersMixin._render_character(bot, CID, rec, "на крыше", "ru")
    finally:
        G.generate_image_with_comfy = orig_render

    check("the picture was sent", len(calls) == 1, calls)
    if calls:
        _, sent_path, kb = calls[0]
        check("to the right image path", sent_path == out_path, sent_path)
        check("WITH an image-action keyboard, like every other picture path",
              bool(kb and kb.get("inline_keyboard")), kb)
    entry = next((e for e in _sess(bot).image_log if e.get("path") == out_path), None)
    check("registered in the image log", entry is not None, _sess(bot).image_log)
    if entry:
        check("with the message id the send actually returned",
              entry.get("msg_id") == 4242, entry)


def test_delivered_render_becomes_the_current_picture_for_follow_ups():
    """Regression (live report, item 15 follow-up): character rendering runs
    off the task queue on a throwaway _scoped_ctx copy, so a mutation to
    ctx.last_image_path never reaches sess. A plain-text question right
    after the render ("what's drawn here?") is seeded from
    sess.last_image_path (tg_tasks.py:1006) -- if this was never set, the
    next turn thinks there is no current picture at all and asks the user
    to upload a photo, even though one was just delivered with a keyboard."""
    import image_generate as G
    bot = _bot()
    C.upsert("hero", name="Герой", trigger="hero", status="ready",
             lora_file=_adapter(), tg_enabled=True)
    rec = C.get("hero")
    out_path = _adapter("rendered3.png")
    bot._send_photo = lambda cid, path, **k: True
    orig_render = G.generate_image_with_comfy
    G.generate_image_with_comfy = lambda ctx, prompt, **k: out_path
    import tg_characters as TC
    try:
        TC.CharactersMixin._render_character(bot, CID, rec, "на крыше", "ru")
    finally:
        G.generate_image_with_comfy = orig_render
    sess = _sess(bot)
    check("sess.last_image_path points at the delivered render",
          sess.last_image_path == out_path, sess.last_image_path)
    check("sess.last_image_prompt was captured too",
          bool(sess.last_image_prompt), sess.last_image_prompt)


def test_a_failed_send_is_reported_not_silent():
    """The old code ignored _send_photo's return value entirely; a failed
    send looked exactly like a successful one from the caller's side."""
    import image_generate as G
    bot = _bot()
    C.upsert("hero", name="Герой", trigger="hero", status="ready",
             lora_file=_adapter(), tg_enabled=True)
    rec = C.get("hero")
    out_path = _adapter("rendered2.png")
    bot._send_photo = lambda cid, path, **k: False       # Telegram rejected it
    orig_render = G.generate_image_with_comfy
    G.generate_image_with_comfy = lambda ctx, prompt, **k: out_path
    import tg_characters as TC
    try:
        bot.sent.clear()
        TC.CharactersMixin._render_character(bot, CID, rec, "на крыше", "ru")
    finally:
        G.generate_image_with_comfy = orig_render
    check("the user is told the render failed, not left guessing",
          bool(_texts(bot)), _texts(bot))


def _main():
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        try:
            fn()
        except Exception as exc:
            check(fn.__name__ + " raised", False, exc)
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d/%d passed" % (len(RESULTS) - len(bad), len(RESULTS)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_main())
