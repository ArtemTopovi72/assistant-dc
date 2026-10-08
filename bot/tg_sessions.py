"""Per-chat conversation state and the JSON store that persists it.

Lifted out of tg_bot.py. A _Session is everything the bot remembers about one
chat between updates — history, pending prefix, menu, the image log, research
depth, language. _Store is the thread-safe JSON file behind them.

Neither reads a module-level path: _Store is handed its file by TelegramBot,
which is what keeps redirect_data_dir() honest for this half of the state.
tg_bot re-exports both, because the suites build tg_bot._Session directly.
"""
import json
import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger("assistant.tg_bot")

import copy as _copy_mod


class _Session:
    # A picture pointed at by a button that led nowhere (Анимация/Стиль menu
    # left open) stayed "pointed at" until the next message, hours later: a
    # text Ozon search went down the vision path with the picture keyboard
    # (live 2026-09-27 20:24, a 16:56 photo). The gesture expires.
    TARGET_TTL_S = 15 * 60

    # Everything that makes the bot wait for the user's NEXT message (a topic,
    # a photo, a name, a city…). ✖️ Отмена under the prompt clears all of it
    # (user 2026-09-27: «всегда иметь возможность отменить … в любом сценарии»).
    _WAIT_STR = ("pending_prefix", "pending_instruction", "pending_photo", "pending_style_target", "pending_outfit_target",
                 "pending_animate_target", "cover_state", "cover_src", "song_draft",
                 # the step-by-step flows too: «Мэшап · трек 1 из 2» had no way out
                 "clone_state", "restyle_state", "continue_state", "book_state", "anim_voice_state", "voice_naming",
                 # ✍️ Свой вариант: the forwarded material waiting for the user's request
                 "fwd_own", "lyrics_state")
    _WAIT_BOOL = ("awaiting_animate_photo", "awaiting_style_photo")
    _NOT_CANCELLABLE = ("", "awaiting_login", "awaiting_name", "awaiting_password")

    FWD_KEEP = 5

    def remember_fwd(self, fid: str, text: str) -> None:
        self.fwd_recent[fid] = text
        while len(self.fwd_recent) > self.FWD_KEEP:
            self.fwd_recent.pop(next(iter(self.fwd_recent)))

    def waiting_for_input(self) -> bool:
        return (str(getattr(self, "reg_state", "")) not in self._NOT_CANCELLABLE
                or any(getattr(self, a, "") for a in self._WAIT_STR + self._WAIT_BOOL))

    def drop_waiting(self) -> None:
        if str(getattr(self, "reg_state", "")) not in self._NOT_CANCELLABLE:
            self.reg_state = ""
        for a in self._WAIT_STR:
            if hasattr(self, a):
                setattr(self, a, "")
        for a in self._WAIT_BOOL:
            if hasattr(self, a):
                setattr(self, a, False)

    REPLY_MODES = ("text", "both", "voice")

    @property
    def voice_on(self) -> bool:
        return self.reply_mode != "text"

    @voice_on.setter
    def voice_on(self, v: bool) -> None:
        self.reply_mode = ("both" if self.reply_mode == "text" else self.reply_mode) if v else "text"

    @property
    def target_image(self) -> str:
        import time
        if self._target and time.time() - self._target_ts > self.TARGET_TTL_S:
            self._target = ""
        return self._target

    @target_image.setter
    def target_image(self, v: str) -> None:
        import time
        self._target, self._target_ts = v or "", time.time()

    def __init__(self, chat_id: int, data: dict = None):
        d = data or {}
        self.chat_id        = chat_id
        self.history: list  = d.get("history", [])
        # Text by default: a voice note repeating the text doubled every answer (live 2026-09-29).
        self.reply_mode: str = d.get("reply_mode") if d.get("reply_mode") in self.REPLY_MODES else "text"
        # Model reasoning, per user; OFF unless they turn it on in Settings.
        self.think: bool = bool(d.get("think", False))
        # 📎 also send every picture as a file: sendPhoto is re-encoded to JPEG by Telegram
        self.photo_file: bool = bool(d.get("photo_file", False))
        self.pending_prefix = d.get("pending_prefix", "")
        self.last_link = None
        self.reg_state: str  = d.get("reg_state", "")   # ""|"awaiting_name"|"awaiting_password"
        self.reg_name:  str  = d.get("reg_name",  "")   # temp storage during registration
        self.is_admin: bool  = d.get("is_admin",  False)
        self.total_user_turns: int = d.get("total_user_turns", 0)  # per-user compaction counter
        # Per-session memory, isolated from the GUI's ctx.session_memory so Telegram
        # turns never bleed into the main app's conversation and vice-versa.
        self.tg_memory: list = d.get("tg_memory", [])
        # Per-session working image, isolated from the GUI's ctx.last_image_* for the
        # same reason as tg_memory. ctx.last_image_prompt is what redraw/edit reuse as
        # the base prompt (tools.py), so leaving it global meant one chat's prompt
        # steered another's — and survived a context clear.
        self.last_image_path:   str = d.get("last_image_path", "")
        # Every picture this chat has seen, newest LAST, capped at _IMAGE_LOG_MAX.
        # One "current image" slot was the whole problem: "describe the picture"
        # three messages later silently meant whatever was drawn most recently, so
        # the answer described an image the user was not looking at. Each entry is
        # {"id", "path", "msg_id", "label", "src", "ts"} and `msg_id` is what makes
        # a Telegram REPLY resolvable back to a file.
        self.image_log: list = d.get("image_log", [])
        # Set for ONE turn when the user pointed at a specific image (replied to it,
        # or pressed a button under it). Consumed by the turn, never inherited.
        self.target_image: str = d.get("target_image", "")
        self._target_ts: float = float(d.get("target_image_ts", 0) or 0)
        # A message the user replied to, quoted for the turn that follows, so
        # "what did you mean by that" refers to the line they pointed at.
        self.quoted_text: str = d.get("quoted_text", "")
        # Which character the next prompt draws. Session state, not a global:
        # two chats can stand in front of two different characters at once,
        # and a shared slot would render one user's prompt as the other's face.
        self.char_slug: str = d.get("char_slug", "")
        # Reference photos collected while a character is armed (file ids)
        # and the caption that came with them -- see tg_characters.
        self.char_ref_ids: list = list(d.get("char_ref_ids", []) or [])
        self.char_ref_caption: str = d.get("char_ref_caption", "")
        # The instruction held back while we ask WHICH picture it is about.
        self.pending_instruction: str = d.get("pending_instruction", "")
        # The picture the PREVIOUS turn was about: sent by the user, pointed
        # at, or delivered by the bot. A follow-up that plainly needs a picture
        # ("now remove the lettering") continues on it instead of asking
        # "which of the 6?" -- a chat that has drawn a car, read a receipt and
        # just recoloured a dress is not a fresh choice between six pictures
        # (live 2026-09-13, mega journey). Cleared by a turn with no picture.
        self.turn_image: str = d.get("turn_image", "")
        # The alphabet of the user's last TYPED message ("en"/"ru"/""). The
        # reply follows it for the turn, and for the button presses, emoji and
        # links that come after it: an English question in a Russian-language
        # chat got "Столица Австралии — Канберра" and the English "draw a
        # lighthouse" was followed by Russian captions on upscale/outpaint
        # (live 2026-09-13, mega journey). Only a message written wholly in one
        # script moves it; mixed text and machine payloads inherit.
        self.turn_lang: str = d.get("turn_lang", "")
        # «answer in English from now on»: sticks until asked otherwise.
        self.lang_pin: str = d.get("lang_pin", "")
        # When the pin last served a turn: a pin left from yesterday expired
        # instead of answering a Russian chat in English for good (10-03).
        self.lang_pin_ts: float = float(d.get("lang_pin_ts", 0) or 0)
        # The transcript of a forwarded voice note, held while we ask whether the
        # user wants it verbatim or summarised.
        self.fwd_transcript: str = d.get("fwd_transcript", "")
        # Identifies WHICH forward fwd_transcript belongs to. The choice buttons
        # (fwdv:text/sum/both) carry no per-message identity of their own — a
        # second forward arriving before the first is answered used to silently
        # overwrite the single fwd_transcript slot, so tapping the FIRST "which
        # do you want?" prompt (still on screen, indistinguishable from the
        # second) delivered the SECOND voice's content under it. Reproduced
        # live: forward A, ignore the prompt, forward B, tap A's button — B's
        # transcript comes back. Stamped fresh each time fwd_transcript is
        # (re)armed and carried in the keyboard's callback_data; a button whose
        # id no longer matches is stale and says so instead of silently acting
        # on whatever is currently pending.
        self.fwd_transcript_id: str = d.get("fwd_transcript_id", "")
        # True once this forward has already been acted on. The transcript is
        # deliberately KEPT after delivery so the inline keyboard -- which stays
        # on screen and clickable -- still works for the other two choices; only
        # the "next thing you type" shortcut is retired, so a much later
        # "перескажи" about something else cannot reach back to an old voice.
        self.fwd_transcript_done: bool = bool(d.get("fwd_transcript_done", False))
        # Every recent forward by id, so each on-screen prompt's buttons reach
        # ITS voice: two voices forwarded together made the first prompt answer
        # "that voice is gone" (live 2026-09-27 19:18).
        self.fwd_recent: dict = dict(d.get("fwd_recent") or {})
        # ✍️ Свой вариант pressed: the transcript the next typed message is about.
        self.fwd_own: str = d.get("fwd_own", "")
        # 🎙 Clone voice: "" | "want_audio" | "want_text" (tg_voice_clone)
        self.clone_state: str = d.get("clone_state", "")
        # 🎚 Remix (was Cover): "" | "want_audio" | "want_text" (tg_cover); the song on disk.
        self.cover_state: str = d.get("cover_state", "")
        self.cover_src: str = d.get("cover_src", "")
        # whose voice sings a cover: "" the original singer's (auto), "none" YuE2's own (no
        # conversion, the clearest words), or a trained star voice (rvc_voice.stars()). Kept.
        self.cover_voice: str = d.get("cover_voice", "")
        # 🎨 Restyle video: "" | "want_video" | "want_text" (tg_restyle)
        self.restyle_state: str = d.get("restyle_state", "")
        self.restyle_src: str = d.get("restyle_src", "")
        # ▶️ Continue video: "" | "want_video" | "want_text" (tg_continue)
        self.continue_state: str = d.get("continue_state", "")
        self.continue_src: str = d.get("continue_src", "")
        self.continue_tail: str = d.get("continue_tail", "")
        # photos of NEW people to bring into the continuation (sent after the clip)
        self.continue_people: list = list(d.get("continue_people") or [])
        # 📚 Audiobook: "" | "want_voice" | "want_book" (tg_audiobook); the narrator's reference
        self.book_state: str = d.get("book_state", "")
        self.book_ref: str = d.get("book_ref", "")
        self.book_ref_text: str = d.get("book_ref_text", "")
        # 🎙 voices for 🎬 Animate (tg_anim_voices): "" | "collect", and the samples
        self.anim_voice_state: str = d.get("anim_voice_state", "")
        self.song_draft: str = d.get("song_draft", "")     # ready lyrics awaiting «как есть / новый»
        # ✨/✍️ lyrics (tg_lyrics): "" | "improve" | "write", and the last result
        self.lyrics_state: str = d.get("lyrics_state", "")
        self.lyrics_last: str = d.get("lyrics_last", "")
        self.anim_voices: list = list(d.get("anim_voices") or [])
        # a clip request waiting for «свои голоса / стандартные», and the answer
        self.voice_pending: str = d.get("voice_pending", "")
        # 📚 the voice library (tg_voice_library) and the voice waiting for a name
        self.voices: list = list(d.get("voices") or [])
        self.voice_naming: str = d.get("voice_naming", "")
        self.voice_choice: str = d.get("voice_choice", "")
        self.clone_ref: str = d.get("clone_ref", "")
        self.clone_ref_text: str = d.get("clone_ref_text", "")
        # 🗣 Assistant voice: a frozen copy of a clone, so later clone
        # experiments do not change how the bot answers. "" = house voice.
        self.assistant_ref: str = d.get("assistant_ref", "")
        self.assistant_ref_text: str = d.get("assistant_ref_text", "")
        self.last_image_prompt: str = d.get("last_image_prompt", "")
        self.last_deck: dict = d.get("last_deck") or {}
        # Pinned facts (remember_fact) are ALSO global on ctx — same leak as
        # session memory, but worse: facts are injected on every single turn, so
        # one user's "remember that my address is …" would surface in another
        # user's prompts. Swapped per session around graph.invoke().
        self.tg_facts: list = d.get("tg_facts", [])
        self.lang: str = d.get("lang", "")          # ""=not chosen yet
        self.lang_chosen: bool = bool(d.get("lang_chosen", False))  # by /lang or the 🌐 button
        self.use_docs: bool = d.get("use_docs", False)
        # Image output size, chosen in Draw ▸ 📐 Size. Empty means "not chosen" —
        # which is not the same as the default value: an unchosen aspect lets the
        # prompt text decide orientation, a chosen one outranks it (image.fix_image_params).
        self.image_aspect:  str = d.get("image_aspect", "")
        self.image_quality: str = d.get("image_quality", "")
        # How hard 🔬 Deep Research digs, chosen in Search ▸ 🔬 Depth. Empty means
        # "not chosen" and resolves to DR_DEFAULT_DEPTH — the difference matters
        # only for the ✅ in the picker, since both behave the same.
        self.dr_depth: str = d.get("dr_depth", "")
        # Song settings, chosen in Creativity ▸ 🎛 Song settings. Empty means
        # "not chosen" -> the songwriter picks what suits the topic; a chosen
        # value becomes an EXPLICIT REQUIREMENT in the caption brief, which is
        # the top of MiniMax's own precedence order (see tg_music.py).
        self.music_genre:    str = d.get("music_genre", "")
        self.music_tempo:    str = d.get("music_tempo", "")
        self.music_vocal:    str = d.get("music_vocal", "")
        self.music_voice:    str = d.get("music_voice", "")     # 🎙 a star's RVC voice or ""
        self.music_duration: str = d.get("music_duration", "")
        # Weight preset (fast/quality/max). Unlike the others this has no
        # "no opinion": a render always uses SOME weights, so an empty
        # value resolves to music.DEFAULT_PRESET rather than to Auto.
        self.music_quality:  str = d.get("music_quality", "")
        # DiT sampler steps (music.STEPS_RANGE); "" = config.MUSIC_STEPS.
        self.music_steps:    str = d.get("music_steps", "")
        # Long-video settings (Creativity ▸ 🎬 Video) and the video waiting
        # for a ▶️ (see tg_video). Empty = default.
        self.video_chunk:    str = d.get("video_chunk", "")
        self.video_frames:   str = d.get("video_frames", "")
        self.video_out:      str = d.get("video_out", "")
        self.video_long:     str = d.get("video_long", "")
        self.long_video:     dict | None = d.get("long_video") or None
        self.last_task_text: str = d.get("last_task_text", "")   # for 🔄 Retry
        # Identity of the request last_task_text belongs to. Telegram buttons never
        # expire, so a Retry button can still be showing under an OLD failed
        # message after several unrelated turns. Without an id tying the button
        # to the specific request it was sent for, tapping that stale button
        # silently retried whatever the user asked MOST RECENTLY instead of the
        # request that actually failed. The button now carries this id
        # (retry:<id>) and the handler refuses to fire if it no longer matches.
        # Both fields are set at the moment each failure is actually DELIVERED
        # (tg_tasks.py's two `except Exception` retry-button sites, and the
        # crash-recovery notice), never at task-push time: a chat may have a
        # second task queued right behind this one — or admitted to run
        # concurrently, see _mark_interruptible — that hasn't started yet, and
        # setting these fields when it was merely PUSHED let that still-pending
        # task overwrite last_task_id before this one even got a chance to
        # fail, so its own genuinely-fresh Retry button was refused as "stale"
        # the instant it was shown (reproduced live via the adversarial
        # harness). Tying the write to delivery instead keeps the "newest
        # DELIVERED outcome wins" contract intact.
        self.last_task_id: str = d.get("last_task_id", "")
        # Which submenu the user is standing in ("", "draw", "search", "settings",
        # "library"). ↩ Back is one button shared by every submenu, and _LABEL2KEY
        # maps LABEL->key, so a second "back" key with the same label would silently
        # overwrite the first — the destination has to come from state, not the
        # button. Without it Back in Settings ▸ 📚 Documents jumped two levels, to
        # the main menu instead of back to Settings.
        self.menu: str = d.get("menu", "")
        # An instruction armed by a button that needs an image the user hasn't sent
        # yet (📷 Analyze Photo). Consumed by the next caption-less photo/album.
        self.pending_photo: str = d.get("pending_photo", "")
        # 🎭 Style pressed under a picture: the id of the TARGET image, waiting
        # for the user's next photo (the style reference). Consumed by the very
        # next photo/album in tg_resolve.py, which short-circuits straight to a
        # transfer_image task instead of the normal caption/vision flow.
        self.pending_style_target: str = d.get("pending_style_target", "")
        # 👗 pressed under a picture: the id of the picture to re-dress. The next
        # photo (caption or not) is the CLOTHING reference, not a new picture to
        # edit; plain text answers through pending_prefix as before.
        self.pending_outfit_target: str = d.get("pending_outfit_target", "")
        # 🎞 Animate photo pressed with no picture yet: waiting for the next
        # photo/album, which becomes the animation subject. Consumed in
        # tg_resolve.py, which then offers the animate-preset keyboard instead
        # of running the normal caption/vision flow on it.
        self.awaiting_animate_photo: bool = bool(d.get("awaiting_animate_photo", False))
        # The id of the picture an animate-preset/custom press should animate,
        # armed once that photo has arrived (mirrors pending_style_target).
        self.pending_animate_target: str = d.get("pending_animate_target", "")
        # 🎭 Change style pressed from the Creativity menu with no picture yet
        # (item 7: style must work for ANY photo, not only one already in the
        # chat) -- waiting for the next photo/album, which becomes
        # pending_style_target once it arrives, same as the existing
        # under-a-picture style flow from there on.
        self.awaiting_style_photo: bool = bool(d.get("awaiting_style_photo", False))
        # The last weather location this chat actually resolved (lat/lon/name/
        # country), so the "48 hours" / "pick a date" follow-up buttons under a
        # forecast can re-fetch WITHOUT re-typing (and re-geocoding, and possibly
        # re-running the LLM city-correction) the city all over again.
        self.wtw_loc: dict = d.get("wtw_loc", {}) or {}
        self._lock           = threading.Lock()

    def get_history(self):
        with self._lock: return _copy_mod.deepcopy(self.history)

    def set_history(self, msgs):
        with self._lock: self.history = msgs

    def clear_history(self):
        with self._lock: self.history = []

    def clear_context(self):
        """Full context reset for /clear and 🗑 Clear Chat.

        clear_history() alone was NOT a context clear: session memory still held the
        `generate: <prompt>` entries and last_image_prompt still held the previous
        prompt, so after clearing and asking for a plain image the model happily
        re-applied the earlier "add text" instruction. Everything that can carry an
        instruction into the next turn is reset here.
        """
        with self._lock:
            self.history = []
            self.tg_memory = []
            self.lang_pin = self.turn_lang = ""     # a fresh start speaks the chat's language
            self.last_link = None
            self.pending_prefix = ""
            # Navigation state, but it decides where ⬅ Back goes. Leaving it set
            # after a context wipe means Back walks up a menu the user is no
            # longer standing in.
            self.menu = ""
            self.pending_photo = ""
            self.pending_style_target = ""
            self.pending_outfit_target = ""
            self.awaiting_animate_photo = False
            self.pending_animate_target = ""
            self.awaiting_style_photo = False
            self.last_image_path = ""
            self.last_image_prompt = ""
            self.last_deck = {}
            # The register goes with the conversation: after a wipe, replying to an
            # old picture must not silently reach back into the cleared context.
            self.image_log = []
            self.target_image = ""
            self.quoted_text = ""
            self.pending_instruction = ""
            self.fwd_transcript = ""
            self.fwd_transcript_id = ""
            self.fwd_transcript_done = False
            self.fwd_recent = {}
            self.clone_state = ""
            self.cover_state = ""
            self.restyle_state = ""
            self.continue_state = ""
            self.continue_src = self.continue_tail = ""
            self.continue_people = []
            self.book_state = ""
            self.anim_voice_state = ""
            self.anim_voices = []
            self.voice_pending = ""
            self.voice_choice = ""
            self.voice_naming = ""
            self.lyrics_state = ""
            self.song_draft = ""
            # Pinned facts too. They were kept out of the wipe because they have
            # their own 🧠 button — but they are injected into EVERY turn, so a
            # user who cleared the chat and was then told their own earlier steps
            # back was reading facts that survived the clear. "Clear" has to mean
            # the model remembers nothing about this conversation.
            self.tg_facts = []
            self.total_user_turns = 0

    def get_tg_memory(self):
        with self._lock: return list(self.tg_memory)

    def set_tg_memory(self, items):
        with self._lock: self.tg_memory = list(items)

    def get_tg_facts(self):
        with self._lock: return list(self.tg_facts)

    def set_tg_facts(self, items):
        with self._lock: self.tg_facts = list(items)

    def to_dict(self):
        with self._lock:
            return {
                "history":          self.history,
                "reply_mode":       self.reply_mode,
                "think":            self.think,
                "photo_file":       self.photo_file,
                "pending_prefix":   self.pending_prefix,
                "reg_state":        self.reg_state,
                "reg_name":         self.reg_name,
                "is_admin":         self.is_admin,
                "total_user_turns": self.total_user_turns,
                "tg_memory":        self.tg_memory,
                "last_image_path":   self.last_image_path,
                "image_log":         self.image_log,
                "target_image":      self.target_image,
                "target_image_ts":   self._target_ts,
                "turn_lang":         self.turn_lang,
                "lang_pin":          self.lang_pin,
                "lang_pin_ts":       self.lang_pin_ts,
                "quoted_text":       self.quoted_text,
                "char_slug":         self.char_slug,
                "char_ref_ids":      self.char_ref_ids,
                "char_ref_caption":  self.char_ref_caption,
                "pending_instruction": self.pending_instruction,
                "fwd_transcript":    self.fwd_transcript,
                "fwd_transcript_id": self.fwd_transcript_id,
                "fwd_transcript_done": self.fwd_transcript_done,
                "fwd_recent":        self.fwd_recent,
                "fwd_own":           self.fwd_own,
                "clone_state":         self.clone_state,
                "cover_state":         self.cover_state,
                "cover_src":           self.cover_src,
                "cover_voice":         self.cover_voice,
                "restyle_state":       self.restyle_state,
                "restyle_src":         self.restyle_src,
                "continue_state":      self.continue_state,
                "continue_src":        self.continue_src,
                "continue_tail":       self.continue_tail,
                "continue_people":     self.continue_people,
                "book_state":          self.book_state,
                "book_ref":            self.book_ref,
                "book_ref_text":       self.book_ref_text,
                "anim_voice_state":    self.anim_voice_state,
                "song_draft":          self.song_draft,
                "lyrics_state":        self.lyrics_state,
                "lyrics_last":         self.lyrics_last,
                "anim_voices":         self.anim_voices,
                "voice_pending":       self.voice_pending,
                "voices":              self.voices,
                "voice_naming":        self.voice_naming,
                "voice_choice":        self.voice_choice,
                "clone_ref":           self.clone_ref,
                "clone_ref_text":      self.clone_ref_text,
                "assistant_ref":       self.assistant_ref,
                "assistant_ref_text":  self.assistant_ref_text,
                "last_image_prompt": self.last_image_prompt,
                "last_deck":         self.last_deck,
                "tg_facts":          self.tg_facts,
                "lang":              self.lang,
                "lang_chosen":       self.lang_chosen,
                "use_docs":          self.use_docs,
                "image_aspect":      self.image_aspect,
                "image_quality":     self.image_quality,
                "dr_depth":          self.dr_depth,
                "music_genre":       self.music_genre,
                "music_tempo":       self.music_tempo,
                "music_vocal":       self.music_vocal,
                "music_voice":       self.music_voice,
                "music_duration":    self.music_duration,
                "music_quality":     self.music_quality,
                "music_steps":       self.music_steps,
                "video_chunk":       self.video_chunk,
                "video_frames":      self.video_frames,
                "video_out":         self.video_out,
                "video_long":        self.video_long,
                "long_video":        self.long_video,
                "last_task_text":    self.last_task_text,
                "last_task_id":      self.last_task_id,
                "menu":              self.menu,
                "pending_photo":     self.pending_photo,
                "pending_style_target": self.pending_style_target,
                "pending_outfit_target": self.pending_outfit_target,
                "awaiting_animate_photo": self.awaiting_animate_photo,
                "pending_animate_target": self.pending_animate_target,
                "awaiting_style_photo":   self.awaiting_style_photo,
                "wtw_loc":           self.wtw_loc,
            }


class _Store:
    def __init__(self, path: Path):
        self._path = path; self._lock = threading.Lock(); self._data: dict = {}
        self._load()

    def _load(self):
        try:
            if self._path.exists():
                with open(self._path, encoding="utf-8") as fh:
                    self._data = json.load(fh)
        except Exception as exc:
            # Move the damaged file aside: the next put() rewrites the store
            # from the (now empty) memory copy, which erased every user's
            # session, facts and language for good.
            keep = self._path.with_name(f"{self._path.name}.unreadable-{int(time.time())}")
            try:
                self._path.replace(keep)
            except OSError:
                keep = self._path
            logger.error("session store unreadable (%s) -- starting empty, kept as %s", exc, keep)
            self._data = {}
        if not isinstance(self._data, dict):
            logger.error("session store holds %s, not a dict -- starting empty",
                         type(self._data).__name__)
            self._data = {}

    def _save(self):
        try:
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, ensure_ascii=False)
            tmp.replace(self._path)
        except Exception as exc:
            logger.warning("session store save error: %s", exc)

    def get(self, chat_id: int) -> _Session:
        with self._lock: return _Session(chat_id, self._data.get(str(chat_id)))

    def put(self, sess: _Session):
        with self._lock:
            self._data[str(sess.chat_id)] = sess.to_dict()
            self._save()

    def delete(self, chat_id: int):
        with self._lock:
            self._data.pop(str(chat_id), None)
            self._save()
