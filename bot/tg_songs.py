"""Two audio flows the 🎨 Creativity menu offers: write a song, build a mashup.

Lifted out of tg_accounts.py. Both are menu-first and multi-turn -- the user
presses the button, then sends what the flow asked for -- and both do the
actual work on a background thread so the poll loop keeps serving other chats.

Songs arm `song_topic` and hand the topic to music.py. Unlike the weather
flow's `wtw_city`, the state is cleared once the attempt FINISHES either way:
a song topic has no "didn't resolve, try again" case worth staying armed for.

Mashups collect two tracks before anything can run, so they keep their own
`mashup_state` rather than a reg_state: the vocal, then the backing.

Paths are read through ``tg_bot`` at call time (_MASHUP_DIR in particular is
rebound by redirect_data_dir(), which a by-value import would never see).
"""
from __future__ import annotations

import html as _html_mod
import os
import re as _re
import threading
import turn_trace

import mashup as _mashup_mod
import music as _music_mod


# A song asked for in plain words. Live, 2026-09-12: "сочини короткую весёлую
# песню про кота" reached the chat model, which WROTE four lines of lyrics and
# read them out as a voice note. The song engine was a menu tap away and the
# user had no way to know that. A request that names a song and a making verb
# is the song flow, whichever way it arrives.


def _read(text: str, after_song: bool = False) -> dict:
    """The model's read of a chat message (agent/intent.py). It replaced six
    word lists (song verbs x nouns, lyrics-only, seconds, «на него», redo,
    rewrite) -- «как научиться петь» and «расскажи про гимн» each had their
    own exclusion."""
    import intent
    return intent.read(None, text or "", "The assistant just sent a song." if after_song else "")


# The song runs as a TASK, not on a bare thread: a thread had no status
# line, no ⛔ Stop, no queue position, and the chat never looked busy, so a
# second message ran the agent on top of a two-minute render. The task's
# user_text is this machine payload; _run_task_inner recognises it.
_SONG_TASK_RE = _re.compile(r"^\[song(?::(\d+))?\]\s*(.*)$", _re.DOTALL)


# "а теперь сочини на него песню" right after the bot wrote lyrics: the
# words are THOSE words, not a new song about the sentence. Live,
# 2026-09-12: four lines about autumn were followed by a song about city
# lights. The topic then carries the lyrics behind this marker.
LYRICS_MARK = "[lyrics]"


def refers_to_previous_text(text: str) -> bool:
    return bool((text or "").strip()) and _read(text)["song_uses_previous_text"]


def looks_like_lyrics(text: str) -> bool:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return 3 <= len(lines) <= 60 and max(len(ln) for ln in lines) <= 90


_SECTION_LABEL_RE = _re.compile(r"^\s*\[?\s*(?:intro|verse|pre-?chorus|chorus|bridge|outro|hook)\b[^\]:]*\]?\s*:?\s*$", _re.I)


def tag_lyrics(text: str) -> str:
    """Section tags Music3 recognises around plain lines: verses of four,
    the last stanza repeated as the chorus."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    # Already laid out («Куплет 1:», «Припев:», «[Chorus]», «Verse 2:»): keep
    # it; music.py maps the names to tags. Re-chunking by four put an extra
    # [verse] over every label and sang the bridge as the chorus.
    if any(_music_mod._RU_TAG_RE.match(ln) or _SECTION_LABEL_RE.match(ln) for ln in lines):
        return "\n".join(lines)
    # The author's blank lines are the stanzas; four-line chunks only for a wall of text.
    stanzas = [[ln.strip() for ln in b.splitlines() if ln.strip()]
               for b in _re.split(r"\n\s*\n", (text or "").strip())]
    if len(stanzas) < 2:
        stanzas = [lines[i:i + 4] for i in range(0, len(lines), 4)]
    if any(stanzas.count(st) > 1 for st in stanzas):
        # The author already repeats the refrain: tag it, add nothing.
        return _repeats_are_chorus("\n".join(
            ["[intro]"] + [ln for st in stanzas for ln in ["[verse]"] + st] + ["[outro]"]))
    out = ["[intro]"]
    for i, st in enumerate(stanzas):
        out.append("[verse]" if i < len(stanzas) - 1 or len(stanzas) == 1 else "[chorus]")
        out.extend(st)
    if len(stanzas) >= 2:
        out.append("[chorus]"); out.extend(stanzas[-1])
    out.append("[outro]")
    return "\n".join(out)


def song_title(lyrics: str) -> str:
    """The first chorus line (what a song is called by), else the first sung line."""
    lines = [ln.strip() for ln in (lyrics or "").splitlines() if ln.strip()]
    sung = lambda xs: [ln for ln in xs if not ln.startswith("[")]
    for i, ln in enumerate(lines):
        if ln.lower().startswith("[chorus"):
            rest = sung(lines[i + 1:i + 2])
            if rest:
                return rest[0].strip(" ,.!?;:—-")
    first = sung(lines)
    return first[0].strip(" ,.!?;:—-") if first else ""


_STRUCTURE_SYS = (
    "You lay out a song's lyrics for a music engine. The user's words are FINAL: copy "
    "every sung line exactly, same order, same spelling, same punctuation -- never "
    "rewrite, add, drop or merge a line. Only insert section tags on their own lines: "
    "[intro] [verse] [pre-chorus] [chorus] [bridge] [outro]. A repeated block is the "
    "chorus. Reply with the tagged lyrics only.")


def _sung(text: str) -> list:
    """The lines that will be sung, normalised for comparison."""
    out = []
    for ln in (text or "").splitlines():
        t = ln.strip()
        if not t or _re.match(r"^\[[^\]]*\]$", t) or _music_mod.is_stage_direction(t):
            continue
        out.append(_re.sub(r"\W+", " ", t).strip().lower())
    return out


def author_notes(text: str) -> str:
    """The author's stage directions («(Интро: тяжёлый рифф…)») -- not sung,
    but they describe the sound, so they go to the style."""
    notes = [ln.strip(" ()") for ln in (text or "").splitlines()
             if ln.strip() and _music_mod.is_stage_direction(ln.strip())]
    return "; ".join(notes)[:400]


def _repeats_are_chorus(tagged: str) -> str:
    """A block sung twice word for word is the chorus: the model tagged the
    refrain «Эй, эй, не грусти» [pre-chorus] both times."""
    parts = _re.split(r"^(\[[^\]\n]+\])[ \t]*$", tagged, flags=_re.M)
    bodies = [parts[k + 1].strip() for k in range(1, len(parts) - 1, 2)]
    for k in range(1, len(parts) - 1, 2):
        if parts[k] != "[chorus]" and parts[k + 1].strip() and bodies.count(parts[k + 1].strip()) > 1:
            parts[k] = "[chorus]"
    return "".join(parts)


def structure_lyrics(ctx, given: str) -> str:
    """Tag the user's own words. Laid-out text keeps its layout; plain text goes
    to the model, whose answer is taken ONLY if every sung line survived verbatim
    (the user: «а ллм не сможет как-то учесть?»). Otherwise the four-line chunks."""
    if ctx is None or any(_music_mod._RU_TAG_RE.match(l.strip()) or _SECTION_LABEL_RE.match(l.strip())
                          for l in given.splitlines() if l.strip()):
        return tag_lyrics(given)
    try:
        import llm
        out = llm.call_llm_simple(ctx, _STRUCTURE_SYS, given, temperature=0.1,
                                  max_tokens=min(4000, 200 + len(given)))
    except Exception:
        tg_bot.logger.warning("[songs] lyric layout call failed", exc_info=True)
        out = None
    if out and _sung(out) == _sung(given):
        return _repeats_are_chorus(out.strip())
    if out:
        tg_bot.logger.info("[songs] model changed the words while tagging -- chunking instead")
    return tag_lyrics(given)


def song_payload(topic: str, duration: int = 0) -> str:
    return f"[song:{int(duration)}] {topic.strip()}" if duration else f"[song] {topic.strip()}"


def parse_song_payload(text: str) -> tuple:
    """(topic, duration) for a "[song:N] topic" task text, else ("", 0)."""
    m = _SONG_TASK_RE.match((text or "").strip())
    if not m:
        return "", 0
    return m.group(2).strip(), int(m.group(1) or 0)


# The lyrics of the song this chat got last; only the NEXT message may redo it.
# ponytail: in memory, a restart forgets it; move to the session if that matters.
LAST_SONG: dict = {}


def song_followup(text: str) -> bool:
    """«а теперь то же самое в стиле рок» right after a song: redo it, same words."""
    t = (text or "").strip()
    return bool(t) and _read(t, after_song=True)["song"] in ("sound", "words")


def song_lang(topic: str, lang: str) -> str:
    """«колыбельная на английском» from a Russian chat was written in Russian."""
    import utils
    return utils.explicit_lang(topic) or lang


def followup_topic(text: str, last_lyrics: str) -> str:
    """The same words sung again for a new sound; rewritten for a change to the words or length."""
    if _read(text, after_song=True)["song"] == "words":
        return text + "\nRewrite these lyrics as asked, keep everything else:\n" + last_lyrics
    return text + "\n" + LYRICS_MARK + last_lyrics


def song_request(text: str) -> tuple:
    """(is a song request, seconds asked for or 0) for a plain chat message."""
    t = (text or "").strip()
    if t.startswith("[song"):
        return False, 0          # already a task payload
    if not t or len(t) > 600:
        return False, 0
    r = _read(t)
    return (True, r["song_seconds"]) if r["song"] == "make" else (False, 0)


def request_seconds(t: str, after_song: bool = False) -> int:
    """The length asked for, in seconds; 0 when none (the model's read)."""
    return _read(t, after_song)["song_seconds"] if (t or "").strip() else 0


def _polished(ctx, lyrics: str, has_stage: bool = True) -> str:
    """The lyric after the lyrics_craft check/revise loop; the original when the
    loop fails or loses the section tags the music engine sings by."""
    try:
        import lyrics_craft
        if has_stage:
            ctx.set_stage("Polishing the lyrics")
        out = (lyrics_craft.polish(ctx, lyrics) or {}).get("text") or ""
    except Exception:
        tg_bot.logger.warning("[songs] polishing the lyrics failed; singing the draft", exc_info=True)
        return lyrics
    tags = lambda t: len(_re.findall(r"^\s*\[[^\]]+\]\s*$", t, _re.M))
    if not out.strip() or (tags(lyrics) and not tags(out)):
        return lyrics
    return out


def snap_duration(secs: int) -> int:
    """The nearest length the engine offers, or 0 when nothing was asked."""
    if not secs:
        return 0
    lo, hi = _music_mod.DURATION_RANGE
    return max(lo, min(int(secs), hi))


class SongsMixin:
    def _start_song_flow(self, chat_id: int, sess, lang: str) -> None:
        sess.reg_state = "song_topic"
        sess.song_draft = ""           # an unanswered «как есть / новый» must not block the next one
        self._store.put(sess)
        # The active settings ride along with the ask: the genre/voice/length
        # the song will come back in is decided BEFORE the topic is typed, and
        # a user who cannot see that here only finds out a few minutes later
        # when the wrong-sounding song arrives.
        self._send_text(chat_id,
                        tg_bot._t("song_topic_prompt", lang)
                        + "\n\n" + _html_mod.escape(tg_bot._music_summary(sess, lang)),
                        parse_mode="HTML")

    def _cb_song_lyrics(self, chat_id: int, data: str) -> None:
        """«🎤 Спеть как есть» / «✍️ Написать новый текст» for ready words."""
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        draft, sess.song_draft = getattr(sess, "song_draft", ""), ""
        if sess.reg_state == "song_topic":
            sess.reg_state = ""
        self._store.put(sess)
        if not draft:
            self._send_text(chat_id, tg_bot._t("song_topic_prompt", lang))
            return
        if data.endswith(":polish"):
            self._run_cancellable(chat_id, tg_bot._t("lyr_working_improve", lang),
                                  self._song_polish_then_sing, chat_id, lang, draft, lang=lang)
            return
        topic = (LYRICS_MARK + "\n" + draft) if data.endswith(":keep") else draft
        self._start_song_generation(chat_id, topic, lang)

    def _start_song_generation(self, chat_id: int, topic: str, lang: str,
                               duration: int = 0) -> None:
        """Runs the actual generation off-thread, same reasoning as
        _start_weather_lookup: build_structured_caption/generate_music can
        call the LLM and a slow local model, and this fires from the poll
        loop thread that every other chat's updates are waiting behind.

        `duration` is a length the user named in the request itself; it wins
        over the 🎛 setting for this one song.

        Queued as a task (see song_payload) so it gets the status line, the
        ⛔ Stop button, the queue position and the busy accounting every other
        long job has."""
        self._enqueue_item(chat_id, {"type": "text",
                                     "text": song_payload(topic, duration)})

    def _generate_song(self, chat_id: int, topic: str, lang: str,
                       duration: int = 0, ctx=None) -> bool:
        """Write and render one song, deliver it. Returns True when a file went
        out. `ctx` is the task's scoped context: its stage callback carries
        the status line and its cancel event is honoured between the writing
        and the render, and before delivery."""
        if ctx is None:
            try:
                ctx = self._get_ctx()
            except Exception:
                ctx = None
        cancelled = lambda: bool(ctx is not None and getattr(ctx, "is_cancelled", None)
                                 and ctx.is_cancelled())
        _has_stage = ctx is not None and hasattr(ctx, "set_stage")
        try:
            # Ask FIRST, so "isn't set up yet" is only ever said when that is
            # actually true. generate_music also raises MusicUnavailable for a
            # render that produced no file, which is a transient failure on a
            # perfectly well-configured engine -- reporting that as "not set
            # up" sent the user off checking an installation that was fine.
            # Readiness is checked for the preset this chat actually chose:
            # "max" can be a 28GB download away while "fast" is ready, and
            # checking the default would promise a song we cannot render.
            _sess0 = self._get_session(chat_id)
            preset = tg_bot._music_preset(_sess0)
            # None (not resolve_steps()'s display default) so an untouched
            # session actually gets the turbo LoRA build_workflow gates on
            # `steps is None` -- see resolve_steps_for_generate's docstring.
            steps = _tg_music.resolve_steps_for_generate(_sess0)
            ok_engine, why = _music_mod.engine_available(ctx, preset)
            if not ok_engine:
                tg_bot.logger.warning("[songs] engine unavailable for chat %s: %s",
                                      chat_id, why)
                self._send_text(chat_id, tg_bot._t("song_unavailable", lang))
                return

            # Whatever the user picked in 🎛 Song settings. Read HERE rather
            # than captured when the thread was spawned, so a setting changed
            # while a topic was being typed still applies to this song.
            sess = self._get_session(chat_id)
            duration = duration or tg_bot._music_duration(sess)
            prefs = tg_bot._music_prefs(sess)
            # Whatever is on Auto, the bot picks itself BEFORE writing -- one
            # short call -- so the choice is explicit in the brief and can be
            # reported to the user when the song arrives.
            auto = _tg_music.auto_fields(sess)
            if not duration:
                auto["duration"] = True
            chosen = {}
            if any(auto.values()):
                if _has_stage:
                    ctx.set_stage("Choosing the sound")
                chosen = _music_mod.choose_auto_params(
                    ctx, topic, lang, genre=auto["genre"], tempo=auto["tempo"],
                    vocal=auto["vocal"], duration=auto["duration"],
                    fixed=dict(prefs, **({"seconds": duration} if duration else {})))
                if chosen:
                    picked = _music_mod.prefs_from(
                        genre=chosen.get("genre", ""),
                        tempo=f"bpm:{chosen['bpm']}" if chosen.get("bpm") else "",
                        vocal=chosen.get("vocal", ""))
                    prefs = dict(picked, **prefs)      # the user's own choices win
                    if chosen.get("seconds"):
                        duration = int(chosen["seconds"])
            if not duration:
                duration = _config.MUSIC_DEFAULT_SECONDS
            # The render length is an upper bound -- Music3 stops when the
            # words run out -- so the songwriter is told the same duration the
            # render will ask for, or it writes two verses for a 240s slot.
            if _has_stage:
                ctx.set_stage("Writing the lyrics")
            given = ""
            if LYRICS_MARK in topic:
                wish, _, given = topic.partition(LYRICS_MARK)
                given = given.strip()
                topic = wish.strip() + "\na song whose words are: " + " / ".join(
                    ln.strip() for ln in given.splitlines() if ln.strip())[:300]
                _notes = author_notes(given)
                if _notes:                  # the author's own sound directions
                    topic += ". The author's notes on the sound: " + _notes
            import steer as _steer
            notes = _steer.take(ctx)
            caption = _music_mod.build_structured_caption(
                ctx, _steer.with_notes(topic, notes), song_lang(topic, lang), duration_s=duration, prefs=prefs)
            # Last call before the render: a wish that came in while the
            # lyrics were being written rewrites them, the render is minutes.
            late = _steer.take(ctx)
            if late and not cancelled():
                notes += late
                tg_bot.logger.info("[songs] rewriting for %d late wish(es): %r", len(late), late[0][:80])
                caption = _music_mod.build_structured_caption(
                    ctx, _steer.with_notes(topic, notes), song_lang(topic, lang), duration_s=duration, prefs=prefs)
            if given:                   # their own words stay; wishes steer the sound
                caption = dict(caption, lyrics=structure_lyrics(ctx, given))
            elif caption.get("lyrics") and not cancelled():
                # Words the bot wrote itself go through the checks and revisions
                # of ✨ Improve lyrics before they are sung (owner 10-03: «под
                # капотом генерится хороший текст, причёсывается автоматом»).
                caption = dict(caption, lyrics=_polished(ctx, caption["lyrics"], _has_stage))
            if cancelled():
                return False
            wav_path = _music_mod.generate_music(
                ctx, caption.get("lyrics", ""), caption.get("style", ""),
                duration_s=duration, preset=preset, steps=steps)
            if cancelled():
                return False
            t = song_title(caption.get("lyrics", ""))
            if t:                                   # the file is named after the song
                import nice_names
                nice_names.TITLES[chat_id] = t
            ok = self._send_audio(chat_id, wav_path)
            if ok:
                LAST_SONG[chat_id] = caption.get("lyrics", "")
            if not ok:
                tg_bot.logger.error("[songs] delivery failed for chat %s (%s)",
                                    chat_id, wav_path)
                self._send_text(chat_id, tg_bot._t("song_error", lang))
            elif chosen:
                line = _tg_music.chose_line(chosen, lang)
                if line:
                    self._send_text(chat_id, line, parse_mode="HTML")
            return bool(ok)
        except _music_mod.SongwritingFailed:
            if cancelled():
                return False        # Stop interrupted the writer; not a failure
            # The engine is fine; the writer model fumbled. Worth retrying, and
            # worth saying so -- this used to be reported as "not set up yet".
            tg_bot.logger.warning("[songs] songwriting failed for chat %s",
                                  chat_id, exc_info=True)
            try:
                self._send_text(chat_id, tg_bot._t("song_lyrics_failed", lang))
            except Exception:
                pass
        except _music_mod.MusicUnavailable:
            if cancelled():
                return False
            # Reached only if the engine went away BETWEEN the check above and
            # the render, or the render itself produced nothing.
            tg_bot.logger.warning("[songs] render unavailable for chat %s",
                                  chat_id, exc_info=True)
            try:
                self._send_text(chat_id, tg_bot._t("song_error", lang))
            except Exception:
                pass
        except Exception:
            if cancelled():
                return False
            tg_bot.logger.exception("[songs] generation failed for chat %s", chat_id)
            try:
                self._send_text(chat_id, tg_bot._t("song_error", lang))
            except Exception:
                pass
        finally:
            try:
                sess = self._get_session(chat_id)
                if sess.reg_state == "song_topic":
                    sess.reg_state = ""
                    self._store.put(sess)
            except Exception:
                pass
        return False

    # ── mashup: the voice of one track over the music of another ─────────────
    def _start_mashup_flow(self, chat_id: int, sess, lang: str) -> None:
        """Arm the two-track capture and ask for the first one.

        Menu-first by design: the user presses 🎚 Мэшап, then sends the tracks
        one at a time and is told which is which. The alternative -- watching
        for any two audio messages in a row -- would have to guess which of
        them the voice comes from, and that is the one thing a mashup cannot
        guess for you.
        """
        if not _mashup_mod.engine_available():
            self._send_text(chat_id, tg_bot._t("mash_off", lang),
                            keyboard=self._main_menu_kb(sess, lang))
            return
        sess.mashup_state = "want1"
        sess.mashup_vocal_path = ""
        sess.mashup_vocal_speech = False
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("mash_ask1", lang), parse_mode="HTML")

    def _mashup_dir(self, chat_id: int) -> str:
        """Per-chat scratch space for the two tracks.

        Read off tg_bot._MASHUP_DIR at call time rather than captured at import:
        redirect_data_dir() rebinds it, and a by-value copy would leave test
        runs writing tracks into the live tree -- the same mistake that once
        put test chat ids into the production user database.
        """
        d = os.path.join(str(tg_bot._MASHUP_DIR), str(chat_id))
        os.makedirs(d, exist_ok=True)
        return d

    def _mashup_take_audio(self, chat_id: int, sess, lang: str, file_id: str,
                           is_voice: bool, suffix: str = ".ogg") -> bool:
        """Store one of the two tracks. Returns True if it was consumed here.

        Downloads on the poll thread deliberately: a Telegram audio file is a
        few megabytes over a local-ish HTTP hop, and the alternative (spawning
        a thread per track) would let the SECOND track arrive and start a
        mashup while the FIRST was still being written to disk.
        """
        data = self._dl_bytes(file_id)
        if not data:
            self._send_text(chat_id, tg_bot._t("mash_need_audio", lang))
            return True
        slot = "1" if sess.mashup_state == "want1" else "2"
        path = os.path.join(self._mashup_dir(chat_id), "track%s%s" % (slot, suffix))
        try:
            with open(path, "wb") as fh:
                fh.write(data)
        except Exception:
            tg_bot.logger.exception("[mashup] could not save track for chat %s", chat_id)
            self._send_text(chat_id, tg_bot._t("mash_need_audio", lang))
            return True

        if sess.mashup_state == "want1":
            sess.mashup_vocal_path = path
            sess.mashup_vocal_speech = bool(is_voice)
            sess.mashup_state = "want2"
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("mash_got1", lang))
            self._send_text(chat_id, tg_bot._t("mash_ask2", lang), parse_mode="HTML")
            return True

        vocal_path = sess.mashup_vocal_path
        is_speech = bool(sess.mashup_vocal_speech)
        # Disarmed BEFORE the slow work starts: leaving it armed meant the next
        # audio message the user sent while waiting was swallowed as "track 2"
        # of a mashup that was already running.
        sess.mashup_state = ""
        sess.mashup_vocal_path = ""
        sess.mashup_vocal_speech = False
        self._store.put(sess)
        if not vocal_path or not os.path.exists(vocal_path):
            self._send_text(chat_id, tg_bot._t("mash_failed", lang,
                                               why=tg_bot._t("mash_need_audio", lang)))
            return True
        self._send_text(chat_id, tg_bot._t("mash_work", lang))
        turn_trace.spawn(self._run_mashup, chat_id, lang, vocal_path, path, is_speech,
                         name=f"mashup-{chat_id}")
        return True

    def _run_mashup(self, chat_id: int, lang: str, vocal_path: str,
                    instr_path: str, is_speech: bool) -> None:
        """Separate, match, mix and deliver -- off the poll thread.

        Two Demucs passes plus a time-stretch is tens of seconds of GPU work,
        and this is reached from the loop every other chat's updates queue
        behind, exactly like _start_weather_lookup and _generate_song.
        """
        sess = self._get_session(chat_id)
        try:
            out = _mashup_mod.default_out_path(str(chat_id))
            rep = _mashup_mod.make_mashup(
                vocal_path, instr_path, out, vocal_is_speech=is_speech,
                progress=lambda st: self._activity.log(
                    chat_id, "stage", "Mashup: %s" % st))
            caption = tg_bot._t(
                "mash_done", lang, secs=rep["seconds"],
                kv=rep["key_vocal"] or "?", bv=rep["bpm_vocal"] or "?",
                ki=rep["key_instr"] or "?", bi=rep["bpm_instr"] or "?",
                stretch=rep["stretch"], semis=rep["semitones"], took=rep["took"])
            # A pair that cannot work is delivered anyway -- the user asked for
            # it -- but with the reason attached. Silently handing back four
            # minutes of two songs fighting each other, with no explanation,
            # is what made the feature look broken rather than misused.
            if rep.get("warnings"):
                caption += "\n\n" + tg_bot._t("mash_mismatch", lang,
                                              why="; ".join(rep["warnings"]))
            if not self._send_audio(chat_id, out, caption):
                tg_bot.logger.error("[mashup] delivery failed for chat %s (%s)",
                                    chat_id, out)
                self._send_text(chat_id, tg_bot._t("mash_failed", lang, why="delivery"))
        except _mashup_mod.MashupUnavailable as exc:
            # Carries a sentence written for the user -- see mashup.py's module
            # docstring -- so it is passed through rather than replaced.
            tg_bot.logger.warning("[mashup] refused for chat %s: %s", chat_id, exc)
            try:
                self._send_text(chat_id, tg_bot._t("mash_failed", lang, why=str(exc)))
            except Exception:
                pass
        except Exception as exc:
            tg_bot.logger.exception("[mashup] failed for chat %s", chat_id)
            try:
                self._send_text(chat_id, tg_bot._t("mash_failed", lang, why=str(exc)))
            except Exception:
                pass
        finally:
            # The separator holds GPU memory the next image or song render
            # needs; this machine has 24GB for everything at once.
            try:
                _mashup_mod.release_separator()
            except Exception:
                pass
            try:
                self._send_text(chat_id, tg_bot._t("main_menu", lang),
                                keyboard=self._main_menu_kb(sess, lang))
            except Exception:
                pass


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)
import tg_music as _tg_music  # noqa: E402
import config as _config  # noqa: E402
