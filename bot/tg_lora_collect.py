"""Telegram: an admin drops reference photos straight into a LoRA dataset.

Built for exactly one workflow: the operator's brother is curating pictures
for a Fallout-STYLE adapter (not a person's likeness) -- Frank Horrigan, NCR
soldiers, the Master, whatever, all sharing the same pre-rendered CGI look.
Style training wants variety of subject under one constant visual style, so
unlike the identity pipeline (characters.py, prep_lora_dataset.py) there is no
stripping of content out of captions here -- the curator's own caption IS the
training caption, verbatim.

Everything lands in the SAME runtime/lora_datasets/<slug>/ shape the desktop
Персонажи tab already understands, through the SAME characters.py registry --
this is not a parallel dataset mechanism, just a second way to feed the one
that already exists.

Gated to admins only: this writes files to disk on every photo a chat sends
while armed, which is a foot-gun in front of an ordinary user who fat-fingers
the button.
"""
import html as _html_mod
import time

import characters as _chars
import lora_training as _lt

STYLE_SLUG = "fallout_style"
STYLE_NAME = "Fallout style"


class LoraCollectMixin:

    def _cb_start_lora_collect(self, chat_id: int) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        if not sess.is_admin:
            self._send_text(chat_id, tg_bot._t("admin_only", lang))
            return
        rec = _chars.get(STYLE_SLUG)
        if not rec:
            out_dir = _lt.DATASET_ROOT / STYLE_SLUG
            _chars.upsert(STYLE_SLUG, name=STYLE_NAME, trigger=STYLE_SLUG,
                          status="dataset", dataset_dir=str(out_dir))
        sess.reg_state = "lora_collect"
        self._store.put(sess)
        n = _lt.dataset_size(_chars.get(STYLE_SLUG)["dataset_dir"])
        self._send_text(chat_id, tg_bot._t("lora_collect_armed", lang, n=n),
                        parse_mode="HTML")

    def _lora_collect_photo(self, chat_id: int, sess, lang: str,
                            file_id: str, caption: str) -> None:
        self._lora_collect_save(chat_id, sess, lang, [file_id], caption)

    def _lora_collect_album(self, chat_id: int, sess, lang: str,
                            file_ids: list, caption: str) -> None:
        # Telegram puts the ONE caption on only one message of the album; by
        # the time it reaches here _flush_album has already found it and
        # attached it to every file in the group. A shared caption across a
        # multi-angle turnaround of the SAME scene is normal for a style
        # dataset -- it is not the "49 identical captions" problem, which was
        # about one caption covering an entire unrelated 49-image set.
        self._lora_collect_save(chat_id, sess, lang, file_ids, caption)

    def _lora_collect_save(self, chat_id: int, sess, lang: str,
                           file_ids: list, caption: str) -> None:
        rec = _chars.get(STYLE_SLUG)
        if not rec:
            # The arming step creates this; reaching here without it means
            # reg_state survived a restart with no matching registry entry.
            sess.reg_state = ""
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("lora_collect_gone", lang))
            return
        out_dir = _lt.DATASET_ROOT / rec["slug"]
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for fid in file_ids:
            data = self._dl_bytes(fid)
            if not data:
                continue
            stem = "%s_%d_%d" % (rec["slug"], int(time.time() * 1000), saved)
            (out_dir / (stem + ".jpg")).write_bytes(data)
            # The curator's caption IS the training caption -- no stripping,
            # unlike the identity pipeline. An uncaptioned drop still needs a
            # caption FILE (the trainer expects one beside every image), so it
            # gets the bare trigger word rather than being silently skipped.
            (out_dir / (stem + ".txt")).write_text(
                caption if caption else rec.get("trigger") or rec["slug"],
                encoding="utf-8")
            saved += 1
        if not saved:
            self._send_text(chat_id, tg_bot._t("lora_collect_failed", lang))
            return
        total = _lt.dataset_size(out_dir)
        self._send_text(chat_id, tg_bot._t(
            "lora_collect_saved", lang, n=saved, total=total,
            caption=_html_mod.escape(caption) if caption else
            tg_bot._t("lora_collect_no_caption", lang)), parse_mode="HTML")


# Imported at the BOTTOM and read as tg_bot.<name> at call time, never by
# value -- same cycle convention as every other tg_* mixin (see tg_characters.py).
import tg_bot  # noqa: E402
