"""Per-chat document library, status and report delivery.

Split out of tg_bot.py. Owns the chat-scoped document library (path, open,
stats, indexing an uploaded file, listing and clearing it) and the read-only
user surfaces built from it: /status, the pinned-facts page, the language
menu and report-file delivery.

_LIBRARY_DIR is read as tg_bot._LIBRARY_DIR because redirect_data_dir()
rebinds it; the rest of the shared helpers are read the same way so they keep
one definition and one patch point.
"""
from __future__ import annotations

import html as _html_mod
import re
import shutil
import tempfile
import time
from pathlib import Path

import requests

import library


class LibraryMixin:
    def _library_path(self, chat_id: int) -> Path:
        tg_bot._LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
        return tg_bot._LIBRARY_DIR / f"lib_{chat_id}.db"

    def _open_library(self, chat_id: int):
        """Open this user's library. One index per chat — a shared one would let
        any user retrieve passages from another user's private documents.

        SQLite connections belong to the thread that created them, so callers open
        and close per operation rather than caching an instance on the bot.

        Goes through the knowledge_client boundary; the default backend is
        in-process (library.Library), so the per-chat index file is unchanged.
        """
        import knowledge_client
        return knowledge_client.open_library(self._library_path(chat_id))

    def _library_stats(self, chat_id: int) -> dict:
        path = self._library_path(chat_id)
        if not path.exists():
            return {"documents": 0, "chunks": 0, "docs": []}
        lib = None
        try:
            lib = self._open_library(chat_id)
            docs = lib.documents()
            return {"documents": len(docs),
                    "chunks": sum(d.get("chunks", 0) for d in docs),
                    "docs": docs}
        except Exception as exc:
            tg_bot.logger.warning("library stats failed chat=%s: %s", chat_id, exc)
            return {"documents": 0, "chunks": 0, "docs": []}
        finally:
            if lib is not None:
                try: lib.close()
                except Exception: pass

    def _index_document(self, chat_id: int, sess, path: str, fname: str,
                        sandboxed: bool = False, quiet: bool = False) -> None:
        """Ingest one uploaded file into this user's hybrid index.

        `sandboxed` says the raw bytes already landed somewhere useful. The
        library's opinion of the extension is then not news the user needs:
        see the unsupported branch below.

        `quiet` is for a file small enough to ALSO be inlined and answered
        directly this same turn (see tg_resolve.py). Live, 2026-09-19: forwarding
        a short .md note with "read this and summarize" got "📚 Индексирую... —
        это займёт минуту" ahead of the actual answer, and the success reply
        silently switched sess.menu to "library" and swapped in the Documents
        keyboard — the user never asked to open the library, they asked a
        question that was already being answered inline. Indexing (for 📄 My
        documents) still runs; only the chat-visible announcement and the
        menu/keyboard side effects are skipped.
        """
        lang = self._lang(sess)
        # A file whose extension the indexer can never handle (a forwarded
        # video/GIF/animation sent as a Telegram "document", an image, an
        # archive, ...) used to sail straight into "Indexing... this may take
        # a minute" and then fail with a bare "RuntimeError" -- a real error
        # class name that tells the user nothing about WHY, for a case that
        # was never going to work. Reject it up front with a message that
        # actually explains what's supported, and skip the pointless attempt.
        ext = Path(fname).suffix.lower()
        if ext not in library.SUPPORTED_EXTS:
            # ...unless the file was accepted somewhere else. Measured live: a
            # .jar the assistant had just ASKED FOR was answered with "not a
            # document I can add to your library ... send it along with your
            # question" -- while sitting in the sandbox, sent with a question.
            # Every clause was wrong for that user, because this branch only
            # knew about one of the file's two destinations.
            if not sandboxed and not quiet:
                self._send_text(chat_id,
                    tg_bot._t("lib_unsupported", lang, name=_html_mod.escape(fname),
                       exts=", ".join(sorted(library.SUPPORTED_EXTS))),
                    parse_mode="HTML")
            return
        if not quiet:
            self._send_text(chat_id,
                            tg_bot._t("lib_indexing", lang, name=_html_mod.escape(fname)),
                            parse_mode="HTML")
            self._api_post("sendChatAction", {"chat_id": chat_id, "action": "typing"})
        # Copy out of the temp file under its real name: the extractor dispatches
        # on the suffix, and more importantly the stored document title is what the
        # user sees in 📄 My documents — "tmpz7k1.pdf" would be useless.
        staged = ""
        lib = None
        try:
            # Stage under a SANITIZED basename: `< > ? | " *` are legal in a
            # Telegram filename (Android/iOS/Linux senders) but Windows rejects
            # them in a path, so `report<v2>.txt` etc. could never be staged at
            # all. The real display name is untouched — it is stored separately
            # as `title=` by lib.build/add_document and is what 📄 My documents
            # and the confirmation message show.
            safe_name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", Path(fname).name) or "file"
            staged = str(Path(tempfile.mkdtemp(prefix="tgdoc_")) / safe_name)
            shutil.copyfile(path, staged)
            lib = self._open_library(chat_id)
            # A stable per-chat+filename id, not the staged temp path: every
            # upload lands in a FRESH mkdtemp that is rmtree'd right after, so
            # keying dedup on that abspath (the old default) meant a re-upload
            # never matched its own previous copy and just duplicated it.
            sid = f"tg:{chat_id}:{fname}"
            before = {d.get("title"): d.get("chunks", 0) for d in lib.documents()}
            stats = lib.ingest([staged], source_ids={staged: sid},
                               titles={staged: fname})
            docs = lib.documents()
            chunks = sum(d.get("chunks", 0) for d in docs)
            # Per-file delta, not the whole library's total — "two.txt indexed
            # (15 passages)" for a 3-word note (the library's 15th) read like the
            # file itself had exploded into 15 chunks.
            this_file_chunks = next((d.get("chunks", 0) for d in docs
                                     if d.get("title") == fname), None)
            if this_file_chunks is None:
                this_file_chunks = chunks - sum(before.values())
            errors = stats.get("errors") or []
            if errors and not stats.get("documents_indexed"):
                raise RuntimeError(errors[0])
            sess.use_docs = True
            if not quiet:
                # The menu is NOT switched to 📚 Documents: an index from an upload
                # left the chat «in Documents» unseen, and every later file was
                # indexed instead of read (owner 10-03: «он сам открыл вкладку»).
                # A stale prefix from an earlier menu (e.g. "generate an image of: "
                # armed by 🎨 Draw, then abandoned for an upload) otherwise survives
                # this transition and glues itself onto the next ordinary question,
                # like every other menu-changing branch already clears it here.
                sess.pending_prefix = ""
            self._store.put(sess)
            if not quiet:
                self._send_text(chat_id,
                    tg_bot._t("lib_indexed", lang, name=_html_mod.escape(fname), chunks=this_file_chunks),
                    parse_mode="HTML", keyboard=(tg_bot._library_kb(True, lang)
                                                   if sess.menu == "library"
                                                   else self._state_kb(sess, lang)))
            self._activity.log(chat_id, "system",
                               f"[library] indexed {fname}: {stats}")
        except Exception as exc:
            # Never echo the exception text verbatim: a Windows path-rejection
            # error carries the staged temp path, which carries the OS username
            # (C:\Users\<name>\AppData\Local\Temp\tgdoc_...) — a raw filesystem
            # detail that has no business reaching a Telegram user. Log it, don't
            # display it.
            tg_bot.logger.exception("library index failed chat=%s file=%s", chat_id, fname)
            # In quiet mode the file is also being answered inline this turn --
            # a background indexing failure for 📄 My documents shouldn't derail
            # that answer with an unrelated error the user didn't ask about.
            if not quiet:
                self._send_text(chat_id,
                    tg_bot._t("lib_failed", lang, name=_html_mod.escape(fname),
                       err=_html_mod.escape(type(exc).__name__)),
                    parse_mode="HTML")
        finally:
            if lib is not None:
                try: lib.close()
                except Exception: pass
            if staged:
                try: shutil.rmtree(Path(staged).parent, ignore_errors=True)
                except Exception: pass

    def _send_library_list(self, chat_id: int, sess) -> None:
        lang = self._lang(sess)
        st = self._library_stats(chat_id)
        if not st["documents"]:
            self._send_text(chat_id, tg_bot._t("lib_empty", lang),
                            keyboard=tg_bot._library_kb(sess.use_docs, lang))
            return
        lines = [tg_bot._t("lib_header", lang, n=st["documents"], chunks=st["chunks"])]
        shown = st["docs"][:40]
        for d in shown:
            lines.append(f"• <b>{_html_mod.escape(str(d.get('title') or '?'))}</b>"
                         f" — {d.get('chunks', 0)}")
        # The header claims the FULL count, so a silent 40-item cap with no note
        # made the list look complete when it wasn't (a message that just ends
        # mid-line, no "showing N of M", no way to see the rest).
        if len(st["docs"]) > len(shown):
            lines.append(tg_bot._t("lib_truncated", lang, shown=len(shown), n=len(st["docs"])))
        # Route through _split_html rather than the blind 4096-char slice
        # _send_text falls back to: that slice is tag-blind and can land inside
        # a <b>, and either way a long list was simply cut off with no warning.
        chunks = tg_bot._split_html("\n".join(lines))
        for i, chunk in enumerate(chunks):
            self._send_text(chat_id, chunk, parse_mode="HTML",
                            keyboard=(tg_bot._library_kb(sess.use_docs, lang)
                                      if i == len(chunks) - 1 else None))

    def _clear_library(self, chat_id: int) -> int:
        lib = None
        try:
            if not self._library_path(chat_id).exists():
                return 0
            lib = self._open_library(chat_id)
            n = len(lib.documents())
            lib.purge_all()
            return n
        except Exception as exc:
            tg_bot.logger.warning("library purge failed chat=%s: %s", chat_id, exc)
            return 0
        finally:
            if lib is not None:
                try: lib.close()
                except Exception: pass

    def _send_facts(self, chat_id: int, sess) -> None:
        """Show what the assistant has pinned about this user, with a way out.

        remember_fact writes facts that are injected into EVERY turn, and until now
        there was no way to see them or take one back.
        """
        lang = self._lang(sess)
        facts = sess.get_tg_facts()
        if not facts:
            self._send_text(chat_id, tg_bot._t("facts_none", lang),
                            keyboard=tg_bot._settings_kb(sess.reply_mode, sess.is_admin, lang))
            return
        lines = [tg_bot._t("facts_header", lang)]
        for i, f in enumerate(facts[-40:], 1):
            text = str((f or {}).get("text", "")).strip() if isinstance(f, dict) else str(f)
            if text:
                lines.append(f"{i}. {_html_mod.escape(text[:300])}")
        # Up to 40 facts x ~300 chars can reach ~12k chars — well past the blind
        # 4096-char slice _send_text used to apply, which silently swallowed the
        # back half of the list with no "showing N of M" and no way to see the
        # rest. Route through the same tag-safe splitter used elsewhere for this
        # exact problem, and only attach the Forget-all button to the last chunk.
        chunks = tg_bot._split_html("\n".join(lines))
        kb = {"inline_keyboard": [[
                {"text": tg_bot._t("facts_forget", lang), "callback_data": "facts_clear"}]]}
        for i, chunk in enumerate(chunks):
            self._send_text(chat_id, chunk, parse_mode="HTML",
                            keyboard=kb if i == len(chunks) - 1 else None)

    def _send_status(self, chat_id: int, sess) -> None:
        """Health probe. A downed ComfyUI or LM Studio used to surface only as a
        generic failure in the middle of a request."""
        lang = self._lang(sess)
        import config as _cfg_mod

        def _probe(url: str, timeout: float = 3.0) -> str:
            try:
                r = requests.get(url, timeout=timeout)
                return "🟢" if r.status_code < 500 else f"🟡 HTTP {r.status_code}"
            except Exception:
                return "🔴"

        llm   = _probe(f"{getattr(_cfg_mod, 'LM_STUDIO_BASE', '')}/v1/models")
        try:
            import comfy_client as _cc
            if llm == "🟢" and _cc.card_is_exclusive():
                # the server answers but the model is lent to a render: a chat
                # message now waits minutes, and 🟢 said the opposite
                llm = "🟡 " + tg_bot._t("status_llm_busy", lang)
        except Exception:
            pass
        comfy = _probe(f"{getattr(_cfg_mod, 'COMFY_URL', '')}/system_stats")
        up    = time.time() - (self._start_time or time.time())
        with self._task_lock:
            running = sum(len(v) for v in self._running_task.values())
        used  = self._user_store.usage_today(chat_id)
        lines = [
            tg_bot._t("status_title", lang),
            tg_bot._t("status_llm", lang, v=llm),
            tg_bot._t("status_comfy", lang, v=comfy),
            tg_bot._t("status_queue", lang, waiting=self._backend.depth(), running=running),
            (tg_bot._t("status_uptime_min", lang, m=int(max(up, 0) // 60)) if up < 3600 else
             tg_bot._t("status_uptime", lang, h=f"{up / 3600:.1f}")),
        ] + ([tg_bot._t("status_backend", lang,
                        name=_html_mod.escape(self._backend.name()))]
             # operator detail, same rule as /settings
             if getattr(sess, "is_admin", False) else []) + [
            "",
            tg_bot._t("status_usage", lang),
        ]
        for k in (tg_bot.KIND_TASK, tg_bot.KIND_IMAGE, tg_bot.KIND_RESEARCH):
            limit = tg_bot._quota_limit(k)
            cap = f"/{limit}" if limit > 0 else ""
            lines.append(f"• {tg_bot._kind_label(k, lang)}: <b>{used.get(k, 0)}{cap}</b>")
        st = self._library_stats(chat_id)
        lines.append(tg_bot._t("status_docs", lang, n=st["documents"],
                        state=tg_bot._t("on" if sess.use_docs else "off", lang)))
        self._send_text(chat_id, "\n".join(lines), parse_mode="HTML",
                        keyboard=tg_bot._settings_kb(sess.reply_mode, sess.is_admin, lang))

    def _send_report_file(self, chat_id: int, topic: str, report: str,
                          header: str, lang: str) -> bool:
        """Deliver a long report as a Markdown attachment.

        Returns True only when the attachment actually reached the user. The
        caller uses that to decide whether the report ALSO has to be pasted into
        the chat: once the file arrives, repeating 38k characters as a dozen
        bubbles just buries the conversation — but if the upload failed, the text
        is the only copy the user gets and must still be sent.
        """
        tmpdir = ""
        try:
            safe = re.sub(r"[^\w\- ]+", "", topic).strip()[:60] or "report"
            tmpdir = tempfile.mkdtemp(prefix="tgreport_")
            path = Path(tmpdir) / f"{safe}.md"
            path.write_text(f"# {topic}\n\n{report}\n", encoding="utf-8")
            caption = tg_bot._t("dr_file_cap", lang, n=len(report))
            if self._send_document(chat_id, str(path), caption):
                return True
            tg_bot.logger.warning("report file delivery failed chat=%s", chat_id)
        except Exception:
            tg_bot.logger.exception("report file build failed chat=%s", chat_id)
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
        return False

    def _send_lang_menu(self, chat_id: int, lang: str) -> None:
        self._send_text(chat_id, tg_bot._t("lang_choose", lang),
                        keyboard={"inline_keyboard": [
                            [{"text": "🇬🇧 English", "callback_data": "lang:en"}],
                            [{"text": "🇷🇺 Русский", "callback_data": "lang:ru"}]]})


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)
