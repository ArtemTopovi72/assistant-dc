"""Background workers: every long call the main window must not block on.

Nine QThread subclasses, one per slow operation — loading the models, switching
one, a redraw, a hand fix, an artifact fix, a library build, deep research, an
agent turn, and a folder scan. They were scattered through gui.py between the
widgets they serve; collected here they read as what they are, a single list of
everything that takes long enough to need a thread.

AssistantWindow is the only caller of all nine. It STAYS in gui.py, which is
why this move needed no test changes: roughly twenty suites patch
gui.ModelLoader and friends, and because the CALLER did not move it still
resolves those names as gui.py globals, so the fakes intercept exactly as
before. (Moving a callee alone is safe; it is moving the caller that kills a
monkeypatch — learned at 16b2521, and the reason the workers were left until
the tabs that used to own some of them were gone.)

Heavy imports stay inside run(): a worker module imported at startup should not
drag in the model stack, and each run() is a separate slow path anyway.
"""
import copy

from PyQt5.QtCore import Qt, QThread, pyqtSignal

from audio import transcribe_audio_array

import logging
logger = logging.getLogger("assistant.gui")


# --------------------------------------------------------------------------- #
# Workers
# --------------------------------------------------------------------------- #
class ModelLoader(QThread):
    ready = pyqtSignal(object, object, object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, model_name, no_think, reasoning_effort="high"):
        super().__init__()
        self.model_name, self.no_think = model_name, no_think
        self.reasoning_effort = reasoning_effort

    def run(self):
        try:
            from app_runtime import build_runtime   # not from assistant: that is the entry point
            ctx, base_state, graph = build_runtime(
                self.model_name, self.no_think, status_cb=self.progress.emit,
                reasoning_effort=self.reasoning_effort)
            ctx.gui_mode = True
            self.ready.emit(ctx, base_state, graph)
        except Exception as exc:
            logger.exception("Model load failed")
            self.failed.emit(str(exc))


class ModelSwitchWorker(QThread):
    """Unload the current LM Studio model and load a newly chosen one off the UI thread."""
    done = pyqtSignal(bool, str)

    def __init__(self, base_url, model_id):
        super().__init__()
        self.base_url, self.model_id = base_url, model_id

    def run(self):
        try:
            from lmstudio import ensure_exclusive
            ok, msg = ensure_exclusive(self.base_url, self.model_id)
            self.done.emit(ok, msg)
        except Exception as exc:
            logger.exception("Model switch failed")
            self.done.emit(False, str(exc))


class RedrawWorker(QThread):
    """Run a redraw / enhance / region edit on the last image off the UI thread."""
    done = pyqtSignal(str)      # new image path ("" on failure)
    failed = pyqtSignal(str)

    def __init__(self, ctx, source_path, prompt, mode, region="", engine="firered"):
        super().__init__()
        self.ctx, self.source_path, self.prompt, self.mode = ctx, source_path, prompt, mode
        self.region = region
        self.engine = engine or "firered"

    def run(self):
        try:
            import image as _image
            if self.mode == "inpaint":
                from image import inpaint_region_with_comfy
                # Region instruction edit via FireRed.
                path = inpaint_region_with_comfy(
                    self.ctx, self.source_path, self.region, self.prompt, engine=self.engine)
                wf = f"contained-{self.engine} (GUI Redraw btn, region given)"
            else:
                # "redraw" and "enhance" (empty instruction) both re-render the
                # whole picture the same way the redraw_image tool does: our own
                # Ideogram picture from its boxes, a user's photo through FireRed.
                # Both used to run the old model graphs whose checkpoint was removed.
                from tool_image_handlers import redraw_whole_image
                instructions = self.prompt or (
                    "improve the overall quality: sharpen fine detail, clean up noise "
                    "and artifacts")
                path, wf = redraw_whole_image(self.ctx, self.source_path, instructions)
                path = _image.assert_deliverable(path, where="gui.RedrawWorker",
                                                 source_path=self.source_path)
            try:
                _image.log_edit_decision(
                    request=f"region={self.region!r} instructions={self.prompt!r}",
                    classifier=f"GUI 'Redraw last' button mode={self.mode}",
                    tool="RedrawWorker", workflow=wf,
                    returned_file=path, source=self.source_path)
            except Exception:
                pass
            self.done.emit(path or "")
        except Exception as exc:
            logger.exception("Redraw failed")
            self.failed.emit(str(exc))


class FixHandsWorker(QThread):
    """Repair deformed hands/fingers off the UI thread via image.fix_hands.
    A drawn mask (mask_override) makes it work even on hands no auto-detector
    can localize; without one it auto-detects (MeshGraphormer + SAM/Florence)."""
    done = pyqtSignal(str)      # new image path ("" on failure)
    failed = pyqtSignal(str)

    def __init__(self, ctx, source_path, mask_path=None, engine="firered"):
        super().__init__()
        self.ctx, self.source_path, self.mask_path = ctx, source_path, mask_path
        self.engine = engine or "firered"

    def run(self):
        try:
            import image as _image
            path = _image.fix_hands(self.ctx, self.source_path,
                                    mask_override=self.mask_path, engine=self.engine)
            try:
                _image.log_edit_decision(
                    request="fix hands/fingers (GUI button)",
                    classifier="hand_repair" + (":manual-mask" if self.mask_path else ":auto"),
                    tool="FixHandsWorker",
                    workflow=f"MeshGraphormer/SAM/Florence + {self.engine}",
                    returned_file=path, source=self.source_path)
            except Exception:
                pass
            self.done.emit(path or "")
        except Exception as exc:
            logger.exception("Fix-hands failed")
            self.failed.emit(str(exc))


class FixArtifactWorker(QThread):
    """Generic 'select-and-fix' off the UI thread: the user paints over ANY awkward
    region (a harsh transition, a smear, a stray object, a melted edge) and describes
    what's wrong; the contained FireRed engine regenerates ONLY that masked region
    and feather-composites it back, so every pixel outside the selection is untouched."""
    done = pyqtSignal(str)      # new image path ("" on failure)
    failed = pyqtSignal(str)

    def __init__(self, ctx, source_path, mask_path, instruction,
                 protect_face=True, engine="firered"):
        super().__init__()
        self.ctx, self.source_path, self.mask_path = ctx, source_path, mask_path
        self.instruction = instruction
        self.protect_face = protect_face
        self.engine = engine or "firered"

    def run(self):
        try:
            import image as _image
            path = _image.edit_region_contained_via_firered(
                self.ctx, self.source_path, "selected area", self.instruction,
                mask_override=self.mask_path, protect_face=self.protect_face,
                engine=self.engine)
            try:
                _image.log_edit_decision(
                    request=f"fix artifact (GUI): {self.instruction[:80]}",
                    classifier="artifact_repair:manual-mask",
                    tool="FixArtifactWorker",
                    workflow=f"contained masked inpaint + {self.engine}",
                    returned_file=path, source=self.source_path)
            except Exception:
                pass
            self.done.emit(path or "")
        except Exception as exc:
            logger.exception("Fix-artifact failed")
            self.failed.emit(str(exc))


class LibraryBuildWorker(QThread):
    """Build/extend the document Library off the UI thread (extract → chunk → embed).

    Creates its OWN Library (SQLite connection) on this thread — connections can't
    cross threads. The GUI keeps a separate reader connection on the same file."""
    progress = pyqtSignal(str, int, int)   # stage ('extract'|'embed'), done, total
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, paths, db_path=None):
        super().__init__()
        self.paths = list(paths)
        self.db_path = db_path
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            import knowledge_client
            lib = knowledge_client.open_library(self.db_path)
            try:
                stats = lib.ingest(
                    self.paths,
                    progress=lambda s, d, t: self.progress.emit(s, d, t),
                    cancel=lambda: self._cancel)
            finally:
                lib.close()
            self.done.emit(stats)
        except Exception as exc:
            logger.exception("Library build failed")
            self.failed.emit(str(exc))


class DeepResearchWorker(QThread):
    """Run the Ultra Search / deep-research pipeline off the UI thread."""
    progress = pyqtSignal(str, dict, str)   # phase, stats, message
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, ctx, topic, depth="standard", overrides=None):
        super().__init__()
        self.ctx, self.topic, self.depth = ctx, topic, depth
        self.overrides = overrides or {}

    def run(self):
        # Goes through the research service boundary (research_api.py /
        # research_client.py), not through deep_research directly: the client
        # owns apply/run/restore as one indivisible operation, and the manual
        # overrides can no longer leak into a later run. out_lang is left None
        # so the client derives it from the typed topic (the Research tab has no
        # separate language control, and a Russian request was coming back as an
        # English paper) — see research_api.OUT_LANG_FROM_TOPIC_MARKER.
        try:
            import research_client
            result = research_client.open_research().run(
                self.ctx, self.topic, depth=self.depth, overrides=self.overrides,
                progress=lambda phase, stats, msg: self.progress.emit(phase, stats, msg),
            )
            self.done.emit(result)
        except Exception as exc:
            # A bad knob surfaces here as a failed-signal (the UI un-busies and
            # shows the error), never as an uncaught thread death that leaves
            # the UI stuck "busy".
            logger.exception("Deep research failed")
            self.failed.emit(str(exc))


class RequestWorker(QThread):
    recognized = pyqtSignal(str)
    info = pyqtSignal(str)        # transient status line (e.g. RAG retrieval note)
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, ctx, graph, base_state, *, text=None, audio=None, image=None,
                 use_db=False, db_k=None):
        super().__init__()
        self.ctx, self.graph, self.base_state = ctx, graph, base_state
        self.text, self.audio, self.image = text, audio, image
        self.use_db = use_db
        self.db_k = db_k

    def run(self):
        import turn_trace
        _trace = turn_trace.start(chat="gui", text=(self.text or "<voice>")[:1500])
        _outcome = "crashed"
        # Every error path ends in failed.emit; DirectConnection runs this in the
        # worker thread, so the outcome is known before finish().
        _failed = []
        self.failed.connect(_failed.append, Qt.DirectConnection)
        try:
            self._run()
            _outcome = "failed: " + _failed[0][:200] if _failed else "done"
        finally:
            turn_trace.finish(_trace, outcome=_outcome)

    def _run(self):
        try:
            text = self.text
            if text is None and self.audio is not None:
                self.ctx.set_stage("Transcribing")
                text = transcribe_audio_array(self.ctx, self.audio)
                if not text:
                    self.failed.emit("Didn't catch that — try again.")
                    return
                self.recognized.emit(text)
            if not text:
                self.failed.emit("Empty input.")
                return
            # RAG: retrieve from the document library and wrap the question. Runs here
            # (worker thread) so the embed/search never blocks the UI, and so typed AND
            # voice queries are augmented uniformly (voice is known only after transcribe).
            # The model sees the augmented prompt; memory/display keep the original text.
            send_text = text
            if self.use_db:
                try:
                    import knowledge_client
                    self.ctx.set_stage("Searching documents")
                    lib = knowledge_client.open_library()   # own connection on this thread
                    try:
                        # Cross-lingual: if the corpus is in a different script than the
                        # question (e.g. Russian book, English question), translate the
                        # query into the corpus language(s) and retrieve over all variants.
                        extra = []
                        try:
                            targets = knowledge_client.cross_lingual_targets(lib, text)
                            for lang in targets[:2]:
                                tq = self._translate_query(text, lang)
                                if tq:
                                    extra.append(tq)
                            if extra:
                                self.info.emit("🌐 Also searching in: " + ", ".join(targets[:2]))
                        except Exception:
                            logger.exception("cross-lingual query expansion failed")
                        send_text, note = knowledge_client.build_rag_prompt(
                            lib, text, k=self.db_k, extra_queries=extra)
                    finally:
                        lib.close()
                    if note:
                        self.info.emit(note)
                except Exception:
                    logger.exception("RAG augmentation failed; answering without retrieval")
                    send_text = text
            self.ctx.remember("user", text, {"source": "gui"})
            state = copy.deepcopy(self.base_state)
            state["user_input"] = send_text
            state["image_data"] = self.image
            state["session_memory_text"] = self.ctx.memory_text()
            # Database mode is a CLOSED-BOOK turn over the user's own documents: the model
            # must answer from the retrieved passages, not the web. Withhold the web tools
            # (search/find_photo) for this turn only, then restore the user's setting — so a
            # "Use Database" question can't be silently answered by a web search.
            _prev_web = getattr(self.ctx, "web_search_enabled", True)
            if self.use_db:
                self.ctx.web_search_enabled = False
            try:
                final = self.graph.invoke(state)
            finally:
                self.ctx.web_search_enabled = _prev_web
            msgs = copy.deepcopy(final.get("messages", self.base_state["messages"]))
            # Every few turns, fold older history into a running summary so the next
            # turn doesn't resend (or reprocess) the whole transcript. Off the UI
            # thread already (this is the worker), so the extra LLM call is fine.
            from graph import compact_history_if_needed
            self.base_state["messages"] = compact_history_if_needed(self.ctx, msgs)
            self.done.emit(dict(final))
        except Exception as exc:
            logger.exception("Request failed")
            self.failed.emit(str(exc))

    def _translate_query(self, text, lang):
        """Translate a search query into `lang` for cross-lingual retrieval. Returns the
        translation or None. Uses the loaded chat model; never raises into the turn."""
        try:
            import llm
            import re as _re
            msgs = [{"role": "system",
                     "content": f"Translate the user's search query into {lang}. "
                                "Reply with ONLY the translation — no quotes, no notes."},
                    {"role": "user", "content": text}]
            r = llm.send_to_lm_studio(self.ctx, msgs, tools=[], tool_choice="none",
                                      temperature=0.0, max_tokens=120, prefill="<think></think>")
            out = ((r or {}).get("content") or "").strip()
            out = _re.sub(r"<think>.*?</think>", "", out, flags=_re.S).strip()
            return out or None
        except Exception:
            logger.exception("query translation failed")
            return None


class ScanWorker(QThread):
    """Whole-book scan (map-reduce) for exhaustive 'find all X' questions that top-k
    retrieval can't answer. Reads EVERY indexed chunk in batches (map: extract relevant
    facts) then synthesizes one answer (reduce). Slow but complete. Own SQLite connection."""
    progress = pyqtSignal(int, int)
    info = pyqtSignal(str)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, ctx, question, paths=None):
        super().__init__()
        self.ctx, self.question, self.paths = ctx, question, paths
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _llm(self, system, user, *, max_tokens, temperature):
        import llm
        import re as _re
        r = llm.send_to_lm_studio(
            self.ctx, [{"role": "system", "content": system},
                       {"role": "user", "content": user}],
            tools=[], tool_choice="none", temperature=temperature,
            max_tokens=max_tokens, prefill="<think></think>")
        out = ((r or {}).get("content") or "")
        return _re.sub(r"<think>.*?</think>", "", out, flags=_re.S).strip()

    def run(self):
        try:
            import knowledge_client
            lib = knowledge_client.open_library()
            try:
                chunks = lib.all_chunks(self.paths)
            finally:
                lib.close()
            if not chunks:
                self.failed.emit("The database is empty — add documents and Build first.")
                return

            def mapf(batch, q):
                if self._cancel:
                    return ""
                note = self._llm(
                    "Extract every fact in the TEXT that is relevant to the QUESTION, as short "
                    "bullet points, using ONLY the TEXT. Quote names/specifics exactly. If "
                    "nothing in the TEXT is relevant, reply with exactly: NONE",
                    f"QUESTION: {q}\n\nTEXT:\n{batch}", max_tokens=500, temperature=0.0)
                return "" if (not note or note.strip().upper().startswith("NONE")) else note

            def reducef(notes, q):
                if not notes:
                    return "I scanned the document(s) but found nothing relevant to that question."
                joined = "\n".join(notes)
                ans = self._llm(
                    "You are given NOTES extracted from a book that are relevant to the QUESTION. "
                    "Write a complete, de-duplicated answer using ONLY the notes. If something is "
                    "not covered by the notes, say it is not specified. Do not invent specifics.",
                    f"QUESTION: {q}\n\nNOTES:\n{joined[:24000]}", max_tokens=1400, temperature=0.2)
                return ans or joined

            self.info.emit(f"🔍 Scanning {len(chunks)} passages across the document(s)…")
            answer = knowledge_client.map_reduce_scan(
                chunks, self.question, mapf, reducef,
                progress=lambda i, n: self.progress.emit(i, n),
                cancel=lambda: self._cancel)
            if self._cancel:
                self.failed.emit("Scan cancelled.")
                return
            self.done.emit(answer)
        except Exception as exc:
            logger.exception("Whole-book scan failed")
            self.failed.emit(str(exc))
