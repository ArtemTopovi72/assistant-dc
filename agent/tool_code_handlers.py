"""The coding agent's tools: files, archives, code, dependencies.

They all read `ctx.sandbox` — the caller decides WHOSE sandbox this turn gets,
so a handler can never pick the wrong user's files. A turn with no sandbox
attached gets a tool error explaining why, which is also what a user without the
grant sees: these schemas are not even sent to the model in that case, so the
error is a belt to the retrieval step's braces.

Every SandboxError becomes a [TOOL ERROR] with the sandbox's own message. Those
messages are written to be read BY the model — "that text appears 3 times, add
more surrounding lines" is a correction it can act on, where "invalid argument"
is not.
"""
from __future__ import annotations

import logging
from pathlib import Path
import re

from code_sandbox import SandboxError
import code_runner as _runner
import sandbox_access as _access

logger = logging.getLogger("assistant.tools.code")

_NO_SANDBOX = (
    "[TOOL ERROR] No sandbox is attached to this conversation, so there are no "
    "files to work with. Tell the user their account does not have sandbox "
    "access yet and that an administrator can grant it."
)


def _box(ctx):
    box = getattr(ctx, "sandbox", None)
    if box is None:
        raise _NoSandbox()
    return box


class _NoSandbox(Exception):
    pass


def _guard(fn):
    """One place where every sandbox failure becomes a model-readable error."""
    def wrapper(ctx, state, args: dict) -> str:
        try:
            return fn(ctx, state, args)
        except _NoSandbox:
            return _NO_SANDBOX
        except SandboxError as exc:
            return f"[TOOL ERROR] {exc}"
        except Exception as exc:                      # never leak a traceback
            logger.exception("sandbox tool %s failed", fn.__name__)
            return f"[TOOL ERROR] {type(exc).__name__}: {exc}"
    wrapper.__name__ = fn.__name__
    wrapper.__wrapped__ = fn
    return wrapper


# -- files -------------------------------------------------------------------

@_guard
def _handle_list_files(ctx, state, args: dict) -> str:
    entries = _box(ctx).list_dir(args.get("path") or ".", int(args.get("depth") or 3))
    if not entries:
        return "The folder is empty."
    # A real mod's 3-level tree runs to hundreds of lines -- a third of the
    # 20k context. Cap it; the tail tells the model how to narrow down.
    if len(entries) > _LIST_SHOWN:
        entries = entries[:_LIST_SHOWN] + [
            f"... {len(entries) - _LIST_SHOWN} more entries not shown -- list a "
            f"subfolder, lower depth, or search_files for a name"]
    return "\n".join(entries)


_LIST_SHOWN = 150


IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"})


@_guard
def _handle_open_image(ctx, state, args: dict) -> str:
    """Make a picture IN THE SANDBOX the one the vision tools look at.

    The requirement was "any input file -- code, an archive, a photo". Photos
    were the hole: read_file refuses a PNG (correctly -- reading it produces
    noise), and inspect_image only ever looked at state["image_path"], which a
    file dropped in the working folder never reaches. So a user could send a
    screenshot, see it listed, and be told it could not be read.

    This does not duplicate the vision path, it POINTS it at the file: the
    existing inspect_image, with all its retries and its prompt, does the
    looking on the next round.
    """
    box = _box(ctx)
    rel = (args.get("path") or "").strip()
    full = box.resolve(rel)
    if not full.exists():
        raise SandboxError(f"'{rel}' does not exist.{box._nearby(rel)}")
    if full.is_dir():
        raise SandboxError(f"'{rel}' is a directory, not a picture.")
    if full.suffix.lower() not in IMAGE_SUFFIXES:
        raise SandboxError(
            f"'{rel}' is not an image ({', '.join(sorted(IMAGE_SUFFIXES))}). "
            f"Use read_file for text.")
    state["image_path"] = str(full)
    try:
        ctx.last_image_path = str(full)
    except Exception:
        pass
    return (f"Opened '{rel}' ({full.stat().st_size} bytes). It is now the "
            f"current picture — call inspect_image to look at it.")


_WINDOW_READ_BYTES = 20_000_000
# SWE-agent ablation: a 100-line viewer beat 30 lines and the whole file.
# 250 lines of a 20k-token window was a quarter of the context per read.
READ_WINDOW = 120


def _read_for_view(box, rel: str) -> str:
    """read_text, except a big text file is still readable in windows: the
    window, not the whole file, is what goes into the model's context."""
    try:
        return box.read_text(rel)
    except SandboxError as exc:
        full = box.resolve(rel)
        if ("limit for one read" in str(exc) and box.is_texty(full)
                and full.stat().st_size <= _WINDOW_READ_BYTES):
            return full.read_text(encoding="utf-8", errors="replace")
        raise


@_guard
def _handle_read_file(ctx, state, args: dict) -> str:
    """SWE-agent's file viewer: a window of lines, not the whole file. With a
    20k context one 200 KB file read whole was the entire budget -- the model
    either got cut off or lost everything it had read before."""
    box = _box(ctx)
    text = _read_for_view(box, args.get("path") or "")
    if not text.strip():
        return f"'{args.get('path')}' is empty."
    lines = text.splitlines()
    total = len(lines)
    start = max(1, int(args.get("start") or 1))
    size = int(args.get("lines") or READ_WINDOW)
    if start > total:
        return (f"{args.get('path')} has only {total} lines; start={start} is past "
                f"the end.")
    end = min(total, start + size - 1)
    body = "\n".join(f"{i:>5}| {lines[i - 1]}" for i in range(start, end + 1))
    where = (f"lines {start}-{end} of {total}" if (start > 1 or end < total)
             else f"{total} lines")
    more = (f"\n[... {total - end} more lines. Call read_file with start={end + 1} "
            f"for the next window, or search_files / code_outline to jump to the "
            f"part you need.]" if end < total else "")
    return (f"{args.get('path')} — {where}, shown with line numbers; "
            f"the 'N| ' prefix is added here and is NOT part of the file. Code "
            f"reading this file will see the original lines.\n{body}{more}"
            + _tag_note(str(args.get("path") or "")))


# Bench deliver_symptom 2026-09-23, 3 of 3 no-think runs: to make Thief see
# MoreVillagers' workstations the model wrote "#c:villager_job_sites" INTO
# morevillagers' own tag -- the reverse of what it does, so the delivered jar
# protected nothing. The direction of a tag reference is the whole task and
# the model gets it backwards, so it is stated where the file is read.
_TAG_PATH_RE = re.compile(r"(^|/)data/([^/]+)/tags/.+\.json$", re.I)


def _tag_note(path: str) -> str:
    m = _TAG_PATH_RE.search(path.replace("\\", "/"))
    if not m:
        return ""
    return ("\n[NOTE] Minecraft tag file: it defines the tag "
            f"'{m.group(2)}:<path after tags/<type>/>' and its \"values\" are the "
            "MEMBERS of THAT tag. An entry '#ns:other' pulls other's blocks INTO "
            "this tag -- it never puts this tag's blocks into ns:other. To make "
            "blocks count for a tag another mod checks, list them in THAT tag: "
            "edit its file, or create data/<ns>/tags/<type>/<path>.json with "
            "\"replace\": false.")


def _syntax_note(box, rel: str) -> str:
    """A Python file that no longer parses, said at write time -- SWE-agent's
    lint-on-edit. Not a refusal: a file written in parts is legitimately
    incomplete between calls, so the model is told, not blocked."""
    if not rel.lower().endswith(".py"):
        return ""
    try:
        compile(box.read_text(rel), rel, "exec")
    except SyntaxError as exc:
        return (f"\n[WARNING: {rel} does not parse now -- line {exc.lineno}: "
                f"{exc.msg}. Fix it with edit_file unless you are still writing "
                f"the rest of the file.]")
    except Exception:
        pass
    return ""


def _edit_context(box, rel: str, new: str, around: int = 3) -> str:
    """The changed lines as the file now has them, numbered like read_file.

    mini-swe-agent / SWE-agent show the result of every edit: a model that
    only reads "Edited x.py" goes on believing its edit looks the way it
    intended, and the next edit_file quotes text that is not there."""
    try:
        text = box.read_text(rel)
    except Exception:
        return ""
    at = text.find(new) if new else -1
    if at < 0:
        return ""
    lines = text.splitlines()
    first = text.count("\n", 0, at)
    last = first + max(0, new.count("\n"))
    lo, hi = max(0, first - around), min(len(lines), last + around + 1)
    body = "\n".join(f"{i + 1:>5}| {lines[i]}" for i in range(lo, hi))
    return f"\nThe file around the change now reads:\n{body}"


_NOTHING_CHANGED = (
    "Nothing changed: {path} already has exactly this content -- this change "
    "is already applied. Do NOT repeat it. Go to the next step: if the file is "
    "inside an unpacked archive, call pack_archive on that folder now.")


_CODE_EXT = (".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".cs", ".go", ".rs",
             ".rb", ".php", ".sh", ".lua", ".kt", ".swift", ".html", ".css")
# A file of code this long is repaired in place, never rewritten: sandbox bench
# 2026-09-24, the regex engine was rewritten whole seven times in one turn -- four
# of the rewrites cut off at the token ceiling, every one re-risking the parts
# that already worked. One typo in a 1000-line file must cost one edit_file.
REWRITE_MAX_LINES = 40
_REWRITE_REFUSED_TAIL = (
    "is not rewritten whole. Fix it in place: read_file the part you need "
    "(start=..., lines=...), then edit_file with the exact old text and its "
    "replacement (or start_line/end_line) -- one edit per fix.")
_OPEN_W_RE = re.compile(
    r"""open\(\s*(?:r|f|rf|fr)?['"]([^'"]+)['"]\s*,\s*(?:mode\s*=\s*)?['"]w""")
_WRITE_TEXT_RE = re.compile(
    r"""Path\(\s*(?:r|f)?['"]([^'"]+)['"]\s*\)\s*\.write_(?:text|bytes)\(""")


def _rewrites_code_file(box, code: str) -> str:
    """The existing long code file a run_code script overwrites, or ""."""
    for rx in (_OPEN_W_RE, _WRITE_TEXT_RE):
        for m in rx.finditer(code or ""):
            rel = m.group(1).strip()
            if rel.startswith("/work/"):          # the sandbox's own mount point
                rel = rel[len("/work/"):]
            rel = rel.lstrip("./")
            if not rel.lower().endswith(_CODE_EXT):
                continue
            try:
                p = box.resolve(rel)
                if p.is_file() and len(p.read_bytes().splitlines()) > REWRITE_MAX_LINES:
                    return rel
            except Exception:
                continue
    return ""


@_guard
def _handle_write_file(ctx, state, args: dict) -> str:
    box = _box(ctx)
    path = args.get("path") or ""
    content = args.get("content") or ""
    prior = _before(box, path)
    n_new = len(content.splitlines())
    if args.get("append"):
        if prior is None:
            raise SandboxError(f"{path} does not exist yet -- create it with write_file "
                               f"(no append) first.")
        old = prior.decode("utf-8", "replace")
        sep = "" if (not old or old.endswith("\n")) else "\n"
        written = box.write_text(path, old + sep + content)
        _snapshot(box, written, prior)
        total = len((old + sep + content).splitlines())
        return (f"Appended {n_new} lines to {written} (now {total} lines)."
                + _syntax_note(box, written))
    if prior is not None and path.lower().endswith(_CODE_EXT):
        n_old = len(prior.splitlines())
        if n_old > REWRITE_MAX_LINES and prior != content.encode("utf-8"):
            raise SandboxError(
                f"NOT written: {path} already exists ({n_old} lines) and is not rewritten "
                f"whole. Fix it in place: read_file the part you need (start=..., "
                f"lines=...), then edit_file with the exact old text and its "
                f"replacement (or start_line/end_line) -- one edit per fix. To add "
                f"code at the end use write_file with append=true.")
    # A complete call is accepted whatever its length: its tokens are already
    # spent, and refusing it only throws them away (sandbox bench 2026-09-24,
    # nine refused 170-700 line writes in a row). The ~150-line advice in the
    # description is about not getting CUT OFF, which a refusal cannot fix.
    # Bench deliver_symptom 2026-09-23: the same write_file five times in a
    # row, each answered "Wrote ...", until the round budget ran out and the
    # fix was never packed. Say plainly that nothing changed.
    if prior is not None and prior == (args.get("content") or "").encode("utf-8"):
        return _NOTHING_CHANGED.format(path=args.get("path"))
    written = box.write_text(args.get("path") or "", args.get("content") or "")
    _snapshot(box, written, prior)
    return f"Wrote {written}." + _syntax_note(box, written)


@_guard
def _handle_edit_file(ctx, state, args: dict) -> str:
    box = _box(ctx)
    new = args.get("new") or ""
    prior = _before(box, args.get("path") or "")
    s_ln, e_ln = int(args.get("start_line") or 0), int(args.get("end_line") or 0)
    if not (args.get("old") or "") and s_ln:
        # SWE-agent's editor: replace lines N-M as numbered by read_file. A
        # small model that cannot copy `old` byte for byte can still name the
        # lines it just read.
        path, how = _replace_lines(box, args.get("path") or "", s_ln, e_ln or s_ln, new)
    elif not (args.get("old") or ""):
        raise SandboxError("Give either `old` (the exact text to replace) or "
                           "start_line/end_line (as numbered by read_file).")
    else:
        path, how = box.replace_flexible(args.get("path") or "", args.get("old") or "", new)
    # SWE-agent's lint gate: an edit that turns a parsing .py file into one
    # that does not is taken back and refused, with the reason. A broken edit
    # left in place is what sends the next ten calls chasing their own
    # SyntaxError. (write_file only warns: files are legitimately written in
    # parts.)
    if prior is not None and path.lower().endswith(".py"):
        broke = _parse_error(box.read_text(path))
        if broke and not _parse_error(prior.decode("utf-8", "replace")):
            box.resolve(path).write_bytes(prior)
            raise SandboxError(
                f"Edit NOT applied: it would break {path} -- {broke}. The file is "
                f"unchanged. Check the indentation and brackets of `new` (it "
                f"replaces `old` exactly, so it must fit where `old` was) and "
                f"try again.")
    if prior is not None and box.resolve(path).read_bytes() == prior:
        return _NOTHING_CHANGED.format(path=path)
    _snapshot(box, path, prior)
    return (f"Edited {path}." + how + _edit_context(box, path, new)
            + _syntax_note(box, path))


def _replace_lines(box, rel: str, start: int, end: int, new: str) -> tuple:
    text = box.read_text(rel)
    lines = text.splitlines(keepends=True)
    if start < 1 or end < start or end > len(lines):
        raise SandboxError(f"Lines {start}-{end} are outside {rel} (it has "
                           f"{len(lines)} lines). read_file it again for the numbers.")
    # read_file shows "   12| code"; a copied prefix must not land in the file
    body = "\n".join(re.sub(r"^\s*\d+\| ", "", ln) for ln in new.split("\n"))
    if body and not body.endswith("\n"):
        body += "\n"
    box.write_text(rel, "".join(lines[:start - 1]) + body + "".join(lines[end:]))
    return box.relative(box.resolve(rel)), f" (lines {start}-{end} replaced)"


def _parse_error(src: str) -> str:
    try:
        compile(src, "<edit>", "exec")
    except SyntaxError as exc:
        return f"line {exc.lineno}: {exc.msg}"
    except Exception:
        return ""
    return ""


def _agent_library_key(ctx) -> str:
    """One knowledge base per sandbox owner -- the same isolation reasoning as
    code_sandbox.sandbox_for: a shared index would let one chat's rag_search
    read another chat's private documents."""
    box = getattr(ctx, "sandbox", None)
    try:
        if box is not None:
            return box.root.name
    except Exception:
        pass
    return "default"


def _open_agent_library(ctx):
    """Opens the SAME per-chat index tg_library.py uses (lib_<key>.db in
    tg_bot._LIBRARY_DIR), so a file added here shows up in the user's own
    📄 My documents and vice versa -- one RAG index, not a second one grown
    inside the sandbox. Read as an attribute (not imported by name) so
    tg_bot's own test redirection (redirect_data_dir) still applies here."""
    import knowledge_client
    try:
        import tg_bot
        lib_dir = tg_bot._LIBRARY_DIR
    except Exception:
        lib_dir = Path(__file__).resolve().parents[1] / "runtime" / "tg_libraries"
    lib_dir.mkdir(parents=True, exist_ok=True)
    return knowledge_client.open_library(str(lib_dir / f"lib_{_agent_library_key(ctx)}.db"))


@_guard
def _handle_rag_add(ctx, state, args: dict) -> str:
    box = _box(ctx)
    rel = (args.get("path") or "").strip()
    full = box.resolve(rel)
    if not full.exists():
        raise SandboxError(f"'{rel}' does not exist.{box._nearby(rel)}")
    if full.is_dir():
        raise SandboxError(f"'{rel}' is a directory, not a file -- name one file in it.")
    import library as _lib
    if full.suffix.lower() not in _lib.SUPPORTED_EXTS:
        raise SandboxError(
            f"'{rel}' cannot be indexed ({full.suffix or 'no extension'}). "
            f"Supported: {', '.join(sorted(_lib.SUPPORTED_EXTS))}.")
    lib = _open_agent_library(ctx)
    try:
        sid = f"agent:{_agent_library_key(ctx)}:{full.name}"
        stats = lib.ingest([str(full)], source_ids={str(full): sid},
                           titles={str(full): full.name})
        errors = stats.get("errors") or []
        if errors and not stats.get("documents_indexed"):
            raise SandboxError(str(errors[0]))
        chunks = next((d.get("chunks", 0) for d in lib.documents()
                       if d.get("title") == full.name), 0)
    finally:
        try: lib.close()
        except Exception: pass
    _record(box, "rag_add", f"indexed {rel} into the knowledge base ({chunks} passages)")
    return (f"Indexed '{rel}' into the knowledge base ({chunks} passages). "
            f"rag_search can now retrieve it.")


@_guard
def _handle_rag_search(ctx, state, args: dict) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        return "[TOOL ERROR] rag_search needs a 'query'."
    lib = _open_agent_library(ctx)
    try:
        if lib.is_empty():
            return ("The knowledge base is empty for this chat -- nothing has "
                    "been indexed yet. Use rag_add on a file already in the "
                    "working folder, or answer from what you already know.")
        hits = lib.retrieve(query, k=int(args.get("k") or 5))
    finally:
        try: lib.close()
        except Exception: pass
    if not hits:
        return f"No stored passages match {query!r}."
    lines = [f"Passages from the knowledge base matching {query!r}:"]
    for h in hits:
        title = getattr(h, "title", None) or "?"
        text = (getattr(h, "text", "") or "").strip().replace("\n", " ")
        lines.append(f"- [{title}] {text[:600]}")
    return "\n".join(lines)


@_guard
def _handle_search_files(ctx, state, args: dict) -> str:
    hits = _box(ctx).search(args.get("pattern") or "", args.get("path") or ".",
                            offset=int(args.get("offset") or 0))
    if not hits:
        return (f"No line matches {args.get('pattern')!r}. The text may be "
                f"spelled differently, or live in a non-text file.")
    return "\n".join(hits)


@_guard
def _handle_find_content(ctx, state, args: dict) -> str:
    """Search the working folder by content: pictures by description, text by words.

    Live 2026-09-14: «найди мост» over 173 photos. The tools on offer could
    look at ONE picture per two calls, so the task did not fit in a turn and
    the model narrated a plan instead of acting -- twice. This is the tool
    that makes the task fit: contact sheets + verification (photo_search),
    grep for text, one call for the whole folder.
    """
    box = _box(ctx)
    query = (args.get("query") or "").strip()
    rel = (args.get("path") or ".").strip() or "."
    kind = (args.get("kind") or "any").strip().lower()
    folder = box.resolve(rel)
    if not folder.exists():
        raise SandboxError(f"'{rel}' does not exist.{box._nearby(rel)}")
    if not folder.is_dir():
        folder = folder.parent
    lines = []
    if kind in ("any", "text"):
        try:
            hits = box.search(re.escape(query), rel)
        except Exception:
            hits = []
        if hits:
            lines.append(f"Text files mentioning {query!r}:")
            lines.extend(hits[:20])
    if kind in ("any", "photos"):
        import photo_search as _ps
        photos = _ps.list_photos(folder)
        if photos:
            _stage = getattr(ctx, "set_stage", None)
            if callable(_stage):
                try: _stage(f"Looking through {len(photos)} pictures")
                except Exception: pass
            res = _ps.find_in_photos(ctx, folder, query,
                                     work_dir=box.root / ".contact_sheets")
            matches = res["matches"]
            if matches:
                first = matches[0]
                state["image_path"] = str(first)
                state["image_status"] = "ok"
                try:
                    ctx.last_image_path = str(first)
                    ctx.found_image_paths = [str(m) for m in matches]
                except Exception:
                    pass
                _remember_found(box, matches)
                _record(box, "find_content", f"found {len(matches)} of {res['scanned']} photos "
                        f"showing {query!r}: " + ", ".join(_rel(box, m) for m in matches))
                lines.append(f"Pictures showing {query!r} ({len(matches)} of "
                             f"{res['scanned']} looked at):")
                lines.extend(str(m.relative_to(box.root)).replace("\\", "/")
                             for m in matches)
                lines.append("The first one is now the current picture and will "
                             "be sent to the user; name the others by path.")
            else:
                lines.append(f"None of the {res['scanned']} pictures shows "
                             f"{query!r} (checked every one).")
    if not lines:
        return (f"Nothing in '{rel}' matches {query!r} -- no text file contains "
                f"it and there are no pictures to look at.")
    return "\n".join(lines)


@_guard
def _handle_delete_path(ctx, state, args: dict) -> str:
    """Live 2026-09-14: «не захламляй: удали архив и папку, коллаж оставь».

    Without this the only way to delete was run_code, which a FILES-level
    account does not have -- so a plain clean-up request was impossible for
    exactly the users the working folder was made for.
    """
    box = _box(ctx)
    what = box.remove(args.get("path") or "")
    # The deleted thing must not stay the "current picture" or the deliverable.
    for key in ("image_path", "document_path"):
        cur = str(state.get(key) or "")
        if cur and not Path(cur).exists():
            state.pop(key, None)
    return f"Deleted {what}."


@_guard
def _handle_dedupe_photos(ctx, state, args: dict) -> str:
    """Journey 33, four runs: told in the description, the schema and the run
    result to call assistant_tools.dedupe, the model wrote its own exact-bytes
    hash every time and reported the copies gone. A tool it can CALL is the
    lever that works (find_content taught the same lesson)."""
    import importlib.util
    import shutil
    box = _box(ctx)
    spec = importlib.util.spec_from_file_location(
        "assistant_tools", Path(__file__).resolve().parents[1] / "docker" / "sandbox" / "assistant_tools.py")
    at = importlib.util.module_from_spec(spec); spec.loader.exec_module(at)
    rels = [str(p).strip() for p in (args.get("paths") or []) if str(p).strip()]
    lines_head = ""
    if rels:
        paths = []
        for r in rels:
            full = box.resolve(r)
            if not full.exists():
                raise SandboxError(f"'{r}' does not exist.{box._nearby(r)}")
            paths.append(full)
    else:
        rel_dir = (args.get("path") or "").strip()
        # ctx is a per-turn copy (tg_bot._scoped_ctx), so the attribute set
        # by find_content last turn is gone; the sandbox file is what lasts.
        found = [Path(p) for p in (getattr(ctx, "found_image_paths", None)
                                   or _recall_found(box))
                 if Path(p).exists()]
        if found and not args.get("whole_folder"):
            # Journey 33: with no `paths` the model de-duplicated the WHOLE
            # archive (173 photos) and then picked eight arbitrary "unique"
            # pictures -- sunsets -- for a collage of bridges. The set it
            # meant is the one find_content just returned; pin on that state.
            paths = found
            lines_head = (f"(the {len(found)} photos find_content found; pass "
                          f"whole_folder=true to compare every picture in a folder)")
        else:
            folder = box.resolve(rel_dir or ".")
            if not folder.is_dir():
                raise SandboxError(f"'{args.get('path')}' is not a folder.")
            paths = at.list_images(folder)
    if not paths:
        return "[TOOL ERROR] No pictures to compare."
    _stage = getattr(ctx, "set_stage", None)
    if callable(_stage):
        try: _stage(f"Comparing {len(paths)} pictures")
        except Exception: pass
    groups = at.group_near_duplicates(paths)
    kept, lines = [], []
    for g in groups:
        if len(g) == 1:
            kept.append(g[0]); continue
        scored = sorted(((at.sharpness(p), p) for p in g), reverse=True)
        best = scored[0][1]; kept.append(best)
        lines.append("group: " + ", ".join(_rel(box, p) for p in g)
                     + f" -> kept {_rel(box, best)} (sharpest, {scored[0][0]:.0f})")
    keep_dir = (args.get("keep_dir") or "").strip()
    if keep_dir:
        dest = box.resolve(keep_dir); dest.mkdir(parents=True, exist_ok=True)
        for p in kept:
            shutil.copy2(p, dest / p.name)
        lines.append(f"copied the {len(kept)} kept photos into {keep_dir}/")
    head = (f"{len(paths)} photos -> {len(kept)} unique "
            f"({len(paths) - len(kept)} near-duplicates dropped, sharpest kept). "
            f"Use ONLY the kept list below for the collage.{lines_head}")
    _record(box, "dedupe_photos", f"{len(paths)} photos -> {len(kept)} kept (sharpest of each "
            f"group), dropped {len(paths) - len(kept)}: " + "; ".join(lines[:8]))
    return "\n".join([head, "kept:"] + [_rel(box, p) for p in kept] + lines)


_FOUND_FILE = Path(".agent") / "found_photos.json"
_RECORD_FILE = Path(".agent") / "record.txt"


def _record(box, tool: str, what: str) -> None:
    """A one-line-per-event log of what the file tools did in this working
    folder. Torture run #4 2026-09-14: after history compaction the model
    answered «сколько мостов нашёл?» with «я текстовый помощник, я не
    проводил поиск». The folder remembers what the chat forgot; the record
    is shown to the model every turn while the folder has files."""
    try:
        f = box.root / _RECORD_FILE
        f.parent.mkdir(parents=True, exist_ok=True)
        with f.open("a", encoding="utf-8") as fh:
            fh.write(f"{tool}: {what[:600]}\n")
    except Exception:
        logger.debug("could not append the working-folder record", exc_info=True)


def read_record(box, limit: int = 12) -> str:
    try:
        lines = (box.root / _RECORD_FILE).read_text(encoding="utf-8").splitlines()
    except Exception:
        lines = []
    out = "\n".join(lines[-limit:])
    plan = plan_text(box)          # an open plan rides along every turn
    if plan:
        out = (out + "\n" if out else "") + plan
    return out


def _remember_found(box, matches) -> None:
    """What find_content found, kept in the sandbox so the NEXT turn's
    dedupe_photos works on the same set (per user, survives a restart)."""
    try:
        import json
        f = box.root / _FOUND_FILE
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps([str(m) for m in matches]), encoding="utf-8")
    except Exception:
        logger.debug("could not remember find_content result", exc_info=True)


def _recall_found(box) -> list:
    try:
        import json
        return list(json.loads((box.root / _FOUND_FILE).read_text(encoding="utf-8")))
    except Exception:
        return []


def _rel(box, p) -> str:
    try:
        return str(Path(p).relative_to(box.root)).replace("\\", "/")
    except Exception:
        return str(p)


# -- archives ----------------------------------------------------------------

@_guard
def _handle_unpack_archive(ctx, state, args: dict) -> str:
    where = _box(ctx).unpack(args.get("path") or "", args.get("dest") or None)
    return f"Unpacked into {where}. List it to see what is inside."


@_guard
def _handle_pack_archive(ctx, state, args: dict) -> str:
    """Builds the deliverable and records it where the surfaces look for it.

    Setting document_path is what makes the file actually reach the user: the
    Telegram and GUI layers deliver from that key, and the honesty guards treat
    a pack that sets nothing as a tool that produced nothing — which is exactly
    right, since a zip nobody receives is not a result.
    """
    box = _box(ctx)
    # Bench deliver_symptom 2026-09-23 (thinking on): a correct fix inside
    # morevillagers-1.21.1_unpacked, then pack_archive(".") -- the whole work
    # folder, so every file sat under "<mod>_unpacked/..." and no game could
    # load it. An archive of the root that holds unpacked mods is refused.
    try:
        base = box.resolve(args.get("path") or "")
        inner = [d.name for d in base.iterdir() if d.is_dir() and d.name.endswith("_unpacked")]
    except Exception:
        inner = []
    if inner:
        raise SandboxError(
            f"Not packed: '{args.get('path')}' contains the unpacked folders "
            f"{', '.join(sorted(inner))}, so the archive would have them as an "
            f"extra top-level folder and the game could not load it. Pack the "
            f"one folder you changed, e.g. pack_archive(path='{sorted(inner)[0]}', "
            f"output='<name>_fixed.jar').")
    out = box.pack(args.get("path") or "", args.get("output") or "result.zip")
    # Only ONE file reaches the user: the surfaces deliver from a single
    # document_path, so a second pack replaces the first rather than adding to
    # it. The model has no way to know that unless it is told, and it does not
    # guess: measured 3/3 on bench case deliver_fix, it correctly edited Thief's
    # protection tag, packed that folder, and then packed the untouched
    # morevillagers folder as well "for completeness" -- so the archive the user
    # actually received was the one with no fix in it, under an answer
    # describing the fix. Saying which file is now the deliverable is the whole
    # correction; refusing the second pack would be worse, since a re-pack after
    # fixing a mistake is legitimate and must win.
    previous = str(state.get("document_path") or "").strip()
    state["document_path"] = str(out)
    # "success", not "ok". The Telegram delivery block sends the file only when
    # this reads exactly "success" (tg_tasks), so "ok" meant every archive the
    # agent packed was silently dropped -- while this tool's own return value
    # told the model "it will be sent to the user", and the model then told the
    # user it had been. The bench could not see it either: its check read
    # document_path and never the status the reader actually requires.
    state["document_status"] = "success"
    if previous and Path(previous).name != out.name:
        return (f"Packed {args.get('path')} into {out.name}. WARNING: this "
                f"REPLACED {Path(previous).name} as the one file the user will "
                f"receive — only one can be sent. If {Path(previous).name} was "
                f"the archive with your fix in it, pack that one again now, or "
                f"put everything the user needs into a single folder and pack "
                f"that instead.")
    return f"Packed {args.get('path')} into {out.name}; it will be sent to the user."


# -- execution ---------------------------------------------------------------

@_guard
def _handle_run_code(ctx, state, args: dict) -> str:
    box = _box(ctx)
    if not _access.may_run_code(getattr(ctx, "sandbox_user", None)):
        return ("[TOOL ERROR] This account may keep and edit files but not run "
                "code. Do the work with the file tools, or tell the user an "
                "administrator has to raise their sandbox level.")
    hijacked = _rewrites_code_file(box, args.get("code") or "")
    if hijacked:
        # Sandbox bench 2026-09-24: refused a whole-file write_file, the model
        # said "let's use run_code to overwrite rx.py completely" and did --
        # with the code inside a Python string, so '\\' became '\' and the
        # file stopped parsing.
        return (f"[TOOL ERROR] NOT run: this script rewrites {hijacked} whole, and "
                f"{hijacked} {_REWRITE_REFUSED_TAIL}")
    before = _pictures_in(box)
    res = _runner.run_python(
        box, args.get("code") or "",
        timeout=args.get("timeout") or _runner.DEFAULT_TIMEOUT,
        allow_host=_access.allow_host_execution(getattr(ctx, "sandbox_user", None)),
    )
    if not res.ok:
        # A failed run is a real result the model must reason about, not a
        # malfunction: the traceback is the most useful thing it will see all
        # turn. Prefixed so the loop counts it as a failure and does not let the
        # answer claim the work is done.
        return f"[TOOL ERROR] {res.as_tool_result()}"
    out = res.as_tool_result()
    if not (res.output or "").strip():
        # SWE-agent: silence is ambiguous -- the model reads "exit=0" alone as
        # "nothing happened" and runs it again, or invents what it printed.
        out += ("\nThe script ran successfully and printed nothing. If you need "
                "to see a value, print() it.")
    return (out + _deliver_new_pictures(box, state, before, ctx)
            + _hand_rolled_dedupe_note(args.get("code") or ""))


_HAND_ROLLED_HASH_RE = re.compile(
    r"(?:imagehash|\bhash\(|tobytes\(\)|hashlib|md5|sha1|phash|dhash|average_hash)",
    re.IGNORECASE)
_DEDUPE_INTENT_RE = re.compile(r"duplicat|dedup|дубл|uniq|уникал", re.IGNORECASE)


def _hand_rolled_dedupe_note(code: str) -> str:
    """The model wrote its own photo 'hash' three runs in a row (2026-09-14),
    each time keeping every re-shot and reporting the duplicates gone. The
    schema and the description both name assistant_tools.dedupe; this is the
    third place -- the result it reads right before answering."""
    if "assistant_tools" in code or not _DEDUPE_INTENT_RE.search(code):
        return ""
    if not _HAND_ROLLED_HASH_RE.search(code):
        return ""
    return ("\n[NOTE: this script hand-rolls an image hash. That finds only "
            "byte-identical files and MISSES re-shots of the same scene, so "
            "the copies are still in the result. Call the dedupe_photos tool "
            "on these paths (same-scene grouping, keeps the sharpest, lists "
            "every group) and rebuild the collage from its `kept:` list "
            "before you answer.]")


def _pictures_in(box) -> dict:
    """{path: mtime} of every picture in the working folder, hidden dirs skipped."""
    out = {}
    try:
        for p in box.root.rglob("*"):
            if any(part.startswith(".") for part in p.relative_to(box.root).parts):
                continue
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                out[p] = p.stat().st_mtime
    except Exception:
        pass
    return out


def _deliver_new_pictures(box, state, before: dict, ctx) -> str:
    """A script that wrote a picture has produced the result; hand it over.

    Journey 32, 2026-09-14: the script built bridge_collage.jpg, but the
    current picture was still the first bridge that find_content had set, so
    the chat received a single bridge photo and the collage only inside a
    zip. Without this there is no route from run_code to the user's eyes --
    test_image_delivery_contract documented exactly that hole.
    """
    after = _pictures_in(box)
    new = [p for p, m in after.items() if before.get(p) != m]
    if not new:
        return ""
    new.sort(key=lambda p: after[p])
    newest = new[-1]
    _record(box, "run_code", "a script wrote " + ", ".join(_rel(box, p) for p in new[-4:]))
    state["image_path"] = str(newest)
    state["image_status"] = "ok"
    try:
        ctx.last_image_path = str(newest)
    except Exception:
        pass
    names = ", ".join(str(p.relative_to(box.root)).replace("\\", "/") for p in new[-5:])
    note = (f"\n[the script wrote {len(new)} picture(s): {names}. "
            f"'{newest.relative_to(box.root)}' is now the current picture and "
            f"will be sent to the user.]")
    # Eyes: the picture is looked at once per turn (code_visual_check).
    if not state.get("_visual_checked"):
        state["_visual_checked"] = True
        try:
            import code_visual_check as _cvc
            defects = _cvc.critique(ctx, str(newest), state.get("user_input", ""))
        except Exception:
            defects = ""
        if defects:
            logger.info("visual check of %s: %s", newest.name, defects.replace("\n", " | ")[:200])
            note += _cvc.note_for(defects)
    return note


@_guard
def _handle_install_packages(ctx, state, args: dict) -> str:
    box = _box(ctx)
    if not _access.may_run_code(getattr(ctx, "sandbox_user", None)):
        return ("[TOOL ERROR] This account may not install packages. Ask the "
                "user to have an administrator raise their sandbox level.")
    pkgs = args.get("packages") or []
    if isinstance(pkgs, str):
        pkgs = [p for p in pkgs.replace(",", " ").split() if p]
    res = _runner.install(
        box, pkgs,
        allow_host=_access.allow_host_execution(getattr(ctx, "sandbox_user", None)))
    return res.as_tool_result() if res.ok else f"[TOOL ERROR] {res.as_tool_result()}"


# -- coding-agent kit: outline, tests, plan, undo ------------------------------
# The four habits a working coding agent has and ours lacked (2026-09-23): see
# the shape of a project before reading it, run the tests as a first-class step
# with the failures pulled out, keep a checklist for a long job, and take back
# its own bad edit instead of hand-reconstructing the old text.
_UNDO_DIR = Path(".agent") / "undo"
_UNDO_KEEP = 10
_PLAN_FILE = Path(".agent") / "plan.json"
_OUTLINE_SUFFIXES = {".py", ".java", ".kt", ".js", ".ts", ".go", ".rs", ".cs",
                     ".cpp", ".c", ".h"}
_DECL_RE = re.compile(
    r"^\s*(?:(?:public|private|protected|static|final|abstract|export|default|"
    r"async|open|internal|override|data|sealed|pub|fn)\s+)*"
    r"(class|interface|enum|record|object|fun|func|function|fn|struct|trait|def)"
    r"\s+([A-Za-z_]\w*)")
_JAVA_METHOD_RE = re.compile(
    r"^\s*(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?"
    r"[\w<>\[\],.? ]+\s+([a-zA-Z_]\w*)\s*\(")


def _undo_slot(box, rel: str) -> Path:
    safe = re.sub(r"[^\w.-]+", "__", rel.replace("\\", "/"))
    return box.root / _UNDO_DIR / safe


def _before(box, rel: str):
    """The file's bytes before a write (None = it did not exist). Read BEFORE
    the change but stored only AFTER it succeeds, so a refused edit leaves no
    undo point that would 'restore' identical content."""
    try:
        full = box.resolve(rel)
        return full.read_bytes() if full.is_file() else None
    except Exception:
        return None


def _snapshot(box, rel: str, prior) -> None:
    """Keep the file as it was before write_file/edit_file changed it. A file
    that did not exist yet is recorded as a '.new' marker, so undo deletes it."""
    try:
        slot = _undo_slot(box, rel)
        slot.mkdir(parents=True, exist_ok=True)
        import time
        stamp = f"{time.time_ns()}"
        if prior is not None:
            (slot / f"{stamp}.bak").write_bytes(prior)
        else:
            (slot / f"{stamp}.new").write_bytes(b"")
        olds = sorted(slot.iterdir())
        for f in olds[:-_UNDO_KEEP]:
            f.unlink(missing_ok=True)
    except Exception:
        logger.debug("undo snapshot failed for %s", rel, exc_info=True)


@_guard
def _handle_undo_edit(ctx, state, args: dict) -> str:
    box = _box(ctx)
    rel = (args.get("path") or "").strip()
    full = box.resolve(rel)
    rel = box.relative(full)
    slot = _undo_slot(box, rel)
    snaps = sorted(slot.iterdir()) if slot.is_dir() else []
    if not snaps:
        raise SandboxError(f"No earlier version of '{rel}' is kept -- only changes "
                           f"made with write_file/edit_file can be undone.")
    last = snaps[-1]
    if last.suffix == ".new":
        if full.exists():
            full.unlink()
        what = f"removed {rel} (it did not exist before your write)"
    else:
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(last.read_bytes())
        what = f"restored {rel} to the version before your last change"
    last.unlink()
    _record(box, "undo_edit", what)
    left = len(snaps) - 1
    return (f"Undone: {what}. {left} older version(s) still kept; call undo_edit "
            f"again to go further back." + _syntax_note(box, rel) if full.exists()
            else f"Undone: {what}.")


def _outline_py(text: str) -> list:
    import ast
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [f"  (SyntaxError line {exc.lineno}: {exc.msg})"]
    out = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out.append(f"  {node.lineno}: class {node.name}")
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append(f"  {sub.lineno}:   def {sub.name}()")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(f"  {node.lineno}: def {node.name}()")
    return out


def _outline_other(text: str) -> list:
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        m = _DECL_RE.match(line)
        if m:
            out.append(f"  {n}: {m.group(1)} {m.group(2)}")
            continue
        m = _JAVA_METHOD_RE.match(line)
        if m and m.group(1) not in {"if", "for", "while", "switch", "return", "new"}:
            out.append(f"  {n}:   {m.group(1)}()")
    return out


_WORD_RE = re.compile(r"[A-Za-z_]\w{2,}")


def _rank_by_references(files) -> list:
    """Aider's repo-map idea, without tree-sitter: a file matters in proportion
    to how often the names it DEFINES are used in OTHER files. When the map
    does not fit, the core module everything imports is shown and a leaf
    script is cut -- not whatever sorts first alphabetically."""
    loaded = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        rows = _outline_py(text) if f.suffix.lower() == ".py" else _outline_other(text)
        names = {m.split()[-1].rstrip("()") for m in rows if m.strip() and "Error" not in m}
        from collections import Counter
        loaded.append((f, text, rows, names, Counter(_WORD_RE.findall(text))))
    total = {}
    for *_x, words in loaded:
        for w, n in words.items():
            total[w] = total.get(w, 0) + n
    ranked = []
    for f, text, rows, names, words in loaded:
        score = sum(total.get(n, 0) - words.get(n, 0) for n in names)
        ranked.append((score, str(f), f, text, rows))
    ranked.sort(key=lambda r: (-r[0], r[1]))
    return [(f, text, rows) for _s, _n, f, text, rows in ranked]


@_guard
def _handle_code_outline(ctx, state, args: dict) -> str:
    """Classes and functions per source file, with line numbers -- the map a
    model needs before it can decide which file to read."""
    box = _box(ctx)
    start = box.resolve(args.get("path") or ".")
    files = [start] if start.is_file() else sorted(
        p for p in start.rglob("*")
        if p.is_file() and p.suffix.lower() in _OUTLINE_SUFFIXES
        and ".agent" not in p.parts and "__pycache__" not in p.parts)
    if not files:
        return "No source files here (looked for " + ", ".join(sorted(_OUTLINE_SUFFIXES)) + ")."
    budget, parts, shown = 6000, [], 0
    for f, text, rows in _rank_by_references(files[:400]):
        block = f"{box.relative(f)} ({text.count(chr(10)) + 1} lines)\n" + "\n".join(rows[:40])
        if len(rows) > 40:
            block += f"\n  ... {len(rows) - 40} more"
        if budget - len(block) < 0 and parts:
            break
        budget -= len(block)
        parts.append(block)
        shown += 1
    tail = "" if shown == len(files) else (
        f"\n[{shown} of {len(files)} files shown -- call code_outline on a "
        f"subfolder for the rest]")
    return "\n".join(parts) + tail


_FAILED_RE = re.compile(r"^(FAILED|ERROR) (\S+)(?: - (.*))?$", re.MULTILINE)
# With -q the verdict line has no "====" frame ("1 failed, 2 passed in 0.1s").
_SUMMARY_RE = re.compile(
    r"^=*\s*(\d+ (?:passed|failed|errors?|skipped|deselected)\b.*?|no tests ran.*?)\s*=*$",
    re.MULTILINE)


def _test_digest(output: str) -> str:
    """Pull the failures and the verdict out of pytest's output, so the model
    reads 'test_x: assert 3 == 4' instead of scrolling 300 lines."""
    fails = _FAILED_RE.findall(output)
    summary = _SUMMARY_RE.findall(output)
    lines = []
    if summary:
        lines.append("Result: " + summary[-1].strip())
    for kind, test, msg in fails[:15]:
        lines.append(f"  {kind} {test}" + (f" -- {msg[:200]}" if msg else ""))
    if len(fails) > 15:
        lines.append(f"  ... {len(fails) - 15} more failures")
    return "\n".join(lines)


@_guard
def _handle_run_tests(ctx, state, args: dict) -> str:
    box = _box(ctx)
    user = getattr(ctx, "sandbox_user", None)
    if not _access.may_run_code(user):
        return ("[TOOL ERROR] This account may keep and edit files but not run "
                "code, so tests cannot be run here.")
    host = _access.allow_host_execution(user)
    target = (args.get("path") or ".").strip() or "."
    box.resolve(target)                     # refuses paths outside the folder
    extra = (args.get("filter") or "").strip()
    argv = ["-q", "-rfE", "--tb=short", "-p", "no:cacheprovider", target]
    if extra:
        argv += ["-k", extra]
    # The project root on sys.path: a tests/ folder without __init__.py cannot
    # otherwise import the package beside it. Measured on the real model: every
    # run was "ERROR tests/test_x.py" and the turn went on "fixing" good imports.
    script = ("import os, sys\nsys.path.insert(0, os.getcwd())\n"
              "try:\n    import pytest\nexcept ImportError:\n    sys.exit(97)\n"
              f"sys.exit(pytest.main({argv!r}))\n")
    timeout = args.get("timeout") or 300
    res = _runner.run_python(box, script, timeout=timeout, allow_host=host)
    if res.code == 97:
        inst = _runner.install(box, ["pytest"], allow_host=host)
        if not inst.ok:
            return f"[TOOL ERROR] pytest is missing and could not be installed: {inst.as_tool_result()}"
        res = _runner.run_python(box, script, timeout=timeout, allow_host=host)
    out = res.output or ""
    if res.code == 5:
        return ("[TOOL ERROR] No tests were found under '" + target + "'. pytest "
                "collects files named test_*.py with functions named test_*. "
                "Write one with write_file, then call run_tests again.")
    digest = _test_digest(out)
    _record(box, "run_tests", (digest.splitlines() or ["exit " + str(res.code)])[0])
    tail = out[-3500:]
    body = (digest + "\n\n--- pytest output (tail) ---\n" + tail) if digest else tail
    if res.code == 0:
        return "All tests passed.\n" + body
    return "[TOOL ERROR] Tests failed (exit " + str(res.code) + ").\n" + body


def _load_plan(box) -> list:
    import json
    try:
        data = json.loads((box.root / _PLAN_FILE).read_text(encoding="utf-8"))
        return [s for s in data if isinstance(s, dict) and s.get("step")]
    except Exception:
        return []


_MARK = {"done": "[x]", "doing": "[>]", "todo": "[ ]"}


def plan_text(box) -> str:
    """The open plan, shown to the model every turn next to the record, so a
    long job survives history compaction. A finished plan is not shown."""
    steps = _load_plan(box)
    if not steps or all(s.get("status") == "done" for s in steps):
        return ""
    return "Your plan for this job:\n" + "\n".join(
        f"  {_MARK.get(s.get('status'), '[ ]')} {i}. {s['step']}"
        for i, s in enumerate(steps, 1))


@_guard
def _handle_update_plan(ctx, state, args: dict) -> str:
    import json
    box = _box(ctx)
    raw = args.get("steps") or []
    steps = []
    for s in raw:
        if isinstance(s, dict):
            step, status = str(s.get("step") or "").strip(), s.get("status") or "todo"
        else:
            step, status = str(s).strip(), "todo"
        if step:
            steps.append({"step": step[:200],
                          "status": status if status in _MARK else "todo"})
    f = box.root / _PLAN_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    if not steps:
        f.unlink(missing_ok=True)
        return "Plan cleared."
    if sum(s["status"] == "doing" for s in steps) > 1:
        raise SandboxError("Only ONE step may be 'doing' at a time.")
    f.write_text(json.dumps(steps, ensure_ascii=False, indent=1), encoding="utf-8")
    done = sum(s["status"] == "done" for s in steps)
    return (f"Plan saved ({done}/{len(steps)} done). It is shown to you every turn; "
            f"mark steps done as you finish them.\n" + (plan_text(box) or "All steps done."))


CODE_TOOL_NAMES = frozenset({
    "list_files", "read_file", "open_image", "write_file", "edit_file",
    "search_files", "find_content", "delete_path", "dedupe_photos",
    "unpack_archive", "pack_archive", "run_code", "install_packages",
    "rag_add", "rag_search",
    "code_outline", "run_tests", "update_plan", "undo_edit",
})

# The subset that needs more than `files`. Kept next to the handlers so a new
# executing tool cannot be added without deciding which side of the line it is
# on.
EXECUTING_TOOL_NAMES = frozenset({"run_code", "install_packages", "run_tests"})
