"""A directory the coding agent may touch, and nothing outside it.

Every path the model supplies is untrusted input: it is generated text, not a
user's click. So the containment check is the whole point of this module, and
it is deliberately paranoid about the ways a path escapes on Windows, which are
more numerous than on POSIX:

  * ``..`` segments, including ones that only escape after normalisation;
  * absolute paths (``C:/Windows/System32``);
  * drive-RELATIVE paths (``C:notes.txt`` means "notes.txt on C:'s current
    directory", which is not the sandbox and usually not even predictable);
  * UNC and device paths (``//server/share``, ``//./PIPE``);
  * symlinks and directory junctions pointing out of the tree, which is why the
    check runs on the RESOLVED path and never on the string.

The rule is one line and has no exceptions: after resolution, the path must be
the root or live underneath it. Everything else raises.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# Text the agent is allowed to open. A modpack is mostly JSON, TOML, config and
# scripts; opening a jar or a texture as text produces a screenful of mojibake
# that teaches the model nothing and costs a fortune in tokens.
TEXT_SUFFIXES = frozenset({
    ".txt", ".md", ".json", ".json5", ".jsonc", ".toml", ".yaml", ".yml",
    ".cfg", ".conf", ".ini", ".properties", ".env", ".xml", ".csv", ".tsv",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".kt", ".gradle", ".c",
    ".h", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".lua",
    ".sh", ".bat", ".ps1", ".sql", ".html", ".css", ".scss", ".svg",
    ".mcmeta", ".lang", ".snbt", ".zs", ".gitignore",
})

# A single file the model can read in one call. Past this the useful move is to
# search it, not to pour it into the context: 200 KB is already ~60k tokens,
# more than the whole per-request budget on this deployment.
MAX_READ_BYTES = 200_000
MAX_WRITE_BYTES = 2_000_000
MAX_LIST_ENTRIES = 400
# Ceiling on what one unpack may write. A zip's own size says nothing about its
# contents: a few-KB "zip bomb" member inflates to gigabytes, and the member was
# read into memory whole. 2 GB is far above any real modpack.
MAX_UNPACK_BYTES = int(os.getenv("SANDBOX_MAX_UNPACK_BYTES", str(2 * 1024 ** 3)))
MAX_UNPACK_FILES = 100_000
# Working debris the agent creates and the user must never receive: the folder
# unpack() makes, and the private area code_runner installs packages into.
UNPACK_SUFFIX = "_unpacked"
AGENT_DIR = ".agent"

_DRIVE_RELATIVE_RE = re.compile(r"^[A-Za-z]:(?![\\/])")
_DRIVE_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class SandboxError(Exception):
    """A path or an operation that left the sandbox. Never a bug — it is the
    expected answer to a model that guessed, so the message is written to be
    read BY the model and is surfaced as a tool error."""


class Sandbox:
    """A rooted view of the filesystem.

    `root` is created if missing and resolved once. Resolving it once matters:
    if the root itself is reached through a symlink, comparing a resolved child
    against an unresolved root would reject every legitimate path.
    """

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # -- containment ------------------------------------------------------
    def resolve(self, rel: str) -> Path:
        """Turn a model-supplied path into a real one inside the sandbox.

        Raises SandboxError rather than returning a fallback: silently
        redirecting an escape to somewhere "safe" would let the model believe it
        edited C:/Windows/System32/drivers/etc/hosts and report success.
        """
        raw = (rel or "").strip().strip('"').strip("'")
        if not raw:
            raise SandboxError("No path given. Pass a path relative to the "
                               "project root, e.g. 'config/settings.json'.")
        raw = raw.replace("\\", "/")
        if raw.startswith("//"):
            raise SandboxError(
                f"'{rel}' is a network or device path. Only paths inside the "
                f"project are allowed.")
        if _DRIVE_RELATIVE_RE.match(raw):
            raise SandboxError(
                f"'{rel}' is a drive-relative path, which does not mean what it "
                f"looks like. Use a path relative to the project root.")
        candidate = Path(raw)
        # `_DRIVE_ABSOLUTE_RE` as well as is_absolute(): on a POSIX host
        # "C:/Windows/win.ini" is a RELATIVE path, so it was quietly honoured
        # as a folder literally named "C:" -- the edit "succeeded" somewhere
        # the model did not mean, unlike on the Windows host it was written for.
        if candidate.is_absolute() or candidate.drive or _DRIVE_ABSOLUTE_RE.match(raw):
            raise SandboxError(
                f"'{rel}' is an absolute path. Use a path relative to the "
                f"project root instead.")

        # strict=False: the target may not exist yet (a file being created).
        # The PARENT chain is still resolved, so a symlinked directory in the
        # middle cannot smuggle the result out of the tree.
        full = (self.root / candidate).resolve(strict=False)
        if full != self.root and self.root not in full.parents:
            raise SandboxError(
                f"'{rel}' points outside the project. You can only read and "
                f"write files under the project root.")
        return full

    def _contains(self, p: Path) -> bool:
        """Is `p` -- after following every link -- still inside the root?

        resolve() guards the paths the model names; this guards the ones a
        WALK finds. run_code's container writes into this tree, and code there
        can leave a symlink to an absolute path, which the host then follows:
        search and pack would read or zip up a file from outside the sandbox.
        """
        try:
            full = p.resolve(strict=False)
        except OSError:
            return False
        return full == self.root or self.root in full.parents

    def _nearby(self, rel: str, limit: int = 12) -> str:
        """What IS there, phrased as the correction to a wrong guess.

        Measured end to end: asked to fix "the Thief jar", the model called
        unpack_archive("Thief.jar"), got "does not exist", and told the user to
        upload a file that was already sitting in the folder under its real name
        (thief-1.21.1.jar). A tool error that carries the answer turns that dead
        end into one more round; advice in a description does not, because by
        then the model has already decided the file is missing.
        """
        try:
            parent = (self.root / rel).parent
            if not parent.is_dir() or (self.root not in parent.parents
                                       and parent != self.root):
                parent = self.root
            names = sorted(p.name + ("/" if p.is_dir() else "")
                           for p in parent.iterdir())
        except OSError:
            return ""
        if not names:
            return " That folder is empty."
        shown = ", ".join(names[:limit])
        more = f" (+{len(names) - limit} more)" if len(names) > limit else ""
        where = self.relative(parent)
        where = "the project root" if where in ("", ".") else where
        return f" {where} contains: {shown}{more}."

    def relative(self, full: Path) -> str:
        """The path as the model should see and re-send it."""
        try:
            return full.relative_to(self.root).as_posix()
        except ValueError:          # pragma: no cover - resolve() prevents it
            return full.as_posix()

    # -- reading ----------------------------------------------------------
    def list_dir(self, rel: str = ".", depth: int = 1) -> list[str]:
        """Entries under `rel`, `depth` levels deep, as root-relative paths.

        Depth > 1 is the SWE-agent / Claude Code habit of seeing a tree at
        once. Measured on the real model (deliver_symptom): with one level per
        call it spent 12 of its 16 rounds walking data/ -> thief/ -> tags/ ->
        block/ one folder at a time and ran out before editing anything.
        """
        full = self.resolve(rel) if rel not in ("", ".", "./") else self.root
        if not full.exists():
            raise SandboxError(f"'{rel}' does not exist.{self._nearby(rel)}")
        if not full.is_dir():
            raise SandboxError(f"'{rel}' is a file, not a directory.")
        out: list[str] = []
        cut = [False]

        def walk(d, level):
            for p in sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
                # Our own private area is not the user's file. It holds
                # installed packages and the scripts run_code writes, and
                # showing it invites "what is .agent?" -- or worse, an edit.
                if p.name == AGENT_DIR:
                    continue
                if len(out) >= MAX_LIST_ENTRIES:
                    cut[0] = True
                    return
                name = self.relative(p)
                if p.is_dir():
                    if level < depth and not p.is_symlink():
                        out.append(f"{name}/")
                        walk(p, level + 1)
                    elif depth == 1:
                        # The one-level listing /files parses stays exactly as
                        # it always was: "name/".
                        out.append(f"{name}/")
                    else:
                        try:
                            n = sum(1 for _ in p.iterdir())
                        except OSError:
                            n = 0
                        out.append(f"{name}/ ({n} entries, not expanded)" if n else f"{name}/")
                else:
                    out.append(f"{name} ({p.stat().st_size} B)")

        walk(full, 1)
        if cut[0]:
            out.append(f"… truncated at {MAX_LIST_ENTRIES} entries -- list a subfolder, "
                       f"or use search_files to find a file by name")
        return out

    def read_text(self, rel: str) -> str:
        full = self.resolve(rel)
        if not full.exists():
            raise SandboxError(f"'{rel}' does not exist.{self._nearby(rel)}")
        if full.is_dir():
            raise SandboxError(f"'{rel}' is a directory. List it instead.")
        size = full.stat().st_size
        if size > MAX_READ_BYTES:
            raise SandboxError(
                f"'{rel}' is {size} bytes, over the {MAX_READ_BYTES} limit for "
                f"one read. Search it for the part you need instead.")
        if not self.is_texty(full):
            raise SandboxError(
                f"'{rel}' is not a text file. Reading it would produce noise, "
                f"not content.")
        # errors="replace": a stray byte in an otherwise readable config must
        # not turn into an exception the model cannot act on.
        return full.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def is_texty(full: Path) -> bool:
        if full.suffix.lower() in TEXT_SUFFIXES:
            return True
        # Any other suffix is sniffed too: a whitelist refused crash.log and
        # assistant_app.log as «not a text file» (live 10-03), and .out/.err/.nfo
        # are the same story. Extensionless files (LICENSE, Dockerfile) likewise.
        try:
            return b"\0" not in full.read_bytes()[:2048]
        except OSError:
            return False

    # -- writing ----------------------------------------------------------
    def write_text(self, rel: str, content: str) -> str:
        full = self.resolve(rel)
        data = content or ""
        if len(data.encode("utf-8")) > MAX_WRITE_BYTES:
            raise SandboxError(f"Refusing to write more than {MAX_WRITE_BYTES} bytes.")
        if full.is_dir():
            raise SandboxError(f"'{rel}' is a directory.")
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(data, encoding="utf-8", newline="")
        return self.relative(full)

    def replace_once(self, rel: str, old: str, new: str) -> str:
        """Exact single-occurrence replacement.

        Requiring the match to be unique is what makes this safe to hand to a
        model: an `old` that appears twice means the model has not identified
        the site it thinks it has, and replacing the first hit would edit the
        wrong line while reporting success.
        """
        if not old:
            raise SandboxError("`old` is empty — use write_file to create or "
                               "overwrite a file.")
        text = self.read_text(rel)
        n = text.count(old)
        if n == 0:
            raise SandboxError(
                f"That exact text is not in '{rel}'. Read the file and copy the "
                f"lines you mean, including their indentation."
                + self._closest(text, old))
        if n > 1:
            raise SandboxError(
                f"That text appears {n} times in '{rel}', so it does not "
                f"identify one place. Include more surrounding lines.")
        self.write_text(rel, text.replace(old, new, 1))
        return self.relative(self.resolve(rel))

    def replace_flexible(self, rel: str, old: str, new: str) -> tuple:
        """replace_once, plus Aider's fallback for the commonest near-miss.

        Measured failure: the model copies the right lines with the wrong
        indentation or trailing spaces (a tab for four spaces, a block quoted
        from memory one level shallower). Exact match fails, the model retries
        with the same mistake. Aider's answer: if the lines match once every
        line is stripped, and match in ONE place, apply there -- re-indented by
        the offset between what was quoted and what is in the file.

        Returns (path, note). The note tells the model a fuzzy match was used,
        so it learns what the file really looks like.
        """
        try:
            return self.replace_once(rel, old, new), ""
        except SandboxError as exc:
            if "not in" not in str(exc):
                raise
            err = exc
        text = self.read_text(rel)
        want = [ln.strip() for ln in (old or "").strip("\n").splitlines()]
        if not want or not any(want):
            raise err
        have = text.splitlines(keepends=True)
        stripped = [ln.strip() for ln in have]
        hits = [i for i in range(len(have) - len(want) + 1)
                if stripped[i:i + len(want)] == want]
        if len(hits) != 1:
            raise err
        i = hits[0]
        real_first = have[i]
        quoted_first = (old or "").strip("\n").splitlines()[0]
        real_ind = real_first[:len(real_first) - len(real_first.lstrip())]
        quoted_ind = quoted_first[:len(quoted_first) - len(quoted_first.lstrip())]

        def _ind(s):
            return s[:len(s) - len(s.lstrip())]

        # Only a CONSISTENT shift is safe to repair: every quoted line off by
        # the same prefix. Measured on the real model: `old` quoted an INI
        # block with one line indented and the next not; repairing that
        # indented `temperature` under `num_ctx`, which configparser reads as
        # a continuation -- the key vanished while the edit "succeeded".
        quoted = (old or "").strip("\n").splitlines()
        for q, r in zip(quoted, have[i:i + len(want)]):
            if not q.strip():
                continue
            qi, ri = _ind(q), _ind(r.rstrip("\r\n"))
            if not qi.startswith(quoted_ind) or ri != real_ind + qi[len(quoted_ind):]:
                raise err
        new_lines = (new or "").splitlines()
        fixed = []
        for ln in new_lines:
            if ln.startswith(quoted_ind):
                ln = real_ind + ln[len(quoted_ind):]
            elif ln.strip():
                ln = real_ind + ln.lstrip()
            fixed.append(ln)
        eol = "\r\n" if real_first.endswith("\r\n") else "\n"
        block = eol.join(fixed) + (eol if new_lines else "")
        out = "".join(have[:i]) + block + "".join(have[i + len(want):])
        self.write_text(rel, out)
        return (self.relative(self.resolve(rel)),
                f" (Your `old` did not match exactly -- whitespace/indentation "
                f"differed -- so it was matched line by line at line {i + 1} and "
                f"`new` was re-indented to fit. Copy indentation exactly next time.)")

    # -- reset ------------------------------------------------------------
    def reset(self) -> int:
        """Empty the sandbox and keep the root. Returns how many entries went.

        The escape hatch for "the agent wrecked it": a run that half-unpacked an
        archive, a venv that will not import, a tree the model has confused
        itself with. Cheaper to explain than any repair.

        Deletes the CONTENTS, never the root, and never anything reached
        through resolve() — it walks the root's own entries directly, so a
        symlink inside the sandbox is unlinked rather than followed into
        whatever it points at.
        """
        import shutil

        removed = 0
        for p in list(self.root.iterdir()):
            try:
                if p.is_symlink() or p.is_file():
                    p.unlink()
                else:
                    shutil.rmtree(p, ignore_errors=True)
                removed += 1
            except OSError:
                continue
        return removed

    def remove(self, rel: str) -> str:
        """Delete ONE entry -- a file or a whole folder -- inside the sandbox.

        The root itself ("", ".", "*", "all") empties the folder through
        reset(): live 2026-09-14 «удали все файлы» hit a refusal here and the
        model reported «Я удалил все файлы» over 17 surviving entries. It is
        the user's own working folder and /reset_sandbox does the same, so
        the honest tool is one that actually does it. Any other path goes
        through resolve() so a symlink cannot point the deletion outside.
        Returns what was removed, for the reply.
        """
        import shutil
        rel = (rel or "").strip()
        if rel.strip("./\\").lower() in ("", "*", "all", "everything"):
            n = self.reset()
            return f"everything in the working folder ({n} entries)"
        full = self.resolve(rel)
        if full == self.root:
            n = self.reset()
            return f"everything in the working folder ({n} entries)"
        if not full.exists() and not full.is_symlink():
            raise SandboxError(f"'{rel}' does not exist.{self._nearby(rel)}")
        if full.is_symlink() or full.is_file():
            full.unlink()
            return f"file {rel}"
        n = sum(1 for _ in full.rglob("*") if _.is_file())
        shutil.rmtree(full)
        return f"folder {rel} ({n} files)"

    def usage(self) -> tuple[int, int]:
        """(files, bytes) — what the owner is holding, for the UI and quotas."""
        files = total = 0
        for p in self.root.rglob("*"):
            if p.is_file() and not p.is_symlink():
                files += 1
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return files, total

    @staticmethod
    def _closest(haystack: str, needle: str, k: int = 3) -> str:
        """The lines that most resemble what the model tried to replace.

        Same principle as the missing-file listing: an error that carries its
        own correction is worth several rounds of guessing. An exact-match edit
        usually fails on a detail the model cannot see from memory -- two spaces
        instead of four, a trailing comma, a quote style -- and showing the real
        line fixes it in one step where "copy it exactly" does not.
        """
        import difflib

        probe = (needle or "").strip().splitlines()
        if not probe:
            return ""
        first = probe[0]
        lines = (haystack or "").splitlines()
        best = difflib.get_close_matches(first, lines, n=k, cutoff=0.5)
        if not best:
            return ""
        shown = "".join(f"{chr(10)}    {b.rstrip()}" for b in best)
        return f" The closest lines actually in the file are:{shown}"

    # -- searching --------------------------------------------------------

    def search(self, pattern: str, rel: str = ".", max_hits: int = 60,
               max_per_file: int = 6, offset: int = 0,
               max_files: int = 12) -> list[str]:
        """Grep the tree. The tool that makes a 40 MB jar usable at all.

        Reading files one by one to find where a block id is registered is
        hopeless inside a per-request budget; searching for it is one call.

        `max_per_file` is why this is not a plain grep. The total cap alone let
        ONE verbose file swallow the whole budget: measured on a real 183-file
        mod jar, `job_site|workstation|block|villager` returned 61 lines of
        which 38 came from a single advancement file, and the lang file that
        actually held the eight block ids never appeared at all. The model then
        answered, correctly given what it was shown, that the ids are not in the
        archive. A truncated search that looks exhaustive is worse than one that
        admits it: the sample has to be spread across FILES, because "which
        files mention this" is the question a broad search is really asking.
        """
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise SandboxError(f"Bad search pattern: {exc}")
        base = self.resolve(rel) if rel not in ("", ".", "./") else self.root

        # Pass 1: WHICH files match, and how much. Collecting the whole picture
        # first is what makes the budget divisible; walking and emitting in one
        # pass can only ever spend it on whatever rglob happens to reach first.
        per_file: list[tuple[str, bool, list[tuple[int, str]]]] = []
        scanned = 0
        for path in base.rglob("*"):
            if path.is_symlink() or not path.is_file() or not self._contains(path):
                continue
            scanned += 1
            _rel = self.relative(path)
            # The PATH counts as content. Measured end to end: asked to edit the
            # tag "break_protected/medium", the model searched for exactly that
            # and found nothing three times over, because the string lives in
            # the file's location and not in any line of it -- which is how
            # Minecraft data packs, and most convention-over-configuration
            # layouts, are organised. Reported separately, so a name match is
            # not mistaken for a line of the file.
            name_hit = bool(rx.search(_rel))
            lines: list[tuple[int, str]] = []
            if self.is_texty(path):
                try:
                    if path.stat().st_size <= MAX_READ_BYTES:
                        for n, line in enumerate(path.read_text(
                                encoding="utf-8", errors="replace").splitlines(), 1):
                            if rx.search(line):
                                lines.append((n, line.strip()[:200]))
                except OSError:
                    pass
            if name_hit or lines:
                per_file.append((_rel, name_hit, lines))

        if not per_file:
            return []

        # A pattern that matches nearly everything has told the model nothing,
        # and paging through it is fifteen calls to learn that. Measured on the
        # real jar: `job_site|workstation|block|villager` matched 181 of 183
        # files, because "block" appears in every recipe and advancement. The
        # model read the truncated sample, saw only advancement files, and
        # concluded the ids were not in the archive. Saying so up front is the
        # difference between a bad search and a wrong answer.
        if scanned >= 8 and len(per_file) > (scanned * 2) // 3:
            # Answer the question it is really asking, then say why the lines
            # were withheld. A bare refusal sent the model round in circles:
            # measured, it retried `.*\.json`, `.*\.json|.*\.txt`,
            # `.*\.json|.*\.mcfunction|.*\.nbt` six times, burned all 16
            # rounds, and finished by claiming a fix it had never written. A
            # pattern like that is a request for the FILE LIST, which is cheap;
            # it was the per-line output that had to be refused, not the query.
            names = [rel for rel, _n, _l in per_file][:80]
            listing = "\n".join("  " + n for n in names)
            more = ("\n  … and %d more" % (len(per_file) - len(names))
                    if len(per_file) > len(names) else "")
            return [f"That pattern matches {len(per_file)} of the {scanned} "
                    f"files here, which is too broad to be useful — one of its "
                    f"terms is probably a word that appears everywhere. Search "
                    f"for something specific to what you want: an exact id "
                    f"('morevillagers:'), a key you expect ('block.'), or a "
                    f"full path fragment. Repeating a broad pattern will get "
                    f"this same answer. Here are the files it matched, which "
                    f"is probably what you wanted:\n{listing}{more}"]

        files = per_file[offset:] if offset else per_file
        if not files:
            return [f"No more matches. {len(per_file)} files matched in total; "
                    f"you have already seen them all."]

        # Pass 2: page by FILE, not by line. Paging by lines is what produced
        # the failure this was written for -- twenty verbose advancement files
        # filled the 60-line budget and the lang file holding the answer was
        # never reached, on page one or any other. A page of whole files always
        # advances, so asking again always shows something new.
        page = files[:max_files]
        share = max(1, min(max_per_file, max_hits // max(1, len(page))))
        hits: list[str] = []
        shown = 0
        for _rel, name_hit, lines in page:
            shown += 1
            if name_hit:
                hits.append(f"{_rel}: (filename match)")
            for n, text in lines[:share]:
                hits.append(f"{_rel}:{n}: {text}")
            if len(lines) > share:
                # Said per file, not once at the end: the model needs to know
                # WHICH file it has only seen part of, to decide whether to
                # open it.
                hits.append(f"{_rel}: … {len(lines) - share} more matching "
                            f"lines in this file — read_file it to see them")

        rest = len(per_file) - offset - shown
        if offset or rest > 0:
            hits.insert(0, f"{len(per_file)} files match; showing "
                            f"{offset + 1}-{offset + shown} of them.")
        if rest > 0:
            # Iterative by design: a search that stops with no way to continue
            # forces the model to guess a narrower pattern, and a guess that
            # misses reads as "not in the archive".
            hits.append(f"… {rest} more matching files not shown. Search again "
                        f"with the same pattern and offset={offset + shown} to "
                        f"see them.")
        return hits

    # -- archives ---------------------------------------------------------
    def unpack(self, archive_rel: str, dest_rel: str | None = None) -> str:
        """Extract a zip/jar INTO the sandbox.

        Every member name goes through resolve(), which is what stops zip-slip:
        an archive entry called '../../../../Windows/System32/x.dll' is a normal
        thing to find in a hostile file, and the extraction helpers in the
        standard library have historically honoured it.
        """
        import zipfile

        src = self.resolve(archive_rel)
        if not src.exists():
            raise SandboxError(
                f"'{archive_rel}' does not exist.{self._nearby(archive_rel)}")
        # Beside the archive, not at the root. A modpack is mods/*.jar, and two
        # mods called thief.jar in different folders both unpacked to
        # "thief_unpacked" -- the second silently merging into the first, so
        # the model would then read one mod's tag file believing it was the
        # other's. Nesting the destination makes that collision impossible and
        # keeps a 200-mod pack from littering the root.
        if not dest_rel:
            parent = self.relative(src.parent)
            stem = src.stem + UNPACK_SUFFIX
            dest_rel = stem if parent in ("", ".") else f"{parent}/{stem}"
        dest = self.resolve(dest_rel)
        dest.mkdir(parents=True, exist_ok=True)
        if not zipfile.is_zipfile(src):
            return self._unpack_other(src, archive_rel, dest, dest_rel)
        written = 0
        budget = [MAX_UNPACK_BYTES]
        with zipfile.ZipFile(src) as zf:
            members = [i for i in zf.infolist() if not i.is_dir()]
            declared = sum(i.file_size for i in members)
            if len(members) > MAX_UNPACK_FILES or declared > MAX_UNPACK_BYTES:
                raise SandboxError(self._too_big(archive_rel, len(members), declared))
            for info in members:
                # resolve() raises on anything that leaves the tree.
                target = self.resolve(f"{dest_rel}/{info.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as fh, open(target, "wb") as out:
                    # Streamed and counted: the declared sizes in the header
                    # are the archive's own claim, not a fact.
                    self._copy_capped(fh, out, budget, archive_rel)
                written += 1
        return f"{self.relative(dest)} ({written} files)"

    @staticmethod
    def _too_big(archive_rel: str, files: int, size: int) -> str:
        return (f"'{archive_rel}' unpacks to {size} bytes in {files} files, over the "
                f"{MAX_UNPACK_BYTES}-byte / {MAX_UNPACK_FILES}-file limit for one "
                f"unpack. It is too large (or a decompression bomb) -- tell the user.")

    @staticmethod
    def _copy_capped(src, dst, budget: list, archive_rel: str) -> None:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                return
            budget[0] -= len(chunk)
            if budget[0] < 0:
                raise SandboxError(
                    f"'{archive_rel}' unpacks to more than {MAX_UNPACK_BYTES} bytes, "
                    f"the limit for one unpack. It is too large (or a decompression "
                    f"bomb) -- tell the user.")
            dst.write(chunk)

    def _unpack_other(self, src: Path, archive_rel: str, dest: Path, dest_rel: str) -> str:
        """rar / 7z / tar[.gz] / anything 7-Zip reads, through py7zz.

        py7zz ships its own 7zz binary inside the wheel, which is the point:
        live 2026-09-22 a user's mods.rar hit "not a zip/jar archive" and the
        model spent two turns on rarfile, which is only a wrapper around an
        unrar executable that neither this host nor the run_code image has.

        7-Zip extracts into a private temp dir first; every file is then copied
        in through resolve(), the same zip-slip gate the zip branch uses, and
        anything that is not a regular file (a symlink in a tar) is dropped.
        """
        import shutil
        import tempfile
        try:
            import py7zz
        except ImportError:
            raise SandboxError(
                f"'{archive_rel}' is not a zip/jar and the rar/7z/tar extractor "
                "(py7zz) is not installed on the server. Tell the user.")
        written = 0
        # 7-Zip writes the whole archive out before a single file is checked,
        # so the size gate has to come first, from the archive's listing. That
        # listing is the archive's own claim; the copy below is counted too.
        try:
            with py7zz.SevenZipFile(str(src), "r") as sz:
                listed = [i for i in sz.infolist() if not i.is_dir()]
            declared = sum(int(getattr(i, "file_size", 0) or 0) for i in listed)
        except Exception:
            listed, declared = [], 0     # unlistable: extract_archive says why
        if len(listed) > MAX_UNPACK_FILES or declared > MAX_UNPACK_BYTES:
            raise SandboxError(self._too_big(archive_rel, len(listed), declared))
        budget = [MAX_UNPACK_BYTES]
        with tempfile.TemporaryDirectory(prefix="unpack_") as tmp:
            try:
                py7zz.extract_archive(str(src), tmp)
            except Exception as exc:
                # 7-Zip's own "Cannot open the file as archive" means the file is
                # no archive at all (a .txt, a .jar that is really a class file);
                # a model told "damaged or password-protected" goes looking for a
                # password. Its raw message also carries server temp paths.
                if "cannot open the file as archive" in str(exc).lower():
                    raise SandboxError(
                        f"'{archive_rel}' is not an archive -- not a zip, jar, rar, "
                        "7z or tar. Read it as a file instead.")
                raise SandboxError(
                    f"'{archive_rel}' could not be opened as an archive "
                    f"({str(exc).splitlines()[-1][:160] if str(exc).strip() else type(exc).__name__}). "
                    "It may be damaged or password-protected -- tell the user.")
            tmp_root = Path(tmp)
            for f in tmp_root.rglob("*"):
                if f.is_symlink() or not f.is_file():
                    continue
                rel = f.relative_to(tmp_root).as_posix()
                target = self.resolve(f"{dest_rel}/{rel}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(f, "rb") as fh, open(target, "wb") as out:
                    self._copy_capped(fh, out, budget, archive_rel)
                written += 1
        return f"{self.relative(dest)} ({written} files)"

    def pack(self, dir_rel: str, out_rel: str) -> Path:
        """Zip a directory back up, or hand a single file straight through.

        Returns the real path, for delivery. A lone file (a .jar the user
        wants exactly as it is, not re-wrapped in a zip) is delivered
        verbatim, under its own real name -- `out_rel` is ignored for it.
        Before this, a single file raised SandboxError and the model had no
        real way to satisfy "just the jar, not a zip": observed live, it
        answered with a fabricated `sandbox:/mnt/data/...` markdown link --
        a ChatGPT-code-interpreter path this bot has never had -- instead of
        admitting the tool it actually has only zips folders.
        """
        import zipfile

        base = self.resolve(dir_rel)
        if base.is_file():
            return base
        if not base.is_dir():
            raise SandboxError(f"'{dir_rel}' is not a file or a directory.")
        out = self.resolve(out_rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(base.rglob("*")):
                if p.is_symlink() or not p.is_file() or p == out or not self._contains(p):
                    continue
                rel = p.relative_to(base)
                # Leave our own working debris out of what the user receives.
                # Measured on the modpack round trip: the pack that went back
                # carried mods/thief_unpacked/ -- a second, extracted copy of a
                # mod that had already been repacked into its jar beside it.
                # The archive was right and also obviously wrong, at double the
                # size. The folder being packed is never skipped, only nested
                # ones: packing "thief_unpacked" itself is the normal case.
                if any(part == AGENT_DIR or part.endswith(UNPACK_SUFFIX)
                       for part in rel.parts[:-1]):
                    continue
                zf.write(p, rel.as_posix())
                count += 1
        if not count:
            raise SandboxError(f"'{dir_rel}' has no files to pack.")
        return out


# -- one sandbox per user ---------------------------------------------------
# Everything lives under runtime/, which is already on the never-commit list, so
# a user's files cannot reach the repository by accident.
SANDBOX_BASE = Path(__file__).resolve().parents[1] / "runtime" / "sandboxes"

_KEY_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]")


def redirect_sandbox_base(path) -> Path:
    """Point every sandbox at `path`. For tests, and only for tests.

    tg_bot.redirect_data_dir moves the user database out of the operator's live
    data, and every suite that builds a bot is required to call it -- but it
    never covered this tree. A suite that drives a document upload for a user
    WITH sandbox access therefore wrote a real file into the operator's real
    sandbox: writing tests/test_tg_sandbox_upload.py left a stray
    runtime/sandboxes/999877 behind on the first run, which is how this was
    found. Same discipline, second door.
    """
    global SANDBOX_BASE
    SANDBOX_BASE = Path(path)
    SANDBOX_BASE.mkdir(parents=True, exist_ok=True)
    return SANDBOX_BASE


def sandbox_key(owner) -> str:
    """A directory name from a chat/user id.

    The id is external input, so it is reduced to a conservative alphabet
    rather than trusted: a key containing '..' or a separator would put one
    user's sandbox inside another's, or outside the base entirely.
    """
    key = _KEY_SAFE_RE.sub("_", str(owner or "").strip())[:64]
    return key or "default"


def sandbox_for(owner, base=None) -> "Sandbox":
    """The sandbox belonging to one user. Created on first use.

    A root per user is not tidiness, it is the isolation itself: two people
    uploading `config.json` must not overwrite each other, and no prompt can
    make one of them read the other's files, because the other's files are not
    reachable from their root at all.
    """
    root = Path(base or SANDBOX_BASE) / sandbox_key(owner)
    return Sandbox(root)
