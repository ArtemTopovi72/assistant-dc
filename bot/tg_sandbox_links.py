"""Sandbox file names in a reply become tap-to-download links.

A name the bot writes ("app.py", "out/report.csv") is wrapped in a deep link
t.me/<bot>?start=f_<hash>. Tapping it sends "/start f_<hash>" to the bot, which
finds the file by that hash in THIS user's sandbox and sends it. The hash is
of (chat, path), so nothing is stored and a link survives a bot restart; a link
forwarded to someone else resolves inside THEIR sandbox, i.e. to nothing.
"""
import hashlib
import html as _html
import os
import re

_MAX_FILES = 400                 # a huge unpacked modpack is not linkified file by file
_SEG_RE = re.compile(r"(<a\b.*?</a>|<pre\b.*?</pre>|<[^>]+>)", re.S | re.I)


def token(chat_id, rel: str) -> str:
    return "f_" + hashlib.sha1(f"{chat_id}:{rel}".encode("utf-8")).hexdigest()[:16]


def sandbox_files(box) -> list:
    """Root-relative posix paths of the files in a sandbox (bounded)."""
    out = []
    for dirpath, dirs, files in os.walk(box.root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            out.append(os.path.relpath(os.path.join(dirpath, f), box.root).replace(os.sep, "/"))
            if len(out) >= _MAX_FILES:
                return out
    return out


def find(box, chat_id, tok: str):
    """The sandbox path a link token points at, or None."""
    for rel in sandbox_files(box):
        if token(chat_id, rel) == tok:
            return rel
    return None


def linkify(reply_html: str, chat_id, box, bot_username: str) -> str:
    """Wrap every sandbox file name in the reply (outside links and <pre>) in its link."""
    if not bot_username or box is None:
        return reply_html
    files = sandbox_files(box)
    if not files:
        return reply_html
    # A bare name links only when it is unique in the sandbox; a path always does.
    by_name = {}
    for rel in files:
        by_name.setdefault(rel.rsplit("/", 1)[-1], []).append(rel)
    names = {rel: rel for rel in files}
    names.update({n: rels[0] for n, rels in by_name.items() if len(rels) == 1})
    pat = re.compile(r"(?<![\w/.-])(" + "|".join(
        re.escape(_html.escape(n, quote=False)) for n in sorted(names, key=len, reverse=True))
        + r")(?![\w/-]|\.\w)")

    def _link(m):
        rel = names[_html.unescape(m.group(1))]
        return f'<a href="https://t.me/{bot_username}?start={token(chat_id, rel)}">{m.group(1)}</a>'

    out = []
    for part in _SEG_RE.split(reply_html):
        low = part[:4].lower()
        if part.startswith("<") and (low.startswith("<a") or low.startswith("<pre")):
            out.append(part)                    # already a link / a code block
        elif part.startswith("<"):
            out.append(part)                    # a tag
        else:
            out.append(pat.sub(_link, part))
    # <code>name</code> -> link around the name: Telegram cannot nest <a> in <code>.
    return re.sub(r"<code>(<a href=\"[^\"]+\">[^<]+</a>)</code>", r"\1", "".join(out))
