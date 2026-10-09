"""Telegram markup: markdown -> Telegram HTML, and tag-safe splitting.

Split out of tg_bot.py. Pure text in, pure text out -- no network, no session,
no localization. That is what made it safe to move: nothing here reads bot
state, and tg_bot re-exports every name, so tg_library / tg_tasks /
tg_transport keep resolving `tg_bot._split_html` and friends unchanged.

Load-bearing, despite being "just formatting": Telegram REJECTS a message
whose markup is malformed, and a rejected edit deletes the reply the user was
reading. _balance_html and _split_html exist to guarantee every chunk is
independently well-formed, which is why splitting happens on tag boundaries and
not on character counts.
"""
import html as _html_mod
import re

_MAX_TEXT = 4096


# ── text helpers ──────────────────────────────────────────────────────────────

_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")


def _md_tables_to_fences(text: str) -> str:
    """A markdown pipe table -> a fenced block with aligned columns. Telegram
    has no tables; "| --- |" arrived as raw pipes (live 2026-09-28)."""
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        if (i + 1 < len(lines) and "|" in lines[i] and _TABLE_SEP.match(lines[i + 1])
                and not text[:sum(len(x) + 1 for x in lines[:i])].count("```") % 2):
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            rows = [[re.sub(r"\*\*|__|`", "", c).strip() for c in ln.strip().strip("|").split("|")]
                    for ln in [lines[i]] + lines[i + 2:j]]
            n = max(len(r) for r in rows)
            rows = [r + [""] * (n - len(r)) for r in rows]
            w = [max(len(r[k]) for r in rows) for k in range(n)]
            out.append("```")
            out += ["  ".join(r[k].ljust(w[k]) for k in range(n)).rstrip() for r in rows]
            out.append("```")
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


_TEX_SYMBOLS = {
    "rightarrow": "→", "to": "→", "leftarrow": "←", "Rightarrow": "⇒", "Leftarrow": "⇐",
    "leftrightarrow": "↔", "times": "×", "cdot": "·", "div": "÷", "pm": "±", "approx": "≈",
    "neq": "≠", "ne": "≠", "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "infty": "∞",
    "sum": "Σ", "prod": "∏", "sqrt": "√", "alpha": "α", "beta": "β", "gamma": "γ",
    "delta": "δ", "Delta": "Δ", "pi": "π", "sigma": "σ", "mu": "μ", "lambda": "λ",
    "theta": "θ", "omega": "ω", "degree": "°", "ldots": "…", "dots": "…", "%": "%",
}
_TEX_SPAN = re.compile(
    r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|\\\((.+?)\\\)|(?<![\\\w$])\$([^$\n]+?)\$(?![\w$])", re.DOTALL)
_SUP = dict(zip("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))


def _latex_to_text(text: str) -> str:
    r"""$\rightarrow$ / $x^2$ / \frac{a}{b} -> plain Unicode. Telegram renders no
    TeX; a raw `$\rightarrow$` reached the chat (live 2026-10-04). Code is left alone."""
    def one(m):
        e = next(g for g in m.groups() if g is not None)
        e = re.sub(r"\\(?:text|mathrm|mathbf|operatorname)\{([^}]*)\}", r"\1", e)
        e = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", e)
        e = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", e)
        e = re.sub(r"\^\{?(\d+)\}?", lambda k: "".join(_SUP[c] for c in k.group(1)), e)
        e = re.sub(r"\\([A-Za-z]+|%)", lambda k: _TEX_SYMBOLS.get(k.group(1), k.group(1)), e)
        return re.sub(r"[{}]", "", e).strip()
    return "".join(p if p.startswith("`") else _TEX_SPAN.sub(one, p)
                   for p in re.split(r"(```.*?```|`[^`\n]*`)", text, flags=re.DOTALL))


def _md_to_html(text: str) -> str:
    """Markdown -> the small HTML subset Telegram accepts (b/i/u/s/a/code/pre).

    Deep-research reports are mostly links and headings, so those must survive:
    a bare [title](url) rendered as literal brackets is the difference between a
    usable answer and a wall of text. Everything is escaped FIRST, then the markup
    is re-introduced, so user text can never inject tags.
    """
    # NUL is the placeholder marker used further down for stashed tags. A NUL in
    # the source text (model output, an extracted PDF) would be read back as a
    # placeholder index and raise IndexError — which the delivery try/except then
    # swallowed, so the user silently received NOTHING. Drop it up front.
    t = _html_mod.escape(_md_tables_to_fences(_latex_to_text(text.replace("\x00", ""))), quote=False)
    # fenced + inline code first, so markup inside them is left alone.
    # A fenced block is put on lines of its own: every emphasis rule below is
    # line-bounded, so newline isolation is what makes a multi-line <pre> block
    # impossible to half-wrap.
    # The fence's language rides along: Telegram labels the block with it.
    t = re.sub(r"```([\w+#-]*)[ \t]*\n?(.*?)```",
               lambda m: ("\n<pre><code class=\"language-%s\">%s</code></pre>\n" % (m.group(1).lower(), m.group(2))
                          if m.group(1) else "\n<pre>" + m.group(2) + "</pre>\n"), t, flags=re.DOTALL)
    # Inline code only OUTSIDE <pre>: `x` inside a fenced block became <code>
    # and the quoted snippet lost its backticks (live 2026-09-28).
    t = "".join(p if p.startswith("<pre>") else re.sub(r"`([^`\n]+)`", r"<code>\1</code>", p)
                for p in re.split(r"(<pre>.*?</pre>)", t, flags=re.DOTALL))

    # Every rule from here on rewrites spans of text, and none of them can be
    # allowed to cut an existing element in half: "[`](http://)`" produced
    # '<a href="http://"><code></a></code>' — crossed tags that Telegram refuses,
    # losing the whole message. Finished elements are therefore replaced by opaque
    # placeholders and put back at the very end.
    protected = []

    def _stash(m):
        protected.append(m.group(0))
        return "\x00%d\x00" % (len(protected) - 1)

    # Phase 1: code/pre only. Bare URLs must NOT be hidden yet — the link rule
    # below still has to see the target inside [label](url).
    t = re.sub(r"<(code|pre)\b[^>]*>.*?</\1>", _stash, t, flags=re.DOTALL)

    # [label](url) -> <a href="url">label</a>. Only http(s) targets: Telegram
    # rejects the message outright on a malformed href, which would drop the whole
    # chunk. &amp; is what escaping turned & into — put it back inside the href.
    def _link(m):
        label, url = m.group(1), m.group(2).replace("&amp;", "&")
        if not url.lower().startswith(("http://", "https://", "tg://")):
            return m.group(0)
        return f'<a href="{_html_mod.escape(url, quote=True)}">{label}</a>'
    # The URL may itself contain a balanced pair of parentheses — Wikipedia titles
    # are full of them ("Mercury_(planet)"), and research reports are mostly
    # Wikipedia links. A non-greedy \S+? stopped at the FIRST ")" and shipped a
    # truncated, dead href plus a stray ")" in the text.
    # \x00 is excluded from every URL body: a stashed <code> sitting inside a
    # target ("<http://`e`>") would otherwise be restored INSIDE the href
    # attribute, producing href="http://<code>e</code>" — malformed markup that
    # Telegram rejects.
    t = re.sub(r"\[([^\]\n]+)\]\(((?:[^\s()\x00]|\([^\s()\x00]*\))+)\)", _link, t)
    # Hide finished links before the autolink rule runs: a bare <http://x> sitting
    # inside a [label](url) would otherwise be turned into a SECOND <a> nested in
    # the first, and Telegram has no nested links.
    t = re.sub(r"<a\b[^>]*>.*?</a>", _stash, t, flags=re.DOTALL)
    # bare autolinks (<https://…> survives escaping as &lt;https://…&gt;)
    t = re.sub(r"&lt;(https?://[^\s&\x00]+)&gt;", r'<a href="\1">\1</a>', t)
    # Phase 2: links are built, so hide them from the emphasis rules along with
    # bare (unlinkified) URLs — "site.ru/_foo_/bar" would otherwise become
    # "site.ru/<i>foo</i>/bar" and the link would break. Escaping ran first, so
    # the only real tags present are ones we created.
    # The emphasis rules are line-bounded while <pre> is not, which is the other
    # half of the crossed-tag problem: "# ```\n```" used to become
    # "<b><pre></b>\n</pre>".
    t = re.sub(r"<a\b[^>]*>.*?</a>|<[^>]+>|https?://[^\s\x00]+",
               _stash, t, flags=re.DOTALL)

    def _stash_elems(s: str, names: str) -> str:
        return re.sub(r"<(%s)\b[^>]*>.*?</\1>" % names, _stash, s, flags=re.DOTALL)

    # bold/italic. Both are line-bounded and reject a leading "* " bullet, so a
    # markdown list is not swallowed into one giant <i> spanning the whole report.
    # Each pass hides what it produced before the next one runs: the passes have
    # different delimiters but overlapping spans, so "_**x_**" resolved to
    # "<i><b>x</i></b>" — crossed tags, and Telegram drops the message. Hiding the
    # finished element also lets the next pass nest around it cleanly.
    t = re.sub(r"\*\*(?!\s)([^\n*]+?)(?<!\s)\*\*", r"<b>\1</b>", t)
    t = _stash_elems(t, "b")
    t = re.sub(r"(?<![\w*])\*(?!\s)([^\n*]+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = _stash_elems(t, "i")
    # Underscore emphasis (_italic_, __bold__). Models mix both markdown dialects,
    # and an unhandled "_" reached users as a literal stray character. Delimiters
    # must sit on a word boundary, so snake_case identifiers and
    # bare_urls/with_underscores are left alone. Slashes are excluded from the
    # italic body for the same reason (an unlinkified URL path).
    t = re.sub(r"(?<!\w)__(?!\s)([^\n_]+?)(?<!\s)__(?!\w)", r"<b>\1</b>", t)
    t = _stash_elems(t, "b")
    t = re.sub(r"(?<!\w)_(?!\s)([^\n_/]+?)(?<!\s)_(?!\w)", r"<i>\1</i>", t)
    t = _stash_elems(t, "i")
    # ~~strike~~, a --- rule, > quotes: they reached the chat as literal «~~», «---» and «&gt;»
    t = re.sub(r"~~(?!\s)([^\n~]+?)(?<!\s)~~", r"<s>\1</s>", t)
    t = _stash_elems(t, "s")
    t = re.sub(r"(?m)^[ \t]{0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$", "──────────", t)
    t = re.sub(r"(?m)(?:^[ \t]{0,3}&gt;[ \t]?.*(?:\n|$))+",
               lambda m: "<blockquote>" + "\n".join(
                   re.sub(r"^[ \t]{0,3}&gt;[ \t]?", "", ln) for ln in m.group(0).rstrip("\n").split("\n"))
               + "</blockquote>" + ("\n" if m.group(0).endswith("\n") else ""), t)
    # ATX headings -> bold line (Telegram has no heading tag). Last, because by
    # now a line holds nothing but plain text and placeholders, so wrapping the
    # whole line cannot cut through anything. "# **X**" still nests correctly.
    # [ \t] — NOT \s — around the optional closing hashes. \s matches NEWLINES, so
    # the trailing `\s*#*\s*$` ate the line break after every heading (and the
    # blank line after it): "# Title\n\n## Abstract" collapsed to
    # "<b>Title</b><b>Abstract</b>", which reached the user as the single fused
    # word "…PhysiologyAbstract". Leading indent is capped the same way, so a
    # heading cannot start by consuming the previous line's newline either.
    t = re.sub(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", r"<b>\1</b>", t)
    # Bullets before the restore: a code block whose line starts with "- " would
    # otherwise have its source rewritten into a bullet.
    t = re.sub(r"(?m)^[ \t]{2,}[-*+][ \t]+", "    ◦ ", t)      # a nested item keeps its step in
    t = re.sub(r"(?m)^\s{0,3}[-*+]\s+", "• ", t)
    # Restore is iterative: placeholders nest (a heading holds a bold, which holds
    # a link, which holds a code span), so one pass would leave a raw \x00 marker
    # in the delivered text. Bounded and fixed-point, so nothing can spin.
    for _ in range(len(protected) + 1):
        if "\x00" not in t:
            break
        prev = t
        t = re.sub(r"\x00(\d+)\x00",
                   lambda m: protected[int(m.group(1))]
                   if int(m.group(1)) < len(protected) else m.group(0), t)
        if t == prev:
            break
    return t


def _html_to_plain(text: str) -> str:
    """Degrade Telegram HTML to readable plain text.

    Used when Telegram refuses to parse a message: the formatting is cosmetic but
    the answer is not, so we resend unformatted rather than dropping it. Link
    targets are kept inline — in a research report the URLs ARE the content.
    """
    t = re.sub(r"<a\s[^>]*href=\"([^\"]*)\"[^>]*>(.*?)</a>",
               lambda m: (m.group(2) if m.group(2).strip() == m.group(1).strip()
                          else f"{m.group(2)} ({m.group(1)})"),
               text or "", flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"<[^>]+>", "", t)
    return _html_mod.unescape(t)


# Telegram closes/reopens nothing for us: a chunk that ends inside <b> is rejected
# with "Unclosed tag" and the whole message is lost. Tracked so _split_html can
# close them at a chunk boundary and reopen them in the next chunk.
_TAG_RE = re.compile(r"<(/?)(b|i|u|s|code|pre|a|blockquote)(\s[^>]*)?>", re.IGNORECASE)


def _balance_html(chunk: str, carry: list) -> tuple:
    """Close tags left open at the end of `chunk`, reopening `carry` at its start.
    Returns (balanced_chunk, new_carry) where new_carry is the open-tag stack."""
    stack = list(carry)
    for m in _TAG_RE.finditer(chunk):
        closing, name = m.group(1), m.group(2).lower()
        if closing:
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == name:
                    del stack[i]
                    break
        else:
            stack.append((name, m.group(0)))
    prefix = "".join(open_tag for _, open_tag in carry)
    suffix = "".join(f"</{name}>" for name, _ in reversed(stack))
    return prefix + chunk + suffix, stack


# Telegram caps a media caption at 1024 characters — a quarter of the text limit,
# so a caption overflows far more easily than a message does.
_MAX_CAPTION = 1024


def _safe_caption(text: str, limit: int = _MAX_CAPTION) -> str:
    """Trim an HTML caption to Telegram's cap without leaving broken markup.

    A blind `caption[:1024]` is the same defect that once cost whole messages on
    the TEXT path: the cut can land inside `<a href="...">` or between `<b>` and
    `</b>`, and Telegram answers "can't parse entities" and rejects the ENTIRE
    request. Captions are worse than text, because sendMessage has a plain-text
    retry to fall back on and sendDocument/sendVideo did not — a rejected caption
    meant the research report or the rendered clip was never delivered at all.

    Cut back out of any tag we landed inside, then close whatever is still open.
    """
    s = str(text or "")
    if len(s) <= limit:
        return s
    cut = limit
    lt, gt = s.rfind("<", 0, cut), s.rfind(">", 0, cut)
    if lt > gt:                       # the cut is inside a tag — go before it
        cut = lt
    sp = s.rfind(" ", max(0, cut - 120), cut)
    if sp > 0:
        lt2, gt2 = s.rfind("<", 0, sp), s.rfind(">", 0, sp)
        if lt2 <= gt2:                # tags carry spaces too (<a href="a b">)
            cut = sp
    if cut <= 0:
        # A single tag longer than the whole budget: no safe cut exists, so drop
        # the markup rather than emit a half-written tag.
        return _html_to_plain(s)[:limit]
    balanced, _ = _balance_html(s[:cut], [])
    return balanced if len(balanced) <= limit else _html_to_plain(s)[:limit]


# What Telegram actually says when the MARKUP is the problem. Matching loosely on
# "entit" is a trap: HTTP 413 comes back as "Request Entity Too Large", which is
# about the file size and is not fixed by stripping tags — treating it as a markup
# error burns a retry re-uploading a clip that will never fit.
_MARKUP_ERROR_RE = re.compile(
    r"can't parse entities|can't parse|unclosed|unsupported start tag|"
    r"can't find end tag|unexpected end tag|entities?\s+(are\s+)?(too|nested|"
    r"overlap)|wrong (http url|entity)|byte offset",
    re.IGNORECASE)


def _is_markup_error(description: str) -> bool:
    """True only when Telegram rejected a request for its FORMATTING alone.

    Anything else (blocked bot, bad chat_id, oversize upload, network) must not
    trigger a resend: the retry would fail identically, and for a 413 it would
    re-upload the whole file to do it.
    """
    return bool(_MARKUP_ERROR_RE.search(str(description or "")))


def _is_caption_parse_error(resp) -> bool:
    """_is_markup_error for a raw requests.Response.

    The media senders post multipart with `requests` directly instead of going
    through _api_post, so they hold a Response, not the decoded dict.
    """
    try:
        if getattr(resp, "status_code", 0) == 200:
            return False
        return _is_markup_error((resp.json() or {}).get("description", ""))
    except Exception:
        return False


def _split_html(text: str, limit: int = _MAX_TEXT) -> list[str]:
    """Split HTML-formatted text into Telegram-sized chunks WITHOUT breaking tags.

    Reserve headroom for the reopen-prefix/close-suffix _balance_html adds, then
    split on paragraph -> line -> word boundaries (never a blind mid-tag slice:
    that produced "<b" at a chunk edge and Telegram rejected the whole message).
    """
    if len(text) <= limit:
        return [text]

    def _tag_safe(s: str, cut: int) -> int:
        """Move `cut` back to before the enclosing tag if it lands inside one."""
        lt, gt = s.rfind("<", 0, cut), s.rfind(">", 0, cut)
        return lt if lt > gt else cut

    def _hard_split(s: str, budget: int) -> list:
        """Last resort for a single oversized run: break on the latest safe point
        that is not inside a tag."""
        out = []
        while len(s) > budget:
            cut = _tag_safe(s, budget)
            sp = s.rfind(" ", max(0, cut - 120), cut)
            # A space is NOT automatically a safe cut: tags carry spaces of their
            # own (<a href="a b">), and cutting there emitted a half-written tag
            # that Telegram rejected, losing the whole message. Re-check it.
            if sp > 0:
                sp = _tag_safe(s, sp)
                if sp > 0:
                    cut = sp
            if cut <= 0:
                cut = _tag_safe(s, budget)
            if cut <= 0:
                cut = budget          # a single tag longer than the budget: nothing
            out.append(s[:cut])       # safe is possible, keep forward progress
            s = s[cut:]
        if s:
            out.append(s)
        return out

    def _assemble(budget: int) -> list:
        pieces: list = []
        for para in re.split(r"\n{2,}", text):
            if len(para) <= budget:
                pieces.append(para)
                continue
            line_buf = ""
            for line in para.splitlines(keepends=True):
                if len(line) > budget:
                    if line_buf:
                        pieces.append(line_buf); line_buf = ""
                    pieces.extend(_hard_split(line, budget))
                elif len(line_buf) + len(line) > budget:
                    pieces.append(line_buf); line_buf = line
                else:
                    line_buf += line
            if line_buf:
                pieces.append(line_buf)

        chunks, buf = [], ""
        for p in pieces:
            if buf and len(buf) + len(p) + 2 > budget:
                chunks.append(buf.rstrip()); buf = ""
            buf += p + "\n\n"
        if buf.strip():
            chunks.append(buf.rstrip())

        out, carry = [], []
        for c in chunks:
            balanced, carry = _balance_html(c, carry)
            out.append(balanced)
        return out or [text[:budget]]

    # The reopen prefix is not a fixed cost: it is every open tag carried across
    # the boundary, and one <a href="…330 chars…"> alone dwarfs a fixed reserve.
    # Overshooting is not cosmetic — the caller truncates at the limit, which cuts
    # the closing tags off and Telegram rejects the message. So measure the real
    # overflow and re-assemble with a budget that accounts for it.
    budget = max(256, limit - 64)
    out = _assemble(budget)
    for _ in range(6):
        worst = max((len(c) for c in out), default=0)
        if worst <= limit or budget <= 256:
            break
        budget = max(256, budget - (worst - limit) - 32)
        out = _assemble(budget)
    return out
