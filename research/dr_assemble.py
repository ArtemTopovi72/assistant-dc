"""The deterministic parts of the document — everything the model does NOT write.

The synthesised prose is only part of a report. Around it go the run header
(topic, depth, trust mix, quarantine notice), the citation-lineage table built
verbatim from the OpenAlex brief (so the counts are real rather than
model-recalled), the contradiction note, and the tiered source appendix — plus
`_deterministic_digest_report`, the evidence-bound fallback delivered when
synthesis comes back empty twice, so a run never ends in a dead "no output" stub.

All of it is assembled from data, never generated, and all of it is written in
the reader's language.
"""
import logging
import math
import re
from typing import Optional

from dr_lang import _lab, _norm_out_lang, count_label
from dr_math import _repair_latex_artifacts
from dr_extract import _fix_mojibake
from dr_outline import _clean_report_title
from dr_policy import _TRUST_LABEL, _TRUST_ORDER
from dr_relevance import _term_in_any

logger = logging.getLogger("assistant.research")


def _parse_cite_entry(text: str, role: str):
    """Parse one OpenAlex bullet ('"Title" (Year) — Authors — cited_by=N — …')."""
    mt = re.search(r'"([^"]+)"', text)
    title = (mt.group(1) if mt else re.split(r"\s+—\s+|\s+\(", text)[0]).strip()
    my = re.search(r"\((\d{4})\)", text)
    year = my.group(1) if my else ""
    mc = re.search(r"cited_by=(\d+)", text)
    cites = mc.group(1) if mc else "?"
    auth = ""
    for p in [p.strip() for p in text.split(" — ")][1:]:
        if p.lower().startswith(("cited_by", "doi:")):
            break
        if re.match(r"^[A-ZА-Я]", p):
            auth = p
            break
    return (title.replace("|", "/")[:80], year, auth.replace("|", "/")[:60], cites, role)


def _citation_section_from_brief(body: str, terms: Optional[list] = None) -> str:
    """Build a deterministic 'Citation Lineage' table straight from the OpenAlex
    brief, so citation counts + descendants ALWAYS appear when data exists.

    `terms` (topic entity terms) filters out OFF-TOPIC works: OpenAlex sorts by
    citation count and its descendant lookup can return loosely-related giants
    (quantum-computing / hyperspectral / hotel-review papers on an "Adam optimizer"
    run). A row is kept only if its title names a topic term (word-boundary). If
    fewer than 2 on-topic rows survive, the whole table is suppressed — a misleading
    citation lineage is worse than none."""
    rows, role = [], "related"
    for ln in body.splitlines():
        s = ln.strip()
        low = s.lower()
        if low.startswith("most-influential"):
            role = "influential"; continue
        if low.startswith("likely seminal"):
            payload = s.split(":", 1)[1].strip() if ":" in s else s
            rows.append(_parse_cite_entry(payload, "seminal")); continue
        if low.startswith("principal descendants"):
            role = "descendant"; continue
        if low.startswith("related high-impact"):
            role = "related"; continue
        if s.startswith("- "):
            rows.append(_parse_cite_entry(s[2:], role))
    rows = [r for r in rows if r and r[0]]
    if not rows:
        return ""
    # Relevance gate: keep a seminal row always (it anchors the lineage), but drop
    # influential/related/descendant rows whose title names no topic term.
    if terms:
        filtered = [r for r in rows
                    if r[4] == "seminal" or _term_in_any(r[0].lower(), terms)]
        on_topic = [r for r in filtered if r[4] != "seminal"]
        if len(on_topic) < 2:
            logger.info("citation lineage suppressed — only %d on-topic work(s) "
                        "from OpenAlex (off-topic citation graph)", len(on_topic))
            return ""
        rows = filtered
    # Keep the most informative role per paper (seminal > descendant > influential).
    _prio = {"seminal": 0, "descendant": 1, "influential": 2, "related": 3}
    best = {}
    order = []
    for title, year, auth, cites, role in rows:
        key = title.lower()[:50]
        if key not in best:
            order.append(key)
        if key not in best or _prio.get(role, 9) < _prio.get(best[key][4], 9):
            best[key] = (title, year, auth, cites, role)
    out = ["## Citation Lineage (OpenAlex)", "",
           "| Paper | Year | Authors | Citations | Role |",
           "| :--- | :--- | :--- | :--- | :--- |"]
    for key in order:
        title, year, auth, cites, role = best[key]
        out.append(f"| {title} | {year} | {auth or '—'} | {cites} | {role} |")
    out.append("\n*Citation counts and works from OpenAlex (api.openalex.org).*")
    return "\n".join(out)


def _append_once(report: str, block: str) -> str:
    """Append `block` unless its heading is already present in `report`.

    The sources appendix is added at several different exit points (normal close,
    survey close, forced close) and on some paths two of them ran — the reader got
    "Список источников" twice at the end of the same document.
    """
    if not block:
        return report
    stripped = block.strip()
    head = stripped.splitlines()[0].strip() if stripped else ""
    if head and head in report:
        logger.debug("skipping a duplicate %r block", head[:40])
        return report
    return report + block


_MD_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
_MD_BROKEN_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\n]*(?:\s|…|\.\.\.)[^)\n]*)\)")

_LINK_UNVERIFIED = {
    "ru": "прямой ссылки в источниках нет",
    "en": "no direct link in the sources",
}


def _norm_url(url: str) -> str:
    """Compare URLs the way a reader would: scheme and case and a trailing
    slash do not make it a different page."""
    u = (url or "").strip().rstrip("/")
    u = re.sub(r"^https?://", "", u, flags=re.I)
    return re.sub(r"^www\.", "", u, flags=re.I).lower()


_PLACEHOLDERS = ("не указано", "не указан", "не указана", "нет данных",
                 "not stated in sources", "not stated", "not specified",
                 "n/a", "unknown")


def _is_placeholder(part: str) -> bool:
    s = part.strip().strip("*_ ").lower()
    return bool(s) and any(s == ph or s.startswith(ph) for ph in _PLACEHOLDERS)


def strip_empty_fields(report: str) -> tuple:
    """Drop RUNS of "not stated" from the ` · ` meta line of an option.

    The options list is described to the model as a fixed row of fields, and it
    fills every one. Measured: an answer about buying a used electric car listed
    each car as "Nissan · от 2 500 000 ₽ · не указано · не указано · не
    указано" -- the three tour-shaped fields (duration, departure time,
    departure point) that a car cannot have.

    A SINGLE placeholder is informative: on a bookable tour, "price: not stated"
    tells the reader to check. Two or more in a row are a form the answer did
    not need, so a run of them is removed and a lone one is kept.

    Returns (report, fields_removed).
    """
    if not report:
        return report, 0
    removed = 0
    out_lines = []
    for line in report.splitlines():
        if "·" not in line or not line.lstrip().startswith(("-", "*", "•")):
            out_lines.append(line)
            continue
        head, sep, meta = line.partition("—")
        if not sep:
            head, sep, meta = line.partition("-")
            if not sep or "·" not in meta:
                out_lines.append(line)
                continue
        parts = meta.split("·")
        keep, run = [], []
        for part in parts:
            if _is_placeholder(part):
                run.append(part)
                continue
            if len(run) == 1:
                keep.append(run[0])
            else:
                removed += len(run)
            run = []
            keep.append(part)
        if len(run) == 1:
            keep.append(run[0])
        else:
            removed += len(run)
        if not [k for k in keep if k.strip()]:
            out_lines.append(head.rstrip().rstrip("—-").rstrip())
            continue
        out_lines.append(head + sep + "·".join(keep).rstrip())
    return "\n".join(out_lines), removed


def _name_tokens(s: str) -> set:
    # Split on every separator: «GV-7090WP-C-HQ» in the text is gv/7090wp/hq in its URL.
    return {t for t in re.findall(r"[a-z0-9]+", s.lower())
            if len(t) >= 3 or (len(t) == 2 and any(c.isdigit() for c in t))} - {
        "www", "com", "https", "http", "html", "product", "products"}


def _page_named(text: str, url: str, briefs: list) -> Optional[str]:
    """The one collected page whose URL or title names what the link names. A selection
    report linked «Atlas 5GigE IP67» to a thinklucid path it made up; thinklucid had two
    collected pages, so the domain repair gave up, while the visratek page
    «atlas-5gige-ip67-8-9-mp-color-imx255» was the model's own evidence for that row."""
    want = _name_tokens(text) | {t for t in _name_tokens(url.split("//", 1)[-1].split("/", 1)[-1])
                                 if any(c.isdigit() for c in t)}
    if not want:
        return None
    pages = [(u, _name_tokens(u + " " + (b.get("title") or "")))
             for b in briefs or [] for u in [(b.get("url") or "").strip()] if u]
    df = {}
    for _, have in pages:
        for t in have & want:
            df[t] = df.get(t, 0) + 1
    # A word on one page names it; one on every page («lucid», «camera») does not; a
    # part number («7090wp») counts double.
    weight = {t: (2.0 if re.search(r"\d", t) and re.search(r"[a-z]", t) else 1.0) / n
              for t, n in df.items()}
    scored = sorted(((sum(weight[t] for t in have & want), u) for u, have in pages),
                    key=lambda x: -x[0])
    if scored and scored[0][0] >= 1.0 and (len(scored) == 1 or scored[0][0] > 1.5 * scored[1][0]):
        return scored[0][1]
    return None


def audit_report_links(report: str, briefs: list, out_lang: str = "en") -> tuple:
    """Every link in the report must be a URL we actually collected.

    The practical-options prompt already forbids inventing, guessing or
    SHORTENING a URL. Measured on a real run, the model did it anyway: three
    different used cars were each linked to `https://carsplus.online` -- the
    bare homepage of a source whose actual URL was
    `carsplus.online/news/luchshie-elektrokary` -- in a report that said in its
    own text that the sources carried no purchase pages. A link that looks like
    a listing and is not one is worse than no link: the reader clicks it, and
    the price and model beside it were never on that page.

    So the prompt asks and this checks. Two repairs, in order:
      * exactly one collected URL on that domain -> use it (the model shortened
        or mangled a real link, which is the common case);
      * otherwise -> drop the link and keep the text, marked plainly.

    Returns (report, {"checked", "repaired", "stripped"}).
    """
    if not report:
        return report, {"checked": 0, "repaired": 0, "stripped": 0}
    lang = "ru" if str(out_lang).lower().startswith("ru") else "en"
    known, by_domain = {}, {}
    for b in briefs or []:
        url = (b.get("url") or "").strip()
        if not url:
            continue
        known[_norm_url(url)] = url
        by_domain.setdefault((b.get("domain") or _norm_url(url).split("/")[0]).lower(),
                             set()).add(url)
    stats = {"checked": 0, "repaired": 0, "stripped": 0}

    def _fix(m):
        text, url = m.group(1), m.group(2)
        stats["checked"] += 1
        if _norm_url(url) in known:
            return m.group(0)
        dom = _norm_url(url).split("/")[0]
        same = by_domain.get(dom) or set()
        if len(same) == 1:
            stats["repaired"] += 1
            return "[%s](%s)" % (text, next(iter(same)))
        named = _page_named(text, url, briefs)
        if named:
            stats["repaired"] += 1
            return "[%s](%s)" % (text, named)
        stats["stripped"] += 1
        return "%s (%s)" % (text, _LINK_UNVERIFIED[lang])

    def _fix_broken(m):
        # «[SONICAM13M-3E](https://cont contrastech.com/SONICAM…)»: a space or an ellipsis
        # inside the parentheses, so _MD_LINK never saw it and it shipped as a dead link.
        text = m.group(1)
        stats["checked"] += 1
        named = _page_named(text, m.group(2).replace(" ", ""), briefs)
        if named:
            stats["repaired"] += 1
            return "[%s](%s)" % (text, named)
        stats["stripped"] += 1
        return "%s (%s)" % (text, _LINK_UNVERIFIED[lang])

    report = _MD_LINK.sub(_fix, report)
    return _MD_BROKEN_LINK.sub(_fix_broken, report), stats


_EMPTY_CELL = re.compile(r"^(?:—|-|–|n/?a|)$", re.I)


def drop_empty_columns(report: str) -> str:
    """Remove a table column that no row fills: every matrix shipped a «Цена» column of
    dashes, which reads as a field the report forgot rather than a fact it lacked. A
    requirement column of «❓ не указано» stays: that gap is the answer."""
    lines = report.split("\n")
    i = 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            j += 1
        rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines[i:j]]
        width = len(rows[0])
        if j - i >= 3 and all(len(r) == width for r in rows):
            body = rows[2:]
            keep = [k for k in range(width)
                    if any(not _EMPTY_CELL.match(r[k].strip(" *")) for r in body)]
            if 0 < len(keep) < width:
                lines[i:j] = ["| " + " | ".join(r[k] for k in keep) + " |" for r in rows]
        i = j
    return "\n".join(lines)


def demote_failing_rows(report: str) -> tuple:
    """Move candidate rows that fail two or more requirements to the ruled-out list.

    Told in both the writing and the checking prompt, the model still kept two 8.1 MP
    2840x2840 cameras (❌ 8.3 MP, ❌ 4K) as candidates with a footnote saying the rule
    allowed it. Returns (report, moved)."""
    lines = report.split("\n")
    cand = next((i for i, ln in enumerate(lines)
                 if re.match(r"##\s*(Кандидаты|Candidates)", ln.strip(), re.I)), None)
    out_at = next((i for i, ln in enumerate(lines)
                   if re.match(r"##\s*(Отсеяно|Ruled out)", ln.strip(), re.I)), None)
    if cand is None or out_at is None or out_at < cand:
        return report, 0
    header, moved, keep = None, [], []
    for i in range(cand + 1, out_at):
        ln = lines[i]
        if not ln.lstrip().startswith("|"):
            keep.append(ln)
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if header is None:
            header = cells
        elif ln.count("❌") >= 2 and not set("".join(cells)) <= set(":- "):
            fails = [f"{h}: {c.replace('❌', '').strip()}"
                     for h, c in zip(header, cells) if "❌" in c]
            moved.append(f"*   {cells[0]} — " + "; ".join(fails))
            continue
        keep.append(ln)
    if not moved:
        return report, 0
    tail = lines[out_at + 1:]
    while tail and not tail[0].strip():
        tail = tail[1:]
    return "\n".join(lines[:cand + 1] + keep + [lines[out_at], ""] + moved + tail), len(moved)


def promote_passing_rows(report: str) -> tuple:
    """Rows in a ruled-out TABLE with no ❌ go back to the candidates table: run 10 listed
    TRT089S (every cell ✅ or ❓) among the rejects. Only when both tables have the same
    columns. Returns (report, moved)."""
    lines = report.split("\n")

    def section(pat):
        start = next((i for i, ln in enumerate(lines) if re.match(pat, ln.strip(), re.I)), None)
        if start is None:
            return None, None
        end = next((j for j in range(start + 1, len(lines)) if lines[j].startswith("## ")),
                   len(lines))
        return start, end
    cs, ce = section(r"##\s*(Кандидаты|Candidates)")
    rs, re_ = section(r"##\s*(Отсеяно|Ruled out)")
    if cs is None or rs is None or rs < cs:
        return report, 0
    ctab = [i for i in range(cs + 1, ce) if lines[i].lstrip().startswith("|")]
    rtab = [i for i in range(rs + 1, re_) if lines[i].lstrip().startswith("|")]
    if len(ctab) < 2 or len(rtab) < 3 or lines[ctab[0]].count("|") != lines[rtab[0]].count("|"):
        return report, 0
    up = [i for i in rtab[2:] if "❌" not in lines[i]]
    if not up:
        return report, 0
    rows = [lines[i] for i in up]
    out = [ln for i, ln in enumerate(lines) if i not in set(up)]
    at = ctab[-1] + 1          # ruled-out comes after the candidates, so this index holds
    return "\n".join(out[:at] + rows + out[at:]), len(rows)


def drop_false_none_claim(report: str) -> tuple:
    """The line above the candidates table exists only to say no model meets everything.
    Run 18 wrote it over seven rows without a single ❌. Dropped when a row fails nothing.
    Returns (report, dropped)."""
    lines = report.split("\n")
    cs = next((i for i, ln in enumerate(lines) if re.match(r"##\s*(Кандидаты|Candidates)", ln.strip(), re.I)), None)
    if cs is None:
        return report, False
    tab = next((i for i in range(cs + 1, len(lines))
                if lines[i].lstrip().startswith("|") or lines[i].startswith("## ")), None)
    if tab is None or not lines[tab].lstrip().startswith("|"):
        return report, False
    rows = []
    for ln in lines[tab + 2:]:
        if not ln.lstrip().startswith("|"):
            break
        rows.append(ln)
    above = [i for i in range(cs + 1, tab) if lines[i].strip()]
    if not above or not any("❌" not in r for r in rows):
        return report, False
    return "\n".join(ln for i, ln in enumerate(lines) if i not in set(above)), True


_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def rank_by_closeness(report: str) -> str:
    """Candidates sorted by ❌ count, then by distance to the stated numbers. Told that a
    stated resolution is a target, the writer still put a 19.6 MP camera above three
    8.9 MP ones for an 8.3 MP spec. A column whose header carries a number is a target
    (log distance, rounded so near-equal rows keep the writer's order); «≥ 25 fps» is a
    floor and costs nothing above it; a header number a cell is ten times off («4K» vs
    4096x2160) is a label, not a target."""
    lines = report.split("\n")
    cs = next((i for i, ln in enumerate(lines) if re.match(r"##\s*(Кандидаты|Candidates)", ln.strip(), re.I)), None)
    if cs is None:
        return report
    head = next((i for i in range(cs + 1, len(lines))
                 if lines[i].lstrip().startswith("|") or lines[i].startswith("## ")), None)
    if head is None or not lines[head].lstrip().startswith("|"):
        return report
    end = head + 2
    while end < len(lines) and lines[end].lstrip().startswith("|"):
        end += 1
    rows = lines[head + 2:end]
    cols = [c.strip() for c in lines[head].strip().strip("|").split("|")]
    targets = {}
    for j, c in enumerate(cols):
        m = _NUM.search(c)
        if m and not c.startswith(("≥", ">=")):
            targets[j] = float(m.group().replace(",", "."))

    def cost(row):
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        d = 0.0
        for j, t in targets.items():
            m = _NUM.search(cells[j]) if j < len(cells) else None
            v = float(m.group().replace(",", ".")) if m else 0
            if t > 0 and v > 0 and 0.1 < v / t < 10:
                d += abs(math.log(v / t))
        return (row.count("❌"), round(d, 1))
    ranked = sorted(rows, key=cost)
    return "\n".join(lines[:head + 2] + ranked + lines[end:])


_SELECTION_SECTIONS = (("Требования", "Requirements"), ("Что из этого следует", "What the numbers force"),
                       ("Кандидаты", "Candidates"), ("Отсеяно", "Ruled out"),
                       ("Рекомендация", "Recommendation"), ("Уточнить", "To confirm"))


def selection_form_missing(report: str) -> list:
    """The required matrix sections a report lacks. Handed 150 model lines, the writer
    renamed and dropped half of them («Отчет по кандидатам», «Резюме»)."""
    heads = [ln.strip().lower() for ln in (report or "").splitlines() if ln.startswith("## ")]
    return [f"{ru} / {en}" for ru, en in _SELECTION_SECTIONS
            if not any(ru.lower() in h or en.lower() in h for h in heads)]


def _gost_entries(by_tier: dict) -> list:
    """One numbered list, ГОСТ Р 7.0.100-2018 for an online resource:
    «Заглавие. — Текст : электронный // Сайт. — URL: … (дата обращения: ДД.ММ.ГГГГ).»
    Most authoritative first; a Russian reader asked for this form over the tier list."""
    from datetime import date
    today = date.today().strftime("%d.%m.%Y")
    out, n = [], 0
    for tier in _TRUST_ORDER:
        for b in sorted(by_tier.get(tier) or [], key=lambda b: (b.get("domain") or "", b["url"])):
            n += 1
            title = re.sub(r"\s+", " ", (b.get("title") or "").strip())
            if len(title) > 100:     # a search engine glued several results' titles: «…CameraTriton10 8.1 MP…»
                title = re.split(r"(?<=[a-zа-я])(?=[A-ZА-Я][a-zа-я0-9])", title)[0]
            title = title.rstrip(" .…") or b["url"]
            out.append(f"{n}. {title}. — Текст : электронный // {b.get('domain') or ''}. — "
                       f"URL: {b['url']} (дата обращения: {today}).")
    return out


def _sources_appendix(briefs: list, out_lang: str = "en") -> str:
    """Source appendix grouped by trust tier (most → least authoritative), then
    by domain, so the reader sees at a glance what the report actually rests on."""
    by_tier: dict = {}
    for b in briefs:
        by_tier.setdefault(b.get("trust", "COMMUNITY"), []).append(b)

    lines = ["", f"# {_lab(out_lang, 'appendix')}", ""]
    if _norm_out_lang(out_lang) == "ru":
        return "\n".join(lines + _gost_entries(by_tier)) + "\n"
    for tier in _TRUST_ORDER:
        bucket = by_tier.get(tier)
        if not bucket:
            continue
        lines += [f"## {_TRUST_LABEL[tier]} — {len(bucket)}", ""]
        by_dom: dict = {}
        for b in bucket:
            by_dom.setdefault(b["domain"], []).append(b)
        # Blank lines around each list: without them the PDF's Markdown ran a domain and
        # its pages together into one paragraph.
        for dom in sorted(by_dom):
            lines += [f"**{dom}**", ""]
            for b in by_dom[dom]:
                title = b.get("title") or b["url"]
                lines.append(f"- {title} — {b['url']}")
            lines.append("")
    return "\n".join(lines)


def _report_header(topic: str, briefs: list, stats: dict, depth: str,
                   core_topic: Optional[str] = None, out_lang: str = "en",
                   title: Optional[str] = None) -> str:
    """Front-matter block prepended to every report: title, run metadata, and a
    one-line source-quality breakdown so credibility is visible up top."""
    from datetime import date
    counts = {t: 0 for t in _TRUST_ORDER}
    for b in briefs:
        counts[b.get("trust", "COMMUNITY")] = counts.get(b.get("trust", "COMMUNITY"), 0) + 1
    mix = " · ".join(f"{counts[t]} {t.lower()}" for t in _TRUST_ORDER if counts[t])
    L = lambda k: _lab(out_lang, k)
    extra = ""
    q = stats.get("quarantined")
    if q:
        extra = " · " + ", ".join(f"{v} {k}" for k, v in q.items()) + " " + L("filtered")
    # Title from the DISTILLED subject (clean), not the raw request which may be a
    # long multilingual instruction; the raw request stays as the *Query:* subtitle.
    # For a non-English document the H1 must be in the DOCUMENT's language. The
    # distilled `core_topic` comes out of the English-first pipeline, so a Russian
    # report was headed "# Comparison of K-700 and K-744 Kirovets tractors" above
    # an otherwise Russian text. The user's own wording is already in the right
    # language and is a perfectly good title.
    if title:
        pass
    elif _norm_out_lang(out_lang) != "en" and (topic or "").strip():
        title = _clean_report_title(topic)
    else:
        title = _clean_report_title(core_topic or topic)
    # Keep the full query as a small subtitle ONLY when we actually shortened it, so
    # the exact search spec stays traceable without polluting the H1.
    query_line = (f"*{L('query')}: {topic.strip()}*\n\n"
                  if title != topic.strip() else "")
    return (
        f"# {title}\n\n"
        f"{query_line}"
        f"*{L('generated')} {date.today().isoformat()} · {L('depth')}: {depth} · "
        f"{count_label(stats.get('sources', 0), L('sources_searched'), out_lang)}, "
        f"{count_label(stats.get('pages', 0), L('pages_read'), out_lang)}, "
        f"{count_label(len(briefs), L('briefs_used'), out_lang)}{extra}*  \n"
        f"*{L('source_quality')}: {mix or 'n/a'}*\n"
    )


def provenance_footer(briefs: list, stats: dict, depth: str,
                      out_lang: str = "en") -> str:
    """How the report was made, in the run's own counters, for the SURVEY path.

    Survey mode writes its own title, abstract and table of contents, so the
    run header is deliberately not prepended -- which left the primary product
    with no provenance at all. Measured on a live run: a 34,000-character
    document with no statement of the depth it ran at, how many sources were
    searched, or how many pages were actually read. A document that looks like
    a paper and cannot be audited is the worse half of both.

    The last line is the one that costs something to say: when reflection asked
    for another pass and the depth did not allow one, the report says so
    instead of ending as though nothing was left open.
    """
    from datetime import date
    L = lambda k: _lab(out_lang, k)
    counts = {}
    for b in (briefs or []):
        t = b.get("trust", "COMMUNITY")
        counts[t] = counts.get(t, 0) + 1
    mix = " · ".join(f"{counts[t]} {t.lower()}" for t in _TRUST_ORDER if counts.get(t))
    stats = stats or {}
    parts = [f"*{L('generated')} {date.today().isoformat()} · {L('depth')}: {depth} · "
             f"{count_label(stats.get('sources', 0), L('sources_searched'), out_lang)}, "
             f"{count_label(stats.get('pages', 0), L('pages_read'), out_lang)}, "
             f"{count_label(len(briefs or []), L('briefs_used'), out_lang)}*"]
    q = stats.get("quarantined")
    if q:
        parts.append("*" + ", ".join(f"{v} {k}" for k, v in q.items())
                     + " " + L("filtered") + "*")
    if mix:
        parts.append(f"*{L('source_quality')}: {mix}*")
    refl = stats.get("reflection") or {}
    if refl.get("needs_second_pass") and not stats.get("reflection_iterations"):
        parts.append(f"*{L('gap_open')}*")
    head = "\n\n## " + L("how_made") + "\n\n"
    return head + "  \n".join(parts) + "\n"


def _contradiction_note(sig: dict, out_lang: str = "en") -> str:
    """Render the contradiction-pass result as a short report block. Always shown
    (even 'none found') so the reader knows disconfirmation was actively attempted.

    Localised: heading and body follow `out_lang`, so a Russian document never
    carries an English section.
    """
    if not sig:
        return ""
    ru = _norm_out_lang(out_lang) == "ru"
    head = "## Проверка на противоречия" if ru else "## Contradiction Check"
    if not sig.get("has_contradictions"):
        body = ("_Выполнен отдельный проход на опровержение; убедительных "
                "противоречащих данных не найдено._") if ru else (
               "_Ran a dedicated disconfirmation pass; no credible contradicting "
               "evidence was found._")
        return f"{head}\n{body}\n\n"
    doms = ", ".join(sig.get("domains", [])[:6]) or "—"
    if ru:
        return (f"{head}\nСила опровержения: **{sig['strength']}** "
                f"({sig['count']} противоречащих источников, {sig['strong_sources']} "
                f"сильных). Источники: {doms}. Они учтены наравне с подтверждающими "
                f"данными выше, а не отброшены молча.\n\n")
    return (f"{head}\nDisconfirmation strength: **{sig['strength']}** "
            f"({sig['count']} contradicting source(s), {sig['strong_sources']} strong). "
            f"Sources: {doms}. These are weighed against the confirming evidence above "
            f"and not silently dropped.\n\n")


def _deterministic_digest_report(topic: str, cluster_digests: list, briefs: list) -> str:
    """Build a readable, evidence-bound summary WITHOUT the LLM, from the cluster
    digests + briefs. Used when the synthesis reduce fails — guarantees the user
    still gets a structured, cited answer instead of a dead stub."""
    def _clean(txt: str, limit: int) -> str:
        # Briefs/claims already arrive as bullet lists ("- fact"); strip leading list
        # markers so the digest's own "- " doesn't produce "- - fact", collapse
        # whitespace, and repair any encoding mojibake.
        t = _fix_mojibake((txt or "").strip().replace("\n", " ")) or ""
        t = re.sub(r"^\s*[-*•]\s+", "", t)
        t = re.sub(r"\s{2,}", " ", t).strip()
        return t[:limit].rstrip()

    out = [f"## Key findings (assembled from {len(briefs)} sources)\n",
           "_Compiled directly from sources without model synthesis — each point is cited._\n"]
    if cluster_digests:
        for d in cluster_digests:
            tier = "major" if d.get("major") else "minor"
            ents = ", ".join(d.get("entities", [])[:5]) or "—"
            out.append(f"\n### {d.get('label', 'thread')} ({tier} thread, "
                       f"{d.get('size', 0)} source(s), weight {d.get('evidence_weight', 0)})")
            out.append(f"Key entities: {ents}\n")
            for rc in d.get("representative_claims", [])[:2]:
                txt = _clean(rc.get("text", ""), 320)
                if txt:
                    out.append(f"- {txt} _( {rc.get('domain', '')}, {rc.get('trust', '')} )_")
    else:
        for b in briefs[:10]:
            txt = _clean(b.get("brief", ""), 280)
            if txt:
                out.append(f"- {txt} _( {b.get('domain', '')}, {b.get('trust', '')} )_")
    out.append("\n")
    return "\n".join(out) + "\n"


def _build_equations_section(briefs: list) -> str:
    """Build a 'Key equations (verbatim from sources)' Markdown section from the
    raw equation blocks preserved on each brief. Deduped across sources, attributed
    to the source domain. NO count cap — the user wants ALL formulas; dedup alone
    bounds it. Returns '' when no equations were preserved."""
    seen, rows = set(), []
    for b in briefs or []:
        for eq in b.get("equations", []) or []:
            eq = _repair_latex_artifacts(eq)   # strip ar5iv layout primitives (\vskip/\cr)
            key = re.sub(r"\s+", "", eq)
            if key in seen:
                continue
            seen.add(key)
            rows.append((eq, b.get("domain", "?"), b.get("trust", "")))
    if not rows:
        return ""
    lines = ["## Key equations (verbatim from sources)",
             "_Copied unchanged from the source text; not paraphrased or "
             "regenerated by the model._\n"]
    for eq, dom, trust in rows:
        tag = f"— _{dom}{(', ' + trust) if trust else ''}_"
        # Equation on its own block (blank lines around it) with the attribution on a
        # SEPARATE line, so a math renderer treats the LaTeX as one clean display block
        # and markdown doesn't fold the source tag into the formula.
        lines.append(f"\n{eq.strip()}\n\n{tag}\n")
    return "\n".join(lines)
