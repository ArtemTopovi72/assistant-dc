"""A link in a research report must be a page we actually read.

The prompt forbids inventing, guessing or SHORTENING a URL, but a model does it
anyway (several different cars linked to one bare homepage whose collected URL
was a specific article), and a link that looks like a listing and is not one is
worse than no link: the reader clicks it and the price beside it was never on
that page.

So the report is audited. Two repairs, in order: if the domain has exactly one
collected URL, that is what the model mangled -- use it; otherwise drop the link
and keep the text, marked plainly.

Offline: no LLM, no search, no GPU.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dr_assemble import audit_report_links, _norm_url  # noqa: E402

OK = BAD = 0


def check(name, cond, detail=""):
    global OK, BAD
    if cond:
        OK += 1
        print("PASS  " + name)
    else:
        BAD += 1
        print("FAIL  " + name + ((": " + str(detail)) if detail else ""))
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise AssertionError(str(name) + ((": " + str(detail)) if detail else ""))


BRIEFS = [
    {"url": "https://carsplus.online/news/luchshie-elektrokary", "domain": "carsplus.online"},
    {"url": "https://auto.mail.ru/article/101297-luchshie/", "domain": "auto.mail.ru"},
    {"url": "https://x.ru/a", "domain": "x.ru"},
    {"url": "https://x.ru/b", "domain": "x.ru"},
]

print("=" * 66)
print("A URL WE COLLECTED IS LEFT ALONE")
print("=" * 66)

rep = "See [the guide](https://auto.mail.ru/article/101297-luchshie/) for prices."
out, st = audit_report_links(rep, BRIEFS, "ru")
check("an exact match survives untouched", out == rep, out)
check("and it is counted as checked", st["checked"] == 1 and st["repaired"] == 0
      and st["stripped"] == 0, st)

for variant in ("http://auto.mail.ru/article/101297-luchshie/",
                "https://www.auto.mail.ru/article/101297-luchshie",
                "https://AUTO.MAIL.RU/article/101297-luchshie/"):
    o, s = audit_report_links("[x](%s)" % variant, BRIEFS, "en")
    check("scheme/www/case/slash do not make it a different page (%s)"
          % variant.split("//")[1][:22], s["stripped"] == 0 and s["repaired"] == 0,
          (o, s))

print()
print("=" * 66)
print("A SHORTENED URL IS REPAIRED TO THE ONE WE HAVE")
print("=" * 66)

out, st = audit_report_links("[Nissan Leaf](https://carsplus.online)", BRIEFS, "ru")
check("the homepage becomes the collected deep link",
      "carsplus.online/news/luchshie-elektrokary" in out, out)
check("and it is still a link", out.startswith("[Nissan Leaf](https://"), out)
check("counted as a repair", st == {"checked": 1, "repaired": 1, "stripped": 0}, st)

out, st = audit_report_links("[a](https://carsplus.online/made/up/path)", BRIEFS, "ru")
check("an invented path on a known domain is repaired too",
      "luchshie-elektrokary" in out, out)

print()
print("=" * 66)
print("AN UNVERIFIABLE LINK IS DROPPED, NOT LEFT LOOKING REAL")
print("=" * 66)

out, st = audit_report_links("[Ghost](https://never-seen.example/deal)", BRIEFS, "ru")
check("a domain we never read loses its link", "](" not in out, out)
check("the text survives", out.startswith("Ghost"), out)
check("and says why, in the report's language",
      "прямой ссылки в источниках нет" in out, out)
check("counted as stripped", st == {"checked": 1, "repaired": 0, "stripped": 1}, st)

out, _ = audit_report_links("[Ghost](https://never-seen.example/deal)", BRIEFS, "en")
check("an English report says it in English",
      "no direct link in the sources" in out, out)

out, st = audit_report_links("[Zeekr](https://x.ru/zzz)", BRIEFS, "ru")
check("an ambiguous domain is NOT silently repaired to one of two",
      st["stripped"] == 1 and st["repaired"] == 0, (out, st))

print()
print("=" * 66)
print("THE AUDIT NEVER COSTS THE REPORT")
print("=" * 66)

check("an empty report passes through", audit_report_links("", BRIEFS)[0] == "")
check("no briefs means every link is unverifiable, not a crash",
      audit_report_links("[a](https://x.ru/a)", [])[1]["stripped"] == 1)
check("a brief with no url does not break the index",
      audit_report_links("[a](https://x.ru/a)", [{"domain": "x.ru"}])[1]["checked"] == 1)
prose = "No links here at all, just text with a bare https://x.ru/a in it."
check("bare URLs in prose are not touched",
      audit_report_links(prose, BRIEFS)[0] == prose)
check("a relative link is not mistaken for a URL",
      audit_report_links("[a](#section)", BRIEFS)[1]["checked"] == 0)

multi = ("- **[A](https://carsplus.online)** — one\n"
         "- **[B](https://auto.mail.ru/article/101297-luchshie/)** — two\n"
         "- **[C](https://never-seen.example/x)** — three\n")
out, st = audit_report_links(multi, BRIEFS, "ru")
check("a whole list is audited entry by entry",
      st == {"checked": 3, "repaired": 1, "stripped": 1}, st)
check("...and the untouched entry is byte-identical",
      "[B](https://auto.mail.ru/article/101297-luchshie/)" in out, out)

check("_norm_url is stable", _norm_url("HTTPS://WWW.X.ru/a/") == "x.ru/a")

print()
print("=" * 66)
print("A FORM THE ANSWER DID NOT NEED IS NOT PRINTED THREE TIMES")
print("=" * 66)

from dr_assemble import strip_empty_fields  # noqa: E402

# The options prompt describes a fixed row of tour-shaped fields, and a used
# car has no duration, no departure time and no meeting point -- so each car
# came out as "... · не указано · не указано · не указано".
car = "- **[Nissan Leaf](https://x.ru/a)** — Nissan · от 2 500 000 ₽ · не указано · не указано · не указано"
out, n = strip_empty_fields(car)
check("a run of placeholders is removed", "не указано" not in out, out)
check("the real fields survive", "Nissan · от 2 500 000 ₽" in out, out)
check("and they are counted", n == 3, n)

# The run must also be caught in the MIDDLE of the row, not only at its end --
# the two positions are handled by different branches, and a mutation that kept
# every placeholder in the middle passed the end-of-row cases untouched.
mid = "- **[X](https://x.ru/a)** — Seller · не указано · не указано · 2021 год"
out, n = strip_empty_fields(mid)
check("a run in the MIDDLE of the row is removed too",
      "не указано" not in out and "Seller" in out and "2021 год" in out, out)
check("...and counted", n == 2, n)

# A LONE one is informative: on a bookable tour, "price: not stated" tells the
# reader to go and check.
tour = "- **[Tour](https://x.ru/b)** — Operator · not stated in sources · 3 hours"
out, n = strip_empty_fields(tour)
check("a single placeholder is kept", out == tour, out)
check("...and nothing is counted", n == 0, n)

out, n = strip_empty_fields("- **[Bare](https://x.ru/c)** — не указано · не указано")
check("a line of nothing but placeholders loses its dash too",
      out.rstrip().endswith("**"), out)

prose = "plain text line with · a dot in it"
check("prose with a dot is untouched", strip_empty_fields(prose)[0] == prose)
check("a heading with a dot is untouched",
      strip_empty_fields("## Heading · with a dot")[0] == "## Heading · with a dot")
check("an empty report passes through", strip_empty_fields("")[0] == "")
check("English placeholders are recognised too",
      "n/a" not in strip_empty_fields(
          "- **[X](https://x.ru/a)** — Seller · n/a · unknown · N/A")[0].lower())

multi = car + "\n" + tour + "\nplain line"
out, n = strip_empty_fields(multi)
check("a whole list is swept line by line",
      out.count("не указано") == 0 and "not stated in sources" in out
      and out.endswith("plain line"), out)

print()


# ── a translator wrapper is not a source ─────────────────────────────────────
# Example source entry:
#   tr-page.yandex.ru/translate?lang=en-ru&url=https://en.wikipedia.org/wiki/Tannin
# cited as though the translator were the source. It is not: a citation must
# point at the page a reader can check. The wrapper also hides the real host
# from every downstream decision — that Wikipedia article was scored as an
# unknown yandex subdomain rather than as an encyclopaedia.
import dr_urls as _U  # noqa: E402

_WRAPPED = "https://tr-page.yandex.ru/translate?lang=en-ru&url=https://en.wikipedia.org/wiki/Tannin"

check("a translator wrapper is unwrapped to the real page",
      _U.unwrap_url(_WRAPPED) == "https://en.wikipedia.org/wiki/Tannin",
      _U.unwrap_url(_WRAPPED))
check("a plain url is left exactly as it is",
      _U.unwrap_url("https://en.wikipedia.org/wiki/Tannin")
      == "https://en.wikipedia.org/wiki/Tannin")
check("a reader proxy that embeds the target in the PATH is unwrapped too",
      _U.unwrap_url("https://r.jina.ai/https://example.org/a")
      == "https://example.org/a")
check("a wrapper with no target is not mangled",
      _U.unwrap_url("https://tr-page.yandex.ru/translate?lang=en-ru")
      == "https://tr-page.yandex.ru/translate?lang=en-ru")
check("a relative target is refused rather than half-unwrapped",
      _U.unwrap_url("https://tr-page.yandex.ru/translate?url=/wiki/Tannin")
      == "https://tr-page.yandex.ru/translate?url=/wiki/Tannin")
check("garbage in, garbage out — no exception", _U.unwrap_url("not a url") == "not a url")
check("an empty url is survivable", _U.unwrap_url("") == "")

# And the real point: unwrapping changes what the pipeline THINKS the host is.
check("the unwrapped host is the one trust is scored on",
      _U._domain(_U.unwrap_url(_WRAPPED)) == "en.wikipedia.org",
      _U._domain(_U.unwrap_url(_WRAPPED)))
check("...which it was not before", _U._domain(_WRAPPED) != "en.wikipedia.org")

# The function being right is not enough — it has to be WIRED IN. Drive the
# real intake with a wrapped hit and look at what gets stored.
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import dr_collect as _C  # noqa: E402
import dr_progress as _P  # noqa: E402

import types  # noqa: E402
_raw = lambda ctx, q, n, **kw: [
    {"href": _WRAPPED, "title": "Tannin", "body": "tannins in tea"},
    {"href": "https://en.wikipedia.org/wiki/Tannin", "title": "Tannin", "body": "dup"},
]
_srcs = _C.collect_sources(types.SimpleNamespace(is_cancelled=lambda: False, deps=types.SimpleNamespace(raw_search_results=_raw)),
                           ["tannin"], 5, _P._Progress(None),
                           Path(tempfile.mkdtemp()), {"category": "general"})

check("the source that reaches the pipeline is the unwrapped page",
      [s["href"] for s in _srcs] == ["https://en.wikipedia.org/wiki/Tannin"], _srcs)
check("...and the wrapped copy is then recognised as the SAME page, not a second one",
      len(_srcs) == 1, _srcs)
# collect_sources stores the href; the domain is derived downstream from it, so
# that is what this asserts on.
check("...so every later host decision sees the encyclopaedia",
      _srcs and _U._domain(_srcs[0]["href"]) == "en.wikipedia.org", _srcs)


print()
print("=" * 66)
print("A MADE-UP PATH IS REPAIRED TO THE PAGE THAT NAMES THE MODEL")
print("=" * 66)
# A camera matrix linked «Atlas 5GigE IP67» to an invented thinklucid path; thinklucid had two
# pages, so the domain repair gave up and the row lost its link while the visratek page for
# exactly that camera was in the briefs.
_CAMS = [
    {"url": "https://www.visratek.com/en/product/lucid/atlas-5gige-ip67-8-9-mp-color-imx255/",
     "domain": "visratek.com", "title": "Atlas 5GigE IP67 8.9 MP Color (IMX255)"},
    {"url": "https://www.visratek.com/en/product/lucid/triton-8-9-mp-color-imx267/",
     "domain": "visratek.com", "title": "Triton 8.9 MP Color (IMX267)"},
    {"url": "https://thinklucid.com/product/phoenix-8-9-mp-imx267/", "domain": "thinklucid.com",
     "title": "Phoenix 8.9 MP Camera (Sony IMX267)"},
    {"url": "https://thinklucid.com/triton2-gige-machine-vision/", "domain": "thinklucid.com",
     "title": "Triton2 2.5GigE"},
]
out, st = audit_report_links("[Atlas 5GigE IP67](https://thinklucid.com/product/atlas-ip67/)", _CAMS, "ru")
check("the model's name finds its page on another domain", "atlas-5gige-ip67-8-9-mp-color-imx255" in out, out)
out, st = audit_report_links("[Phoenix 8.9 MP](https://thinklucid.com/phoenix)", _CAMS, "ru")
check("and on the same domain among several", "phoenix-8-9-mp-imx267" in out and st["repaired"] == 1, out)
out, st = audit_report_links("[Basler ace](https://thinklucid.com/made-up)", _CAMS, "ru")
check("a name no page carries is still dropped", "](" not in out and st["stripped"] == 1, out)

out, st = audit_report_links("[SONICAM13M-3E](https://cont contrastech.com/SONICAM13M-3E_high...)", _CAMS, "ru")
check("a link with a space or an ellipsis in it never ships as a dead link",
      "](" not in out and "прямой ссылки" in out and st["stripped"] == 1, out)
out, st = audit_report_links("[Atlas 5GigE IP67](https://thinklucid.com/atlas ip67...)", _CAMS, "ru")
check("and is repaired when a page names the model", "atlas-5gige-ip67" in out, out)

_CAMS.append({"url": "https://www.1stvision.com/cameras/models/IDS-Imaging/GV-7090WP-M/C",
              "domain": "1stvision.com", "title": "IDS GV-7090WP-M/C"})
out, st = audit_report_links("[GV-7090WP-C-HQ](https://en.ids-imaging.com/gv-7090wp.html)", _CAMS, "ru")
check("a hyphenated part number matches its URL's pieces", "GV-7090WP-M/C" in out, out)

print()
print("=" * 66)
print("A CANDIDATE FAILING TWO REQUIREMENTS IS RULED OUT")
print("=" * 66)
from dr_assemble import demote_failing_rows  # noqa: E402
_M = ("## Кандидаты\n\n| Модель | 8.3 Мп | 4K | fps |\n| :--- | :--- | :--- | :--- |\n"
      "| [ATP089S](https://a/1) | ✅ 8.9 MP | ✅ 4096x2160 | ✅ 58 |\n"
      "| [ATX081S](https://a/2) | ❌ 8.1 MP | ❌ 2840x2840 | ✅ 136 |\n"
      "| [TRI089S](https://a/3) | ✅ 8.9 MP | ✅ 4096x2160 | ❌ 12.5 |\n\n"
      "## Отсеяно\n\n*   [MV-CH089](https://a/4) — 13 fps\n\n## Рекомендация\n\nx\n")
out, n = demote_failing_rows(_M)
cand = out.split("## Отсеяно")[0]
check("the two-fail row leaves the candidates", "ATX081S" not in cand and n == 1, out)
check("and is ruled out with the values that fail", "ATX081S](https://a/2) — 8.3 Мп: 8.1 MP; 4K: 2840x2840"
      in out.split("## Отсеяно")[1], out)
check("a one-fail row stays a candidate", "TRI089S" in cand, out)
check("the existing ruled-out lines and later sections survive",
      "MV-CH089" in out and "## Рекомендация" in out, out)

from dr_assemble import promote_passing_rows, selection_form_missing  # noqa: E402
_R = ("## Кандидаты\n\n| Модель | 8.3 Мп | fps |\n| :--- | :--- | :--- |\n| [A](https://a/1) | ✅ 8.9 | ✅ 58 |\n\n"
      "## Отсеяно\n\n| Модель | 8.3 Мп | fps |\n| :--- | :--- | :--- |\n"
      "| [TRT089S](https://a/2) | ✅ 8.9 | ✅ 31 |\n| [Phoenix](https://a/3) | ✅ 8.9 | ❌ 13 |\n\n## Уточнить\n\nx\n")
out, n = promote_passing_rows(_R)
check("a ruled-out row with no ❌ goes back to the candidates",
      n == 1 and "TRT089S" in out.split("## Отсеяно")[0] and "Phoenix" in out.split("## Отсеяно")[1], out)
check("a report with renamed sections is caught",
      len(selection_form_missing("## Требования\n## Отчет по кандидатам\n## Резюме\n")) == 5)
check("the six sections in either language pass",
      selection_form_missing("## Требования / Requirements\n## What the numbers force\n## Кандидаты\n"
                             "## Ruled out\n## Рекомендация\n## Уточнить\n") == [])

print("%d passed, %d failed" % (OK, BAD))
sys.exit(1 if BAD else 0)