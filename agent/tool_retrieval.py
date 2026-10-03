"""Dynamic tool retrieval: send the model the schemas this turn can use.

Mechanic #1 of the catalog, and on this deployment the highest-value one by a
wide margin. The full set is 15 schemas, about 6800 tokens, which was the ENTIRE
per-request budget of a model loaded at 12288 with parallel 2 -- so a tool-bearing
round did not merely cost more, it was rejected outright and the turn was lost.
Doubling the context bought room at the price of VRAM shared with another
application; narrowing the payload is the fix that costs nothing.

Lexical BM25 over the schemas is not an option here: the descriptions are
English and the users write Russian, so term overlap is zero on exactly the
turns that matter. Which tools a message needs is read by the model
(agent/intent.py `wants`) -- it replaced a per-tool regex cue table, which a
paraphrase or a typo slipped past and an unrelated word tripped.

The safety rule is that dropping a needed tool is far worse than sending a few
extra: a missing schema is a capability the model cannot use no matter how well
it routes. So retrieval only ever NARROWS from a matched set, and when nothing
matches it hands back the full list untouched.
"""
import os
import config as _cfg_env   # env_int/env_float: a bad .env value falls back, never crashes the import


# Cheap, broadly useful, and the ones a turn most often turns out to need after
# the first round. Kept whenever there is room rather than matched for.
_CORE = ("search", "calculate", "remember_fact")

# The file tools work as a KIT, not individually: unpacking an archive is
# useless without listing it, reading it and editing it, and the model cannot
# ask for a schema it was not sent. Measured while wiring this up -- "распакуй
# мод и добавь блоки в тег" retrieved unpack_archive alone, which is an agent
# that can open the box and then do nothing with the contents. They are small
# (about 90 tokens each), so the whole kit costs less than one image schema.
# open_image rides with the kit for the same reason inspect_image rides with
# the editors: a picture in the working folder is unreadable by read_file, and
# the model cannot call a schema it was never sent -- it would tell the user to
# re-send a photo that is already sitting in their folder.
_CODE_KIT = ("list_files", "read_file", "open_image", "edit_file", "write_file",
             "search_files", "find_content", "dedupe_photos", "delete_path", "unpack_archive",
             "pack_archive", "code_outline", "update_plan", "undo_edit")

# Verification is a prerequisite of honest image work (see the escalation guard
# in graph_personality): whenever an editing tool survives retrieval, the tool
# that CHECKS it has to survive too, or the agent is structurally unable to
# verify its own output.
# The part of the kit a non-empty working folder earns on EVERY turn: enough
# to notice and open what is there. payload_audit 2026-09-23: pinning the
# whole kit on every turn cost ~3700 tokens and 14 schemas on "какая погода".
# The rest arrives as soon as the words ask for it or the model starts using
# the folder (any kit tool already called this turn).
_LOOK_KIT = ("list_files", "read_file", "open_image", "unpack_archive")

_EDITORS = frozenset({"inpaint_image", "redraw_image", "transfer_image",
                      "fix_hands", "fix_artifact", "generate_image"})


def _name(schema):
    return schema.get("function", {}).get("name", "")


def _embed_on():
    # On by default since 2026-09-24: with the fast path no longer swallowing
    # claimed actions, para 14/30 -> 18/30 with it, gate/route unchanged.
    v = os.getenv("TOOLS_EMBED")
    if v is not None:
        return v.strip() == "1"
    return not os.getenv("F5_TEST_RUN")


_EMBED_CACHE = {}      # tuple of tool names -> (names, unit vectors)
_EMBEDDER = None


def _unit(v):
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def _embed_pick(text, schemas, k=None, floor=None):
    """Names of the tools whose descriptions are nearest to `text`.

    Empty set on any failure (no embedding model served): the caller then
    falls back to the core set exactly as without this option.
    """
    global _EMBEDDER
    k = k or _cfg_env.env_int("TOOLS_EMBED_K", 4)
    floor = _cfg_env.env_float("TOOLS_EMBED_FLOOR", 0.30) if floor is None else floor
    try:
        if _EMBEDDER is None:
            import knowledge
            _EMBEDDER = knowledge.Embedder(timeout=10.0)
        names = tuple(_name(s) for s in schemas)
        if names not in _EMBED_CACHE:
            docs = [f"{_name(s)}: {s.get('function', {}).get('description', '')[:600]}"
                    for s in schemas]
            vecs = _EMBEDDER.embed(docs)
            if not vecs:
                return set()
            _EMBED_CACHE[names] = [_unit(v) for v in vecs]
        q = _EMBEDDER.embed([text])
        if not q:
            return set()
        q = _unit(q[0])
        sims = sorted(((sum(a * b for a, b in zip(q, v)), n)
                       for n, v in zip(names, _EMBED_CACHE[names])), reverse=True)
        return {n for s, n in sims[:k] if s >= floor}
    except Exception:
        return set()


def select_tools(text, schemas, *, always=(), limit=12, wants=None):
    """Retrieval (see _select_tools), then a STABLE order: the core tools first,
    always in the same order, the per-turn picks after them. The tool block sits
    right after the system prompt, so a block that starts the same way every turn
    lets the server reuse the cached prefix instead of recomputing the history."""
    out = _select_tools(text, schemas, always=always, limit=limit, wants=wants)
    rank = {n: i for i, n in enumerate(_CORE)}
    return sorted(out, key=lambda s: rank.get(_name(s), len(rank)))   # stable sort keeps the rest in order


def _select_tools(text, schemas, *, always=(), limit=12, wants=None):
    """Narrow `schemas` to what this turn plausibly needs.

    `always` are names that must survive whatever the cues say -- the caller
    passes the tools already used this turn, so a retry is never made impossible
    by the retrieval step.

    Returns the full list unchanged when nothing matches: no evidence is a
    reason to send everything, not a reason to guess.
    """
    text = text or ""
    # The tools the model's read of the message picked (agent/intent.py);
    # None when that read failed -- then no evidence, the core set below.
    matched = set(wants or ())
    keep = set(always) | matched
    if not matched:
        if text.strip() and _embed_on():
            # Paraphrase fallback (TOOLS_EMBED=1): no cue word, so add the
            # tools whose descriptions the request resembles. Added one by
            # one -- a weak hit on update_plan must not pull in the 14-tool
            # file kit.
            emb_pick = _embed_pick(text, schemas)
        else:
            emb_pick = set()
        # payload_audit 2026-09-23: "привет" went out with 13 schemas (32 with
        # a working folder), ~4-10k tokens of tools for a greeting. With no
        # cue the core set plus the pinned tools is sent; TOOLS_NOMATCH=all
        # restores the old send-everything fallback for A/B runs.
        if os.getenv("TOOLS_NOMATCH", "core").strip().lower() == "all":
            return list(schemas)
        keep |= set(_CORE)
        if keep & set(_CODE_KIT) - set(_LOOK_KIT):
            keep.update(_CODE_KIT)
        if emb_pick:
            keep |= emb_pick
        return [s for s in schemas if _name(s) in keep] or list(schemas)

    if keep & _EDITORS:
        keep.add("inspect_image")
    if (matched | (set(always) - set(_LOOK_KIT))) & (set(_CODE_KIT) | {"run_code", "run_tests"}):
        keep.update(_CODE_KIT)
        keep.add("inspect_image")     # open_image only POINTS; this one looks
    # install_packages rides with run_code, for the same reason inspect_image
    # rides with the editors: a run that stops at ModuleNotFoundError has
    # exactly one cure, and the model cannot call a schema it was never sent.
    # Measured -- "построй питоном график" retrieves run_code but matches none
    # of install_packages' cues (установи / pip / dependenc), so three runs in a
    # row died on a missing matplotlib; the model tried pip from inside the
    # script, and then told the user the chart had been saved. It was not
    # ignoring the advice to install; it had no installer. Pin on the STATE the
    # turn can reach, never on the words the user happened to use.
    if "run_code" in keep:
        keep.add("install_packages")
        keep.add("run_tests")
    # The core fill must not hand back a tool the match already argues against.
    # Measured on the routing bench: `раздели 987654 на 321 и округли` matched
    # calculate, the fill added search anyway, and on one run of two the model
    # computed the answer and then went to the web as well. A calculation is not
    # a web question, and offering the wrong tool is most of the invitation.
    contradicted = {"search"} if "calculate" in matched else set()
    # A chart built from the user's own numbers is not a picture to invent.
    # "сохрани картинкой" matches generate_image's `картинк\w*`, so a plotting
    # request arrived offering an image generator, and the diffusion tool is a
    # far more inviting way to produce something that looks like an answer than
    # writing code is. Same shape as calculate/search above.
    # An explicit "нарисуй" is the user choosing a picture: "нарисуй график продаж
    # по месяцам, как картинку" lost generate_image here and the model looped on
    # inspect_image with nothing to draw with (live journey 34). Both are offered.
    if "run_code" in matched and "generate_image" not in (always or ()) \
            and "generate_image" not in matched:
        contradicted.add("generate_image")
        keep.discard("generate_image")
    for n in _CORE:
        if len(keep) >= limit:
            break
        if n in contradicted and n not in matched and n not in (always or ()):
            continue
        keep.add(n)

    out = [s for s in schemas if _name(s) in keep]
    if out or os.getenv("TOOLS_NOMATCH", "core").strip().lower() == "all":
        return out or list(schemas)
    # What matched is not offered here (an image editor with no picture, code
    # tools with no working folder): the core set, not everything.
    return [s for s in schemas if _name(s) in _CORE] or list(schemas)
