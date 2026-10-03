"""rag_add / rag_search: a host-mediated bridge from the network-isolated code
sandbox to the EXISTING knowledge base (library.py / knowledge_client), not a
second RAG engine grown inside Docker. Also: rag_add/rag_search must land in
the SAME per-chat index tg_library.py already shows in the user's own
Documents list, so "закинь этот файл в RAG" from the agent and an upload from
the Telegram UI end up in one place, not two.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

import tools as T
import sandbox_access as A
from code_sandbox import Sandbox


class _Ctx:
    def __init__(self, sandbox=None, user=None):
        self.sandbox = sandbox
        self.sandbox_user = user


class _User:
    def __init__(self, level):
        self.prefs = {"sandbox": level}


@pytest.fixture(autouse=True)
def _isolated_library_dir(tmp_path_factory, monkeypatch):
    """Every test in this module gets its own tg_bot._LIBRARY_DIR -- otherwise
    a rag_add here would write into the LIVE tg_libraries/ folder (see
    tg_bot.redirect_data_dir's own docstring: exactly this kind of leak was
    found live once already)."""
    import tg_bot
    d = tmp_path_factory.mktemp("libdir")
    monkeypatch.setattr(tg_bot, "_LIBRARY_DIR", d)
    yield d


@pytest.fixture
def box(tmp_path_factory):
    return Sandbox(tmp_path_factory.mktemp("sbx"))


@pytest.fixture
def ctx(box):
    return _Ctx(box, _User(A.CODE))


def _call(ctx, name, **args):
    return T.execute_tool(ctx, {}, name, args)


# --- rag_add ------------------------------------------------------------

def test_rag_add_indexes_an_existing_file(ctx, box):
    box.write_text("notes.txt", "The launch window opens at dawn on Tuesday.")
    out = _call(ctx, "rag_add", path="notes.txt")
    assert "Indexed" in out and "[TOOL ERROR]" not in out, out


def test_rag_add_refuses_a_missing_file(ctx, box):
    out = _call(ctx, "rag_add", path="ghost.txt")
    assert "[TOOL ERROR]" in out
    assert "does not exist" in out


def test_rag_add_refuses_an_unsupported_extension(ctx, box):
    box.write_text("image.png", "not really a png, just bytes")
    out = _call(ctx, "rag_add", path="image.png")
    assert "[TOOL ERROR]" in out
    assert "cannot be indexed" in out


def test_rag_add_without_a_sandbox_is_refused(box):
    out = _call(_Ctx(None, None), "rag_add", path="notes.txt")
    assert "[TOOL ERROR]" in out and "sandbox" in out.lower()


# --- rag_search -----------------------------------------------------------

def test_rag_search_on_an_empty_base_says_so(ctx):
    out = _call(ctx, "rag_search", query="launch window")
    assert "empty" in out.lower()


def test_rag_search_finds_what_rag_add_indexed(ctx, box):
    box.write_text("notes.txt", "The launch window opens at dawn on Tuesday, "
                                "not at noon as previously announced.")
    add_out = _call(ctx, "rag_add", path="notes.txt")
    assert "Indexed" in add_out, add_out
    out = _call(ctx, "rag_search", query="launch window")
    assert "launch window" in out.lower()
    assert "notes.txt" in out


def test_rag_search_needs_a_query(ctx):
    out = _call(ctx, "rag_search", query="")
    assert "[TOOL ERROR]" in out


# --- isolation: two different sandboxes must not see each other's documents -

def test_two_chats_do_not_share_a_knowledge_base(tmp_path_factory):
    box_a = Sandbox(tmp_path_factory.mktemp("chat_a"))
    box_b = Sandbox(tmp_path_factory.mktemp("chat_b"))
    ctx_a = _Ctx(box_a, _User(A.CODE))
    ctx_b = _Ctx(box_b, _User(A.CODE))
    box_a.write_text("secret.txt", "The password rotation happens every Friday.")
    add_out = _call(ctx_a, "rag_add", path="secret.txt")
    assert "Indexed" in add_out, add_out
    out_b = _call(ctx_b, "rag_search", query="password rotation")
    assert "empty" in out_b.lower(), out_b


# --- schema: withheld unless a sandbox is actually attached ----------------

def test_rag_tools_are_in_the_gated_code_tool_set():
    from tool_code_handlers import CODE_TOOL_NAMES
    assert "rag_add" in CODE_TOOL_NAMES
    assert "rag_search" in CODE_TOOL_NAMES


if __name__ == "__main__":
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pytest", __file__, "-q"])
    sys.exit(r.returncode)
