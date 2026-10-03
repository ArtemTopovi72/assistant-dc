"""Granting and revoking sandbox rights.

The suite is written around one property: there is no route through this module
that turns "I could not tell" into permission. A missing user, an unreadable
prefs blob, a typo in a hand-edited row, a store that raises — every one of them
has to come back `off`, because the alternative is a stranger executing code on
someone's desktop.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import sandbox_access as A


class _U:
    def __init__(self, prefs=None):
        self.prefs = prefs


class _Store:
    def __init__(self, users=None):
        self.users = users or {}
        self.saved = []

    def get(self, chat_id):
        return self.users.get(chat_id)

    def put(self, user):
        self.saved.append(user)


# --- default deny -----------------------------------------------------------

@pytest.mark.parametrize("user", [
    None, _U(None), _U({}), _U({"sandbox": ""}), _U({"sandbox": "yes"}),
    _U({"sandbox": "CODE "}), _U({"sandbox": 1}), _U("not-a-dict"),
])
def test_anything_unclear_is_off(user):
    """`"CODE "` is in the list deliberately: it is not exactly a level, and a
    permission check is the wrong place to be lenient about whitespace… except
    that normalize DOES strip and lowercase, so this one is expected to pass as
    `code`. It is asserted below on its own."""
    if isinstance(user, _U) and isinstance(user.prefs, dict) \
            and str(user.prefs.get("sandbox", "")).strip().lower() in A.LEVELS:
        pytest.skip("covered by test_whitespace_and_case_are_tolerated")
    assert A.level_for(user) == A.OFF


def test_whitespace_and_case_are_tolerated():
    """A level typed into an admin box with a stray space is still that level;
    an unrecognised WORD is not."""
    assert A.level_for(_U({"sandbox": "  CODE "})) == A.CODE
    assert A.level_for(_U({"sandbox": "codee"})) == A.OFF


def test_a_raising_prefs_denies():
    class Hostile:
        @property
        def prefs(self):
            raise RuntimeError("db is on fire")
    assert A.level_for(Hostile()) == A.OFF


# --- the ladder -------------------------------------------------------------

def test_each_level_grants_exactly_what_it_says():
    off, files, code, host = (_U({"sandbox": lv}) for lv in A.LEVELS)

    assert not A.may_use_files(off) and not A.may_run_code(off)
    assert A.may_use_files(files) and not A.may_run_code(files)
    assert A.may_use_files(code) and A.may_run_code(code)
    assert A.may_use_files(host) and A.may_run_code(host)


def test_host_execution_is_the_top_level_only():
    """Unisolated execution must never be reachable by 'code' — the container
    being down is exactly when it would matter."""
    for lv in (A.OFF, A.FILES, A.CODE):
        assert not A.allow_host_execution(_U({"sandbox": lv})), lv
    assert A.allow_host_execution(_U({"sandbox": A.HOST}))


def test_the_ladder_is_ordered():
    for weaker, stronger in zip(A.LEVELS, A.LEVELS[1:]):
        assert A.at_least(_U({"sandbox": stronger}), weaker)
        assert not A.at_least(_U({"sandbox": weaker}), stronger)


# --- granting and revoking --------------------------------------------------

def test_grant_and_revoke_round_trip():
    store = _Store({7: _U({})})
    assert A.set_level(store, 7, A.CODE) == A.CODE
    assert A.level_for(store.get(7)) == A.CODE
    assert A.set_level(store, 7, A.OFF) == A.OFF
    assert not A.may_use_files(store.get(7))


def test_granting_persists_through_the_store():
    """No in-process cache: the bot must see a revocation on its next message."""
    store = _Store({7: _U({})})
    A.set_level(store, 7, A.FILES)
    assert store.saved, "the change was never written"


def test_granting_keeps_other_prefs():
    store = _Store({7: _U({"badge": "🐈", "sandbox": "off"})})
    A.set_level(store, 7, A.CODE)
    assert store.get(7).prefs["badge"] == "🐈"


def test_granting_a_nonsense_level_revokes_rather_than_grants():
    store = _Store({7: _U({"sandbox": A.HOST})})
    assert A.set_level(store, 7, "superuser") == A.OFF
    assert not A.may_use_files(store.get(7))


def test_granting_to_an_unknown_user_is_an_error_not_a_grant():
    with pytest.raises(KeyError):
        A.set_level(_Store(), 999, A.CODE)


# --- it reads a real user row -----------------------------------------------

def test_it_works_against_the_real_user_store(tmp_path):
    """The prefs column exists so a per-user option needs no migration; this
    checks the assumption rather than trusting it."""
    import tg_userstore as S
    store = S._UserStore(tmp_path / "users.db")
    u = S._User(chat_id=42, name="brother", status="approved")
    store.put(u)
    assert A.level_for(store.get(42)) == A.OFF
    A.set_level(store, 42, A.CODE)
    assert A.level_for(store.get(42)) == A.CODE
    assert A.may_run_code(store.get(42))
    assert not A.allow_host_execution(store.get(42))
