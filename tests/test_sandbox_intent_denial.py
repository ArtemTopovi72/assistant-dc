from types import SimpleNamespace

from graph_personality import _sandbox_execution_denied


def test_execution_denial_requires_explicit_code_intent():
    no_access = SimpleNamespace(sandbox=None, sandbox_user=None)
    assert _sandbox_execution_denied(no_access, {'ok': True, 'must_call': 'run_code'})
    assert not _sandbox_execution_denied(no_access, {'ok': True, 'must_call': 'search'})
    assert not _sandbox_execution_denied(no_access, {'ok': False, 'must_call': 'run_code'})
    files_only = SimpleNamespace(sandbox=object(), sandbox_user={'prefs': {'sandbox': 'files'}})
    assert _sandbox_execution_denied(files_only, {'ok': True, 'must_call': 'run_code'})
    assert not _sandbox_execution_denied(files_only, {'ok': True, 'must_call': 'list_files'})
    allowed = SimpleNamespace(sandbox=object(), sandbox_user={'prefs': {'sandbox': 'code'}})
    assert not _sandbox_execution_denied(allowed, {'ok': True, 'must_call': 'run_code'})
