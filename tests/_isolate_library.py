"""Keep a GUI suite off the LIVE document database.

Importing this points library.DEFAULT_DB at a per-process temp file.

Why it exists: building an AssistantWindow builds the Database tab, and the
tab's _get_library() opens memory/library.db -- the real one. When the app is
actually running it holds that file, so the suite blocked inside
KnowledgeBase._init_schema on SQLite's busy_timeout: 30 seconds PER STATEMENT,
across a schema script of a dozen of them. That is what "the test suite hangs"
was. It is also a data hazard in the other direction: a suite that got the lock
would write its fixtures into the user's real library.

Same class of bug, and the same fix, as tg_bot.redirect_data_dir (a suite
writing into the live tg_users.db) and gui_telegram_tab.redirect_settings (a
suite overwriting the live BotFather token).

    import _isolate_library  # noqa: F401  -- before importing gui
"""
import tempfile
from pathlib import Path

try:
    import library as _library
    _library.redirect_db(Path(tempfile.mkdtemp(prefix="testlib_")) / "library.db")
except Exception:  # library is optional in some environments
    pass
