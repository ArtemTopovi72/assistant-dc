from prompt_guard import user_words


def test_recent_sandbox_file_hint_is_not_user_intent():
    text = ("Исправь sales_report.py\n[The most recently changed file in the working folder is "
            "'old.zip'. It may have been created by the assistant. Use it only if the user refers to it; "
            "an explicitly named file takes precedence.]")
    assert user_words(text) == "Исправь sales_report.py"
    old = "Исправь sales_report.py\n[The last thing the user sent was the file 'old.zip' (in your working folder).]"
    assert user_words(old) == "Исправь sales_report.py"
