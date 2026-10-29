from types import SimpleNamespace

from gui_dashboard import DashboardMixin


def test_video_editor_opens_sandbox_and_focuses_the_prompt():
    calls = []
    prompt = SimpleNamespace(
        setPlaceholderText=lambda text: calls.append(("hint", text)),
        setFocus=lambda: calls.append(("focus",)),
    )
    host = SimpleNamespace(
        _focus_tab=lambda key: calls.append(("tab", key)),
        pages=SimpleNamespace(setCurrentIndex=lambda index: calls.append(("page", index))),
        code_tab=SimpleNamespace(prompt=prompt),
    )
    DashboardMixin._open_video_editor(host)
    assert calls[0] == ("tab", "code")
    assert ("page", 2) in calls
    assert calls[-1] == ("focus",)
