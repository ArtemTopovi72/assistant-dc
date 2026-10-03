"""Interface language: English by default, every label Russian when chosen.

Translation happens where text meets Qt (gui_i18n.install), so the check is the
built window itself: tabs, buttons, placeholders and dynamic statuses come out
Russian, while combo boxes still answer the code in English.
"""
import os, re, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen"); os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt5.QtWidgets import QApplication, QComboBox, QPushButton
app = QApplication.instance() or QApplication([])
import gui_i18n
import gui_ru

PH = re.compile(r"%[-+ 0#]*\d*(?:\.\d+)?[sdif]")


def test_table_placeholders_match():
    bad = [k for k, v in gui_ru.RU.items() if len(PH.findall(k)) != len(PH.findall(v))]
    assert not bad, bad


def test_russian_window():
    gui_i18n.install("ru")
    import gui
    w = gui.AssistantWindow("x", True)
    texts = {b.text() for b in w.findChildren(QPushButton)}
    assert "⚙  Настройки" in texts and "■  Стоп" in texts, sorted(texts)[:20]
    assert w.input.placeholderText() == "Напишите сообщение и нажмите Enter…"
    # a formatted status is matched by its template
    assert gui_i18n.tr("Loading 'gemma'…") == "Загружаю «gemma»…"
    assert gui_i18n.tr("Indexed: 3 documents · 10 passages · 50% embedded · embedding model bge") \
        == "В базе: документов 3 · фрагментов 10 · эмбеддингов 50% · модель эмбеддингов bge"
    assert gui_i18n.tr("↑ Up") == "↑ Вверх"
    # combos show Russian but the code keeps reading English values
    c = QComboBox(); c.addItems(["Short", "Detailed"]); c.addItem("Auto", "auto")
    assert QComboBox.itemText.__wrapped__(c, 0) == "Коротко"
    c.setCurrentText("Detailed")
    assert c.currentText() == "Detailed" and c.findText("Auto") == 2 and c.itemData(2) == "auto"
    w.ctx = None; w.close(); w.deleteLater(); app.processEvents()


def test_live_switch_back_and_forth():
    """Settings → Language switches the running window, no restart."""
    gui_i18n.install("en")
    import gui
    w = gui.AssistantWindow("x", True)
    texts = lambda: {b.text() for b in w.findChildren(QPushButton)}
    assert "⚙  Settings" in texts()
    gui_i18n.set_language("ru")
    assert "⚙  Настройки" in texts() and w.pages.tabText(0) == "Команды", w.pages.tabText(0)
    assert w.input.placeholderText() == "Напишите сообщение и нажмите Enter…"
    gui_i18n.set_language("en")
    assert "⚙  Settings" in texts() and w.pages.tabText(0) == "Commands"
    w.ctx = None; w.close(); w.deleteLater(); app.processEvents()


if __name__ == "__main__":
    test_table_placeholders_match(); test_russian_window(); test_live_switch_back_and_forth(); print("ok")
