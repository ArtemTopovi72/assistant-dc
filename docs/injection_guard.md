# Защита от инъекций: два слоя

1. `agent/prompt_guard.py` — дешёвые регулярки: факты/сводки, рамки `<<<ATTACHED FILE`, `user_words`.
2. `agent/injection_scan.py` — модель Horizon-Labs/prompt-injection-guard-base (mmBERT 308M, Apache-2.0, 30 языков, CPU ~30 мс на кусок).
   Скачать: `huggingface_hub.snapshot_download("Horizon-Labs/prompt-injection-guard-base", local_dir="models_ext/prompt-guard")`.
   `scrub(text, source)` вырезает только помеченные куски (порог `INJECTION_GUARD_THRESHOLD`, 0.5); без модели текст проходит как есть.
   Подключено: веб-поиск (`research/search.py`, до дистилляции), вложенные файлы (`bot/tg_resolve.py`), описание видео по ссылке (`bot/tg_links.py`).
Замер на русском: прямые команды, HTML-комментарий для ИИ, DAN, «yappi» — 0.997–1.0; рецепт, фраза в художественном тексте «забудь всё», договор — ≤0.06.
Также страницы Deep Research (`research/dr_crawl.py`). Не подключено (кандидаты): вывод песочницы, RAG-чанки.
