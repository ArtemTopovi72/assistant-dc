"""Paraphrase cases: a tool is needed, but none of the retrieval cue words appear.

These exercise the NO-MATCH fallback of tool_retrieval.select_tools -- the only
path an embedding fallback (TOOLS_EMBED=1) can change. Checked offline by
bench/tool_recall.py: every case here must miss all the regex cues.
"""

PARA = [
    dict(id="para_weather", cat="para", text="мне завтра в Казань, зонт брать?",
         expect="search"),
    dict(id="para_rate", cat="para", text="почём сегодня доллар у ЦБ?",
         expect="search"),
    dict(id="para_score", cat="para", text="как вчера сыграл Зенит?",
         expect="search"),
    dict(id="para_draw", cat="para", text="изобрази закат над морем в стиле Айвазовского",
         expect="generate_image"),
    dict(id="para_draw2", cat="para", text="хочу обои на рабочий стол: неоновый Токио ночью",
         expect="generate_image"),
    dict(id="para_video", cat="para", text="сделай короткий ролик, где волны бьются о маяк",
         expect="generate_video"),
    dict(id="para_calc", cat="para", text="сколько выйдет 17% от 348 900?",
         expect="calculate"),
    dict(id="para_remember", cat="para", text="запиши себе: у меня аллергия на орехи",
         expect="remember_fact"),
    dict(id="para_forget", cat="para", text="выкинь из головы всё, что знаешь про мою работу",
         expect="forget_facts"),
    dict(id="para_shop", cat="para", text="подбери мне недорогие беспроводные наушники, до трёх тысяч",
         expect="ozon_search"),
    dict(id="para_slides", cat="para", text="собери мне слайды для выступления про кофе",
         expect="create_presentation"),
    dict(id="para_report", cat="para",
         text="мне нужен обстоятельный разбор рынка электросамокатов с цифрами и источниками",
         expect="deep_research"),
    dict(id="para_py", cat="para", text="прогони вот это и скажи, что выведет: print(sum(range(10**6)))",
         expect="run_code"),
    dict(id="para_clip", cat="para", text="глянь, что я только что скопировал",
         expect="read_clipboard"),
    dict(id="para_photo_real", cat="para", text="покажи, как выглядит настоящий Байкал зимой",
         expect="find_photo", forbid=["generate_image"]),
]
