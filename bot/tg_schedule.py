"""⏰ Tasks: the user's reminders, scheduled tasks and Ozon price watches, with buttons.

Adding goes through the agent (the 🔔/🔁/📉 buttons arm a prefix): it reads «через
2 часа», «по пятницам в 18» and a pasted link better than a form would. Seeing,
changing and deleting is here, in code: the 📋 list is an inline message whose
buttons («sch:...») open a card per item and edit it in place.

Every press names an item id and acts only on the presser's own item (reminders
checks the owner), so an old card or a forged payload cannot touch another
chat's tasks. A typed answer (new time, text, target price) is a capture in
sess.reg_state «sch_time:<id>» / «sch_text:<id>» / «sch_target:<id>»: it stays
armed on a wrong answer and any button leaves it, like the weather flow.

tg_bot is imported at the BOTTOM and read as ``tg_bot.<name>`` at call time,
the same convention the rest of the tg_* family uses.
"""
from __future__ import annotations

import datetime as _dt
import html as _html
import re
import threading
import time

import price_watch
import reminders

_WEEKDAYS = {"ru": ("пн", "вт", "ср", "чт", "пт", "сб", "вс"),
             "en": ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")}
_EVERY_HOURS = (1, 3, 6, 12, 24)
_WX_TIMES = ("06:00", "07:00", "08:00", "09:00", "10:00")
_TIME = re.compile(r"^\s*(?:(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?\s+)?(\d{1,2})[:.](\d{2})\s*$")


def parse_when(text: str, now: _dt.datetime) -> _dt.datetime | None:
    """«18:30» (today, or tomorrow once past) or «27.10 18:30» / «27.10.2026 18:30»."""
    m = _TIME.match(text or "")
    if not m:
        return None
    day, month, year, hh, mm = m.groups()
    try:
        if day:
            y = int(year) + (2000 if year and len(year) == 2 else 0) if year else now.year
            t = now.replace(year=y, month=int(month), day=int(day), hour=int(hh), minute=int(mm),
                            second=0, microsecond=0)
            if not year and t <= now:
                t = t.replace(year=y + 1)
        else:
            t = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
            if t <= now:
                t += _dt.timedelta(days=1)
    except ValueError:
        return None
    return t if t > now else None


def _rub(n) -> str:
    return f"{int(n):,} ₽".replace(",", " ")


class ScheduleMixin:

    # ── time in the user's zone ──────────────────────────────────────────────

    def _sch_now(self, chat_id: int) -> _dt.datetime:
        from prompts import _user_now
        try:
            from tg_weather import city_tz
            home = self._fact_city(chat_id)
            tz = city_tz(home) if home else ""
        except Exception:
            tg_bot.logger.debug("tasks: no user zone for %s", chat_id, exc_info=True)
            tz = ""
        return _user_now(tz)

    def _sch_fmt(self, ts: float, chat_id: int, lang: str) -> str:
        t = _dt.datetime.fromtimestamp(ts, self._sch_now(chat_id).tzinfo)
        return f"{_WEEKDAYS.get(lang, _WEEKDAYS['en'])[t.weekday()]} {t:%d.%m}, {t:%H:%M}"

    # ── views ────────────────────────────────────────────────────────────────

    def _sch_name(self, item: dict) -> str:
        return item.get("name") if item.get("kind") == "price" else item["text"]

    def _sch_list_view(self, chat_id: int, lang: str) -> tuple:
        items = reminders.pending(chat_id)
        if not items:
            return tg_bot._t("sch_empty", lang), None
        icon = {"price": "📉", "task": "🔁"}
        rows = [[{"text": f"{icon.get(i.get('kind'), '🔔')} {self._sch_name(i)[:40]}",
                  "callback_data": f"sch:o:{i['id']}"}] for i in items]
        rows.append([{"text": tg_bot._t("sch_b_refresh", lang), "callback_data": "sch:l"}])
        lines = [f"{n}. {icon.get(i.get('kind'), '🔔')} {_html.escape(self._sch_name(i)[:60])} — "
                 + (_rub(i["price"]) if i.get("kind") == "price" else self._sch_fmt(i["due"], chat_id, lang))
                 for n, i in enumerate(items, 1)]
        return (tg_bot._t("sch_head", lang, n=len(items)) + "\n\n" + "\n".join(lines),
                {"inline_keyboard": rows})

    def _sch_card_view(self, chat_id: int, item: dict, lang: str) -> tuple:
        t, rid = tg_bot._t, item["id"]
        b = lambda key, data: {"text": t(key, lang), "callback_data": f"sch:{data}"}
        if item.get("kind") == "price":
            text = "\n".join([
                f"<b>{t('sch_kind_price', lang)}</b>",
                f"<a href=\"{_html.escape(item['product'])}\">{_html.escape(item.get('name') or '')}</a>",
                t("sch_low", lang, price=_rub(item["price"])),
                t("sch_target", lang, price=_rub(item["target"])) if item.get("target")
                else t("sch_no_target", lang),
                t("sch_every", lang, h=f"{item['every'] / 3600:g}",
                  when=self._sch_fmt(item["due"], chat_id, lang)),
                t("sch_until", lang, date=self._sch_fmt(item["until"], chat_id, lang).split(",")[0])
                if item.get("until") else ""])
            unit = "ч" if lang == "ru" else "h"
            rows = [[{"text": ("✅ " if item["every"] == h * 3600 else "") + f"{h} {unit}",
                      "callback_data": f"sch:ev:{rid}:{h}"} for h in _EVERY_HOURS],
                    [b("sch_b_target", f"tg:{rid}")] + ([b("sch_b_notarget", f"tg0:{rid}")] if item.get("target") else []),
                    [b("sch_b_check", f"chk:{rid}"), b("sch_b_extend", f"ext:{rid}")]]
        else:
            rep = {86400: "sch_daily", 7 * 86400: "sch_weekly"}.get(item.get("every"), "sch_once")
            text = "\n".join([
                f"<b>{t('sch_kind_task' if item.get('kind') == 'task' else 'sch_kind_text', lang)}</b>",
                f"«{_html.escape(item['text'])}»",
                t("sch_when", lang, when=self._sch_fmt(item["due"], chat_id, lang), rep=t(rep, lang))])
            rows = [[b("sch_b_snooze10", f"sn:{rid}:10"), b("sch_b_snooze60", f"sn:{rid}:60"),
                     b("sch_b_snooze1d", f"sn:{rid}:1440")],
                    [b("sch_b_time", f"t:{rid}"), b("sch_b_text", f"x:{rid}")],
                    [{"text": ("✅ " if rep == key else "") + t(key, lang), "callback_data": f"sch:rep:{rid}:{v}"}
                     for key, v in (("sch_once", "none"), ("sch_daily", "daily"), ("sch_weekly", "weekly"))],
                    [b("sch_b_run" if item.get("kind") == "task" else "sch_b_send", f"run:{rid}")]]
        rows.append([b("sch_b_delete", f"del:{rid}"), b("sch_b_list", "l")])
        return "\n".join(x for x in text.split("\n") if x), {"inline_keyboard": rows}

    def _send_tasks_list(self, chat_id: int, lang: str) -> None:
        text, kb = self._sch_list_view(chat_id, lang)
        self._send_text(chat_id, text, parse_mode="HTML", keyboard=kb)

    def _send_weather_times(self, chat_id: int, lang: str) -> None:
        self._send_text(chat_id, tg_bot._t("sch_wx_pick", lang), keyboard={"inline_keyboard": [
            [{"text": x, "callback_data": f"sch:wx:{x}"} for x in _WX_TIMES]]})

    # ── buttons ──────────────────────────────────────────────────────────────

    def _cb_schedule(self, chat_id: int, msg: dict, data: str) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        mid = msg.get("message_id")
        parts = data.split(":")
        op, rid = parts[1], (parts[2] if len(parts) > 2 else "")

        def show(text, kb):
            self._edit_text(chat_id, mid, text, parse_mode="HTML", keyboard=kb)

        def card(item=None):
            item = item or reminders.get(chat_id, rid)
            if item:
                show(*self._sch_card_view(chat_id, item, lang))
            else:
                show(tg_bot._t("sch_gone", lang), self._sch_list_view(chat_id, lang)[1])

        if op == "l":
            return show(*self._sch_list_view(chat_id, lang))
        if op == "wx":
            return self._sch_add_weather(chat_id, lang, ":".join(parts[2:4]))
        item = reminders.get(chat_id, rid)
        if not item:
            return card()
        now = time.time()
        try:
            if op == "o":
                return card(item)
            if op == "del":
                return show(tg_bot._t("sch_del_ask", lang, what=_html.escape(self._sch_name(item)[:60])),
                            {"inline_keyboard": [[
                                {"text": tg_bot._t("sch_del_yes", lang), "callback_data": f"sch:delok:{rid}"},
                                {"text": tg_bot._t("sch_del_no", lang), "callback_data": f"sch:o:{rid}"}]]})
            if op == "delok":
                reminders.remove(chat_id, rid)
                text, kb = self._sch_list_view(chat_id, lang)
                return show(tg_bot._t("sch_deleted", lang, what=_html.escape(self._sch_name(item)[:60]))
                            + "\n\n" + text, kb)
            if op == "sn":
                return card(reminders.update(chat_id, rid, due=max(item["due"], now) + int(parts[3]) * 60))
            if op == "rep":
                every = {"daily": 86400, "weekly": 7 * 86400}.get(parts[3])
                return card(reminders.update(chat_id, rid, every=every))
            if op == "ev":
                hours = int(parts[3])
                if hours not in _EVERY_HOURS:
                    return
                return card(reminders.update(chat_id, rid, every=hours * 3600, due=now + hours * 3600))
            if op == "tg0":
                return card(reminders.update(chat_id, rid, target=None))
            if op == "ext":
                return card(reminders.update(chat_id, rid, until=max(item.get("until") or now, now) + 30 * 86400))
            if op in ("t", "x", "tg"):
                field = {"t": "time", "x": "text", "tg": "target"}[op]
                sess.reg_state = f"sch_{field}:{rid}"
                self._store.put(sess)
                return self._send_text(chat_id, tg_bot._t({"t": "sch_ask_time", "x": "sch_ask_text",
                                                           "tg": "sch_ask_target"}[op], lang), parse_mode="HTML")
            if op == "run":
                if item.get("kind") == "task":
                    return self._run_scheduled(chat_id, item["text"])
                return self._send_text(chat_id, tg_bot._t("sch_remind_now", lang, text=item["text"]))
            if op == "chk":
                threading.Thread(target=self._sch_check_now, args=(chat_id, item, lang), daemon=True,
                                 name="tasks-check").start()
        except (ValueError, IndexError) as exc:
            tg_bot.logger.warning("tasks button %r: %s", data, exc)

    def _sch_check_now(self, chat_id: int, item: dict, lang: str) -> None:
        try:
            d = price_watch.read(item)
        except Exception as exc:
            tg_bot.logger.warning("tasks: check now %s failed: %s", item.get("product"), exc)
            d = {}
        if not d.get("price"):
            return self._send_text(chat_id, tg_bot._t("sch_check_failed", lang))
        self._send_text(chat_id, tg_bot._t("sch_now_price", lang, name=_html.escape(item.get("name") or ""),
                                           price=_rub(d["price"]), low=_rub(item["price"])), parse_mode="HTML")
        if d["price"] < item["price"]:
            reminders.update(chat_id, item["id"], price=d["price"])   # said now: not again at the next check

    def _sch_add_weather(self, chat_id: int, lang: str, hhmm: str) -> None:
        when = parse_when(hhmm, self._sch_now(chat_id))
        if not when or hhmm not in _WX_TIMES:
            return
        reminders.add(chat_id, (when - self._sch_now(chat_id)).total_seconds(),
                      tg_bot._t("sch_wx_text", lang), every_s=86400, lang=lang, kind="task")
        self._send_text(chat_id, tg_bot._t("sch_wx_done", lang, time=hhmm))

    # ── typed answers ────────────────────────────────────────────────────────

    def _sch_capture(self, chat_id: int, sess, lang: str, text: str) -> None:
        field, _, rid = sess.reg_state.partition(":")
        item = reminders.get(chat_id, rid)
        if not item:
            self._clear_capture(sess)
            return self._send_text(chat_id, tg_bot._t("sch_gone", lang))
        text = (text or "").strip()
        try:
            if field == "sch_time":
                when = parse_when(text, self._sch_now(chat_id))
                if not when:
                    return self._send_text(chat_id, tg_bot._t("sch_bad_time", lang), parse_mode="HTML")
                item = reminders.update(chat_id, rid, due=when.timestamp())
            elif field == "sch_text":
                if not text:
                    return self._send_text(chat_id, tg_bot._t("sch_ask_text", lang))
                item = reminders.update(chat_id, rid, text=text[:500])
            elif field == "sch_target":
                digits = re.sub(r"[\s ₽руб.]", "", text.lower())
                if not digits.isdigit() or not int(digits):
                    return self._send_text(chat_id, tg_bot._t("sch_bad_target", lang), parse_mode="HTML")
                item = reminders.update(chat_id, rid, target=int(digits))
        except ValueError as exc:
            tg_bot.logger.warning("tasks capture %s: %s", sess.reg_state, exc)
        self._clear_capture(sess)
        if item:
            card_text, kb = self._sch_card_view(chat_id, item, lang)
            self._send_text(chat_id, tg_bot._t("sch_saved", lang) + "\n\n" + card_text,
                            parse_mode="HTML", keyboard=kb)


import tg_bot  # noqa: E402  (bottom import: see module docstring)
