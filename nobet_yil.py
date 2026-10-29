# -*- coding: utf-8 -*-
"""Analiz > Yıl özeti: kayıtlı aylardan kişi başı yıllık nöbet dağılımı ve tercih karşılanma oranı."""
import calendar
import json
from datetime import date

import pandas as pd
import streamlit as st

from db import load_auto_saved_assignments, log_unexpected
from nobet_core import as_bool_frame, resolve_alias
from nobet_gunler import _column_day, holiday_text_for_period, parse_holiday_days

SUTUNLAR = ["Nöbet", "Cumartesi", "Pazar", "Cuma", "Tatil", "Tercih karşılandı"]


def _tercihler(pref_json):
    """Kayıtlı tercih JSON'undan {(gün sütunu, kişi): değer}; okunamazsa boş."""
    try:
        data = json.loads(pref_json) if pref_json else None
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    result = {}
    for column, people in data.items():
        if isinstance(people, dict):
            for person, value in people.items():
                try:
                    result[(str(column), str(person))] = int(value)
                except (TypeError, ValueError):
                    continue
    return result


def yil_ozeti(schedules, prefs, year, aliases, team, holidays_by_period):
    """Kayıtlı aylardan kişi başı yıl özeti: (DataFrame, ay_sayısı).

    ``schedules``: {(yıl, ay): bool çizelge}, ``prefs``: {(yıl, ay): tercih JSON'u}.
    Tercih karşılandı: yeşil (tercih) günlerin kaçında nöbet yazıldı.
    Eski adla kayıtlı aylar güncel ada çevrilir; ekipte olmayan ama nöbeti olan kişi de listelenir."""
    totals = {}
    months = 0
    for (y, m), frame in sorted(schedules.items()):
        if y != year or frame is None or frame.empty:
            continue
        months += 1
        frame = as_bool_frame(frame)
        frame = frame.rename(index={p: resolve_alias(p, aliases, team) for p in frame.index})
        if frame.index.has_duplicates:
            frame = frame.groupby(level=0).any()
        holidays, _ = parse_holiday_days(holiday_text_for_period(holidays_by_period, y, m),
                                         calendar.monthrange(y, m)[1])
        wanted = {(c, resolve_alias(p, aliases, team)): v for (c, p), v in _tercihler(prefs.get((y, m))).items()}
        for person in frame.index:
            row = totals.setdefault(person, dict.fromkeys(SUTUNLAR[:-1], 0) | {"istenen": 0, "karsilanan": 0})
            for column in frame.columns:
                day = _column_day(column)
                worked = bool(frame.at[person, column])
                if worked and day is not None:
                    row["Nöbet"] += 1
                    try:
                        weekday = date(y, m, day).weekday()
                    except ValueError:
                        weekday = None
                    row["Cumartesi"] += weekday == 5
                    row["Pazar"] += weekday == 6
                    row["Cuma"] += weekday == 4
                    row["Tatil"] += day in holidays
                if wanted.get((str(column), person)) == 1:
                    row["istenen"] += 1
                    row["karsilanan"] += worked
    ordered = [p for p in team if p in totals] + [p for p in totals if p not in team]
    rows = []
    for person in ordered:
        t = totals[person]
        rows.append({
            "Kişi": person, **{c: int(t[c]) for c in SUTUNLAR[:-1]},
            "Tercih karşılandı": f"{t['karsilanan']}/{t['istenen']}" if t["istenen"] else "—",
        })
    return pd.DataFrame(rows, columns=["Kişi", *SUTUNLAR]), months


def render_year_summary(owner, yil, isimler, holidays_by_period):
    """Açılınca yılın on iki ayını bir sorguyla okur; kapalıyken veritabanına gitmez."""
    st.markdown("**📆 Yıl özeti**")
    if not st.toggle(f"{yil} yılının kayıtlı aylarını göster", key="yil_ozeti_ac"):
        return
    try:
        schedules, prefs = load_auto_saved_assignments(
            [(yil, m) for m in range(1, 13)], owner=owner, include_preferences=True,
        )
    except Exception as _error:
        log_unexpected("Yıl özeti", _error)
        st.warning("⚠️ Yıl özeti okunamadı. Birazdan tekrar deneyin.")
        return
    table, months = yil_ozeti(schedules, prefs, yil, st.session_state.get("name_aliases"),
                              isimler, holidays_by_period)
    if table.empty:
        st.info(f"{yil} için kayıtlı ay yok.")
        return
    st.caption(f"{months} kayıtlı ay: {yil}. Tercih karşılandı = yeşil (tercih) günlerin kaçında nöbet yazıldı.")
    st.dataframe(table, hide_index=True, width="stretch")

