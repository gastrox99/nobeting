# -*- coding: utf-8 -*-
"""Salt okunur paylaşım sayfası (?paylas=...).

Ekip arkadaşları giriş yapmadan yalnızca listeyi görür; yalnızca paylaşım
bağlantısının sahibine ait liste okunur.
"""
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from db import list_shared_months, load_schedule, resolve_share_token
from nobet_core import (
    TR_AY_ADLARI,
    TR_GUN_KISA,
    add_overflow_roles,
    as_bool_frame,
    build_person_ics,
    escape_html,
    now_tr,
    on_duty_summary,
    parse_schedule_import_date,
    shared_day_states,
    shared_default_month,
    shared_updated_text,
    shared_visible_months,
)


def _rows_from_schedule(schedule, year, month):
    _rows = []
    _days = {}
    for _col in schedule.columns:
        try:
            _days[int(str(_col).split()[0])] = _col
        except (ValueError, IndexError):
            continue
    for _day in sorted(_days):
        _assigned = schedule.index[schedule[_days[_day]]].tolist()
        _weekday = TR_GUN_KISA[date(year, month, _day).weekday()]
        _rows.append({
            "Tarih": f"{_day:02d}.{month:02d}.{year} {_weekday}",
            "Nöbetçiler": ", ".join(_assigned) or "-",
        })
    return _rows


def render_shared_view(token, db_ready):
    st.markdown("<h1 style='text-align: center; font-size: 2.2rem;'>Nöbet Listesi</h1>", unsafe_allow_html=True)
    _target = resolve_share_token(token) if db_ready else None
    if not _target:
        st.error("Bu paylaşım bağlantısı geçersiz ya da kapatılmış.")
        return
    _owner, _year, _month = _target
    _months = list_shared_months(_owner)
    _updated = {(_y, _m): _t for _y, _m, _t in _months}
    _today = now_tr().date()
    if (_year, _month) == (0, 0):
        # Kalıcı bağlantı: geçmiş aylar, bu ay ve sonraki ay gösterilir (daha
        # ileri aylar taslaktır); bu ay önce açılır.
        _periods = shared_visible_months(list(_updated), _today)
        if not _periods:
            st.info("Henüz paylaşılan bir nöbet listesi yok.")
            return
        _year, _month = st.selectbox(
            "Ay:", _periods,
            index=shared_default_month(_periods, _today),
            format_func=lambda _p: f"{TR_AY_ADLARI[_p[1]]} {_p[0]}",
        )
    else:
        st.subheader(f"{TR_AY_ADLARI[_month]} {_year}")
    _rows, _roles = _shared_rows(_owner, _year, _month)
    if not _rows:
        st.info("Bu ay için henüz çizelge yok.")
        return
    _updated_text = shared_updated_text(_updated.get((_year, _month)))
    if _updated_text:
        st.caption(f"Son güncelleme: {_updated_text}")
    _on_duty = on_duty_summary(_rows, _roles, _year, _month, _today)
    _tomorrow = _today + timedelta(days=1)
    if (
        (_today.year, _today.month) == (_year, _month)
        and _tomorrow.month != _month
        and (_tomorrow.year, _tomorrow.month) in _updated
    ):
        # Ayın son günü: yarının nöbetçisi sonraki ayın listesinde.
        _next_rows, _next_roles = _shared_rows(_owner, _tomorrow.year, _tomorrow.month)
        if _next_rows:
            _on_duty += on_duty_summary(_next_rows, _next_roles, _tomorrow.year, _tomorrow.month, _today)
    if _on_duty:
        for _col, (_label, _text) in zip(st.columns(len(_on_duty)), _on_duty):
            with _col:
                st.markdown(
                    f"<div style='border:1px solid rgba(128,128,128,.35);border-radius:8px;padding:8px 12px;'>"
                    f"<div style='font-size:12px;opacity:.7;'>{escape_html(_label)} nöbetçi</div>"
                    f"<div style='font-weight:600;'>{escape_html(_text)}</div></div>",
                    unsafe_allow_html=True,
                )
        st.write("")
    _people = sorted({
        _row.get(_role) for _row in _rows for _role in _roles
        if _row.get(_role) not in (None, "", "-") and _role != "Nöbetçiler"
    })
    _selected = st.selectbox("Kendi nöbetlerini görmek için adını seç:", ["Tümü"] + _people) if _people else "Tümü"
    _states = shared_day_states(_rows, _year, _month, _today)
    _shown = [
        (_row, _state) for _row, _state in zip(_rows, _states)
        if _selected == "Tümü" or any(_row.get(_role) == _selected for _role in _roles)
    ]
    _table = pd.DataFrame(
        [{_k: _row.get(_k, "-") for _k in ["Tarih"] + _roles} for _row, _ in _shown],
        columns=["Tarih"] + _roles,
    )
    # Geçmiş günler soluk, bugün vurgulu görünür.
    _row_styles = {
        "past": "color: rgba(128,128,128,.55);",
        "today": "font-weight: 700; background-color: rgba(255,200,0,.18);",
        "future": "",
    }
    _shown_states = [_state for _, _state in _shown]
    st.dataframe(
        _table.style.apply(lambda _r: [_row_styles[_shown_states[_r.name]]] * len(_r), axis=1),
        hide_index=True, width="stretch",
    )
    if _selected != "Tümü":
        st.download_button(
            "📅 Nöbetlerimi takvime ekle",
            build_person_ics(_selected, _rows, _roles, _year, _month).encode("utf-8"),
            f"nobet_{_year}_{_month:02d}.ics", "text/calendar",
            on_click="ignore", width="stretch",
        )
    st.caption("Bu sayfa salt okunurdur ve liste sahibi değişiklik yaptıkça güncellenir.")


def _shared_rows(_owner, year, month):
    """Paylaşım sayfası için ayın satırları ve görevleri: (satırlar, görevler); yoksa (None, None)."""
    _loaded = load_schedule(f"Otomatik_{year}_{month:02d}", year, month, owner=_owner)
    _rows = _loaded[4] if _loaded and len(_loaded) > 4 and _loaded[4] else None
    _schedule = _loaded[1] if _loaded else None
    if _schedule is not None:
        _schedule = as_bool_frame(_schedule)
    if _rows:
        _roles = []
        for _row in _rows:
            for _key in _row:
                if _key != "Tarih" and _key not in _roles:
                    _roles.append(_key)
        if _schedule is not None and not _schedule.empty:
            # Görev sayısından fazla atanan kişiler de listede görünsün.
            _day_columns = {}
            for _col in _schedule.columns:
                try:
                    _day_columns[int(str(_col).split()[0])] = _col
                except (ValueError, IndexError):
                    continue
            _row_columns = []
            for _row in _rows:
                _parsed = parse_schedule_import_date(_row.get("Tarih"))
                _row_columns.append(_day_columns.get(_parsed["day"]) if _parsed else None)
            _rows, _roles = add_overflow_roles(_rows, _roles, _schedule, _row_columns)
        return _rows, _roles
    if _schedule is not None and not _schedule.empty:
        return _rows_from_schedule(_schedule, year, month), ["Nöbetçiler"]
    return None, None
