# -*- coding: utf-8 -*-
"""Paylaşım sayfası (?paylas=...).

Ekip arkadaşları giriş yapmadan listeyi görür; yalnızca paylaşım bağlantısının
sahibine ait liste okunur. "Talep Gönder" sekmesinden tercih/izin günlerini
yazarlar; talepler çizelgeye doğrudan yazılmaz, liste sahibi onaylar.
"""
from datetime import date, timedelta

import pandas as pd
import json
from urllib.parse import unquote

import streamlit as st

from db import (
    list_shared_months,
    list_team_requests,
    load_app_settings,
    load_schedule,
    resolve_share_token,
    submit_team_request,
)
from nobet_core import (
    TALEP_METIN_SINIRI,
    TALEP_TURLERI,
    TR_AY_ADLARI,
    TR_GUN_KISA,
    add_overflow_roles,
    as_bool_frame,
    build_person_ics,
    escape_html,
    now_tr,
    on_duty_summary,
    format_day_ranges,
    parse_schedule_import_date,
    parse_team_request,
    request_open_months,
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


# Paylaşım sayfası girişsizdir; aynı listeyi art arda isteyenler veritabanını yormasın
# diye ay listesi ve ay satırları kısa süre önbellekte tutulur (jeton her seferinde sorulur).
@st.cache_data(ttl=10, show_spinner=False)
def _cached_shared_months(owner):
    return list_shared_months(owner)


@st.cache_data(ttl=10, show_spinner=False)
def _cached_shared_rows(owner, year, month):
    return _shared_rows(owner, year, month)


def render_shared_view(token, db_ready):
    st.markdown("<h1 style='text-align: center; font-size: 2.2rem;'>Nöbet Listesi</h1>", unsafe_allow_html=True)
    _target = resolve_share_token(token) if db_ready else None
    if not _target:
        st.error("Bu paylaşım bağlantısı geçersiz ya da kapatılmış.")
        return
    _owner, _year, _month = _target
    _team = _team_of(_owner)
    if not _team:
        # Ekip girilmemişse eskisi gibi yalnızca liste (talep gönderilemez).
        _render_list(_owner, _year, _month, None)
        return
    # Önce ad sorulur; liste ve talep sekmeleri o kişiye göre açılır. Seçilen
    # ad telefonda bu link için çerezde saklanır, sonraki açılışta hazır gelir.
    # (localStorage bileşeni açılışta sayfayı bir kez daha çalıştırıp formu bozuyordu.)
    _cookie = "nobet_ad_" + "".join(_c for _c in str(token) if _c.isalnum())[:16]
    if "paylasim_ad_kayitli" not in st.session_state:
        try:
            st.session_state.paylasim_ad_kayitli = unquote(st.context.cookies.get(_cookie) or "")
        except Exception:
            st.session_state.paylasim_ad_kayitli = ""
        if st.session_state.paylasim_ad_kayitli in _team:
            st.session_state.paylasim_kisi = st.session_state.paylasim_ad_kayitli
    _person = st.selectbox("Adın:", _team, index=None, placeholder="Adını seç", key="paylasim_kisi")
    if not _person:
        st.info("Nöbetlerini görmek ve talep göndermek için adını seç.")
        return
    _tab_list, _tab_request = st.tabs(["📋 Nöbet Listesi", "📝 Talep Gönder"])
    with _tab_list:
        _render_list(_owner, _year, _month, _person)
    with _tab_request:
        _render_request_form(_owner, _person)
    # Sekmelerden sonra yazılır: üstte geçici bir öğe sekmeleri ilk sekmeye döndürürdü.
    if _person and _person != st.session_state.paylasim_ad_kayitli:
        st.session_state.paylasim_ad_kayitli = _person
        st.html(
            "<script>document.cookie = " + json.dumps(_cookie + "=")
            + " + encodeURIComponent(" + json.dumps(_person).replace("<", "\\u003c")
            + ") + '; path=/; max-age=31536000; SameSite=Lax';</script>",
            unsafe_allow_javascript=True,
        )


def _team_of(_owner):
    _settings = load_app_settings(owner=_owner) or {}
    return list(dict.fromkeys(
        _name.strip() for _name in str(_settings.get("team") or "").split(",") if _name.strip()
    ))


def _render_list(_owner, _year, _month, _person):
    _months = _cached_shared_months(_owner)
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
    _rows, _roles = _cached_shared_rows(_owner, _year, _month)
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
        _next_rows, _next_roles = _cached_shared_rows(_owner, _tomorrow.year, _tomorrow.month)
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
    if _person:
        _selected = "Tümü" if st.toggle("Tüm listeyi göster", key="paylasim_tumu") else _person
    else:
        _selected = st.selectbox("Kendi nöbetlerini görmek için adını seç:", ["Tümü"] + _people) if _people else "Tümü"
    _states = shared_day_states(_rows, _year, _month, _today)
    _shown = [
        (_row, _state) for _row, _state in zip(_rows, _states)
        if _selected == "Tümü" or any(_row.get(_role) == _selected for _role in _roles)
    ]
    if not _shown:
        st.info(f"{TR_AY_ADLARI[_month]} {_year} listesinde nöbetin yok.")
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
    st.caption("Liste, liste sahibi değişiklik yaptıkça güncellenir.")


# Kutu başlıkları: ekibin anlayacağı dille, parantezde uygulamadaki adı.
_REQUEST_LABELS = {
    "tercih": "👍 Nöbet istediğim günler (Tercih)",
    "kacin": "🤏 Verilebilir ama kaçınılsın (Kaçın)",
    "izin": "🏖️ İzinli olduğum günler (İzin)",
    "musait_degil": "🚫 Nöbet tutamadığım günler (Müsait Değil)",
}


def _render_request_form(_owner, _person):
    st.caption(
        "Günleri virgülle ayırın, aralık için tire kullanın: 3,5,10-14. "
        "Talebin liste sahibine gider; onaylanınca çizelgeye işlenir."
    )
    _today = now_tr().date()
    _periods = request_open_months(_today)
    # Çoğunlukla gelecek ayın listesi hazırlanır; o ay önce seçili gelir.
    _year, _month = st.selectbox(
        "Hangi ay:", _periods, index=1 if len(_periods) > 1 else 0,
        format_func=lambda _p: f"{TR_AY_ADLARI[_p[1]]} {_p[0]}", key="talep_ay",
    )
    _done = st.session_state.get("talep_gonderildi")
    if _done == (_person, _year, _month):
        st.success("✅ Talebin iletildi. Liste sahibi onaylayınca çizelgeye işlenir; "
                   "o zamana kadar buradan güncelleyebilirsin.")
    _pending = next((
        _r for _r in list_team_requests(_owner)
        if (_r["person"], _r["year"], _r["month"]) == (_person, _year, _month)
    ), None)
    if _pending and _done != (_person, _year, _month):
        st.info("Bu ay için onay bekleyen talebin aşağıda. Değiştirip güncellersen eskisinin yerine geçer.")
    _prefill = (_pending or {}).get("days") or {}
    with st.form(f"talep_{_person}_{_year}_{_month}", border=False):
        _texts = {
            _key: st.text_input(
                _REQUEST_LABELS[_key],
                value=format_day_ranges(_prefill.get(_key) or []).replace(" ", ""),
                max_chars=TALEP_METIN_SINIRI,
                placeholder="Örn: 3,5,10-14",
                key=f"talep_{_key}_{_person}_{_year}_{_month}",
            )
            for _key, _, _ in TALEP_TURLERI
        }
        # Bekleyen talep varsa yenisi onun yerine geçer (iki talep oluşmaz).
        _sent = st.form_submit_button(
            "🔄 Talebi güncelle" if _pending else "📨 Talebi gönder", type="primary", width="stretch",
        )
    if not _sent:
        return
    _days, _errors = parse_team_request(_texts, _year, _month, now_tr().date())
    if _errors:
        st.error("Talep gönderilmedi:\n\n" + "\n\n".join(f"- {_e}" for _e in _errors))
        return
    _result = submit_team_request(_owner, _year, _month, _person, _days)
    if _result == "ok":
        st.session_state.talep_gonderildi = (_person, _year, _month)
        st.rerun()
    elif _result == "limit":
        st.error("Bekleyen talep sayısı doldu; liste sahibi eski talepleri onaylayınca tekrar deneyin.")
    else:
        st.error("Talep kaydedilemedi. Biraz sonra tekrar deneyin.")


def _shared_rows(_owner, year, month):
    """Paylaşım sayfası için ayın satırları ve görevleri: (satırlar, görevler); yoksa (None, None)."""
    _loaded = load_schedule(f"Otomatik_{year}_{month:02d}", year, month, owner=_owner, include_preferences=False)
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
