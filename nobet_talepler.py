# -*- coding: utf-8 -*-
"""Ekip talepleri: paylaşım linkinden gelen tercih/izin istekleri.

Link girişsiz olduğu için talepler çizelgeye doğrudan yazılmaz; liste sahibi
burada onaylar (tercihlere işlenir) ya da reddeder (silinir).
"""
import time

import streamlit as st

from db import delete_team_request, list_team_requests
from nobet_core import (
    TR_AY_ADLARI,
    apply_team_request,
    escape_html,
    resolve_alias,
    team_request_summary,
)

_CACHE_KEY = "team_requests_cache"
_CACHE_SECONDS = 60


def _requests(owner):
    _cached = st.session_state.get(_CACHE_KEY)
    if not _cached or time.monotonic() - _cached[0] > _CACHE_SECONDS:
        _cached = (time.monotonic(), list_team_requests(owner))
        st.session_state[_CACHE_KEY] = _cached
    return _cached[1]


def _approve(owner, request, person, save_undo_state):
    save_undo_state()
    st.session_state.pref_df, _changed = apply_team_request(
        st.session_state.pref_df, person, request["days"]
    )
    # Tercih Belirle modunda da bekletmeden kaydedilsin.
    st.session_state.save_preferences_now = True
    delete_team_request(owner, request["id"])
    st.session_state.pop(_CACHE_KEY, None)
    st.toast(f"{person} talebi onaylandı ({_changed} gün işlendi).", icon="✅")


def _reject(owner, request):
    delete_team_request(owner, request["id"])
    st.session_state.pop(_CACHE_KEY, None)
    st.toast(f"{request['person']} talebi reddedildi.", icon="🗑️")


def render_team_requests(owner, yil, ay, isimler, save_undo_state):
    _all = _requests(owner)
    if not _all:
        return
    _here = [_r for _r in _all if (_r["year"], _r["month"]) == (yil, ay)]
    _others = {}
    for _r in _all:
        if (_r["year"], _r["month"]) != (yil, ay):
            _others[(_r["year"], _r["month"])] = _others.get((_r["year"], _r["month"]), 0) + 1
    with st.expander(f"📥 Ekip talepleri ({len(_all)})", expanded=bool(_here)):
        if not _here:
            st.caption("Bu ay için bekleyen talep yok.")
        for _r in _here:
            _person = resolve_alias(_r["person"], st.session_state.get("name_aliases"), isimler)
            _known = _person in isimler
            _lines = " · ".join(
                f"<b>{escape_html(_label)}:</b> {escape_html(_text)}"
                for _label, _text in team_request_summary(_r["days"])
            )
            _when = _r["created_tr"].strftime("%d.%m %H:%M") if _r.get("created_tr") else ""
            _cols = st.columns([6, 1, 1], vertical_alignment="center")
            with _cols[0]:
                st.markdown(
                    f"<div><b>{escape_html(_person)}</b> "
                    f"<span style='opacity:.6;font-size:12px;'>{escape_html(_when)}</span></div>"
                    f"<div style='font-size:14px;'>{_lines}</div>",
                    unsafe_allow_html=True,
                )
                if not _known:
                    st.caption("Bu isim ekipte yok; yalnızca reddedilebilir.")
            with _cols[1]:
                st.button("✅ Onayla", key=f"talep_onay_{_r['id']}", disabled=not _known, width="stretch",
                          on_click=_approve, args=(owner, _r, _person, save_undo_state))
            with _cols[2]:
                st.button("❌ Reddet", key=f"talep_ret_{_r['id']}", width="stretch",
                          on_click=_reject, args=(owner, _r))
        if _others:
            st.caption(
                "Diğer aylarda bekleyen talepler: "
                + ", ".join(f"{TR_AY_ADLARI[_m]} {_y} ({_n})" for (_y, _m), _n in sorted(_others.items()))
                + ". O aya geçince burada görünür."
            )
        st.button("🔄 Talepleri yenile", key="talep_yenile",
                  on_click=lambda: st.session_state.pop(_CACHE_KEY, None))
