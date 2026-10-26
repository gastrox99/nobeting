# -*- coding: utf-8 -*-
"""Ekip talepleri: paylaşım linkinden gelen tercih/izin istekleri.

Link girişsiz olduğu için talepler çizelgeye doğrudan yazılmaz; liste sahibi
burada onaylar (tercihlere işlenir) ya da reddeder; sonuç talep sahibine paylaşım linkinde görünür.
"""
import time

import streamlit as st

from db import decide_team_request, list_team_requests
from nobet_core import (
    TR_AY_ADLARI,
    apply_team_request,
    escape_html,
    resolve_alias,
    team_request_summary,
)

_CACHE_KEY = "team_requests_cache"
_CACHE_SECONDS = 60
# Onaylanıp tercihlere işlenen, ay kaydedilince onaylandı olarak işaretlenecek talepler: [(id, yıl, ay)].
_APPROVED_KEY = "team_requests_approved"


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
    if _changed:
        # Tercih Belirle modunda da bekletmeden kaydedilsin. Talep, ay gerçekten
        # kaydedilince onaylandı olarak işaretlenir (finish_approved_requests); kayıt başka yerdeki
        # bir değişiklik yüzünden durursa talep kaybolmaz, yeniden onaylanabilir.
        st.session_state.save_preferences_now = True
        st.session_state.setdefault(_APPROVED_KEY, []).append(
            (request["id"], request["year"], request["month"])
        )
    else:
        # Tercihler zaten talep gibiymiş; kaydedilecek bir şey yok.
        decide_team_request(owner, request["id"], "approved")
        st.session_state.pop(_CACHE_KEY, None)
    st.toast(f"{person} talebi onaylandı ({_changed} gün işlendi).", icon="✅")


def _reject(owner, request):
    decide_team_request(owner, request["id"], "rejected")
    st.session_state.pop(_CACHE_KEY, None)
    st.toast(f"{request['person']} talebi reddedildi.", icon="🗑️")


def finish_approved_requests(owner, yil, ay, saved):
    """Kayıt denendikten sonra çağrılır: kaydedildiyse onaylanan talepleri işaretler,
    kaydedilemediyse talepler listede yeniden görünür."""
    _approved = st.session_state.pop(_APPROVED_KEY, None)
    if not _approved:
        return
    st.session_state.pop(_CACHE_KEY, None)
    if saved:
        for _id, _y, _m in _approved:
            if (_y, _m) == (yil, ay):
                decide_team_request(owner, _id, "approved")
    else:
        # Liste bu çalışmada talep gizlenerek çizildi; yeniden çiz ki görünsün.
        st.rerun()


def render_pending_badge(owner):
    """Ay çubuğunun altında onay bekleyen talep sayısı; Ayarlar açıkken de görünür."""
    _n = len(_requests(owner))
    if _n:
        st.caption(f"🔔 {_n} ekip talebi onay bekliyor")


def render_team_requests(owner, yil, ay, isimler, save_undo_state):
    # Onaylanıp kaydı beklenenler listede ikinci kez görünmesin.
    _approved_ids = {_id for _id, _, _ in st.session_state.get(_APPROVED_KEY) or []}
    _all = [_r for _r in _requests(owner) if _r["id"] not in _approved_ids]
    if not _all:
        return
    _here = [_r for _r in _all if (_r["year"], _r["month"]) == (yil, ay)]
    _others = {}
    for _r in _all:
        if (_r["year"], _r["month"]) != (yil, ay):
            _others[(_r["year"], _r["month"])] = _others.get((_r["year"], _r["month"]), 0) + 1
    with st.expander(f"📥 Ekip talepleri ({len(_all)})", expanded=bool(_here)):
        # Ay kaydedilemiyorsa (başka yerde değişmiş ya da okunamamış) onaylanan
        # talep yalnızca bu sekmede kalıp silinirdi; onay o zaman kapalı.
        _blocked = st.session_state.get("autosave_blocked_period") == (yil, ay)
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
                elif _blocked:
                    st.caption("Bu ay şu an kaydedilemiyor; sayfayı yenileyip tekrar onaylayın.")
            with _cols[1]:
                st.button("✅ Onayla", key=f"talep_onay_{_r['id']}", disabled=not _known or _blocked, width="stretch",
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
