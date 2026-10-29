# -*- coding: utf-8 -*-
"""Kişi adı değişince ekrandaki verinin yeni adla taşınması (nobet_durum.apply_renames)."""
import pandas as pd
import streamlit as st

from nobet_durum import apply_renames


def _yukle():
    st.session_state.clear()
    cols = ["1 Per", "2 Cum"]
    st.session_state.schedule_bool = pd.DataFrame(
        [[True, False], [False, True]], index=["Ali", "Ayşe"], columns=cols)
    st.session_state.pref_df = pd.DataFrame(0, index=["Ali", "Ayşe"], columns=cols)
    st.session_state.locked_cells = {("Ali", "1 Per"), ("Ayşe", "2 Cum")}
    st.session_state.cached_rows_liste = [{"Tarih": "1 Per", "AYB": "Ali", "GYB": "Ayşe"}]
    st.session_state.cached_key = (2026, 10, ("Ali", "Ayşe"), 1, 2)
    st.session_state.prev_month_context = {
        "last_shift": {"Ali": 3},
        "history_counts": {"Ali": 2, "Ayşe": 1},
        "holiday_counts": {"Ayşe": 1},
        "monthly_totals": [{"Ali": 4}, {"Ayşe": 5}],
        "monthly_available": [{"Ali": 20}],
        "monthly_holidays": [{"Ayşe": 1}],
    }


def test_ad_degisikligi_ekrandaki_butun_veriyi_tasir():
    _yukle()
    apply_renames({"Ali": "Ali Can"})
    assert list(st.session_state.schedule_bool.index) == ["Ali Can", "Ayşe"]
    assert bool(st.session_state.schedule_bool.at["Ali Can", "1 Per"])
    assert list(st.session_state.pref_df.index) == ["Ali Can", "Ayşe"]
    assert st.session_state.locked_cells == {("Ali Can", "1 Per"), ("Ayşe", "2 Cum")}
    assert st.session_state.cached_rows_liste[0]["AYB"] == "Ali Can"
    assert st.session_state.cached_key[2] == ("Ali Can", "Ayşe")
    assert st.session_state.name_aliases == {"Ali": "Ali Can"}
    ctx = st.session_state.prev_month_context
    assert ctx["last_shift"] == {"Ali Can": 3}
    assert ctx["history_counts"] == {"Ali Can": 2, "Ayşe": 1}
    assert ctx["monthly_totals"] == [{"Ali Can": 4}, {"Ayşe": 5}]
    assert ctx["monthly_available"] == [{"Ali Can": 20}]
    assert ctx["monthly_holidays"] == [{"Ayşe": 1}]


def test_ad_degisikligi_olmayan_veriye_dokunmaz():
    _yukle()
    apply_renames({"Veli": "Veli Can"})
    assert list(st.session_state.schedule_bool.index) == ["Ali", "Ayşe"]
    assert st.session_state.locked_cells == {("Ali", "1 Per"), ("Ayşe", "2 Cum")}
    assert st.session_state.prev_month_context["last_shift"] == {"Ali": 3}
