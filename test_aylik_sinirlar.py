# -*- coding: utf-8 -*-
"""Kişi başı nöbet sınırları (en fazla nöbet / hafta sonu) aya özeldir:
bu ay için yazılan sınır önceki ayda görünmez; yeni aya önceki ayınkiler kopyalanır."""
import copy

from nobet_core import apply_app_settings, collect_app_settings
from nobet_ekip import limits_for_period


def test_sinir_ayrilir_yeni_aya_kopyalanir_eski_aya_donunce_kendi_siniri_gelir():
    ss = {"limits_text": "Ali:0-6"}
    limits_for_period(ss, "2026-10")
    ss["limits_text"] = "Ali:0-4"               # Ekim için yazıldı
    limits_for_period(ss, "2026-11")               # yeni ay: Ekim'inkiler kopyalanır
    assert ss["limits_text"] == "Ali:0-4"
    ss["limits_text"] = "Ali:0-3"               # Kasım'a özel
    limits_for_period(ss, "2026-10")
    assert ss["limits_text"] == "Ali:0-4"       # Ekim Kasım'dan etkilenmedi
    limits_for_period(ss, "2026-11")
    assert ss["limits_text"] == "Ali:0-3"


def test_eski_kayit_ilk_acilista_bugunku_siniri_devralir_ve_ayarla_kaydolur():
    ss = {}
    apply_app_settings(ss, {"limits": "Ali:0-5", "month": 10, "year": 2026})
    limits_for_period(ss, "2026-10")
    limits_for_period(ss, "2026-09")               # hiç kaydı olmayan eski ay: bugünküyü devralır
    assert ss["limits_text"] == "Ali:0-5"
    saved = collect_app_settings(ss)
    assert saved["limits_by_period"] == {"0000-00": "Ali:0-5", "2026-10": "Ali:0-5", "2026-09": "Ali:0-5"}
    ss2 = {}
    apply_app_settings(ss2, copy.deepcopy(saved))
    ss2["limits_text"] = saved["limits"]
    ss2["limits_by_period"]["2026-09"] = "Ali:0-2"
    limits_for_period(ss2, "2026-09")
    assert ss2["limits_text"] == "Ali:0-2"


def test_bu_ay_yazilan_sinir_hic_gorulmemis_onceki_aya_yansimaz():
    ss = {"limits_text": "Ali:0-6"}            # eski tek ayar
    limits_for_period(ss, "2026-12")
    ss["limits_text"] = "Ali:0-4"               # Aralık için yazıldı
    limits_for_period(ss, "2026-11")               # önceki ay: eski ayar, yeni sınır değil
    assert ss["limits_text"] == "Ali:0-6"
    limits_for_period(ss, "2026-12")
    assert ss["limits_text"] == "Ali:0-4"
    limits_for_period(ss, "2027-01")               # ileri yeni ay: Aralık'ınki kopyalanır
    assert ss["limits_text"] == "Ali:0-4"


def test_uygulamada_ay_oku_sinirlari_aya_ozel_tutar():
    from test_bozuk_kayit import _app, _stop
    at, patches = _app(lambda *a, **k: (None, None, None, None, None), lambda *a, **k: True, team="Ali, Ayşe")
    at.session_state["ay_val"] = 11
    at.session_state["limits_text"] = "Ali:0-6"        # eski tek ayar
    try:
        at.run()
        at.session_state["limits_text"] = "Ali:0-4"     # Kasım için yazıldı
        at.run()
        at.button(key="month_prev").click().run()
        assert [e.value for e in at.exception] == []
        assert at.session_state["ay_val"] == 10
        assert at.session_state["limits_text"] == "Ali:0-6"
        assert at.session_state["person_limits"]["Ali"]["max"] == 6
        at.button(key="month_next").click().run()
        assert at.session_state["limits_text"] == "Ali:0-4"
        assert at.session_state["person_limits"]["Ali"]["max"] == 4
    finally:
        _stop(patches)
