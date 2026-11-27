# -*- coding: utf-8 -*-
"""Kişi başı nöbet sınırları (en fazla nöbet / hafta sonu) aya özeldir:
bu ay için yazılan sınır önceki ayda görünmez; yeni aya önceki ayınkiler kopyalanır."""
import copy

from nobet_core import apply_app_settings, collect_app_settings
from nobet_ekip import constraints_for_period


def test_sinir_ayrilir_yeni_aya_kopyalanir_eski_aya_donunce_kendi_siniri_gelir():
    ss = {"limits_text": "Ali:0-6"}
    constraints_for_period(ss, "2026-10")
    ss["limits_text"] = "Ali:0-4"               # Ekim için yazıldı
    constraints_for_period(ss, "2026-11")               # yeni ay: Ekim'inkiler kopyalanır
    assert ss["limits_text"] == "Ali:0-4"
    ss["limits_text"] = "Ali:0-3"               # Kasım'a özel
    constraints_for_period(ss, "2026-10")
    assert ss["limits_text"] == "Ali:0-4"       # Ekim Kasım'dan etkilenmedi
    constraints_for_period(ss, "2026-11")
    assert ss["limits_text"] == "Ali:0-3"


def test_eski_kayit_ilk_acilista_bugunku_siniri_devralir_ve_ayarla_kaydolur():
    ss = {}
    apply_app_settings(ss, {"limits": "Ali:0-5", "month": 10, "year": 2026})
    constraints_for_period(ss, "2026-10")
    constraints_for_period(ss, "2026-09")               # hiç kaydı olmayan eski ay: bugünküyü devralır
    assert ss["limits_text"] == "Ali:0-5"
    saved = collect_app_settings(ss)
    assert saved["limits_by_period"] == {"0000-00": "Ali:0-5", "2026-10": "Ali:0-5", "2026-09": "Ali:0-5"}
    ss2 = {}
    apply_app_settings(ss2, copy.deepcopy(saved))
    ss2["limits_text"] = saved["limits"]
    ss2["limits_by_period"]["2026-09"] = "Ali:0-2"
    constraints_for_period(ss2, "2026-09")
    assert ss2["limits_text"] == "Ali:0-2"


def test_bu_ay_yazilan_sinir_hic_gorulmemis_onceki_aya_yansimaz():
    ss = {"limits_text": "Ali:0-6"}            # eski tek ayar
    constraints_for_period(ss, "2026-12")
    ss["limits_text"] = "Ali:0-4"               # Aralık için yazıldı
    constraints_for_period(ss, "2026-11")               # önceki ay: eski ayar, yeni sınır değil
    assert ss["limits_text"] == "Ali:0-6"
    constraints_for_period(ss, "2026-12")
    assert ss["limits_text"] == "Ali:0-4"
    constraints_for_period(ss, "2027-01")               # ileri yeni ay: Aralık'ınki kopyalanır
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


def test_yasak_birlikte_ve_sabit_gorev_de_aya_ozel():
    ss = {"forbidden_pairs_text": "Ali-Can", "together_pairs_text": "", "fixed_roles": {"Ali": "AYB"}}
    constraints_for_period(ss, "2026-12")
    ss["forbidden_pairs_text"] = "Ali-Deniz"
    ss["together_pairs_text"] = "Can-Ece"
    ss["fixed_roles"] = {"Ali": "GYB"}
    constraints_for_period(ss, "2026-11")              # önceki ay: eski ayarlar
    assert ss["forbidden_pairs_text"] == "Ali-Can"
    assert ss["together_pairs_text"] == ""
    assert ss["fixed_roles"] == {"Ali": "AYB"}
    ss["fixed_roles"]["Can"] = "AYB"                   # yerinde değişiklik Aralık'a sızmaz
    constraints_for_period(ss, "2026-12")
    assert ss["forbidden_pairs_text"] == "Ali-Deniz" and ss["together_pairs_text"] == "Can-Ece"
    assert ss["fixed_roles"] == {"Ali": "GYB"}
    constraints_for_period(ss, "2027-01")              # yeni ay: Aralık kopyalanır
    assert ss["fixed_roles"] == {"Ali": "GYB"}
    ss["fixed_roles"]["Ece"] = "AYB"
    constraints_for_period(ss, "2026-12")
    assert ss["fixed_roles"] == {"Ali": "GYB"}


def test_aylik_kisitlar_ayarla_kaydolur_ve_geri_yuklenir():
    ss = {"fixed_roles": {"Ali": "AYB"}, "forbidden_pairs_text": "Ali-Can"}
    constraints_for_period(ss, "2026-12")
    saved = collect_app_settings(ss)
    assert saved["fixed_roles_by_period"]["2026-12"] == {"Ali": "AYB"}
    assert saved["forbidden_by_period"]["2026-12"] == "Ali-Can"
    ss2 = {}
    apply_app_settings(ss2, copy.deepcopy(saved))
    assert ss2["fixed_roles_by_period"] == saved["fixed_roles_by_period"]


def test_ad_degisikligi_diger_aylarin_kisitlarina_da_uygulanir():
    from nobet_ekip import rename_in_other_periods
    ss = {"limits_text": "Ali:0-4", "forbidden_pairs_text": "Ali-Can", "fixed_roles": {"Ali": "AYB"}}
    constraints_for_period(ss, "2026-12")
    constraints_for_period(ss, "2027-01")
    constraints_for_period(ss, "2026-12")
    ss["limits_text"], ss["forbidden_pairs_text"], ss["fixed_roles"] = "Ali Y:0-4", "Ali Y-Can", {"Ali Y": "AYB"}
    rename_in_other_periods(ss, ["Ali", "Can"], {"Ali": "Ali Y"}, "2026-12")
    constraints_for_period(ss, "2027-01")
    assert ss["limits_text"] == "Ali Y:0-4"
    assert ss["forbidden_pairs_text"] == "Ali Y-Can"
    assert ss["fixed_roles"] == {"Ali Y": "AYB"}
    constraints_for_period(ss, "2026-10")              # eski aylar da yeni adı taşır
    assert ss["limits_text"] == "Ali Y:0-4"


def test_uygulamada_yasak_cift_ay_oku_ile_aya_ozel_kalir():
    from test_bozuk_kayit import _app, _stop
    at, patches = _app(lambda *a, **k: (None, None, None, None, None), lambda *a, **k: True,
                       team="Ali, Ayşe, Can")
    at.session_state["ay_val"] = 11
    try:
        at.run()
        at.session_state["forbidden_pairs_text"] = "Ali-Can"      # Kasım için yazıldı
        at.run()
        assert at.session_state["forbidden_pairs"] == {("Ali", "Can")}
        at.button(key="month_prev").click().run()
        assert [e.value for e in at.exception] == []
        assert at.session_state["forbidden_pairs"] == set()
        at.button(key="month_next").click().run()
        assert at.session_state["forbidden_pairs"] == {("Ali", "Can")}
    finally:
        _stop(patches)
