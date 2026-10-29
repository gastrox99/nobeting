"""Ekip talepleri: gün ayrıştırma, özet ve tercih tablosuna işleme."""

from datetime import date

import pandas as pd

from nobet_core import (
    apply_team_request,
    format_day_ranges,
    parse_team_request,
    request_open_months,
    team_request_summary,
)

TODAY = date(2026, 10, 19)


def test_acik_aylar_bu_ay_ve_sonraki_iki_ay():
    assert request_open_months(TODAY) == [(2026, 10), (2026, 11), (2026, 12)]
    assert request_open_months(date(2026, 12, 5)) == [(2026, 12), (2027, 1), (2027, 2)]


def test_kutular_ayrisir():
    days, errors = parse_team_request(
        {"tercih": "3,5,10-12", "kacin": " 20 ", "izin": "", "musait_degil": "28-30"}, 2026, 11, TODAY,
    )
    assert errors == []
    assert days == {"tercih": [3, 5, 10, 11, 12], "kacin": [20], "musait_degil": [28, 29, 30]}


def test_hatali_parca_ve_ay_disi_gun_reddedilir():
    days, errors = parse_team_request({"tercih": "3,abc", "izin": "30-31"}, 2026, 11, TODAY)
    assert days == {}
    assert any("'abc'" in e for e in errors)
    assert any("'30-31'" in e for e in errors)


def test_gecmis_ay_ve_gecmis_gun_kapali():
    assert parse_team_request({"tercih": "3"}, 2026, 9, TODAY)[1] == ["Bu ay için talep gönderilemez."]
    assert parse_team_request({"tercih": "3"}, 2027, 2, TODAY)[1] == ["Bu ay için talep gönderilemez."]
    days, errors = parse_team_request({"izin": "15-21"}, 2026, 10, TODAY)
    assert days == {} and any("geçmiş günler" in e and "15-18" in e for e in errors)
    assert parse_team_request({"izin": "19-21"}, 2026, 10, TODAY) == ({"izin": [19, 20, 21]}, [])


def test_ayni_gun_iki_kutuda_olamaz_ve_bos_talep():
    days, errors = parse_team_request({"tercih": "5", "izin": "4-6"}, 2026, 11, TODAY)
    assert days == {} and any("5. gün" in e for e in errors)
    assert parse_team_request({"tercih": " "}, 2026, 11, TODAY)[1] == ["En az bir kutuya gün yazın."]


def test_uzun_metin_reddedilir():
    assert parse_team_request({"tercih": "1," * 60}, 2026, 11, TODAY)[0] == {}


def test_ozet_araliklari_birlestirir():
    assert format_day_ranges([12, 3, 10, 11, 5]) == "3, 5, 10-12"
    assert format_day_ranges([]) == ""
    assert team_request_summary({"izin": [1, 2, 3], "tercih": [7]}) == [("Tercih", "7"), ("İzin", "1-3")]


def test_onay_yalnizca_yazili_gunleri_degistirir():
    pref = pd.DataFrame(0, index=["Ali", "Ayşe"], columns=["1 Paz", "2 Pzt", "3 Sal"])
    pref.at["Ali", "3 Sal"] = 2
    result, changed = apply_team_request(pref, "Ali", {"tercih": [1], "izin": [2], "musait_degil": [31]})
    assert changed == 2
    assert result.loc["Ali"].tolist() == [1, 3, 2]
    assert result.loc["Ayşe"].tolist() == [0, 0, 0]
    assert pref.loc["Ali"].tolist() == [0, 0, 2]  # özgün tablo değişmez
    assert apply_team_request(pref, "Yok", {"tercih": [1]})[1] == 0


def _istek(person, year=2026, month=10):
    return {"id": 1, "year": year, "month": month, "person": person, "days": {}, "status": "pending"}


def test_talep_durumu_gonderen_ve_gondermeyeni_ekip_sirasiyla_ayirir():
    from nobet_talepler import talep_durumu
    istekler = [_istek("Can"), _istek("Ali", 2026, 11), _istek("Eski Ad")]
    gonderen, eksik = talep_durumu(["Ali", "Ayşe", "Can", "Yeni Ad"], istekler, 2026, 10, {"Eski Ad": "Yeni Ad"})
    assert gonderen == ["Can", "Yeni Ad"]          # başka aydaki talep sayılmaz, eski ad güncel adıyla
    assert eksik == ["Ali", "Ayşe"]


def test_hatirlatma_metni_ay_baglanti_ve_eksikleri_icerir():
    from nobet_talepler import hatirlatma_metni
    metin = hatirlatma_metni(["Ali", "Ayşe"], 2026, 11, "https://x.app?paylas=abc")
    assert "Kasım 2026" in metin and "https://x.app?paylas=abc" in metin
    assert "Ali, Ayşe" in metin
    assert "Henüz göndermeyenler" not in hatirlatma_metni([], 2026, 11, "https://x.app")


def test_uygulamada_talep_durumu_gonderenleri_sayar():
    from test_bozuk_kayit import _app, _stop, _october_columns, _corrupt_month_loader
    at, patches = _app(
        _corrupt_month_loader(_october_columns()), lambda *a, **k: True, team="Ali, Ayşe, Can",
        list_team_requests=lambda *a, **k: [dict(_istek("Can"), created_tr=None)],
        get_share_link=lambda *a: "tok123",
    )
    try:
        at.run()
        assert [e.value for e in at.exception] == []
        etiketler = [e.label for e in at.expander]
    finally:
        _stop(patches)
    assert any("Talep durumu (1/3 gönderdi)" in etiket for etiket in etiketler)
