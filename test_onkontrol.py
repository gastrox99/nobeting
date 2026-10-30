# -*- coding: utf-8 -*-
"""Simülasyon öncesi ön kontrol (nobet_onkontrol.py)."""
import pandas as pd

from nobet_onkontrol import onkontrol_uyarilari

GUNLER = [f"{d} Çar" for d in range(1, 11)]


def _izinler(**kisiler):
    frame = pd.DataFrame(False, index=list(kisiler), columns=GUNLER)
    for person, days in kisiler.items():
        for day in days:
            frame.at[person, GUNLER[day - 1]] = True
    return frame


def test_yeterli_kisi_varsa_uyari_yok():
    assert onkontrol_uyarilari(_izinler(Ali=[], Ayşe=[], Can=[], Ece=[]), GUNLER, 2, 1) == []


def test_musait_kisi_yetmeyen_gunler_listelenir():
    uyari = onkontrol_uyarilari(_izinler(Ali=[3, 4], Ayşe=[4], Can=[4, 5], Ece=[]), GUNLER, 2, 0)
    assert len(uyari) == 1 and "3. gün" not in uyari[0] and "5. gün" not in uyari[0]
    assert "4. gün (1 müsait)" in uyari[0]


def test_toplam_kapasite_ve_limit_hesabi():
    # 10 gün x 2 kişi = 20 nöbet gerekir; dinlenme 1 gün → kişi başı en fazla 5; limitler 2'şer → kapasite 8.
    limits = {p: {"max": 2} for p in ("Ali", "Ayşe", "Can", "Ece")}
    uyari = onkontrol_uyarilari(_izinler(Ali=[], Ayşe=[], Can=[], Ece=[]), GUNLER, 2, 1, limits)
    assert any("kapasitesi 8" in u and "gereken 20" in u and "en az 12" in u for u in uyari)


def test_paylasim_kontrolleri_hepsi_yesilse_bos():
    from nobet_onkontrol import paylasim_kontrolleri
    assert paylasim_kontrolleri(0, [], None) == []
    uyari = paylasim_kontrolleri(2, [14, 22], "Katsayı eski.")
    assert uyari[0].startswith("2 ekip talebi") and "14, 22" in uyari[1] and uyari[2] == "Katsayı eski."


def test_neden_yok_cevabi_kisiyi_bulur():
    from nobet_onkontrol import neden_yok_cevabi
    reasons = [("dinlenmede", ["Ali (13. gün nöbetli)", "Can (geçen ayın sonunda nöbetli)"]),
               ("uygun, elle atanabilir", ["Ece"])]
    assert neden_yok_cevabi(reasons, "Ali") == "Ali: dinlenmede (13. gün nöbetli)"
    assert neden_yok_cevabi(reasons, "Ece") == "Ece: uygun, elle atanabilir"
    assert neden_yok_cevabi(reasons, "Ayşe") == "Ayşe o gün zaten nöbetçi."


def test_ics_satir_sonu_kacirilir():
    from nobet_core import _ics_escape
    assert _ics_escape("a\r\nb\rc\nd") == "a\\nb\\nc\\nd"


def test_ay_sonu_cakismasi_sonraki_ayin_ilk_nobetiyle_olculur():
    from nobet_onkontrol import ay_sonu_cakismalari, ilk_nobet_gunleri
    # Ekim 31 gün, dinlenme 2: 30. gün nöbeti + Kasım 1 → fark 2 → çakışır; Kasım 3 → fark 4 → uygun.
    uyari = ay_sonu_cakismalari({"Ali": 30, "Ayşe": 30, "Can": -2}, {"Ali": 1, "Ayşe": 3, "Can": 1}, 31, 2)
    assert len(uyari) == 1 and "Ali" in uyari[0] and "30. gün" in uyari[0] and "1. günü" in uyari[0]
    frame = pd.DataFrame(False, index=["Ali", "Ayşe"], columns=["2 Cum", "5 Pzt"])
    frame.at["Ali", "5 Pzt"] = True
    frame.at["Ali", "2 Cum"] = True
    assert ilk_nobet_gunleri(frame) == {"Ali": 2}


def test_yedek_adaylari_hafta_sonu_az_tutana_oncelik_verir():
    from nobet_core import build_gun_detaylari
    from nobet_onkontrol import yedek_adaylari
    days = build_gun_detaylari(2026, 11, 30, [])
    cols = list(days)
    names = ["Ali", "Ayşe", "Can", "Deniz", "Ece", "Fuat"]
    schedule = pd.DataFrame(False, index=names, columns=cols)
    unwanted = pd.DataFrame(False, index=names, columns=cols)
    col = "14 Cmt"
    for p in ("Ali", "Fuat"):
        schedule.at[p, col] = True
    schedule.at["Ayşe", "7 Cmt"] = True      # 1 hafta sonu
    for c in ("2 Pzt", "3 Sal", "4 Çar"):
        schedule.at["Can", c] = True         # 0 hafta sonu, 3 nöbet
    schedule.at["Ece", "5 Per"] = True       # 0 hafta sonu, 1 nöbet
    unwanted.at["Deniz", col] = True
    adaylar, nedenler = yedek_adaylari(schedule, col, "Ali", days, unwanted, 1)
    assert adaylar == ["Ece", "Can", "Ayşe"]
    assert ("izinli / müsait değil", ["Deniz"]) in nedenler
    assert schedule.at["Ali", col]          # öneri çizelgeyi değiştirmez


def test_telafi_gunleri_ayni_ay_icinde_sonraki_gunlerden_yedegin_nobeti():
    from nobet_core import build_gun_detaylari
    from nobet_onkontrol import telafi_gunleri
    days = build_gun_detaylari(2026, 11, 30, [])
    names = ["Ali", "Ece"]
    schedule = pd.DataFrame(False, index=names, columns=list(days))
    unwanted = pd.DataFrame(False, index=names, columns=list(days))
    schedule.at["Ece", "14 Cmt"] = True      # yedek yazıldı (Ali çıktı)
    for c in ("12 Per", "19 Per", "26 Per", "29 Pzt"):
        schedule.at["Ece", c] = True
    schedule.at["Ali", "18 Çar"] = True      # 19. günle dinlenme çakışır
    unwanted.loc["Ali", "29 Pzt"] = True      # müsait değil
    assert telafi_gunleri(schedule, "14 Cmt", "Ali", "Ece", days, unwanted, 1) == ["26 Per"]
