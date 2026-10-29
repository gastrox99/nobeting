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
