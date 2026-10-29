# -*- coding: utf-8 -*-
"""Analiz > Yıl özeti (nobet_yil.py)."""
import json

import pandas as pd

import nobet_core
from nobet_yil import yil_ozeti


def _ay(year, month, atamalar):
    """atamalar: {kişi: [gün, ...]}; sütunlar nobet_core ile aynı biçimde."""
    gd = nobet_core.build_gun_detaylari(year, month, nobet_core.calendar.monthrange(year, month)[1], set())
    columns = list(gd)
    frame = pd.DataFrame(False, index=list(atamalar), columns=columns)
    for person, days in atamalar.items():
        for day in days:
            frame.at[person, columns[day - 1]] = True
    return frame, columns


def test_yil_ozeti_gun_turlerini_ve_tercih_karsilanmayi_sayar():
    # Ekim 2026: 2 Cum, 3 ve 10 Cmt, 4 Paz; Kasım 2026: 6 ve 13 Cum.
    ekim, ek_cols = _ay(2026, 10, {"Ali": [2, 3, 10], "Ayşe": [4]})
    kasim, ks_cols = _ay(2026, 11, {"Ali": [13], "Ayşe": [6]})
    prefs = {(2026, 10): json.dumps({ek_cols[2]: {"Ali": 1}, ek_cols[8]: {"Ali": 1}, ek_cols[3]: {"Ayşe": 2}})}
    tablo, aylar = yil_ozeti(
        {(2026, 10): ekim, (2026, 11): kasim, (2025, 12): ekim}, prefs, 2026, {}, ["Ali", "Ayşe"], {},
    )
    assert aylar == 2                                    # başka yılın kaydı sayılmaz
    ali, ayse = (tablo.set_index("Kişi").loc[p] for p in ("Ali", "Ayşe"))
    assert (ali["Nöbet"], ali["Cumartesi"], ali["Pazar"], ali["Cuma"]) == (4, 2, 0, 2)
    assert (ayse["Nöbet"], ayse["Cumartesi"], ayse["Pazar"], ayse["Cuma"]) == (2, 0, 1, 1)
    assert ali["Tercih karşılandı"] == "1/2"            # 3. gün istedi ve aldı, 9. gün istedi ve almadı
    assert ayse["Tercih karşılandı"] == "—"             # kaçın işareti tercih sayılmaz


def test_yil_ozeti_eski_ad_guncel_ada_cevrilir_ve_bos_yil_bos_tablo_verir():
    ekim, _ = _ay(2026, 10, {"Eski": [5]})
    tablo, aylar = yil_ozeti({(2026, 10): ekim}, {}, 2026, {"Eski": "Yeni"}, ["Yeni"], {})
    assert list(tablo["Kişi"]) == ["Yeni"] and aylar == 1
    bos, aylar = yil_ozeti({}, {}, 2026, {}, ["Yeni"], {})
    assert bos.empty and aylar == 0


def test_kisi_ve_ortalama_yalniz_kendi_satirini_verir():
    from nobet_yil import kisi_ve_ortalama
    ekim, _ = _ay(2026, 10, {"Ali": [2, 3], "Ayşe": [4]})
    tablo, _ = yil_ozeti({(2026, 10): ekim}, {}, 2026, {}, ["Ali", "Ayşe"], {})
    benim, ortalama = kisi_ve_ortalama(tablo, "Ali")
    assert benim["Nöbet"] == 2 and ortalama["Nöbet"] == 1.5
    assert "Kişi" not in benim and "Ayşe" not in str(benim)
    assert kisi_ve_ortalama(tablo, "Yok") is None
