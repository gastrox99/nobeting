# -*- coding: utf-8 -*-
"""Yüklenen yedek dosyasının doğrulanması (16 Kasım 2026 incelemesi).

Her ayın "ay ayarları" (kilitli nöbetler ve saklı kişiler) bölümü yalnızca
sözlük mü diye denetleniyordu; boyutu denetlenmediği için bu bölüme konan
megabaytlarca veri olduğu gibi veritabanına yazılıyordu.
"""
import json

from nobet_core import BACKUP_FORMAT, BACKUP_VERSION, parse_backup


def _yedek(ay_ayarlari, ay_sayisi=1):
    return json.dumps({
        "format": BACKUP_FORMAT, "version": BACKUP_VERSION, "settings": None,
        "schedules": [{
            "name": f"Otomatik_2026_{ay:02d}", "year": 2026, "month": ay,
            "team": ["Ali", "Ece"], "days": ["1 Sal", "2 Çar"], "assigned": [["Ali", "1 Sal"]],
            "preferences": None, "settings": ay_ayarlari, "rows": None,
        } for ay in range(1, ay_sayisi + 1)],
    }, ensure_ascii=False).encode("utf-8")


def test_ay_ayarlari_cok_buyukse_yedek_reddedilir():
    yedek, hatalar = parse_backup(_yedek({"x": "A" * 1_000_000}))
    assert yedek is None
    assert hatalar and "ay ayarları" in hatalar[0]


def test_gercekci_en_buyuk_ay_ayarlari_kabul_edilir():
    # 50 kişilik ekipte her günün her nöbeti elle sabitlenmiş ve 20 kişi saklı: gerçekte
    # olabilecek en büyük kayıt bile sınırın epey altında kalmalı.
    gunler = [f"{g} Pzt" for g in range(1, 32)]
    kisiler = [f"Dr. Ayşe Yılmaz Öztürk {i}" for i in range(50)]
    ayarlar = {
        "locked": [[k, g] for k in kisiler for g in gunler],
        "hidden": {f"Eski Kişi {i}": {"shifts": list(range(1, 32)), "prefs": {str(g): 3 for g in range(1, 32)}}
                   for i in range(20)},
    }
    yedek, hatalar = parse_backup(_yedek(ayarlar))
    assert hatalar == []
    assert yedek["schedules"][0]["settings"] == ayarlar


# --- Biçimi bozuk ay ayarları ve çok derin iç içe dosya ---
# Ay ayarlarının içi denetlenmediği için "locked": 5 gibi bir değer doğrulamadan
# geçiyor, geri yüklendikten sonra o ay her açılışta program hatası veriyordu.

def _ay_yuklenir(ay_ayarlari):
    """Yedek doğrulamadan geçerse o ayın uygulamada açılmasını dener; geçmezse None."""
    import pandas as pd
    import nobet_core

    yedek, hatalar = parse_backup(_yedek(ay_ayarlari))
    if hatalar:
        return None
    kayit = yedek["schedules"][0]
    gun_detaylari = nobet_core.build_gun_detaylari(2026, 1, 31, [])
    cizelge = pd.DataFrame(
        {gun: {kisi: (kisi, gun) in set(kayit["assigned"]) for kisi in kayit["team"]} for gun in kayit["days"]},
        index=kayit["team"],
    )
    return nobet_core.restore_auto_saved_schedule(
        {}, lambda ad, yil, ay: (kayit["team"], cizelge, None, kayit["settings"], None),
        "Otomatik_2026_01", 2026, 1, ["Ali", "Ece"], list(gun_detaylari), gun_detaylari, ["A", "B"],
    )


def test_bicimi_bozuk_ay_ayarlari_reddedilir_ay_acilirken_hata_vermez():
    for bozuk in (
        {"locked": 5},
        {"locked": True},
        {"locked": {"Ali": "1 Sal"}},
        {"hidden": 7},
        {"hidden": ["Zed"]},
        {"hidden": {"Zed": {"shifts": 5}}},
        {"hidden": {"Zed": {"shifts": None}}},
        {"hidden": {"Zed": {"shifts": "12"}}},
        {"hidden": {"Zed": {"prefs": [1, 2]}}},
        {"hidden": {"Zed": "x"}},
    ):
        # Doğrulamadan geçen her yedek uygulamada hatasız açılmalı; bunlar hiç geçmemeli.
        assert _ay_yuklenir(bozuk) is None, bozuk


def test_uygulamanin_yazdigi_ay_ayarlari_kabul_edilir():
    for saglam in (
        {},
        {"locked": [], "hidden": {}},
        {"locked": [["Ali", "1 Sal"]], "hidden": {"Zed": {"shifts": [1, 2], "prefs": {"3": 3}}}},
        {"locked": None, "hidden": None},
        {"hidden": {"Zed": {"shifts": [1]}}},
        {"hidden": {"Zed": {"prefs": {"3": 4}}}},
        {"baska": "eski sürümden kalan alan"},
    ):
        assert _ay_yuklenir(saglam) is True, saglam


def test_cok_derin_ic_ice_dosya_program_hatasi_vermeden_reddedilir():
    yedek, hatalar = parse_backup(b"[" * 200_000 + b"]" * 200_000)
    assert yedek is None
    assert hatalar == ["Dosya bir Nöbet yedeği değil (JSON okunamadı)."]
