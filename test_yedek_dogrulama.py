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
