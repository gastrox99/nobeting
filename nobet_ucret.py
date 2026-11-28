# -*- coding: utf-8 -*-
"""nobet_ucret.py - Ünvana göre nöbet saat ücreti hesapları."""
import os
import calendar
from datetime import date, timedelta

from nobet_gunler import (
    TR_RELIGIOUS_HOLIDAYS, build_gun_detaylari, holiday_text_for_period, official_half_days,
    official_holidays, parse_holiday_days, personal_daily_work_hours, required_work_hours,
)


# ── Ünvana göre nöbet saat ücreti ─────────────────────────────────────────
# 657 sayılı DMK Ek Madde 33: nöbet ücreti = gösterge × memur aylık katsayısı.
# 8. Dönem Toplu Sözleşme göstergelere 10 puan ekler (aşağıdakiler eklenmiş
# hâlidir). 20.00-08.00 arası gece nöbeti %10, acil/yoğun bakım/ameliyathane
# gibi riskli birimler %50 artırımlıdır. Nöbet ücretinden yalnızca damga
# vergisi (binde 7,59) kesilir; aşağıdaki tutarlar kesinti sonrası nettir.
NOBET_UNVANLARI = {
    "Eğitim Görevlisi": 160,
    "Uzman Tabip": 160,
    "Tabip": 145,
    "Tıpta uzmanlık tüzüğüne göre": 145,
    "Diş Tabibi ve Eczacılar": 130,
    "Sağlık Personeli (Lisans-Önlisans)": 100,
    "Sağlık Personeli (Lise)": 85,
    "Diğer Personel": 65,
    # 4/B sözleşmeli personel, memur tutarının yaklaşık %85,89'unu alır
    # (2026 tablosundan çıkarılan karşılık gösterge).
    "4/B Sağlık Personeli (Lisans-Önlisans)": 85.89,
    "4/B Sağlık Personeli (Lise)": 73.01,
    "4/B Diğer Personel": 55.83,
}
# Önceki sürümlerde kaydedilmiş ünvan adları.
ESKI_UNVAN_ADLARI = {
    "Pratisyen Tabip": "Tabip",
    "Diş Tabibi / Eczacı": "Diş Tabibi ve Eczacılar",
    "Hemşire / Ebe (yüksek öğrenim)": "Sağlık Personeli (Lisans-Önlisans)",
    "Sağlık Personeli (lise)": "Sağlık Personeli (Lise)",
}
ELLE_UNVAN = "Elle gir"

# Memur aylık katsayısı her Ocak ve Temmuz'da Hazine ve Maliye Bakanlığı
# genelgesiyle değişir; yeni dönem çıkınca buraya bir satır eklenir.
# Anahtar: (yıl, 1 = Ocak-Haziran, 2 = Temmuz-Aralık).
MEMUR_AYLIK_KATSAYI = {
    (2026, 1): 1.387871,
    (2026, 2): 1.575512,
}
DAMGA_VERGISI = 0.00759
GECE_ARTIRIMI = 1.10
GECE_FARKI = 0.10
RISKLI_BIRIM_ARTIRIMI = 1.50
# Resmi ve dini bayram (tatil) günlerinde tutulan nöbet %25 fazla ödenir.
TATIL_ARTIRIMI = 0.25
# Nöbet 08.00'de başlar, 24 saat sürer; gece 20.00-08.00 (UCRET_KURALLARI.md).
NOBET_BASLANGIC_SAATI = 8
NOBET_SURESI = 24
GECE_BASLANGIC, GECE_BITIS = 20, 8
# Arifede bayram 13.00'te başlar.
ARIFE_BAYRAM_SAATI = 13
# Genel tatiller: zorunlu mesaiden düşer ama bayram değildir (%25 yok);
# dini bayram gününe denk gelirse bayram sayılır.
GENEL_TATILLER = frozenset({(1, 1), (5, 1), (7, 15)})
_DONEM_ADI = {1: "Ocak-Haziran", 2: "Temmuz-Aralık"}
# Ücret türleri: gündüz, gece (20-08), resmi/dini bayram gündüz ve gece.
UCRET_TURLERI = ("gunduz", "gece", "tatil", "tatil_gece")

# Yayımlanan net saat ücreti tabloları; olan dönemde formül yerine bunlar
# kullanılır (tablolar kuruş yuvarlamasıyla formülden 1-2 kuruş sapabilir).
# Sıra: normal, gece, riskli, riskli gece, bayram, bayram gece, bayram
# riskli, bayram riskli gece.
NOBET_UCRET_TABLOSU = {
    (2026, 2): {
        "Eğitim Görevlisi": (250.17, 275.19, 375.25, 412.78, 312.71, 343.98, 469.07, 515.97),
        "Uzman Tabip": (250.17, 275.19, 375.25, 412.78, 312.71, 343.98, 469.07, 515.97),
        "Tabip": (226.72, 249.39, 340.07, 374.08, 283.39, 311.73, 425.09, 467.60),
        "Tıpta uzmanlık tüzüğüne göre": (226.72, 249.39, 340.07, 374.08, 283.39, 311.73, 425.09, 467.60),
        "Diş Tabibi ve Eczacılar": (203.28, 223.59, 304.89, 335.38, 254.08, 279.49, 381.12, 419.23),
        "Sağlık Personeli (Lisans-Önlisans)": (156.36, 171.99, 234.55, 257.99, 195.44, 214.99, 293.17, 322.48),
        "Sağlık Personeli (Lise)": (132.90, 146.19, 199.35, 219.29, 166.13, 182.74, 249.19, 274.11),
        "Diğer Personel": (101.63, 111.78, 152.46, 167.68, 127.04, 139.74, 190.65, 209.61),
        "4/B Sağlık Personeli (Lisans-Önlisans)": (134.30, 147.73, 201.45, 221.59, 167.87, 184.66, 251.81, 276.99),
        "4/B Sağlık Personeli (Lise)": (114.15, 125.57, 171.25, 188.35, 142.69, 156.96, 214.04, 235.44),
        "4/B Diğer Personel": (87.28, 96.02, 130.84, 144.03, 109.12, 120.03, 163.58, 180.04),
    },
}


def katsayi_donemi(yil, ay):
    """Seçilen ay için kullanılacak katsayı dönemi: (anahtar, tam_eşleşme_mi).

    O dönemin katsayısı tabloda yoksa en yakın önceki (yoksa ilk) dönem
    kullanılır ve tam_eşleşme False döner.
    """
    key = (int(yil), 1 if int(ay) <= 6 else 2)
    if key in MEMUR_AYLIK_KATSAYI:
        return key, True
    known = sorted(MEMUR_AYLIK_KATSAYI)
    earlier = [k for k in known if k < key]
    return (earlier[-1] if earlier else known[0]), False


def donem_adi(key):
    return f"{_DONEM_ADI[key[1]]} {key[0]}"


def katsayi_uyarisi(yil, ay):
    """Seçilen ayın katsayısı tabloda yoksa görünür uyarı metni, varsa None."""
    key, tam = katsayi_donemi(yil, ay)
    if tam:
        return None
    return ("Bu ayın memur katsayısı henüz açıklanmadı; ücret hesaplanmadı (tahmin edilmez). "
            "Katsayı eklenince tutarlar görünür; isterseniz saat ücretini kendiniz girebilirsiniz.")


def unvan_saat_ucreti(unvan, yil, ay, riskli=False):
    """Ünvanın o ayki nöbet saat ücretleri (net, TL).

    Döner: {"gunduz", "gece", "tatil", "tatil_gece", "donem", "tam"}.
    Dönemin yayımlanan tablosu varsa tutarlar oradan, yoksa gösterge ×
    katsayı × artırımlar formülünden gelir. Bilinmeyen ünvanda None.
    """
    gosterge = NOBET_UNVANLARI.get(unvan)
    if gosterge is None:
        return None
    donem, tam = katsayi_donemi(yil, ay)
    tablo = NOBET_UCRET_TABLOSU.get(donem, {}).get(unvan) if tam else None
    if tablo:
        secili = tablo[2:4] + tablo[6:8] if riskli else tablo[0:2] + tablo[4:6]
        ucretler = dict(zip(UCRET_TURLERI, secili))
    else:
        brut = gosterge * MEMUR_AYLIK_KATSAYI[donem]
        if riskli:
            brut *= RISKLI_BIRIM_ARTIRIMI
        carpanlar = (1, GECE_ARTIRIMI, 1 + TATIL_ARTIRIMI, (1 + TATIL_ARTIRIMI) * GECE_ARTIRIMI)
        ucretler = {
            tur: round(brut * carpan * (1 - DAMGA_VERGISI), 2)
            for tur, carpan in zip(UCRET_TURLERI, carpanlar)
        }
    return {**ucretler, "donem": donem, "tam": tam}


def saat_ucretleri(gunduz, tatil_artirimi=0.0, gece_artirimi=0.0):
    """Elle girilen gündüz ücretinden gece ve bayram ücretleri."""
    gece = 1 + gece_artirimi
    tatil = 1 + tatil_artirimi
    return {
        "gunduz": gunduz,
        "gece": round(gunduz * gece, 2),
        "tatil": round(gunduz * tatil, 2),
        "tatil_gece": round(gunduz * tatil * gece, 2),
    }


def _dini_bayram_mi(gun):
    for start, length, _ in TR_RELIGIOUS_HOLIDAYS.get(gun.year, []) + TR_RELIGIOUS_HOLIDAYS.get(gun.year - 1, []):
        if 0 <= (gun - start).days < length:
            return True
    return False


def _bayram_gunu_mu(gun, tatil):
    """Tam gün bayram mı: tatil ve (genel tatil değil ya da dini bayram)."""
    return tatil and ((gun.month, gun.day) not in GENEL_TATILLER or _dini_bayram_mi(gun))


def haftalik_ucret_saatleri(gun_detaylari, nobet_gunleri, gunluk_mesai, ucretler, resmi_tatil_gunleri=None):
    """Haftalık (Pzt-Paz) fazla mesai ve ücret dökümü (UCRET_KURALLARI.md, Bölüm 4).

    ``nobet_gunleri``: kişinin nöbet başlattığı gün numaraları (08.00'den 24 saat).
    ``gunluk_mesai``: {gün: zorunlu saat} (``personal_daily_work_hours``).
    ``resmi_tatil_gunleri``: %25 alan tatil günleri; elle eklenen tatil almaz.
    Verilmezse tüm tatiller. Bayram saat saat belirlenir: bayram günü
    00-24, arife 13-24; bayram ertesi sabahı normaldir.
    Her hafta fazla saat önce bayram gece, normal gece, bayram gündüz, normal
    gündüz saatlerine yazılır; eksik haftaların açığı aynı ayın ödenecek
    saatlerinden (normal gece, bayram gece, normal gündüz, bayram gündüz
    sırasıyla) düşülür, sonraki aya devretmez.
    Döner: (nöbet saati, fazla mesai saati, zorunlu mesai, satırlar);
    ``satirlar``: {tür: (saat, saat ücreti, tutar)}, türler ``UCRET_TURLERI``.
    """
    if not isinstance(ucretler, dict):
        ucretler = dict.fromkeys(UCRET_TURLERI, ucretler)
    gunler = {}
    for detail in gun_detaylari.values():
        gunler[detail["day_num"]] = (_detay_tarihi(detail), bool(detail["holiday"]))
    ilk = min(gunler.values())[0]
    yil, ay = ilk.year, ilk.month
    arifeler = {gun for gun, _ in official_half_days(yil, ay)}
    sonraki = ilk.replace(day=28) + timedelta(days=4)
    sonraki_tatil = {gun for gun, _ in official_holidays(sonraki.year, sonraki.month)}

    def bayram(gun, saat):
        if gun.month == ay:
            tatil = gunler[gun.day][1] and (resmi_tatil_gunleri is None or gun.day in resmi_tatil_gunleri)
            if gun.day in arifeler and saat >= ARIFE_BAYRAM_SAATI:
                return True
        else:
            tatil = gun.day in sonraki_tatil
        return _bayram_gunu_mu(gun, tatil)

    haftalar = {}
    for gun_no, (gun, _) in gunler.items():
        hafta = haftalar.setdefault(gun.isocalendar()[:2], dict.fromkeys(("R", "A") + UCRET_TURLERI, 0))
        hafta["R"] += gunluk_mesai.get(gun_no, 0)
    for gun_no in set(nobet_gunleri):
        baslangic = gunler[gun_no][0]
        hafta = haftalar[baslangic.isocalendar()[:2]]
        hafta["A"] += NOBET_SURESI
        for i in range(NOBET_SURESI):
            t = NOBET_BASLANGIC_SAATI + i
            gun, saat = baslangic + timedelta(days=t // 24), t % 24
            gece = saat >= GECE_BASLANGIC or saat < GECE_BITIS
            b = bayram(gun, saat)
            hafta[("tatil_gece" if b else "gece") if gece else ("tatil" if b else "gunduz")] += 1

    odenecek = dict.fromkeys(UCRET_TURLERI, 0)
    acik = 0
    for hafta in haftalar.values():
        fark = hafta["A"] - hafta["R"]
        if fark <= 0:
            acik -= fark
            continue
        for tur in ("tatil_gece", "gece", "tatil", "gunduz"):
            pay = min(fark, hafta[tur])
            odenecek[tur] += pay
            fark -= pay
    for tur in ("gece", "tatil_gece", "gunduz", "tatil"):
        dusulen = min(acik, odenecek[tur])
        odenecek[tur] -= dusulen
        acik -= dusulen

    satirlar = {
        tur: (odenecek[tur], ucretler[tur], odenecek[tur] * ucretler[tur]) for tur in UCRET_TURLERI
    }
    zorunlu = sum(hafta["R"] for hafta in haftalar.values())
    return len(set(nobet_gunleri)) * NOBET_SURESI, sum(odenecek.values()), zorunlu, satirlar


def _detay_tarihi(detail):
    """``build_gun_detaylari`` satırının tarihi ("01.10.2026 Per" -> date)."""
    gun, ay, yil = detail["full_date"].split()[0].split(".")
    return date(int(yil), int(ay), int(gun))


def ayarlardaki_saat_ucretleri(settings, yil, ay):
    """Liste sahibinin kayıtlı ayarlarından o ayın saat ücretleri.

    Ana sayfadaki seçimle aynıdır: ünvan seçiliyse resmi tablo (gündüz ücreti
    o dönem için elle değiştirildiyse ondan türetilen gece/bayram ücretleri),
    "Elle gir"de her saat aynı ücret. Ücret girilmemişse None.
    Döner: {"ucretler", "unvan", "riskli", "donem", "tam", "elle"}.
    """
    settings = settings or {}
    unvan = _unvan(settings.get("title")) or ELLE_UNVAN
    riskli = settings.get("risky_unit") is True
    try:
        elle_ucret = float(settings.get("hourly_rate"))
    except (TypeError, ValueError):
        elle_ucret = None
    if elle_ucret is not None and (elle_ucret != elle_ucret or elle_ucret < 0):
        elle_ucret = None
    if unvan == ELLE_UNVAN:
        # Hiç ücret girilmemiş ayarlarda kutu 1 TL'de durur; o zaman gösterilmez.
        if not elle_ucret or elle_ucret <= 1:
            return None
        return {"ucretler": elle_ucret, "unvan": None, "riskli": False,
                "donem": None, "tam": True, "elle": True}
    info = unvan_saat_ucreti(unvan, yil, ay, riskli)
    kaynak = f"{unvan}|{info['donem'][0]}-{info['donem'][1]}|{int(riskli)}|g"
    elle = (
        settings.get("hourly_rate_source") == kaynak
        and elle_ucret is not None
        and abs(elle_ucret - info["gunduz"]) > 0.005
    )
    ucretler = (
        saat_ucretleri(elle_ucret, TATIL_ARTIRIMI, GECE_FARKI) if elle
        else {tur: info[tur] for tur in UCRET_TURLERI}
    )
    if not info["tam"] and not elle:
        return None  # Katsayı yoksa hesap durur (UCRET_KURALLARI.md).
    return {"ucretler": ucretler, "unvan": unvan, "riskli": riskli,
            "donem": info["donem"], "tam": info["tam"], "elle": elle}


UCRET_TURU_ADI = {
    "gunduz": "Gündüz",
    "gece": "Gece (20-08)",
    "tatil": "Bayram gündüz (%25)",
    "tatil_gece": "Bayram gece (%25)",
}


def kisi_ucret_dokumu(nobet_gunleri, yil, ay, settings, preferences=None, person=None):
    """Bir kişinin o ayki tahmini nöbet ücreti dökümü (paylaşım sayfası için).

    ``nobet_gunleri``: kişinin nöbet tuttuğu gün numaraları. Tatiller, arifeler
    ve izinli günlerin mesaiden düşmesi Analiz'deki Ücret Özeti ile aynıdır.
    Ücret ayarı yoksa None döner.
    """
    oran = ayarlardaki_saat_ucretleri(settings, yil, ay)
    if oran is None:
        return None
    gun_sayisi = calendar.monthrange(yil, ay)[1]
    tatil_metni = holiday_text_for_period((settings or {}).get("holidays_by_period"), yil, ay)
    tatil_gunleri, _ = parse_holiday_days(tatil_metni, gun_sayisi)
    gun_detaylari = build_gun_detaylari(yil, ay, gun_sayisi, tatil_gunleri)
    yarim_gunler = official_half_days(yil, ay)
    zorunlu, _ = required_work_hours(gun_detaylari, yarim_gunler)
    gunluk = personal_daily_work_hours(gun_detaylari, yarim_gunler, preferences, [person])[person]
    resmi = frozenset(gun for gun, _ in official_holidays(yil, ay))
    tatiller = set(tatil_gunleri)
    gunler = sorted({int(gun) for gun in nobet_gunleri if 1 <= int(gun) <= gun_sayisi})
    bayram = [gun for gun in gunler if _bayram_gunu_mu(date(yil, ay, gun), gun in tatiller and gun in resmi)]
    saat, fm_saat, mesai, satirlar = haftalik_ucret_saatleri(
        gun_detaylari, gunler, gunluk, oran["ucretler"], resmi)
    return {
        **oran,
        "nobet": len(gunler),
        "bayram_nobet": len(bayram),
        "saat": saat,
        "zorunlu": zorunlu,
        "mesai": mesai,
        "fazla_mesai": fm_saat,
        "satirlar": satirlar,
        "toplam": sum(tutar for _, _, tutar in satirlar.values()),
    }


def _unvan(value):
    value = ESKI_UNVAN_ADLARI.get(value, value)
    return value if value == ELLE_UNVAN or value in NOBET_UNVANLARI else None


# Damga en sonda: yükleme yarıda kalırsa sonraki çalışma yeniden dener (nobet_tazele.py).
_nobet_loaded_stamp = os.stat(__file__).st_mtime_ns
