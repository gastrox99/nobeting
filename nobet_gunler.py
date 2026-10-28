# -*- coding: utf-8 -*-
"""nobet_gunler.py - Resmi tatiller, arifeler, mesai saati ve ay/gün yardımcıları."""
import os
from datetime import date

import pandas as pd


# Tercih kodları. İzin (3) ve Müsait Değil (4) günlerine nöbet yazılmaz;
# yalnızca İzin zorunlu mesaiden düşer.
PREF_LEAVE = 3
PREF_UNAVAILABLE = 4
BLOCKED_PREFS = frozenset({PREF_LEAVE, PREF_UNAVAILABLE})
MAX_PREF_CODE = 4

TR_GUN_KISA = {0: "Pzt", 1: "Sal", 2: "Çar", 3: "Per", 4: "Cum", 5: "Cmt", 6: "Paz"}


def period_key(yil, ay):
    """Aya özel ayarlar (ör. tatiller) için anahtar: "2026-10"."""
    return f"{int(yil):04d}-{int(ay):02d}"


def parse_holiday_days(text_input, max_day):
    """Tatil metnini (geçerli günler, geçersiz parçalar) olarak ayrıştırır.

    Bir aralıkta uçlardan biri ay sınırını aşıyorsa aralığın tamamı geçersiz
    sayılır; böylece hatalı giriş sessizce ay içine kırpılmaz.
    """
    if not text_input or (isinstance(text_input, float) and pd.isna(text_input)):
        return [], []

    valid_days = set()
    invalid_parts = []
    for raw_part in str(text_input).split(','):
        part = raw_part.strip()
        if not part:
            continue

        if '-' in part:
            range_parts = [item.strip() for item in part.split('-')]
            if (
                len(range_parts) != 2
                or not all(item.isascii() and item.isdecimal() for item in range_parts)
            ):
                invalid_parts.append(part)
                continue
            start, end = map(int, range_parts)
            if start < 1 or end > max_day or start > end:
                invalid_parts.append(part)
                continue
            valid_days.update(range(start, end + 1))
            continue

        if not part.isascii() or not part.isdecimal():
            invalid_parts.append(part)
            continue
        day = int(part)
        if 1 <= day <= max_day:
            valid_days.add(day)
        else:
            invalid_parts.append(part)

    return sorted(valid_days), invalid_parts


def build_gun_detaylari(yil, ay, gun_sayisi, tatil_gunleri):
    """Ayın her günü için uygulamanın kullandığı sütun ve meta veriyi üretir.

    Anahtarlar "1 Pzt" biçimindedir. Tatiller de nöbet tutulan günlerdir;
    çıkarılmaz, ``holiday`` ile işaretlenir ve hafta sonu gibi özel gün sayılır.
    """
    tatiller = set(tatil_gunleri)
    gun_detaylari = {}
    for gun in range(1, gun_sayisi + 1):
        gun_tarihi = date(yil, ay, gun)
        weekday = gun_tarihi.weekday()
        gun_adi = TR_GUN_KISA[weekday]
        gun_detaylari[f"{gun} {gun_adi}"] = {
            "day_num": gun,
            "weekend": weekday >= 5,
            "holiday": gun in tatiller,
            "full_date": gun_tarihi.strftime("%d.%m.%Y") + " " + gun_adi,
            "weekday": weekday,
            "week": (gun_tarihi.toordinal() - weekday) // 7,
        }
    return gun_detaylari


TR_AY_ADLARI = {1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
                7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"}


# Türkiye resmi tatilleri (arife yarım günleri dahil edilmez).
TR_FIXED_HOLIDAYS = {
    (1, 1): "Yılbaşı",
    (4, 23): "Ulusal Egemenlik ve Çocuk Bayramı",
    (5, 1): "Emek ve Dayanışma Günü",
    (5, 19): "Atatürk'ü Anma, Gençlik ve Spor Bayramı",
    (7, 15): "Demokrasi ve Milli Birlik Günü",
    (8, 30): "Zafer Bayramı",
    (10, 29): "Cumhuriyet Bayramı",
}
# Dini bayramlar her yıl değişir: (ilk gün, gün sayısı, ad). Diyanet takvimi.
TR_RELIGIOUS_HOLIDAYS = {
    2024: [(date(2024, 4, 10), 3, "Ramazan Bayramı"), (date(2024, 6, 16), 4, "Kurban Bayramı")],
    2025: [(date(2025, 3, 30), 3, "Ramazan Bayramı"), (date(2025, 6, 6), 4, "Kurban Bayramı")],
    2026: [(date(2026, 3, 20), 3, "Ramazan Bayramı"), (date(2026, 5, 27), 4, "Kurban Bayramı")],
    2027: [(date(2027, 3, 9), 3, "Ramazan Bayramı"), (date(2027, 5, 16), 4, "Kurban Bayramı")],
    2028: [(date(2028, 2, 26), 3, "Ramazan Bayramı"), (date(2028, 5, 5), 4, "Kurban Bayramı")],
    # 2029: Diyanet'in kesin takvimi çıkınca kontrol edin.
    2029: [(date(2029, 2, 14), 3, "Ramazan Bayramı"), (date(2029, 4, 24), 4, "Kurban Bayramı")],
}


def religious_holidays_known(yil):
    """Bu yılın dini bayram tarihleri listede var mı?"""
    return yil in TR_RELIGIOUS_HOLIDAYS


def official_holidays(yil, ay):
    """Ayın resmi tatilleri: [(gün, ad), ...] gün sırasıyla."""
    found = {}
    for (month, day), name in TR_FIXED_HOLIDAYS.items():
        if month == ay:
            found[day] = name
    for start, length, name in TR_RELIGIOUS_HOLIDAYS.get(yil, []) + TR_RELIGIOUS_HOLIDAYS.get(yil - 1, []):
        for offset in range(length):
            current = date.fromordinal(start.toordinal() + offset)
            if current.year == yil and current.month == ay:
                found[current.day] = name
    return sorted(found.items())


# Arifeler yarım gün (öğleden sonra tatil): zorunlu mesaide 4 saat sayılır.
TR_FIXED_HALF_DAYS = {(10, 28): "Cumhuriyet Bayramı arifesi"}


def official_half_days(yil, ay):
    """Ayın arife yarım günleri: [(gün, ad), ...] gün sırasıyla."""
    found = {day: name for (month, day), name in TR_FIXED_HALF_DAYS.items() if month == ay}
    for start, _, name in TR_RELIGIOUS_HOLIDAYS.get(yil, []) + TR_RELIGIOUS_HOLIDAYS.get(yil + 1, []):
        eve = date.fromordinal(start.toordinal() - 1)
        if eve.year == yil and eve.month == ay:
            found[eve.day] = f"{name} arifesi"
    return sorted(found.items())


def required_work_hours(gun_detaylari, half_days=()):
    """Zorunlu mesai: hafta içi iş günü 8 saat, iş gününe düşen arife 4 saat.

    ``half_days``: [(gün, ad), ...]. Hafta sonuna ya da tatil olarak
    işaretlenmiş güne düşen arife zaten sayılmaz. Dönen değer
    (saat, hesaba katılan arifeler).
    """
    half = dict(half_days)
    hours, counted = 0, []
    for detail in gun_detaylari.values():
        if detail["weekend"] or detail["holiday"]:
            continue
        if detail["day_num"] in half:
            hours += 4
            counted.append((detail["day_num"], half[detail["day_num"]]))
        else:
            hours += 8
    return hours, counted


def personal_work_hours(gun_detaylari, half_days, preferences, people):
    """Kişi başına zorunlu mesai: İzin (3) işaretli iş günü mesaiden düşer.

    İzinli iş günü 8 saat, izinli arife 4 saat düşer; hafta sonu ve tatil
    zaten mesaide sayılmadığı için düşmez. Müsait Değil (4) mesaiyi
    değiştirmez. Dönen değer {kişi: saat}.
    """
    half = dict(half_days)
    work_hours = {}
    for column, detail in gun_detaylari.items():
        if detail["weekend"] or detail["holiday"]:
            continue
        work_hours[column] = 4 if detail["day_num"] in half else 8
    total = sum(work_hours.values())
    result = {}
    for person in people:
        leave = 0
        if preferences is not None and person in preferences.index:
            for column, hours in work_hours.items():
                if column not in preferences.columns:
                    continue
                try:
                    code = int(preferences.at[person, column])
                except (TypeError, ValueError):
                    continue
                if code == PREF_LEAVE:
                    leave += hours
        result[person] = total - leave
    return result


def official_holidays_summary(yil, ay):
    """["20-22 Mart Ramazan Bayramı", "29 Ekim Cumhuriyet Bayramı"] gibi kısa açıklama."""
    groups = []
    for day, name in official_holidays(yil, ay):
        if groups and groups[-1][2] == name and groups[-1][1] == day - 1:
            groups[-1][1] = day
        else:
            groups.append([day, day, name])
    month = TR_AY_ADLARI[ay]
    return [
        f"{start} {month} {name}" if start == end else f"{start}-{end} {month} {name}"
        for start, end, name in groups
    ]


def default_holiday_text(yil, ay):
    """Tatil kutusunun bu ay için varsayılan metni (resmi tatiller)."""
    return holidays_to_text([day for day, _ in official_holidays(yil, ay)])


def holiday_text_for_period(holidays_by_period, yil, ay):
    """Ayın tatil metni: elle değiştirildiyse kaydı, değilse güncel resmi tatiller.

    Varsayılan metin ayarlara yazılmaz; böylece koddaki bayram tarihleri
    düzeltilince (ör. 2029) daha önce açılmış aylara da yansır.
    """
    return (holidays_by_period or {}).get(period_key(yil, ay), default_holiday_text(yil, ay))


def prune_default_holidays(holidays_by_period):
    """Resmi tatillerle aynı olan (elle değiştirilmemiş) ay kayıtlarını siler."""
    for key in list(holidays_by_period or {}):
        try:
            yil, ay = (int(part) for part in key.split("-"))
            default = default_holiday_text(yil, ay)
        except (ValueError, KeyError):
            continue
        if holidays_by_period[key] == default:
            del holidays_by_period[key]
    return holidays_by_period


def holidays_to_text(days):
    """[20, 21, 22, 29] -> "20-22, 29" (tatil kutusuna yazılan biçim)."""
    days = sorted(set(days))
    parts, start = [], None
    for index, day in enumerate(days):
        if start is None:
            start = day
        if index == len(days) - 1 or days[index + 1] != day + 1:
            parts.append(str(start) if start == day else f"{start}-{day}")
            start = None
    return ", ".join(parts)


def previous_period(yil, ay, back=1):
    """(yıl, ay)'dan ``back`` ay öncesini döner."""
    index = yil * 12 + (ay - 1) - back
    return index // 12, index % 12 + 1


def _column_day(column):
    try:
        return int(str(column).split()[0])
    except (ValueError, IndexError):
        return None


# Damga en sonda: yükleme yarıda kalırsa sonraki çalışma yeniden dener (nobet_tazele.py).
_nobet_loaded_stamp = os.stat(__file__).st_mtime_ns
