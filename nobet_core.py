# -*- coding: utf-8 -*-
"""
nobet_core.py - Streamlit bağımsız, test edilebilir saf fonksiyonlar
nobet.py'deki iş mantığı buraya taşınarak unit test yazılabilir hale getirildi.
"""
import pandas as pd
import numpy as np
import random
import calendar
from io import BytesIO
from datetime import datetime


def parse_unwanted_days(text_input, max_day):
    """Müsait olmayan günlerin metin girdisini listeye çevirir."""
    if not text_input or (isinstance(text_input, float) and pd.isna(text_input)):
        return []
    days = set()
    parts = str(text_input).split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            if '-' in part:
                start, end = map(int, part.split('-'))
                start = max(1, start)
                end = min(max_day, end)
                if start <= end:
                    days.update(range(start, end + 1))
            else:
                d = int(part)
                if 1 <= d <= max_day:
                    days.add(d)
        except ValueError:
            continue
    return sorted(days)


def validate_inputs(isimler, yil, ay, gun_sayisi, tatil_gunleri, nobet_ucreti, min_bosluk, kisi_sayisi=2):
    """Tüm girdileri doğrular, (is_valid, errors, warnings) döner."""
    errors = []
    warnings = []

    # Ekip doğrulama
    if not isimler or len(isimler) == 0:
        errors.append("❌ En az 1 kişi ekleyin")
    elif len(isimler) > 50:
        errors.append("❌ Maksimum 50 kişi ekleyebilirsiniz")

    # Yinelenen isim kontrolü
    if len(isimler) != len(set(isimler)):
        errors.append("❌ Aynı isimde 2 kişi olamaz")

    # Ücret doğrulama
    if nobet_ucreti < 0:
        errors.append("❌ Saatlik ücret negatif olamaz")
    elif nobet_ucreti == 0:
        warnings.append("⚠️ Saatlik ücret 0 TL")

    # Tatil günü doğrulama
    invalid_holidays = [h for h in tatil_gunleri if h < 1 or h > gun_sayisi]
    if invalid_holidays:
        errors.append(f"❌ Geçersiz tatil günleri: {invalid_holidays}")

    # Dinlenme süresi doğrulama
    if min_bosluk < 0 or min_bosluk > 7:
        errors.append("❌ Dinlenme süresi 0-7 gün arasında olmalı")

    # Fizibilite uyarıları
    working_days = gun_sayisi - len(tatil_gunleri)
    total_positions_needed = working_days * kisi_sayisi
    team_size = len(isimler)

    if team_size < kisi_sayisi:
        errors.append(f"❌ {kisi_sayisi} kişi nöbet için en az {kisi_sayisi} kişi gerekli")
    elif team_size > 0 and total_positions_needed > team_size * 30:
        avg_per_person = total_positions_needed / team_size
        warnings.append(f"⚠️ Her kişiye ortalama {avg_per_person:.1f} nöbet düşecek (çok fazla)")
    elif team_size > 0 and total_positions_needed < team_size:
        warnings.append(f"⚠️ Nöbetleri dağıtmak için çok fazla kişi var ({team_size} kişi, {total_positions_needed} pozisyon)")

    return len(errors) == 0, errors, warnings


def parse_forbidden_pairs(forbidden_input):
    """Birlikte çalışamayan çiftleri metin girdisinden küme olarak döner."""
    forbidden_pairs = set()
    if not forbidden_input or not forbidden_input.strip():
        return forbidden_pairs
    all_pairs = []
    for line in forbidden_input.strip().split('\n'):
        all_pairs.extend(line.split(','))
    for pair_str in all_pairs:
        pair_str = pair_str.strip()
        if '-' in pair_str:
            parts = pair_str.split('-', 1)
            if len(parts) == 2:
                p1, p2 = parts[0].strip(), parts[1].strip()
                if p1 and p2:
                    forbidden_pairs.add(tuple(sorted((p1, p2))))
    return forbidden_pairs


def parse_person_limits(limits_text):
    """Kişisel limit metnini sözlüğe çevirir: {'Ali': {'min': 5, 'max': 10}}"""
    person_limits = {}
    if not limits_text or not limits_text.strip():
        return person_limits
    for line in limits_text.strip().split('\n'):
        if ':' in line:
            parts = line.split(':')
            name = parts[0].strip()
            if len(parts) == 2 and '-' in parts[1]:
                try:
                    min_val, max_val = map(int, parts[1].split('-'))
                    person_limits[name] = {'min': min_val, 'max': max_val}
                except ValueError:
                    pass
    return person_limits


def build_gun_detaylari(yil, ay, gun_sayisi, tatil_gunleri):
    """Her gün için meta veri sözlüğü oluşturur."""
    gun_detaylari = {}
    ay_isimleri = {1:"Oca", 2:"Şub", 3:"Mar", 4:"Nis", 5:"May", 6:"Haz",
                   7:"Tem", 8:"Ağu", 9:"Eyl", 10:"Eki", 11:"Kas", 12:"Ara"}
    gun_isimleri = {0:"Pzt", 1:"Sal", 2:"Çar", 3:"Per", 4:"Cum", 5:"Cmt", 6:"Paz"}

    for gun in range(1, gun_sayisi + 1):
        if gun in tatil_gunleri:
            continue
        weekday = calendar.weekday(yil, ay, gun)
        is_weekend = weekday >= 5
        full_date = f"{gun} {ay_isimleri[ay]} {gun_isimleri[weekday]}"
        col_key = f"G{gun:02d}"
        gun_detaylari[col_key] = {
            'day_num': gun,
            'weekend': is_weekend,
            'holiday': gun in tatil_gunleri,
            'full_date': full_date,
            'weekday': weekday,
        }
    return gun_detaylari


def run_scheduling_core(isimler, sutunlar, df_unwanted_bool, gun_detaylari,
                        min_bosluk, kisi_sayisi, forbidden_pairs=None,
                        person_limits=None, df_preferred=None,
                        simulation_count=10, progress_callback=None):
    """
    Saf zamanlama algoritması (Streamlit bağımsız).
    progress_callback(int) isteğe bağlı ilerleme bildirimi için kullanılır.
    """
    best_schedule = None
    best_score = float('inf')

    for attempt in range(simulation_count):
        if progress_callback:
            progress_callback(attempt + 1)

        stat_total = {i: 0 for i in isimler}
        stat_special = {i: 0 for i in isimler}
        stat_consecutive_weekend = {i: 0 for i in isimler}
        last_weekend_shift = {i: -10 for i in isimler}
        pair_history = {}
        last_shift_day = {i: -10 for i in isimler}

        temp_schedule = pd.DataFrame(
            {col: [False] * len(isimler) for col in sutunlar}, index=isimler
        )

        def get_decision_score(p, is_sp, col):
            total = stat_total[p] + (random.random() * 0.5)
            sp_count = stat_special[p]
            consecutive_penalty = stat_consecutive_weekend[p] * 200
            pref_bonus = 0
            if df_preferred is not None and p in df_preferred.index and col in df_preferred.columns:
                pref_val = df_preferred.at[p, col]
                if pref_val == 1:
                    pref_bonus = -500
                elif pref_val == 2:
                    pref_bonus = 300
            limit_penalty = 0
            if person_limits and p in person_limits:
                max_limit = person_limits[p].get('max', 999)
                if stat_total[p] >= max_limit:
                    limit_penalty = 50000
            if is_sp:
                return (sp_count * 100) + (total * 10) + consecutive_penalty + pref_bonus + limit_penalty
            else:
                return (total * 10) + (sp_count * 1) + pref_bonus + limit_penalty

        empty_shifts = 0
        limit_violations = 0

        for col in sutunlar:
            info = gun_detaylari[col]
            gun_no = info['day_num']
            is_sp = info['weekend'] or info['holiday']
            is_weekend = info['weekend']
            weekend_num = (gun_no - 1) // 7

            adaylar = []
            for k in isimler:
                if df_unwanted_bool.at[k, col]:
                    continue
                if (gun_no - last_shift_day[k]) <= min_bosluk:
                    continue
                if person_limits and k in person_limits:
                    max_limit = person_limits[k].get('max', 999)
                    if stat_total[k] >= max_limit:
                        continue
                adaylar.append(k)

            random.shuffle(adaylar)
            adaylar.sort(key=lambda x: get_decision_score(x, is_sp, col))

            if len(adaylar) >= kisi_sayisi:
                secilenler = []
                for p in adaylar:
                    valid = True
                    if forbidden_pairs:
                        for selected in secilenler:
                            if tuple(sorted((p, selected))) in forbidden_pairs:
                                valid = False
                                break
                    if valid:
                        secilenler.append(p)
                        if len(secilenler) >= kisi_sayisi:
                            break

                # Retry with shuffle if forbidden pairs blocked
                if len(secilenler) < kisi_sayisi and forbidden_pairs and len(adaylar) >= kisi_sayisi:
                    random.shuffle(adaylar)
                    secilenler = []
                    for p in adaylar:
                        valid = True
                        for selected in secilenler:
                            if tuple(sorted((p, selected))) in forbidden_pairs:
                                valid = False
                                break
                        if valid:
                            secilenler.append(p)
                            if len(secilenler) >= kisi_sayisi:
                                break

                if len(secilenler) >= kisi_sayisi:
                    if kisi_sayisi >= 2:
                        pair = tuple(sorted((secilenler[0], secilenler[1])))
                        pair_history[pair] = pair_history.get(pair, 0) + 1
                    for k in secilenler:
                        temp_schedule.at[k, col] = True
                        stat_total[k] += 1
                        if is_sp:
                            stat_special[k] += 1
                        last_shift_day[k] = gun_no
                        if is_weekend:
                            if last_weekend_shift[k] >= 0 and weekend_num == last_weekend_shift[k] + 1:
                                stat_consecutive_weekend[k] += 1
                            elif last_weekend_shift[k] >= 0 and weekend_num > last_weekend_shift[k] + 1:
                                stat_consecutive_weekend[k] = 0
                            last_weekend_shift[k] = weekend_num
                else:
                    empty_shifts += 1
            else:
                empty_shifts += 1

        if person_limits:
            for p, limits in person_limits.items():
                if stat_total.get(p, 0) < limits.get('min', 0):
                    limit_violations += 1

        totals = list(stat_total.values())
        specials = list(stat_special.values())
        consecutive_weekends = sum(stat_consecutive_weekend.values())
        std_dev_total = np.std(totals)
        std_dev_special = np.std(specials)
        range_total = max(totals) - min(totals)

        score = (
            (empty_shifts * 10000) +
            (limit_violations * 5000) +
            (consecutive_weekends * 500) +
            (range_total * 100) +
            (std_dev_total * 10) +
            (std_dev_special * 5)
        )

        if score < best_score:
            best_score = score
            best_schedule = temp_schedule.copy()

    return best_schedule, best_score


def create_print_html(df_liste, df_stats_load, yil, ay):
    """Yazdırma dostu HTML oluşturur."""
    ay_isimleri = {1:"Ocak", 2:"Şubat", 3:"Mart", 4:"Nisan", 5:"Mayıs", 6:"Haziran",
                   7:"Temmuz", 8:"Ağustos", 9:"Eylül", 10:"Ekim", 11:"Kasım", 12:"Aralık"}
    html = f"""<html><head><meta charset="utf-8"><title>Nöbet - {ay_isimleri[ay]} {yil}</title></head>
    <body><h1>Nöbet Listesi - {ay_isimleri[ay]} {yil}</h1>
    <table><tr>{''.join(f'<th>{col}</th>' for col in df_liste.columns)}</tr>"""
    for _, row in df_liste.iterrows():
        html += f"<tr>{''.join(f'<td>{val}</td>' for val in row)}</tr>"
    html += "</table></body></html>"
    return html
