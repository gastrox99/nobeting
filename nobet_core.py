# -*- coding: utf-8 -*-
"""
nobet_core.py - Streamlit bağımsız, test edilebilir saf fonksiyonlar
nobet.py'deki iş mantığı buraya taşınarak unit test yazılabilir hale getirildi.
"""
import pandas as pd
import numpy as np
import random
import calendar
import hashlib
from io import BytesIO
from datetime import datetime
from itertools import combinations as _combinations
from html import escape as _escape_html
from numbers import Integral


def escape_html(value):
    """Convert any displayed value to safe HTML text."""
    return _escape_html(str(value), quote=True)


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


def schedule_fingerprint(schedule):
    """Bir çizelgenin içerik ve eksenlerini kapsayan sabit kimliğini döner."""
    axis = repr((tuple(schedule.index), tuple(schedule.columns))).encode("utf-8")
    values = pd.util.hash_pandas_object(schedule, index=True).values.tobytes()
    return hashlib.sha256(axis + values).hexdigest()


def changed_schedule_columns(previous_schedule, current_schedule):
    """İki çizelge arasındaki değişen gün sütunlarını döner."""
    if (
        previous_schedule is None
        or list(previous_schedule.index) != list(current_schedule.index)
        or list(previous_schedule.columns) != list(current_schedule.columns)
    ):
        return list(current_schedule.columns)
    return [
        column for column in current_schedule.columns
        if not previous_schedule[column].equals(current_schedule[column])
    ]


def build_limit_violation_messages(schedule, names, person_limits):
    """Return the user-facing messages for personal shift-limit violations."""
    messages = []
    for name in names:
        limits = person_limits.get(name, {})
        min_limit = limits.get("min", 0)
        max_limit = limits.get("max", 999)
        total = int(schedule.loc[name].sum())
        if min_limit > 0 and total < min_limit:
            messages.append(
                f"🔻 **{name}**: Min {min_limit} nöbet gerekli, şu an {total} atanmış."
            )
        if max_limit < 999 and total > max_limit:
            messages.append(
                f"🔺 **{name}**: Max {max_limit} nöbet aşıldı, şu an {total} atanmış."
            )
    return messages


def undo_history_state(session_state, current_snapshot):
    """Move the current state to redo history and return the undo snapshot."""
    if not session_state.get("undo_history"):
        return None
    session_state.setdefault("redo_history", []).append(current_snapshot)
    return session_state["undo_history"].pop()


def redo_history_state(session_state, current_snapshot):
    """Move the current state to undo history and return the redo snapshot."""
    if not session_state.get("redo_history"):
        return None
    session_state.setdefault("undo_history", []).append(current_snapshot)
    return session_state["redo_history"].pop()


def use_compact_schedule_editor(person_count, day_count, threshold=500):
    """Çok büyük çizelgelerde hücre başına bileşen yerine tek tablo kullanılır."""
    return person_count * day_count > threshold


def normalize_preference_grid(preferences):
    """Tercih tablosunu yalnızca geçerli 0-3 tam sayı kodlarıyla normalleştirir."""
    normalized = preferences.apply(pd.to_numeric, errors="coerce")
    invalid = (
        normalized.isna().any().any()
        or ((normalized < 0) | (normalized > 3)).any().any()
        or (normalized % 1 != 0).any().any()
    )
    return None if invalid else normalized.astype(int)


def build_schedule_analysis(
    schedule,
    rows_liste,
    role_names,
    isimler,
    gun_detaylari,
    person_limits,
    zorunlu_saat,
    nobet_ucreti,
):
    """Çizelge görünümünde kullanılan saf analiz tablolarını üretir."""
    all_role_counts = {role: {isim: 0 for isim in isimler} for role in role_names}
    for row in rows_liste:
        for role in role_names:
            person = row.get(role, "-")
            if person in all_role_counts[role]:
                all_role_counts[role][person] += 1

    weekend_columns = [column for column in schedule.columns if gun_detaylari[column]["weekend"]]
    special_columns = [
        column for column in schedule.columns
        if gun_detaylari[column]["weekend"] or gun_detaylari[column]["holiday"]
    ]
    totals = schedule.sum(axis=1)
    weekend_totals = schedule[weekend_columns].sum(axis=1) if weekend_columns else pd.Series(0, index=isimler)
    special_totals = schedule[special_columns].sum(axis=1) if special_columns else pd.Series(0, index=isimler)

    stats_load = []
    stats_finance = []
    for isim in isimler:
        toplam = int(totals.get(isim, 0))
        saat = toplam * 24
        fm_saat = max(0, saat - zorunlu_saat)
        ucret = fm_saat * nobet_ucreti
        limits = person_limits.get(isim, {})
        min_limit = limits.get("min", 0)
        max_limit = limits.get("max", 999)
        if min_limit > 0 and max_limit < 999:
            limit_text = f"{min_limit}-{max_limit}"
        elif min_limit > 0:
            limit_text = f"≥{min_limit}"
        elif max_limit < 999:
            limit_text = f"≤{max_limit}"
        else:
            limit_text = "-"

        stats_load.append({
            "İsim": isim,
            "Toplam": toplam,
            "Özel": int(special_totals.get(isim, 0)),
            **{role: all_role_counts[role].get(isim, 0) for role in role_names},
            "Limit": limit_text,
            "✓": "🟢" if min_limit <= toplam <= max_limit else "🔴",
        })
        stats_finance.append({
            "İsim": isim,
            "Nöbet": int(saat),
            "Mesai": int(zorunlu_saat),
            "FM": int(fm_saat),
            "Ücret (TL)": round(ucret, 2),
        })

    pair_matrix = pd.DataFrame(0, index=isimler, columns=isimler, dtype=int)
    for column in schedule.columns:
        assigned = schedule.index[schedule[column]].tolist()
        for first, second in _combinations(assigned, 2):
            pair_matrix.loc[first, second] += 1
            pair_matrix.loc[second, first] += 1
    pair_display = pair_matrix.astype(str)
    for isim in isimler:
        pair_display.loc[isim, isim] = "-"

    return (
        pd.DataFrame(stats_load).set_index("İsim"),
        pd.DataFrame(stats_finance).set_index("İsim"),
        pair_display,
    )


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
    unique_holidays = sorted(set(tatil_gunleri))
    invalid_holidays = [h for h in unique_holidays if h < 1 or h > gun_sayisi]
    if invalid_holidays:
        errors.append(f"❌ Geçersiz tatil günleri: {invalid_holidays}")

    # Dinlenme süresi doğrulama
    if min_bosluk < 0 or min_bosluk > 7:
        errors.append("❌ Dinlenme süresi 0-7 gün arasında olmalı")

    # Fizibilite uyarıları
    working_days = gun_sayisi - len([h for h in unique_holidays if 1 <= h <= gun_sayisi])
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
    """Kişisel limit metnini sözlüğe çevirir: {'Ali': {'min': 5, 'max': 10}}.

    Geçersiz satırlar (boş isim, negatif değer, min > max, hatalı format)
    sessizce atlanır; UI katmanı (nobet.py) kullanıcıya uyarı gösterir.
    """
    person_limits = {}
    if not limits_text or not limits_text.strip():
        return person_limits
    for line in limits_text.strip().split('\n'):
        line = line.strip()
        if not line or ':' not in line:
            continue
        name_part, _, range_part = line.partition(':')
        name = name_part.strip()
        range_part = range_part.strip()
        if not name or '-' not in range_part:
            continue
        try:
            # rsplit so that a negative min like "-1-10" is parsed correctly
            min_str, max_str = range_part.rsplit('-', 1)
            min_val = int(min_str)
            max_val = int(max_str)
        except ValueError:
            continue
        if min_val < 0 or max_val < 0 or min_val > max_val:
            continue
        person_limits[name] = {'min': min_val, 'max': max_val}
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
        week_val = (datetime(yil, ay, gun).toordinal() - weekday) // 7
        gun_detaylari[col_key] = {
            'day_num': gun,
            'weekend': is_weekend,
            'holiday': gun in tatil_gunleri,
            'full_date': full_date,
            'weekday': weekday,
            'week': week_val,
        }
    return gun_detaylari


def _find_valid_group(adaylar, kisi_sayisi, forbidden_pairs):
    """Sıralı listeden forbidden_pairs'e uymayan kisi_sayisi boyutunda grup bulur.
    Önce greedy (sıra korunur), başarısız olursa tam kombinasyon araması yapar."""
    secilenler = []
    for p in adaylar:
        valid = True
        if forbidden_pairs:
            for s in secilenler:
                if tuple(sorted((p, s))) in forbidden_pairs:
                    valid = False
                    break
        if valid:
            secilenler.append(p)
            if len(secilenler) >= kisi_sayisi:
                return secilenler
    # Greedy yetmedi; tüm kombinasyonları dene (geçerli varsa mutlaka bulur)
    if len(secilenler) < kisi_sayisi and forbidden_pairs and len(adaylar) >= kisi_sayisi:
        for combo in _combinations(adaylar, kisi_sayisi):
            ok = True
            for i in range(len(combo)):
                for j in range(i + 1, len(combo)):
                    if tuple(sorted((combo[i], combo[j]))) in forbidden_pairs:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return list(combo)
    return secilenler


def run_scheduling_core(isimler, sutunlar, df_unwanted_bool, gun_detaylari,
                        min_bosluk, kisi_sayisi, forbidden_pairs=None,
                        person_limits=None, df_preferred=None,
                        simulation_count=10, progress_callback=None):
    """
    Saf zamanlama algoritması (Streamlit bağımsız).
    progress_callback(int) isteğe bağlı ilerleme bildirimi için kullanılır.

    simulation_count pozitif bir tamsayı olmalıdır; aksi halde ValueError yükseltilir.
    """
    if isinstance(simulation_count, bool) or not isinstance(simulation_count, Integral) or simulation_count <= 0:
        raise ValueError("Simülasyon sayısı pozitif bir tamsayı olmalıdır.")

    best_schedule = None
    best_score = float('inf')

    for attempt in range(int(simulation_count)):
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
            weekend_num = info.get('week', (gun_no - 1) // 7)

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
                secilenler = _find_valid_group(adaylar, kisi_sayisi, forbidden_pairs)

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
    month_name = escape_html(ay_isimleri[ay])
    html = f"""<html><head><meta charset="utf-8"><title>Nöbet - {month_name} {yil}</title></head>
    <body><h1>Nöbet Listesi - {month_name} {yil}</h1>
    <table><tr>{''.join(f'<th>{escape_html(col)}</th>' for col in df_liste.columns)}</tr>"""
    for _, row in df_liste.iterrows():
        html += f"<tr>{''.join(f'<td>{escape_html(val)}</td>' for val in row)}</tr>"
    html += "</table></body></html>"
    return html
