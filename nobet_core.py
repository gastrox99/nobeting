# -*- coding: utf-8 -*-
"""
nobet_core.py - Streamlit bağımsız, test edilebilir saf fonksiyonlar
nobet.py'deki iş mantığı buraya taşınarak unit test yazılabilir hale getirildi.
"""
import pandas as pd
import numpy as np
import random
import hashlib
import json
import math
import re
from datetime import date, datetime, timedelta, timezone
from itertools import combinations as _combinations
from html import escape as _escape_html
from numbers import Integral


def normalize_owner_email(value):
    """Kullanıcıyı eşleştirmek için e-postayı tek biçime getirir."""
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def escape_html(value):
    """Convert any displayed value to safe HTML text."""
    return _escape_html(str(value), quote=True)


# Türkiye 2016'dan beri yaz saati uygulamıyor; sunucu UTC'de çalışsa da
# "bugün" ve yazdırma saati Türkiye saatine göre olsun.
TR_TZ = timezone(timedelta(hours=3))


def now_tr():
    """Türkiye saatiyle şu an."""
    return datetime.now(TR_TZ)


def as_bool_frame(frame):
    """Kayıttan gelen çizelgeyi bool'a çevirir; eksik hücre (NaN) atanmamış sayılır."""
    return frame.map(lambda value: False if pd.isna(value) else bool(value)).astype(bool)


_TR_MONTHS = {
    "oca": 1, "ocak": 1,
    "şub": 2, "sub": 2, "şubat": 2, "subat": 2,
    "mar": 3, "mart": 3,
    "nis": 4, "nisan": 4,
    "may": 5, "mayıs": 5, "mayis": 5,
    "haz": 6, "haziran": 6,
    "tem": 7, "temmuz": 7,
    "ağu": 8, "agu": 8, "ağustos": 8, "agustos": 8,
    "eyl": 9, "eylül": 9, "eylul": 9,
    "eki": 10, "ekim": 10,
    "kas": 11, "kasım": 11, "kasim": 11,
    "ara": 12, "aralık": 12, "aralik": 12,
}

_ASCII_FOLD = str.maketrans("çğıöşü", "cgiosu")


def _tr_lower(text):
    """Türkçe küçük harf: "NİSAN" → "nisan", "KASIM" → "kasım".

    Python'un lower()/casefold() işlevi "İ" harfini "i̇" (noktalı iki karakter)
    yaptığı için büyük harfli NİSAN, HAZİRAN ve EKİM tanınmıyordu.
    """
    return text.replace("İ", "i").replace("I", "ı").lower()


_TR_MONTHS_ASCII = {key.translate(_ASCII_FOLD): month for key, month in _TR_MONTHS.items()}


def _month_from_name(name):
    lowered = _tr_lower(name)
    if lowered in _TR_MONTHS:
        return _TR_MONTHS[lowered]
    # "NISAN" gibi Türkçe karaktersiz yazımlar
    return _TR_MONTHS_ASCII.get(lowered.translate(_ASCII_FOLD))


def _valid_import_date(day, month=None, year=None):
    """Return date components only when an explicitly supplied date is valid."""
    if month is None:
        return {"day": day, "month": None, "year": None} if 1 <= day <= 31 else None
    if year is None:
        return {"day": day, "month": month, "year": None} if 1 <= day <= 31 else None
    try:
        datetime(year, month, day)
    except (TypeError, ValueError):
        return None
    return {"day": day, "month": month, "year": year}


def parse_schedule_import_date(value):
    """Parse dates used by imported schedule spreadsheets.

    The exported schedule historically used ``1 Haz Çar`` (without a year),
    while users may upload real Excel dates or strings such as
    ``01.06.2026``.  The return value keeps missing month/year information as
    ``None`` so the caller can preserve support for the legacy format without
    pretending that its year was verified.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (pd.Timestamp, datetime, date)):
        return _valid_import_date(value.day, value.month, value.year)

    if isinstance(value, (Integral, float, np.integer, np.floating)) and not isinstance(value, bool):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = math.nan
        if math.isfinite(numeric_value):
            if numeric_value.is_integer() and 1 <= int(numeric_value) <= 31:
                return _valid_import_date(int(numeric_value))
            # Excel's serial date representation (1899-12-30 origin).
            if numeric_value > 31:
                try:
                    timestamp = pd.Timestamp("1899-12-30") + pd.to_timedelta(
                        numeric_value, unit="D"
                    )
                    return _valid_import_date(
                        timestamp.day, timestamp.month, timestamp.year
                    )
                except (TypeError, ValueError, OverflowError):
                    return None

    text = str(value).strip()
    if not text:
        return None

    # Numeric date forms: dd.mm.yyyy, dd/mm/yyyy, yyyy-mm-dd and yyyy/mm/dd.
    match = re.match(r"^(\d{1,2})[./](\d{1,2})[./](\d{4})(?:\s|$)", text)
    if match:
        day, month, year = (int(part) for part in match.groups())
        return _valid_import_date(day, month, year)
    match = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:\s|$)", text)
    if match:
        year, month, day = (int(part) for part in match.groups())
        return _valid_import_date(day, month, year)

    # Localized labels: "1 Haz Çar", "1 Haziran 2026 Çar".
    match = re.match(r"^(\d{1,2})\s+([^\W\d_]+)(?:\s+(\d{4}))?", text, re.IGNORECASE)
    if match:
        day = int(match.group(1))
        month = _month_from_name(match.group(2))
        year = int(match.group(3)) if match.group(3) else None
        if month is not None:
            return _valid_import_date(day, month, year)

    # Keep accepting manually prepared sheets that contain only a day number.
    match = re.match(r"^(\d{1,2})(?:\s|$)", text)
    if match:
        return _valid_import_date(int(match.group(1)))
    return None


# Excel yükleme sınırları: uygulamanın kendi Excel'i birkaç KB ve en fazla
# 31 gün satırı içerir. Büyük dosyalar sunucunun belleğini doldurabiliyordu.
EXCEL_IMPORT_MAX_BYTES = 2 * 1024 * 1024
EXCEL_IMPORT_MAX_ROWS = 40


def import_schedule_table(table, isimler, sutunlar, gun_detaylari, role_names, yil, ay):
    """Excel'den okunan "Tarih + görev sütunları" tablosunu çizelgeye çevirir.

    Session state'e dokunmaz; sonucu sözlük olarak döner:
    format_ok, schedule, rows, matched, skipped_names, date_mismatch,
    unreadable_rows. Tarihlerden biri başka aya/yıla aitse date_mismatch dolu
    olur ve çağıran yüklemeyi iptal etmelidir.
    """
    result = {
        "format_ok": False, "schedule": None, "rows": None, "matched": 0,
        "skipped_names": [], "date_mismatch": None, "unreadable_rows": 0,
    }
    role_columns = [column for column in table.columns if column != "Tarih"]
    if "Tarih" not in table.columns or not role_columns:
        return result
    result["format_ok"] = True
    day_count = len(sutunlar)
    column_by_day = {gun_detaylari[column]["day_num"]: column for column in sutunlar}
    schedule = pd.DataFrame(False, index=isimler, columns=sutunlar)
    rows_by_day = {}
    skipped = set()
    for _, source_row in table.iterrows():
        parsed = parse_schedule_import_date(source_row.get("Tarih"))
        if parsed is None:
            result["unreadable_rows"] += 1
            continue
        day = parsed["day"]
        if (
            (parsed["month"] is not None and parsed["month"] != ay)
            or (parsed["year"] is not None and parsed["year"] != yil)
            or day > max(column_by_day, default=day_count)
        ):
            if result["date_mismatch"] is None:
                result["date_mismatch"] = parsed
            continue
        column = column_by_day.get(day)
        if column is None:
            continue
        row = {"Tarih": gun_detaylari[column]["full_date"]}
        for index, role in enumerate(role_names):
            source_column = role_columns[index] if index < len(role_columns) else None
            person = str(source_row.get(source_column, "-")).strip() if source_column else "-"
            if person and person not in ("-", "nan") and person in schedule.index:
                schedule.at[person, column] = True
                row[role] = person
            else:
                if person and person not in ("-", "nan"):
                    skipped.add(person)
                row[role] = "-"
        rows_by_day[day] = row
    rows = []
    for column in sutunlar:
        day = gun_detaylari[column]["day_num"]
        if day in rows_by_day:
            rows.append(rows_by_day[day])
            result["matched"] += 1
        else:
            empty = {"Tarih": gun_detaylari[column]["full_date"]}
            empty.update({role: "-" for role in role_names})
            rows.append(empty)
    result["schedule"] = schedule
    result["rows"] = rows
    result["skipped_names"] = sorted(skipped)
    return result


PREFERENCE_GRID_STATES = (
    {
        "key": "assigned_neutral",
        "symbol": "✓",
        "color": "#475569",
        "css_class": "preference-state-0",
        "description": "Atandı",
    },
    {
        "key": "assigned_preferred",
        "symbol": "✓",
        "color": "#166534",
        "css_class": "preference-state-1",
        "description": "Atandı + Tercih",
    },
    {
        "key": "assigned_avoid",
        "symbol": "!",
        "color": "#92400e",
        "css_class": "preference-state-2",
        "description": "Atandı + Kaçınma",
    },
    {
        "key": "assigned_unavailable",
        "symbol": "×",
        "color": "#991b1b",
        "css_class": "preference-state-3",
        "description": "Atandı + İzin",
    },
    {
        "key": "conflict",
        "symbol": "🔺",
        "color": "#7f1d1d",
        "css_class": "preference-state-conflict",
        "description": "Çakışma",
    },
    {
        "key": "neutral",
        "soft": True,
        "symbol": "·",
        "color": "#475569",
        "css_class": "preference-free-0",
        "description": "Nötr",
    },
    {
        "key": "preferred",
        "soft": True,
        "symbol": "T",
        "color": "#166534",
        "css_class": "preference-free-1",
        "description": "Tercih",
    },
    {
        "key": "avoid",
        "soft": True,
        "symbol": "K",
        "color": "#92400e",
        "css_class": "preference-free-2",
        "description": "Kaçınma",
    },
    {
        "key": "unavailable",
        "soft": True,
        "symbol": "Y",
        "color": "#991b1b",
        "css_class": "preference-free-3",
        "description": "İzin",
    },
)

_PREFERENCE_GRID_STATE_BY_KEY = {
    state["key"]: state for state in PREFERENCE_GRID_STATES
}


def get_preference_grid_state(is_assigned, preference_value, has_conflict=False):
    """Return the canonical symbol and styling for one schedule-grid cell."""
    try:
        preference_value = int(preference_value)
    except (TypeError, ValueError):
        preference_value = 0

    if has_conflict:
        state_key = "conflict"
    elif is_assigned:
        state_key = {
            1: "assigned_preferred",
            2: "assigned_avoid",
            3: "assigned_unavailable",
        }.get(preference_value, "assigned_neutral")
    else:
        state_key = {
            1: "preferred",
            2: "avoid",
            3: "unavailable",
        }.get(preference_value, "neutral")

    return dict(_PREFERENCE_GRID_STATE_BY_KEY[state_key])


def name_column_width(names, char_px=9, minimum=96, maximum=220):
    """İsim sütunu genişliği (px): renk noktası + en uzun isim + sayı rozeti.

    Çok uzun isimler ``maximum``'da kesilir ve "…" ile kısaltılır.
    """
    longest = max((len(str(name)) for name in names), default=0)
    width = 11 + 4 + longest * char_px + 4 + 30 + 10  # nokta, boşluk, isim, boşluk, rozet, kenar
    return int(max(minimum, min(maximum, width)))


def get_preference_grid_legend():
    """Return the legend entries in the same order as the cell states."""
    return [dict(state) for state in PREFERENCE_GRID_STATES]


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


def missing_shift_coverage(schedule, columns, required_people):
    """Gerekli kişi sayısına ulaşmayan günleri (sütun, atanan kişi) olarak döner."""
    missing = []
    for column in columns:
        assigned = int(schedule[column].sum())
        if assigned < required_people:
            missing.append((column, assigned))
    return missing


MISSING_DAY_FREE_LABEL = "uygun, elle atanabilir"


def explain_missing_day(schedule, col, gun_detaylari, df_unwanted_bool, min_bosluk,
                        person_limits=None, forbidden_pairs=None, initial_last_shift=None):
    """Eksik nöbetçili bir gün için atanmamış her kişinin neden uygun olmadığını döner.

    Sonuç sıralı (neden, [açıklamalı isimler]) listesidir. "uygun" grubundaki
    kişiler hiçbir kurala takılmıyor; o güne elle atanabilirler.
    """
    limits = person_limits or {}
    initial = initial_last_shift or {}
    day = gun_detaylari[col]["day_num"]
    assigned = [p for p in schedule.index if bool(schedule.at[p, col])]
    groups = {"unwanted": [], "rest": [], "limit": [], "forbidden": [], "free": []}
    for person in schedule.index:
        if person in assigned:
            continue
        if person in df_unwanted_bool.index and col in df_unwanted_bool.columns and bool(df_unwanted_bool.at[person, col]):
            groups["unwanted"].append(person)
            continue
        near = [
            gun_detaylari[other]["day_num"] for other in schedule.columns
            if other != col and other in gun_detaylari and bool(schedule.at[person, other])
            and abs(gun_detaylari[other]["day_num"] - day) <= min_bosluk
        ]
        previous = initial.get(person)
        if near:
            groups["rest"].append(f"{person} ({min(near, key=lambda d: abs(d - day))}. gün nöbetli)")
            continue
        if previous is not None and day - previous <= min_bosluk:
            groups["rest"].append(f"{person} (geçen ayın sonunda nöbetli)")
            continue
        max_limit = limits.get(person, {}).get("max", 999)
        if int(schedule.loc[person].sum()) >= max_limit:
            groups["limit"].append(f"{person} (en fazla {max_limit})")
            continue
        mates = [m for m in assigned if tuple(sorted((person, m))) in (forbidden_pairs or [])]
        if mates:
            groups["forbidden"].append(f"{person} ({', '.join(mates)} ile)")
            continue
        groups["free"].append(person)
    labels = [
        ("unwanted", "izinli"),
        ("rest", "dinlenmede"),
        ("limit", "nöbet limitinde"),
        ("forbidden", "birlikte çalışamaz"),
        ("free", MISSING_DAY_FREE_LABEL),
    ]
    return [(label, groups[key]) for key, label in labels if groups[key]]


def format_missing_day_reason(reasons):
    """explain_missing_day sonucunu tek satırlık metne çevirir."""
    if not reasons:
        return "Ekipte bu güne atanabilecek başka kimse yok."
    return " · ".join(f"{label}: {', '.join(names)}" for label, names in reasons)


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


def synchronize_changed_role_rows(
    rows_liste,
    previous_schedule,
    current_schedule,
    role_names,
    day_columns,
    day_details,
):
    """Keep role rows aligned with changed schedule columns.

    Existing valid role placements are retained and newly assigned people fill
    open roles. A copied list is returned so it can be saved atomically with
    the current schedule.
    """
    synchronized_rows = [dict(row) for row in (rows_liste or [])]
    for column in changed_schedule_columns(previous_schedule, current_schedule):
        row_index = day_columns.index(column)
        current_row = (
            synchronized_rows[row_index]
            if row_index < len(synchronized_rows)
            else {"Tarih": day_details[column]["full_date"]}
        )
        assigned_people = current_schedule.index[current_schedule[column]].tolist()
        new_row = {"Tarih": current_row.get("Tarih", day_details[column]["full_date"])}
        assigned_in_roles = []
        for role_name in role_names:
            person = current_row.get(role_name, "-")
            if person in assigned_people and person != "-":
                new_row[role_name] = person
                assigned_in_roles.append(person)
            else:
                new_row[role_name] = "-"
        for person in assigned_people:
            if person not in assigned_in_roles:
                for role_name in role_names:
                    if new_row[role_name] == "-":
                        new_row[role_name] = person
                        break
        if row_index < len(synchronized_rows):
            synchronized_rows[row_index] = new_row
        else:
            synchronized_rows.append(new_row)
    return synchronized_rows


def assignment_warnings(
    schedule,
    person,
    column,
    columns,
    day_details,
    min_rest,
    person_limits=None,
    preferences=None,
    forbidden_pairs=None,
    people_per_shift=None,
    previous_last_shift=None,
):
    """Bir kişi bir güne atandıktan hemen sonra gösterilecek uyarıları döner.

    ``schedule`` atama yapılmış hâlidir; kişi o gün atanmamışsa liste boştur.
    """
    if (
        person not in schedule.index
        or column not in schedule.columns
        or not bool(schedule.at[person, column])
    ):
        return []

    warnings = []
    day = day_details[column]["day_num"]

    nearby_days = sorted(
        day_details[other]["day_num"]
        for other in columns
        if other != column
        and other in schedule.columns
        and bool(schedule.at[person, other])
        and abs(day_details[other]["day_num"] - day) <= min_rest
    )
    if nearby_days:
        days_text = ", ".join(str(d) for d in nearby_days)
        warnings.append(
            f"{person}: {days_text}. günde de nöbeti var — {min_rest} gün dinlenme kuralına uymuyor."
        )
    previous_day = (previous_last_shift or {}).get(person)
    if previous_day is not None and day - previous_day <= min_rest:
        ago = "son günü" if previous_day == 0 else f"son gününden {-previous_day} gün önce"
        warnings.append(
            f"{person}: önceki ayın {ago} nöbeti var — {min_rest} gün dinlenme kuralına uymuyor."
        )

    max_limit = (person_limits or {}).get(person, {}).get("max", 999)
    total = int(schedule.loc[person].sum())
    if max_limit < 999 and total > max_limit:
        warnings.append(f"{person}: toplam {total} nöbet, en fazla {max_limit} olmalı.")

    if (
        preferences is not None
        and person in preferences.index
        and column in preferences.columns
    ):
        try:
            preference = int(preferences.at[person, column])
        except (TypeError, ValueError):
            preference = 0
        if preference == 3:
            warnings.append(f"{person}: {day}. gün izinli olarak işaretli.")
        elif preference == 2:
            warnings.append(f"{person}: {day}. gün kaçınmak istediği gün.")

    others = [p for p in schedule.index[schedule[column]].tolist() if p != person]
    for other in others:
        if tuple(sorted((person, other))) in (forbidden_pairs or set()):
            warnings.append(f"{day}. gün: {person} ve {other} birlikte çalışamaz.")

    if people_per_shift is not None and len(others) + 1 > people_per_shift:
        warnings.append(
            f"{day}. gün: {len(others) + 1} kişi atandı, en fazla {people_per_shift} olmalı."
        )

    return warnings


def apply_role_edits(original_row, edited_values, role_names, assigned_today):
    """Görev yeri düzenlemesini uygular: (yeni_satır, değişti_mi).

    Bir göreve o gün başka görevde olan kişi yazılırsa ikisi yer değiştirir;
    böylece aynı kişi iki birimde görünmez, diğeri de listeden düşmez.
    O gün nöbeti olmayan isimler yok sayılır.
    """
    new_row = dict(original_row)
    changed = False
    for role in role_names:
        value = str(edited_values.get(role, "-")).strip()
        # Yalnızca kullanıcının gerçekten değiştirdiği hücreler işlenir.
        if value == str(original_row.get(role, "-")).strip():
            continue
        if not value or value in ("nan", "-") or value not in assigned_today:
            continue
        if new_row.get(role) == value:
            continue
        previous = new_row.get(role, "-")
        for other in role_names:
            if other != role and new_row.get(other) == value:
                new_row[other] = previous
                break
        new_row[role] = value
        changed = True
    return new_row, changed


def ignored_role_edits(original_row, edited_values, role_names, assigned_today):
    """apply_role_edits'in uygulamadığı düzenlemeler: [(görev, yazılan), ...].

    Kullanıcıya neden değişmediğini söylemek için: o gün nöbeti olmayan bir isim
    yazılmışsa ya da hücre boşaltılmışsa (kişiyi görevden almak nöbeti
    kaldırmakla olur).
    """
    ignored = []
    for role in role_names:
        value = str(edited_values.get(role, "-")).strip()
        if value == str(original_row.get(role, "-")).strip():
            continue
        if not value or value in ("nan", "-") or value not in assigned_today:
            ignored.append((role, value if value not in ("nan",) else ""))
    return ignored


RESERVED_ROLE_NAMES = ("Tarih",)


def normalize_role_names(text, count):
    """Görev adlarını virgüllü metinden tam ``count`` tane, benzersiz olarak üretir.

    Aynı ad iki kez yazılınca ("AYB, AYB") ya da bir göreve "Tarih" denince
    satırdaki kişiler birbirinin üzerine yazılıp listeden kayboluyordu.
    Tekrar eden adlar "AYB 2" gibi numaralanır; eksikler GörevN ile tamamlanır.
    Dönüş: (adlar, değiştirilen_adlar)
    """
    raw = [part.strip() for part in (text or "").split(",") if part.strip()]
    names = []
    adjusted = []
    used = set(name.casefold() for name in RESERVED_ROLE_NAMES)

    def _unique(base):
        candidate = base
        number = 2
        while candidate.casefold() in used:
            candidate = f"{base} {number}"
            number += 1
        return candidate

    for name in raw[:count]:
        unique = _unique(name)
        if unique != name:
            adjusted.append((name, unique))
        used.add(unique.casefold())
        names.append(unique)
    index = 1
    while len(names) < count:
        candidate = f"Görev{index}"
        index += 1
        if candidate.casefold() in used:
            continue
        used.add(candidate.casefold())
        names.append(candidate)
    return names, adjusted


def detect_renames(old_names, new_names):
    """Ekip listesinde yeniden adlandırılan kişileri bulur: {eski: yeni}.

    Yalnızca kişi sayısı aynı kaldığında ve aynı sıradaki bir isim, listede
    başka yerde olmayan yeni bir isimle değiştirildiğinde ad değişikliği
    sayılır; ekleme/çıkarma/sıra değişikliği ad değişikliği değildir.
    """
    old_names = list(old_names or [])
    new_names = list(new_names or [])
    if not old_names or len(old_names) != len(new_names):
        return {}
    renames = {}
    old_set, new_set = set(old_names), set(new_names)
    for old, new in zip(old_names, new_names):
        if old == new:
            continue
        if old in new_set or new in old_set:
            return {}
        renames[old] = new
    return renames


def rename_people_in_rows(rows, renames):
    """Görev satırlarındaki isimleri yeni adlarına çevirir."""
    if not rows or not renames:
        return rows
    return [
        {key: (renames.get(value, value) if key != "Tarih" else value) for key, value in row.items()}
        for row in rows
    ]


def align_rows_to_roles(rows, old_roles, new_roles, schedule, day_columns, day_details):
    """Görev satırlarını yeni görev düzenine taşır ve çizelgeyle eşitler.

    Görev adları ya da nöbet başına kişi sayısı değiştikten sonra eski bir ay
    açılınca (ya da geri alınınca) satırlar eski düzende kalıyordu; yeni görev
    sütunu olmadığı için ikinci nöbetçi listede görünmüyordu. Kişiler sırayla
    yeni görevlere taşınır, eksikler boş görevlere eklenir.
    """
    old_roles = list(old_roles or [])
    remapped = []
    for row in rows or []:
        new_row = {"Tarih": row.get("Tarih", "")}
        for index, role in enumerate(new_roles):
            new_row[role] = row.get(old_roles[index], "-") if index < len(old_roles) else "-"
        # Kaldırılan görevlerdeki kişiler boş kalan görevlere geçer.
        for old_role in old_roles[len(new_roles):]:
            person = row.get(old_role, "-")
            if person in (None, "", "-") or person in new_row.values():
                continue
            for role in new_roles:
                if new_row[role] in (None, "", "-"):
                    new_row[role] = person
                    break
        remapped.append(new_row)
    return synchronize_changed_role_rows(remapped, None, schedule, list(new_roles), day_columns, day_details)


def drop_unknown_people_from_rows(rows, people):
    """Ekipte olmayan isimleri görev satırlarından "-" ile siler."""
    if not rows:
        return rows
    people = set(people)
    return [
        {
            key: (value if key == "Tarih" or value in people or value in (None, "", "-") else "-")
            for key, value in row.items()
        }
        for row in rows
    ]


def record_renames(aliases, renames):
    """Ad değişikliklerini {eski_ad: güncel_ad} sözlüğüne işler.

    Diğer aylar veritabanında eski adla kalır; açılırken bu sözlükle güncel
    ada çevrilir. Zincirler (A→B, B→C) tek adıma indirilir, geri dönüş
    (A→B, B→A) eski kaydı siler.
    """
    aliases = dict(aliases or {})
    for old, new in (renames or {}).items():
        if old == new:
            continue
        aliases.pop(new, None)
        for key, value in list(aliases.items()):
            if value == old:
                aliases[key] = new
        aliases[old] = new
    return {key: value for key, value in aliases.items() if key != value}


def resolve_alias(name, aliases, team):
    """Kayıttaki eski adı ekipteki güncel adına çevirir.

    Ekipte zaten olan bir isim (ör. eski adı alan yeni biri) değişmez; güncel
    adı ekipte olmayan eski adlar da olduğu gibi kalır.
    """
    if name in team or not aliases:
        return name
    seen = set()
    current = name
    while current in aliases and current not in seen:
        seen.add(current)
        current = aliases[current]
    return current if current in team else name


def _alias_map(names, aliases, team):
    """Kayıttaki isimler için {eski: yeni}; aynı kişiye iki kayıt düşerse dokunmaz."""
    names = list(names)
    mapping = {}
    for name in names:
        target = resolve_alias(name, aliases, team)
        if target != name and target not in names and target not in mapping.values():
            mapping[name] = target
    return mapping


def _clean_hidden(value):
    """Kayıttaki saklı kişileri doğrular: {kişi: {"shifts": [gün], "prefs": {"gün": kod}}}."""
    cleaned = {}
    if not isinstance(value, dict):
        return cleaned
    for person, data in value.items():
        if not isinstance(person, str) or not isinstance(data, dict):
            continue
        shifts = sorted({
            day for day in data.get("shifts", [])
            if isinstance(day, int) and not isinstance(day, bool) and 1 <= day <= 31
        })
        prefs = {}
        raw_prefs = data.get("prefs", {})
        if isinstance(raw_prefs, dict):
            for day, code in raw_prefs.items():
                if (
                    str(day).isdecimal() and 1 <= int(day) <= 31
                    and isinstance(code, int) and not isinstance(code, bool) and 1 <= code <= 3
                ):
                    prefs[str(int(day))] = code
        if shifts or prefs:
            cleaned[person] = {"shifts": shifts, "prefs": prefs}
    return cleaned


def stash_removed_people(hidden, schedule, preferences, team):
    """Ekipten çıkan kişilerin bu aydaki nöbet ve tercihlerini saklar.

    Eskiden ekipten silinen (ya da virgülü bozulan) bir ismin nöbetleri
    hemen kaydedilip kalıcı olarak kayboluyordu. Saklanan veri gün
    numarasıyla tutulur; isim ekibe geri yazılınca ``take_back_people``
    ile geri gelir. Dönüş: (saklananlar, bu çağrıda saklanan kişiler).
    """
    hidden = {person: dict(data) for person, data in (hidden or {}).items()}
    stashed = []
    if schedule is None:
        return hidden, stashed
    team = set(team)
    for person in schedule.index:
        if person in team:
            continue
        shifts = sorted(
            _column_day(column) for column in schedule.columns
            if _column_day(column) is not None and bool(schedule.at[person, column])
        )
        prefs = {}
        if preferences is not None and person in preferences.index:
            for column in preferences.columns:
                day = _column_day(column)
                try:
                    code = int(preferences.at[person, column])
                except (TypeError, ValueError):
                    continue
                if day is not None and 1 <= code <= 3:
                    prefs[str(day)] = code
        if shifts or prefs:
            hidden[person] = {"shifts": shifts, "prefs": prefs}
            stashed.append(person)
    return hidden, stashed


def take_back_people(hidden, schedule, preferences, team):
    """Ekibe geri yazılan kişilerin saklı nöbet ve tercihlerini geri koyar.

    ``schedule`` ve ``preferences`` yerinde güncellenir.
    Dönüş: (kalan saklılar, geri gelen kişiler).
    """
    hidden = dict(hidden or {})
    returned = []
    if schedule is None:
        return hidden, returned
    day_to_column = {_column_day(column): column for column in schedule.columns}
    for person in team:
        if person not in hidden or person not in schedule.index:
            continue
        data = hidden.pop(person)
        for day in data.get("shifts", []):
            column = day_to_column.get(day)
            if column is not None:
                schedule.at[person, column] = True
        if preferences is not None and person in preferences.index:
            for day, code in data.get("prefs", {}).items():
                column = day_to_column.get(int(day))
                if column is not None and column in preferences.columns:
                    preferences.at[person, column] = int(code)
        returned.append(person)
    return hidden, returned


def update_locked_cells(locked, cells_now_assigned, cells_now_free):
    """Elle atanan hücreleri sabitler, elle kaldırılanların sabitliğini siler."""
    updated = set(locked or ())
    updated |= set(cells_now_assigned)
    updated -= set(cells_now_free)
    return updated


def active_locks(locked, schedule):
    """Hâlâ atanmış olan sabit hücreler (ekipten çıkan/kaldırılanlar düşer)."""
    return {
        (person, column) for person, column in (locked or ())
        if person in schedule.index and column in schedule.columns
        and bool(schedule.at[person, column])
    }


def locks_to_json(locked):
    return sorted([person, column] for person, column in (locked or ()))


def locks_from_saved(saved_locks, saved_day_by_column, gun_detaylari, team):
    """Kayıttaki [kişi, sütun] listesini bu ayın sütunlarına eşler."""
    day_to_column = {details["day_num"]: column for column, details in gun_detaylari.items()}
    restored = set()
    for item in saved_locks or ():
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        person, saved_column = item
        day = saved_day_by_column.get(saved_column, _column_day(saved_column))
        column = day_to_column.get(day)
        if person in team and column is not None:
            restored.add((person, column))
    return restored


def find_person_role(rows_liste, role_names, day_index, person):
    """Return the role assigned to a person on a schedule day, if any."""
    if not rows_liste or day_index < 0 or day_index >= len(rows_liste):
        return None

    row = rows_liste[day_index]
    if not isinstance(row, dict):
        return None

    return next(
        (role_name for role_name in (role_names or []) if row.get(role_name) == person),
        None,
    )


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


def update_persistence_feedback(session_state, saved, error_message=None):
    """Keep database persistence feedback visible across a Streamlit rerun."""
    if saved:
        session_state["auto_save_failed"] = False
        session_state.pop("persistence_error", None)
        return None

    message = error_message or "❌ Çizelge kaydedilemedi. Veritabanı bağlantısını kontrol edin."
    session_state["auto_save_failed"] = True
    session_state["persistence_error"] = message
    return message


def restore_auto_saved_schedule(
    session_state,
    load_schedule_fn,
    schedule_name,
    yil,
    ay,
    isimler,
    sutunlar,
    gun_detaylari,
    role_names=None,
    on_error=None,
    aliases=None,
):
    """Otomatik kaydı session state'e geri yükler; başarısız olursa state'e dokunmaz.

    ``on_error`` yalnızca yükleyici bir istisna oluşturduğunda çağrılır. Böylece
    veritabanında henüz otomatik kayıt bulunmaması ile gerçek bir DB hatası
    birbirinden ayrılabilir.

    ``aliases`` ({eski_ad: güncel_ad}) ile kayıttaki eski adlar güncel ekibe
    eşlenir. Ekipte olmayan kişilerin verisi silinmez, ``hidden_people``
    içinde saklanır.
    """
    try:
        loaded = load_schedule_fn(schedule_name, yil, ay)
    except Exception as exc:
        if on_error is not None:
            on_error(exc)
        return False

    if not isinstance(loaded, tuple) or len(loaded) < 4:
        return False

    _, saved_schedule, saved_preferences, saved_settings = loaded[:4]
    saved_rows = loaded[4] if len(loaded) > 4 else None
    if saved_schedule is None or saved_schedule.empty:
        return False
    # Eksik hücre (NaN) bool'a çevrilince True olurdu; atanmamış say.
    saved_schedule = as_bool_frame(saved_schedule)

    # Ad değişikliğinden önceki kayıtlar eski adla durur; güncel ada çevir.
    renames = _alias_map(saved_schedule.index, aliases, isimler)
    saved_hidden = _clean_hidden(saved_settings.get("hidden")) if isinstance(saved_settings, dict) else {}
    if renames:
        saved_schedule = saved_schedule.rename(index=renames)
        if saved_preferences is not None:
            saved_preferences = saved_preferences.rename(index=renames)
        if saved_rows:
            saved_rows = rename_people_in_rows(saved_rows, renames)
        if isinstance(saved_settings, dict) and saved_settings.get("locked"):
            saved_settings = dict(saved_settings)
            saved_settings["locked"] = [
                [renames.get(item[0], item[0]), item[1]]
                for item in saved_settings["locked"]
                if isinstance(item, (list, tuple)) and len(item) == 2
            ]
    hidden_renames = _alias_map(saved_hidden, aliases, isimler)
    saved_hidden = {hidden_renames.get(person, person): data for person, data in saved_hidden.items()}

    restored_schedule = pd.DataFrame(False, index=isimler, columns=sutunlar)
    saved_days = {}
    for saved_column in saved_schedule.columns:
        try:
            saved_days[int(str(saved_column).split()[0])] = saved_column
        except (ValueError, IndexError):
            continue

    for person in isimler:
        if person not in saved_schedule.index:
            continue
        for column in sutunlar:
            saved_column = saved_days.get(gun_detaylari[column]["day_num"])
            if saved_column is not None:
                try:
                    restored_schedule.at[person, column] = bool(saved_schedule.at[person, saved_column])
                except Exception:
                    continue

    session_state["schedule_bool"] = restored_schedule
    if isinstance(saved_settings, dict) and saved_settings.get("locked"):
        session_state["locked_cells"] = active_locks(
            locks_from_saved(
                saved_settings["locked"],
                {column: day for day, column in saved_days.items()},
                gun_detaylari,
                isimler,
            ),
            restored_schedule,
        )

    restored_rows = None
    if saved_rows:
        saved_rows_by_day = {}
        for saved_row in saved_rows:
            try:
                saved_day = int(str(saved_row.get("Tarih", "")).split(".")[0])
            except (ValueError, TypeError, IndexError):
                continue
            saved_rows_by_day[saved_day] = dict(saved_row)

        if saved_rows_by_day:
            restored_rows = []
            for column in sutunlar:
                saved_row = saved_rows_by_day.get(gun_detaylari[column]["day_num"])
                if saved_row is None:
                    restored_rows = None
                    break
                saved_row["Tarih"] = gun_detaylari[column].get(
                    "full_date", saved_row.get("Tarih", "")
                )
                restored_rows.append(saved_row)
            # Ekipten çıkarılan kişiler listede kalmasın.
            if restored_rows:
                restored_rows = drop_unknown_people_from_rows(restored_rows, isimler)

    if restored_rows:
        restored_role_names = []
        for restored_row in restored_rows:
            for restored_role_name in restored_row:
                if (
                    restored_role_name != "Tarih"
                    and restored_role_name not in restored_role_names
                ):
                    restored_role_names.append(restored_role_name)
        active_role_names = role_names or restored_role_names
        if list(restored_role_names) != list(active_role_names):
            # Kayıt eski görev düzeniyle yapılmış: satırları güncel görevlere taşı.
            restored_rows = align_rows_to_roles(
                restored_rows, restored_role_names, active_role_names,
                restored_schedule, sutunlar, gun_detaylari,
            )
            restored_role_names = list(active_role_names)
        session_state["cached_rows_liste"] = restored_rows
        session_state["cached_role_names"] = restored_role_names or active_role_names
        session_state["cached_key"] = (
            yil, ay, tuple(isimler), tuple(sutunlar), tuple(active_role_names)
        )
        session_state["cached_role_schedule"] = restored_schedule.copy(deep=True)
        session_state["should_regenerate_assignments"] = False
    else:
        # Legacy schedules do not contain role rows; regenerate them normally.
        session_state["should_regenerate_assignments"] = True

    if saved_preferences is not None:
        restored_preferences = pd.DataFrame(0, index=isimler, columns=sutunlar)
        for person in isimler:
            if person not in saved_preferences.index:
                continue
            for column in sutunlar:
                saved_column = saved_days.get(gun_detaylari[column]["day_num"])
                if saved_column is not None and saved_column in saved_preferences.columns:
                    try:
                        restored_preferences.at[person, column] = int(
                            saved_preferences.at[person, saved_column]
                        )
                    except Exception:
                        continue
        session_state["pref_df"] = restored_preferences

    # Ekipte olmayanların verisi silinmesin: saklanır, ekibe dönünce geri gelir.
    hidden, _ = stash_removed_people(saved_hidden, saved_schedule, saved_preferences, isimler)
    if session_state.get("pref_df") is None and any(
        hidden.get(person, {}).get("prefs") for person in isimler
    ):
        session_state["pref_df"] = pd.DataFrame(0, index=isimler, columns=sutunlar)
    hidden, returned = take_back_people(
        hidden, restored_schedule, session_state.get("pref_df"), isimler
    )
    session_state["hidden_people"] = hidden
    if returned and not session_state.get("should_regenerate_assignments"):
        # Geri gelen kişiler görev satırlarına yerleşsin.
        session_state["cached_role_schedule"] = None

    return True


# Kalıcı ayar adı -> (session_state anahtarı, doğrulayıcı)
def _int_in(low, high):
    def _check(value):
        if isinstance(value, bool):
            return None
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        return number if low <= number <= high and number == value else None
    return _check


def _text(value):
    return value if isinstance(value, str) else None


def _period_texts(value):
    """{"2026-10": "29", ...} biçimindeki aylık metinleri doğrular."""
    if not isinstance(value, dict):
        return None
    cleaned = {
        key: text for key, text in value.items()
        if isinstance(key, str) and re.fullmatch(r"\d{4}-\d{2}", key) and isinstance(text, str)
    }
    return cleaned


def period_key(yil, ay):
    """Aya özel ayarlar (ör. tatiller) için anahtar: "2026-10"."""
    return f"{int(yil):04d}-{int(ay):02d}"


def _bool(value):
    return value if isinstance(value, bool) else None


def _non_negative_float(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _aliases(value):
    """{eski_ad: güncel_ad} biçimindeki ad değişikliği kayıtlarını doğrular."""
    if not isinstance(value, dict):
        return None
    return {
        old: new for old, new in value.items()
        if isinstance(old, str) and isinstance(new, str) and old and new and old != new
    }


# Yıl seçimi aralığı (eskiden 2030'da bitiyordu; Aralık 2030'da varsayılan
# "gelecek ay" 2031 olacağı için uygulama açılmazdı).
YEAR_MIN = 2024
YEAR_MAX = 2100

APP_SETTINGS_FIELDS = {
    "team": ("isimler_text", _text),
    "year": ("yil_val", _int_in(YEAR_MIN, YEAR_MAX)),
    "month": ("ay_val", _int_in(1, 12)),
    "people_per_shift": ("kisi_sayisi_val", _int_in(1, 5)),
    "rest_days": ("min_bosluk_val", _int_in(0, 3)),
    "holidays_by_period": ("holidays_by_period", _period_texts),
    "hourly_rate": ("nobet_ucreti_val", _non_negative_float),
    "role_names": ("role_names_text", _text),
    "forbidden_pairs": ("forbidden_pairs_text", _text),
    "limits": ("limits_text", _text),
    "balance_history": ("balance_history", _bool),
    "name_aliases": ("name_aliases", _aliases),
    "last_backup": ("last_backup_at", _text),
}


def apply_app_settings(session_state, settings):
    """Kayıtlı ayarları session state'e yazar; geçersiz değerleri atlar."""
    if not isinstance(settings, dict):
        return
    for name, (state_key, check) in APP_SETTINGS_FIELDS.items():
        if name in settings:
            value = check(settings[name])
            if value is not None:
                session_state[state_key] = value


def collect_app_settings(session_state):
    """Session state'teki güncel ayarları kaydedilecek sözlüğe çevirir."""
    settings = {
        name: session_state.get(state_key)
        for name, (state_key, _check) in APP_SETTINGS_FIELDS.items()
        if session_state.get(state_key) is not None
    }
    # Derin kopya: iç içe sözlükler (aylık tatiller) sonradan değişince
    # "son kaydedilen" kopya da değişip kaydı engellemesin.
    return json.loads(json.dumps(settings, ensure_ascii=False))


def persistence_fingerprint(schedule, preferences, rows, locked=None, hidden=None):
    """Veritabanına yazılan anlık görüntünün (çizelge+tercih+roller+sabitler+saklılar) kimliği."""
    parts = [
        schedule_fingerprint(schedule) if schedule is not None else "-",
        schedule_fingerprint(preferences) if preferences is not None else "-",
        json.dumps(rows or [], ensure_ascii=False, sort_keys=True, default=str),
        json.dumps(locks_to_json(locked), ensure_ascii=False),
        json.dumps(hidden or {}, ensure_ascii=False, sort_keys=True),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


PERIOD_STATE_KEYS = (
    "schedule_bool",
    "pref_df",
    "cached_rows_liste",
    "cached_role_names",
    "cached_role_schedule",
    "cached_key",
    "undo_history",
    "redo_history",
    "incomplete_simulation",
    "reset_pending",
    "prev_month_context",
    "locked_cells",
    "hidden_people",
)


def switch_schedule_period(session_state, new_period, save_fn=None):
    """Ay/yıl değiştiğinde eski dönemin verisini kaydedip state'i temizler.

    Eski çizelgenin yeni aya taşınması, otomatik kayıtla birlikte o ayın
    kayıtlı çizelgesinin üzerine yazılmasına yol açıyordu. Dönem değişince
    önce giden dönem ``save_fn(period, schedule, preferences, rows)`` ile
    kaydedilir, ardından döneme bağlı tüm state silinir; böylece çağıran taraf
    yeni dönemin kaydını sıfırdan yükleyebilir. Dönem değiştiyse True döner.
    """
    old_period = session_state.get("loaded_period")
    if old_period is None or tuple(old_period) == tuple(new_period):
        return False

    schedule = session_state.get("schedule_bool")
    if save_fn is not None and schedule is not None:
        try:
            save_fn(
                tuple(old_period),
                schedule,
                session_state.get("pref_df"),
                session_state.get("cached_rows_liste"),
            )
        except Exception:
            pass

    for key in PERIOD_STATE_KEYS:
        session_state.pop(key, None)
    session_state.pop("loaded_period", None)
    return True


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


def preference_summary(schedule, preferences, people):
    """Analiz'deki ekip tercih özeti: kişi başına tercih ve kaçınma sonuçları.

    Tercih kodları: 1 = tercih, 2 = kaçın, 3 = izin. Kimsenin işaretli
    günü yoksa None döner.
    """
    rows = []
    any_marked = False
    for person in people:
        counts = {1: [0, 0], 2: [0, 0], 3: [0, 0]}  # [işaretli gün, o güne düşen nöbet]
        for column in schedule.columns:
            try:
                code = int(preferences.at[person, column])
            except (KeyError, TypeError, ValueError):
                code = 0
            if code in counts:
                counts[code][0] += 1
                if person in schedule.index and bool(schedule.at[person, column]):
                    counts[code][1] += 1
        any_marked = any_marked or any(marked for marked, _ in counts.values())
        rows.append({
            "İsim": person,
            "🟢 Tercih günü": counts[1][0],
            "Tercihte nöbet": counts[1][1],
            "Başarı": f"%{round(counts[1][1] / counts[1][0] * 100)}" if counts[1][0] else "-",
            "🟡 Kaçın günü": counts[2][0],
            "Kaçına düşen": counts[2][1],
            "🔴 İzinde nöbet": counts[3][1],
        })
    if not any_marked:
        return None
    return pd.DataFrame(rows).set_index("İsim")


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
    friday_columns = [column for column in schedule.columns if is_friday_column(gun_detaylari[column])]
    friday_totals = schedule[friday_columns].sum(axis=1) if friday_columns else pd.Series(0, index=isimler)
    special_totals = schedule[special_columns].sum(axis=1) if special_columns else pd.Series(0, index=isimler)
    holiday_columns = [column for column in schedule.columns if gun_detaylari[column]["holiday"]]
    holiday_totals = schedule[holiday_columns].sum(axis=1) if holiday_columns else pd.Series(0, index=isimler)

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
            "Cuma": int(friday_totals.get(isim, 0)),
            "Tatil": int(holiday_totals.get(isim, 0)),
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

    # Fizibilite uyarıları — tatiller de nöbet tutulan günlerdir.
    total_positions_needed = gun_sayisi * kisi_sayisi
    team_size = len(isimler)
    # Dinlenme kuralıyla bir kişi en fazla her (min_bosluk + 1) günde bir nöbet tutar.
    max_per_person = -(-gun_sayisi // (max(0, min_bosluk) + 1))
    capacity = team_size * max_per_person

    if team_size < kisi_sayisi:
        errors.append(f"❌ {kisi_sayisi} kişi nöbet için en az {kisi_sayisi} kişi gerekli")
    elif team_size > 0 and capacity < total_positions_needed:
        warnings.append(
            f"⚠️ {min_bosluk} gün dinlenme kuralıyla {team_size} kişi bu ay en fazla "
            f"{capacity} nöbet tutabilir, ama {total_positions_needed} nöbet gerekiyor — "
            "bazı günler eksik kalacak. Dinlenmeyi azaltın veya ekibe kişi ekleyin."
        )
    elif team_size > 0 and total_positions_needed < team_size:
        warnings.append(f"⚠️ Nöbetleri dağıtmak için çok fazla kişi var ({team_size} kişi, {total_positions_needed} pozisyon)")

    return len(errors) == 0, errors, warnings


def _split_pair(pair_str, team):
    """"Ali-Ayşe" → ("Ali", "Ayşe"). İsimde tire varsa ("Ayşe-Nur-Ali") ekipteki
    isimlere uyan bölme seçilir; uyan yoksa ilk tireden bölünür."""
    positions = [i for i, ch in enumerate(pair_str) if ch == "-"]
    if team:
        team = set(team)
        for i in positions:
            left, right = pair_str[:i].strip(), pair_str[i + 1:].strip()
            if left in team and right in team:
                return left, right
    left, right = pair_str.split("-", 1)
    return left.strip(), right.strip()


def parse_forbidden_pairs(forbidden_input, team=None):
    """Birlikte çalışamayan çiftleri metin girdisinden küme olarak döner.

    ``team`` verilirse isminde tire olan kişiler doğru ayrılır."""
    forbidden_pairs = set()
    if not forbidden_input or not forbidden_input.strip():
        return forbidden_pairs
    all_pairs = []
    for line in forbidden_input.strip().split('\n'):
        all_pairs.extend(line.split(','))
    for pair_str in all_pairs:
        pair_str = pair_str.strip()
        if '-' in pair_str:
            p1, p2 = _split_pair(pair_str, team)
            if p1 and p2:
                forbidden_pairs.add(tuple(sorted((p1, p2))))
    return forbidden_pairs


def parse_person_limits_with_issues(limits_text):
    """Kişisel limitleri ayrıştırır: (limitler, hatalı_satırlar, hata_mesajları).

    "İsim:min-max" biçimindeki her satırı okur. Biçimi bozuk satırlar ile
    negatif / min > max değerler atlanır ve kullanıcıya gösterilmek üzere
    ayrıca döndürülür.
    """
    person_limits = {}
    invalid_lines = []
    errors = []
    if not limits_text or not limits_text.strip():
        return person_limits, invalid_lines, errors
    for line in limits_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        name_part, separator, range_part = line.partition(':')
        name = name_part.strip()
        range_part = range_part.strip()
        if not separator or not name or '-' not in range_part:
            invalid_lines.append(line)
            continue
        try:
            # rsplit so that a negative min like "-1-10" is parsed correctly
            min_str, max_str = range_part.rsplit('-', 1)
            min_val = int(min_str)
            max_val = int(max_str)
        except ValueError:
            invalid_lines.append(line)
            continue
        if min_val < 0 or max_val < 0:
            errors.append(f"**{name}**: negatif değer ({range_part}) — atlandı")
        elif min_val > max_val:
            errors.append(f"**{name}**: min ({min_val}) > max ({max_val}) — atlandı")
        else:
            if name in person_limits:
                errors.append(
                    f"**{name}** birden fazla satırda — son satır ({min_val}-{max_val}) kullanılıyor"
                )
            person_limits[name] = {'min': min_val, 'max': max_val}
    return person_limits, invalid_lines, errors


def parse_person_limits(limits_text):
    """Kişisel limit metnini sözlüğe çevirir: {'Ali': {'min': 5, 'max': 10}}."""
    return parse_person_limits_with_issues(limits_text)[0]


def pair_label(pair):
    """Seçim listesinde ikilinin adı: "Ali – Ayşe"."""
    first, second = sorted(pair)
    return f"{first} – {second}"


def forbidden_pair_choices(team, current_pairs):
    """Ayarlardaki seçim listesi: {etiket: ikili}. Ekipten olası bütün ikililer
    ve kayıtlı olup ekipte olmayan isimli ikililer (uyarı gösterilsin diye)."""
    pairs = {tuple(sorted(pair)) for pair in _combinations(sorted(set(team)), 2)}
    pairs |= {tuple(sorted(pair)) for pair in current_pairs}
    return {pair_label(pair): pair for pair in sorted(pairs)}


def forbidden_pairs_to_text(pairs):
    """İkilileri ayarlarda saklanan metne çevirir: "Ali-Ayşe, Can-Ece"."""
    return ", ".join(f"{a}-{b}" for a, b in sorted(tuple(sorted(pair)) for pair in pairs))


def person_limits_to_text(limits):
    """Kişisel limitleri saklanan metne çevirir: "Ali:3-8" satırları (999 = sınırsız)."""
    return "\n".join(f"{name}:{value['min']}-{value['max']}" for name, value in limits.items())


def unknown_names(names, team):
    """Ekipte olmayan isimleri sıralı ve tekrarsız döner."""
    team_set = set(team)
    return sorted({name for name in names if name not in team_set})


TR_GUN_KISA = {0: "Pzt", 1: "Sal", 2: "Çar", 3: "Per", 4: "Cum", 5: "Cmt", 6: "Paz"}


def _balance_fridays(schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
                     df_preferred, min_bosluk, forbidden_pairs, fixed_assignments,
                     initial_last_shift, max_moves=200):
    """Cuma sayılarındaki farkı takasla azaltır; toplamlar değişmez.

    Çok Cuması olan kişinin bir Cuma nöbeti, az Cuması olan kişinin bir hafta
    içi nöbetiyle yer değiştirir. Hafta sonu/tatil nöbetlerine dokunulmaz.
    """
    friday_cols = [c for c in sutunlar if is_friday_column(gun_detaylari[c])]
    weekday_cols = [
        c for c in sutunlar
        if not gun_detaylari[c]["weekend"] and not gun_detaylari[c]["holiday"] and c not in friday_cols
    ]
    return _balance_by_swaps(
        schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool, df_preferred,
        min_bosluk, forbidden_pairs, fixed_assignments, initial_last_shift,
        friday_cols, weekday_cols, max_moves=max_moves,
    )


def _balance_holidays(schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
                      df_preferred, min_bosluk, forbidden_pairs, fixed_assignments,
                      initial_last_shift, history_holiday, max_moves=200):
    """Bayram/tatil sayılarındaki farkı (varsa son 12 ay dahil) takasla azaltır.

    Çok tatil nöbeti olanın bir tatil nöbeti, az olanın tatil olmayan bir
    hafta sonu nöbetiyle yer değiştirir. Toplam, hafta sonu/tatil ve Cuma
    sayıları değişmez; yalnızca hangi özel günün kimde olduğu değişir.
    """
    holiday_cols = [c for c in sutunlar if gun_detaylari[c]["holiday"]]
    weekend_cols = [
        c for c in sutunlar if gun_detaylari[c]["weekend"] and not gun_detaylari[c]["holiday"]
    ]
    return _balance_by_swaps(
        schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool, df_preferred,
        min_bosluk, forbidden_pairs, fixed_assignments, initial_last_shift,
        holiday_cols, weekend_cols, extra_counts=history_holiday, max_moves=max_moves,
        swap_ok=_no_new_back_to_back_weekends(schedule, sutunlar, gun_detaylari),
    )


def _balance_specials(schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
                      df_preferred, min_bosluk, forbidden_pairs, fixed_assignments,
                      initial_last_shift, history_special, max_moves=200):
    """Hafta sonu/tatil sayılarındaki farkı (önceki aylar dahil) takasla azaltır.

    Simülasyon seçimi hafta sonu dengesini aya yayma ve bayram takibiyle
    birlikte tartar; bu yüzden fark bazen 2 kalabiliyordu. Çok hafta
    sonu/tatil nöbeti olanın bir özel gün nöbeti, az olanın Cuma olmayan bir
    hafta içi nöbetiyle yer değiştirir. Toplam ve Cuma sayıları değişmez.
    Önce art arda hafta sonu oluşturmayan takaslar denenir; fark yine 2 ya
    da daha fazla kalırsa denge önce gelir (art arda hafta sonu olabilir).
    """
    special_cols = [c for c in sutunlar if gun_detaylari[c]["weekend"] or gun_detaylari[c]["holiday"]]
    weekday_cols = [
        c for c in sutunlar
        if c not in special_cols and not is_friday_column(gun_detaylari[c])
    ]
    schedule = _balance_by_swaps(
        schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool, df_preferred,
        min_bosluk, forbidden_pairs, fixed_assignments, initial_last_shift,
        special_cols, weekday_cols, extra_counts=history_special, max_moves=max_moves,
        swap_ok=_no_new_back_to_back_weekends(schedule, sutunlar, gun_detaylari),
    )
    return _balance_by_swaps(
        schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool, df_preferred,
        min_bosluk, forbidden_pairs, fixed_assignments, initial_last_shift,
        special_cols, weekday_cols, extra_counts=history_special, max_moves=max_moves,
    )


def _no_new_back_to_back_weekends(schedule, sutunlar, gun_detaylari):
    """Takas art arda hafta sonu nöbeti oluşturmasın (simülasyon bundan kaçınır)."""
    weekend_days = [c for c in sutunlar if gun_detaylari[c]["weekend"]]

    def back_to_back(person):
        weeks = sorted({
            gun_detaylari[c].get("week", (gun_detaylari[c]["day_num"] - 1) // 7)
            for c in weekend_days if bool(schedule.at[person, c])
        })
        return sum(1 for a, b in zip(weeks, weeks[1:]) if b == a + 1)

    def swap_ok(giver, receiver, apply_swap, undo_swap):
        before = back_to_back(giver) + back_to_back(receiver)
        apply_swap()
        after = back_to_back(giver) + back_to_back(receiver)
        undo_swap()
        return after <= before

    return swap_ok


def _balance_by_swaps(schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
                      df_preferred, min_bosluk, forbidden_pairs, fixed_assignments,
                      initial_last_shift, target_cols, partner_cols, extra_counts=None,
                      max_moves=200, swap_ok=None):
    """``target_cols`` sayılarındaki farkı takasla azaltır; toplamlar değişmez.

    Çok sayısı olan kişinin bir hedef gün nöbeti, az olanın bir eş gün
    (``partner_cols``) nöbetiyle yer değiştirir. ``extra_counts`` geçmiş
    aylardan gelen farktır. İkisi de yeni gününde müsait olmalı, kaçınma
    günü olmamalı, dinlenme kuralına ve yasak çiftlere uymalı; sabit
    nöbetler yerinden oynamaz.
    """
    fixed = set(fixed_assignments or ())
    initial = initial_last_shift or {}
    extra = extra_counts or {}
    day_of = {col: gun_detaylari[col]["day_num"] for col in sutunlar}
    target_cols = list(target_cols)
    partner_cols = list(partner_cols)
    if not target_cols or not partner_cols:
        return schedule

    def avoided(person, col):
        if df_preferred is None or person not in df_preferred.index or col not in df_preferred.columns:
            return False
        try:
            return int(df_preferred.at[person, col]) == 2
        except (TypeError, ValueError):
            return False

    def can_take(person, col, dropped):
        if bool(df_unwanted_bool.at[person, col]) or avoided(person, col):
            return False
        day = day_of[col]
        previous = initial.get(person)
        if previous is not None and day - previous <= min_bosluk:
            return False
        for other in sutunlar:
            if other in (col, dropped) or not bool(schedule.at[person, other]):
                continue
            if abs(day_of[other] - day) <= min_bosluk:
                return False
        if forbidden_pairs:
            for mate in schedule.index[schedule[col]]:
                if tuple(sorted((person, mate))) in forbidden_pairs:
                    return False
        return True

    for _ in range(max_moves):
        counts = {
            p: sum(1 for c in target_cols if bool(schedule.at[p, c])) + extra.get(p, 0)
            for p in isimler
        }
        move = None
        for giver in sorted(isimler, key=lambda p: -counts[p]):
            for receiver in sorted(isimler, key=lambda p: counts[p]):
                if counts[giver] - counts[receiver] <= 1:
                    break
                for tcol in target_cols:
                    if not bool(schedule.at[giver, tcol]) or bool(schedule.at[receiver, tcol]):
                        continue
                    if (giver, tcol) in fixed:
                        continue
                    for pcol in partner_cols:
                        if not bool(schedule.at[receiver, pcol]) or bool(schedule.at[giver, pcol]):
                            continue
                        if (receiver, pcol) in fixed:
                            continue
                        # Takası önce uygula, iki tarafı da yeni hâliyle kontrol et.
                        schedule.at[giver, tcol] = False
                        schedule.at[receiver, pcol] = False
                        ok = can_take(receiver, tcol, pcol) and can_take(giver, pcol, tcol)
                        schedule.at[giver, tcol] = True
                        schedule.at[receiver, pcol] = True
                        if ok and swap_ok is not None:
                            def _apply(g=giver, r=receiver, t=tcol, q=pcol):
                                schedule.at[g, t] = False
                                schedule.at[g, q] = True
                                schedule.at[r, q] = False
                                schedule.at[r, t] = True

                            def _undo(g=giver, r=receiver, t=tcol, q=pcol):
                                schedule.at[g, t] = True
                                schedule.at[g, q] = False
                                schedule.at[r, q] = True
                                schedule.at[r, t] = False

                            ok = swap_ok(giver, receiver, _apply, _undo)
                        if ok:
                            move = (giver, receiver, tcol, pcol)
                            break
                    if move:
                        break
                if move:
                    break
            if move:
                break
        if move is None:
            break
        giver, receiver, tcol, pcol = move
        schedule.at[giver, tcol] = False
        schedule.at[giver, pcol] = True
        schedule.at[receiver, pcol] = False
        schedule.at[receiver, tcol] = True
    return schedule


def _fill_open_days(schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
                    df_preferred, min_bosluk, forbidden_pairs, person_limits,
                    initial_last_shift, kisi_sayisi):
    """Dengeleme takaslarından sonra doldurulabilir hâle gelen boş yerleri doldurur.

    Takas birini bir günden çekince o kişi, daha önce kimsenin alamadığı
    bir güne uygun hâle gelebilir. Aday izinli olmamalı, dinlenme kuralına,
    yasak çiftlere ve max limitine uymalı. Kaçınma günü olmayan ve en az
    nöbeti olan önce gelir; kaçınma günü yalnızca başka aday yoksa verilir.
    """
    initial = initial_last_shift or {}
    limits = person_limits or {}
    day_of = {col: gun_detaylari[col]["day_num"] for col in sutunlar}

    def avoided(person, col):
        if df_preferred is None or person not in df_preferred.index or col not in df_preferred.columns:
            return False
        try:
            return int(df_preferred.at[person, col]) == 2
        except (TypeError, ValueError):
            return False

    def special(col):
        info = gun_detaylari[col]
        return bool(info.get("weekend") or info.get("holiday"))

    def eligible(person, col):
        if bool(schedule.at[person, col]) or bool(df_unwanted_bool.at[person, col]):
            return False
        day = day_of[col]
        previous = initial.get(person)
        if previous is not None and day - previous <= min_bosluk:
            return False
        for other in sutunlar:
            if other != col and bool(schedule.at[person, other]) and abs(day_of[other] - day) <= min_bosluk:
                return False
        if forbidden_pairs:
            for mate in schedule.index[schedule[col]]:
                if tuple(sorted((person, mate))) in forbidden_pairs:
                    return False
        max_limit = limits.get(person, {}).get("max")
        if max_limit is not None and int(schedule.loc[person].sum()) >= max_limit:
            return False
        return True

    for col in sutunlar:
        while int(schedule[col].sum()) < kisi_sayisi:
            candidates = [p for p in isimler if eligible(p, col)]
            if not candidates:
                break
            sp = special(col)
            best = min(candidates, key=lambda p: (
                avoided(p, col),
                int(schedule.loc[p].sum()),
                sum(1 for c in sutunlar if special(c) and bool(schedule.at[p, c])) if sp else 0,
                isimler.index(p),
            ))
            schedule.at[best, col] = True
    return schedule


def is_friday_column(info):
    """Tatil olmayan Cuma mı? (Tatile denk gelen Cuma zaten özel gün sayılır.)"""
    return info.get("weekday") == 4 and not info.get("weekend") and not info.get("holiday")


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


def _balance_totals(schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
                    df_preferred, min_bosluk, forbidden_pairs, person_limits,
                    fixed_assignments, initial_last_shift, history_special,
                    max_moves=500, history_total=None, history_holiday=None):
    """Toplam nöbet farkını kurallara uyarak en fazla 1'e indirir.

    Gün gün ilerleyen seçim ay sonunda toplamlarda kayma bırakabiliyordu
    (ör. 62 nöbet / 9 kişide biri 6, biri 8). En çok nöbeti olandan en az
    olana, aynı gün içinde, şu koşullarla bir nöbet aktarılır:
    alan kişi o gün müsait, kaçınma günü değil, dinlenme kuralına ve yasak
    çiftlere uyuyor, max limitini aşmıyor; veren kişinin nöbeti sabit değil
    ve min limitinin altına düşmüyor; hafta sonu/tatil dengesi bozulmuyor.
    Tercih edilen günler mümkün olduğunca korunur.

    Toplam farkı zaten 1 iken önceki aylar (``history_total``) hesaba katılır:
    geçmişiyle birlikte 2 ya da daha fazla önde olan kişinin fazla nöbeti,
    geçmişi az olana geçer. Bu ayın farkı yine 1'i aşmaz.
    Tatil günü aktarılırken son 12 ayın tatil sayısı (``history_holiday``)
    bozulmaz.
    """
    limits = person_limits or {}
    history_total = history_total or {}
    history_holiday = history_holiday or {}
    fixed = set(fixed_assignments or ())
    history = history_special or {}
    initial = initial_last_shift or {}
    day_of = {col: gun_detaylari[col]["day_num"] for col in sutunlar}
    special_cols = {col for col in sutunlar if gun_detaylari[col]["weekend"] or gun_detaylari[col]["holiday"]}
    friday_cols = {col for col in sutunlar if is_friday_column(gun_detaylari[col])}
    holiday_cols = {col for col in sutunlar if gun_detaylari[col]["holiday"]}

    def pref(person, col):
        if df_preferred is None or person not in df_preferred.index or col not in df_preferred.columns:
            return 0
        try:
            return int(df_preferred.at[person, col])
        except (TypeError, ValueError):
            return 0

    def rest_ok(person, col):
        day = day_of[col]
        previous = initial.get(person)
        if previous is not None and day - previous <= min_bosluk:
            return False
        for other in sutunlar:
            if other != col and bool(schedule.at[person, other]) and abs(day_of[other] - day) <= min_bosluk:
                return False
        return True

    for _ in range(max_moves):
        totals = {p: int(schedule.loc[p].sum()) for p in isimler}
        specials = {
            p: sum(1 for c in special_cols if bool(schedule.at[p, c])) + history.get(p, 0)
            for p in isimler
        }
        fridays = {p: sum(1 for c in friday_cols if bool(schedule.at[p, c])) for p in isimler}
        holidays = {
            p: sum(1 for c in holiday_cols if bool(schedule.at[p, c])) + history_holiday.get(p, 0)
            for p in isimler
        }
        overall = {p: totals[p] + history_total.get(p, 0) for p in isimler}
        best_move = None
        for giver in sorted(isimler, key=lambda p: (-totals[p], -overall[p])):
            if totals[giver] - 1 < limits.get(giver, {}).get("min", 0):
                continue
            for receiver in sorted(isimler, key=lambda p: (totals[p], overall[p])):
                gap = totals[giver] - totals[receiver]
                if gap <= 0:
                    break
                # Fark 1 ise yalnızca geçmişiyle birlikte belirgin önde olan verir.
                if gap == 1 and overall[giver] - overall[receiver] < 2:
                    continue
                if totals[receiver] + 1 > limits.get(receiver, {}).get("max", 999):
                    continue
                for col in sutunlar:
                    if not bool(schedule.at[giver, col]) or bool(schedule.at[receiver, col]):
                        continue
                    if (giver, col) in fixed or bool(df_unwanted_bool.at[receiver, col]):
                        continue
                    if pref(receiver, col) == 2:
                        continue
                    # Yalnızca geçmiş yüzünden yapılan aktarım verenin tercih ettiği günü almaz.
                    if gap == 1 and pref(giver, col) == 1:
                        continue
                    if col in special_cols and specials[giver] <= specials[receiver]:
                        continue
                    if col in friday_cols and fridays[giver] <= fridays[receiver]:
                        continue
                    if col in holiday_cols and holidays[giver] <= holidays[receiver]:
                        continue
                    others = [p for p in schedule.index[schedule[col]] if p not in (giver, receiver)]
                    if forbidden_pairs and any(tuple(sorted((receiver, o))) in forbidden_pairs for o in others):
                        continue
                    if not rest_ok(receiver, col):
                        continue
                    # Önce hafta içi; verenin tercih ettiği gün en son, alanın tercih ettiği gün önce.
                    score = (
                        (10 if col in special_cols else 0)
                        + (4 if col in friday_cols else 0)
                        + (5 if pref(giver, col) == 1 else 0)
                        - (3 if pref(receiver, col) == 1 else 0)
                    )
                    if best_move is None or score < best_move[0]:
                        best_move = (score, giver, receiver, col)
                if best_move is not None:
                    break
            if best_move is not None:
                break
        if best_move is None:
            break
        _, giver, receiver, col = best_move
        schedule.at[giver, col] = False
        schedule.at[receiver, col] = True
    return schedule


# Aya yayma ağırlıkları: beklenen aralıktan her eksik gün için seçim cezası
# ve simülasyonlar arasında seçimde kısa aralık başına ceza. Denge
# kurallarından (toplam ×10, Cuma ×40, hafta sonu ×100) bilerek hafiftir.
SPREAD_WEIGHT = 3
SPREAD_SCORE_WEIGHT = 4


def run_scheduling_core(isimler, sutunlar, df_unwanted_bool, gun_detaylari,
                        min_bosluk, kisi_sayisi, forbidden_pairs=None,
                        person_limits=None, df_preferred=None,
                        simulation_count=10, progress_callback=None,
                        initial_last_shift=None, history_special=None,
                        fixed_assignments=None, history_total=None,
                        history_holiday=None):
    """
    Saf zamanlama algoritması (Streamlit bağımsız).
    progress_callback(int) isteğe bağlı ilerleme bildirimi için kullanılır.

    simulation_count pozitif bir tamsayı olmalıdır; aksi halde ValueError yükseltilir.

    initial_last_shift: {kişi: gün} önceki ayın son nöbeti, bu aya göre
        (0 = önceki ayın son günü, -1 = ondan önceki gün). Ay geçişinde
        dinlenme kuralı için kullanılır.
    history_special: {kişi: fark} önceki aylardaki hafta sonu/tatil
        nöbetlerinin ekip ortalamasından farkı; dengelemede hesaba katılır.
    fixed_assignments: {(kişi, sütun)} elle sabitlenmiş nöbetler. Aynen
        korunur; dinlenme kuralı, max limit ve yasak çiftlerde hesaba katılır,
        simülasyon yalnızca kalan boşlukları doldurur.
    history_total: {kişi: fark} önceki aylardaki toplam nöbetlerin ay
        ortalamasından farkı. Bu ayın toplam farkı yine en fazla 1 kalır;
        geçmişi fazla olan, bu ayın "1 fazla" alanlarından olmaz.
    history_holiday: {kişi: fark} son 12 aydaki tatil/bayram nöbetlerinin
        ekip ortalamasından farkı; tatil günleri azı olana verilir.

    Nöbetler aya yayılır: kişinin son nöbetinden bu yana geçen gün,
    ekip için beklenen aralıktan (kişi sayısı / günlük nöbetçi) kısaysa
    hafif ceza alır. Bu ceza denge kurallarından daha hafiftir.
    """
    initial_last_shift = initial_last_shift or {}
    history_special = history_special or {}
    history_total = history_total or {}
    history_holiday = history_holiday or {}
    # Beklenen aralık: herkes eşit nöbet alırsa iki nöbet arası kaç gün olur.
    ideal_gap = len(isimler) / max(1, kisi_sayisi)
    fixed_by_col = {}
    fixed_days = {}
    for person, column in (fixed_assignments or ()):
        if person in isimler and column in gun_detaylari:
            fixed_by_col.setdefault(column, []).append(person)
            fixed_days.setdefault(person, set()).add(gun_detaylari[column]["day_num"])
    for column in fixed_by_col:
        fixed_by_col[column].sort(key=isimler.index)
    if isinstance(simulation_count, bool) or not isinstance(simulation_count, Integral) or simulation_count <= 0:
        raise ValueError("Simülasyon sayısı pozitif bir tamsayı olmalıdır.")

    best_schedule = None
    best_score = float('inf')

    for attempt in range(int(simulation_count)):
        if progress_callback:
            progress_callback(attempt + 1)

        stat_total = {i: 0 for i in isimler}
        stat_special = {i: 0 for i in isimler}
        stat_friday = {i: 0 for i in isimler}
        stat_holiday = {i: 0 for i in isimler}
        short_gaps = 0
        stat_consecutive_weekend = {i: 0 for i in isimler}
        last_weekend_shift = {i: -10 for i in isimler}
        last_shift_day = {i: initial_last_shift.get(i, -10) for i in isimler}

        temp_schedule = pd.DataFrame(
            {col: [False] * len(isimler) for col in sutunlar}, index=isimler
        )
        future_fixed = {i: len(fixed_days.get(i, ())) for i in isimler}

        def get_decision_score(p, is_sp, col, is_friday=False, is_holiday=False, day=None):
            total = stat_total[p] + history_total.get(p, 0) + (random.random() * 0.5)
            sp_count = stat_special[p] + history_special.get(p, 0)
            consecutive_penalty = stat_consecutive_weekend[p] * 200
            pref_bonus = 0
            if df_preferred is not None and p in df_preferred.index and col in df_preferred.columns:
                pref_val = df_preferred.at[p, col]
                if pref_val == 1:
                    pref_bonus = -500
                elif pref_val == 2:
                    pref_bonus = 300
            limit_penalty = 0
            min_bonus = 0
            if person_limits and p in person_limits:
                max_limit = person_limits[p].get('max', 999)
                if stat_total[p] >= max_limit:
                    limit_penalty = 50000
                # Minimumuna ulaşmamış kişi öne alınır (çoğunlukla hafta içi).
                if stat_total[p] < person_limits[p].get('min', 0):
                    min_bonus = -250
            # Son nöbete çok yakın günler hafifçe geri itilir (aya yayma).
            spread_penalty = 0
            if day is not None:
                gap = day - last_shift_day[p]
                if gap < ideal_gap:
                    spread_penalty = (ideal_gap - gap) * SPREAD_WEIGHT
            if is_sp:
                # Bayram/tatil: son 12 ayda az tutan öne; hafta sonu dengesinden (×100) hafif.
                holiday_penalty = (
                    (stat_holiday[p] + history_holiday.get(p, 0)) * 60 if is_holiday else 0
                )
                return ((sp_count * 100) + (total * 10) + consecutive_penalty + pref_bonus
                        + limit_penalty + min_bonus + holiday_penalty + spread_penalty)
            # Cuma da dengelenir; hafta sonu/tatil dengesinden (×100) daha hafif.
            friday_penalty = stat_friday[p] * 40 if is_friday else 0
            return ((total * 10) + (sp_count * 1) + friday_penalty + pref_bonus
                    + limit_penalty + min_bonus + spread_penalty)

        empty_shifts = 0
        limit_violations = 0

        for col in sutunlar:
            info = gun_detaylari[col]
            gun_no = info['day_num']
            is_sp = info['weekend'] or info['holiday']
            is_weekend = info['weekend']
            is_friday = is_friday_column(info)
            is_holiday = bool(info['holiday'])
            weekend_num = info.get('week', (gun_no - 1) // 7)

            # Sabit nöbetler önce ve aynen yerleşir.
            sabitler = fixed_by_col.get(col, [])
            for k in sabitler:
                future_fixed[k] -= 1
            needed = max(0, kisi_sayisi - len(sabitler))

            adaylar = []
            for k in isimler:
                if k in sabitler:
                    continue
                if df_unwanted_bool.at[k, col]:
                    continue
                if (gun_no - last_shift_day[k]) <= min_bosluk:
                    continue
                # Yaklaşan sabit nöbetinin dinlenme aralığına girme.
                if any(0 < f - gun_no <= min_bosluk for f in fixed_days.get(k, ())):
                    continue
                if person_limits and k in person_limits:
                    max_limit = person_limits[k].get('max', 999)
                    # Kalan sabit nöbetleri için yer bırak.
                    if stat_total[k] + future_fixed[k] >= max_limit:
                        continue
                if forbidden_pairs and any(
                    tuple(sorted((k, f))) in forbidden_pairs for f in sabitler
                ):
                    continue
                adaylar.append(k)

            random.shuffle(adaylar)
            adaylar.sort(key=lambda x: get_decision_score(x, is_sp, col, is_friday, is_holiday, gun_no))

            # Yeterli aday yoksa günü tamamen boş bırakma: uyan adayları ata,
            # eksik kalan her nöbetçiyi say (eskiden 1 kişi müsaitken gün boştu).
            secilenler = (
                _find_valid_group(adaylar, needed, forbidden_pairs)[:needed]
                if adaylar and needed else []
            )
            empty_shifts += needed - len(secilenler)
            for k in list(sabitler) + secilenler:
                temp_schedule.at[k, col] = True
                stat_total[k] += 1
                if is_sp:
                    stat_special[k] += 1
                if is_friday:
                    stat_friday[k] += 1
                if is_holiday:
                    stat_holiday[k] += 1
                if gun_no - last_shift_day[k] < ideal_gap * 0.6:
                    short_gaps += 1
                last_shift_day[k] = gun_no
                if is_weekend:
                    if last_weekend_shift[k] >= 0 and weekend_num == last_weekend_shift[k] + 1:
                        stat_consecutive_weekend[k] += 1
                    elif last_weekend_shift[k] >= 0 and weekend_num > last_weekend_shift[k] + 1:
                        stat_consecutive_weekend[k] = 0
                    last_weekend_shift[k] = weekend_num

        if person_limits:
            for p, limits in person_limits.items():
                if stat_total.get(p, 0) < limits.get('min', 0):
                    limit_violations += 1

        totals = list(stat_total.values())
        specials = [stat_special[i] + history_special.get(i, 0) for i in isimler]
        consecutive_weekends = sum(stat_consecutive_weekend.values())
        std_dev_total = np.std(totals)
        std_dev_special = np.std(specials)
        range_total = max(totals) - min(totals)
        # Hafta sonu/tatil farkı Cuma farkından daha ağır cezalandırılır.
        range_special = max(specials) - min(specials)
        fridays = list(stat_friday.values())
        range_friday = max(fridays) - min(fridays)
        holidays = [stat_holiday[i] + history_holiday.get(i, 0) for i in isimler]
        range_holiday = max(holidays) - min(holidays)

        score = (
            (empty_shifts * 10000) +
            (limit_violations * 5000) +
            (consecutive_weekends * 500) +
            (range_total * 100) +
            (range_special * 60) +
            (range_friday * 30) +
            (range_holiday * 20) +
            (short_gaps * SPREAD_SCORE_WEIGHT) +
            (std_dev_total * 10) +
            (std_dev_special * 5)
        )

        if score < best_score:
            best_score = score
            best_schedule = temp_schedule.copy()

    if best_schedule is not None:
        best_schedule = _balance_totals(
            best_schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
            df_preferred, min_bosluk, forbidden_pairs, person_limits,
            {(p, c) for c, people in fixed_by_col.items() for p in people},
            initial_last_shift, history_special,
            history_total=history_total, history_holiday=history_holiday,
        )
        best_schedule = _balance_specials(
            best_schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
            df_preferred, min_bosluk, forbidden_pairs,
            {(p, c) for c, people in fixed_by_col.items() for p in people},
            initial_last_shift, history_special,
        )
        best_schedule = _balance_fridays(
            best_schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
            df_preferred, min_bosluk, forbidden_pairs,
            {(p, c) for c, people in fixed_by_col.items() for p in people},
            initial_last_shift,
        )
        if any(gun_detaylari[c]["holiday"] for c in sutunlar):
            best_schedule = _balance_holidays(
                best_schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
                df_preferred, min_bosluk, forbidden_pairs,
                {(p, c) for c, people in fixed_by_col.items() for p in people},
                initial_last_shift, history_holiday,
            )
        best_schedule = _fill_open_days(
            best_schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
            df_preferred, min_bosluk, forbidden_pairs, person_limits,
            initial_last_shift, kisi_sayisi,
        )
    return best_schedule, best_score


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


def last_shift_offsets(prev_schedule, prev_month_days):
    """Önceki ayın son nöbetlerini bu aya göre gün olarak döner.

    0 önceki ayın son günü, -1 bir önceki gün… Nöbeti olmayan kişi yer almaz.
    """
    offsets = {}
    if prev_schedule is None or prev_schedule.empty:
        return offsets
    for person in prev_schedule.index:
        days = [
            _column_day(column) for column in prev_schedule.columns
            if _column_day(column) is not None and bool(prev_schedule.at[person, column])
        ]
        if days:
            offsets[person] = max(days) - prev_month_days
    return offsets


def special_shift_counts(schedule, yil, ay, holidays=()):
    """Kişi başına hafta sonu veya tatil günü nöbet sayısı."""
    holiday_set = set(holidays)
    counts = {}
    if schedule is None or schedule.empty:
        return counts
    for column in schedule.columns:
        day = _column_day(column)
        if day is None:
            continue
        try:
            is_special = date(yil, ay, day).weekday() >= 5 or day in holiday_set
        except ValueError:
            continue
        if not is_special:
            continue
        for person in schedule.index[schedule[column].astype(bool)]:
            counts[person] = counts.get(person, 0) + 1
    return counts


def history_special_adjustment(history_counts, team, clip=2):
    """Geçmiş özel gün nöbetlerini ekip ortalamasına göre farka çevirir.

    Fark ±``clip`` ile sınırlanır; böylece yeni gelen biri geçmişi olmadığı
    için bütün hafta sonlarını almaz.
    """
    if not team:
        return {}
    mean = sum(history_counts.get(person, 0) for person in team) / len(team)
    return {
        person: max(-clip, min(clip, history_counts.get(person, 0) - mean))
        for person in team
    }


def schedule_totals(schedule):
    """Kişi başına toplam nöbet (çizelgedeki herkes, nöbetsizler 0)."""
    if schedule is None:
        return {}
    return {person: int(schedule.loc[person].astype(bool).sum()) for person in schedule.index}


def holiday_shift_counts(schedule, holidays):
    """Kişi başına tatil (bayram, yılbaşı vb.) günü nöbet sayısı; herkes yer alır."""
    holiday_set = set(holidays)
    if schedule is None or schedule.empty:
        return {}
    counts = {person: 0 for person in schedule.index}
    for column in schedule.columns:
        if _column_day(column) in holiday_set:
            for person in schedule.index[schedule[column].astype(bool)]:
                counts[person] += 1
    return counts


def history_adjustment(monthly_counts, team, clip=2, monthly_available=None):
    """Aylık sayıları kişi başına "ortalamadan fark"a çevirir ve toplar.

    ``monthly_counts``: her ay için {kişi: sayı}; o ay çizelgede olan herkes
    yer alır (0 dahil). Her ay yalnızca o ay ekipte olanların ortalamasıyla
    karşılaştırılır; o ay ekipte olmayanın farkı 0'dır. Böylece yeni gelen,
    geçmişi olmadığı için "borçlu" sayılmaz. Sonuç ±``clip`` ile sınırlanır.

    ``monthly_available``: (isteğe bağlı) her ay için {kişi: müsait gün}.
    Verilirse kişinin payı müsait olduğu güne göre oranlanır: izinli ya da
    müsait değil olduğu için az tutan "eksik" sayılmaz.
    """
    team = list(team or [])
    totals = {person: 0.0 for person in team}
    monthly_available = list(monthly_available or ())
    for index, counts in enumerate(monthly_counts or ()):
        present = [person for person in team if person in counts]
        if len(present) < 2:
            continue
        month_sum = sum(counts[person] for person in present)
        available = monthly_available[index] if index < len(monthly_available) else None
        if available:
            weights = {person: max(0, available.get(person, 0)) for person in present}
            weight_sum = sum(weights.values())
            if weight_sum <= 0:
                continue
            expected = {person: month_sum * weights[person] / weight_sum for person in present}
        else:
            expected = {person: month_sum / len(present) for person in present}
        for person in present:
            totals[person] += counts[person] - expected[person]
    return {person: max(-clip, min(clip, round(value, 2))) for person, value in totals.items()}


def unavailable_day_counts(pref_json):
    """Kayıtlı tercih JSON'undan kişi başına izin (3) gün sayısı."""
    try:
        data = json.loads(pref_json) if pref_json else None
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    counts = {}
    for column in data.values():
        if not isinstance(column, dict):
            continue
        for person, value in column.items():
            try:
                if int(value) == 3:
                    counts[str(person)] = counts.get(str(person), 0) + 1
            except (TypeError, ValueError):
                continue
    return counts


def _row_day_and_weekday(row):
    tarih = str(row.get("Tarih", "")).strip()
    parsed = parse_schedule_import_date(tarih)
    weekday = tarih.split()[-1] if tarih.split() else ""
    return (parsed["day"] if parsed else None), weekday


def build_share_text(rows, role_names, yil, ay, special_days=()):
    """WhatsApp'a yapıştırmaya hazır günlük liste."""
    special = set(special_days)
    lines = [f"📅 {TR_AY_ADLARI[ay]} {yil} Nöbet Listesi", ""]
    for row in rows:
        day, weekday = _row_day_and_weekday(row)
        label = f"{day} {TR_AY_ADLARI[ay]} {weekday}".strip() if day else str(row.get("Tarih", ""))
        if day in special:
            label = f"*{label}*"
        parts = [
            f"{role}: {row.get(role)}" for role in role_names
            if row.get(role) not in (None, "", "-")
        ]
        lines.append(f"{label} — {' · '.join(parts) if parts else 'boş'}")
    return "\n".join(lines)


def shared_default_month(months, today):
    """Paylaşım sayfasında ilk açılacak ay: bu ay, yoksa bugünden önceki son ay.

    ``months``: sıralı [(yıl, ay), ...]. Bugünden önce ay yoksa ilki seçilir.
    Dönen değer listedeki sıra (boş listede None).
    """
    if not months:
        return None
    current = (today.year, today.month)
    if current in months:
        return months.index(current)
    earlier = [i for i, period in enumerate(months) if period <= current]
    return earlier[-1] if earlier else 0


def shared_day_states(rows, yil, ay, today):
    """Her satır için "past", "today" ya da "future" (tarihi okunamayan: "future")."""
    states = []
    for row in rows:
        day, _ = _row_day_and_weekday(row)
        try:
            row_date = date(yil, ay, day) if day else None
        except ValueError:
            row_date = None
        if row_date is None or row_date > today:
            states.append("future")
        elif row_date == today:
            states.append("today")
        else:
            states.append("past")
    return states


def on_duty_summary(rows, role_names, yil, ay, today):
    """Bugün ve yarın nöbetçi olanlar: [("Bugün", "Ali (AYB) · Ayşe (GYB)"), ...].

    Yalnızca bu ayın satırlarındaki günler yazılır; nöbetçisi olmayan gün "boş".
    """
    wanted = {today: "Bugün", today + timedelta(days=1): "Yarın"}
    found = {}
    for row in rows:
        day, _ = _row_day_and_weekday(row)
        try:
            row_date = date(yil, ay, day) if day else None
        except ValueError:
            continue
        if row_date not in wanted:
            continue
        parts = []
        for role in role_names:
            value = row.get(role)
            if value in (None, "", "-"):
                continue
            parts.append(str(value) if role == "Nöbetçiler" else f"{value} ({role})")
        found[row_date] = " · ".join(parts) or "boş"
    return [(wanted[d], found[d]) for d in sorted(found)]


def shared_updated_text(moment):
    """Son güncelleme zamanı: "17 Ekim 2026, 14:30" (bilinmiyorsa boş)."""
    if not isinstance(moment, datetime):
        return ""
    return f"{moment.day} {TR_AY_ADLARI[moment.month]} {moment.year}, {moment:%H:%M}"


def _ics_escape(text):
    return (
        str(text).replace("\\", "\\\\").replace(";", "\\;")
        .replace(",", "\\,").replace("\n", "\\n")
    )


def _ics_fold(line):
    """RFC 5545: satırlar 75 baytta katlanır (devam satırı boşlukla başlar)."""
    out, current = [], b""
    for char in line:
        encoded = char.encode("utf-8")
        if len(current) + len(encoded) > (75 if not out else 74):
            out.append(current.decode("utf-8"))
            current = b""
        current += encoded
    out.append(current.decode("utf-8"))
    return "\r\n ".join(out)


def build_person_ics(person, rows, role_names, yil, ay):
    """Kişinin nöbetlerini telefon takvimine eklenecek .ics olarak üretir.

    Her nöbet tüm gün etkinliğidir; başlıkta ve açıklamada görev yeri (birim)
    ve o gün birlikte nöbet tutanlar ile onların birimleri yazar.
    """
    person_hash = hashlib.sha1(person.encode("utf-8")).hexdigest()[:10]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Adil Nobet//nobeting//TR",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        f"X-WR-CALNAME:{_ics_escape(person + ' nöbetleri')}",
    ]
    for row in rows:
        roles = [role for role in role_names if row.get(role) == person]
        if not roles:
            continue
        day, _ = _row_day_and_weekday(row)
        if not day:
            continue
        try:
            start = date(yil, ay, day)
        except ValueError:
            continue
        end = date.fromordinal(start.toordinal() + 1)
        unit = ", ".join(roles)
        partners = [
            f"{row.get(role)} ({role})" for role in role_names
            if role not in roles and row.get(role) not in (None, "", "-")
        ]
        description = f"Görev yeri: {unit}"
        if partners:
            description += "\nBirlikte: " + ", ".join(partners)
        lines += [
            "BEGIN:VEVENT",
            f"UID:nobet-{start:%Y%m%d}-{person_hash}@nobeting",
            f"DTSTAMP:{stamp}",
            f"DTSTART;VALUE=DATE:{start:%Y%m%d}",
            f"DTEND;VALUE=DATE:{end:%Y%m%d}",
            f"SUMMARY:{_ics_escape('Nöbet — ' + unit)}",
            f"DESCRIPTION:{_ics_escape(description)}",
            "TRANSP:OPAQUE",
            # Tüm gün etkinliği gece yarısı başlar; -6 saat = önceki gün 18:00.
            "BEGIN:VALARM",
            "ACTION:DISPLAY",
            f"DESCRIPTION:{_ics_escape('Yarın nöbetin var — ' + unit)}",
            "TRIGGER:-PT6H",
            "END:VALARM",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(_ics_fold(line) for line in lines) + "\r\n"


def add_overflow_roles(rows, role_names, schedule, columns):
    """Görev sayısından fazla atanan kişileri "Ek 1", "Ek 2"… sütunlarına ekler.

    Bir güne görev sayısından fazla kişi atanınca (ya da nöbet başına kişi
    sayısı sonradan düşürülünce) fazla kişi Günlük Liste, WhatsApp, Excel,
    paylaşım bağlantısı ve takvim dosyasında hiç görünmüyordu.
    Dönüş: (satırlar, görevler); fazla kişi yoksa girdiler aynen döner.
    """
    role_names = list(role_names)
    extras = []
    for index, column in enumerate(columns):
        row = rows[index] if index < len(rows) else {}
        placed = {row.get(role) for role in role_names}
        assigned = (
            schedule.index[schedule[column].astype(bool)].tolist()
            if column in schedule.columns else []
        )
        extras.append([person for person in assigned if person not in placed])
    most = max((len(extra) for extra in extras), default=0)
    if not most:
        return rows, role_names
    used = {name.casefold() for name in role_names} | {"tarih"}
    extra_roles = []
    number = 1
    while len(extra_roles) < most:
        name = f"Ek {number}"
        number += 1
        if name.casefold() not in used:
            extra_roles.append(name)
    out = []
    for index, row in enumerate(rows):
        new_row = dict(row)
        extra = extras[index] if index < len(extras) else []
        for position, role in enumerate(extra_roles):
            new_row[role] = extra[position] if position < len(extra) else "-"
        out.append(new_row)
    return out, role_names + extra_roles


def create_print_html(df_liste, df_stats_load, yil, ay):
    """Yazdırma dostu HTML oluşturur (günlük liste + nöbet yükü özeti)."""
    ay_isimleri = {1:"Ocak", 2:"Şubat", 3:"Mart", 4:"Nisan", 5:"Mayıs", 6:"Haziran",
                   7:"Temmuz", 8:"Ağustos", 9:"Eylül", 10:"Ekim", 11:"Kasım", 12:"Aralık"}
    month_name = escape_html(ay_isimleri[ay])
    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Nöbet Listesi - {month_name} {yil}</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            h1 {{ text-align: center; color: #333; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .weekend {{ background-color: #e3f2fd !important; }}
            @media print {{
                body {{ margin: 0; }}
                button {{ display: none; }}
            }}
        </style>
    </head>
    <body>
        <h1>Nöbet Listesi - {month_name} {yil}</h1>
        <table>
            <tr>{''.join(f'<th>{escape_html(col)}</th>' for col in df_liste.columns)}</tr>
    """

    for _, row in df_liste.iterrows():
        tarih = str(row.get('Tarih', ''))
        css_class = 'weekend' if 'Cmt' in tarih or 'Paz' in tarih else ''
        html += f"<tr class='{css_class}'>{''.join(f'<td>{escape_html(val)}</td>' for val in row)}</tr>"

    html += """
        </table>
        <h2>Nöbet Yükü Özeti</h2>
        <table>
            <tr><th>İsim</th>""" + ''.join(f'<th>{escape_html(col)}</th>' for col in df_stats_load.columns) + "</tr>"

    for idx, row in df_stats_load.iterrows():
        html += f"<tr><td><strong>{escape_html(idx)}</strong></td>{''.join(f'<td>{escape_html(val)}</td>' for val in row)}</tr>"

    html += """
        </table>
        <p style="text-align: center; color: #666; margin-top: 30px;">
            Oluşturulma: """ + now_tr().strftime("%d.%m.%Y %H:%M") + """
        </p>
    </body>
    </html>
    """
    return html


# --- YEDEK (tüm veriyi indirme ve geri yükleme) ---
BACKUP_FORMAT = "nobeting-yedek"
BACKUP_VERSION = 1
BACKUP_MAX_BYTES = 20 * 1024 * 1024
_BACKUP_MAX_SCHEDULES = 1200
_BACKUP_MAX_PEOPLE = 300
_BACKUP_MAX_TEXT = 255
_BACKUP_MAX_CELLS = 500_000


def _parse_json_text(value):
    if value in (None, ""):
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def build_backup(settings, schedules, created_at=None):
    """Veritabanı satırlarından indirilecek yedek sözlüğünü kurar.

    ``schedules``: db.export_owner_data'nın döndürdüğü kayıtlar
    (name, year, month, team_members, pref_json, settings_json,
    rows_liste_json, cells=[(kişi, gün, atandı)]).
    """
    items = []
    for row in schedules or ():
        team = _parse_json_text(row.get("team_members"))
        if not isinstance(team, list):
            team = [m.strip() for m in str(row.get("team_members") or "").split(",") if m.strip()]
        # Uygulama da çizelgeyi ekip listesine göre okur; listede olmayanın
        # hücreleri görünmez, yedeğe de girmez (yoksa geri yükleme reddederdi).
        members = set(str(member) for member in team)
        cells = [cell for cell in (row.get("cells") or []) if str(cell[0]) in members]
        items.append({
            "name": row["name"],
            "year": int(row["year"]),
            "month": int(row["month"]),
            "team": [str(member) for member in team],
            "days": list(dict.fromkeys(str(day) for _, day, _ in cells)),
            "assigned": [[str(person), str(day)] for person, day, assigned in cells if assigned],
            "preferences": _parse_json_text(row.get("pref_json")),
            "settings": _parse_json_text(row.get("settings_json")),
            "rows": _parse_json_text(row.get("rows_liste_json")),
        })
    items.sort(key=lambda item: (item["year"], item["month"], item["name"]))
    return {
        "format": BACKUP_FORMAT,
        "version": BACKUP_VERSION,
        "created_at": (created_at or now_tr()).strftime("%Y-%m-%d %H:%M"),
        "settings": settings if isinstance(settings, dict) else None,
        "schedules": items,
    }


def backup_to_bytes(backup):
    return json.dumps(backup, ensure_ascii=False, indent=1).encode("utf-8")


def _short_text(value, limit=_BACKUP_MAX_TEXT):
    return isinstance(value, str) and 0 < len(value) <= limit


def parse_backup(raw):
    """Yüklenen yedek dosyasını doğrular.

    Dönen değer (yedek, hatalar). Hata varsa yedek None'dır; dosya hiçbir
    şeye yazılmadan reddedilir.
    """
    if raw is None:
        return None, ["Dosya okunamadı."]
    if len(raw) > BACKUP_MAX_BYTES:
        return None, ["Dosya çok büyük (en fazla 20 MB)."]
    try:
        data = json.loads(raw.decode("utf-8-sig") if isinstance(raw, (bytes, bytearray)) else raw)
    except (UnicodeDecodeError, ValueError):
        return None, ["Dosya bir Nöbet yedeği değil (JSON okunamadı)."]
    if not isinstance(data, dict) or data.get("format") != BACKUP_FORMAT:
        return None, ["Dosya bir Nöbet yedeği değil."]
    if data.get("version") != BACKUP_VERSION:
        return None, ["Yedek dosyasının sürümü desteklenmiyor."]
    settings = data.get("settings")
    if settings is not None and not isinstance(settings, dict):
        return None, ["Yedekteki ayarlar bozuk."]
    schedules = data.get("schedules")
    if not isinstance(schedules, list) or len(schedules) > _BACKUP_MAX_SCHEDULES:
        return None, ["Yedekteki çizelge listesi bozuk."]

    errors = []
    clean = []
    seen = set()
    skipped = 0
    for index, item in enumerate(schedules, start=1):
        label = f"{index}. kayıt"
        if not isinstance(item, dict):
            errors.append(f"{label}: bozuk.")
            continue
        name, year, month = item.get("name"), item.get("year"), item.get("month")
        if not _short_text(name):
            errors.append(f"{label}: adı geçersiz.")
            continue
        if (isinstance(year, bool) or not isinstance(year, int) or not YEAR_MIN <= year <= YEAR_MAX
                or isinstance(month, bool) or not isinstance(month, int) or not 1 <= month <= 12):
            errors.append(f"{label} ({name}): yıl/ay geçersiz.")
            continue
        # Uygulama yalnızca "Otomatik_YYYY_AA" kayıtlarını kullanır; diğerleri
        # (eski elle kayıtlar) görünmeden yer kaplardı, geri yüklenmez.
        if name != f"Otomatik_{year}_{month:02d}":
            skipped += 1
            continue
        key = (name, year, month)
        if key in seen:
            errors.append(f"{label} ({name} {month}/{year}): iki kez var.")
            continue
        seen.add(key)
        team, days, assigned = item.get("team"), item.get("days"), item.get("assigned")
        if (not isinstance(team, list) or len(team) > _BACKUP_MAX_PEOPLE
                or not all(_short_text(person) for person in team)):
            errors.append(f"{label} ({name} {month}/{year}): ekip listesi geçersiz.")
            continue
        if not isinstance(days, list) or len(days) > 31 or not all(_short_text(day, 50) for day in days):
            errors.append(f"{label} ({name} {month}/{year}): gün listesi geçersiz.")
            continue
        if (not isinstance(assigned, list)
                or not all(isinstance(cell, list) and len(cell) == 2 for cell in assigned)
                or not all(cell[0] in team and cell[1] in days for cell in assigned)):
            errors.append(f"{label} ({name} {month}/{year}): nöbetler geçersiz.")
            continue
        preferences, sched_settings, rows = item.get("preferences"), item.get("settings"), item.get("rows")
        if preferences is not None and not _valid_backup_preferences(preferences):
            errors.append(f"{label} ({name} {month}/{year}): tercihler geçersiz.")
            continue
        if sched_settings is not None and not isinstance(sched_settings, dict):
            errors.append(f"{label} ({name} {month}/{year}): ay ayarları geçersiz.")
            continue
        if rows is not None and not _valid_backup_rows(rows):
            errors.append(f"{label} ({name} {month}/{year}): görev listesi geçersiz.")
            continue
        clean.append({
            "name": name, "year": year, "month": month,
            "team": list(dict.fromkeys(team)), "days": list(dict.fromkeys(days)),
            "assigned": [tuple(cell) for cell in assigned],
            "preferences": preferences, "settings": sched_settings, "rows": rows,
        })
    if not errors and sum(len(item["team"]) * len(item["days"]) for item in clean) > _BACKUP_MAX_CELLS:
        errors.append("Yedek çok büyük (çok fazla kişi/gün).")
    if errors:
        return None, errors[:10]
    return {
        "created_at": str(data.get("created_at") or "")[:40],
        "settings": settings,
        "schedules": clean,
        "skipped": skipped,
    }, []


def _valid_backup_preferences(preferences):
    """Tercihler {gün: {kişi: 0-3}} biçiminde olmalı (pandas to_json çıktısı)."""
    if not isinstance(preferences, dict) or len(preferences) > 31:
        return False
    for day, column in preferences.items():
        if not _short_text(day, 50) or not isinstance(column, dict) or len(column) > _BACKUP_MAX_PEOPLE:
            return False
        for person, value in column.items():
            if not _short_text(person) or isinstance(value, bool):
                return False
            if value is not None and (not isinstance(value, (int, float)) or not 0 <= value <= 3):
                return False
    return True


def _valid_backup_rows(rows):
    """Görev listesi: en fazla 31 satır, her satır kısa metin değerli sözlük."""
    if not isinstance(rows, list) or len(rows) > 31:
        return False
    for row in rows:
        if not isinstance(row, dict) or len(row) > 60:
            return False
        for key, value in row.items():
            if not _short_text(key):
                return False
            if value is not None and not isinstance(value, (str, int, float)):
                return False
            if isinstance(value, str) and len(value) > _BACKUP_MAX_TEXT:
                return False
    return True


def backup_summary(backup):
    """"14 ay (Ocak 2026 – Şubat 2027), ekip ve ayarlar" gibi kısa açıklama."""
    schedules = backup.get("schedules") or []
    parts = []
    if schedules:
        periods = sorted({(item["year"], item["month"]) for item in schedules})
        first, last = periods[0], periods[-1]
        span = f"{TR_AY_ADLARI[first[1]]} {first[0]}"
        if last != first:
            span += f" – {TR_AY_ADLARI[last[1]]} {last[0]}"
        parts.append(f"{len(periods)} ay ({span})")
    if backup.get("settings"):
        parts.append("ekip ve ayarlar")
    return ", ".join(parts) if parts else "boş yedek"


# Uygulamada saklanan yedekler: her ay için Yedekle tuşuyla alınan tek yedek
# ve son geri yüklemeden önceki hâlin tek kopyası (year=0, tüm veri).
SNAPSHOT_MANUAL = "Yedekle"
SNAPSHOT_BEFORE_RESTORE = "Geri yükleme öncesi"


def snapshot_time_text(item):
    """"2026-10-16 12:05" -> "16.10.2026 12:05"."""
    text = str(item.get("created_tr", ""))
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M").strftime("%d.%m.%Y %H:%M")
    except ValueError:
        return text


def snapshot_label(item):
    """Listede görünen satır: "Mart 2026 · 10.03.2026 14:20 yedeklendi"."""
    try:
        period = f"{TR_AY_ADLARI[int(item.get('month'))]} {int(item.get('year'))}"
    except (TypeError, ValueError, IndexError, KeyError):
        period = "?"
    return f"{period} · {snapshot_time_text(item)} yedeklendi"


def month_backup(backup, year, month):
    """Tüm veriden yalnızca o ayın otomatik kaydını içeren yedek (ayarlar hariç)."""
    name = f"Otomatik_{year}_{month:02d}"
    return {
        **backup,
        "settings": None,
        "schedules": [item for item in backup.get("schedules") or [] if item.get("name") == name],
    }


def backup_reminder(last_backup_text, today=None, days=30):
    """Son yedek eskiyse ya da hiç yoksa hatırlatma metni, değilse None."""
    today = today or now_tr().date()
    try:
        last = datetime.strptime(str(last_backup_text or "")[:10], "%Y-%m-%d").date()
    except ValueError:
        return "Henüz yedek indirilmedi."
    age = (today - last).days
    if age >= days:
        return f"Son yedek {age} gün önce indirildi."
    return None
