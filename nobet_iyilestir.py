# -*- coding: utf-8 -*-
"""İyileştir: mevcut çizelgeyi OR-Tools (CP-SAT) ile daha adil hâle getirir.

Simülasyonun yerine geçmez; onun bulduğu listeden başlar. Kurallar
(izin, dinlenme, günlük kişi sayısı, yasak ikili, en fazla nöbet, en fazla
hafta sonu, kilitli nöbetler) aynen uygulanır. Öncelikli dengeler (boş nöbet, toplam, hafta
sonu/tatil, tatil, Cuma) mevcut listeden kötü olamaz; kişisel limiti
olanların nöbet sayısı değişmez. Çözücü bunların
içinde tercihleri, art arda hafta sonlarını ve nöbet aralıklarını da
iyileştirir. ortools yalnızca düğmeye basılınca yüklenir.
"""
import os

import pandas as pd

from nobet_core import (
    alternate_day_chains, has_total_limit, is_friday_column, pair_repeat_allowed, pair_repeat_excess, special_cap, spread_targets,
)

# Önem sırası (kullanıcı: hafta sonu/tatil > Cuma; toplam farkı en fazla 1 kuralı).
PRIMARY_METRICS = ("unfilled", "min_short", "total", "special", "holiday", "friday", "total_history")
SECONDARY_METRICS = ("consecutive_weekends", "alt_chain", "avoid", "pair_repeat", "missed_preference", "long_gap", "spread")
METRIC_LABELS = {
    "unfilled": "boş nöbet",
    "total": "toplam nöbet farkı",
    "special": "hafta sonu/tatil farkı",
    "holiday": "tatil farkı",
    "friday": "Cuma farkı",
    "total_history": "önceki aylarla toplam farkı",
    "consecutive_weekends": "art arda hafta sonu nöbeti",
    "alt_chain": "üst üste gün aşırı nöbet",
    "avoid": "kaçınılan güne düşen nöbet",
    "pair_repeat": "aynı ikilinin fazladan birlikte nöbeti",
    "missed_preference": "karşılanmayan tercih",
    "long_gap": "uzun nöbetsiz dönem (gün)",
}
WEIGHTS = {
    "unfilled": 10_000_000,
    "min_short": 1_000_000,
    "total": 200_000,
    "special": 100_000,
    "holiday": 20_000,
    "friday": 5_000,
    "total_history": 1_000,
    "consecutive_weekends": 60,
    "alt_chain": 50,
    "avoid": 40,
    "pair_repeat": 30,
    "missed_preference": 20,
    "long_gap": 15,
    "spread": 1,
}
# Geçmiş aylardan gelen farklar kesirli olabilir (ör. 2/9); çözücü yalnız tam
# sayı kabul ettiği için bu ölçüler 100 ile çarpılıp kuruşa yuvarlanır.
SCALE = 100
SCALED_METRICS = ("special", "holiday", "total_history")


def _history(values):
    """Geçmiş farklarını iki basamağa yuvarlar (çözücü ve ölçüler aynı sayıyı görsün)."""
    return {p: round(float(v), 2) for p, v in (values or {}).items()}


def _is_special(info):
    return bool(info["weekend"] or info["holiday"])


def _pref(df_preferred, person, column):
    if df_preferred is None or person not in df_preferred.index or column not in df_preferred.columns:
        return 0
    try:
        return int(df_preferred.at[person, column])
    except (TypeError, ValueError):
        return 0


def _spread_pairs(people, columns, gun_detaylari, min_bosluk, kisi_sayisi):
    """Aya yayma cezası: ideal aralıktan yakın iki nöbet → (kişi, sütun1, sütun2, ceza)."""
    ideal = len(people) / max(1, kisi_sayisi)
    days = {c: gun_detaylari[c]["day_num"] for c in columns}
    pairs = []
    for i, first in enumerate(columns):
        for second in columns[i + 1:]:
            distance = days[second] - days[first]
            if min_bosluk < distance < ideal:
                weight = int(round(ideal - distance))
                if weight > 0:
                    pairs.append((first, second, weight))
    return pairs


def _long_gap_windows(people, columns, gun_detaylari, kisi_sayisi, person_limits=None):
    """Uzun boşluk ölçüsü: kişi başı, beklenen aralığın 2 katı uzunluğundaki
    ardışık gün pencereleri → (kişi, [sütunlar]). Hiç nöbet düşmeyen her
    pencere 1 puandır; boşluk pencereden uzadıkça her gün 1 puan artar."""
    gaps, _ = spread_targets(people, len(columns), kisi_sayisi, person_limits)
    ordered = sorted(columns, key=lambda c: gun_detaylari[c]["day_num"])
    windows = []
    for p in people:
        width = int(round(2 * gaps[p])) + 1
        for start in range(0, len(ordered) - width + 1):
            windows.append((p, ordered[start:start + width]))
    return windows


def schedule_metrics(schedule, people, columns, gun_detaylari, kisi_sayisi, min_bosluk,
                     person_limits=None, df_preferred=None, history_special=None,
                     history_total=None, history_holiday=None):
    """Çizelgenin adalet ölçüleri (hepsinde az = iyi). İyileştir öncesi/sonrası karşılaştırılır."""
    person_limits = person_limits or {}
    history_special = history_special or {}
    history_total = history_total or {}
    history_holiday = history_holiday or {}

    def assigned(person, column):
        return bool(person in schedule.index and column in schedule.columns and schedule.at[person, column])

    def spread(values):
        return round(max(values) - min(values), 2) if values else 0

    totals = {p: sum(assigned(p, c) for c in columns) for p in people}
    free = [p for p in people if not has_total_limit(person_limits.get(p, {}))]
    # Hafta sonu sınırı olanlar hafta sonu dengesine katılmaz (simülasyondaki gibi).
    sp_people = [p for p in people if special_cap(person_limits, p) >= 999] or list(people)
    has_holiday = any(gun_detaylari[c]["holiday"] for c in columns)
    has_friday = any(is_friday_column(gun_detaylari[c]) for c in columns)
    weeks = sorted({gun_detaylari[c]["week"] for c in columns if gun_detaylari[c]["weekend"]})
    consecutive = 0
    for p in people:
        worked = {w for w in weeks if any(
            assigned(p, c) for c in columns if gun_detaylari[c]["weekend"] and gun_detaylari[c]["week"] == w
        )}
        consecutive += sum(1 for w in worked if w + 1 in worked)
    pairs = _spread_pairs(people, columns, gun_detaylari, min_bosluk, kisi_sayisi)
    windows = _long_gap_windows(people, columns, gun_detaylari, kisi_sayisi, person_limits)
    return {
        "unfilled": sum(max(0, kisi_sayisi - sum(assigned(p, c) for p in people)) for c in columns),
        "min_short": sum(
            max(0, person_limits[p].get("min", 0) - totals[p]) for p in people if p in person_limits
        ),
        "total": spread([totals[p] for p in free]) if len(free) >= 2 else 0,
        "special": spread([
            sum(assigned(p, c) for c in columns if _is_special(gun_detaylari[c])) + history_special.get(p, 0)
            for p in sp_people
        ]),
        "holiday": spread([
            sum(assigned(p, c) for c in columns if gun_detaylari[c]["holiday"]) + history_holiday.get(p, 0)
            for p in people
        ]) if has_holiday else 0,
        "friday": spread([
            sum(assigned(p, c) for c in columns if is_friday_column(gun_detaylari[c])) for p in people
        ]) if has_friday else 0,
        "total_history": spread([totals[p] + history_total.get(p, 0) for p in free])
        if history_total and len(free) >= 2 else 0,
        "consecutive_weekends": consecutive,
        "avoid": sum(1 for p in people for c in columns if assigned(p, c) and _pref(df_preferred, p, c) == 2),
        "missed_preference": sum(
            1 for p in people for c in columns if not assigned(p, c) and _pref(df_preferred, p, c) == 1
        ),
        "pair_repeat": pair_repeat_excess(schedule, people, columns, kisi_sayisi),
        "alt_chain": alternate_day_chains(
            schedule, people, columns, {c: gun_detaylari[c]["day_num"] for c in columns}
        ),
        "long_gap": sum(1 for p, window in windows if not any(assigned(p, c) for c in window)),
        "spread": sum(
            weight for p in people for first, second, weight in pairs
            if assigned(p, first) and assigned(p, second)
        ),
    }


def weighted_score(metrics):
    return sum(WEIGHTS[name] * metrics[name] for name in WEIGHTS)


def improvement_lines(before, after):
    """Kullanıcıya gösterilecek değişiklikler: ["Cuma farkı 2 → 1", ...]."""
    lines = []
    for name in PRIMARY_METRICS + SECONDARY_METRICS:
        if name in METRIC_LABELS and after[name] != before[name]:
            lines.append(f"{METRIC_LABELS[name]} {before[name]:g} → {after[name]:g}")
    if not lines and after["spread"] < before["spread"]:
        lines.append("nöbetler aya daha düzenli yayıldı")
    return lines


def rule_breaking_shifts(current, people, columns, df_unwanted_bool, gun_detaylari, min_bosluk,
                         forbidden_pairs=None, fixed_assignments=None, initial_last_shift=None):
    """Kilitli olmayan, kurala uymayan nöbet sayısı (izin günü, dinlenme, yasak ikili).

    İyileştir çözüm bulamadığında nedenini söylemek için kullanılır.
    """
    fixed = set(fixed_assignments or ())
    initial = initial_last_shift or {}
    days = {c: gun_detaylari[c]["day_num"] for c in columns}

    def assigned(p, c):
        return bool(p in current.index and c in current.columns and current.at[p, c])

    bad = set()
    for p in people:
        mine = [c for c in columns if assigned(p, c)]
        for c in mine:
            if (p, c) in fixed:
                continue
            if p in df_unwanted_bool.index and c in df_unwanted_bool.columns and bool(df_unwanted_bool.at[p, c]):
                bad.add((p, c))
            if p in initial and days[c] - initial[p] <= min_bosluk:
                bad.add((p, c))
            if any(o != c and abs(days[o] - days[c]) <= min_bosluk for o in mine):
                bad.add((p, c))
    for c in columns:
        for a, b in (forbidden_pairs or ()):
            if assigned(a, c) and assigned(b, c):
                bad.update(cell for cell in ((a, c), (b, c)) if cell not in fixed)
    return len(bad)


def improve_schedule(current, people, columns, df_unwanted_bool, gun_detaylari, min_bosluk, kisi_sayisi,
                     forbidden_pairs=None, person_limits=None, df_preferred=None,
                     fixed_assignments=None, initial_last_shift=None, history_special=None,
                     history_total=None, history_holiday=None, time_limit=5.0):
    """Mevcut çizelgeden daha iyisini arar.

    Dönen değer (yeni_çizelge, önceki_ölçüler, yeni_ölçüler). Daha iyisi
    bulunamazsa yeni_çizelge None olur. Öncelikli ölçülerin hiçbiri
    mevcut listeden kötüleşmez.
    """
    from ortools.sat.python import cp_model

    person_limits = person_limits or {}
    history_special = _history(history_special)
    history_total = _history(history_total)
    history_holiday = _history(history_holiday)
    initial_last_shift = initial_last_shift or {}
    fixed = {(p, c) for p, c in (fixed_assignments or ()) if p in people and c in columns}
    metric_args = dict(
        person_limits=person_limits, df_preferred=df_preferred, history_special=history_special,
        history_total=history_total, history_holiday=history_holiday,
    )
    before = schedule_metrics(current, people, columns, gun_detaylari, kisi_sayisi, min_bosluk, **metric_args)
    days = {c: gun_detaylari[c]["day_num"] for c in columns}

    model = cp_model.CpModel()
    x = {(p, c): model.NewBoolVar(f"x_{i}_{j}") for i, p in enumerate(people) for j, c in enumerate(columns)}

    # Kurallar: simülasyonla aynı. Kilitli nöbet her zaman kalır.
    for p in people:
        for c in columns:
            if (p, c) in fixed:
                model.Add(x[p, c] == 1)
                continue
            if p in df_unwanted_bool.index and c in df_unwanted_bool.columns and bool(df_unwanted_bool.at[p, c]):
                model.Add(x[p, c] == 0)
            if p in initial_last_shift and days[c] - initial_last_shift[p] <= min_bosluk:
                model.Add(x[p, c] == 0)
        for i, first in enumerate(columns):
            for second in columns[i + 1:]:
                if days[second] - days[first] <= min_bosluk and not ((p, first) in fixed and (p, second) in fixed):
                    model.Add(x[p, first] + x[p, second] <= 1)
        sp_cap = special_cap(person_limits, p)
        if sp_cap < 999:
            locked_sp = sum(1 for c in columns if (p, c) in fixed and _is_special(gun_detaylari[c]))
            model.Add(sum(x[p, c] for c in columns if _is_special(gun_detaylari[c])) <= max(sp_cap, locked_sp))
        if has_total_limit(person_limits.get(p, {})):
            locked = sum(1 for c in columns if (p, c) in fixed)
            cap = max(person_limits[p].get("max", 999), locked)
            model.Add(sum(x[p, c] for c in columns) <= cap)
            # Limitli kişi toplam dengesinde sayılmaz; nöbet sayısı da değişmesin
            # (yoksa çözücü fazla/eksik nöbeti ona yükleyip diğerlerini dengeliyordu).
            current_total = sum(
                1 for c in columns if p in current.index and c in current.columns and bool(current.at[p, c])
            )
            if current_total <= cap:
                if current_total >= person_limits[p].get("min", 0):
                    model.Add(sum(x[p, c] for c in columns) == current_total)
                else:
                    # Alt sınırın altındaki kişinin nöbeti azalmaz; çözücü eksiği kapatabilir.
                    model.Add(sum(x[p, c] for c in columns) >= current_total)
    for c in columns:
        for a, b in (forbidden_pairs or ()):
            if a in people and b in people and not ((a, c) in fixed and (b, c) in fixed):
                model.Add(x[a, c] + x[b, c] <= 1)

    def range_of(exprs, terms_name, bound=1000):
        high = model.NewIntVar(-bound, bound, f"hi_{terms_name}")
        low = model.NewIntVar(-bound, bound, f"lo_{terms_name}")
        model.AddMaxEquality(high, exprs)
        model.AddMinEquality(low, exprs)
        value = model.NewIntVar(0, 2 * bound, terms_name)
        model.Add(value == high - low)
        return value

    expr = {}
    unfilled = []
    for c in columns:
        locked = sum(1 for p in people if (p, c) in fixed)
        gap = model.NewIntVar(0, kisi_sayisi, f"u_{days[c]}")
        model.Add(sum(x[p, c] for p in people) <= max(kisi_sayisi, locked))
        model.Add(sum(x[p, c] for p in people) + gap >= kisi_sayisi)
        unfilled.append(gap)
    expr["unfilled"] = sum(unfilled)
    totals = {p: sum(x[p, c] for c in columns) for p in people}
    shorts = []
    for p, limits in person_limits.items():
        if p in people and limits.get("min", 0) > 0:
            short = model.NewIntVar(0, limits["min"], f"short_{len(shorts)}")
            model.Add(short >= limits["min"] - totals[p])
            shorts.append(short)
    expr["min_short"] = sum(shorts)
    free = [p for p in people if not has_total_limit(person_limits.get(p, {}))]
    sp_people = [p for p in people if special_cap(person_limits, p) >= 999] or list(people)
    expr["total"] = range_of([totals[p] for p in free], "total") if len(free) >= 2 else 0

    def count(p, test):
        return sum(x[p, c] for c in columns if test(gun_detaylari[c]))

    def scaled_range(values, history, terms_name, group):
        # SCALE ile çarpılmış tam sayılar: kesirli geçmiş farkı modele girebilsin.
        return range_of(
            [SCALE * values[p] + int(round(SCALE * history.get(p, 0))) for p in group],
            terms_name, bound=1000 * SCALE,
        )

    expr["special"] = scaled_range({p: count(p, _is_special) for p in people}, history_special, "special", sp_people)
    expr["holiday"] = scaled_range(
        {p: count(p, lambda d: d["holiday"]) for p in people}, history_holiday, "holiday", people
    ) if any(gun_detaylari[c]["holiday"] for c in columns) else 0
    expr["friday"] = range_of(
        [count(p, is_friday_column) for p in people], "friday"
    ) if any(is_friday_column(gun_detaylari[c]) for c in columns) else 0
    expr["total_history"] = scaled_range(
        totals, history_total, "total_history", free
    ) if history_total and len(free) >= 2 else 0

    weeks = sorted({gun_detaylari[c]["week"] for c in columns if gun_detaylari[c]["weekend"]})
    consecutive = []
    for p in people:
        worked = {}
        for w in weeks:
            worked[w] = model.NewBoolVar("")
            model.AddMaxEquality(worked[w], [
                x[p, c] for c in columns if gun_detaylari[c]["weekend"] and gun_detaylari[c]["week"] == w
            ])
        for w in weeks:
            if w + 1 in worked:
                both = model.NewBoolVar("")
                model.Add(both >= worked[w] + worked[w + 1] - 1)
                consecutive.append(both)
    expr["consecutive_weekends"] = sum(consecutive)
    expr["avoid"] = sum(x[p, c] for p in people for c in columns if _pref(df_preferred, p, c) == 2)
    expr["missed_preference"] = sum(1 - x[p, c] for p in people for c in columns if _pref(df_preferred, p, c) == 1)
    spread_terms = []
    for p in people:
        for first, second, weight in _spread_pairs(people, columns, gun_detaylari, min_bosluk, kisi_sayisi):
            both = model.NewBoolVar("")
            model.Add(both >= x[p, first] + x[p, second] - 1)
            spread_terms.append(weight * both)
    expr["spread"] = sum(spread_terms)
    # Aynı ikili: beklenenden fazla ortak nöbet (schedule_metrics ile aynı ölçü).
    pair_terms = []
    if kisi_sayisi >= 2 and len(people) >= 2:
        allowed = pair_repeat_allowed(len(people), len(columns), kisi_sayisi)
        for i, a in enumerate(people):
            for b in people[i + 1:]:
                together = []
                for c in columns:
                    both = model.NewBoolVar("")
                    model.Add(both >= x[a, c] + x[b, c] - 1)
                    together.append(both)
                excess = model.NewIntVar(0, len(columns), "")
                model.Add(excess >= sum(together) - allowed)
                pair_terms.append(excess)
    expr["pair_repeat"] = sum(pair_terms)
    long_gap_terms = []
    for p, window in _long_gap_windows(people, columns, gun_detaylari, kisi_sayisi, person_limits):
        empty = model.NewBoolVar("")
        model.Add(sum(x[p, c] for c in window) + empty >= 1)
        long_gap_terms.append(empty)
    expr["long_gap"] = sum(long_gap_terms)
    # Gün aşırı zincir: d, d+2, d+4 üçlüsünün hepsi doluysa 1 (schedule_metrics ile aynı ölçü).
    by_day = {days[c]: c for c in columns}
    chain_terms = []
    for p in people:
        for d, c in by_day.items():
            if d + 2 in by_day and d + 4 in by_day:
                triple = model.NewBoolVar("")
                model.Add(triple >= x[p, c] + x[p, by_day[d + 2]] + x[p, by_day[d + 4]] - 2)
                chain_terms.append(triple)
    expr["alt_chain"] = sum(chain_terms)

    # Öncelikli dengeler mevcut listeden kötü olamaz.
    for name in PRIMARY_METRICS:
        limit = int(round(SCALE * before[name])) if name in SCALED_METRICS else before[name]
        model.Add(expr[name] <= limit)
    model.Minimize(sum(
        (WEIGHTS[name] // SCALE if name in SCALED_METRICS else WEIGHTS[name]) * expr[name] for name in WEIGHTS
    ))
    for p in people:
        for c in columns:
            model.AddHint(x[p, c], bool(p in current.index and c in current.columns and current.at[p, c]))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = max(1, min(4, os.cpu_count() or 1))
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, before, before

    improved = pd.DataFrame(False, index=list(current.index), columns=list(current.columns))
    for (p, c), var in x.items():
        if solver.Value(var):
            improved.at[p, c] = True
    after = schedule_metrics(improved, people, columns, gun_detaylari, kisi_sayisi, min_bosluk, **metric_args)
    worse = any(after[name] > before[name] for name in PRIMARY_METRICS)
    if worse or weighted_score(after) >= weighted_score(before):
        return None, before, before
    return improved, before, after


def improve_if_needed(schedule, people, columns, df_unwanted_bool, gun_detaylari, min_bosluk, kisi_sayisi,
                      **kwargs):
    """Simülasyon sonucunda boş nöbet ya da 1'den büyük toplam farkı varsa bir kez
    ``improve_schedule`` dener. Dönen değer: (çizelge, düzeldi_mi); çözücü yoksa,
    hata verirse ya da daha iyisini bulamazsa çizelge aynen döner."""
    metric_keys = ("person_limits", "df_preferred", "history_special", "history_total", "history_holiday")
    try:
        before = schedule_metrics(
            schedule, people, columns, gun_detaylari, kisi_sayisi, min_bosluk,
            **{key: kwargs[key] for key in metric_keys if key in kwargs},
        )
        if before["unfilled"] == 0 and before["total"] <= 1:
            return schedule, False
        improved, _, _ = improve_schedule(
            schedule, people, columns, df_unwanted_bool, gun_detaylari, min_bosluk, kisi_sayisi, **kwargs,
        )
    except Exception:
        return schedule, False
    if improved is None:
        return schedule, False
    return improved, True
