"""İyileştir düğmesinin çözücüsü (nobet_iyilestir): kurallar korunur, öncelikli dengeler kötüleşmez."""

import random

import pandas as pd
import pytest

pytest.importorskip("ortools")

import nobet_core as n  # noqa: E402
from nobet_iyilestir import (  # noqa: E402
    PRIMARY_METRICS,
    improve_schedule,
    improvement_lines,
    rule_breaking_shifts,
    schedule_metrics,
    weighted_score,
)


def _month(yil=2026, ay=11, team=8):
    days = 30 if ay == 11 else 31
    hol, _ = n.parse_holiday_days(n.default_holiday_text(yil, ay), days)
    gd = n.build_gun_detaylari(yil, ay, days, hol)
    cols = list(gd)
    names = [f"K{i}" for i in range(team)]
    return gd, cols, names


def _check_rules(schedule, names, cols, gd, unwanted, gap, k, pairs, limits, fixed):
    day = {c: gd[c]["day_num"] for c in cols}
    for p, c in fixed:
        assert schedule.at[p, c]
    for p in names:
        mine = sorted(day[c] for c in cols if schedule.at[p, c])
        assert all(b - a > gap for a, b in zip(mine, mine[1:])), p
        for c in cols:
            if schedule.at[p, c] and (p, c) not in fixed:
                assert not unwanted.at[p, c]
        if p in limits:
            assert len(mine) <= limits[p]["max"]
    for c in cols:
        assert schedule[c].sum() <= k
        for a, b in pairs:
            assert not (schedule.at[a, c] and schedule.at[b, c])


def test_kurallar_korunur_ve_oncelikli_dengeler_kotulesmez():
    gd, cols, names = _month()
    rng = random.Random(3)
    unwanted = pd.DataFrame(False, index=names, columns=cols)
    prefs = pd.DataFrame(0, index=names, columns=cols)
    for p in names:
        for c in rng.sample(cols, 3):
            unwanted.at[p, c] = True
        for c in rng.sample(cols, 3):
            if not unwanted.at[p, c]:
                prefs.at[p, c] = rng.choice((1, 2))
    pairs = {("K0", "K1")}
    limits = {"K2": {"min": 0, "max": 5}}
    fixed = {("K3", cols[4])}
    unwanted.at["K3", cols[4]] = False
    gap, k = 1, 2
    random.seed(3)
    current, _ = n.run_scheduling_core(names, cols, unwanted, gd, gap, k, pairs, limits, prefs,
                                       simulation_count=30, fixed_assignments=fixed)
    improved, before, after = improve_schedule(
        current, names, cols, unwanted, gd, gap, k, forbidden_pairs=pairs, person_limits=limits,
        df_preferred=prefs, fixed_assignments=fixed, time_limit=3,
    )
    assert improved is not None
    _check_rules(improved, names, cols, gd, unwanted, gap, k, pairs, limits, fixed)
    for name in PRIMARY_METRICS:
        assert after[name] <= before[name], name
    assert weighted_score(after) < weighted_score(before)
    assert after == schedule_metrics(improved, names, cols, gd, k, gap, person_limits=limits, df_preferred=prefs)


def test_onceki_ayin_son_nobeti_ve_dinlenme_korunur():
    gd, cols, names = _month(team=6)
    unwanted = pd.DataFrame(False, index=names, columns=cols)
    current = pd.DataFrame(False, index=names, columns=cols)
    # Herkes sırayla; K0 önceki ayın son günü nöbetçi (0) olduğu için 1. gün tutamaz.
    for i, c in enumerate(cols):
        current.at[names[(i + 1) % len(names)], c] = True
    improved, _, _ = improve_schedule(current, names, cols, unwanted, gd, 2, 1,
                                      initial_last_shift={"K0": 0}, time_limit=2)
    result = improved if improved is not None else current
    day = {c: gd[c]["day_num"] for c in cols}
    assert not any(result.at["K0", c] for c in cols if day[c] <= 2)
    _check_rules(result, names, cols, gd, unwanted, 2, 1, set(), {}, set())


def test_daha_iyisi_yoksa_liste_degismez():
    gd, cols, names = _month(team=1)
    unwanted = pd.DataFrame(False, index=names, columns=cols)
    current = pd.DataFrame(True, index=names, columns=cols)
    improved, before, after = improve_schedule(current, names, cols, unwanted, gd, 0, 1, time_limit=2)
    assert improved is None
    assert before == after


def test_degisiklik_metni():
    before = dict.fromkeys(PRIMARY_METRICS, 0) | {"consecutive_weekends": 3, "avoid": 0,
                                                  "missed_preference": 2, "spread": 5}
    after = dict(before, friday=0, consecutive_weekends=0, missed_preference=1)
    before = dict(before, friday=2)
    assert improvement_lines(before, after) == [
        "Cuma farkı 2 → 0", "art arda hafta sonu nöbeti 3 → 0", "karşılanmayan tercih 2 → 1",
    ]
    assert improvement_lines(after, dict(after, spread=1)) == ["nöbetler aya daha düzenli yayıldı"]


@pytest.mark.parametrize("max_limit", [6, 9])
def test_limitli_kisinin_nobet_sayisi_degismez(max_limit):
    # Eskiden limitli kişi toplam dengesinde sayılmadığı için 4'e düşüyor ya da
    # 11'e çıkıyordu (toplam farkı 1 → 4).
    gd, cols, names = _month()
    rng = random.Random(13)
    unwanted = pd.DataFrame(False, index=names, columns=cols)
    prefs = pd.DataFrame(0, index=names, columns=cols)
    for p in names:
        for c in rng.sample(cols, 4):
            unwanted.at[p, c] = True
        for c in rng.sample(cols, 4):
            if not unwanted.at[p, c]:
                prefs.at[p, c] = rng.choice((1, 2))
    limits = {"K0": {"min": 0, "max": max_limit}}
    random.seed(13)
    current, _ = n.run_scheduling_core(names, cols, unwanted, gd, 1, 2, person_limits=limits,
                                       df_preferred=prefs, simulation_count=10)
    improved, _, _ = improve_schedule(current, names, cols, unwanted, gd, 1, 2, person_limits=limits,
                                      df_preferred=prefs, time_limit=3)
    result = improved if improved is not None else current
    assert result.loc["K0"].sum() == current.loc["K0"].sum()
    free = result.drop(index="K0").sum(axis=1)
    assert free.max() - free.min() <= 1


def test_kurala_uymayan_nobet_sayilir():
    gd, cols, names = _month(team=4)
    unwanted = pd.DataFrame(False, index=names, columns=cols)
    current = pd.DataFrame(False, index=names, columns=cols)
    current.at["K0", cols[0]] = True
    current.at["K0", cols[1]] = True          # dinlenme kuralı (1 gün)
    current.at["K1", cols[5]] = True
    unwanted.at["K1", cols[5]] = True          # izin günü
    current.at["K2", cols[9]] = True
    current.at["K3", cols[9]] = True           # birlikte çalışamaz
    assert rule_breaking_shifts(current, names, cols, unwanted, gd, 1,
                                forbidden_pairs={("K2", "K3")}) == 5
    # Kilitli nöbet sayılmaz; önceki ayın son nöbeti de dinlenmeye katılır.
    assert rule_breaking_shifts(current, names, cols, unwanted, gd, 1, forbidden_pairs={("K2", "K3")},
                                fixed_assignments={("K1", cols[5])}) == 4
    clean = pd.DataFrame(False, index=names, columns=cols)
    clean.at["K0", cols[0]] = True
    assert rule_breaking_shifts(clean, names, cols, unwanted, gd, 1) == 0
    assert rule_breaking_shifts(clean, names, cols, unwanted, gd, 1, initial_last_shift={"K0": 0}) == 1
