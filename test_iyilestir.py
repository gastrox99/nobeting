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
    before = dict.fromkeys(PRIMARY_METRICS, 0) | {"consecutive_weekends": 3, "alt_chain": 0, "avoid": 0, "pair_repeat": 0, "long_gap": 0,
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


def test_kesirli_gecmis_farki_cozucuyu_bozmaz():
    # Canlıda "Önceki aylarla dengele" açıkken 9 kişide geçmiş farkı 2/9 gibi
    # kesirli çıkıyordu ve çözücü "Failed to convert integer linear expression" veriyordu.
    gd, cols, names = _month(team=9)
    unwanted = pd.DataFrame(False, index=names, columns=cols)
    history_special = n.history_special_adjustment({p: (3 if p == "K0" else 2) for p in names}, names)
    history_total = n.history_adjustment([{p: (7 if p in ("K0", "K1") else 6) for p in names}], names)
    history_holiday = n.history_adjustment([{p: (1 if p == "K2" else 0) for p in names}], names)
    assert any(v != int(v) for v in history_special.values())
    random.seed(1)
    current, _ = n.run_scheduling_core(names, cols, unwanted, gd, 1, 2, simulation_count=10)
    improved, before, after = improve_schedule(
        current, names, cols, unwanted, gd, 1, 2, history_special=history_special,
        history_total=history_total, history_holiday=history_holiday, time_limit=3,
    )
    result = improved if improved is not None else current
    _check_rules(result, names, cols, gd, unwanted, 1, 2, set(), {}, set())
    for name in PRIMARY_METRICS:
        assert after[name] <= before[name], name


def _special_count(schedule, person, cols, gd):
    return sum(1 for c in cols if schedule.at[person, c] and (gd[c]["weekend"] or gd[c]["holiday"]))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_hafta_sonu_siniri_simulasyon_ve_iyilestirde_asilmaz(seed):
    gd, cols, names = _month(team=9)
    rng = random.Random(seed)
    unwanted = pd.DataFrame(False, index=names, columns=cols)
    for p in names:
        for c in rng.sample(cols, 3):
            unwanted.at[p, c] = True
    limits = {"K0": {"min": 0, "max": 999, "max_special": 1}, "K1": {"min": 0, "max": 999, "max_special": 0}}
    random.seed(seed)
    current, _ = n.run_scheduling_core(names, cols, unwanted, gd, 1, 2, person_limits=limits,
                                       simulation_count=20)
    assert _special_count(current, "K0", cols, gd) <= 1
    assert _special_count(current, "K1", cols, gd) == 0
    # Yalnız hafta sonu sınırı olan kişi toplam dengesinde kalır.
    totals = current.sum(axis=1)
    assert totals.max() - totals.min() <= 1
    assert current.sum().min() == 2  # boş gün yok
    improved, _, _ = improve_schedule(current, names, cols, unwanted, gd, 1, 2, person_limits=limits,
                                      time_limit=3)
    result = improved if improved is not None else current
    assert _special_count(result, "K0", cols, gd) <= 1
    assert _special_count(result, "K1", cols, gd) == 0
    others = [_special_count(result, p, cols, gd) for p in names if p not in limits]
    assert max(others) - min(others) <= 1


def test_hafta_sonu_siniri_uyari_ve_neden():
    gd, cols, names = _month(team=3)
    weekend = [c for c in cols if gd[c]["weekend"]]
    schedule = pd.DataFrame(False, index=names, columns=cols)
    schedule.at["K0", weekend[0]] = True
    schedule.at["K0", weekend[2]] = True
    limits = {"K0": {"min": 0, "max": 999, "max_special": 1}}
    messages = n.build_limit_violation_messages(schedule, names, limits, weekend)
    assert messages and "hafta sonu" in messages[0]
    warnings = n.assignment_warnings(schedule, "K0", weekend[2], cols, gd, 0, limits)
    assert any("hafta sonu" in w for w in warnings)
    schedule.at["K0", weekend[2]] = False
    reasons = dict(n.explain_missing_day(schedule, weekend[4], gd,
                                         pd.DataFrame(False, index=names, columns=cols), 0, limits))
    assert any("K0" in r for r in reasons.get("nöbet limitinde", []))


def _fixed_pairs_schedule(names, cols, k=2):
    """Her gün aynı ikililer sırayla: (K0,K1), (K2,K3), ... — eski davranışın abartısı."""
    schedule = pd.DataFrame(False, index=names, columns=cols)
    groups = [names[i:i + k] for i in range(0, len(names), k)]
    for i, c in enumerate(cols):
        for p in groups[i % len(groups)]:
            schedule.at[p, c] = True
    return schedule


def test_ayni_ikili_olcusu():
    gd, cols, names = _month(team=10)
    assert n.pair_repeat_allowed(10, 30, 2) == 1
    assert n.pair_repeat_allowed(8, 30, 2) == 2
    # 5 ikili × 6 ortak nöbet; her biri 1'i aşan 5 fazla → 25.
    assert n.pair_repeat_excess(_fixed_pairs_schedule(names, cols), names, cols, 2) == 25
    assert n.pair_repeat_excess(_fixed_pairs_schedule(names, cols), names, cols, 1) == 0


def test_iyilestir_ayni_ikiliyi_dagitir():
    gd, cols, names = _month(team=10)
    unwanted = pd.DataFrame(False, index=names, columns=cols)
    current = _fixed_pairs_schedule(names, cols)
    improved, before, after = improve_schedule(current, names, cols, unwanted, gd, 1, 2, time_limit=5)
    assert improved is not None
    assert before["pair_repeat"] == 25
    assert after["pair_repeat"] < 5
    assert all(after[m] <= before[m] for m in PRIMARY_METRICS)
    assert "aynı ikilinin fazladan birlikte nöbeti 25 → " in " ".join(improvement_lines(before, after))


def _clumped_schedule(names, cols):
    # K0 ve K1 ayın ilk 11 gününde 6 nöbeti bitirir, sonra 19 gün boş kalır.
    schedule = pd.DataFrame(False, index=names, columns=cols)
    rest = []
    for i, c in enumerate(cols):
        if i % 2 == 0 and i <= 10:
            schedule.loc[["K0", "K1"], c] = True
        else:
            rest.append(c)
    others = names[2:]
    for j, c in enumerate(rest):
        schedule.at[others[(2 * j) % len(others)], c] = True
        schedule.at[others[(2 * j + 1) % len(others)], c] = True
    return schedule


def test_uzun_bosluk_olcusu():
    gd, cols, names = _month(team=10)
    metrics = schedule_metrics(_clumped_schedule(names, cols), names, cols, gd, 2, 1)
    # 12-30 arası 19 boş gün; 11 günlük pencereden 9 tane, iki kişi → 18.
    assert metrics["long_gap"] == 18


def test_iyilestir_uzun_boslugu_azaltir():
    gd, cols, names = _month(team=10)
    unwanted = pd.DataFrame(False, index=names, columns=cols)
    improved, before, after = improve_schedule(
        _clumped_schedule(names, cols), names, cols, unwanted, gd, 1, 2, time_limit=5,
    )
    assert improved is not None
    assert before["long_gap"] == 18
    assert after["long_gap"] < 3
    assert all(after[m] <= before[m] for m in PRIMARY_METRICS)
    assert "uzun nöbetsiz dönem (gün) 18 → " in " ".join(improvement_lines(before, after))


def _alternate_day_schedule(names, cols):
    """K0-K1 1,3,5,7. günler (4'lü gün aşırı zincir), K2-K3 9,11,13 (3'lü); kalan günler sırayla."""
    schedule = pd.DataFrame(False, index=names, columns=cols)
    for d in (1, 3, 5, 7):
        schedule.loc[["K0", "K1"], cols[d - 1]] = True
    for d in (9, 11, 13):
        schedule.loc[["K2", "K3"], cols[d - 1]] = True
    others = names[4:]
    j = 0
    for d in range(1, len(cols) + 1):
        if d in (1, 3, 5, 7, 9, 11, 13):
            continue
        for _ in range(2):
            schedule.at[others[j % len(others)], cols[d - 1]] = True
            j += 1
    return schedule


def test_gun_asiri_zincir_olcusu():
    gd, cols, names = _month(team=10)
    schedule = _alternate_day_schedule(names, cols)
    # 4'lü zincir 2, 3'lü zincir 1 sayılır: K0, K1 2+2; K2, K3 1+1.
    assert n.alternate_day_chains(schedule, ["K0", "K1", "K2", "K3"], cols) == 6
    assert schedule_metrics(schedule, ["K0", "K1", "K2", "K3"], cols, gd, 2, 1)["alt_chain"] == 6
    two = pd.DataFrame(False, index=["K0"], columns=cols)
    two.loc["K0", [cols[0], cols[2], cols[6], cols[8]]] = True
    assert n.alternate_day_chains(two, ["K0"], cols) == 0


def test_iyilestir_gun_asiri_zinciri_kirar():
    gd, cols, names = _month(team=10)
    unwanted = pd.DataFrame(False, index=names, columns=cols)
    current = _alternate_day_schedule(names, cols)
    improved, before, after = improve_schedule(current, names, cols, unwanted, gd, 1, 2, time_limit=5)
    assert improved is not None
    assert before["alt_chain"] >= 6
    assert after["alt_chain"] == 0
    assert all(after[m] <= before[m] for m in PRIMARY_METRICS)
    assert "üst üste gün aşırı nöbet" in " ".join(improvement_lines(before, after))


def test_cozucu_cekirdek_bilgisini_loga_yazar(capsys):
    gd, cols, names = _month(team=4)
    unwanted = pd.DataFrame(False, index=names, columns=cols)
    random.seed(1)
    current, _ = n.run_scheduling_core(names, cols, unwanted, gd, 1, 2, set(), {}, None, simulation_count=5)
    improve_schedule(current, names, cols, unwanted, gd, 1, 2, time_limit=1)
    out = capsys.readouterr().out
    assert "İyileştir çekirdek:" in out
    assert "cpu_count=" in out and "affinity=" in out and "cgroup=" in out and "num_workers=" in out
