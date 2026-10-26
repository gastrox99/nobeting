# -*- coding: utf-8 -*-
"""Birlikte nöbet tutmak isteyen ikililer: ayar, simülasyon, dengeleme ve İyileştir."""
import json
import random
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import db
from nobet_core import (
    TOGETHER_WEIGHT,
    _balance_totals,
    _pick_varied_group,
    apply_app_settings,
    build_gun_detaylari,
    collect_app_settings,
    pair_repeat_excess,
    parse_together_pairs,
    rename_in_pairs_text,
    run_scheduling_core,
    together_split_count,
)


def _month(team=10, yil=2026, ay=11):
    gd = build_gun_detaylari(yil, ay, 30, [])
    cols = list(gd)
    names = [f"K{i}" for i in range(team)]
    unwanted = pd.DataFrame(False, index=names, columns=cols)
    return gd, cols, names, unwanted


def _ranges(schedule, names, cols, gd):
    totals = [int(schedule.loc[p].sum()) for p in names]
    specials = [sum(bool(schedule.at[p, c]) for c in cols if gd[c]["weekend"] or gd[c]["holiday"]) for p in names]
    return max(totals) - min(totals), max(specials) - min(specials)


def test_metin_okunur_ve_yasak_ikili_onceliklidir():
    pairs = parse_together_pairs("Ali-Ayşe, Can-Ece", ["Ali", "Ayşe", "Can", "Ece"])
    assert pairs == {("Ali", "Ayşe"), ("Can", "Ece")}
    assert parse_together_pairs("Ali-Ayşe, Can-Ece", None, {("Can", "Ece")}) == {("Ali", "Ayşe")}
    assert parse_together_pairs("", None) == set()


def test_ayar_kaydedilir_geri_yuklenir_ve_ad_degisince_tasinir():
    state = {"together_pairs_text": "Ali-Ayşe"}
    settings = collect_app_settings(state)
    assert settings["together_pairs"] == "Ali-Ayşe"
    restored = {}
    apply_app_settings(restored, json.loads(json.dumps(settings)))
    assert restored == state
    assert rename_in_pairs_text("Ali-Ayşe", ["Ali", "Ayşe"], {"Ayşe": "Ayşegül"}) == "Ali-Ayşegül"


def test_ayrik_gun_sayisi_ve_tekrar_cezasi_muafiyeti():
    cols = ["1", "2", "3", "4"]
    schedule = pd.DataFrame(False, index=["A", "B", "C"], columns=cols)
    for c in cols:
        schedule.at["A", c] = schedule.at["B", c] = True
    schedule.at["B", "4"] = False
    schedule.at["C", "4"] = True
    assert together_split_count(schedule, {("A", "B")}, cols) == 1
    assert pair_repeat_excess(schedule, ["A", "B", "C"], cols, 2) > 0
    assert pair_repeat_excess(schedule, ["A", "B", "C"], cols, 2, exempt={("B", "A")}) == 0


def test_secimde_es_one_alinir_esi_gelemeyen_geri_itilir():
    scores = {"A": 0, "C": 10, "B": 100}
    partners = {"A": {"B"}, "B": {"A"}}
    # A ilk seçilir, eşi B puanı kötü olsa da yanına gelir.
    assert _pick_varied_group(["A", "C", "B"], 2, scores, [], {}, set(), partners) == ["A", "B"]
    # B o gün gelemiyorsa A, eşi olmayan bir güne konmaz (başka aday varken).
    scores = {"A": 0, "C": 10, "D": 20}
    assert _pick_varied_group(["A", "C", "D"], 2, scores, [], {}, set(), partners) == ["C", "D"]
    assert TOGETHER_WEIGHT > 100


def test_simulasyon_ikilileri_ayni_gune_koyar_dengeyi_bozmaz():
    gd, cols, names, unwanted = _month()
    together = {("K0", "K1"), ("K2", "K3")}
    for seed in range(3):
        random.seed(seed)
        schedule, _ = run_scheduling_core(
            names, cols, unwanted, gd, 1, 2, simulation_count=30, together_pairs=together,
        )
        assert schedule.sum().sum() == 60
        assert together_split_count(schedule, together, cols) <= 2
        total_range, special_range = _ranges(schedule, names, cols, gd)
        assert total_range <= 1 and special_range <= 1
    # Karşılaştırma: ikili verilmezse aynı kişiler ayın çoğunda ayrı düşer.
    random.seed(0)
    plain, _ = run_scheduling_core(names, cols, unwanted, gd, 1, 2, simulation_count=30)
    assert together_split_count(plain, together, cols) >= 10


def test_esi_izinliyken_kurallar_ve_yasak_korunur():
    gd, cols, names, unwanted = _month(team=8)
    for c in cols[3:8]:
        unwanted.at["K0", c] = True
    random.seed(1)
    schedule, _ = run_scheduling_core(
        names, cols, unwanted, gd, 1, 2, forbidden_pairs={("K2", "K3")}, simulation_count=30,
        together_pairs={("K0", "K1"), ("K2", "K3")},
    )
    for c in cols:
        assert not (schedule.at["K2", c] and schedule.at["K3", c])
        assert not (unwanted.at["K0", c] and schedule.at["K0", c])
    assert together_split_count(schedule, {("K0", "K1")}, cols) <= 2


def test_tek_kisilik_nobette_ikili_yok_sayilir():
    gd, cols, names, unwanted = _month(team=5)
    random.seed(0)
    schedule, _ = run_scheduling_core(
        names, cols, unwanted, gd, 1, 1, simulation_count=5, together_pairs={("K0", "K1")},
    )
    assert (schedule.sum() == 1).all()


def test_gecmis_icin_toplam_aktarimi_ikiliyi_ayirmaz():
    gd, cols, names, unwanted = _month(team=4)
    cols = cols[:4]
    schedule = pd.DataFrame(False, index=names, columns=cols)
    for c in cols[:2]:
        schedule.at["K0", c] = schedule.at["K1", c] = True
    for c in cols[2:]:
        schedule.at["K2", c] = schedule.at["K3", c] = True
    schedule.at["K0", cols[2]] = True
    schedule.at["K2", cols[2]] = False
    # K0 3, K2 1 nöbet: fark 2, aktarım yapılır ama ikiliyi ayırmayan gün seçilir.
    result = _balance_totals(
        schedule.copy(), names, cols, gd, unwanted.loc[:, cols], None, 0, set(), {}, set(), {}, {},
        together_pairs={("K0", "K1")},
    )
    totals = {p: int(result.loc[p].sum()) for p in names}
    assert max(totals.values()) - min(totals.values()) <= 1
    assert together_split_count(result, {("K0", "K1")}, cols) == 0
    # Geçmiş yüzünden (bu ay fark 1) yapılacak aktarım ikiliyi ayırmaz.
    four = ["K0", "K1", "K2", "K3"]
    even = pd.DataFrame(False, index=four, columns=cols[:3])
    for c in cols[:2]:
        even.at["K0", c] = even.at["K1", c] = True
    even.at["K2", cols[2]] = even.at["K3", cols[2]] = True
    args = (four, cols[:3], gd, unwanted.loc[four, cols[:3]], None, 0, set(), {}, set(), {}, {})
    history = {"K0": 3, "K1": 3}
    # İkili yokken geçmişi fazla olanın nöbeti aktarılır; ikili varsa aktarılmaz.
    moved = _balance_totals(even.copy(), *args, history_total=history)
    assert int(moved.loc["K0"].sum()) + int(moved.loc["K1"].sum()) < 4
    kept = _balance_totals(even.copy(), *args, history_total=history, together_pairs={("K0", "K1")})
    assert kept.equals(even)


def test_iyilestir_ayrik_gunleri_azaltir():
    pytest.importorskip("ortools")
    from nobet_iyilestir import improve_schedule

    gd, cols, names, unwanted = _month(team=6)
    schedule = pd.DataFrame(False, index=names, columns=cols)
    # K0 ve K1 hiç aynı gün değil; liste kurallara uygun ve dengeli.
    order = ["K0", "K2", "K1", "K3", "K4", "K5"]
    for i, c in enumerate(cols):
        schedule.at[order[(2 * i) % 6], c] = True
        schedule.at[order[(2 * i + 1) % 6], c] = True
    together = {("K0", "K1")}
    before_split = together_split_count(schedule, together, cols)
    improved, before, after = improve_schedule(
        schedule, names, cols, unwanted, gd, 1, 2, together_pairs=together, time_limit=10,
    )
    assert improved is not None
    assert before["together_split"] == before_split
    assert after["together_split"] < before_split
    assert after["total"] <= before["total"] and after["special"] <= before["special"]


def test_ayarlarda_birlikte_tablosu_isaretlenir():
    from streamlit.testing.v1 import AppTest
    fakes = dict(
        init_db=lambda: True, get_last_error=lambda: None, is_database_configured=lambda: False, SCHEMA_VERSION="test",
        load_app_settings=lambda **k: None, save_app_settings=lambda *a, **k: True,
        load_schedule=lambda *a, **k: None, save_schedule=lambda *a, **k: True,
        load_auto_saved_assignments=lambda *a, **k: ({}, {}), list_backup_snapshots=lambda **k: [],
        get_share_link=lambda *a: None, ping_db=lambda: True,
        get_schedule_version=lambda *a, **k: None, get_settings_version=lambda *a, **k: None,
    )
    with patch.dict("os.environ", {"DATABASE_URL": ""}), patch.multiple(db, **fakes):
        at = AppTest.from_file(str(Path(__file__).with_name("nobet.py")), default_timeout=120)
        at.session_state["isimler_text"] = "Ali, Ayşe, Can"
        at.session_state["storage_init"] = {}
        at.session_state["yil_val"], at.session_state["ay_val"] = 2026, 10
        at.session_state["settings_expanded"] = True  # Ayarlar açık
        at.run()
        assert [e.value for e in at.exception] == []
        at.checkbox(key="w_birlikte::Ali::Can").check()
        at.run()
        assert [e.value for e in at.exception] == []
        assert at.session_state["together_pairs_text"] == "Ali-Can"
        assert at.session_state["forbidden_pairs_text"] == ""
        assert at.session_state["together_pairs"] == {("Ali", "Can")}
        # Aynı ikili yasak da işaretlenirse yasak geçerli olur.
        at.checkbox(key="w_yasak::Ali::Can").check()
        at.run()
        assert at.session_state["together_pairs"] == set()
