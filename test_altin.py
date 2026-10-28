# -*- coding: utf-8 -*-
"""Altın çıktı testi: aynı tohumla nöbet listesi, puan, rol dağıtımı ve ücret
dökümü birebir aynı kalmalı. Kod taşınırken ya da yeniden düzenlenirken
algoritmanın sessizce değişmediğini gösterir. İyileştir (OR-Tools) katılmaz.

Bir algoritma değişikliği bilerek sonucu değiştiriyorsa özetler o PR'da
yeniden üretilir ve nedeni PR açıklamasına yazılır.
"""
import calendar
import hashlib
import json
import random

import pandas as pd

import nobet_core as n


def _ozet(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _senaryo(team, yil, ay, tatil, gap, k, seed, forbidden=None, limits=None, together=None):
    gd = n.build_gun_detaylari(yil, ay, calendar.monthrange(yil, ay)[1], tatil)
    cols = list(gd)
    names = [f"K{i}" for i in range(team)]
    rng = random.Random(seed)
    unwanted = pd.DataFrame(False, index=names, columns=cols)
    preferred = pd.DataFrame(0, index=names, columns=cols)
    for p in names:
        for c in rng.sample(cols, 3):
            unwanted.at[p, c] = True
        for c in rng.sample(cols, 3):
            if not unwanted.at[p, c]:
                preferred.at[p, c] = rng.choice((1, 2))
    random.seed(seed)
    schedule, score = n.run_scheduling_core(
        names, cols, unwanted, gd, gap, k, forbidden or set(), limits or {}, preferred,
        simulation_count=15, together_pairs=together,
    )
    day_people = [[p for p in names if schedule.at[p, c]] for c in cols]
    rows, first = n.distribute_roles(cols, day_people, ["Acil", "Servis"], {}, rng=random.Random(seed))
    settings = {"title": next(iter(n.NOBET_UNVANLARI))}
    pay = {
        p: n.kisi_ucret_dokumu([int(gd[c]["day_num"]) for c in cols if schedule.at[p, c]],
                               yil, ay, settings, person=p)
        for p in names
    }
    assert all(v is not None for v in pay.values())
    return {
        "liste": _ozet(schedule.astype(int).to_dict()),
        "puan": _ozet(score),
        "roller": _ozet([rows, first]),
        "ucret": _ozet(pay),
    }


def test_ayni_tohumla_cikti_degismez():
    assert _senaryo(8, 2026, 11, [], 1, 2, 1) == {
        "liste": "01abf28450d8bce9",
        "puan": "a60d1ab6bf882340",
        "roller": "8fcf820f7c0f0b5e",
        "ucret": "1241e91069f60ded",
    }
    assert _senaryo(6, 2026, 2, [3], 2, 1, 7) == {
        "liste": "be006c4fdbd6ff2a",
        "puan": "488ff305e0d68493",
        "roller": "55f8b21b06dcdb54",
        "ucret": "6add67a8194b1b55",
    }
    assert _senaryo(
        8, 2026, 12, [25, 26], 1, 2, 42,
        forbidden={("K0", "K1")},
        limits={"K2": {"max": 6, "min": 0, "max_special": 2}, "K3": {"min": 8, "max": 999}},
        together={("K4", "K5")},
    ) == {
        "liste": "e53b37f7202d5934",
        "puan": "410f47d7434b430c",
        "roller": "d6a8cbfd83987b6d",
        "ucret": "1c537265b18d2d3a",
    }
