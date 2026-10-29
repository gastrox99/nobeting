# -*- coding: utf-8 -*-
"""Önceki aylar bağlamı (nobet.py _load_previous_context): ay geçişi dinlenmesi, aylar arası
denge ve izinli günler. Diğer testler bu yükleyiciyi boş sonuçla taklit ettiği için burada
gerçek bir Eylül kaydıyla çalıştırılır."""
import json

import pandas as pd

import nobet_core
from test_bozuk_kayit import _app, _stop, _messages

TEAM = "Ali, Ayşe, Can"


def _eylul():
    gd = nobet_core.build_gun_detaylari(2026, 9, 30, set())
    columns = list(gd)
    frame = pd.DataFrame(False, index=["Ali", "Ayşe", "Can"], columns=columns)
    frame.at["Ali", columns[-1]] = True          # 30 Eylül: ay geçişi dinlenmesi için
    frame.at["Ayşe", next(c for c in columns if gd[c]["weekend"])] = True
    frame.at["Can", columns[0]] = True
    pref_json = json.dumps({columns[4]: {"Can": 3, "Ali": 4}, columns[5]: {"Can": 3}})
    return frame, pref_json


def _calistir(loader):
    at, patches = _app(
        lambda *a, **k: (None, None, None, None, None), lambda *a, **k: True,
        team=TEAM, load_auto_saved_assignments=loader,
    )
    try:
        at.run()
        assert [e.value for e in at.exception] == []
        return at.session_state["prev_month_context"], _messages(at)
    finally:
        _stop(patches)


def test_onceki_ay_son_nobet_denge_ve_izinli_gunleri_yuklenir():
    frame, pref_json = _eylul()
    ctx, _ = _calistir(lambda periods, **k: ({(2026, 9): frame}, {(2026, 9): pref_json}))
    assert ctx["months"] == 1 and "failed_at" not in ctx
    assert ctx["last_shift"]["Ali"] == 0                       # 30 Eylül: ay sonundan 0 gün geride
    assert "Ali" in ctx["last_shift"]
    assert sum(ctx["history_counts"].values()) >= 1             # Ayşe'nin hafta sonu nöbeti
    assert ctx["monthly_totals"] == [{"Ali": 1, "Ayşe": 1, "Can": 1}]
    # Can'ın iki izin günü ve Ali'nin bir müsait değil günü o ayın "uygun gün"ünden düşer.
    assert ctx["monthly_available"] == [{"Ali": 29, "Ayşe": 30, "Can": 28}]


def test_onceki_aylar_okunamazsa_uyari_gosterilir_ve_yeniden_denenmek_uzere_isaretlenir():
    def kirik(*args, **kwargs):
        raise RuntimeError("bağlantı koptu")
    ctx, messages = _calistir(kirik)
    assert "failed_at" in ctx and ctx["months"] == 0
    assert any("Önceki aylar okunamadı" in m for m in messages)
