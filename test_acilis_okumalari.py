"""İlk açılışta veritabanı okuma sayısı (13 Kasım 2026, ilk açılış yavaşlığı).

Canlıda her okuma ayrı bağlantı açıyor ve ~1,2 sn sürüyor. Aynı tablodan iki kez
okunan veriler tek okumada gelmeli; sonuç değişmemeli.
"""
import pandas as pd

import nobet_core
from test_bozuk_kayit import _app, _stop

TEAM = "Ali, Ayşe, Can"


def _talep(id_, person, status):
    return {"id": id_, "year": 2026, "month": 10, "person": person, "days": {},
            "created_tr": None, "status": status, "decided_tr": None}


def test_ekip_talepleri_acilista_bir_kez_okunur_bekleyenler_suzulur():
    calls = []

    def list_team_requests(owner, raise_on_error=False, include_decided=False):
        calls.append(include_decided)
        rows = [_talep(1, "Can", "pending"), _talep(2, "Ali", "approved")]
        return rows if include_decided else [r for r in rows if r["status"] == "pending"]

    at, patches = _app(
        lambda *a, **k: (None, None, None, None, None), lambda *a, **k: True, team=TEAM,
        list_team_requests=list_team_requests,
    )
    try:
        at.run()
        assert [e.value for e in at.exception] == []
        captions = [c.value for c in at.caption]
        labels = [e.label for e in at.expander]
    finally:
        _stop(patches)
    assert len(calls) == 1
    # Onaylanan talep "onay bekliyor" sayısına girmez, ama "gönderdi" sayısına girer.
    assert any("1 ekip talebi onay bekliyor" in c for c in captions)
    assert any("Talep durumu (2/3 gönderdi)" in label for label in labels)


def test_onceki_aylar_ve_sonraki_ay_tek_okumada_gelir():
    gd = nobet_core.build_gun_detaylari(2026, 11, 30, set())
    columns = list(gd)
    kasim = pd.DataFrame(False, index=["Ali", "Ayşe", "Can"], columns=columns)
    kasim.at["Ali", columns[0]] = True      # 1 Kasım
    kasim.at["Can", columns[2]] = True      # 3 Kasım
    calls = []

    def loader(periods, owner="", include_preferences=False):
        calls.append(list(periods))
        frames = {(2026, 11): kasim} if (2026, 11) in periods else {}
        return (frames, {}) if include_preferences else frames

    at, patches = _app(
        lambda *a, **k: (None, None, None, None, None), lambda *a, **k: True, team=TEAM,
        load_auto_saved_assignments=loader,
    )
    try:
        at.run()
        assert [e.value for e in at.exception] == []
        first_shift = at.session_state["next_month_first_shift"]
        ctx = at.session_state["prev_month_context"]
    finally:
        _stop(patches)
    assert len(calls) == 1 and (2026, 11) in calls[0] and (2026, 9) in calls[0]
    assert first_shift == {"Ali": 1, "Can": 3}
    # Sonraki ay önceki aylar dengesine karışmaz ve oturumda tutulmaz.
    assert ctx["months"] == 0 and "next_frame" not in ctx
