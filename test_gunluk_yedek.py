"""Günlük otomatik kopya: günün ilk açılışında tüm veri saklanır, son 7 gün kalır."""

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import db
from nobet_core import SNAPSHOT_DAILY, SNAPSHOT_MANUAL, TR_TZ, daily_snapshot_due


def test_gunluk_kopya_bugun_alinmadiysa_gerekir():
    today = date(2026, 10, 27)
    assert daily_snapshot_due([], today)
    assert daily_snapshot_due([{"reason": SNAPSHOT_DAILY, "created_tr": "2026-10-26 23:59"}], today)
    # Ay yedeği günlük kopyanın yerini tutmaz.
    assert daily_snapshot_due([{"reason": SNAPSHOT_MANUAL, "created_tr": "2026-10-27 09:00"}], today)
    assert not daily_snapshot_due([{"reason": SNAPSHOT_DAILY, "created_tr": "2026-10-27 00:05"}], today)


def _run_app(snapshots, runs=2, now=None, **extra):
    import streamlit
    from streamlit.testing.v1 import AppTest

    saved = []
    fakes = dict(
        init_db=lambda: True, get_last_error=lambda: None, is_database_configured=lambda: True,
        SCHEMA_VERSION="gunluk-yedek-test",
        load_app_settings=lambda **k: None, save_app_settings=lambda *a, **k: True,
        load_schedule=lambda *a, **k: None, save_schedule=lambda *a, **k: True,
        load_auto_saved_assignments=lambda *a, **k: ({}, {}),
        list_backup_snapshots=lambda **k: list(snapshots),
        export_owner_data=lambda owner="": ({"team": "Ali, Ayşe"}, []),
        save_backup_snapshot=lambda *a: saved.append(a),
        get_share_link=lambda *a: None, list_team_requests=lambda *a, **k: [],
        list_recent_errors=lambda *a, **k: [], ping_db=lambda: True,
    )
    fakes.update(extra)
    patches = [
        patch.dict("os.environ", {"DATABASE_URL": "", "GIRISSIZ_KULLANIM": "1"}),
        patch.multiple(db, create=True, **fakes),
    ]
    for item in patches:
        item.start()
    streamlit.cache_resource.clear()
    try:
        at = AppTest.from_file(str(Path(__file__).with_name("nobet.py")), default_timeout=120)
        at.session_state["isimler_text"] = "Ali, Ayşe"
        at.session_state["storage_init"] = {}
        at.session_state["yil_val"], at.session_state["ay_val"] = 2026, 10
        for _ in range(runs):
            at.run()
            assert not at.exception, at.exception
        if now is not None:
            with patch("nobet_core.now_tr", return_value=now), patch("nobet_yedek.now_tr", return_value=now):
                at.run()
    finally:
        for item in reversed(patches):
            item.stop()
        streamlit.cache_resource.clear()
    return saved


def _today():
    return datetime.now(TR_TZ).date()


def test_acilista_tum_verinin_gunluk_kopyasi_bir_kez_alinir():
    today = _today()
    saved = _run_app([])
    assert len(saved) == 1
    _owner, year, month, created, reason, _summary, data = saved[0]
    assert (year, month, reason) == (0, today.isoweekday(), SNAPSHOT_DAILY)
    assert created[:10] == today.isoformat()
    assert '"team": "Ali, Ayşe"' in data


def test_bugunun_kopyasi_varsa_yeniden_alinmaz():
    today = f"{_today().isoformat()} 08:00"
    assert _run_app([{"id": 1, "reason": SNAPSHOT_DAILY, "year": 0, "month": 1, "created_tr": today}]) == []


def test_acik_kalan_oturumda_ertesi_gun_yeniden_alinir():
    today = _today()
    tomorrow = datetime.now(TR_TZ) + timedelta(days=1)
    saved = _run_app([], now=tomorrow)
    assert [s[2] for s in saved] == [today.isoweekday(), tomorrow.isoweekday()]


def test_kopya_alinamazsa_uygulama_calismaya_devam_eder():
    def fail(owner=""):
        raise RuntimeError("bağlantı koptu")

    errors = []
    assert _run_app([], export_owner_data=fail, log_unexpected=lambda action, exc: errors.append(action)) == []
    assert errors == ["Günlük yedek"]
