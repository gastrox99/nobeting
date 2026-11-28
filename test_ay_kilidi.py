# -*- coding: utf-8 -*-
"""Geçmiş ay kilidi: biten ay 5 gün sonra kilitlenir, açık aylar listesiyle açılır."""
import os
import uuid
from datetime import date

import pandas as pd
import pytest

import db

TEST_DB = os.environ.get("TEST_DATABASE_URL")
needs_db = pytest.mark.skipif(not TEST_DB, reason="TEST_DATABASE_URL tanımlı değil")


def test_ay_bitince_bes_gun_sonra_kilitlenir():
    assert not db.month_is_locked(2026, 10, [], today=date(2026, 10, 31))
    assert not db.month_is_locked(2026, 10, [], today=date(2026, 11, 5))
    assert db.month_is_locked(2026, 10, [], today=date(2026, 11, 6))
    assert db.month_is_locked(2025, 12, [], today=date(2026, 1, 6))
    assert not db.month_is_locked(2025, 12, [], today=date(2026, 1, 5))


def test_acik_ay_kilitli_sayilmaz():
    assert not db.month_is_locked(2026, 9, ["2026-09"], today=date(2026, 11, 20))
    assert db.month_is_locked(2026, 9, ["2026-08"], today=date(2026, 11, 20))
    assert db.month_is_locked(2026, 9, None, today=date(2026, 11, 20))


@pytest.fixture
def owner(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", TEST_DB)
    assert db.init_db(), db.get_last_error()
    name = f"kilit-test-{uuid.uuid4().hex[:10]}@example.com"
    yield name
    conn = db._connect()
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM schedules WHERE owner = %s", (name,))
        cur.execute("DELETE FROM app_settings WHERE key = %s", (db.settings_key(name),))
    conn.close()


def _save(owner, year, month):
    frame = pd.DataFrame(False, index=["Ali", "Ayşe"], columns=["1 Per"])
    frame.at["Ali", "1 Per"] = True
    return db.save_schedule(f"Otomatik_{year}_{month:02d}", year, month, ["Ali", "Ayşe"], frame,
                            owner=owner, expected_version=db.FORCE)


def _assigned(owner, year, month):
    _, schedule, *_ = db.load_schedule(f"Otomatik_{year}_{month:02d}", year, month, owner=owner)
    return [p for p in schedule.index if schedule.at[p, "1 Per"]]


@needs_db
def test_kilitli_aya_kayit_reddedilir_acilinca_yazilir_tekrar_kilitlenir(owner, monkeypatch):
    monkeypatch.setattr(db, "_today", lambda: date(2026, 11, 20))
    with pytest.raises(db.MonthLocked):
        _save(owner, 2026, 9)
    assert db.save_app_settings({"open_months": ["2026-09"]}, owner=owner, expected_version=db.FORCE)
    assert _save(owner, 2026, 9) is True
    assert _assigned(owner, 2026, 9) == ["Ali"]
    with pytest.raises(db.MonthLocked):  # yalnız Eylül açıldı
        _save(owner, 2026, 8)
    assert db.save_app_settings({"open_months": []}, owner=owner, expected_version=db.FORCE)
    with pytest.raises(db.MonthLocked):
        _save(owner, 2026, 9)


@needs_db
def test_kilit_gelmemis_aya_dokunmaz(owner, monkeypatch):
    monkeypatch.setattr(db, "_today", lambda: date(2026, 11, 5))
    assert _save(owner, 2026, 10) is True  # Ekim'in 5 günlük süresi dolmadı
    assert _save(owner, 2026, 12) is True


@needs_db
def test_geri_yukleme_kilitli_aya_yazmaz_ve_atlanani_bildirir(owner, monkeypatch):
    from nobet_core import backup_to_bytes, build_backup, parse_backup
    monkeypatch.setattr(db, "_today", lambda: date(2026, 10, 2))  # Eylül henüz açık
    assert _save(owner, 2026, 9) is True
    assert _save(owner, 2026, 10) is True
    backup, errors = parse_backup(backup_to_bytes(build_backup(*db.export_owner_data(owner=owner))))
    assert errors == []
    frame = pd.DataFrame(False, index=["Ali", "Ayşe"], columns=["1 Per"])
    frame.at["Ayşe", "1 Per"] = True
    for month in (9, 10):
        assert db.save_schedule(f"Otomatik_2026_{month:02d}", 2026, month, ["Ali", "Ayşe"], frame,
                                owner=owner, expected_version=db.FORCE)
    monkeypatch.setattr(db, "_today", lambda: date(2026, 11, 2))  # Eylül kilitlendi, Ekim değil
    skipped = []
    assert db.restore_owner_data(backup, owner=owner, skipped_out=skipped)
    assert skipped == [(2026, 9)]
    assert _assigned(owner, 2026, 9) == ["Ayşe"]  # kilitli ay değişmedi
    assert _assigned(owner, 2026, 10) == ["Ali"]  # açık ay geri yüklendi
    # Kilit açılınca aynı yedek o ayı da yükler.
    assert db.save_app_settings({"open_months": ["2026-09"]}, owner=owner, expected_version=db.FORCE)
    skipped = []
    backup["settings"] = None
    assert db.restore_owner_data(backup, owner=owner, skipped_out=skipped)
    assert skipped == [] and _assigned(owner, 2026, 9) == ["Ali"]


@needs_db
def test_geri_yukleme_acik_aylar_listesini_degistirmez(owner, monkeypatch):
    from nobet_core import backup_to_bytes, build_backup, parse_backup
    # Yedek, Eylül'ün açık olduğu bir anda alındı.
    monkeypatch.setattr(db, "_today", lambda: date(2026, 11, 20))
    assert db.save_app_settings({"team": "Ali", "open_months": ["2026-09"]}, owner=owner, expected_version=db.FORCE)
    assert _save(owner, 2026, 9) is True
    backup, errors = parse_backup(backup_to_bytes(build_backup(*db.export_owner_data(owner=owner))))
    assert errors == []
    # Sonra Eylül kilitlendi, Ekim açıldı: geri yükleme bu kilit durumunu bozmamalı.
    assert db.save_app_settings({"team": "Ali", "open_months": ["2026-10"]}, owner=owner, expected_version=db.FORCE)
    assert db.restore_owner_data(backup, owner=owner, skipped_out=[])
    assert db.load_app_settings(owner=owner)["open_months"] == ["2026-10"]
    # Ayarların geri kalanı yedekten gelir.
    assert db.load_app_settings(owner=owner)["team"] == "Ali"
