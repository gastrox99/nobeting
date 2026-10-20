"""Yedek bölümü: gerçek PostgreSQL ile indirme, geri yükleme ve uygulama içi yedekler.

TEST_DATABASE_URL tanımlı değilse atlanır (GitHub'daki "Testler" iş akışı
geçici bir PostgreSQL açıp bu değişkeni verir). Gerçek veritabanının adresini
buraya vermeyin: testler kendi sahip adlarıyla kayıt yazar ve siler.
"""

import json
import os
import uuid

import pandas as pd
import pytest

TEST_DB = os.environ.get("TEST_DATABASE_URL")
if not TEST_DB:
    pytest.skip("TEST_DATABASE_URL tanımlı değil", allow_module_level=True)

import db  # noqa: E402
from nobet_core import (  # noqa: E402
    SNAPSHOT_BEFORE_RESTORE,
    SNAPSHOT_MANUAL,
    backup_summary,
    backup_to_bytes,
    build_backup,
    month_backup,
    parse_backup,
)


@pytest.fixture(autouse=True)
def test_database(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", TEST_DB)
    assert db.init_db(), db.get_last_error()


@pytest.fixture
def owner():
    name = f"yedek-test-{uuid.uuid4().hex[:10]}@example.com"
    yield name
    conn = db._connect()
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM schedules WHERE owner = %s", (name,))
        cur.execute("DELETE FROM backup_snapshots WHERE owner = %s", (name,))
        cur.execute("DELETE FROM app_settings WHERE key = %s", (db.settings_key(name),))
    conn.close()


DAYS = ["1 Per", "2 Cum", "3 Cmt"]
TEAM = ["Ali", "Ayşe", "Can"]


def _save_month(owner, year, month, assigned, team=TEAM):
    schedule = pd.DataFrame(False, index=team, columns=DAYS)
    for person, day in assigned:
        schedule.at[person, day] = True
    prefs = pd.DataFrame(0, index=team, columns=DAYS)
    prefs.at[team[0], DAYS[0]] = 3
    rows = [{"Tarih": f"{i + 1} gün", "AYB": person} for i, (person, _) in enumerate(assigned)]
    settings = {"locked": [list(assigned[0])] if assigned else [], "hidden": {}}
    assert db.save_schedule(
        f"Otomatik_{year}_{month:02d}", year, month, team, schedule,
        pref_df=prefs, settings_dict=settings, rows_liste=rows, owner=owner, expected_version=db.FORCE,
    )


def _assigned(owner, year, month):
    _, schedule, _, _, _ = db.load_schedule(f"Otomatik_{year}_{month:02d}", year, month, owner=owner)
    if schedule is None:
        return None
    return {(person, day) for person in schedule.index for day in schedule.columns if schedule.at[person, day]}


def _full_backup(owner):
    settings, schedules = db.export_owner_data(owner=owner)
    parsed, errors = parse_backup(backup_to_bytes(build_backup(settings, schedules)))
    assert errors == []
    return parsed


def test_indirilen_yedek_baska_hesaba_aynen_geri_yuklenir(owner):
    _save_month(owner, 2026, 9, [("Ali", "1 Per"), ("Can", "3 Cmt")])
    _save_month(owner, 2026, 10, [("Ayşe", "2 Cum")])
    assert db.save_app_settings({"isimler_text": "Ali, Ayşe, Can", "kisi_sayisi": 2}, owner=owner, expected_version=db.FORCE)

    backup = _full_backup(owner)
    assert backup_summary(backup) == "2 ay (Eylül 2026 – Ekim 2026), ekip ve ayarlar"

    other = owner.replace("yedek-test", "yedek-hedef")
    try:
        assert db.restore_owner_data(backup, owner=other)
        # JSON metinleri farklı yazılabilir (ör. "ş" ya da kaçışlı); içerik aynı olmalı.
        assert {**_full_backup(other), "created_at": ""} == {**backup, "created_at": ""}
        _, _, prefs, month_settings, rows = db.load_schedule("Otomatik_2026_09", 2026, 9, owner=other)
        assert int(prefs.at["Ali", "1 Per"]) == 3
        assert month_settings["locked"] == [["Ali", "1 Per"]]
        assert rows[0] == {"Tarih": "1 gün", "AYB": "Ali"}
    finally:
        conn = db._connect()
        with conn, conn.cursor() as cur:
            cur.execute("DELETE FROM schedules WHERE owner = %s", (other,))
            cur.execute("DELETE FROM app_settings WHERE key = %s", (db.settings_key(other),))
        conn.close()


def test_geri_yukleme_yalnizca_yedekteki_aylari_degistirir(owner):
    _save_month(owner, 2026, 9, [("Ali", "1 Per")])
    backup = _full_backup(owner)
    _save_month(owner, 2026, 9, [("Can", "2 Cum")])
    _save_month(owner, 2026, 11, [("Ayşe", "3 Cmt")])

    assert db.restore_owner_data(backup, owner=owner)
    assert _assigned(owner, 2026, 9) == {("Ali", "1 Per")}
    # Yedekte olmayan aya dokunulmaz.
    assert _assigned(owner, 2026, 11) == {("Ayşe", "3 Cmt")}


def test_ekip_degisince_eski_kisinin_hucreleri_kalmaz(owner):
    _save_month(owner, 2026, 9, [("Ali", "1 Per")], team=["Ali", "Ayşe"])
    backup = _full_backup(owner)
    _save_month(owner, 2026, 9, [("Zeki", "3 Cmt")], team=["Ali", "Ayşe", "Zeki"])

    assert db.restore_owner_data(backup, owner=owner)
    team, schedule, _, _, _ = db.load_schedule("Otomatik_2026_09", 2026, 9, owner=owner)
    assert team == ["Ali", "Ayşe"]
    assert set(schedule.index) == {"Ali", "Ayşe"}
    assert _assigned(owner, 2026, 9) == {("Ali", "1 Per")}


def test_yarida_kalan_geri_yukleme_hicbir_seyi_degistirmez(owner):
    _save_month(owner, 2026, 9, [("Ali", "1 Per")])
    _save_month(owner, 2026, 10, [("Ali", "1 Per")])
    backup = _full_backup(owner)
    backup["schedules"][0]["assigned"] = [("Can", "3 Cmt")]
    # İkinci ay veritabanı hatasına yol açar (yıl sayı değil).
    backup["schedules"][1]["year"] = "bozuk"

    with pytest.raises(Exception):
        db.restore_owner_data(backup, owner=owner)
    assert _assigned(owner, 2026, 9) == {("Ali", "1 Per")}


def test_ay_yedegi_her_ay_icin_tek_ve_uzerine_yazar(owner):
    _save_month(owner, 2026, 9, [("Ali", "1 Per")])
    _save_month(owner, 2026, 10, [("Ayşe", "2 Cum")])

    def take(year, month, created):
        settings, schedules = db.export_owner_data(owner=owner)
        bk = month_backup(build_backup(settings, schedules), year, month)
        db.save_backup_snapshot(owner, year, month, created, SNAPSHOT_MANUAL,
                                backup_summary(bk), backup_to_bytes(bk).decode("utf-8"))

    take(2026, 9, "2026-09-30 10:00")
    take(2026, 10, "2026-10-31 10:00")
    _save_month(owner, 2026, 9, [("Can", "3 Cmt")])
    take(2026, 9, "2026-10-01 09:00")

    snaps = db.list_backup_snapshots(owner=owner)
    assert [(s["year"], s["month"], s["created_tr"]) for s in snaps] == [
        (2026, 10, "2026-10-31 10:00"), (2026, 9, "2026-10-01 09:00"),
    ]
    september = next(s for s in snaps if s["month"] == 9)
    parsed, errors = parse_backup(db.load_backup_snapshot(september["id"], owner=owner).encode("utf-8"))
    assert errors == []
    assert parsed["settings"] is None
    assert [item["name"] for item in parsed["schedules"]] == ["Otomatik_2026_09"]
    assert set(parsed["schedules"][0]["assigned"]) == {("Can", "3 Cmt")}


def test_geri_yukleme_oncesi_kopya_tek_ve_ay_yedeklerinden_ayri(owner):
    _save_month(owner, 2026, 9, [("Ali", "1 Per")])
    payload = backup_to_bytes(_full_backup_raw(owner)).decode("utf-8")
    db.save_backup_snapshot(owner, 2026, 9, "2026-10-01 09:00", SNAPSHOT_MANUAL, "1 ay", payload)
    db.save_backup_snapshot(owner, 0, 0, "2026-10-02 09:00", SNAPSHOT_BEFORE_RESTORE, "1 ay", payload)
    db.save_backup_snapshot(owner, 0, 0, "2026-10-03 09:00", SNAPSHOT_BEFORE_RESTORE, "1 ay", payload)

    snaps = db.list_backup_snapshots(owner=owner)
    assert sorted((s["reason"], s["created_tr"]) for s in snaps) == [
        (SNAPSHOT_BEFORE_RESTORE, "2026-10-03 09:00"), (SNAPSHOT_MANUAL, "2026-10-01 09:00"),
    ]


def test_baska_kullanicinin_yedegi_gorunmez_ve_okunmaz(owner):
    db.save_backup_snapshot(owner, 2026, 9, "2026-10-01 09:00", SNAPSHOT_MANUAL, "1 ay", "{}")
    snap_id = db.list_backup_snapshots(owner=owner)[0]["id"]
    stranger = "yabanci-" + owner
    assert db.list_backup_snapshots(owner=stranger) == []
    assert db.load_backup_snapshot(snap_id, owner=stranger) is None
    assert db.load_backup_snapshot(snap_id, owner=owner) == "{}"


def test_yedek_dosyasi_json_olarak_okunabilir(owner):
    _save_month(owner, 2026, 9, [("Ali", "1 Per")])
    data = json.loads(backup_to_bytes(_full_backup_raw(owner)))
    assert data["format"] == "nobeting-yedek"
    assert data["schedules"][0]["team"] == TEAM
    assert data["schedules"][0]["days"] == DAYS


def _full_backup_raw(owner):
    settings, schedules = db.export_owner_data(owner=owner)
    return build_backup(settings, schedules)
