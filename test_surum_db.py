"""Kayıt sürümü denetimi ve paylaşım okuması: gerçek PostgreSQL ile.

TEST_DATABASE_URL tanımlı değilse atlanır (bkz. test_paylasim_db.py).
"""

import os
import uuid

import pandas as pd
import pytest

TEST_DB = os.environ.get("TEST_DATABASE_URL")
if not TEST_DB:
    pytest.skip("TEST_DATABASE_URL tanımlı değil", allow_module_level=True)

import db  # noqa: E402


@pytest.fixture(autouse=True)
def test_database(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", TEST_DB)
    assert db.init_db(), db.get_last_error()


@pytest.fixture
def owner():
    name = f"surum-test-{uuid.uuid4().hex[:10]}@example.com"
    yield name
    conn = db._connect()
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM schedules WHERE owner = %s", (name,))
        cur.execute("DELETE FROM app_settings WHERE key = %s", (db.settings_key(name),))
    conn.close()


def _schedule(person):
    frame = pd.DataFrame(False, index=["Ali", "Ayşe"], columns=["1 Per"])
    frame.at[person, "1 Per"] = True
    return frame


def _save(owner, person, **kwargs):
    return db.save_schedule("Otomatik_2026_10", 2026, 10, ["Ali", "Ayşe"], _schedule(person), owner=owner, **kwargs)


def _assigned(owner):
    _, schedule, _, _, _ = db.load_schedule("Otomatik_2026_10", 2026, 10, owner=owner)
    return list(schedule.index[schedule["1 Per"]])


def test_kayit_yoksa_surum_bos_metin(owner):
    assert db.get_schedule_version("Otomatik_2026_10", 2026, 10, owner=owner) == ""
    assert db.get_settings_version(owner=owner) == ""


def test_eski_surumle_yazan_oturum_reddedilir(owner):
    phone, desktop = db.get_schedule_version("Otomatik_2026_10", 2026, 10, owner=owner), None
    out = {}
    assert _save(owner, "Ali", expected_version=phone, version_out=out)
    phone = out["version"]
    # Bilgisayar oturumu ilk sürümü ("") okumuştu; telefonun yazdığından sonra yazamaz.
    desktop = ""
    with pytest.raises(db.SaveConflict):
        _save(owner, "Ayşe", expected_version=desktop)
    assert _assigned(owner) == ["Ali"]
    # Güncel sürümü bilen oturum yazabilir ve yeni sürümü alır.
    out2 = {}
    assert _save(owner, "Ayşe", expected_version=phone, version_out=out2)
    assert out2["version"] != phone
    assert db.get_schedule_version("Otomatik_2026_10", 2026, 10, owner=owner) == out2["version"]
    assert _assigned(owner) == ["Ayşe"]


def test_surum_verilmezse_denetim_yapilmaz(owner):
    assert _save(owner, "Ali")
    assert _save(owner, "Ayşe")
    assert _assigned(owner) == ["Ayşe"]


def test_ayar_kaydinda_surum_denetimi(owner):
    out = {}
    assert db.save_app_settings({"a": 1}, owner=owner, expected_version="", version_out=out)
    with pytest.raises(db.SaveConflict):
        db.save_app_settings({"a": 2}, owner=owner, expected_version="")
    assert db.load_app_settings(owner=owner) == {"a": 1}
    assert db.save_app_settings({"a": 3}, owner=owner, expected_version=out["version"])
    assert db.load_app_settings(owner=owner) == {"a": 3}


def test_paylasim_okumasi_tercih_verisini_okumaz(owner):
    prefs = pd.DataFrame(0, index=["Ali", "Ayşe"], columns=["1 Per"])
    prefs.at["Ali", "1 Per"] = 3
    assert _save(owner, "Ali", pref_df=prefs)
    assert db.load_schedule("Otomatik_2026_10", 2026, 10, owner=owner)[2] is not None
    assert db.load_schedule("Otomatik_2026_10", 2026, 10, owner=owner, include_preferences=False)[2] is None
