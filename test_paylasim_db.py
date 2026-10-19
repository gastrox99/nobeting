"""Kalıcı paylaşım bağlantısı: gerçek PostgreSQL ile.

TEST_DATABASE_URL tanımlı değilse atlanır (GitHub'daki "Testler" iş akışı
geçici bir PostgreSQL açıp bu değişkeni verir). Gerçek veritabanının adresini
buraya vermeyin: testler kendi sahip adlarıyla kayıt yazar ve siler.
"""

import os
import uuid
from datetime import datetime, timedelta

import pandas as pd
import pytest

TEST_DB = os.environ.get("TEST_DATABASE_URL")
if not TEST_DB:
    pytest.skip("TEST_DATABASE_URL tanımlı değil", allow_module_level=True)

import db  # noqa: E402
from nobet_core import now_tr  # noqa: E402


@pytest.fixture(autouse=True)
def test_database(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", TEST_DB)
    assert db.init_db(), db.get_last_error()


@pytest.fixture
def owner():
    name = f"paylasim-test-{uuid.uuid4().hex[:10]}@example.com"
    yield name
    conn = db._connect()
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM schedules WHERE owner = %s", (name,))
        cur.execute("DELETE FROM share_links WHERE owner = %s", (name,))
        cur.execute("DELETE FROM team_requests WHERE owner = %s", (name,))
    conn.close()


def _save(owner, year, month, assigned=True, name=None):
    schedule = pd.DataFrame(False, index=["Ali", "Ayşe"], columns=["1 Per", "2 Cum"])
    if assigned:
        schedule.at["Ali", "1 Per"] = True
    assert db.save_schedule(name or f"Otomatik_{year}_{month:02d}", year, month,
                            ["Ali", "Ayşe"], schedule, owner=owner)


def test_kalici_baglanti_tum_aylari_acar(owner):
    token = db.create_share_link(owner, 0, 0)
    assert db.resolve_share_token(token) == (owner, 0, 0)
    assert db.get_share_link(owner, 0, 0) == token


def test_paylasilan_aylar_yalniz_nobeti_olan_otomatik_kayitlar(owner):
    _save(owner, 2026, 11)
    _save(owner, 2026, 9)
    _save(owner, 2026, 10, assigned=False)       # boş ay görünmez
    _save(owner, 2026, 12, name="Elle kayıt")    # otomatik olmayan kayıt görünmez
    months = db.list_shared_months(owner)
    assert [(y, m) for y, m, _ in months] == [(2026, 9), (2026, 11)]
    updated = months[-1][2]
    # Kayıt zamanı Türkiye saatine çevrilir.
    now = now_tr().replace(tzinfo=None)
    assert isinstance(updated, datetime)
    assert abs(updated - now) < timedelta(minutes=5)


def test_baska_kullanicinin_aylari_gorunmez(owner):
    _save(owner, 2026, 11)
    assert db.list_shared_months(owner + ".baska") == []


def test_tum_baglantilar_kapanir(owner):
    monthly = db.create_share_link(owner, 2026, 10)
    permanent = db.create_share_link(owner, 0, 0)
    assert db.resolve_share_token(monthly) == (owner, 2026, 10)
    assert db.delete_all_share_links(owner)
    assert db.resolve_share_token(monthly) is None
    assert db.resolve_share_token(permanent) is None


def test_ekip_talebi_yeniden_gonderilince_yerine_gecer_ve_silinir(owner):
    assert db.submit_team_request(owner, 2026, 11, "Ali", {"tercih": [3]}) == "ok"
    assert db.submit_team_request(owner, 2026, 11, "Ali", {"izin": [5, 6]}) == "ok"
    assert db.submit_team_request(owner, 2026, 12, "Ali", {"kacin": [1]}) == "ok"
    requests = db.list_team_requests(owner)
    assert [(r["person"], r["month"], r["days"]) for r in requests] == [
        ("Ali", 11, {"izin": [5, 6]}), ("Ali", 12, {"kacin": [1]}),
    ]
    assert db.list_team_requests(owner + ".baska") == []
    # Başka kullanıcının talebi silinemez.
    assert db.delete_team_request(owner + ".baska", requests[0]["id"])
    assert len(db.list_team_requests(owner)) == 2
    assert db.delete_team_request(owner, requests[0]["id"])
    assert [r["month"] for r in db.list_team_requests(owner)] == [12]


def test_ekip_talebi_siniri(owner, monkeypatch):
    monkeypatch.setattr(db, "TEAM_REQUEST_LIMIT", 2)
    assert db.submit_team_request(owner, 2026, 11, "Ali", {"tercih": [3]}) == "ok"
    assert db.submit_team_request(owner, 2026, 11, "Ayşe", {"tercih": [3]}) == "ok"
    assert db.submit_team_request(owner, 2026, 11, "Can", {"tercih": [3]}) == "limit"
    # Kendi bekleyen talebini güncellemek sınıra takılmaz.
    assert db.submit_team_request(owner, 2026, 11, "Ali", {"izin": [4]}) == "ok"
