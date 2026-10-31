"""Uzman listesinin aya kaydı: gerçek PostgreSQL ile.

TEST_DATABASE_URL tanımlı değilse atlanır (GitHub'daki "Testler" iş akışı
geçici bir PostgreSQL açıp bu değişkeni verir).
"""

import os
import uuid
from datetime import date

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
    name = f"uzman-test-{uuid.uuid4().hex[:10]}@example.com"
    yield name
    conn = db._connect()
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM uzman_lists WHERE owner = %s", (name,))
    conn.close()


def test_kaydedilen_liste_ayla_geri_gelir(owner):
    assert db.load_uzman_list(owner, 2026, 11) is None
    assert db.save_uzman_list(owner, 2026, 11, "uzman_kasim.xlsx", {1: ("ÇAĞRI Ş.", "AYŞE K."), 2: ("MURAT Ö.", "")})
    assert db.load_uzman_list(owner, 2026, 11) == ("uzman_kasim.xlsx", {1: ("ÇAĞRI Ş.", "AYŞE K."), 2: ("MURAT Ö.", "")})
    # Başka ay ve başka kullanıcı ayrı.
    assert db.load_uzman_list(owner, 2026, 12) is None
    assert db.load_uzman_list(owner + "x", 2026, 11) is None


def test_yeni_yukleme_eskisinin_yerine_gecer(owner):
    db.save_uzman_list(owner, 2026, 11, "eski.xlsx", {1: ("A", "B"), 2: ("C", "D")})
    db.save_uzman_list(owner, 2026, 11, "yeni.xlsx", {3: ("E", "F")})
    assert db.load_uzman_list(owner, 2026, 11) == ("yeni.xlsx", {3: ("E", "F")})


def test_kilitli_aya_yazilmaz(owner, monkeypatch):
    monkeypatch.setattr(db, "_today", lambda: date(2026, 12, 20))
    with pytest.raises(db.MonthLocked):
        db.save_uzman_list(owner, 2026, 10, "x.xlsx", {1: ("A", "B")})
    assert db.load_uzman_list(owner, 2026, 10) is None
    assert db.save_uzman_list(owner, 2026, 12, "x.xlsx", {1: ("A", "B")})
