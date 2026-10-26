"""Son hatalar: db.py hataları küçük bir tabloya da yazar, Ayarlar'da gösterilir.

TEST_DATABASE_URL tanımlı değilse atlanır (bkz. test_yedek_db.py).
"""

import os

import psycopg2
import pytest

TEST_DB = os.environ.get("TEST_DATABASE_URL")
if not TEST_DB:
    pytest.skip("TEST_DATABASE_URL tanımlı değil", allow_module_level=True)

import db  # noqa: E402


@pytest.fixture(autouse=True)
def temiz_tablo(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", TEST_DB)
    assert db.init_db(), db.get_last_error()
    conn = db._connect()
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM error_log")
    conn.close()


def test_hata_tabloya_sifresiz_yazilir():
    db._log_error("Kayıt", RuntimeError("postgresql://kim:gizli@sunucu/db bozuldu"))
    rows = db.list_recent_errors()
    assert len(rows) == 1
    assert rows[0]["action"] == "Kayıt"
    assert "RuntimeError" in rows[0]["message"]
    assert "gizli" not in rows[0]["message"] and "kim:***@" in rows[0]["message"]
    assert rows[0]["created_tr"]


def test_beklenmeyen_hata_surum_etiketiyle_yazilir():
    db.log_unexpected("Otomatik kayıt", ValueError("kötü değer"))
    row = db.list_recent_errors()[0]
    assert "ValueError" in row["message"] and "kötü değer" in row["message"]
    assert row["version"]


def test_yalniz_son_yirmi_hata_tutulur_yeniden_eskiye():
    for i in range(25):
        db._log_error("Deneme", RuntimeError(f"hata {i}"))
    rows = db.list_recent_errors()
    assert len(rows) == db.ERROR_LOG_KEEP == 20
    assert rows[0]["message"].endswith("hata 24")
    assert rows[-1]["message"].endswith("hata 5")


def test_baglanti_hatasi_tabloya_yazilmaz():
    db._log_error("Ping", psycopg2.OperationalError("uyku"))
    assert db.list_recent_errors() == []


def test_hata_tablosu_yazamazsa_asil_islem_bozulmaz(monkeypatch, capsys):
    monkeypatch.setattr(db.psycopg2, "connect", lambda *a, **k: (_ for _ in ()).throw(OSError("yok")))
    db._log_error("Kayıt", RuntimeError("x"))
    assert "Kayıt error" in capsys.readouterr().out
    assert db.list_recent_errors() is None
