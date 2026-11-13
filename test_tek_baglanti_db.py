"""Sayfa çalışması içinde tek bağlantı (db.begin_run_connection): gerçek PostgreSQL ile.

Canlıda her yeni bağlantı ~1 sn sürdüğü için açılıştaki okumalar aynı bağlantıyı
paylaşır; ama bu bir havuz değil: çalışma bitince ya da boşta kalınca kapanır,
bozulan bağlantı atılır. TEST_DATABASE_URL tanımlı değilse atlanır.
"""

import os
import threading
import time

import psycopg2
import pytest

TEST_DB = os.environ.get("TEST_DATABASE_URL")
if not TEST_DB:
    pytest.skip("TEST_DATABASE_URL tanımlı değil", allow_module_level=True)

import db  # noqa: E402


@pytest.fixture(autouse=True)
def test_database(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", TEST_DB)
    yield
    db.end_run_connection()


def _backend_pid(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT pg_backend_pid()")
        return cur.fetchone()[0]


def test_paylasim_yokken_her_cagri_kendi_baglantisini_acar_ve_kapatir():
    first = db._connect()
    first.close()
    second = db._connect()
    try:
        assert first.closed and second is not first
    finally:
        second.close()


def test_calisma_icinde_okumalar_tek_baglantiyi_paylasir_sonda_kapanir():
    db.begin_run_connection()
    first = db._connect()
    pid = _backend_pid(first)
    first.close()
    assert not first.closed                       # paylaşılan bağlantı açık kalır
    second = db._connect()
    assert second is first and _backend_pid(second) == pid
    second.close()
    db.end_run_connection()
    assert first.closed
    third = db._connect()                         # çalışma bitti: yine ayrı bağlantı
    try:
        assert third is not first
    finally:
        third.close()


def test_ic_ice_cagri_paylasilan_baglantiyi_almaz():
    db.begin_run_connection()
    outer = db._connect()
    inner = db._connect()
    try:
        assert inner is not outer
    finally:
        inner.close()
        outer.close()
    assert inner.closed and not outer.closed


def test_hatali_sorgu_sonrakini_bozmaz_ve_oturum_ayari_tasinmaz():
    db.begin_run_connection()
    conn = db._connect()
    conn.set_session(isolation_level=psycopg2.extensions.ISOLATION_LEVEL_REPEATABLE_READ, readonly=True)
    with pytest.raises(psycopg2.Error):
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM olmayan_tablo")
    conn.close()
    again = db._connect()
    try:
        assert again is conn
        with again, again.cursor() as cur:            # salt okunur kalsaydı yazamazdı
            cur.execute("CREATE TEMP TABLE tek_baglanti_deneme (x int)")
            cur.execute("INSERT INTO tek_baglanti_deneme VALUES (1)")
    finally:
        again.close()


def test_kopan_baglanti_atilir_sonraki_okuma_yenisini_acar():
    db.begin_run_connection()
    conn = db._connect()
    pid = _backend_pid(conn)
    conn.close()
    killer = psycopg2.connect(TEST_DB)
    killer.autocommit = True
    with killer.cursor() as cur:                     # Neon uyuyunca bağlantının kopması gibi
        cur.execute("SELECT pg_terminate_backend(%s)", (pid,))
    killer.close()
    conn = db._connect()
    with pytest.raises(psycopg2.Error):
        _backend_pid(conn)
    conn.close()
    fresh = db._connect()
    try:
        assert fresh is not conn and _backend_pid(fresh) != pid
    finally:
        fresh.close()


def test_bosta_kalan_baglanti_kendiliginden_kapanir(monkeypatch):
    monkeypatch.setattr(db, "RUN_CONNECTION_IDLE_SECONDS", 0)
    db.begin_run_connection()
    conn = db._connect()
    conn.close()
    deadline = time.monotonic() + 5
    while not conn.closed and time.monotonic() < deadline:
        time.sleep(0.2)
    assert conn.closed


def test_biten_is_parcaciginin_baglantisi_kapanir():
    box = {}

    def run():
        db.begin_run_connection()
        box["conn"] = db._connect()
        box["conn"].close()                        # st.stop gibi: end_run_connection çağrılmadı

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    deadline = time.monotonic() + 3  # boşta kalma süresinden (5 sn) önce
    while not box["conn"].closed and time.monotonic() < deadline:
        time.sleep(0.2)
    assert box["conn"].closed


def test_calisma_sonunda_olcum_doner():
    db.begin_run_connection()
    for _ in range(2):
        conn = db._connect()
        _backend_pid(conn)
        conn.close()
    olcum = db.end_run_connection()
    assert olcum["uses"] == 2 and len(olcum["connect_ms"]) == 1   # iki okuma, tek bağlantı
    assert 0 < olcum["db_seconds"] <= olcum["page_seconds"]
    assert db.end_run_connection() is None                         # çalışma yoksa ölçüm yok
