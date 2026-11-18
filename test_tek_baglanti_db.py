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
    # Her okumanın adı ve süresi (ms) kayda girer: hangi okumanın yavaş olduğu görülsün.
    assert [name for name, _ in olcum["reads"]] == ["test_calisma_sonunda_olcum_doner"] * 2
    assert all(isinstance(ms, int) and ms >= 0 for _, ms in olcum["reads"])
    assert 0 < olcum["db_seconds"] <= olcum["page_seconds"]
    assert db.end_run_connection() is None                         # çalışma yoksa ölçüm yok


# --- Okuma tek ağ turu (14 Kasım 2026, K1) ---
# Paylaşılan bağlantıda her okuma BEGIN + SELECT + ROLLBACK diye üç tur gidiyordu
# (canlıda tur ~145 ms). Okuma işlem açmamalı; yazma yine tek işlemde kalmalı.

def _status(conn):
    return conn.info.transaction_status


def test_paylasilan_baglantida_okuma_islem_acmaz():
    db.begin_run_connection()
    conn = db._connect()
    _backend_pid(conn)
    # İşlem açılmadıysa BEGIN gitmemiş, close() da ROLLBACK göndermez: okuma tek tur.
    assert _status(conn) == psycopg2.extensions.TRANSACTION_STATUS_IDLE
    conn.close()


def _uzman_sahibi():
    import uuid
    return f"tek-tur-{uuid.uuid4().hex[:10]}@example.com"


def _sil(owner):
    conn = psycopg2.connect(TEST_DB)
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM schedules WHERE owner = %s", (owner,))
    conn.close()


def test_surumlu_okuma_tek_anlik_goruntuden_ve_sonraki_okuma_yine_tek_tur(monkeypatch):
    import pandas as pd
    assert db.init_db(), db.get_last_error()
    owner = _uzman_sahibi()
    frame = pd.DataFrame(False, index=["Ali", "Ayşe"], columns=["1 Per"])
    try:
        assert db.save_schedule("Otomatik_2026_10", 2026, 10, ["Ali", "Ayşe"], frame, owner=owner, expected_version=db.FORCE)
        seen = {}
        real = db._schedule_content_version

        def spy(cur, schedule_id):
            with cur.connection.cursor() as probe:
                probe.execute("SELECT current_setting('transaction_isolation'), current_setting('transaction_read_only')")
                seen["session"] = probe.fetchone()
            seen["status"] = _status(cur.connection)
            return real(cur, schedule_id)

        monkeypatch.setattr(db, "_schedule_content_version", spy)
        db.begin_run_connection()
        box = {}
        team, *_ = db.load_schedule("Otomatik_2026_10", 2026, 10, owner=owner, version_out=box, raise_on_error=True)
        assert team == ["Ali", "Ayşe"] and box["version"].startswith("h:")
        # Sürüm ve içerik aynı anlık görüntüden: tek işlem, tekrarlanabilir okuma.
        assert seen["session"] == ("repeatable read", "on")
        assert seen["status"] == psycopg2.extensions.TRANSACTION_STATUS_INTRANS
        conn = db._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT current_setting('transaction_isolation'), current_setting('transaction_read_only')")
                assert cur.fetchone() == ("read committed", "off")   # oturum ayarı taşınmadı
            assert _status(conn) == psycopg2.extensions.TRANSACTION_STATUS_IDLE
        finally:
            conn.close()
    finally:
        _sil(owner)


def test_paylasilan_baglantida_yarida_kalan_kayit_hicbir_sey_yazmaz(monkeypatch):
    import pandas as pd
    assert db.init_db(), db.get_last_error()
    owner = _uzman_sahibi()
    frame = pd.DataFrame(False, index=["Ali", "Ayşe"], columns=["1 Per"])
    frame.at["Ali", "1 Per"] = True
    try:
        assert db.save_schedule("Otomatik_2026_10", 2026, 10, ["Ali", "Ayşe"], frame, owner=owner, expected_version=db.FORCE)
        db.begin_run_connection()

        def boom(*a, **k):                       # eski satırlar silindikten sonra hata
            raise psycopg2.OperationalError("yarıda koptu")

        monkeypatch.setattr(db, "execute_values", boom)
        other = frame.copy()
        other.at["Ali", "1 Per"], other.at["Ayşe", "1 Per"] = False, True
        assert db.save_schedule("Otomatik_2026_10", 2026, 10, ["Ali", "Ayşe"], other, owner=owner, expected_version=db.FORCE) is False
        monkeypatch.undo()
        monkeypatch.setenv("DATABASE_URL", TEST_DB)
        db.end_run_connection()
        _, saved, *_ = db.load_schedule("Otomatik_2026_10", 2026, 10, owner=owner)
        assert list(saved.index[saved["1 Per"]]) == ["Ali"]      # eski kayıt olduğu gibi
    finally:
        _sil(owner)


# --- Eski (girişsiz) veri aktarımı yalnız gerektiğinde yazar (18 Kasım 2026) ---

def test_eski_veri_denetimi_aktarilacak_veriyi_kacirmaz():
    import uuid

    import pandas as pd
    assert db.init_db(), db.get_last_error()
    owner = _uzman_sahibi()
    legacy_name = f"Eski_{uuid.uuid4().hex[:8]}"
    frame = pd.DataFrame(False, index=["Ali"], columns=["1 Per"])
    try:
        assert db.save_schedule(legacy_name, 2099, 1, ["Ali"], frame, owner="", expected_version=db.FORCE)
        db.begin_run_connection()
        assert db.claim_legacy_data(owner)
        team, *_ = db.load_schedule(legacy_name, 2099, 1, owner=owner)
        assert team == ["Ali"]                                    # aktarıldı
        assert db.load_schedule(legacy_name, 2099, 1, owner="")[0] is None
        assert db.claim_legacy_data(owner)                        # ikinci kez: değişiklik yok, hata yok
        assert db.load_schedule(legacy_name, 2099, 1, owner=owner)[0] == ["Ali"]
    finally:
        conn = psycopg2.connect(TEST_DB)
        with conn, conn.cursor() as cur:
            cur.execute("DELETE FROM schedules WHERE name = %s", (legacy_name,))
            cur.execute("DELETE FROM app_settings WHERE key = %s", (db.settings_key(owner),))
        conn.close()
        _sil(owner)


def test_olcumde_okuma_adi_ic_yardimci_degil_disaridan_cagrilan_fonksiyondur():
    db.begin_run_connection()
    db.list_team_requests("olcum-test@example.com")             # _rows / _with_schema_repair içinden açar
    assert [name for name, _ in db.end_run_connection()["reads"]] == ["list_team_requests"]
