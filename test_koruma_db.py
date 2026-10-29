# -*- coding: utf-8 -*-
"""Kayıt koruması (Mimar raporu 20 Ekim 2026, madde 1, 2, 5, 6 ve 9): davranış testleri.

Gerçek PostgreSQL gerektirenler TEST_DATABASE_URL yoksa atlanır; geri kalanı
(kod hatasının gizlenmemesi, uygulama akışı) her zaman çalışır.
"""
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import psycopg2
import pytest

import db

TEST_DB = os.environ.get("TEST_DATABASE_URL")
needs_db = pytest.mark.skipif(not TEST_DB, reason="TEST_DATABASE_URL tanımlı değil")
NAME = "Otomatik_2026_10"


@pytest.fixture
def owner(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", TEST_DB)
    assert db.init_db(), db.get_last_error()
    name = f"koruma-test-{uuid.uuid4().hex[:10]}@example.com"
    yield name
    monkeypatch.setenv("DATABASE_URL", TEST_DB)
    conn = db._connect()
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM schedules WHERE owner = %s", (name,))
        cur.execute("DELETE FROM app_settings WHERE key = %s", (db.settings_key(name),))
    conn.close()


def _frame(person):
    frame = pd.DataFrame(False, index=["Ali", "Ayşe"], columns=["1 Per"])
    frame.at[person, "1 Per"] = True
    return frame


def _save(owner, person, **kwargs):
    return db.save_schedule(NAME, 2026, 10, ["Ali", "Ayşe"], _frame(person), owner=owner, **kwargs)


def _assigned(owner):
    _, schedule, *_ = db.load_schedule(NAME, 2026, 10, owner=owner)
    return [p for p in schedule.index if schedule.at[p, "1 Per"]]


def _unreachable(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:x@localhost:1/yok")


# --- Madde 1: sürüm okunamazsa denetimsiz yazılmaz ---------------------------

@needs_db
def test_okunamayan_surumle_baska_cihazin_kaydi_ezilmez(owner, monkeypatch):
    assert _save(owner, "Ali", expected_version=db.FORCE)
    _unreachable(monkeypatch)
    stale = db.get_schedule_version(NAME, 2026, 10, owner=owner)   # Neon uykuda
    monkeypatch.setenv("DATABASE_URL", TEST_DB)
    assert stale is None
    phone = db.get_schedule_version(NAME, 2026, 10, owner=owner)
    assert _save(owner, "Ayşe", expected_version=phone, writer="telefon")
    with pytest.raises(db.VersionUnknown):
        _save(owner, "Ali", expected_version=stale, writer="eski-sekme")
    assert _assigned(owner) == ["Ayşe"]


@needs_db
def test_okunamayan_ayar_surumuyle_kayit_durur(owner):
    assert db.save_app_settings({"a": 1}, owner=owner, expected_version="")
    with pytest.raises(db.VersionUnknown):
        db.save_app_settings({"a": 2}, owner=owner, expected_version=None)
    assert db.load_app_settings(owner=owner) == {"a": 1}


@needs_db
def test_surum_icerikle_ayni_okumadan_gelir(owner):
    box = {"version": None}
    assert db.load_schedule(NAME, 2026, 10, raise_on_error=True, owner=owner, version_out=box)[0] is None
    assert box["version"] == ""          # kayıt yok: ilk kayıt "" ile denetlenir
    assert _save(owner, "Ali", expected_version="")
    db.load_schedule(NAME, 2026, 10, raise_on_error=True, owner=owner, version_out=box)
    assert box["version"] == db.get_schedule_version(NAME, 2026, 10, owner=owner)
    # Okunan sürümle yazmak çalışır, başka yerde değiştiyse çakışır.
    assert _save(owner, "Ayşe", expected_version=box["version"])
    with pytest.raises(db.SaveConflict):
        _save(owner, "Ali", expected_version=box["version"])

    settings_box = {"version": None}
    assert db.load_app_settings(owner=owner, raise_on_error=True, version_out=settings_box) is None
    assert settings_box["version"] == ""
    assert db.save_app_settings({"a": 1}, owner=owner, expected_version="")
    assert db.load_app_settings(owner=owner, raise_on_error=True, version_out=settings_box) == {"a": 1}
    assert settings_box["version"] == db.get_settings_version(owner=owner)


@needs_db
def test_okuma_basarisizsa_surum_bos_kalir(owner, monkeypatch):
    box = {"version": None}
    _unreachable(monkeypatch)
    with pytest.raises(psycopg2.Error):
        db.load_schedule(NAME, 2026, 10, raise_on_error=True, owner=owner, version_out=box)
    assert box["version"] is None


# --- Madde 2: serileştirme hatası alanı NULL yazmaz -------------------------

@needs_db
def test_serilestirilemeyen_ayar_kaydi_durdurur_eski_veri_kalir(owner):
    prefs = pd.DataFrame(0, index=["Ali", "Ayşe"], columns=["1 Per"])
    prefs.at["Ali", "1 Per"] = 2
    settings = {"locked": [["Ali", "1 Per"]], "hidden": {}}
    out = {}
    assert _save(owner, "Ali", pref_df=prefs, settings_dict=settings, expected_version="", version_out=out)
    bad = {"locked": [["Ali", "1 Per"]], "hidden": {"Veli": {"shifts": [np.int64(3)], "prefs": {}}}}
    with pytest.raises(TypeError):
        _save(owner, "Ayşe", pref_df=prefs, settings_dict=bad, expected_version=out["version"])
    _, schedule, saved_prefs, saved_settings, _ = db.load_schedule(NAME, 2026, 10, owner=owner)
    assert saved_settings == settings
    assert int(saved_prefs.at["Ali", "1 Per"]) == 2
    assert _assigned(owner) == ["Ali"]


# --- Madde 5: aynı ayın ilk kaydı eşzamanlı yapılınca biri ezilmez -----------

@needs_db
def test_eszamanli_ilk_kayitta_ikincisi_cakisir(owner, monkeypatch):
    original = db.execute_values
    first_inside = threading.Event()

    def slow(cur, sql, rows, *args, **kwargs):
        result = original(cur, sql, rows, *args, **kwargs)
        if not first_inside.is_set():
            first_inside.set()
            time.sleep(1.0)          # A satırı ekledi, henüz commit etmedi
        return result

    monkeypatch.setattr(db, "execute_values", slow)
    results = {}

    def save(tag, person):
        try:
            results[tag] = _save(owner, person, expected_version="", writer=tag)
        except db.SaveConflict:
            results[tag] = "çakışma"

    a = threading.Thread(target=save, args=("telefon", "Ali"))
    a.start()
    assert first_inside.wait(5)
    b = threading.Thread(target=save, args=("masaüstü", "Ayşe"))
    b.start()
    a.join()
    b.join()
    assert results == {"telefon": True, "masaüstü": "çakışma"}
    assert _assigned(owner) == ["Ali"]


# --- Madde 9: eşzamanlı şema hazırlığı -------------------------------------

@needs_db
def test_eszamanli_init_db_bos_veritabaninda_hata_vermez(monkeypatch):
    schema = f"koruma_{uuid.uuid4().hex[:8]}"
    admin = psycopg2.connect(TEST_DB)
    admin.autocommit = True
    admin.cursor().execute(f"CREATE SCHEMA {schema}")
    sep = "&" if "?" in TEST_DB else "?"
    monkeypatch.setenv("DATABASE_URL", f"{TEST_DB}{sep}options=-csearch_path%3D{schema}")
    try:
        results = []
        threads = [threading.Thread(target=lambda: results.append(db.init_db())) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert results == [True] * 8
    finally:
        admin.cursor().execute(f"DROP SCHEMA {schema} CASCADE")
        admin.close()


def test_gec_gelen_hata_hazir_bayragini_geri_cevirmez():
    shared = {"ready": False, "error": None}

    def late_failure():
        # B'nin hazırlığı sürerken A başlayıp başarıyla bitirir; B'nin hatası sonra gelir.
        assert db.prepare_schema(shared, {}, "v", lambda: True, lambda: None, now=0)
        return False

    assert db.prepare_schema(shared, {}, "v", late_failure, lambda: "geç hata", now=0)
    assert shared == {"ready": True, "error": None}


def test_hazirlik_basarisiz_oturum_sonra_yeniden_dener():
    shared, session, calls = {"ready": False, "error": None}, {}, []

    def init():
        calls.append(1)
        return len(calls) > 1

    assert not db.prepare_schema(shared, session, "v", init, lambda: "uyku", now=0)
    assert not db.prepare_schema(shared, session, "v", init, lambda: "uyku", now=10)
    assert db.prepare_schema(shared, session, "v", init, lambda: None, now=31)
    assert len(calls) == 2 and shared == {"ready": True, "error": None}
    # Eski biçimdeki oturum durumu ya da yeni şema sürümü süre beklemeden dener.
    for attempted in ("eski-metin", ("eski-surum", 0)):
        tried = []
        db.prepare_schema({"ready": False}, {"_db_init_attempted": attempted}, "v",
                          lambda: tried.append(1) or False, lambda: "e", now=1)
        assert tried == [1]


# --- Madde 6: kod hatası "veritabanı hatası" gibi gizlenmez -----------------

def test_kod_hatasi_yukari_iletilir_db_hatasi_false_doner():
    schedule = _frame("Ali")
    with patch.object(db.psycopg2, "connect", side_effect=RuntimeError("kod hatası")):
        with pytest.raises(RuntimeError):
            db.save_schedule(NAME, 2026, 10, ["Ali", "Ayşe"], schedule, expected_version=db.FORCE)
        with pytest.raises(RuntimeError):
            db.save_app_settings({"a": 1}, expected_version=db.FORCE)
    with patch.object(db.psycopg2, "connect", side_effect=psycopg2.OperationalError("uyku")):
        assert db.save_schedule(NAME, 2026, 10, ["Ali", "Ayşe"], schedule, expected_version=db.FORCE) is False
        assert db.save_app_settings({"a": 1}, expected_version=db.FORCE) is False


def test_beklenmeyen_hata_izi_gunluge_sifresiz_yazilir(capsys):
    try:
        raise ValueError("postgresql://kim:gizli@sunucu/db")
    except ValueError as exc:
        db.log_unexpected("Deneme", exc)
    out = capsys.readouterr().out
    assert "Traceback" in out and "ValueError" in out
    assert "gizli" not in out and "kim:***@" in out


# --- Uygulama akışı (nobet.py): sürüm yükten gelir, hatalar ayrı gösterilir ---

def _app(save_schedule, load_schedule=None, load_app_settings=None, save_app_settings=None,
         team="Ali, Ayşe, Can, Deniz, Ece", **extra):
    import streamlit
    from streamlit.testing.v1 import AppTest

    def no_schedule(*args, version_out=None, **kwargs):
        if version_out is not None:
            version_out["version"] = "h:yuklenen"
        return None, None, None, None, None

    fakes = dict(
        init_db=lambda: True, get_last_error=lambda: None, is_database_configured=lambda: False,
        SCHEMA_VERSION="koruma-test",
        load_app_settings=load_app_settings or (lambda **k: None),
        save_app_settings=save_app_settings or (lambda *a, **k: True),
        load_schedule=load_schedule or no_schedule, save_schedule=save_schedule,
        load_auto_saved_assignments=lambda *a, **k: ({}, {}), list_backup_snapshots=lambda **k: [],
        get_share_link=lambda *a: None, list_team_requests=lambda *a, **k: [],
        ping_db=lambda: True,
    )
    fakes.update(extra)
    # create=True: aynı test eski koda karşı da çalışabilsin (orada başarısız olur).
    patches = [patch.dict("os.environ", {"DATABASE_URL": ""}), patch.multiple(db, create=True, **fakes)]
    for item in patches:
        item.start()
    streamlit.cache_resource.clear()
    at = AppTest.from_file(str(Path(__file__).with_name("nobet.py")), default_timeout=120)
    if team is not None:
        at.session_state["isimler_text"] = team
    at.session_state["storage_init"] = {}
    at.session_state["yil_val"], at.session_state["ay_val"] = 2026, 10
    return at, patches


def _simulate(at):
    at.run()
    next(b for b in at.button if b.label == "⚡ Simülasyon").click()
    at.run()
    assert [e.value for e in at.exception] == []


def _stop(patches):
    import streamlit
    for item in reversed(patches):
        item.stop()
    streamlit.cache_resource.clear()


def test_uygulama_kayitta_yuklemeyle_okunan_surumu_kullanir():
    seen = {}

    def save(*args, expected_version=None, **kwargs):
        seen["schedule"] = expected_version
        return True

    def load_settings(owner="", raise_on_error=False, version_out=None, corrupt_out=None):
        version_out["version"] = "h:ayar"
        return None

    def save_settings(*args, expected_version=None, **kwargs):
        seen["settings"] = expected_version
        return True

    at, patches = _app(save, load_app_settings=load_settings, save_app_settings=save_settings)
    try:
        _simulate(at)
    finally:
        _stop(patches)
    assert seen == {"schedule": "h:yuklenen", "settings": "h:ayar"}


def test_uygulama_surum_bilinmezse_kaydi_durdurur_ve_soyler():
    def unknown(*args, **kwargs):
        raise db.VersionUnknown("x")

    at, patches = _app(unknown)
    try:
        _simulate(at)
        message = at.session_state["persistence_error"]
    finally:
        _stop(patches)
    assert "sürümü okunamadı" in message
    assert "veritabanına yazılamıyor" not in message


def test_uygulama_kod_hatasini_baglanti_hatasi_gibi_gostermez():
    def bug(*args, **kwargs):
        raise TypeError("Object of type int64 is not JSON serializable")

    at, patches = _app(bug)
    try:
        _simulate(at)
        message = at.session_state["persistence_error"]
    finally:
        _stop(patches)
    assert "beklenmeyen bir program hatası (TypeError)" in message
    assert "veritabanına yazılamıyor" not in message


# --- İnceleyici şartları: kilit, anlık görüntü, kilit bekleme süresi -----------

class _SlowInsertCursor(psycopg2.extensions.cursor):
    """İlk INSERT'ten sonra (işlem henüz commit edilmeden) bir süre bekler."""
    target = None
    inside = None

    def execute(self, sql, args=None):
        result = super().execute(sql, args)
        if self.target in str(sql) and not self.inside.is_set():
            self.inside.set()
            time.sleep(1.0)
        return result


@needs_db
def test_eszamanli_ilk_ayar_kaydinda_ikincisi_cakisir(owner, monkeypatch):
    inside = threading.Event()
    slow = type("Slow", (_SlowInsertCursor,), {"target": "INSERT INTO app_settings", "inside": inside})
    monkeypatch.setattr(db, "_connect", lambda: psycopg2.connect(TEST_DB, cursor_factory=slow))
    results = {}

    def save(tag, value):
        try:
            results[tag] = db.save_app_settings({"team": value}, owner=owner, expected_version="", writer=tag)
        except db.SaveConflict:
            results[tag] = "çakışma"

    a = threading.Thread(target=save, args=("telefon", "Ali"))
    a.start()
    assert inside.wait(5)
    b = threading.Thread(target=save, args=("masaüstü", "Ayşe"))
    b.start()
    a.join()
    b.join()
    assert results == {"telefon": True, "masaüstü": "çakışma"}
    assert db.load_app_settings(owner=owner) == {"team": "Ali"}


@needs_db
def test_cizelge_ve_surum_ayni_anlik_goruntuden_okunur(owner, monkeypatch):
    out = {}
    assert _save(owner, "Ali", expected_version="", version_out=out)
    original = db._schedule_content_version
    state = {"busy": False}

    def write_in_between(cur, schedule_id):
        version = original(cur, schedule_id)
        if not state["busy"]:
            # Sürüm okunduktan hemen sonra telefon ayı kaydeder.
            state["busy"] = True
            assert _save(owner, "Ayşe", expected_version=out["version"], writer="telefon")
        return version

    monkeypatch.setattr(db, "_schedule_content_version", write_in_between)
    box = {"version": None}
    _, schedule, *_ = db.load_schedule(NAME, 2026, 10, raise_on_error=True, owner=owner, version_out=box)
    monkeypatch.setattr(db, "_schedule_content_version", original)
    # Okunan içerik okunan sürüme ait olmalı: ikisi de telefondan önceki hâl.
    assert box["version"] == out["version"]
    assert [p for p in schedule.index if schedule.at[p, "1 Per"]] == ["Ali"]
    with pytest.raises(db.SaveConflict):
        _save(owner, "Ali", expected_version=box["version"])


@needs_db
@pytest.mark.parametrize("what", ["schedule", "settings"])
def test_kilit_beklemesi_sinirli_kayit_takilmaz(owner, monkeypatch, what):
    assert _save(owner, "Ali", expected_version=db.FORCE)
    assert db.save_app_settings({"a": 1}, owner=owner, expected_version=db.FORCE)
    monkeypatch.setattr(db, "LOCK_TIMEOUT", "300ms", raising=False)
    holder = psycopg2.connect(TEST_DB)
    try:
        with holder.cursor() as cur:
            # Kaydın ortasında ölmüş bir istemci kilidi tutuyor gibi.
            if what == "schedule":
                cur.execute("SELECT id FROM schedules WHERE owner = %s FOR UPDATE", (owner,))
            else:
                cur.execute("SELECT key FROM app_settings WHERE key = %s FOR UPDATE", (db.settings_key(owner),))
        result = {}

        def save():
            if what == "schedule":
                result["ok"] = _save(owner, "Ayşe", expected_version=db.FORCE)
            else:
                result["ok"] = db.save_app_settings({"a": 2}, owner=owner, expected_version=db.FORCE)

        worker = threading.Thread(target=save, daemon=True)
        worker.start()
        worker.join(5)
        finished = not worker.is_alive()
    finally:
        holder.rollback()
        holder.close()
    worker.join(5)
    assert finished, "kayıt kilidi beklerken takıldı"
    assert result["ok"] is False


# --- İnceleyici şartı 1: Neon'un ilk bağlantısı düşerse ------------------------

def test_baglanti_hatasinda_bir_kez_yeniden_denenir(monkeypatch):
    monkeypatch.setattr(db, "CONNECT_RETRY_DELAY", 0, raising=False)
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise psycopg2.OperationalError("Neon uyanıyor")
        return "bağlantı"

    with patch.object(db.psycopg2, "connect", flaky):
        assert db._connect() == "bağlantı"
    assert len(calls) == 2

    calls.clear()
    with patch.object(db.psycopg2, "connect", side_effect=psycopg2.OperationalError("uyku")) as down:
        with pytest.raises(psycopg2.OperationalError):
            db._connect()
    assert down.call_count == 2          # bir kez yeniden dener, sonsuza kadar değil


TEAM = "Ali, Ayşe, Can, Deniz, Ece"
_TROUBLE = ("okunamadı", "ulaşılamıyor", "bağlanılamadı", "kaydedilemedi", "yazılamıyor")


def _messages(at):
    return [m.value for m in list(at.warning) + list(at.error)]


def _trouble(at):
    return [m for m in _messages(at) if any(word in m for word in _TROUBLE)]


@pytest.fixture
def app_db(monkeypatch):
    """Ayrı şemada gerçek veritabanı; nobet.py girişsiz modda buna bağlanır."""
    import streamlit
    schema = f"uyanma_{uuid.uuid4().hex[:8]}"
    admin = psycopg2.connect(TEST_DB)
    admin.autocommit = True
    admin.cursor().execute(f"CREATE SCHEMA {schema}")
    sep = "&" if "?" in TEST_DB else "?"
    url = f"{TEST_DB}{sep}options=-csearch_path%3D{schema}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("GIRISSIZ_KULLANIM", "1")
    monkeypatch.setattr(db, "CONNECT_RETRY_DELAY", 0.05, raising=False)
    streamlit.cache_resource.clear()
    try:
        yield url
    finally:
        streamlit.cache_resource.clear()
        admin.cursor().execute(f"DROP SCHEMA {schema} CASCADE")
        admin.close()


def _session():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).with_name("nobet.py")), default_timeout=120)
    at.session_state["storage_init"] = {}
    at.session_state["yil_val"], at.session_state["ay_val"] = 2026, 10
    return at


def _seed_app():
    at = _session()
    at.session_state["isimler_text"] = TEAM
    _simulate(at)
    at.run()
    assert _trouble(at) == []
    assert db.load_app_settings()["team"] == TEAM
    assert db.load_schedule(NAME, 2026, 10)[1] is not None


@needs_db
@pytest.mark.parametrize("dropped", [1, 2, 3, 4])
def test_neon_ilk_baglantilardan_biri_duserse_uygulama_normal_acilir(app_db, dropped):
    _seed_app()
    real = psycopg2.connect
    calls = [0]

    def flaky(*args, **kwargs):
        calls[0] += 1
        if calls[0] == dropped:
            raise psycopg2.OperationalError("Neon uyanıyor (deneme)")
        return real(*args, **kwargs)

    with patch.object(db.psycopg2, "connect", flaky):
        at = _session()
        at.run()
        assert at.session_state["isimler_text"] == TEAM      # ekip boş açılmaz
        assert _trouble(at) == []
        # Sürümler biliniyor: kayıt çakışma denetimiyle yapılabilir.
        assert at.session_state["settings_version"]["version"]
        assert at.session_state["schedule_version"]["version"]
        _simulate(at)
        at.run()
        assert _trouble(at) == []
        assert _saved_ok(at)
    assert at.session_state["schedule_version"]["version"] == db.get_schedule_version(NAME, 2026, 10)


@needs_db
def test_neon_uyanmazsa_tek_mesaj_sonra_kendiliginden_yuklenir(app_db, monkeypatch):
    _seed_app()
    calls = [0]

    def down(*args, **kwargs):
        calls[0] += 1
        raise psycopg2.OperationalError("Neon uyanmadı (deneme)")

    with patch.object(db.psycopg2, "connect", down):
        at = _session()
        at.run()
        opened_with = calls[0]
        trouble = _trouble(at)
        team = at.session_state["isimler_text"] if "isimler_text" in at.session_state else ""
    # Tek uyandırma bağlantısı + bir yeniden deneme; diğer okumalar denenmez.
    assert opened_with <= 2
    assert len(trouble) == 1 and "ulaşılamıyor" in trouble[0]
    assert "okunamadı" not in trouble[0]
    assert not team

    # Neon uyandı, yeniden deneme süresi doldu: aynı oturumda ekip ve ay yüklenir.
    monkeypatch.setattr(db, "DB_RETRY_SECONDS", 0, raising=False)
    at.run()
    assert at.session_state["isimler_text"] == TEAM
    assert _trouble(at) == []
    assert at.session_state["schedule_version"]["version"] == db.get_schedule_version(NAME, 2026, 10)
    _simulate(at)
    at.run()
    assert _saved_ok(at)
    assert at.session_state["schedule_version"]["version"] == db.get_schedule_version(NAME, 2026, 10)


def test_sema_hazirligi_sonra_basarili_olunca_ayarlar_yuklenir(monkeypatch):
    inits = [False, True]
    loads = []

    def load_settings(owner="", raise_on_error=False, version_out=None, corrupt_out=None):
        loads.append(1)
        if version_out is not None:
            version_out["version"] = "h:ayar"
        return {"team": TEAM}

    at, patches = _app(lambda *a, **k: True, load_app_settings=load_settings, team=None,
                       init_db=lambda: inits.pop(0), get_last_error=lambda: "uyku")
    try:
        at.run()
        assert not (at.session_state["isimler_text"] if "isimler_text" in at.session_state else "")
        assert any("bağlanılamadı" in m for m in _messages(at))
        monkeypatch.setattr(db, "DB_RETRY_SECONDS", 0, raising=False)
        at.run()
        team = at.session_state["isimler_text"]
        trouble = _trouble(at)
        version = at.session_state["settings_version"]["version"]
    finally:
        _stop(patches)
    assert loads == [1]
    assert team == TEAM and trouble == [] and version == "h:ayar"


def test_baglanti_yokken_degisen_ayarlarin_uzerine_yuklenmez(monkeypatch):
    inits = [False, True]

    def load_settings(owner="", raise_on_error=False, version_out=None, corrupt_out=None):
        if version_out is not None:
            version_out["version"] = "h:ayar"
        return {"team": TEAM}

    at, patches = _app(lambda *a, **k: True, load_app_settings=load_settings, team=None,
                       init_db=lambda: inits.pop(0), get_last_error=lambda: "uyku")
    try:
        at.run()
        at.session_state["isimler_text"] = "Elle, Yazılan"
        monkeypatch.setattr(db, "DB_RETRY_SECONDS", 0, raising=False)
        at.run()
        team = at.session_state["isimler_text"]
        messages = _messages(at)
    finally:
        _stop(patches)
    assert team == "Elle, Yazılan"
    assert any("yenileyin" in m and "yüklenmedi" in m for m in messages)


# --- İnceleyici şartı 3: sürümü bilinmeyen ay "yine de geç" ile değiştirilebilir ---

def _month(at):
    return at.session_state["ay_val"]


def _pick_month(at, month):
    """Ayı seçer; geri alınırsa uyarı bu çalışmada görünür, mesajları döndürür."""
    if not any(box.key == "ay_input_direct" for box in at.selectbox):
        at.button(key="ayarlar_ac_kapa").click().run()  # Ayarlar ekip varken kapalı başlar
    at.selectbox(key="ay_input_direct").set_value(month).run()
    messages = _messages(at)
    at.run()
    return messages


def _saved_ok(at):
    return ("persistence_error" not in at.session_state) or at.session_state["persistence_error"] is None


def test_otomatik_kayitta_surum_bilinmezse_ay_onayla_degisir():
    def unknown(*args, **kwargs):
        raise db.VersionUnknown("x")

    at, patches = _app(unknown)
    try:
        _simulate(at)
        notice = _pick_month(at, 11)
        first = _month(at)
        _pick_month(at, 11)
        second = (_month(at), _messages(at))
    finally:
        _stop(patches)
    assert first == 10
    assert any("sürümü okunamadı" in m and "atılır" in m for m in notice), notice
    assert second[0] == 11
    # Önceki ayın kayıt uyarısı yeni ayda kalmaz.
    assert not any("sürümü okunamadı" in m for m in second[1]), second[1]


def test_ay_degisiminde_surum_bilinmezse_ay_onayla_degisir():
    mode = {"unknown": False}

    def save(*args, version_out=None, **kwargs):
        if mode["unknown"]:
            raise db.VersionUnknown("x")
        if version_out is not None:
            version_out["version"] = "h:yeni"
        return True

    at, patches = _app(save)
    try:
        _simulate(at)
        assert _saved_ok(at)
        # Kaydedilmemiş değişiklik var, ay değişiminde kayıt sürümü bilinmiyor.
        mode["unknown"] = True
        at.session_state["last_saved_fingerprint"] = "kaydedilmemiş"
        notice = _pick_month(at, 11)
        first = _month(at)
        _pick_month(at, 11)
        second = _month(at)
    finally:
        _stop(patches)
    assert first == 10
    assert any("sürümü okunamadı" in m and "atılır" in m for m in notice), notice
    assert second == 11


# --- İyileştirici madde 2: paylaşım linki "geçersiz" değil "ulaşılamıyor" der ---

def _share_app(**fakes):
    at, patches = _app(lambda *a, **k: True, team=None, **fakes)
    at.query_params["paylas"] = "abc123"
    return at, patches


def test_paylasim_linki_veritabani_yokken_gecersiz_demez():
    def broken(token, raise_on_error=False):
        if raise_on_error:
            raise psycopg2.OperationalError("Neon uyanmadı")
        return None

    for fakes in ({"resolve_share_token": broken},
                  {"init_db": lambda: False, "get_last_error": lambda: "uyku",
                   "resolve_share_token": broken}):
        at, patches = _share_app(**fakes)
        try:
            at.run()
            messages = _messages(at)
        finally:
            _stop(patches)
        assert any("ulaşılamıyor" in m and "yenileyin" in m for m in messages), messages
        assert not any("geçersiz" in m for m in messages)


def test_paylasim_linki_gercekten_yoksa_gecersiz_der():
    at, patches = _share_app(resolve_share_token=lambda token, raise_on_error=False: None)
    try:
        at.run()
        messages = _messages(at)
    finally:
        _stop(patches)
    assert any("geçersiz" in m for m in messages)


# --- C4: iki kayıt yeri (bu ay, ay değişimi) ortak yardımcıdan geçer; sonuç aynı kalır ---

def test_bu_ayda_cakisma_kaydi_durdurur_ve_kimin_degistirdigini_soyler():
    def conflict(*args, **kwargs):
        raise db.SaveConflict(args[0], "Telefon", datetime(2026, 10, 1, 9, 5))

    at, patches = _app(conflict)
    try:
        _simulate(at)
        message = at.session_state["persistence_error"]
        reason = at.session_state["autosave_block_reason"]
    finally:
        _stop(patches)
    assert "başka bir yerde" in message and "Telefon" in message
    assert reason == "cakisma"


@pytest.mark.parametrize("kind,month,reason", [("kilitli", 11, None), ("cakisma", 10, "cakisma")])
def test_ay_degisiminde_kilitli_ay_gecer_cakismada_kalir(kind, month, reason):
    calls = []

    def save(*args, version_out=None, **kwargs):
        calls.append(args[:3])
        if mode["kind"] == "kilitli":
            raise db.MonthLocked(args[1], args[2])
        if mode["kind"] == "cakisma":
            raise db.SaveConflict(args[0], "Telefon", None)
        return True

    mode = {"kind": None}
    at, patches = _app(save)
    try:
        _simulate(at)
        mode["kind"] = kind
        calls.clear()
        at.session_state["last_saved_fingerprint"] = "kaydedilmemiş"
        _pick_month(at, 11)
        state = (_month(at), at.session_state["autosave_block_reason"])
    finally:
        _stop(patches)
    assert state == (month, reason)
    assert calls[0] == ("Otomatik_2026_10", 2026, 10)  # önce çıkılan ay kaydedilir
