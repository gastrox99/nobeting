# -*- coding: utf-8 -*-
"""Bozuk kayıttan kurtarma (Mimar raporu 20 Ekim 2026, madde 3; İnceleyici bulgu 2).

Bir ayın kaydında tek bir alan (tercihler, kilitler/saklı kişiler, görev yerleri)
ya da ekip/ayar kaydı okunamıyorsa:
- alan sessizce boş sayılıp ilk dokunuşta kalıcı olarak silinmez,
- sağlam kısımlar (nöbetler) yine gösterilir,
- kullanıcı ayrı ve doğru bir mesaj görür, açık onayla bozuk bölümü boşaltıp devam edebilir.

Gerçek PostgreSQL gerektirenler TEST_DATABASE_URL yoksa atlanır.
"""
import json
import os
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import psycopg2
import pytest

import db

TEST_DB = os.environ.get("TEST_DATABASE_URL")
needs_db = pytest.mark.skipif(not TEST_DB, reason="TEST_DATABASE_URL tanımlı değil")
NAME = "Otomatik_2026_10"
TEAM = "Ali, Ayşe, Can, Deniz, Ece"
BOZUK = "{bozuk"
# Alan -> kullanıcının gördüğü ad.
FIELDS = {
    "pref_json": "Tercihler",
    "settings_json": "Kilitler ve saklı kişiler",
    "rows_liste_json": "Görev yerleri",
}


@pytest.fixture
def owner(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", TEST_DB)
    assert db.init_db(), db.get_last_error()
    name = f"bozuk-test-{uuid.uuid4().hex[:10]}@example.com"
    yield name
    monkeypatch.setenv("DATABASE_URL", TEST_DB)
    conn = db._connect()
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM schedules WHERE owner = %s", (name,))
        cur.execute("DELETE FROM app_settings WHERE key = %s", (db.settings_key(name),))
    conn.close()


def _seed(owner=""):
    frame = pd.DataFrame(False, index=["Ali", "Ayşe"], columns=["1 Per", "2 Cum"])
    frame.at["Ali", "1 Per"] = True
    frame.at["Ayşe", "2 Cum"] = True
    pref = pd.DataFrame(0, index=["Ali", "Ayşe"], columns=["1 Per", "2 Cum"])
    pref.at["Ali", "2 Cum"] = 1
    assert db.save_schedule(
        NAME, 2026, 10, ["Ali", "Ayşe"], frame, owner=owner, expected_version=db.FORCE,
        pref_df=pref, settings_dict={"locked": [["Ali", "1 Per"]], "hidden": {}},
        rows_liste=[{"Gün": "1 Per", "Nöbet": "Ali"}],
    )


def _corrupt(column, value=BOZUK, owner=""):
    conn = db._connect()
    with conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE schedules SET {column} = %s WHERE owner = %s AND name = %s",  # noqa: S608 (sabit alan adı)
            (value, owner, NAME),
        )
    conn.close()


def _raw(column, owner=""):
    conn = db._connect()
    with conn.cursor() as cur:
        cur.execute(f"SELECT {column} FROM schedules WHERE owner = %s AND name = %s", (owner, NAME))  # noqa: S608
        value = cur.fetchone()[0]
    conn.close()
    return value


def _raw_settings(owner=""):
    conn = db._connect()
    with conn.cursor() as cur:
        cur.execute("SELECT value FROM app_settings WHERE key = %s", (db.settings_key(owner),))
        row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def _set_raw_settings(value, owner=""):
    conn = db._connect()
    with conn, conn.cursor() as cur:
        cur.execute("UPDATE app_settings SET value = %s WHERE key = %s", (value, db.settings_key(owner)))
    conn.close()


# --- db.py: bozuk alan ayırt edilir ------------------------------------------

@needs_db
@pytest.mark.parametrize("column,value", [
    ("pref_json", BOZUK),
    ("settings_json", BOZUK),
    ("settings_json", "[1, 2]"),          # JSON ama sözlük değil
    ("rows_liste_json", BOZUK),
    ("rows_liste_json", '{"a": 1}'),      # JSON ama satır listesi değil
])
def test_bozuk_alan_etkilesimli_okumada_sessizce_bos_sayilmaz(owner, column, value):
    _seed(owner)
    _corrupt(column, value, owner)
    with pytest.raises(db.CorruptRecord) as error:
        db.load_schedule(NAME, 2026, 10, raise_on_error=True, owner=owner)
    assert error.value.fields == [column]


@needs_db
@pytest.mark.parametrize("column", list(FIELDS))
def test_bozuk_alan_bildirilir_saglam_kisimlar_ve_surum_gelir(owner, column):
    _seed(owner)
    _corrupt(column, owner=owner)
    corrupt, box = [], {}
    team, schedule, pref, settings, rows = db.load_schedule(
        NAME, 2026, 10, raise_on_error=True, owner=owner, version_out=box, corrupt_out=corrupt,
    )
    assert corrupt == [column]
    assert team == ["Ali", "Ayşe"]
    assert bool(schedule.at["Ali", "1 Per"]) and bool(schedule.at["Ayşe", "2 Cum"])
    assert (pref is None) == (column == "pref_json")
    assert (settings is None) == (column == "settings_json")
    assert (rows is None) == (column == "rows_liste_json")
    # Sürüm ham içerikten: bozuk alan değişmedikçe kayıt bu sürümle yapılabilir.
    assert box["version"] == db.get_schedule_version(NAME, 2026, 10, owner=owner)


@needs_db
def test_paylasim_okumasi_eskisi_gibi_bos_alan_gorur(owner):
    _seed(owner)
    _corrupt("pref_json", owner=owner)
    _, schedule, pref, *_ = db.load_schedule(NAME, 2026, 10, owner=owner)
    assert pref is None and bool(schedule.at["Ali", "1 Per"])


@needs_db
@pytest.mark.parametrize("value", [BOZUK, "[1, 2]", '"metin"'])
def test_bozuk_ayar_kaydi_ayirt_edilir(owner, value):
    assert db.save_app_settings({"team": "Ali, Ayşe"}, owner=owner, expected_version="")
    _set_raw_settings(value, owner)
    with pytest.raises(db.CorruptRecord) as error:
        db.load_app_settings(owner=owner, raise_on_error=True)
    assert error.value.fields == ["app_settings"]
    corrupt, box = [], {}
    assert db.load_app_settings(owner=owner, raise_on_error=True, version_out=box, corrupt_out=corrupt) is None
    assert corrupt == ["app_settings"]
    assert box["version"] == db._settings_content_version(value)
    assert db.load_app_settings(owner=owner) is None   # paylaşım sayfası: eskisi gibi


# --- Uygulama akışı (sahte veritabanı): mesaj ve onaylı düğme ---------------

def _app(load_schedule, save_schedule, load_app_settings=None, save_app_settings=None, team=TEAM, **extra):
    import streamlit
    from streamlit.testing.v1 import AppTest

    fakes = dict(
        init_db=lambda: True, get_last_error=lambda: None, is_database_configured=lambda: False,
        SCHEMA_VERSION="bozuk-test",
        load_app_settings=load_app_settings or (lambda **k: None),
        save_app_settings=save_app_settings or (lambda *a, **k: True),
        load_schedule=load_schedule, save_schedule=save_schedule,
        load_auto_saved_assignments=lambda *a, **k: ({}, {}), list_backup_snapshots=lambda **k: [],
        get_share_link=lambda *a: None, list_team_requests=lambda *a, **k: [],
        ping_db=lambda: True,
    )
    fakes.update(extra)
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


def _stop(patches):
    import streamlit
    for item in reversed(patches):
        item.stop()
    streamlit.cache_resource.clear()


def _messages(at):
    return [m.value for m in list(at.warning) + list(at.error)]


def _button(at, label):
    return next(b for b in at.button if b.label == label)


RECOVER = "Okunamayan bölümü boşaltıp devam et"


def _corrupt_month_loader(columns):
    """Ali'nin 1 Ekim nöbeti sağlam, tercih alanı bozuk bir ay."""
    def load(name, year, month, raise_on_error=False, owner="", version_out=None, corrupt_out=None, **kwargs):
        if corrupt_out is None:
            # Eski çağrı biçimi: bozuk alan sessizce boş sayılır (düzeltmeden önceki hâl).
            pass
        else:
            corrupt_out.append("pref_json")
        if version_out is not None:
            version_out["version"] = "h:bozuk"
        frame = pd.DataFrame(False, index=["Ali", "Ayşe"], columns=columns)
        frame.iloc[0, 0] = True
        return ["Ali", "Ayşe"], frame, None, {"locked": [], "hidden": {}}, None
    return load


def _flip_cell(at, row=1, col=1):
    changed = at.session_state["schedule_bool"].copy()
    changed.iloc[row, col] = not bool(changed.iloc[row, col])
    at.session_state["schedule_bool"] = changed


def _set_preference(at, value=1):
    schedule = at.session_state["schedule_bool"]
    pref = at.session_state["pref_df"] if "pref_df" in at.session_state else None
    if pref is None:
        pref = pd.DataFrame(0, index=schedule.index, columns=schedule.columns)
    pref = pref.copy()
    pref.iloc[0, 1] = value
    at.session_state["pref_df"] = pref
    return pref.columns[1]


def _october_columns():
    import nobet_core
    days = nobet_core.build_gun_detaylari(2026, 10, 31, set())
    return list(days)


def test_uygulama_bozuk_alani_ezmez_nobetleri_gosterir_ve_dogru_mesaji_verir():
    saves = []

    def save(*args, expected_version=None, pref_df=None, **kwargs):
        saves.append((expected_version, pref_df))
        return True

    at, patches = _app(_corrupt_month_loader(_october_columns()), save, team="Ali, Ayşe")
    try:
        at.run()
        assert [e.value for e in at.exception] == []
        schedule = at.session_state["schedule_bool"]
        messages = _messages(at)
        # Kullanıcı nöbet ve tercih değiştirir: bozuk ayda kayıt yapılmamalı.
        _flip_cell(at)
        _set_preference(at)
        at.run()
        assert [e.value for e in at.exception] == []
        saved_before_confirm = list(saves)
        assert RECOVER in [b.label for b in at.button]
    finally:
        _stop(patches)
    assert bool(schedule.iloc[0, 0])                         # sağlam nöbet gösteriliyor
    notice = [m for m in messages if "okunamayan bir bölüm var" in m]
    assert notice and "Tercihler" in notice[0] and "Nöbetler gösteriliyor" in notice[0]
    assert "yedek" in notice[0].lower()
    assert not any("bağlantısını kontrol" in m for m in messages)
    assert saved_before_confirm == []                        # bozuk alanın üzerine yazılmadı


def test_uygulama_onaylaninca_bozuk_alan_bos_kabul_edilip_bilinen_surumle_yazilir():
    saves = []

    def save(*args, expected_version=None, pref_df=None, **kwargs):
        saves.append(expected_version)
        prefs.append(pref_df)
        return True

    prefs = []
    at, patches = _app(_corrupt_month_loader(_october_columns()), save, team="Ali, Ayşe")
    try:
        at.run()
        # Tercih modu (varsayılan): kayıt normalde mod değişene kadar ertelenir;
        # onaydan sonra ertelenmemeli, bu arada girilen tercih de yazılmalı.
        assert at.session_state["edit_mode"] == "tercih"
        column = _set_preference(at)
        at.run()
        assert saves == []
        button = _button(at, RECOVER)
        assert button.disabled                               # açık onay olmadan basılamaz
        next(c for c in at.checkbox if "Tercihler" in c.label).check()
        at.run()
        _button(at, RECOVER).click()
        at.run()
        assert [e.value for e in at.exception] == []
        messages = _messages(at)
        labels = [b.label for b in at.button]
        blocked = at.session_state["autosave_blocked_period"]
        schedule = at.session_state["schedule_bool"]
    finally:
        _stop(patches)
    assert saves == ["h:bozuk"]                              # hemen, yüklemedeki sürümle
    assert prefs[0].at["Ali", column] == 1                   # bekleyen tercih de yazıldı
    assert blocked is None
    assert RECOVER not in labels
    assert not any("okunamayan bir bölüm var" in m for m in messages)
    assert bool(schedule.iloc[0, 0])                         # nöbetler korundu


def test_uygulama_bozuk_ayar_kaydinda_ekip_bos_gelip_uzerine_yazilmaz():
    saved = []

    def load_settings(owner="", raise_on_error=False, version_out=None, corrupt_out=None):
        if version_out is not None:
            version_out["version"] = "h:bozuk-ayar"
        if corrupt_out is not None:
            corrupt_out.append("app_settings")
            return None
        if raise_on_error:
            raise json.JSONDecodeError("bozuk", "{bozuk", 1)
        return None

    def save_settings(settings, owner="", expected_version=None, **kwargs):
        saved.append((settings.get("team"), expected_version))
        return True

    def no_schedule(*args, version_out=None, **kwargs):
        if version_out is not None:
            version_out["version"] = ""
        return None, None, None, None, None

    at, patches = _app(no_schedule, lambda *a, **k: True, load_app_settings=load_settings,
                       save_app_settings=save_settings, team=None)
    try:
        at.run()
        messages = _messages(at)
        at.session_state["isimler_text"] = "Yeni, Kişi"     # kullanıcı ekip yazar
        at.run()
        saved_before = list(saved)
        next(c for c in at.checkbox if "Ekip ve ayarlar" in c.label).check()
        at.run()
        _button(at, RECOVER).click()
        at.run()
        assert [e.value for e in at.exception] == []
    finally:
        _stop(patches)
    notice = [m for m in messages if "okunamayan bir bölüm var" in m]
    assert notice and "Ekip ve ayarlar" in notice[0]
    assert not any("Sayfayı yenileyin" in m for m in notice)
    assert saved_before == []                                # bozuk kayıt ezilmedi
    assert saved == [("Yeni, Kişi", "h:bozuk-ayar")]          # onaydan sonra bilinen sürümle


# --- Gerçek veritabanıyla uçtan uca: bozuk satır ------------------------------

@pytest.fixture
def app_db(monkeypatch):
    """Ayrı şemada gerçek veritabanı; nobet.py girişsiz modda buna bağlanır."""
    import streamlit
    schema = f"bozuk_{uuid.uuid4().hex[:8]}"
    admin = psycopg2.connect(TEST_DB)
    admin.autocommit = True
    admin.cursor().execute(f"CREATE SCHEMA {schema}")
    sep = "&" if "?" in TEST_DB else "?"
    url = f"{TEST_DB}{sep}options=-csearch_path%3D{schema}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("GIRISSIZ_KULLANIM", "1")
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


def _simulate(at):
    next(b for b in at.button if b.label == "⚡ Simülasyon").click()
    at.run()
    assert [e.value for e in at.exception] == []


def _seed_app():
    at = _session()
    at.session_state["isimler_text"] = TEAM
    at.run()
    _simulate(at)
    at.run()
    _, schedule, pref, *_ = db.load_schedule(NAME, 2026, 10)
    assert schedule is not None and schedule.values.any()
    # Tercih alanı dolu olsun (bozulduğunda kaybolacak gerçek veri).
    conn = db._connect()
    with conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE schedules SET pref_json = %s WHERE name = %s",
            (pd.DataFrame(1, index=["Ali"], columns=[schedule.columns[0]]).to_json(), NAME),
        )
    conn.close()
    return schedule


@needs_db
def test_gercek_db_bozuk_tercih_ilk_dokunusta_silinmez_onayla_temizlenir(app_db):
    saved = _seed_app()
    _corrupt("pref_json")
    assigned = {(p, c) for p in saved.index for c in saved.columns if saved.at[p, c]}

    at = _session()
    at.run()
    assert [e.value for e in at.exception] == []
    shown = at.session_state["schedule_bool"]
    assert {(p, c) for p in shown.index for c in shown.columns if shown.at[p, c]} == assigned
    messages = _messages(at)
    assert any("okunamayan bir bölüm var: Tercihler" in m for m in messages)
    assert not any("bağlantısını kontrol" in m for m in messages)

    _simulate(at)                    # ilk dokunuş
    at.run()
    assert _raw("pref_json") == BOZUK                        # sessiz veri kaybı yok

    next(c for c in at.checkbox if "Tercihler" in c.label).check()
    at.run()
    _button(at, RECOVER).click()
    at.run()
    assert [e.value for e in at.exception] == []
    cleaned = _raw("pref_json")
    assert cleaned != BOZUK
    corrupt = []
    _, schedule, *_ = db.load_schedule(NAME, 2026, 10, raise_on_error=True, corrupt_out=corrupt)
    assert corrupt == []
    assert at.session_state["autosave_blocked_period"] is None
    assert at.session_state["schedule_version"]["version"] == db.get_schedule_version(NAME, 2026, 10)


@needs_db
@pytest.mark.parametrize("value", [BOZUK, "[1, 2]"])
def test_gercek_db_bozuk_ayar_kaydi_ekip_boşken_ezilmez(app_db, value):
    _seed_app()
    _set_raw_settings(value)
    at = _session()
    at.run()
    assert [e.value for e in at.exception] == []
    assert any("okunamayan bir bölüm var: Ekip ve ayarlar" in m for m in _messages(at))
    at.session_state["isimler_text"] = "Yeni, Kişi"
    at.run()
    assert _raw_settings() == value                          # ekip boş gelip üzerine yazılmadı
    next(c for c in at.checkbox if "Ekip ve ayarlar" in c.label).check()
    at.run()
    _button(at, RECOVER).click()
    at.run()
    assert [e.value for e in at.exception] == []
    assert db.load_app_settings(raise_on_error=True)["team"] == "Yeni, Kişi"


@needs_db
@pytest.mark.parametrize("column", list(FIELDS))
def test_gercek_db_bozuk_alan_ilk_dokunusta_bos_haliyle_kalici_olmaz(app_db, column):
    """Düzeltmeden önceki hata: bozuk alan boş sayılıyor, ilk kayıt boş hâlini yazıyordu."""
    _seed_app()
    _corrupt(column)
    at = _session()
    at.run()
    _simulate(at)
    at.run()
    assert _raw(column) == BOZUK


@needs_db
def test_gercek_db_sozluk_olmayan_ayar_kaydi_ekip_yazilinca_ezilmez(app_db):
    _seed_app()
    _set_raw_settings("[1, 2]")
    at = _session()
    at.run()
    at.session_state["isimler_text"] = "Yeni, Kişi"
    at.run()
    assert _raw_settings() == "[1, 2]"


@needs_db
def test_gercek_db_bozuk_ayda_yedekle_saglam_yedegin_yerine_gecmez(app_db):
    _seed_app()
    at = _session()
    at.run()
    _button(at, "💾 Yedekle").click()
    at.run()
    [snap] = db.list_backup_snapshots()
    good = db.load_backup_snapshot(snap["id"])
    _corrupt("pref_json")

    at = _session()
    at.run()
    _button(at, "💾 Yedekle").click()
    at.run()
    assert any("yedek alınmadı" in m and "önceki yedeği geri yükleme için saklı kalır" in m for m in _messages(at))
    [snap] = db.list_backup_snapshots()
    assert db.load_backup_snapshot(snap["id"]) == good
    assert BOZUK not in good


def _switching_loader(corrupt_months):
    """Ekim/Kasım: ``corrupt_months`` içindeki aylarda tercih alanı bozuk."""
    import nobet_core

    def load(name, year, month, raise_on_error=False, owner="", version_out=None, corrupt_out=None, **kwargs):
        if version_out is not None:
            version_out["version"] = f"h:{month}"
        columns = list(nobet_core.build_gun_detaylari(year, month, pd.Period(f"{year}-{month}").days_in_month, set()))
        if month in corrupt_months:
            corrupt_out.append("pref_json")
        frame = pd.DataFrame(False, index=["Ali", "Ayşe"], columns=columns)
        frame.iloc[0, 0] = True
        return ["Ali", "Ayşe"], frame, None, {"locked": [], "hidden": {}}, None
    return load


def _notice(at):
    return [m for m in _messages(at) if "okunamayan bir bölüm var" in m]


def test_uygulama_bozuk_aydan_baska_aya_gecip_donunce_durum_dogru():
    saves = []
    corrupt = {10}

    def save(name, year, month, *args, expected_version=None, **kwargs):
        saves.append((month, expected_version))
        return True

    at, patches = _app(_switching_loader(corrupt), save, team="Ali, Ayşe")
    try:
        at.run()
        assert _notice(at)
        _button(at, "›").click()                             # Kasım: kaydı sağlam
        at.run()
        assert at.session_state["loaded_period"] == (2026, 11)
        assert _notice(at) == []
        assert at.session_state["autosave_blocked_period"] is None
        _flip_cell(at)
        at.run()
        assert saves == [(11, "h:11")]                       # Kasım normal kaydediliyor
        corrupt.clear()                                       # Ekim başka yerden düzeltildi
        _button(at, "‹").click()
        at.run()
        assert at.session_state["loaded_period"] == (2026, 10)
        notice = _notice(at)
        stale = at.session_state["schedule_corrupt"] if "schedule_corrupt" in at.session_state else None
        _flip_cell(at)
        at.run()
        assert [e.value for e in at.exception] == []
    finally:
        _stop(patches)
    assert notice == [] and stale is None                    # eski bozuk durumu kalmadı
    assert saves[-1] == (10, "h:10")


def test_uygulama_baska_ayin_bozuk_durumu_bu_ayda_gosterilmez():
    at, patches = _app(_switching_loader(set()), lambda *a, **k: True, team="Ali, Ayşe")
    try:
        at.run()
        at.session_state["schedule_corrupt"] = {"period": (2026, 9), "fields": ["pref_json"]}
        at.run()
        notice = _notice(at)
    finally:
        _stop(patches)
    assert notice == []


def test_uygulama_bozuk_ayda_yedegi_yoksa_saklı_kalir_demez():
    snapshots = []

    def export(owner=""):
        snapshots.append(owner)
        return {}, []

    at, patches = _app(_corrupt_month_loader(_october_columns()), lambda *a, **k: True, team="Ali, Ayşe",
                       export_owner_data=export, save_backup_snapshot=lambda *a, **k: snapshots.append(a))
    try:
        at.run()
        _button(at, "💾 Yedekle").click()
        at.run()
        messages = [m for m in _messages(at) if "yedek alınmadı" in m]
    finally:
        _stop(patches)
    assert snapshots == []
    assert messages and "saklı" not in messages[0]
