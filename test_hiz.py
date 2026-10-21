# -*- coding: utf-8 -*-
"""Hız düzeltmelerinin testleri (dört rollü inceleme, 20 Ekim 2026, madde 5-8).

1. Simülasyon iç döngüsü tercih/izin tablolarını önceden sözlüğe çevirir; sonuç
   eski hücre hücre okumayla birebir aynı olmalı.
2. Kayıt ve yedekten geri yükleme hücreleri tek INSERT ile yazar.
3. Ayarlar paneli kapalıyken içi çizilmez; açık/kapalı oturumda saklanır.
4. Paylaşım sayfasında ekip listesi 10 sn önbellekte, jeton denetimi önbelleksiz.
"""
import hashlib
import inspect
import os
import random
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extensions
import pytest

import db
import nobet_core
from nobet_core import build_gun_detaylari, parse_person_limits, run_scheduling_core

TEST_DB = os.environ.get("TEST_DATABASE_URL")


# --- 1. Simülasyon: önceden hesaplanan tablolar sonucu değiştirmez -----------------

# Eski iç döngü (hücre hücre pandas okuması). Referans fonksiyon, bugünkü
# run_scheduling_core kaynağında yalnız bu iki yer eski hâline döndürülerek kurulur;
# algoritmanın geri kalanı değişirse referans da onunla birlikte değişir.
_ESKI_TERCIH = '''            pref_bonus = 0
            if df_preferred is not None and p in df_preferred.index and col in df_preferred.columns:
                pref_val = df_preferred.at[p, col]
                if pref_val == 1:
                    pref_bonus = -500
                elif pref_val == 2:
                    pref_bonus = 300
'''
_YENI_TERCIH = '''            pref_bonus = pref_bonus_of.get((p, col), 0)
'''
_ESKI_IZIN = "                if df_unwanted_bool.at[k, col]:\n"
_YENI_IZIN = "                if (k, col) in unwanted_cells:\n"


def _eski_run_scheduling_core():
    source = inspect.getsource(run_scheduling_core)
    assert source.count(_YENI_TERCIH) == 1 and source.count(_YENI_IZIN) == 1, \
        "İç döngü değişti; testteki eski hâli de güncelleyin."
    source = source.replace(_YENI_TERCIH, _ESKI_TERCIH).replace(_YENI_IZIN, _ESKI_IZIN)
    namespace = dict(vars(nobet_core))
    exec(compile(source, "eski_run_scheduling_core", "exec"), namespace)
    return namespace["run_scheduling_core"]


def _senaryo(team, seed):
    rng = random.Random(seed)
    gd = build_gun_detaylari(2026, 12, 31, [31])
    cols = list(gd)
    names = [f"K{i}" for i in range(team)]
    unwanted = pd.DataFrame(False, index=names, columns=cols)
    for person in names:
        for col in rng.sample(cols, rng.randint(2, 5)):
            unwanted.at[person, col] = True
    for col in cols[8:15]:
        unwanted.at["K3", col] = True
    preferred = pd.DataFrame(False, index=names, columns=cols)
    for person in names[:5]:
        for col in rng.sample(cols, 2):
            if not unwanted.at[person, col]:
                preferred.at[person, col] = True
    limits = parse_person_limits("K0:2-3\nK1:3-4:1\nK2:0-2\nK5:4-6:2")
    kwargs = dict(
        forbidden_pairs={("K4", "K5")}, person_limits=limits, df_preferred=preferred,
        together_pairs={("K1", "K2")}, initial_last_shift={"K4": 0, "K2": -1},
        history_special={"K0": 1.0, "K1": -1.0}, history_total={"K2": 0.5}, history_holiday={"K5": 1},
        fixed_assignments={("K0", cols[3]), ("K1", cols[20])},
    )
    return names, cols, unwanted, gd, kwargs


def _tercih_bicimi(names, cols, unwanted, team, bicim):
    """Kayıtta görülen tercih biçimleri: bool, 0/1/2, NaN'lı/eksik satır-sütunlu, yok."""
    if bicim == "bool":
        return None  # senaryodaki bool tablo
    if bicim == "yok":
        return "yok"
    rng = np.random.default_rng(team)
    pref = pd.DataFrame(rng.choice([0, 0, 0, 1, 2], size=(team, len(cols))), index=names, columns=cols)
    pref = pref.where(~unwanted, 0)
    if bicim == "nan_eksik":
        pref = pref.astype(float)
        pref.iloc[0, :5] = np.nan
        pref = pref.drop(index=names[-1]).drop(columns=cols[-2:])
    return pref


def _ozet(schedule):
    return hashlib.md5(schedule.values.tobytes()).hexdigest()


@pytest.mark.parametrize("team,kisi", [(6, 1), (8, 2), (14, 2), (20, 3)])
@pytest.mark.parametrize("bicim", ["bool", "int012", "nan_eksik", "yok"])
def test_simulasyon_sonucu_eski_hucre_okumasiyla_birebir_ayni(team, kisi, bicim):
    eski = _eski_run_scheduling_core()
    names, cols, unwanted, gd, kwargs = _senaryo(team, seed=team)
    pref = _tercih_bicimi(names, cols, unwanted, team, bicim)
    if isinstance(pref, pd.DataFrame):
        kwargs = dict(kwargs, df_preferred=pref)
    elif pref == "yok":
        kwargs = dict(kwargs, df_preferred=None)
    for seed in range(3):
        sonuclar = []
        for fonk in (eski, run_scheduling_core):
            random.seed(seed)
            schedule, score = fonk(names, cols, unwanted, gd, 1, kisi, simulation_count=12, **kwargs)
            sonuclar.append((_ozet(schedule), score))
        assert sonuclar[0] == sonuclar[1], (team, kisi, bicim, seed)


def test_simulasyon_izin_tablosu_sayisal_ve_nan_iken_de_ayni():
    # İzin tablosu bool değil de 0/1 ya da NaN içerirse "doğru mu" sorusu aynı yanıtlanır.
    eski = _eski_run_scheduling_core()
    names, cols, unwanted, gd, kwargs = _senaryo(10, seed=3)
    numeric = unwanted.astype(float)
    numeric.iloc[2, 4] = np.nan  # bool(nan) == True: izinli sayılır (eski davranış)
    for seed in range(3):
        sonuclar = []
        for fonk in (eski, run_scheduling_core):
            random.seed(seed)
            schedule, score = fonk(names, cols, numeric, gd, 1, 2, simulation_count=12, **kwargs)
            sonuclar.append((_ozet(schedule), score))
        assert sonuclar[0] == sonuclar[1]


def test_simulasyon_ic_dongusu_pandas_hucresi_okumaz():
    names, cols, unwanted, gd, kwargs = _senaryo(8, seed=1)
    reads = []
    original = pd.core.indexing._AtIndexer.__getitem__

    def counting(self, key):
        reads.append(key)
        return original(self, key)

    def okuma_sayisi(simulation_count):
        reads.clear()
        random.seed(0)
        with patch.object(pd.core.indexing._AtIndexer, "__getitem__", counting):
            run_scheduling_core(names, cols, unwanted, gd, 1, 2, simulation_count=simulation_count, **kwargs)
        return len(reads)

    # Eskiden her denemede binlerce okuma vardı; artık tablo başta bir kez okunur,
    # deneme sayısı arttıkça okuma (sonraki dengeleme adımları dışında) artmaz.
    tek, coklu = okuma_sayisi(1), okuma_sayisi(40)
    assert coklu - tek < 40 * len(names), (tek, coklu)


# --- 2. Kayıt: hücreler tek INSERT ile yazılır -----------------------------------

def test_kayit_ve_geri_yukleme_tek_sayfa_ister():
    """Veritabanı olmadan: execute_values'a satır sayısı kadar sayfa verilir."""
    import inspect as _inspect
    source = _inspect.getsource(db)
    assert source.count("execute_values(cur, ") == 2
    assert source.count("rows, page_size=len(rows))") == 2


class _SayanImlec(psycopg2.extensions.cursor):
    sayac = None

    def execute(self, query, vars=None):
        text = query.decode() if isinstance(query, bytes) else str(query)
        if "INSERT INTO schedule_data" in text and _SayanImlec.sayac is not None:
            _SayanImlec.sayac.append(text.count("),("))
        return super().execute(query, vars)


@pytest.fixture
def sayan_db(monkeypatch):
    if not TEST_DB:
        pytest.skip("TEST_DATABASE_URL tanımlı değil")
    monkeypatch.setenv("DATABASE_URL", TEST_DB)
    assert db.init_db(), db.get_last_error()
    real = psycopg2.connect
    inserts = []
    _SayanImlec.sayac = inserts
    monkeypatch.setattr(db.psycopg2, "connect",
                        lambda *a, **k: real(*a, cursor_factory=_SayanImlec, **k))
    owner = f"hiz-test-{uuid.uuid4().hex[:10]}@example.com"
    yield owner, inserts
    _SayanImlec.sayac = None
    conn = real(TEST_DB)
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM schedules WHERE owner = %s", (owner,))
        cur.execute("DELETE FROM app_settings WHERE key = %s", (db.settings_key(owner),))
    conn.close()


def _buyuk_ay(team=14, days=31):
    names = [f"Kişi{i}" for i in range(team)]
    cols = [f"{d} Gün" for d in range(1, days + 1)]
    schedule = pd.DataFrame(False, index=names, columns=cols)
    for d in range(days):
        schedule.iat[d % team, d] = True
    return names, schedule


def test_kayit_434_hucreyi_tek_insertle_yazar_ve_aynen_okur(sayan_db):
    owner, inserts = sayan_db
    names, schedule = _buyuk_ay()
    assert db.save_schedule("Otomatik_2026_12", 2026, 12, names, schedule,
                            rows_liste=[], owner=owner, expected_version=db.FORCE)
    # Eskiden 100'lük sayfalar: 434 satır için 5 INSERT.
    assert len(inserts) == 1
    assert inserts[0] + 1 == 14 * 31
    _, loaded, _, _, _ = db.load_schedule("Otomatik_2026_12", 2026, 12, owner=owner)
    assert loaded.loc[names, list(schedule.columns)].equals(schedule)


def test_geri_yukleme_her_ayi_tek_insertle_yazar(sayan_db):
    from nobet_core import backup_to_bytes, build_backup, parse_backup
    owner, inserts = sayan_db
    names, schedule = _buyuk_ay()
    for month in (11, 12):
        assert db.save_schedule(f"Otomatik_2026_{month}", 2026, month, names, schedule,
                                rows_liste=[], owner=owner, expected_version=db.FORCE)
    settings, schedules = db.export_owner_data(owner=owner)
    backup, errors = parse_backup(backup_to_bytes(build_backup(settings, schedules)))
    assert errors == []
    inserts.clear()
    assert db.restore_owner_data(backup, owner=owner)
    assert len(inserts) == 2  # ay başına bir INSERT
    _, loaded, _, _, _ = db.load_schedule("Otomatik_2026_12", 2026, 12, owner=owner)
    assert loaded.loc[names, list(schedule.columns)].equals(schedule)


# --- 3. Ayarlar paneli: kapalıyken çizilmez, açık/kapalı oturumda kalır -----------

def _fakes(**extra):
    fakes = dict(
        init_db=lambda: True, get_last_error=lambda: None, is_database_configured=lambda: False,
        SCHEMA_VERSION="hiz-test",
        load_app_settings=lambda **k: None, save_app_settings=lambda *a, **k: True,
        load_schedule=lambda *a, **k: None, save_schedule=lambda *a, **k: True,
        load_auto_saved_assignments=lambda *a, **k: ({}, {}), list_backup_snapshots=lambda **k: [],
        get_share_link=lambda *a: None, list_team_requests=lambda *a, **k: [], ping_db=lambda: True,
        get_schedule_version=lambda *a, **k: None, get_settings_version=lambda *a, **k: None,
    )
    fakes.update(extra)
    return fakes


def _app(team="Ali, Ayşe, Can, Deniz"):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).with_name("nobet.py")), default_timeout=120)
    at.session_state["isimler_text"] = team
    at.session_state["storage_init"] = {}
    at.session_state["yil_val"], at.session_state["ay_val"] = 2026, 10
    return at


@pytest.fixture
def uygulama():
    import streamlit
    streamlit.cache_resource.clear()
    with patch.dict("os.environ", {"DATABASE_URL": ""}), patch.multiple(db, create=True, **_fakes()):
        yield _app
    streamlit.cache_resource.clear()


def _acik(at):
    return any(box.key == "w_ekip_ad_0" for box in at.text_input)


def _ac_kapa(at):
    at.button(key="ayarlar_ac_kapa").click().run()
    assert [e.value for e in at.exception] == []


def test_ayarlar_kapaliyken_ici_cizilmez_acilip_kapanir(uygulama):
    at = uygulama()
    at.run()
    assert [e.value for e in at.exception] == []
    # Ekip varken kapalı başlar: ekip tablosu ve çapraz tablolar çizilmez.
    assert at.button(key="ayarlar_ac_kapa").label == "⚙️ Ayarlar"
    assert not _acik(at)
    assert not any(str(box.key).startswith(("w_yasak::", "w_birlikte::")) for box in at.checkbox)
    assert not any(box.key == "ay_input_direct" for box in at.selectbox)
    _ac_kapa(at)
    assert _acik(at)
    assert any(str(box.key).startswith("w_yasak::") for box in at.checkbox)
    # Açık kalır (başka bir tıklamada da).
    at.run()
    at.run()
    assert _acik(at) and at.session_state["settings_expanded"] is True
    _ac_kapa(at)
    assert not _acik(at)
    at.run()
    assert not _acik(at)


def test_ekip_bosken_ayarlar_acik_baslar(uygulama):
    at = uygulama(team="")
    at.run()
    assert [e.value for e in at.exception] == []
    assert _acik(at)


def test_ayarlar_kapali_iken_degerler_korunur_ve_hesaplar_suruyor(uygulama):
    at = uygulama()
    at.run()
    _ac_kapa(at)
    at.slider(key="kisi_slider_direct").set_value(3).run()
    at.text_input(key="w_role_names_text").input("AYB, GYB, NB").run()
    at.checkbox(key="w_yasak::Ali::Can").check().run()
    # İkinci kişinin adı düzeltilir (kutudan gelen değer).
    at.text_input(key="w_ekip_ad_1").input("Ayşegül").run()
    assert at.session_state["isimler_text"] == "Ali, Ayşegül, Can, Deniz"
    _ac_kapa(at)
    for _ in range(3):
        at.run()
    assert [e.value for e in at.exception] == []
    assert not _acik(at)
    # Kapalıyken de ayarlar hesaplanır ve kayda giden değerler değişmez.
    assert at.session_state["kisi_sayisi_val"] == 3
    assert at.session_state["rol_isimleri"] == ["AYB", "GYB", "NB"]
    assert at.session_state["forbidden_pairs"] == {("Ali", "Can")}
    assert at.session_state["isimler_text"] == "Ali, Ayşegül, Can, Deniz"
    # Üstteki ay oku kapalıyken de çalışır.
    at.button(key="month_next").click().run()
    assert [e.value for e in at.exception] == []
    assert at.session_state["ay_val"] == 11
    # Yeniden açılınca kutular aynı değerlerle gelir, ekip silinmez.
    _ac_kapa(at)
    assert at.selectbox(key="ay_input_direct").value == 11
    assert at.slider(key="kisi_slider_direct").value == 3
    assert at.text_input(key="w_role_names_text").value == "AYB, GYB, NB"
    assert at.checkbox(key="w_yasak::Ali::Can").value is True
    assert [at.text_input(key=f"w_ekip_ad_{i}").value for i in range(4)] == ["Ali", "Ayşegül", "Can", "Deniz"]
    at.run()
    assert at.session_state["isimler_text"] == "Ali, Ayşegül, Can, Deniz"
    assert not at.warning or all("Session State" not in w.value for w in at.warning)


# --- 4. Paylaşım: ekip listesi önbellekte, jeton her seferinde sorulur -------------

def test_paylasim_ekibi_onbellekte_jeton_her_seferinde_sorulur():
    import streamlit
    from streamlit.testing.v1 import AppTest
    import nobet_paylasim
    calls = {"token": 0, "settings": 0}

    def resolve(token, raise_on_error=False):
        calls["token"] += 1
        return ("sahip@example.com", 0, 0) if token == "abc123" else None

    def settings(owner="", raise_on_error=False, version_out=None):
        calls["settings"] += 1
        return {"team": "Ali, Ayşe"} if owner == "sahip@example.com" else None

    nobet_paylasim._cached_team.clear()
    streamlit.cache_resource.clear()
    fakes = _fakes(resolve_share_token=resolve, load_app_settings=settings,
                   list_shared_months=lambda owner: [])
    with patch.dict("os.environ", {"DATABASE_URL": ""}), patch.multiple(db, create=True, **fakes):
        at = AppTest.from_file(str(Path(__file__).with_name("nobet.py")), default_timeout=120)
        at.query_params["paylas"] = "abc123"
        for _ in range(3):
            at.run()
            assert [e.value for e in at.exception] == []
            assert at.selectbox(key="paylasim_kisi").options == ["Ali", "Ayşe"]
    nobet_paylasim._cached_team.clear()
    streamlit.cache_resource.clear()
    assert calls["token"] == 3      # jeton denetimi önbelleksiz
    assert calls["settings"] == 1   # ekip listesi 10 sn önbellekte


def test_paylasim_ekip_onbellegi_sahipleri_karistirmaz_hatayi_saklamaz():
    import nobet_paylasim
    calls = []
    fail = {"on": True}

    def settings(owner="", raise_on_error=False, version_out=None):
        calls.append(owner)
        if fail["on"]:
            if raise_on_error:
                raise psycopg2.OperationalError("uyku")
            return None
        return {"team": {"a@x": "Ali, Can", "b@x": "Ece"}[owner]}

    nobet_paylasim._cached_team.clear()
    try:
        with patch.object(nobet_paylasim, "load_app_settings", settings):
            # Okuma hatası önbelleğe "boş ekip" olarak yazılmaz.
            with pytest.raises(psycopg2.OperationalError):
                nobet_paylasim._cached_team("a@x")
            fail["on"] = False
            assert nobet_paylasim._cached_team("a@x") == ["Ali", "Can"]
            assert nobet_paylasim._cached_team("b@x") == ["Ece"]
            assert nobet_paylasim._cached_team("a@x") == ["Ali", "Can"]
    finally:
        nobet_paylasim._cached_team.clear()
    assert calls == ["a@x", "a@x", "b@x"]


# --- İnceleme bulguları: kapalı panelde eski değer geri gelmez, kapanırken yazılan kaybolmaz ---

@pytest.fixture
def kayitli_uygulama():
    """Ekip kayıtlı açılış; ayar kayıtları ``saved`` listesine düşer."""
    import streamlit
    saved = []
    fakes = _fakes(save_app_settings=lambda settings, *a, **k: saved.append(dict(settings)) or True)
    streamlit.cache_resource.clear()
    with patch.dict("os.environ", {"DATABASE_URL": ""}), patch.multiple(db, create=True, **fakes):
        yield _app, saved
    streamlit.cache_resource.clear()


def test_kapali_panelde_kutudan_secilen_ay_ok_tusunun_eski_ayina_donmez(kayitli_uygulama):
    app, saved = kayitli_uygulama
    at = app()
    at.run()
    # Açmadan iki kez ileri ok: ok, ay kutusunun anahtarına doğrudan yazar.
    at.button(key="month_next").click().run()
    at.button(key="month_next").click().run()
    assert (at.session_state["yil_val"], at.session_state["ay_val"]) == (2026, 12)
    _ac_kapa(at)
    at.selectbox(key="ay_input_direct").set_value(3).run()
    _ac_kapa(at)
    for _ in range(3):
        at.run()
        assert (at.session_state["yil_val"], at.session_state["ay_val"]) == (2026, 3)
    at.button(key="month_next").click().run()
    assert (at.session_state["yil_val"], at.session_state["ay_val"]) == (2026, 4)
    at.run()
    assert [e.value for e in at.exception] == []
    assert saved and (saved[-1]["year"], saved[-1]["month"]) == (2026, 4)
    _ac_kapa(at)
    assert at.selectbox(key="ay_input_direct").value == 4


def test_kapali_panelde_elle_girilen_ucret_varsayilana_donmez(kayitli_uygulama):
    app, saved = kayitli_uygulama
    at = app()
    at.run()
    _ac_kapa(at)
    at.number_input(key="w_nobet_ucreti_val").set_value(250.0).run()
    _ac_kapa(at)
    for _ in range(3):
        at.run()
        assert at.session_state["nobet_ucreti_val"] == 250.0
    assert saved and saved[-1]["hourly_rate"] == 250.0
    _ac_kapa(at)
    assert at.number_input(key="w_nobet_ucreti_val").value == 250.0


def test_ayarlar_kapanirken_yazilan_ekip_ve_kutular_kaybolmaz(kayitli_uygulama):
    """Enter/Tab'a basmadan doğrudan başlığa tıklanınca kutu değerleri aynı çalışmada gelir."""
    app, saved = kayitli_uygulama
    at = app()
    at.run()
    _ac_kapa(at)
    rows = at.session_state["_ekip_rows"]
    at.text_input(key=f"w_ekip_ad_{rows - 1}").input("Ece")
    at.number_input(key="w_ekip_max_0").set_value(5)
    at.checkbox(key="w_birlikte::Ali::Can").check()
    at.text_input(key="w_role_names_text").input("AYB, GYB")
    hk = next(t.key for t in at.text_input if str(t.key).startswith("w_holiday_text_"))
    at.text_input(key=hk).input("3,4")
    _ac_kapa(at)  # hepsi kapanma tıklamasıyla aynı çalışmada
    at.run()
    assert not _acik(at)
    assert at.session_state["isimler_text"] == "Ali, Ayşe, Can, Deniz, Ece"
    assert at.session_state["limits_text"] == "Ali:0-5"
    assert at.session_state["together_pairs_text"] == "Ali-Can"
    assert at.session_state["role_names_text"] == "AYB, GYB"
    assert at.session_state["holidays_by_period"]["2026-10"] == "3,4"
    assert saved and saved[-1]["team"] == "Ali, Ayşe, Can, Deniz, Ece"
    _ac_kapa(at)
    assert [at.text_input(key=f"w_ekip_ad_{i}").value for i in range(5)] == ["Ali", "Ayşe", "Can", "Deniz", "Ece"]
    assert at.checkbox(key="w_birlikte::Ali::Can").value is True
