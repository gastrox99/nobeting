# -*- coding: utf-8 -*-
"""Sonnet kod incelemesi (19 Ekim 2026) düzeltmelerinin testleri."""
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import db
from nobet_core import (
    build_gun_detaylari,
    csv_safe_frame,
    csv_safe_text,
    import_schedule_table,
    team_settings_texts,
)
from nobet_iyilestir import improve_if_needed, improve_schedule


def test_csv_formul_basliyan_metin_korunur():
    assert csv_safe_text("=1+1") == "'=1+1"
    assert csv_safe_text("+90") == "'+90"
    assert csv_safe_text("@ali") == "'@ali"
    assert csv_safe_text("-cmd") == "'-cmd"
    # Boş nöbet yeri ve normal metin değişmez.
    assert csv_safe_text("-") == "-"
    assert csv_safe_text("01.10.2026 Per") == "01.10.2026 Per"
    assert csv_safe_text(5) == 5
    frame = pd.DataFrame({"Tarih": ["01.10.2026 Per"], "AYB": ["=HYPERLINK()"], "GYB": ["-"]})
    safe = csv_safe_frame(frame)
    assert safe.loc[0, "AYB"] == "'=HYPERLINK()" and safe.loc[0, "GYB"] == "-"


def test_isimdeki_iki_nokta_limit_satirini_bozmaz():
    names, limits = team_settings_texts([{"name": "Ali:Veli", "max": 4}])
    assert names == "Ali Veli"
    assert limits == "Ali Veli:0-4"


def _import(table, roles, people=("Ali", "Ayşe", "Can")):
    gd = build_gun_detaylari(2026, 10, 31, [])
    return import_schedule_table(table, list(people), list(gd), gd, roles, 2026, 10)


def test_excel_gorev_sutunlari_basliga_gore_eslesir():
    # Sütunlar uygulamadaki sıranın tersi: başlığa göre okunmalı.
    table = pd.DataFrame({"Tarih": ["01.10.2026 Per"], "GYB": ["Ayşe"], "AYB": ["Ali"]})
    result = _import(table, ["AYB", "GYB"])
    assert result["rows"][0]["AYB"] == "Ali" and result["rows"][0]["GYB"] == "Ayşe"


def test_excel_ek_sutunlari_cizelgeye_eklenir():
    table = pd.DataFrame({
        "Tarih": ["01.10.2026 Per"], "AYB": ["Ali"], "GYB": ["Ayşe"], "Ek 1": ["Can"],
    })
    result = _import(table, ["AYB", "GYB"])
    assert bool(result["schedule"].at["Can", "1 Per"])
    assert result["extra_people"] == 1
    # Ek sütunundaki ekip dışı isim atlananlara yazılır.
    table["Ek 1"] = ["Bilinmeyen"]
    assert _import(table, ["AYB", "GYB"])["skipped_names"] == ["Bilinmeyen"]


def test_excel_basliklar_uymazsa_siraya_gore_eslesir():
    table = pd.DataFrame({"Tarih": ["01.10.2026 Per"], "Birim A": ["Ali"], "Birim B": ["Ayşe"]})
    result = _import(table, ["AYB", "GYB"])
    assert result["rows"][0]["AYB"] == "Ali" and result["rows"][0]["GYB"] == "Ayşe"


def _tight_case():
    people = ["Ali", "Ayşe", "Can", "Deniz"]
    gd = build_gun_detaylari(2026, 11, 30, [])
    cols = list(gd)[:10]
    gd = {c: gd[c] for c in cols}
    unwanted = pd.DataFrame(False, index=people, columns=cols)
    return people, cols, gd, unwanted


def test_iyilestir_gerekirse_toplam_farkini_kapatir():
    people, cols, gd, unwanted = _tight_case()
    schedule = pd.DataFrame(False, index=people, columns=cols)
    for c in cols[0:10:2]:
        schedule.at["Ali", c] = True            # Ali 5
    for c, p in zip(cols[1:10:2], ["Ayşe", "Ayşe", "Ayşe", "Ayşe", "Can"]):
        schedule.at[p, c] = True                # Ayşe 4, Can 1, Deniz 0
    fixed, changed = improve_if_needed(schedule, people, cols, unwanted, gd, 0, 1)
    assert changed
    totals = fixed.sum(axis=1)
    assert totals.max() - totals.min() <= 1
    assert (fixed.sum(axis=0) == 1).all()


def test_iyilestir_gerekmiyorsa_dokunmaz():
    people, cols, gd, unwanted = _tight_case()
    schedule = pd.DataFrame(False, index=people, columns=cols)
    for i, c in enumerate(cols):
        schedule.at[people[i % 4], c] = True
    same, changed = improve_if_needed(schedule, people, cols, unwanted, gd, 0, 1)
    assert not changed and same is schedule


def test_alt_siniri_altindaki_kisinin_nobeti_azalmaz_ama_artabilir():
    people, cols, gd, unwanted = _tight_case()
    schedule = pd.DataFrame(False, index=people, columns=cols)
    for i, c in enumerate(cols):
        schedule.at[people[i % 3], c] = True    # Deniz 0 nöbet
    limits = {"Deniz": {"min": 2, "max": 5}}
    result, before, after = improve_schedule(schedule, people, cols, unwanted, gd, 0, 1, person_limits=limits)
    assert result is not None
    assert result.loc["Deniz"].sum() >= 2
    assert after["min_short"] < before["min_short"]


def test_buyuk_ekipte_yasak_ikili_listesi_kullanilir():
    from streamlit.testing.v1 import AppTest
    fakes = dict(
        init_db=lambda: True, get_last_error=lambda: None, is_database_configured=lambda: False,
        load_app_settings=lambda **k: None, save_app_settings=lambda *a, **k: True,
        load_schedule=lambda *a, **k: None, save_schedule=lambda *a, **k: True,
        load_auto_saved_assignments=lambda *a, **k: ({}, {}), list_backup_snapshots=lambda **k: [],
        get_share_link=lambda *a: None,
        get_schedule_version=lambda *a, **k: None, get_settings_version=lambda *a, **k: None,
    )
    names = ", ".join(f"Kişi{i:02d}" for i in range(27))
    with patch.dict("os.environ", {"DATABASE_URL": ""}), patch.multiple(db, **fakes):
        at = AppTest.from_file(str(Path(__file__).with_name("nobet.py")), default_timeout=120)
        at.session_state["isimler_text"] = names
        at.session_state["storage_init"] = {}
        at.session_state["yil_val"], at.session_state["ay_val"] = 2026, 10
        at.run()
        assert [e.value for e in at.exception] == []
        # 27 kişi: 700'den fazla onay kutusu yerine tek seçim listesi.
        assert len(at.checkbox) < 30
        pairs = at.multiselect(key="w_yasak_liste")
        pairs.select("Kişi00 + Kişi01")
        at.run()
        assert [e.value for e in at.exception] == []
        assert at.session_state["forbidden_pairs_text"] == "Kişi00-Kişi01"
        # İşaretli ikili kaldırılabilir.
        at.multiselect(key="w_yasak_liste").unselect("Kişi00 + Kişi01")
        at.run()
        assert at.session_state["forbidden_pairs_text"] == ""


def test_uygulama_surum_denetimini_kullanir():
    source = Path(__file__).with_name("nobet.py").read_text(encoding="utf-8")
    assert "expected_version=st.session_state.get(\"schedule_version\")" in source
    assert "except SaveConflict:" in source
    assert "_auto_improve(_new_schedule)" in source
    assert "period_switch_blocked" in source


def test_ayni_gun_iki_satir_yuklemeyi_reddeder():
    gd = build_gun_detaylari(2026, 10, 31, [])
    cols = list(gd)
    table = pd.DataFrame({"Tarih": ["1 Eki Per", "1 Eki Per"], "R1": ["Ali", "Ayşe"], "R2": ["Can", "-"]})
    result = import_schedule_table(table, ["Ali", "Ayşe", "Can"], cols, gd, ["R1", "R2"], 2026, 10)
    assert result["duplicate_day"] == 1
    # Yinelenen satır çizelgeye yazılmaz: liste ile çizelge ayrışmaz.
    assert int(result["schedule"][cols[0]].sum()) == 2
    assert result["rows"][0]["R1"] == "Ali"


def test_yedekte_cok_buyuk_ayar_reddedilir():
    from nobet_core import BACKUP_FORMAT, BACKUP_VERSION, parse_backup
    import json
    big = {"x": "a" * (210 * 1024)}
    raw = json.dumps({"format": BACKUP_FORMAT, "version": BACKUP_VERSION, "settings": big, "schedules": []})
    parsed, errors = parse_backup(raw.encode("utf-8"))
    assert parsed is None and errors == ["Yedekteki ayarlar çok büyük."]
    ok = json.dumps({"format": BACKUP_FORMAT, "version": BACKUP_VERSION, "settings": {"x": "a"}, "schedules": []})
    assert parse_backup(ok.encode("utf-8"))[0] is not None


def test_cakisan_ayda_ay_degistirmek_once_uyarir():
    from streamlit.testing.v1 import AppTest
    import streamlit

    def conflict(*args, **kwargs):
        raise db.SaveConflict("x")

    fakes = dict(
        init_db=lambda: True, get_last_error=lambda: None, is_database_configured=lambda: False,
        load_app_settings=lambda **k: None, save_app_settings=lambda *a, **k: True,
        load_schedule=lambda *a, **k: (None, None, None, None, None), save_schedule=conflict,
        load_auto_saved_assignments=lambda *a, **k: ({}, {}), list_backup_snapshots=lambda **k: [],
        get_share_link=lambda *a: None,
        get_schedule_version=lambda *a, **k: "v1", get_settings_version=lambda *a, **k: None,
    )
    with patch.dict("os.environ", {"DATABASE_URL": ""}), patch.multiple(db, **fakes):
        at = AppTest.from_file(str(Path(__file__).with_name("nobet.py")), default_timeout=120)
        at.session_state["isimler_text"] = "Ali, Ayşe, Can, Deniz, Ece"
        at.session_state["storage_init"] = {}
        at.session_state["yil_val"], at.session_state["ay_val"] = 2026, 10
        at.run()
        next(b for b in at.button if b.label == "⚡ Simülasyon").click()
        at.run()
        assert [e.value for e in at.exception] == []
        assert at.session_state["autosave_conflict_period"] == (2026, 10)
        # Ay değiştirme: ilk denemede engellenir ve çakışma uyarısı çıkar.
        at.number_input(key="ay_input_direct").set_value(11) if any(
            n.key == "ay_input_direct" for n in at.number_input
        ) else at.selectbox(key="ay_input_direct").set_value(11)
        at.run()
        assert [e.value for e in at.exception] == []
        assert any("başka bir yerde" in e.value for e in at.error)
        assert at.session_state["ay_input_direct"] == 10
        # İkinci denemede kullanıcı bilerek geçiyor.
        (at.number_input(key="ay_input_direct") if any(
            n.key == "ay_input_direct" for n in at.number_input
        ) else at.selectbox(key="ay_input_direct")).set_value(11)
        at.run()
        assert [e.value for e in at.exception] == []
        assert at.session_state["ay_input_direct"] == 11
    streamlit.cache_resource.clear()
