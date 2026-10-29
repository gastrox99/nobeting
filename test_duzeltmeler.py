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


def test_iyilestir_karsilanmayan_tercih_varsa_dener():
    people, cols, gd, unwanted = _tight_case()
    schedule = pd.DataFrame(False, index=people, columns=cols)
    for i, c in enumerate(cols):
        schedule.at[people[i % 4], c] = True        # dengeli, ama Ali 2. günü istiyor
    preferred = pd.DataFrame(0, index=people, columns=cols)
    preferred.at["Ali", cols[1]] = 1
    fixed, changed = improve_if_needed(
        schedule, people, cols, unwanted, gd, 0, 1, df_preferred=preferred,
    )
    assert changed and bool(fixed.at["Ali", cols[1]])
    totals = fixed.sum(axis=1)
    assert totals.max() - totals.min() <= 1


def test_iyilestir_gerekmiyorsa_dokunmaz():
    people, cols, gd, unwanted = _tight_case()
    schedule = pd.DataFrame(False, index=people, columns=cols)
    for i, c in enumerate(cols):
        schedule.at[people[i % 4], c] = True
    same, changed = improve_if_needed(schedule, people, cols, unwanted, gd, 0, 1)
    assert not changed and same is schedule


def _ay_listesi(weekend_of, friday_of):
    """Her gün bir kişi. Hafta sonu ve Cuma verilen sırayla, kalan günler toplamı dengeler."""
    people = ["Ali", "Ayşe", "Can", "Deniz"]
    gd = build_gun_detaylari(2026, 11, 30, [])
    cols = list(gd)
    unwanted = pd.DataFrame(False, index=people, columns=cols)
    weekends = [c for c in cols if gd[c]["weekend"]]
    fridays = [c for c in cols if gd[c]["weekday"] == 4]
    rest = [c for c in cols if c not in weekends and c not in fridays]
    schedule = pd.DataFrame(False, index=people, columns=cols)
    for col, person in zip(weekends, weekend_of):
        schedule.at[person, col] = True
    for col, person in zip(fridays, friday_of):
        schedule.at[person, col] = True
    counts = {p: int(schedule.loc[p].sum()) for p in people}
    for col in rest:
        person = min(people, key=lambda name: (counts[name], name))
        schedule.at[person, col] = True
        counts[person] += 1
    return people, cols, gd, unwanted, schedule


def test_toplam_denkken_hafta_sonu_veya_cuma_farki_iyilestirir(monkeypatch):
    # Toplam farkı 1. Hafta sonu farkı 6 ya da Cuma farkı 4 iken çözücü çalışmalı.
    from nobet_iyilestir import schedule_metrics

    calls = []

    def fake(*args, **kwargs):
        calls.append(True)
        return None, {}, {}

    monkeypatch.setattr("nobet_iyilestir.improve_schedule", fake)
    people = ["Ali", "Ayşe", "Can", "Deniz"]
    for owners, key in (
        (["Ali"] * 6 + ["Ayşe"] * 3, "special"),
        (["Ali", "Ayşe", "Can", "Deniz"] * 3, "friday"),
    ):
        calls.clear()
        fridays = people if key == "special" else ["Ali"] * 4
        weekends = owners if key == "special" else ["Ali", "Ayşe", "Can", "Deniz", "Ali", "Ayşe", "Can", "Deniz", "Ali"]
        who, cols, gd, unwanted, schedule = _ay_listesi(weekends, fridays)
        metrics = schedule_metrics(schedule, who, cols, gd, 1, 1)
        assert metrics["unfilled"] == 0 and metrics["total"] <= 1
        assert metrics[key] > 1
        improve_if_needed(schedule, who, cols, unwanted, gd, 1, 1)
        assert calls, key


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
        init_db=lambda: True, get_last_error=lambda: None, is_database_configured=lambda: False, SCHEMA_VERSION="test",
        load_app_settings=lambda **k: None, save_app_settings=lambda *a, **k: True,
        load_schedule=lambda *a, **k: None, save_schedule=lambda *a, **k: True,
        load_auto_saved_assignments=lambda *a, **k: ({}, {}), list_backup_snapshots=lambda **k: [],
        get_share_link=lambda *a: None, ping_db=lambda: True,
        get_schedule_version=lambda *a, **k: None, get_settings_version=lambda *a, **k: None,
    )
    names = ", ".join(f"Kişi{i:02d}" for i in range(27))
    with patch.dict("os.environ", {"DATABASE_URL": ""}), patch.multiple(db, **fakes):
        at = AppTest.from_file(str(Path(__file__).with_name("nobet.py")), default_timeout=120)
        at.session_state["isimler_text"] = names
        at.session_state["storage_init"] = {}
        at.session_state["yil_val"], at.session_state["ay_val"] = 2026, 10
        at.session_state["settings_expanded"] = True  # Ayarlar açık
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
    assert "expected_version=_schedule_version.get(\"version\")" in source
    assert "version_out=_schedule_version" in source
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
        init_db=lambda: True, get_last_error=lambda: None, is_database_configured=lambda: False, SCHEMA_VERSION="test",
        load_app_settings=lambda **k: None, save_app_settings=lambda *a, **k: True,
        load_schedule=lambda *a, **k: (None, None, None, None, None), save_schedule=conflict,
        load_auto_saved_assignments=lambda *a, **k: ({}, {}), list_backup_snapshots=lambda **k: [],
        get_share_link=lambda *a: None, ping_db=lambda: True,
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
        at.button(key="ayarlar_ac_kapa").click().run()  # Ayarlar ekip varken kapalı başlar
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


def test_ayar_cakismasinda_farkli_alanlar_birlesir():
    from nobet_core import merge_app_settings
    base = {"team": "Ali, Ayşe", "limits": "", "year": 2026, "month": 11}
    mine = {"team": "Ali, Ayşe, Can", "limits": "", "year": 2026, "month": 11}
    theirs = {"team": "Ali, Ayşe", "limits": "Ali:3", "year": 2026, "month": 12}
    merged, adopt = merge_app_settings(base, mine, theirs)
    # Bu sekmenin ekibi + öbür yerin limiti; bakılan ay bu sekmeninki kalır.
    assert merged == {"team": "Ali, Ayşe, Can", "limits": "Ali:3", "year": 2026, "month": 11}
    assert adopt == {"limits": "Ali:3"}


def test_ayar_cakismasinda_ayni_deger_cakisma_degil_farkli_deger_cakisma():
    from nobet_core import merge_app_settings
    base = {"team": "Ali", "fixed_roles": {}}
    same = {"team": "Ali", "fixed_roles": {"Ali": "AYB"}}
    assert merge_app_settings(base, same, dict(same)) == (same, {})
    assert merge_app_settings(base, same, {"team": "Ali", "fixed_roles": {"Ali": "GYB"}}) is None
    # Bu oturum hiç kaydetmemişse (bilinen kayıt yok) birleştirilmez.
    assert merge_app_settings(None, same, base) is None
    # Öbür yerde silinen alan bu oturumda değişmediyse silinir.
    merged, adopt = merge_app_settings({"team": "Ali", "limits": "Ali:2"}, {"team": "Ali, Can", "limits": "Ali:2"}, {"team": "Ali"})
    assert merged == {"team": "Ali, Can"} and adopt == {}


def test_ayar_cakismasi_uyarisinda_farkli_alanlar_yazilir():
    from nobet_core import settings_difference_labels
    assert settings_difference_labels({"team": "A", "limits": "", "month": 11}, {"team": "A", "limits": "A:2", "month": 12}) == ["nöbet sınırları", "ay"]


def test_cakisma_uyarisinda_son_kaydi_yapan_cihaz():
    from datetime import datetime, timezone
    from nobet_core import conflict_writer_text, device_label, writer_tag
    iphone = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1"
    home_screen = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"
    mac_chrome = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36"
    assert device_label(iphone) == "iPhone · Safari"
    assert device_label(home_screen) == "iPhone · Ana Ekran uygulaması"
    assert device_label(mac_chrome) == "Mac · Chrome"
    assert writer_tag(iphone, "abcdef123") == "iPhone · Safari · oturum abcdef"
    text = conflict_writer_text("iPhone · Safari · oturum 111111", datetime(2026, 10, 20, 17, 58, tzinfo=timezone.utc), "iPhone · Safari · oturum 222222")
    assert "Son kaydı yapan: iPhone · Safari · oturum 111111 20:58" in text
    assert "bu cihaz" in text
    assert conflict_writer_text(None, None, "x") == ""
