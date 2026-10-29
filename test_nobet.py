# -*- coding: utf-8 -*-
"""
test_nobet.py - Nöbet Yönetimi Uygulaması Unit Testleri
Çalıştırmak için: python -m pytest test_nobet.py -v
"""
import unittest
import re
import json
from pathlib import Path
import pandas as pd
import numpy as np
import random
import time
from unittest.mock import patch
import db
from nobet_core import (
    NOBET_UNVANLARI,
    alternate_day_chains,
    katsayi_donemi, katsayi_uyarisi,
    nobet_ucreti_hesapla,
    saat_ucretleri,
    unvan_saat_ucreti,
    parse_holiday_days,
    validate_inputs,
    parse_forbidden_pairs,
    parse_person_limits,
    parse_person_limits_with_issues,
    unknown_names,
    previous_period,
    last_shift_offsets,
    special_shift_counts,
    history_special_adjustment,
    history_adjustment,
    build_backup,
    backup_to_bytes,
    parse_backup,
    backup_summary,
    backup_reminder,
    holiday_shift_counts,
    schedule_totals,
    build_preference_share_text,
    build_share_text,
    build_person_ics,
    shared_default_month,
    shared_visible_months,
    shared_day_states,
    on_duty_summary,
    shared_updated_text,
    preference_overview,
    preference_summary,
    compact_team_rows,
    team_settings_rows,
    team_settings_texts,
    rename_in_pairs_text,
    toggle_pair_text,
    special_cap,
    has_total_limit,
    forbidden_pairs_to_text,
    person_limits_to_text,
    parse_forbidden_pairs,
    parse_person_limits_with_issues,
    official_half_days,
    required_work_hours,
    parse_holiday_days,
    default_holiday_text,
    restore_auto_saved_schedule,
    build_gun_detaylari,
    build_schedule_analysis,
    changed_schedule_columns,
    find_person_role,
    get_preference_grid_legend,
    get_preference_grid_state,
    normalize_preference_grid,
    run_scheduling_core,
    weekend_pair_count,
    _balance_fridays,
    _balance_holidays,
    _balance_specials,
    unavailable_day_counts,
    is_blocked_preference,
    personal_work_hours,
    missing_shift_coverage,
    explain_missing_day,
    format_missing_day_reason,
    schedule_fingerprint,
    synchronize_changed_role_rows,
    switch_schedule_period,
    apply_app_settings,
    assignment_warnings,
    collect_app_settings,
    normalize_owner_email,
    period_key,
    persistence_fingerprint,
    use_compact_schedule_editor,
    build_limit_violation_messages,
    parse_schedule_import_date,
    undo_history_state,
    redo_history_state,
    update_persistence_feedback,
)

# Arayüz nobet.py ve bölüm dosyalarına ayrıldı; kaynak denetimleri hepsine bakar.
APP_FILES = (
    "nobet.py", "nobet_stil.py", "nobet_durum.py", "nobet_paylasim.py", "nobet_disa_aktar.py",
    "nobet_yedek.py", "nobet_tablo.py", "nobet_takvim.py", "nobet_analiz.py",
    "nobet_ekip.py", "nobet_talepler.py", "nobet_yil.py", "nobet_onkontrol.py",
)


def _app_source(*names):
    return "\n".join(
        Path(__file__).with_name(name).read_text(encoding="utf-8") for name in (names or APP_FILES)
    )




class TestParseHolidayDays(unittest.TestCase):
    def test_gecerli_ve_gecersiz_karisik_girdi(self):
        days, invalid = parse_holiday_days("1,5-7,32,abc,9-4", 31)

        self.assertEqual(days, [1, 5, 6, 7])
        self.assertEqual(invalid, ["32", "abc", "9-4"])

    def test_ay_sinirini_asan_aralik_sessizce_kirpilmiyor(self):
        days, invalid = parse_holiday_days("30-32", 31)

        self.assertEqual(days, [])
        self.assertEqual(invalid, ["30-32"])

    def test_gecersiz_tek_gun_ve_hatali_metin_bildirilir(self):
        days, invalid = parse_holiday_days("0,35,izin", 31)

        self.assertEqual(days, [])
        self.assertEqual(invalid, ["0", "35", "izin"])

    def test_unicode_rakam_tek_gun_olarak_gecersizdir(self):
        days, invalid = parse_holiday_days("1,²,3", 31)

        self.assertEqual(days, [1, 3])
        self.assertEqual(invalid, ["²"])

    def test_unicode_rakam_aralik_ucunda_gecersizdir(self):
        days, invalid = parse_holiday_days("1,4-²,7", 31)

        self.assertEqual(days, [1, 7])
        self.assertEqual(invalid, ["4-²"])

    def test_tekrarlanan_gunler_tekilleştirilir(self):
        days, invalid = parse_holiday_days("1,1,2-3,3", 31)

        self.assertEqual(days, [1, 2, 3])
        self.assertEqual(invalid, [])


class TestParseScheduleImportDate(unittest.TestCase):
    def test_tarih_yil_ve_ay_bilgisiyle_parse_edilir(self):
        self.assertEqual(
            parse_schedule_import_date("01.06.2026 Paz"),
            {"day": 1, "month": 6, "year": 2026},
        )

    def test_iso_tarih_parse_edilir(self):
        self.assertEqual(
            parse_schedule_import_date("2026-07-15"),
            {"day": 15, "month": 7, "year": 2026},
        )

    def test_turkce_ay_etiketi_parse_edilir(self):
        self.assertEqual(
            parse_schedule_import_date("15 Haziran 2026 Pzt"),
            {"day": 15, "month": 6, "year": 2026},
        )

    def test_eski_sadece_gun_bicimi_korunur(self):
        self.assertEqual(
            parse_schedule_import_date("15"),
            {"day": 15, "month": None, "year": None},
        )

    def test_excel_seri_tarihi_parse_edilir(self):
        self.assertEqual(
            parse_schedule_import_date(46174),
            {"day": 1, "month": 6, "year": 2026},
        )

    def test_gecersiz_tarih_parse_edilmez(self):
        self.assertIsNone(parse_schedule_import_date("31.02.2026"))


# ------------------------------------------------------------------------------
# 2. validate_inputs testleri
# ------------------------------------------------------------------------------
class TestValidateInputs(unittest.TestCase):

    def _base_params(self, **overrides):
        params = dict(
            isimler=["Ali", "Ayşe", "Mehmet", "Fatma", "Can"],
            yil=2025, ay=1, gun_sayisi=31,
            tatil_gunleri=[], nobet_ucreti=100.0,
            min_bosluk=1, kisi_sayisi=2
        )
        params.update(overrides)
        return params

    def test_gecerli_girdi(self):
        is_valid, errors, warnings = validate_inputs(**self._base_params())
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_bos_ekip(self):
        is_valid, errors, _ = validate_inputs(**self._base_params(isimler=[]))
        self.assertFalse(is_valid)
        self.assertTrue(any("En az 1 kişi" in e for e in errors))

    def test_cok_fazla_kisi(self):
        isimler = [f"Kişi{i}" for i in range(51)]
        is_valid, errors, _ = validate_inputs(**self._base_params(isimler=isimler))
        self.assertFalse(is_valid)
        self.assertTrue(any("50" in e for e in errors))

    def test_yinelenen_isim(self):
        is_valid, errors, _ = validate_inputs(**self._base_params(isimler=["Ali", "Ali", "Mehmet"]))
        self.assertFalse(is_valid)
        self.assertTrue(any("aynı isim" in e.lower() for e in errors))

    def test_negatif_ucret(self):
        is_valid, errors, _ = validate_inputs(**self._base_params(nobet_ucreti=-1))
        self.assertFalse(is_valid)
        self.assertTrue(any("negatif" in e for e in errors))

    def test_sifir_ucret_uyari(self):
        is_valid, _, warnings = validate_inputs(**self._base_params(nobet_ucreti=0))
        self.assertTrue(is_valid)
        self.assertTrue(any("0 TL" in w for w in warnings))

    def test_gecersiz_tatil_gunu(self):
        is_valid, errors, _ = validate_inputs(**self._base_params(tatil_gunleri=[0, 32]))
        self.assertFalse(is_valid)
        self.assertTrue(any("Geçersiz tatil" in e for e in errors))

    def test_gecerli_tatil_gunu(self):
        is_valid, errors, _ = validate_inputs(**self._base_params(tatil_gunleri=[1, 15, 31]))
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_tatiller_nobet_gunu_sayilir_ve_tekrar_etkilemez(self):
        # Tatillerde de nöbet tutulur: 2 gün x 2 kişi = 4 pozisyon, tatil sayısından bağımsız.
        is_valid, errors, warnings = validate_inputs(
            **self._base_params(
                isimler=["Ali", "Ayşe", "Mehmet", "Fatma", "Can"],
                gun_sayisi=2,
                tatil_gunleri=[1, 1],
            )
        )

        self.assertTrue(is_valid)
        self.assertEqual(errors, [])
        self.assertTrue(any("4 pozisyon" in warning for warning in warnings))

    def test_dinlenme_kapasitesi_yetmezse_uyarir(self):
        # 3 kişi, 1 gün dinlenme: kişi başı en fazla 15 nöbet -> 45 < 60
        _, errors, warnings = validate_inputs(
            **self._base_params(isimler=["Ali", "Ayşe", "Can"], gun_sayisi=30, min_bosluk=1)
        )
        self.assertEqual(errors, [])
        self.assertTrue(any("en fazla 45 nöbet" in w and "60 nöbet" in w for w in warnings))

        _, _, warnings = validate_inputs(
            **self._base_params(isimler=["Ali", "Ayşe", "Can", "Deniz"], gun_sayisi=30, min_bosluk=1)
        )
        self.assertFalse(any("dinlenme kuralıyla" in w for w in warnings))

    def test_yetersiz_ekip(self):
        is_valid, errors, _ = validate_inputs(**self._base_params(isimler=["Ali"], kisi_sayisi=2))
        self.assertFalse(is_valid)
        self.assertTrue(any("en az" in e for e in errors))

    def test_dinlenme_suresi_sinir_disi(self):
        is_valid, errors, _ = validate_inputs(**self._base_params(min_bosluk=8))
        self.assertFalse(is_valid)
        self.assertTrue(any("Dinlenme" in e for e in errors))

    def test_cok_az_pozisyon_uyari(self):
        # 5 kişi, 1 günde 2 pozisyon → nöbet sayısı ekipten az
        is_valid, _, warnings = validate_inputs(
            **self._base_params(isimler=["Ali","Ayşe","Mehmet","Fatma","Can"],
                                gun_sayisi=1, tatil_gunleri=[])
        )
        self.assertTrue(is_valid)
        self.assertTrue(any("fazla kişi" in w for w in warnings))


# ------------------------------------------------------------------------------
# 3. parse_forbidden_pairs testleri
# ------------------------------------------------------------------------------
class TestParseForbiddenPairs(unittest.TestCase):

    def test_bos_girdi(self):
        self.assertEqual(parse_forbidden_pairs(""), set())

    def test_tek_cift(self):
        result = parse_forbidden_pairs("Ali-Ayşe")
        self.assertIn(("Ali", "Ayşe"), result)

    def test_coklu_cift(self):
        result = parse_forbidden_pairs("Ali-Ayşe, Mehmet-Fatma")
        self.assertIn(("Ali", "Ayşe"), result)
        self.assertIn(("Fatma", "Mehmet"), result)

    def test_satirla_ayrilmis(self):
        result = parse_forbidden_pairs("Ali-Ayşe\nMehmet-Fatma")
        self.assertEqual(len(result), 2)

    def test_sirasi_onemli_degil(self):
        r1 = parse_forbidden_pairs("Ali-Ayşe")
        r2 = parse_forbidden_pairs("Ayşe-Ali")
        self.assertEqual(r1, r2)

    def test_gecersiz_format(self):
        result = parse_forbidden_pairs("AliAyşe")
        self.assertEqual(result, set())


# ------------------------------------------------------------------------------
# 4. parse_person_limits testleri
# ------------------------------------------------------------------------------
class TestParsePersonLimits(unittest.TestCase):

    def test_bos_girdi(self):
        self.assertEqual(parse_person_limits(""), {})

    def test_tekli_limit(self):
        result = parse_person_limits("Ali:5-10")
        self.assertEqual(result, {"Ali": {"min": 5, "max": 10}})

    def test_coklu_limit(self):
        result = parse_person_limits("Ali:5-10\nAyşe:3-8")
        self.assertEqual(result["Ali"], {"min": 5, "max": 10})
        self.assertEqual(result["Ayşe"], {"min": 3, "max": 8})

    def test_gecersiz_format(self):
        result = parse_person_limits("Ali5-10")
        self.assertEqual(result, {})

    def test_boslukla_isim(self):
        result = parse_person_limits("Ali Veli:2-6")
        self.assertIn("Ali Veli", result)

    def test_bos_isim_atlaniyor(self):
        """':5-10' gibi boş isimli satır atlanmalı"""
        result = parse_person_limits(":5-10")
        self.assertEqual(result, {})

    def test_negatif_min_atlaniyor(self):
        """min negatifse satır atlanmalı"""
        result = parse_person_limits("Ali:-1-10")
        self.assertEqual(result, {})

    def test_negatif_max_atlaniyor(self):
        """max negatifse satır atlanmalı"""
        result = parse_person_limits("Ali:5--2")
        self.assertEqual(result, {})

    def test_min_buyuk_max_atlaniyor(self):
        """min > max ise satır atlanmalı"""
        result = parse_person_limits("Ali:10-5")
        self.assertEqual(result, {})

    def test_hicbir_cizgi_yok(self):
        """'Ali:5' gibi aralık yoksa atlanmalı"""
        result = parse_person_limits("Ali:5")
        self.assertEqual(result, {})

    def test_gecerli_ve_gecersiz_karisik(self):
        """Geçerli satır kabul, geçersiz atlanmalı"""
        result = parse_person_limits("Ali:3-8\nAyşe:10-2\nMehmet:2-6")
        self.assertIn("Ali", result)
        self.assertNotIn("Ayşe", result)  # min > max
        self.assertIn("Mehmet", result)


class TestLimitViolationMessages(unittest.TestCase):
    def test_minimum_limit_ihlali_mesaji_uretilir(self):
        schedule = pd.DataFrame(
            [[True, False], [False, False]],
            index=["Ali", "Ayşe"],
            columns=["G01", "G02"],
        )

        messages = build_limit_violation_messages(
            schedule,
            ["Ali", "Ayşe"],
            {"Ali": {"min": 2, "max": 5}},
        )

        self.assertEqual(
            messages,
            ["🔻 **Ali**: Min 2 nöbet gerekli, şu an 1 atanmış."],
        )

    def test_maksimum_limit_ihlali_mesaji_uretilir(self):
        schedule = pd.DataFrame(
            [[True, True], [False, False]],
            index=["Ali", "Ayşe"],
            columns=["G01", "G02"],
        )

        messages = build_limit_violation_messages(
            schedule,
            ["Ali", "Ayşe"],
            {"Ali": {"min": 0, "max": 1}},
        )

        self.assertEqual(
            messages,
            ["🔺 **Ali**: Max 1 nöbet aşıldı, şu an 2 atanmış."],
        )


class SessionStateMock(dict):
    """Minimal stand-in for Streamlit's session state in history tests."""


class TestAutoScheduleRestore(unittest.TestCase):
    def setUp(self):
        self.isimler = ["Ali", "Ayşe"]
        self.sutunlar = ["1 Pzt", "2 Sal"]
        self.gun_detaylari = {
            "1 Pzt": {"day_num": 1},
            "2 Sal": {"day_num": 2},
        }

    def test_yukleyici_cizelge_ve_tercihleri_session_statee_aktarir(self):
        saved_schedule = pd.DataFrame(
            [[True, False], [False, True]],
            index=self.isimler,
            columns=["1 Çar", "2 Per"],
        )
        saved_preferences = pd.DataFrame(
            [[1, 3], [2, 0]],
            index=self.isimler,
            columns=["1 Çar", "2 Per"],
        )
        session_state = SessionStateMock()

        def mock_load_schedule(name, year, month):
            self.assertEqual((name, year, month), ("Otomatik_2025_01", 2025, 1))
            return self.isimler, saved_schedule, saved_preferences, {}

        restored = restore_auto_saved_schedule(
            session_state, mock_load_schedule, "Otomatik_2025_01", 2025, 1,
            self.isimler, self.sutunlar, self.gun_detaylari,
        )

        self.assertTrue(restored)
        pd.testing.assert_frame_equal(
            session_state["schedule_bool"],
            pd.DataFrame([[True, False], [False, True]], index=self.isimler, columns=self.sutunlar),
        )
        pd.testing.assert_frame_equal(
            session_state["pref_df"],
            pd.DataFrame([[1, 3], [2, 0]], index=self.isimler, columns=self.sutunlar),
        )
        self.assertTrue(session_state["should_regenerate_assignments"])

    def test_yukleyici_yanlis_sayida_deger_dondururse_state_temiz_kalir(self):
        session_state = SessionStateMock()

        def mock_load_schedule(*_):
            return None, None, None

        restored = restore_auto_saved_schedule(
            session_state, mock_load_schedule, "Otomatik_2025_01", 2025, 1,
            self.isimler, self.sutunlar, self.gun_detaylari,
        )

        self.assertFalse(restored)
        self.assertEqual(session_state, {})

    def test_yukleyici_kayitli_rol_atamalarini_aynen_geri_yukler(self):
        saved_schedule = pd.DataFrame(
            [[True, True], [True, True]],
            index=self.isimler,
            columns=["1 Çar", "2 Per"],
        )
        saved_rows = [
            {"Tarih": "01.01.2025 Çar", "Başrol": "Ayşe", "Destek": "Ali"},
            {"Tarih": "02.01.2025 Per", "Başrol": "Ayşe", "Destek": "Ali"},
        ]
        session_state = SessionStateMock()
        day_details = {
            "1 Pzt": {"day_num": 1, "full_date": "01.01.2025 Çar"},
            "2 Sal": {"day_num": 2, "full_date": "02.01.2025 Per"},
        }

        def mock_load_schedule(*_):
            return self.isimler, saved_schedule, None, {}, saved_rows

        restored = restore_auto_saved_schedule(
            session_state, mock_load_schedule, "Otomatik_2025_01", 2025, 1,
            self.isimler, self.sutunlar, day_details, ["Başrol", "Destek"],
        )

        self.assertTrue(restored)
        self.assertEqual(session_state["cached_rows_liste"], saved_rows)
        self.assertEqual(session_state["cached_role_names"], ["Başrol", "Destek"])
        self.assertFalse(session_state["should_regenerate_assignments"])

    def test_eski_gorev_duzeniyle_kayit_guncel_gorevlere_tasinir(self):
        # Kayıt tek görevle (Görev1) yapılmış; şimdi nöbet başına 2 kişi var.
        saved_schedule = pd.DataFrame(
            [[True, False], [True, True]],
            index=self.isimler,
            columns=["1 Çar", "2 Per"],
        )
        saved_rows = [
            {"Tarih": "01.01.2025 Çar", "Görev1": self.isimler[0]},
            {"Tarih": "02.01.2025 Per", "Görev1": self.isimler[1]},
        ]
        session_state = SessionStateMock()
        day_details = {
            "1 Pzt": {"day_num": 1, "full_date": "01.01.2025 Çar"},
            "2 Sal": {"day_num": 2, "full_date": "02.01.2025 Per"},
        }
        restored = restore_auto_saved_schedule(
            session_state, lambda *_: (self.isimler, saved_schedule, None, {}, saved_rows),
            "Otomatik_2025_01", 2025, 1,
            self.isimler, self.sutunlar, day_details, ["AYB", "GYB"],
        )
        self.assertTrue(restored)
        self.assertEqual(session_state["cached_role_names"], ["AYB", "GYB"])
        first, second = session_state["cached_rows_liste"]
        # Gün 1'de iki kişi nöbetçi: ikisi de listede, eski yerleşim korunur.
        self.assertEqual(first["AYB"], self.isimler[0])
        self.assertEqual(first["GYB"], self.isimler[1])
        self.assertEqual(second, {"Tarih": "02.01.2025 Per", "AYB": self.isimler[1], "GYB": "-"})

    def test_gorev_satirlari_yeni_duzene_tasinir(self):
        from nobet_core import align_rows_to_roles
        gun = build_gun_detaylari(2026, 2, 28, [])
        cols = list(gun)[:2]
        schedule = pd.DataFrame([[True, True], [True, False], [False, True]],
                                index=["Ali", "Ayşe", "Can"], columns=cols)
        rows = [{"Tarih": "d1", "A": "Ayşe", "B": "Ali", "C": "-"},
                {"Tarih": "d2", "A": "Can", "B": "-", "C": "Ali"}]
        # Üç görevden ikiye: kaldırılan C'deki kişi boş göreve geçer.
        out = align_rows_to_roles(rows, ["A", "B", "C"], ["X", "Y"], schedule, cols, gun)
        self.assertEqual(out, [{"Tarih": "d1", "X": "Ayşe", "Y": "Ali"},
                               {"Tarih": "d2", "X": "Can", "Y": "Ali"}])

    def test_ayni_kisinin_iki_limiti_bildirilir(self):
        limits, _, errors = parse_person_limits_with_issues("Ali:3-4\nAli:5-6")
        self.assertEqual(limits["Ali"], {"min": 5, "max": 6})
        self.assertTrue(any("birden fazla" in e for e in errors))

    def test_yukleme_hatasi_callback_ile_bildirilebilir_ve_state_degismez(self):
        session_state = SessionStateMock({"existing": "state"})
        errors = []

        def failing_load_schedule(*_):
            raise RuntimeError("database unavailable")

        restored = restore_auto_saved_schedule(
            session_state,
            failing_load_schedule,
            "Otomatik_2025_01",
            2025,
            1,
            self.isimler,
            self.sutunlar,
            self.gun_detaylari,
            on_error=errors.append,
        )

        self.assertFalse(restored)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertEqual(session_state, {"existing": "state"})


class TestRedoHistory(unittest.TestCase):
    def test_undo_current_stateyi_redoya_tasiyip_onceki_snapshoti_dondurur(self):
        session_state = SessionStateMock(
            undo_history=["before-current"],
            redo_history=[],
        )
        current_snapshot = "current"

        restored = undo_history_state(session_state, current_snapshot)

        self.assertEqual(restored, "before-current")
        self.assertEqual(session_state["undo_history"], [])
        self.assertEqual(session_state["redo_history"], ["current"])

    def test_redo_current_stateyi_undoya_tasiyip_redo_snapshotini_dondurur(self):
        session_state = SessionStateMock(
            undo_history=["before-current"],
            redo_history=["after-current"],
        )
        current_snapshot = "current"

        restored = redo_history_state(session_state, current_snapshot)

        self.assertEqual(restored, "after-current")
        self.assertEqual(session_state["undo_history"], ["before-current", "current"])
        self.assertEqual(session_state["redo_history"], [])


class TestPersistenceFeedback(unittest.TestCase):
    def test_excel_kaydi_basarisiz_oldugunda_hata_sonraki_rerunda_kalir(self):
        session_state = SessionStateMock()
        message = (
            "❌ Excel verisi yüklendi ancak çizelge kaydedilemedi. "
            "Veritabanı bağlantısını kontrol edin."
        )

        update_persistence_feedback(session_state, False, message)

        # Aynı session state, Streamlit'in st.rerun() çağrısından sonra da kullanılır.
        self.assertTrue(session_state["auto_save_failed"])
        self.assertEqual(session_state["persistence_error"], message)

    def test_basariyla_kaydedilince_bekleyen_hata_temizlenir(self):
        session_state = SessionStateMock(
            auto_save_failed=True,
            persistence_error="önceki hata",
        )

        update_persistence_feedback(session_state, True)

        self.assertFalse(session_state["auto_save_failed"])
        self.assertNotIn("persistence_error", session_state)


# ------------------------------------------------------------------------------
# 5. build_gun_detaylari testleri
# ------------------------------------------------------------------------------
class TestBuildGunDetaylari(unittest.TestCase):

    def test_ocak_2025_gun_sayisi(self):
        result = build_gun_detaylari(2025, 1, 31, [])
        self.assertEqual(len(result), 31)

    def test_tatiller_nobet_gunu_olarak_isaretlenir(self):
        # Uygulamada tatillerde de nöbet tutulur; gün çıkarılmaz, işaretlenir.
        result = build_gun_detaylari(2025, 1, 31, [1, 2, 3])
        self.assertEqual(len(result), 31)
        self.assertTrue(result["1 Çar"]["holiday"])
        self.assertFalse(result["4 Cmt"]["holiday"])

    def test_anahtarlar_uygulamadaki_sutun_adlari(self):
        result = build_gun_detaylari(2025, 1, 31, [])
        self.assertEqual(list(result)[:3], ["1 Çar", "2 Per", "3 Cum"])

    def test_hafta_sonu_tespiti(self):
        # 2025 Ocak 4 = Cumartesi
        result = build_gun_detaylari(2025, 1, 31, [])
        self.assertTrue(result["4 Cmt"]["weekend"])  # Cumartesi

    def test_hafta_ici_tespiti(self):
        # 2025 Ocak 6 = Pazartesi
        result = build_gun_detaylari(2025, 1, 31, [])
        self.assertFalse(result["6 Pzt"]["weekend"])  # Pazartesi

    def test_gun_numarasi_dogru(self):
        result = build_gun_detaylari(2025, 1, 31, [])
        self.assertEqual(result["15 Çar"]["day_num"], 15)

    def test_tarih_string_formatli(self):
        result = build_gun_detaylari(2025, 1, 31, [])
        self.assertEqual(result["1 Çar"]["full_date"], "01.01.2025 Çar")

    def test_ayni_hafta_sonu_ayni_week(self):
        # Cmt ve Paz aynı hafta sonu -> aynı 'week' değeri olmalı
        result = build_gun_detaylari(2025, 1, 31, [])
        self.assertEqual(result["4 Cmt"]["week"], result["5 Paz"]["week"])

    def test_ardisik_hafta_sonu_week_farki_bir(self):
        # Ardışık hafta sonları tam olarak 1 fark etmeli
        result = build_gun_detaylari(2025, 1, 31, [])
        self.assertEqual(result["11 Cmt"]["week"] - result["4 Cmt"]["week"], 1)

    def test_yil_siniri_week_monoton(self):
        # Ocak 2022: 1 Cmt (ISO hafta 52/2021), 8 Cmt (ISO hafta 1/2022)
        # ISO hafta numarası kullanılsaydı 52 -> 1 olur, +1 mantığı kırılırdı.
        result = build_gun_detaylari(2022, 1, 31, [])
        self.assertEqual(result["8 Cmt"]["week"] - result["1 Cmt"]["week"], 1)


# ------------------------------------------------------------------------------
# 6. run_scheduling_core testleri (algoritma)
# ------------------------------------------------------------------------------
class TestRunSchedulingCore(unittest.TestCase):

    def _build_test_env(self, isimler=None, gun_sayisi=7, kisi_sayisi=2):
        if isimler is None:
            isimler = ["Ali", "Ayşe", "Mehmet", "Fatma"]
        yil, ay = 2025, 1
        tatil = []
        gun_detaylari = build_gun_detaylari(yil, ay, gun_sayisi, tatil)
        sutunlar = list(gun_detaylari.keys())
        df_unwanted = pd.DataFrame(False, index=isimler, columns=sutunlar)
        return isimler, sutunlar, df_unwanted, gun_detaylari, kisi_sayisi

    def test_cikti_dataframe_dogrulugu(self):
        isimler, sutunlar, df_unwanted, gun_detaylari, kisi_sayisi = self._build_test_env()
        schedule, score = run_scheduling_core(
            isimler, sutunlar, df_unwanted, gun_detaylari,
            min_bosluk=1, kisi_sayisi=kisi_sayisi, simulation_count=5
        )
        self.assertIsInstance(schedule, pd.DataFrame)
        self.assertEqual(list(schedule.index), isimler)
        self.assertEqual(list(schedule.columns), sutunlar)

    def test_sifir_simulasyon_reddedilir_ve_girdiyi_degistirmez(self):
        isimler, sutunlar, df_unwanted, gun_detaylari, kisi_sayisi = self._build_test_env()
        previous_schedule = pd.DataFrame(True, index=isimler, columns=sutunlar)
        previous_snapshot = previous_schedule.copy(deep=True)

        with self.assertRaisesRegex(ValueError, "pozitif bir tamsayı"):
            run_scheduling_core(
                isimler, sutunlar, df_unwanted, gun_detaylari,
                min_bosluk=1, kisi_sayisi=kisi_sayisi, simulation_count=0
            )

        pd.testing.assert_frame_equal(previous_schedule, previous_snapshot)

    def test_negatif_simulasyon_reddedilir(self):
        isimler, sutunlar, df_unwanted, gun_detaylari, kisi_sayisi = self._build_test_env()

        with self.assertRaisesRegex(ValueError, "pozitif bir tamsayı"):
            run_scheduling_core(
                isimler, sutunlar, df_unwanted, gun_detaylari,
                min_bosluk=1, kisi_sayisi=kisi_sayisi, simulation_count=-1
            )

    def test_ayni_ikili_her_nobette_yan_yana_dusmez(self):
        # Eskiden 10 kişi/günde 2 kişide bazı ikililer 6 nöbetin 5'ini birlikte tutuyordu.
        isimler = [f"K{i}" for i in range(10)]
        gun_detaylari = build_gun_detaylari(2026, 11, 30, [])
        sutunlar = list(gun_detaylari)
        df_unwanted = pd.DataFrame(False, index=isimler, columns=sutunlar)
        random.seed(11)
        for _ in range(5):
            schedule, _ = run_scheduling_core(
                isimler, sutunlar, df_unwanted, gun_detaylari,
                min_bosluk=1, kisi_sayisi=2, simulation_count=50,
            )
            together = {}
            for c in sutunlar:
                pair = tuple(sorted(schedule.index[schedule[c]]))
                together[pair] = together.get(pair, 0) + 1
            self.assertLessEqual(max(together.values()), 3)
            totals = schedule.sum(axis=1)
            self.assertLessEqual(totals.max() - totals.min(), 1)

    def test_normal_simulasyon_cizelge_uretmege_devam_eder(self):
        isimler, sutunlar, df_unwanted, gun_detaylari, kisi_sayisi = self._build_test_env()

        schedule, score = run_scheduling_core(
            isimler, sutunlar, df_unwanted, gun_detaylari,
            min_bosluk=1, kisi_sayisi=kisi_sayisi, simulation_count=1
        )

        self.assertIsInstance(schedule, pd.DataFrame)
        self.assertTrue(np.isfinite(score))

    def test_her_gun_dogru_kisi_sayisi(self):
        isimler, sutunlar, df_unwanted, gun_detaylari, kisi_sayisi = self._build_test_env(
            isimler=["Ali","Ayşe","Mehmet","Fatma","Can","Zeynep"]
        )
        schedule, _ = run_scheduling_core(
            isimler, sutunlar, df_unwanted, gun_detaylari,
            min_bosluk=0, kisi_sayisi=2, simulation_count=5
        )
        for col in sutunlar:
            assigned = schedule[col].sum()
            self.assertEqual(assigned, 2, f"{col} gününde {assigned} kişi atandı, beklenen 2")

    def test_eksik_simulasyon_sonucu_gun_ve_kisi_sayisiyla_saptanir(self):
        isimler, sutunlar, df_unwanted, gun_detaylari, _ = self._build_test_env(
            gun_sayisi=3
        )
        df_unwanted.loc[:, sutunlar[0]] = True
        schedule, _ = run_scheduling_core(
            isimler, sutunlar, df_unwanted, gun_detaylari,
            min_bosluk=0, kisi_sayisi=2, simulation_count=3
        )
        self.assertEqual(missing_shift_coverage(schedule, sutunlar, 2), [(sutunlar[0], 0)])

    def test_eksik_kapsama_kismi_atamayi_da_saptar(self):
        schedule = pd.DataFrame({
            "G1": [True, False, False],
            "G2": [True, True, False],
            "G3": [False, False, False],
        }, index=["Ali", "Ayşe", "Can"])
        self.assertEqual(missing_shift_coverage(schedule, list(schedule.columns), 2),
                         [("G1", 1), ("G3", 0)])
        self.assertEqual(missing_shift_coverage(schedule, ["G2"], 2), [])

    def test_yasak_cift_atanmasin(self):
        isimler = ["Ali", "Ayşe", "Mehmet", "Fatma", "Can", "Zeynep"]
        _, sutunlar, df_unwanted, gun_detaylari, _ = self._build_test_env(isimler=isimler)
        forbidden = {("Ali", "Ayşe")}
        schedule, _ = run_scheduling_core(
            isimler, sutunlar, df_unwanted, gun_detaylari,
            min_bosluk=0, kisi_sayisi=2,
            forbidden_pairs=forbidden, simulation_count=20
        )
        for col in sutunlar:
            assigned = schedule.index[schedule[col]].tolist()
            if "Ali" in assigned and "Ayşe" in assigned:
                self.fail(f"{col} gününde Ali ve Ayşe birlikte atandı!")

    def test_musait_olmayan_gun_atanmasin(self):
        isimler = ["Ali", "Ayşe", "Mehmet", "Fatma"]
        _, sutunlar, df_unwanted, gun_detaylari, _ = self._build_test_env(isimler=isimler)
        # Ali'yi ilk 3 güne müsait değil yap
        for col in sutunlar[:3]:
            df_unwanted.at["Ali", col] = True
        schedule, _ = run_scheduling_core(
            isimler, sutunlar, df_unwanted, gun_detaylari,
            min_bosluk=0, kisi_sayisi=2, simulation_count=10
        )
        for col in sutunlar[:3]:
            self.assertFalse(schedule.at["Ali", col], f"Ali {col} gününe atandı ama müsait değil!")

    def test_denge_skoru_makul(self):
        isimler = ["Ali", "Ayşe", "Mehmet", "Fatma", "Can", "Zeynep"]
        _, sutunlar, df_unwanted, gun_detaylari, _ = self._build_test_env(
            isimler=isimler, gun_sayisi=28
        )
        schedule, score = run_scheduling_core(
            isimler, sutunlar, df_unwanted, gun_detaylari,
            min_bosluk=1, kisi_sayisi=2, simulation_count=20
        )
        totals = [schedule.loc[p].sum() for p in isimler]
        spread = max(totals) - min(totals)
        self.assertLessEqual(spread, 5, f"Dağılım farkı çok yüksek: {spread} (max-min)")

    def test_tek_kisi_nobeti(self):
        isimler = ["Ali", "Ayşe", "Mehmet"]
        _, sutunlar, df_unwanted, gun_detaylari, _ = self._build_test_env(
            isimler=isimler, gun_sayisi=5, kisi_sayisi=1
        )
        schedule, _ = run_scheduling_core(
            isimler, sutunlar, df_unwanted, gun_detaylari,
            min_bosluk=0, kisi_sayisi=1, simulation_count=5
        )
        for col in sutunlar:
            self.assertEqual(schedule[col].sum(), 1)

    def test_maksimum_limit_asimi(self):
        isimler = ["Ali", "Ayşe", "Mehmet", "Fatma", "Can"]
        _, sutunlar, df_unwanted, gun_detaylari, _ = self._build_test_env(
            isimler=isimler, gun_sayisi=20
        )
        person_limits = {"Ali": {"min": 0, "max": 2}}
        schedule, _ = run_scheduling_core(
            isimler, sutunlar, df_unwanted, gun_detaylari,
            min_bosluk=0, kisi_sayisi=2,
            person_limits=person_limits, simulation_count=10
        )
        ali_total = schedule.loc["Ali"].sum()
        self.assertLessEqual(ali_total, 2, f"Ali'ye {ali_total} nöbet düştü, max 2 olmalıydı")

    def test_tercih_yesil_oncelik(self):
        isimler = ["Ali", "Ayşe", "Mehmet", "Fatma"]
        _, sutunlar, df_unwanted, gun_detaylari, _ = self._build_test_env(isimler=isimler, gun_sayisi=5)
        df_preferred = pd.DataFrame(0, index=isimler, columns=sutunlar)
        # Ali tüm günlerde yeşil (tercih eder)
        for col in sutunlar:
            df_preferred.at["Ali", col] = 1
        schedule, _ = run_scheduling_core(
            isimler, sutunlar, df_unwanted, gun_detaylari,
            min_bosluk=0, kisi_sayisi=2,
            df_preferred=df_preferred, simulation_count=30
        )
        ali_total = schedule.loc["Ali"].sum()
        # Ali yeşil tercihli olduğu için ortalamadan fazla nöbet almalı
        avg = schedule.values.sum() / len(isimler)
        self.assertGreaterEqual(ali_total, avg * 0.8,
                                f"Ali yeşil tercih koymasına rağmen beklenenden az nöbet aldı: {ali_total}")

    def test_aradan_sonra_hafta_sonu_cifti_sayilir(self):
        # Kasım 2026 haftaları: 1, sonra 7-8, ara, 21-22 ve 28-29.
        # İki çift var. Serinin sonu 1 sayardı; seçim 2 saymalı.
        days = build_gun_detaylari(2026, 11, 30, [])
        cols = list(days)
        schedule = pd.DataFrame(False, index=["Ali"], columns=cols)
        by_week = {}
        for col in cols:
            if days[col]["weekend"]:
                by_week.setdefault(days[col]["week"], []).append(col)
        weeks = sorted(by_week)
        for week in (weeks[0], weeks[1], weeks[3], weeks[4]):
            for col in by_week[week]:
                schedule.at["Ali", col] = True
        self.assertEqual(weekend_pair_count(schedule, cols, days), 2)

    def test_unutulan_ciftler_secim_puanina_girer(self):
        # 6 kişi, tohum 0, 20 simülasyon. Seçilen liste aynı kalır.
        # Sayaç serinin sonunda 4 görüyordu; gerçek çift 6. Puan 1000 yükselir.
        names = [f"K{i}" for i in range(6)]
        days = build_gun_detaylari(2026, 11, 30, [])
        cols = list(days)
        unwanted = pd.DataFrame(False, index=names, columns=cols)
        random.seed(0)
        schedule, score = run_scheduling_core(
            names, cols, unwanted, days, min_bosluk=1, kisi_sayisi=2, simulation_count=20,
        )
        self.assertEqual(weekend_pair_count(schedule, cols, days), 6)
        self.assertAlmostEqual(score, 3320.7735026918963, places=4)

    def test_tum_hafta_icinden_kacinan_toplamda_birden_fazla_geride_kalmaz(self):
        # K0 hafta içinin tamamından kaçınıyor. Yalnız hafta sonu yetmezse
        # dengeleme kaçınılan güne de geçebilir; fark yine en fazla 1 kalır.
        names = [f"K{i}" for i in range(8)]
        days = build_gun_detaylari(2026, 11, 30, [])
        cols = list(days)
        unwanted = pd.DataFrame(False, index=names, columns=cols)
        preferred = pd.DataFrame(0, index=names, columns=cols)
        for col in cols:
            if not days[col]["weekend"] and not days[col]["holiday"]:
                preferred.at["K0", col] = 2
        random.seed(4)
        schedule, _ = run_scheduling_core(
            names, cols, unwanted, days, min_bosluk=1, kisi_sayisi=2,
            df_preferred=preferred, simulation_count=30,
        )
        totals = {p: int(schedule.loc[p].sum()) for p in names}
        self.assertLessEqual(max(totals.values()) - min(totals.values()), 1)
        avoid = [
            days[c]["day_num"] for c in cols
            if schedule.at["K0", c] and int(preferred.at["K0", c]) == 2
        ]
        self.assertGreater(len(avoid), 0)
        for p in names:
            got = sorted(days[c]["day_num"] for c in cols if schedule.at[p, c])
            self.assertTrue(all(b - a > 1 for a, b in zip(got, got[1:])), p)
        # Üç pazartesiden kaçınmak farkı zaten kapatıyorsa kaçınılan gün boş kalır.
        light = preferred.copy()
        light.loc[:, :] = 0
        for day in (2, 9, 16):
            col = next(c for c in cols if days[c]["day_num"] == day)
            light.at["K0", col] = 2
        random.seed(4)
        modest, _ = run_scheduling_core(
            names, cols, unwanted, days, min_bosluk=1, kisi_sayisi=2,
            df_preferred=light, simulation_count=30,
        )
        modest_avoid = sum(
            1 for c in cols if modest.at["K0", c] and int(light.at["K0", c]) == 2
        )
        modest_totals = [int(modest.loc[p].sum()) for p in names]
        self.assertLessEqual(max(modest_totals) - min(modest_totals), 1)
        self.assertEqual(modest_avoid, 0)


class TestLargeSchedulePerformance(unittest.TestCase):
    def _large_schedule(self):
        isimler = [f"Kişi {number}" for number in range(50)]
        gun_detaylari = build_gun_detaylari(2025, 1, 31, [])
        sutunlar = list(gun_detaylari.keys())
        schedule = pd.DataFrame(False, index=isimler, columns=sutunlar)
        rows_liste = []
        for index, column in enumerate(sutunlar):
            assigned = [isimler[index % len(isimler)], isimler[(index + 1) % len(isimler)]]
            schedule.loc[assigned, column] = True
            rows_liste.append({
                "Tarih": gun_detaylari[column]["full_date"],
                "Görev1": assigned[0],
                "Görev2": assigned[1],
            })
        return isimler, sutunlar, gun_detaylari, schedule, rows_liste

    def test_tek_hucre_yalnizca_etkilenen_gunu_bildirir(self):
        _, sutunlar, _, schedule, _ = self._large_schedule()
        previous = schedule.copy(deep=True)
        schedule.at[schedule.index[0], sutunlar[15]] = not schedule.at[schedule.index[0], sutunlar[15]]

        self.assertEqual(changed_schedule_columns(previous, schedule), [sutunlar[15]])
        self.assertEqual(changed_schedule_columns(schedule, schedule), [])

    def test_elli_kisi_otuzbir_gun_analizi_hizli_kalir(self):
        isimler, _, gun_detaylari, schedule, rows_liste = self._large_schedule()

        started = time.perf_counter()
        stats_load, stats_finance, pair_display = build_schedule_analysis(
            schedule, rows_liste, ["Görev1", "Görev2"], isimler,
            gun_detaylari, {}, zorunlu_saat=160, nobet_ucreti=1.0,
        )
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 1.0, f"Büyük çizelge analizi çok yavaş: {elapsed:.3f}s")
        self.assertEqual(stats_load.shape[0], 50)
        self.assertEqual(stats_finance.shape[0], 50)
        self.assertEqual(pair_display.shape, (50, 50))
        self.assertTrue(schedule_fingerprint(schedule))

    def test_buyuk_cizelge_hizli_duzenleyiciye_gecer(self):
        self.assertTrue(use_compact_schedule_editor(50, 31))
        self.assertFalse(use_compact_schedule_editor(10, 31))

    def test_hizli_tercih_duzenleyicisi_gecersiz_kodlari_reddeder(self):
        valid = pd.DataFrame(
            [[0, 1], [2, 4]],
            index=["Ali", "Ayşe"],
            columns=["G01", "G02"],
            dtype=object,
        )
        self.assertTrue(normalize_preference_grid(valid).equals(valid.astype(int)))

        for invalid_value in [None, 1.5, 5, -1]:
            invalid = valid.copy()
            invalid.at["Ali", "G01"] = invalid_value
            self.assertIsNone(normalize_preference_grid(invalid))


class TestPersonRoleDisplay(unittest.TestCase):
    def test_kisinin_gundeki_gorevini_doner(self):
        rows = [
            {"Tarih": "01.01.2025 Çar", "Görev1": "Ali", "Görev2": "Ayşe"},
            {"Tarih": "02.01.2025 Per", "Görev1": "Ayşe", "Görev2": "Ali"},
        ]

        self.assertEqual(find_person_role(rows, ["Görev1", "Görev2"], 0, "Ali"), "Görev1")
        self.assertEqual(find_person_role(rows, ["Görev1", "Görev2"], 1, "Ali"), "Görev2")

    def test_gorev_yeri_degistiginde_guncel_rolu_doner(self):
        rows = [{"Tarih": "01.01.2025 Çar", "Görev1": "Ali", "Görev2": "Ayşe"}]

        rows[0]["Görev1"], rows[0]["Görev2"] = rows[0]["Görev2"], rows[0]["Görev1"]

        self.assertEqual(find_person_role(rows, ["Görev1", "Görev2"], 0, "Ali"), "Görev2")

    def test_eslesmeyen_kisi_icin_rol_donmez(self):
        rows = [{"Tarih": "01.01.2025 Çar", "Görev1": "Ali"}]

        self.assertIsNone(find_person_role(rows, ["Görev1", "Görev2"], 0, "Mehmet"))

def _writes(cursor):
    """Kayıt sorguları; sıraya sokan kilit (pg_advisory_xact_lock, lock_timeout) hariç."""
    return [item for item in cursor.executed
            if "pg_advisory_xact_lock" not in item[0] and "lock_timeout" not in item[0]
            and item[0] != "SELECT value FROM app_settings WHERE key = %s"]


class _FakeDbCursor:
    def __init__(self, schedule_row=None, data_rows=None):
        self.schedule_row = schedule_row
        self.data_rows = data_rows or []
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchone(self):
        return self.schedule_row if self.schedule_row is not None else (1,)

    def fetchall(self):
        return self.data_rows


class TestPreferenceGridAccessibility(unittest.TestCase):
    """Streamlit widget'larını hedefleyen mobil tercih stillerini korur."""

    @classmethod
    def setUpClass(cls):
        cls.app_source = _app_source()

    def test_tercih_grid_gercek_anahtarli_container_kullanir(self):
        self.assertIn('grid_key = "preference-grid"', self.app_source)
        self.assertIn("grid_container = st.container(", self.app_source)
        self.assertIn("key=grid_key", self.app_source)
        self.assertIn('legend_container = st.container(key="preference-legend"', self.app_source)
        self.assertIn(".st-key-preference-grid", self.app_source)
        self.assertIn(".st-key-preference-legend", self.app_source)

    def test_mobil_hucreler_bir_haftayi_sigdirir_ve_renk_anlamli_etiketler_tasir(self):
        mobile_rule = (
            "@media (max-width: 768px) {{\n"
            "    /*\n"
            "     * This must follow the shared grid-button rule above."
        )
        self.assertIn(mobile_rule, self.app_source)
        mobile_rule_position = self.app_source.index(mobile_rule)
        shared_button_position = self.app_source.index(
            ".st-key-schedule-grid .stButton > button,"
        )
        shared_hover_position = self.app_source.index(
            ".st-key-schedule-grid .stButton > button:hover,"
        )
        self.assertGreater(mobile_rule_position, shared_button_position)
        self.assertGreater(mobile_rule_position, shared_hover_position)
        mobile_css = self.app_source[mobile_rule_position:]
        # 390 px telefonda 88 px isim + 7 × 36 px gün sığar.
        self.assertIn("min-width: 34px !important", mobile_css)
        self.assertIn("min-height: 40px !important", mobile_css)
        self.assertIn("flex: 0 0 88px !important", mobile_css)
        self.assertIn("transform: none !important", mobile_css)
        for color in ("#166534", "#92400e", "#991b1b"):
            self.assertIn(color, self.app_source)
        self.assertIn("get_preference_grid_state(", self.app_source)

    def test_renk_kurallari_yeni_streamlit_sutunlarini_ve_iki_modu_hedefler(self):
        # Streamlit 1.38+ sütunları data-testid="stColumn" ile işaretler;
        # yalnızca eski "column" adını hedefleyen kurallar hiç eşleşmez.
        self.assertNotRegex(
            self.app_source,
            r"""(?<!:is\()\[data-testid=["']column["']\](?!,\[data-testid=["']stColumn)""",
        )
        for state in ("0", "1", "2", "3", "conflict"):
            self.assertIn(
                ":is(.st-key-preference-grid, .st-key-schedule-grid) "
                ':is([data-testid="column"],[data-testid="stColumn"])'
                f":has(.preference-state-{state}) .stButton > button",
                self.app_source,
            )

    def test_hucre_ve_lejant_ayni_durum_sozlugunu_kullanir(self):
        legend = get_preference_grid_legend()
        self.assertEqual(len(legend), 11)

        for legend_state in legend:
            cell_state = get_preference_grid_state(
                legend_state["key"].startswith("assigned_"),
                {
                    "assigned_neutral": 0,
                    "assigned_preferred": 1,
                    "assigned_avoid": 2,
                    "assigned_unavailable": 3,
                    "assigned_not_available": 4,
                    "conflict": 3,
                    "neutral": 0,
                    "preferred": 1,
                    "avoid": 2,
                    "unavailable": 3,
                    "not_available": 4,
                }[legend_state["key"]],
                legend_state["key"] == "conflict",
            )
            self.assertEqual(cell_state, legend_state)

        self.assertEqual(
            get_preference_grid_state(True, 3, True)["symbol"],
            "🔺",
        )

    def test_grid_hucrelerinde_istenmeyen_tooltip_yoktur(self):
        self.assertNotIn("Değiştirmek için dokunun", self.app_source)
        self.assertNotIn('help=f"{escape_html(person)}', self.app_source)

    def test_simulasyon_manuel_degisiklikte_onay_istemez(self):
        self.assertNotIn("sim_pending_confirm", self.app_source)
        self.assertIn('st.session_state.run_simulation = True', self.app_source)
        self.assertNotIn("100 Simülasyon yapıldı", self.app_source)
        self.assertIn('st.toast("En adil sonuç seçildi. " + simulation_summary(', self.app_source)


# ------------------------------------------------------------------------------
# Çalıştır
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)

class TestFixedShifts(unittest.TestCase):
    def setUp(self):
        from nobet_core import update_locked_cells, active_locks, locks_to_json, locks_from_saved
        self.update_locked_cells = update_locked_cells
        self.active_locks = active_locks
        self.locks_to_json = locks_to_json
        self.locks_from_saved = locks_from_saved
        self.days = build_gun_detaylari(2026, 11, 30, [])
        self.cols = list(self.days)
        self.names = ["Ali", "Ayşe", "Can", "Deniz", "Ece"]

    def test_simulasyon_sabitleri_korur_ve_kurallara_uyar(self):
        unwanted = pd.DataFrame(False, index=self.names, columns=self.cols)
        fixed = {("Ali", self.cols[14]), ("Ali", self.cols[19]), ("Ayşe", self.cols[14])}
        for seed in range(4):
            random.seed(seed)
            schedule, _ = run_scheduling_core(
                self.names, self.cols, unwanted, self.days, 1, 2, simulation_count=20,
                fixed_assignments=fixed, person_limits={"Ali": {"min": 0, "max": 4}},
                forbidden_pairs={("Ayşe", "Deniz")},
            )
            for person, column in fixed:
                self.assertTrue(schedule.at[person, column])
            self.assertEqual(sorted(schedule.index[schedule[self.cols[14]]]), ["Ali", "Ayşe"])
            ali_days = [self.days[c]["day_num"] for c in self.cols if schedule.at["Ali", c]]
            self.assertLessEqual(len(ali_days), 4)
            for neighbour in (14, 16, 19, 21):
                self.assertNotIn(neighbour, ali_days)
            self.assertEqual(missing_shift_coverage(schedule, self.cols, 2), [])

    def test_yasak_cift_sabitin_yanina_konmaz(self):
        unwanted = pd.DataFrame(False, index=self.names, columns=self.cols)
        random.seed(0)
        schedule, _ = run_scheduling_core(
            self.names, self.cols, unwanted, self.days, 1, 2, simulation_count=10,
            fixed_assignments={("Ayşe", self.cols[3])}, forbidden_pairs={("Ayşe", "Deniz")},
        )
        self.assertFalse(schedule.at["Deniz", self.cols[3]])

    def test_kilit_yardimcilari(self):
        locks = self.update_locked_cells(set(), [("Ali", "1 Paz")], [])
        locks = self.update_locked_cells(locks, [("Can", "2 Pzt")], [("Ali", "1 Paz")])
        self.assertEqual(locks, {("Can", "2 Pzt")})
        schedule = pd.DataFrame(False, index=["Ali", "Can"], columns=["1 Paz", "2 Pzt"])
        schedule.at["Can", "2 Pzt"] = True
        self.assertEqual(self.active_locks(locks | {("Eski", "1 Paz")}, schedule), {("Can", "2 Pzt")})
        self.assertEqual(self.locks_to_json(locks), [["Can", "2 Pzt"]])

    def test_sabitler_kaydedilip_geri_yuklenir(self):
        saved = pd.DataFrame(False, index=["Ali", "Can"], columns=["1 Paz", "2 Pzt"])
        saved.at["Ali", "1 Paz"] = True
        saved.at["Can", "2 Pzt"] = True
        days = {"1 Paz": {"day_num": 1, "full_date": "01.11.2026 Paz"},
                "2 Pzt": {"day_num": 2, "full_date": "02.11.2026 Pzt"}}
        state = {}
        settings = {"locked": [["Ali", "1 Paz"], ["Kimse", "2 Pzt"]]}
        self.assertTrue(restore_auto_saved_schedule(
            state, lambda *a: (["Ali", "Can"], saved, None, settings), "x", 2026, 11,
            ["Ali", "Can"], ["1 Paz", "2 Pzt"], days,
        ))
        self.assertEqual(state["locked_cells"], {("Ali", "1 Paz")})

    def test_uygulama_baglantilari(self):
        source = _app_source()
        self.assertIn('fixed_assignments=active_locks(st.session_state.get("locked_cells"), st.session_state.schedule_bool)', source)
        self.assertIn('settings_dict={"locked": locks_to_json(_active_locks),', source)
        self.assertIn("'locked': set(st.session_state.get('locked_cells', set())),", source)
        self.assertIn(':has(.cell-locked) .stButton > button', source)
        self.assertIn("st.session_state.locked_cells = set()\n                st.session_state.hidden_people = {}\n                st.session_state.cached_rows_liste = None", source)


class TestAlgorithmFixes(unittest.TestCase):
    def setUp(self):
        self.days = build_gun_detaylari(2026, 11, 30, [])
        self.cols = list(self.days)

    def test_tek_kisi_musaitse_gun_bos_kalmaz(self):
        names = ["Ali", "Ayşe", "Can", "Deniz"]
        unwanted = pd.DataFrame(False, index=names, columns=self.cols)
        unwanted.loc[["Ayşe", "Can", "Deniz"], self.cols[9]] = True
        random.seed(1)
        schedule, _ = run_scheduling_core(names, self.cols, unwanted, self.days, 1, 2, simulation_count=20)
        self.assertEqual(schedule.index[schedule[self.cols[9]]].tolist(), ["Ali"])
        self.assertEqual(missing_shift_coverage(schedule, self.cols, 2), [(self.cols[9], 1)])

    def test_minimum_limit_saglanir(self):
        names = ["Ali", "Ayşe", "Can", "Deniz"]
        unwanted = pd.DataFrame(False, index=names, columns=self.cols)
        for seed in range(3):
            random.seed(seed)
            schedule, _ = run_scheduling_core(
                names, self.cols, unwanted, self.days, 0, 2, simulation_count=50,
                person_limits={"Ali": {"min": 18, "max": 999}},
            )
            self.assertGreaterEqual(int(schedule.loc["Ali"].sum()), 18)

    def test_kayittaki_eksik_hucre_nobet_sayilmaz(self):
        saved = pd.DataFrame({"1 Paz": [True, None]}, index=["Ali", "Ayşe"])
        state = {}
        days = {"1 Paz": {"day_num": 1, "full_date": "01.11.2026 Paz"}}
        self.assertTrue(restore_auto_saved_schedule(
            state, lambda *a: (["Ali", "Ayşe"], saved, None, {}), "x", 2026, 11,
            ["Ali", "Ayşe"], ["1 Paz"], days,
        ))
        self.assertTrue(state["schedule_bool"].at["Ali", "1 Paz"])
        self.assertFalse(state["schedule_bool"].at["Ayşe", "1 Paz"])

    def test_uygulama_tekrar_eden_isimleri_ayiklar(self):
        source = _app_source()
        self.assertIn("isimler = list(dict.fromkeys(_raw_isimler))", source)
        self.assertNotIn("itertuples(index=False), rows_liste", source)


class TestPreviousMonths(unittest.TestCase):
    def test_onceki_ay_yil_gecisi(self):
        self.assertEqual(previous_period(2026, 1), (2025, 12))
        self.assertEqual(previous_period(2026, 3, back=3), (2025, 12))
        self.assertEqual(previous_period(2026, 10), (2026, 9))

    def test_son_nobet_bu_aya_gore_gun(self):
        prev = pd.DataFrame(False, index=["Ali", "Ayşe", "Can"], columns=["28 Pzt", "29 Sal", "30 Çar"])
        prev.at["Ali", "30 Çar"] = True
        prev.at["Ayşe", "28 Pzt"] = True
        prev.at["Ayşe", "29 Sal"] = True
        self.assertEqual(last_shift_offsets(prev, 30), {"Ali": 0, "Ayşe": -1})

    def test_ozel_gun_sayilari_hafta_sonu_ve_tatil(self):
        # Ekim 2026: 3 Cmt, 4 Paz; 29 tatil
        sched = pd.DataFrame(False, index=["Ali", "Ayşe"], columns=["2 Cum", "3 Cmt", "4 Paz", "29 Per"])
        sched.loc["Ali", ["3 Cmt", "4 Paz", "29 Per"]] = True
        sched.at["Ayşe", "2 Cum"] = True
        self.assertEqual(special_shift_counts(sched, 2026, 10, [29]), {"Ali": 3})

    def test_gecmis_farki_sinirlanir(self):
        adjustment = history_special_adjustment({"Ali": 9, "Ayşe": 3}, ["Ali", "Ayşe", "Yeni"])
        self.assertEqual(adjustment["Ali"], 2)
        self.assertEqual(adjustment["Yeni"], -2)
        self.assertAlmostEqual(adjustment["Ayşe"], -1)

    def _env(self):
        names = ["Ali", "Ayşe", "Can", "Deniz"]
        days = build_gun_detaylari(2026, 10, 31, [])
        cols = list(days)
        return names, cols, pd.DataFrame(False, index=names, columns=cols), days

    def test_onceki_ayin_son_gunu_nobet_tutan_1inde_atanmaz(self):
        names, cols, unwanted, days = self._env()
        random.seed(3)
        for _ in range(5):
            schedule, _ = run_scheduling_core(
                names, cols, unwanted, days, min_bosluk=1, kisi_sayisi=2,
                simulation_count=3, initial_last_shift={"Ali": 0},
            )
            self.assertFalse(schedule.at["Ali", cols[0]])

    def test_onceki_ayin_son_hafta_sonu_bu_ayin_ilk_hafta_sonuna_yazilmaz(self):
        # 31 Ekim 2026 Cumartesi. Ayşe o gün nöbet tuttuysa 7-8 Kasım'a
        # tercihen yazılmaz. 1 Kasım dinlenme kuralıyla zaten boştur.
        names = ["Ayşe", "Mehmet", "Zeynep", "Can", "Elif", "Burak", "Selin", "Emre"]
        days = build_gun_detaylari(2026, 11, 30, [])
        cols = list(days)
        unwanted = pd.DataFrame(False, index=names, columns=cols)
        by_day = {days[c]["day_num"]: c for c in cols}
        random.seed(0)
        schedule, _ = run_scheduling_core(
            names, cols, unwanted, days, min_bosluk=1, kisi_sayisi=2,
            simulation_count=15, initial_last_shift={"Ayşe": 0},
        )
        self.assertFalse(bool(schedule.at["Ayşe", by_day[1]]))
        self.assertFalse(
            bool(schedule.at["Ayşe", by_day[7]]) or bool(schedule.at["Ayşe", by_day[8]])
        )
        from nobet_core import prior_weekend_weeks
        self.assertEqual(prior_weekend_weeks(days, {"Ayşe": 0})["Ayşe"], days[by_day[1]]["week"])
        # 30 Ekim Cuma hafta sonu değildir; ertesi hafta sonu bu yüzden işaretlenmez.
        self.assertNotIn("Ayşe", prior_weekend_weeks(days, {"Ayşe": -1}))
        from nobet_iyilestir import improve_schedule
        improved, _, _ = improve_schedule(
            schedule, names, cols, unwanted, days, 1, 2,
            initial_last_shift={"Ayşe": 0}, time_limit=8,
        )
        result = improved if improved is not None else schedule
        self.assertFalse(
            bool(result.at["Ayşe", by_day[7]]) or bool(result.at["Ayşe", by_day[8]])
        )

    def test_gecmiste_fazla_hafta_sonu_tutan_bu_ay_daha_az_alir(self):
        names, cols, unwanted, days = self._env()
        special_cols = [c for c in cols if days[c]["weekend"]]
        random.seed(5)
        heavy, light = 0, 0
        for _ in range(4):
            schedule, _ = run_scheduling_core(
                names, cols, unwanted, days, min_bosluk=1, kisi_sayisi=2, simulation_count=10,
                history_special={"Ali": 2, "Ayşe": -2, "Can": 0, "Deniz": 0},
            )
            heavy += int(schedule.loc["Ali", special_cols].sum())
            light += int(schedule.loc["Ayşe", special_cols].sum())
        self.assertLess(heavy, light)

    def test_anlik_uyari_onceki_ayi_bilir(self):
        cols = ["1 Per", "2 Cum"]
        days = {"1 Per": {"day_num": 1}, "2 Cum": {"day_num": 2}}
        schedule = pd.DataFrame([[True, False]], index=["Ali"], columns=cols)
        warnings = assignment_warnings(schedule, "Ali", "1 Per", cols, days, 1,
                                       previous_last_shift={"Ali": 0})
        self.assertEqual(len(warnings), 1)
        self.assertIn("önceki ayın son günü", warnings[0])
        self.assertEqual(
            assignment_warnings(schedule, "Ali", "1 Per", cols, days, 1, previous_last_shift={"Ali": -3}),
            [],
        )


class TestCrossMonthBalance(unittest.TestCase):
    """Aylar arası toplam denge, bayram takibi ve nöbetleri aya yayma."""

    def test_toplam_ve_bayram_sayilari(self):
        sched = pd.DataFrame(False, index=["Ali", "Ayşe", "Can"], columns=["20 Cum", "21 Cmt", "23 Pzt"])
        sched.loc["Ali", ["20 Cum", "21 Cmt"]] = True
        sched.at["Ayşe", "23 Pzt"] = True
        self.assertEqual(schedule_totals(sched), {"Ali": 2, "Ayşe": 1, "Can": 0})
        self.assertEqual(holiday_shift_counts(sched, [20, 21, 22]), {"Ali": 2, "Ayşe": 0, "Can": 0})

    def test_gecmis_farki_o_ay_ekipte_olanlara_gore(self):
        months = [
            {"Ali": 9, "Ayşe": 7, "Can": 8},
            {"Ali": 8, "Ayşe": 7, "Can": 6},
        ]
        adjustment = history_adjustment(months, ["Ali", "Ayşe", "Can", "Yeni"])
        self.assertEqual(adjustment["Ali"], 2)  # +1 ve +1
        self.assertEqual(adjustment["Ayşe"], -1)
        self.assertEqual(adjustment["Can"], -1)
        # Yeni gelen geçmişte yoktu: borçlu sayılmaz.
        self.assertEqual(adjustment["Yeni"], 0)
        self.assertEqual(history_adjustment([{"Ali": 30, "Ayşe": 0}], ["Ali", "Ayşe"])["Ali"], 2)

    def test_izinli_ay_eksik_sayilmaz(self):
        # Ayşe ayın yarısında izinliydi: 15 günde 5 nöbet, Ali 30 günde 10 nöbet.
        months = [{"Ali": 10, "Ayşe": 5}]
        plain = history_adjustment(months, ["Ali", "Ayşe"])
        self.assertEqual(plain["Ayşe"], -2)
        fair = history_adjustment(months, ["Ali", "Ayşe"], monthly_available=[{"Ali": 30, "Ayşe": 15}])
        self.assertEqual(fair, {"Ali": 0, "Ayşe": 0})
        # Müsait olduğu halde az tutan yine eksik sayılır.
        short = history_adjustment([{"Ali": 10, "Ayşe": 4}], ["Ali", "Ayşe"],
                                   monthly_available=[{"Ali": 30, "Ayşe": 15}])
        self.assertLess(short["Ayşe"], 0)
        # Müsaitlik bilgisi olmayan ay eski usulle (ortalama) hesaplanır.
        mixed = history_adjustment(months, ["Ali", "Ayşe"], monthly_available=[None])
        self.assertEqual(mixed, plain)

    def test_kayitli_tercihten_musait_degil_gunleri_sayilir(self):
        pref = pd.DataFrame(0, index=["Ali", "Ayşe"], columns=["1 Pzt", "2 Sal", "3 Çar"])
        pref.at["Ali", "1 Pzt"] = 3
        pref.at["Ali", "2 Sal"] = 3
        pref.at["Ayşe", "3 Çar"] = 2  # kaçınma, müsait değil değil
        self.assertEqual(unavailable_day_counts(pref.to_json()), {"Ali": 2})
        self.assertEqual(unavailable_day_counts(None), {})
        self.assertEqual(unavailable_day_counts("bozuk"), {})

    def test_gecmiste_fazla_toplami_olan_bu_ay_1_fazla_almaz(self):
        names = [f"K{i}" for i in range(7)]
        days = build_gun_detaylari(2026, 11, 30, [])
        cols = list(days)
        unwanted = pd.DataFrame(False, index=names, columns=cols)
        random.seed(11)
        for _ in range(5):
            schedule, _ = run_scheduling_core(
                names, cols, unwanted, days, min_bosluk=1, kisi_sayisi=2, simulation_count=10,
                history_total={"K0": 2, "K1": 2, "K2": 2},
            )
            totals = schedule.sum(axis=1)
            # 60 nöbet / 7 kişi: dördü 9, üçü 8 alır; fazlası olanlar 8'lerden olur.
            self.assertLessEqual(totals.max() - totals.min(), 1)
            self.assertTrue(all(totals[p] == totals.min() for p in ["K0", "K1", "K2"]), totals.to_dict())

    def test_son_12_ayda_cok_bayram_tutan_bu_bayramda_az_alir(self):
        # Mart 2026: 20-22 Ramazan Bayramı (Cuma, Cumartesi, Pazar)
        names = [f"K{i}" for i in range(8)]
        days = build_gun_detaylari(2026, 3, 31, [20, 21, 22])
        cols = list(days)
        holiday_cols = [c for c in cols if days[c]["holiday"]]
        special_cols = [c for c in cols if days[c]["weekend"] or days[c]["holiday"]]
        unwanted = pd.DataFrame(False, index=names, columns=cols)
        random.seed(4)
        heavy = 0
        for _ in range(5):
            schedule, _ = run_scheduling_core(
                names, cols, unwanted, days, min_bosluk=1, kisi_sayisi=2, simulation_count=10,
                history_holiday={"K0": 2, "K1": 2},
            )
            heavy += int(schedule.loc[["K0", "K1"], holiday_cols].sum().sum())
            specials = schedule[special_cols].sum(axis=1)
            # Hafta sonu/tatil dengesi bozulmaz.
            self.assertLessEqual(specials.max() - specials.min(), 1)
        # 6 bayram nöbeti / 8 kişi: geçmiş olmasa ikisi 5 ayda ~7-8 alırdı.
        # Art arda hafta sonu kuralı bazı takasları engellediğinden sıfır değil.
        self.assertLessEqual(heavy, 3)

    def test_bayram_takasi_toplam_ve_ozel_gun_sayisini_degistirmez(self):
        names = ["Ali", "Ayşe"]
        days = build_gun_detaylari(2026, 3, 31, [20])
        cols = list(days)
        schedule = pd.DataFrame(False, index=names, columns=cols)
        schedule.at["Ali", "20 Cum"] = True
        schedule.at["Ayşe", "28 Cmt"] = True
        unwanted = pd.DataFrame(False, index=names, columns=cols)
        result = _balance_holidays(
            schedule.copy(), names, cols, days, unwanted, None, 1, set(), set(), {},
            {"Ali": 1, "Ayşe": 0},
        )
        self.assertTrue(result.at["Ayşe", "20 Cum"])
        self.assertTrue(result.at["Ali", "28 Cmt"])
        self.assertEqual(result.sum(axis=1).to_dict(), {"Ali": 1, "Ayşe": 1})
        # Sabit nöbet yerinden oynamaz.
        fixed = _balance_holidays(
            schedule.copy(), names, cols, days, unwanted, None, 1, set(), {("Ali", "20 Cum")}, {},
            {"Ali": 1, "Ayşe": 0},
        )
        self.assertTrue(fixed.at["Ali", "20 Cum"])

    def test_bayram_takasi_art_arda_hafta_sonu_olusturmaz(self):
        # Mart 2026: 20 Cuma bayram; 14-15 ve 28-29 hafta sonları.
        names = ["Ali", "Ayşe"]
        days = build_gun_detaylari(2026, 3, 31, [20])
        cols = list(days)
        schedule = pd.DataFrame(False, index=names, columns=cols)
        schedule.at["Ali", "20 Cum"] = True
        schedule.at["Ali", "14 Cmt"] = True
        schedule.at["Ayşe", "21 Cmt"] = True  # Ali'ye geçerse 14 ve 21 art arda olur
        unwanted = pd.DataFrame(False, index=names, columns=cols)
        result = _balance_holidays(
            schedule.copy(), names, cols, days, unwanted, None, 1, set(), set(), {},
            {"Ali": 1, "Ayşe": 0},
        )
        self.assertTrue(result.at["Ali", "20 Cum"])
        self.assertFalse(result.at["Ali", "21 Cmt"])

    def test_hafta_sonu_takasi_farki_kapatir_toplam_ve_cumayi_korur(self):
        # Kasım 2026: Ali iki hafta sonu, Ayşe hiç; ikisinin de birer hafta içi nöbeti var.
        names = ["Ali", "Ayşe"]
        days = build_gun_detaylari(2026, 11, 30, [])
        cols = list(days)
        schedule = pd.DataFrame(False, index=names, columns=cols)
        schedule.at["Ali", "7 Cmt"] = True
        schedule.at["Ali", "21 Cmt"] = True
        schedule.at["Ali", "11 Çar"] = True
        schedule.at["Ayşe", "17 Sal"] = True
        schedule.at["Ayşe", "27 Cum"] = True
        unwanted = pd.DataFrame(False, index=names, columns=cols)
        result = _balance_specials(
            schedule.copy(), names, cols, days, unwanted, None, 1, set(), set(), {}, {},
        )
        specials = [c for c in cols if days[c]["weekend"]]
        self.assertEqual(result[specials].sum(axis=1).to_dict(), {"Ali": 1, "Ayşe": 1})
        self.assertEqual(result.sum(axis=1).to_dict(), {"Ali": 3, "Ayşe": 2})
        # Cuma yerinden oynamaz.
        self.assertTrue(result.at["Ayşe", "27 Cum"])
        # Sabit ve müsait değil günler korunur.
        fixed = _balance_specials(
            schedule.copy(), names, cols, days, unwanted, None, 1, set(),
            {("Ali", "7 Cmt"), ("Ali", "21 Cmt")}, {}, {},
        )
        self.assertEqual(fixed[specials].sum(axis=1).to_dict(), {"Ali": 2, "Ayşe": 0})
        blocked = unwanted.copy()
        blocked.loc["Ayşe", specials] = True
        kept = _balance_specials(
            schedule.copy(), names, cols, days, blocked, None, 1, set(), set(), {}, {},
        )
        self.assertEqual(kept[specials].sum(axis=1).to_dict(), {"Ali": 2, "Ayşe": 0})

    def test_hafta_sonu_dengesi_art_arda_hafta_sonundan_once_gelir(self):
        # Ayşe'ye geçebilecek her hafta sonu 21 Cmt'ye komşu; yine de fark 2 kalmaz.
        names = ["Ali", "Ayşe"]
        days = build_gun_detaylari(2026, 11, 30, [])
        cols = list(days)
        schedule = pd.DataFrame(False, index=names, columns=cols)
        for col in ("14 Cmt", "28 Cmt", "29 Paz"):
            schedule.at["Ali", col] = True
        schedule.at["Ayşe", "21 Cmt"] = True
        schedule.at["Ayşe", "10 Sal"] = True
        unwanted = pd.DataFrame(False, index=names, columns=cols)
        result = _balance_specials(
            schedule.copy(), names, cols, days, unwanted, None, 1, set(), set(), {}, {},
        )
        specials = [c for c in cols if days[c]["weekend"]]
        self.assertEqual(result[specials].sum(axis=1).to_dict(), {"Ali": 2, "Ayşe": 2})
        self.assertTrue(result.at["Ali", "10 Sal"])

    def test_mayis_2027_hafta_sonu_farki_en_fazla_1(self):
        # Aya yayma ve bayram ağırlığı bu ayda hafta sonu farkını 2'de bırakıyordu.
        names = [f"K{i}" for i in range(8)]
        days = build_gun_detaylari(2027, 5, 31, [1, 16, 17, 18, 19])
        cols = list(days)
        special_cols = [c for c in cols if days[c]["weekend"] or days[c]["holiday"]]
        unwanted = pd.DataFrame(False, index=names, columns=cols)
        random.seed(0)
        for _ in range(3):
            schedule, _ = run_scheduling_core(
                names, cols, unwanted, days, min_bosluk=1, kisi_sayisi=2, simulation_count=30,
            )
            specials = schedule[special_cols].sum(axis=1)
            self.assertLessEqual(specials.max() - specials.min(), 1)
            totals = schedule.sum(axis=1)
            self.assertLessEqual(totals.max() - totals.min(), 1)

    def test_gecmis_olmadan_da_bayram_ay_icinde_dengelenir(self):
        names = [f"K{i}" for i in range(8)]
        days = build_gun_detaylari(2026, 3, 31, [20, 21, 22])
        cols = list(days)
        holiday_cols = [c for c in cols if days[c]["holiday"]]
        unwanted = pd.DataFrame(False, index=names, columns=cols)
        random.seed(8)
        for _ in range(5):
            schedule, _ = run_scheduling_core(
                names, cols, unwanted, days, min_bosluk=1, kisi_sayisi=2, simulation_count=10,
            )
            holidays = schedule[holiday_cols].sum(axis=1)
            self.assertLessEqual(holidays.max() - holidays.min(), 1)

    def test_nobetler_aya_yayilir(self):
        # 8 kişi, günde 2: beklenen aralık 4 gün. Dinlenme 1 gün; yayma
        # kapalıyken günaşırı (aralık ≤2) nöbetler çok daha sık çıkar.
        import nobet_algoritma
        names = [f"K{i}" for i in range(8)]
        days = build_gun_detaylari(2026, 11, 30, [])
        cols = list(days)
        unwanted = pd.DataFrame(False, index=names, columns=cols)

        def short_gaps():
            random.seed(2)
            short = 0
            for _ in range(5):
                schedule, _ = run_scheduling_core(
                    names, cols, unwanted, days, min_bosluk=1, kisi_sayisi=2, simulation_count=20,
                )
                for person in names:
                    worked = [days[c]["day_num"] for c in cols if schedule.at[person, c]]
                    short += sum(1 for a, b in zip(worked, worked[1:]) if b - a <= 2)
                totals = schedule.sum(axis=1)
                self.assertLessEqual(totals.max() - totals.min(), 1)
            return short

        spread = short_gaps()
        with patch.object(nobet_algoritma, "SPREAD_WEIGHT", 0), patch.object(nobet_algoritma, "SPREAD_SCORE_WEIGHT", 0), \
                patch.object(nobet_algoritma, "LONG_GAP_WEIGHT", 0), patch.object(nobet_algoritma, "LONG_GAP_SCORE_WEIGHT", 0):
            unspread = short_gaps()
        self.assertLess(spread * 2, unspread)

    def test_gun_asiri_zincir_azalir(self):
        # Ayın ilk haftası çok kişi müsait değil (Kasım 2026 ekran görüntüsü):
        # kalanlar nöbet-boş-nöbet-boş-nöbet zincirine giriyordu. 2 normal,
        # 3. ve sonrası cezalı.
        import nobet_algoritma
        names = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        days = build_gun_detaylari(2026, 11, 30, [])
        cols = list(days)
        unwanted = pd.DataFrame(False, index=names, columns=cols)
        blocks = {"A": range(1, 9), "B": range(5, 9), "D": range(1, 7), "E": range(1, 7),
                  "G": [14, 15, 20], "H": range(1, 16), "I": range(16, 31)}
        for person, ds in blocks.items():
            for d in ds:
                unwanted.at[person, cols[d - 1]] = True
        limits = {"H": {"min": 0, "max": 4}, "I": {"min": 0, "max": 4}}
        day_of = {c: days[c]["day_num"] for c in cols}

        def chains():
            random.seed(3)
            total = 0
            for _ in range(4):
                schedule, _ = run_scheduling_core(
                    names, cols, unwanted, days, min_bosluk=1, kisi_sayisi=2, simulation_count=30,
                    person_limits=limits,
                )
                total += alternate_day_chains(schedule, names, cols, day_of)
            return total

        penalized = chains()
        with patch.object(nobet_algoritma, "ALT_CHAIN_WEIGHT", 0), patch.object(nobet_algoritma, "ALT_CHAIN_SCORE_WEIGHT", 0):
            plain = chains()
        self.assertLess(penalized * 2, plain)

    def test_uzun_bos_kalan_ve_limitli_kisi_aya_yayilir(self):
        # 10 kişi, günde 2, K0 en fazla 3: eskiden limitini ayın başında
        # doldurup (ör. 1, 8, 10) ayın kalanında boş kalıyordu.
        names = [f"K{i}" for i in range(10)]
        days = build_gun_detaylari(2026, 11, 30, [])
        cols = list(days)
        unwanted = pd.DataFrame(False, index=names, columns=cols)
        random.seed(7)
        k0_long = 0
        for _ in range(8):
            schedule, _ = run_scheduling_core(
                names, cols, unwanted, days, min_bosluk=1, kisi_sayisi=2, simulation_count=20,
                person_limits={"K0": {"min": 0, "max": 3}},
            )
            for person in names:
                worked = [days[c]["day_num"] for c in cols if schedule.at[person, c]]
                points = [0] + worked + [31]
                longest = max(b - a for a, b in zip(points, points[1:]))
                if person == "K0":
                    self.assertEqual(len(worked), 3)
                    k0_long += longest > 14
                else:
                    self.assertLessEqual(longest, 10, (person, worked))
        # Eskiden 8 ayın 8'inde K0'ın son nöbetinden sonra 15+ gün boştu.
        self.assertLessEqual(k0_long, 2)

    def test_uygulama_onceki_aylari_tek_sorguda_okur(self):
        cursor = _FakeDbCursor(data_rows=[
            {"name": "Otomatik_2026_09", "year": 2026, "month": 9,
             "team_members": json.dumps(["Ali", "Ayşe"]), "person": "Ali", "day_col": "5 Cmt"},
            {"name": "Otomatik_2026_09", "year": 2026, "month": 9,
             "team_members": json.dumps(["Ali", "Ayşe"]), "person": None, "day_col": None},
            {"name": "Otomatik_2026_08", "year": 2026, "month": 8,
             "team_members": json.dumps(["Ali"]), "person": None, "day_col": None},
        ])
        with patch.object(db.psycopg2, "connect", return_value=_FakeDbConnection(cursor)):
            loaded = db.load_auto_saved_assignments([(2026, 9), (2026, 8), (2026, 7)], owner="ali@gmail.com")
        sql, params = cursor.executed[0]
        self.assertIn("s.owner = %s", sql)
        self.assertEqual(params[0], "ali@gmail.com")
        self.assertEqual(set(params[1]), {"Otomatik_2026_09", "Otomatik_2026_08", "Otomatik_2026_07"})
        self.assertEqual(set(loaded), {(2026, 9), (2026, 8)})
        self.assertEqual(list(loaded[(2026, 9)].index), ["Ali", "Ayşe"])
        self.assertTrue(loaded[(2026, 9)].at["Ali", "5 Cmt"])
        self.assertFalse(loaded[(2026, 9)].at["Ayşe", "5 Cmt"])
        self.assertEqual(schedule_totals(loaded[(2026, 8)]), {"Ali": 0})
        self.assertEqual(db.load_auto_saved_assignments([]), {})

    def test_onceki_aylarin_tercihleri_ayri_sorguyla_okunur(self):
        cursor = _FakeDbCursor(data_rows=[
            {"name": "Otomatik_2026_09", "year": 2026, "month": 9,
             "team_members": json.dumps(["Ali"]), "person": "Ali", "day_col": "5 Cmt",
             "pref_json": '{"5 Cmt": {"Ali": 3}}'},
        ])
        with patch.object(db.psycopg2, "connect", return_value=_FakeDbConnection(cursor)):
            loaded, prefs = db.load_auto_saved_assignments(
                [(2026, 9)], owner="ali@gmail.com", include_preferences=True,
            )
        self.assertEqual(len(cursor.executed), 2)
        self.assertNotIn("pref_json", cursor.executed[0][0])
        self.assertIn("pref_json", cursor.executed[1][0])
        self.assertEqual(set(loaded), {(2026, 9)})
        self.assertEqual(unavailable_day_counts(prefs[(2026, 9)]), {"Ali": 1})
        self.assertEqual(db.load_auto_saved_assignments([], include_preferences=True), ({}, {}))

    def test_analizde_bayram_sutunu(self):
        days = build_gun_detaylari(2026, 10, 31, [29])
        cols = list(days)
        schedule = pd.DataFrame(False, index=["Ali", "Ayşe"], columns=cols)
        schedule.at["Ali", "29 Per"] = True
        load, _, _ = build_schedule_analysis(schedule, [], [], ["Ali", "Ayşe"], days, {}, 0, 0)
        self.assertEqual(int(load.at["Ali", "Tatil"]), 1)
        self.assertEqual(int(load.at["Ayşe", "Tatil"]), 0)


class TestBackup(unittest.TestCase):
    """Tüm verinin yedeği: indirme dosyası, doğrulama ve geri yükleme."""

    def _rows(self):
        return [{
            "name": "Otomatik_2026_09", "year": 2026, "month": 9,
            "team_members": json.dumps(["Ali", "Ayşe"]),
            "pref_json": json.dumps({"1 Sal": {"Ali": 1, "Ayşe": 0}}),
            "settings_json": json.dumps({"locked": [["Ali", "1 Sal"]], "hidden": {}}),
            "rows_liste_json": json.dumps([{"Tarih": "1 Eyl Sal", "Nöbetçi 1": "Ali"}]),
            "cells": [("Ali", "1 Sal", True), ("Ayşe", "1 Sal", False),
                      ("Ali", "2 Çar", False), ("Ayşe", "2 Çar", True)],
        }]

    def test_yedekte_uygulamanin_kullanmadigi_kayitlar_atlanir(self):
        rows = self._rows()
        rows.append(dict(rows[0], name="Elle Kayıt"))
        rows.append(dict(rows[0], name="Otomatik_2026_10"))  # adı ayla uyuşmuyor
        backup, errors = parse_backup(backup_to_bytes(build_backup(None, rows)))
        self.assertEqual(errors, [])
        self.assertEqual([item["name"] for item in backup["schedules"]], ["Otomatik_2026_09"])
        self.assertEqual(backup["skipped"], 2)

    def test_excel_yukleme_sinirlari(self):
        import nobet_core
        self.assertEqual(nobet_core.EXCEL_IMPORT_MAX_BYTES, 2 * 1024 * 1024)
        self.assertGreaterEqual(nobet_core.EXCEL_IMPORT_MAX_ROWS, 32)  # başlık + 31 gün

    def test_yedek_dosyasi_geri_okunur(self):
        raw = backup_to_bytes(build_backup({"team": "Ali, Ayşe"}, self._rows()))
        backup, errors = parse_backup(raw)
        self.assertEqual(errors, [])
        item = backup["schedules"][0]
        self.assertEqual(item["team"], ["Ali", "Ayşe"])
        self.assertEqual(item["days"], ["1 Sal", "2 Çar"])
        self.assertEqual(set(item["assigned"]), {("Ali", "1 Sal"), ("Ayşe", "2 Çar")})
        self.assertEqual(item["preferences"], {"1 Sal": {"Ali": 1, "Ayşe": 0}})
        self.assertEqual(backup["settings"], {"team": "Ali, Ayşe"})
        self.assertEqual(backup_summary(backup), "1 ay (Eylül 2026), ekip ve ayarlar")

    def test_ekip_disindaki_hucreler_yedege_girmez(self):
        rows = self._rows()
        rows[0]["cells"].append(("Eski Kişi", "1 Sal", True))
        backup, errors = parse_backup(backup_to_bytes(build_backup(None, rows)))
        self.assertEqual(errors, [])
        self.assertNotIn(("Eski Kişi", "1 Sal"), backup["schedules"][0]["assigned"])

    def test_bozuk_yedek_reddedilir(self):
        good = build_backup(None, self._rows())
        self.assertTrue(parse_backup(b"\xff\xfe")[1])
        self.assertTrue(parse_backup(b'{"format": "baska"}')[1])
        self.assertTrue(parse_backup(b"x" * (20 * 1024 * 1024 + 1))[1])
        for mutate in (
            lambda d: d["schedules"][0].update(year=1990),
            lambda d: d["schedules"][0].update(month=13),
            lambda d: d["schedules"][0].update(name=""),
            lambda d: d["schedules"][0].update(team="Ali"),
            lambda d: d["schedules"][0].update(assigned=[["Yabancı", "1 Sal"]]),
            lambda d: d["schedules"][0].update(days=[str(i) for i in range(40)]),
            lambda d: d["schedules"][0].update(rows="x"),
            lambda d: d["schedules"].append(dict(d["schedules"][0])),
            lambda d: d.update(settings=[1]),
            lambda d: d.update(version=99),
            # Küçük dosyadan milyonlarca satır üretilmesin.
            lambda d: d.update(schedules=[
                dict(d["schedules"][0], name=f"Otomatik_{2030 + i // 12}_{i % 12 + 1:02d}",
                     year=2030 + i // 12, month=i % 12 + 1, team=[f"K{j}" for j in range(300)],
                     days=[str(k) for k in range(31)], assigned=[])
                for i in range(60)
            ]),
            lambda d: d["schedules"][0].update(preferences={"1 Sal": {"Ali": 7}}),
            lambda d: d["schedules"][0].update(preferences={"1 Sal": {"Ali": "<b>"}}),
            lambda d: d["schedules"][0].update(rows=[{"Tarih": ["iç içe"]}]),
            lambda d: d["schedules"][0].update(rows=[{"Tarih": "x"}] * 40),
        ):
            data = json.loads(json.dumps(good))
            mutate(data)
            backup, errors = parse_backup(json.dumps(data).encode())
            self.assertIsNone(backup)
            self.assertTrue(errors)

    def test_yedek_hatirlatmasi(self):
        from datetime import date as _date
        today = _date(2026, 10, 14)
        self.assertEqual(backup_reminder(None, today), "Henüz yedek indirilmedi.")
        self.assertIsNone(backup_reminder("2026-10-01 10:00", today))
        self.assertEqual(backup_reminder("2026-09-01 10:00", today), "Son yedek 43 gün önce indirildi.")

    def test_disa_aktarma_kullaniciya_gore(self):
        cursor = _FakeDbCursor(data_rows=[])
        with patch.object(db.psycopg2, "connect", return_value=_FakeDbConnection(cursor)):
            settings, schedules = db.export_owner_data("ali@gmail.com")
        self.assertEqual(cursor.executed[0][1], ("user:ali@gmail.com",))
        self.assertEqual(cursor.executed[1][1], ("ali@gmail.com",))
        self.assertIn("s.owner = %s", cursor.executed[2][0])
        self.assertEqual(cursor.executed[2][1], ("ali@gmail.com",))
        self.assertEqual(schedules, [])

    def test_geri_yukleme_kullanicinin_verisine_tek_islemde_yazar(self):
        backup, _ = parse_backup(backup_to_bytes(build_backup({"team": "Ali"}, self._rows())))
        cursor = _FakeDbCursor()
        inserted = []
        with patch.object(db.psycopg2, "connect", return_value=_FakeDbConnection(cursor)), \
                patch.object(db, "execute_values", side_effect=lambda _c, _sql, rows, **_k: inserted.extend(rows)):
            self.assertTrue(db.restore_owner_data(backup, owner="ali@gmail.com"))
        upsert_sql, upsert_params = cursor.executed[0]
        self.assertIn("ON CONFLICT (owner, name, year, month)", upsert_sql)
        self.assertEqual(upsert_params[7], "ali@gmail.com")
        self.assertEqual(len(inserted), 4)
        self.assertEqual(sum(1 for row in inserted if row[3]), 2)
        self.assertEqual(cursor.executed[-1][1][0], "user:ali@gmail.com")

    def test_geri_yukleme_hatasi_yukari_iletilir(self):
        with patch.object(db.psycopg2, "connect", side_effect=RuntimeError("yok")):
            with self.assertRaises(RuntimeError):
                db.restore_owner_data({"schedules": [], "settings": None})
            with self.assertRaises(RuntimeError):
                db.export_owner_data()

    def test_uygulama_yedegi_kullaniciya_baglar(self):
        source = _app_source()
        self.assertIn("export_owner_data(owner=current_owner)", source)
        # Veritabanı hatası kullanıcıya ham mesajla gösterilmez.
        self.assertIn("Yedek hazırlanamadı", source)
        self.assertIn("restore_owner_data(parsed, owner=current_owner)", source)
        # Geri yüklemeden sonra eski ekran durumu yedeğin üzerine yazılmasın.
        self.assertIn("st.session_state.clear()", source)

    def test_uygulamadaki_yedek_satiri(self):
        from nobet_core import snapshot_label
        item = {"year": 2026, "month": 3, "created_tr": "2026-03-10 14:20", "reason": "Yedekle"}
        self.assertEqual(snapshot_label(item), "Mart 2026 · 10.03.2026 14:20 yedeklendi")
        self.assertIn("bozuk", snapshot_label({"year": None, "month": None, "created_tr": "bozuk"}))

    def test_ay_yedegi_yalnizca_o_ayi_icerir(self):
        from nobet_core import month_backup
        rows = self._rows() + [dict(self._rows()[0], name="Otomatik_2026_11", month=11)]
        backup = build_backup({"team": "Ali"}, rows)
        only = month_backup(backup, 2026, 11)
        self.assertEqual([item["name"] for item in only["schedules"]], ["Otomatik_2026_11"])
        # Ay yedeği ekip/ayarları değiştirmez.
        self.assertIsNone(only["settings"])
        parsed, errors = parse_backup(backup_to_bytes(only))
        self.assertEqual(errors, [])
        self.assertEqual(len(parsed["schedules"]), 1)
        self.assertEqual(month_backup(backup, 2025, 1)["schedules"], [])

    def test_uygulama_yedegi_sahibe_ve_aya_gore_tek(self):
        cursor = _FakeDbCursor()
        with patch.object(db.psycopg2, "connect", return_value=_FakeDbConnection(cursor)):
            db.save_backup_snapshot("ali@gmail.com", 2026, 3, "2026-03-10 14:20", "Yedekle", "1 ay", "{}")
            db.load_backup_snapshot(5, owner="ali@gmail.com")
        delete_sql, delete_params = cursor.executed[0]
        self.assertIn("DELETE FROM backup_snapshots", delete_sql)
        self.assertEqual(delete_params, ("ali@gmail.com", "Yedekle", 2026, 3))
        insert_sql, insert_params = cursor.executed[1]
        self.assertIn("INSERT INTO backup_snapshots", insert_sql)
        self.assertEqual(insert_params[:3], ("ali@gmail.com", 2026, 3))
        load_sql, load_params = cursor.executed[2]
        self.assertIn("owner = %s", load_sql)
        self.assertEqual(load_params, (5, "ali@gmail.com"))

    def test_yedekle_tusu_kayittan_sonra_calisir(self):
        source = _app_source("nobet.py")
        # Yedek, son değişiklik otomatik kaydedildikten sonra alınır.
        self.assertLess(
            source.index('st.error(st.session_state.persistence_error)'),
            source.index("\nrun_requested_snapshot(current_owner, yil, ay)"),
        )
        # Geri yüklemeden önce mevcut hâlin kopyası alınır; alınamazsa geri yüklenmez.
        backup = _app_source("nobet_yedek.py")
        restore = backup[backup.index("def restore_backup"):]
        self.assertLess(
            restore.index("take_snapshot(SNAPSHOT_BEFORE_RESTORE, current_owner)"),
            restore.index("restore_owner_data(parsed"),
        )


class TestShareAndCalendar(unittest.TestCase):
    ROWS = [
        {"Tarih": "03.10.2026 Cmt", "AYB": "Ali", "GYB": "Ayşe"},
        {"Tarih": "05.10.2026 Pzt", "AYB": "Can", "GYB": "Ali"},
        {"Tarih": "06.10.2026 Sal", "AYB": "Can", "GYB": "-"},
    ]

    def test_whatsapp_metni(self):
        text = build_share_text(self.ROWS, ["AYB", "GYB"], 2026, 10, special_days=[3])
        lines = text.split("\n")
        self.assertEqual(lines[0], "📅 Ekim 2026 Nöbet Listesi")
        self.assertIn("*3 Ekim Cmt* — AYB: Ali · GYB: Ayşe", lines)
        self.assertIn("5 Ekim Pzt — AYB: Can · GYB: Ali", lines)
        self.assertIn("6 Ekim Sal — AYB: Can", lines)

    def _unfold(self, ics):
        return ics.replace("\r\n ", "").split("\r\n")

    def test_takvim_dosyasi_birim_ve_partneri_icerir(self):
        ics = build_person_ics("Ali", self.ROWS, ["AYB", "GYB"], 2026, 10)
        self.assertTrue(ics.startswith("BEGIN:VCALENDAR\r\n"))
        self.assertTrue(ics.endswith("END:VCALENDAR\r\n"))
        lines = self._unfold(ics)
        self.assertEqual(lines.count("BEGIN:VEVENT"), 2)
        self.assertIn("DTSTART;VALUE=DATE:20261003", lines)
        self.assertIn("DTEND;VALUE=DATE:20261004", lines)
        self.assertIn("SUMMARY:Nöbet — AYB", lines)
        self.assertIn("SUMMARY:Nöbet — GYB", lines)
        self.assertIn("DESCRIPTION:Görev yeri: AYB\\nBirlikte: Ayşe (GYB)", lines)
        uids = [l for l in lines if l.startswith("UID:")]
        self.assertEqual(len(set(uids)), 2)

    def test_takvim_satirlari_75_bayti_asmaz_ve_ozel_karakterler_kacirilir(self):
        rows = [{"Tarih": "01.10.2026 Per", "Acil Servis; Sarı Alan, Gece Nöbeti Uzun Birim Adı": "Ali",
                 "GYB": "Çağlayan Öztürkmenoğlu"}]
        ics = build_person_ics("Ali", rows, list(rows[0])[1:], 2026, 10)
        for raw_line in ics.split("\r\n"):
            self.assertLessEqual(len(raw_line.encode("utf-8")), 75)
        unfolded = "".join(self._unfold(ics))
        self.assertIn("Acil Servis\\; Sarı Alan\\, Gece", unfolded)


class TestGridAppearance(unittest.TestCase):
    def setUp(self):
        self.source = _app_source()

    def test_atandi_gri_zemin_ustunde_tik(self):
        state = get_preference_grid_state(True, 0)
        self.assertEqual(state["symbol"], "✓")
        self.assertEqual(state["color"], "#475569")
        self.assertNotEqual(state["symbol"], get_preference_grid_state(False, 0)["symbol"])

    def test_karanlik_modda_isimler_okunur(self):
        self.assertIn('st.context.theme.type == "dark"', self.source)
        self.assertIn("color:{ui['name_text']};", self.source)
        self.assertIn("background: {ui['name_bg']} !important;", self.source)
        self.assertNotIn("background: #f8fafc !important;", self.source)

    def test_isim_sutunu_en_uzun_isme_gore(self):
        from nobet_core import name_column_width
        self.assertEqual(name_column_width(["Ali", "Can"]), 96)          # kısa isimler: en az 96
        self.assertGreater(name_column_width(["AYBÜKE", "Ali"]), 96)       # rozet sığsın
        self.assertEqual(name_column_width(["Çok Uzun Bir İsim Soyisim Örneği"]), 220)  # üst sınır
        self.assertEqual(name_column_width([]), 96)

    def test_sayi_rozeti_alt_satira_kaymaz(self):
        self.assertIn("flex: 0 0 {name_col_px}px !important;", self.source)
        self.assertIn("display:flex;flex-wrap:nowrap;", self.source)
        self.assertIn("text-overflow:ellipsis;", self.source)
        self.assertIn("<span class='nm-badge' style='flex-shrink:0;background:{badge_bg}", self.source)
        # Telefonda rozet bilerek adın altına iner (kısa ad + "6 · 2 HS").
        self.assertIn(".grid-name .nm-short {{ display: inline !important;", self.source)

    def test_gunler_alt_satira_kaymaz(self):
        self.assertIn(
            '.st-key-preference-grid [data-testid="stHorizontalBlock"] {{\n    flex-wrap: nowrap !important;',
            self.source,
        )


class TestUiCilasiYardimcilari(unittest.TestCase):
    def test_kisa_adlar_cakismaz(self):
        from nobet_core import short_display_names
        short = short_display_names(["Ayşe Yılmaz", "Ayşe Yıldız", "Mehmet Kaya", "Can", "Ali Rıza Demir"])
        self.assertEqual(short["Mehmet Kaya"], "Mehmet K.")
        self.assertEqual(short["Can"], "Can")
        self.assertEqual(short["Ali Rıza Demir"], "Ali Rıza D.")
        self.assertEqual({short["Ayşe Yılmaz"], short["Ayşe Yıldız"]}, {"Ayşe Yılm.", "Ayşe Yıld."})
        same = short_display_names(["Ece Ak", "Ece Ak "])
        self.assertEqual(len(set(same.values())), 2)

    def test_siradaki_nobet(self):
        from datetime import date
        from nobet_core import next_shift, next_shift_text
        nov = [{"Tarih": "02.11.2026 Pzt", "AYB": "Can", "GYB": "Elif"},
               {"Tarih": "04.11.2026 Çar", "AYB": "Elif", "GYB": "Can"}]
        dec = [{"Tarih": "01.12.2026 Sal", "AYB": "Can", "GYB": "Ali"}]
        today = date(2026, 11, 3)
        shift = next_shift([((2026, 11), nov, ["AYB", "GYB"]), ((2026, 12), dec, ["AYB", "GYB"])], "Can", today)
        self.assertEqual(shift, {"date": date(2026, 11, 4), "roles": ["GYB"], "partners": ["Elif"]})
        self.assertEqual(next_shift_text(shift, today), "4 Kasım Çarşamba · yarın · GYB · Elif ile")
        later = next_shift([((2026, 11), nov, ["AYB", "GYB"]), ((2026, 12), dec, ["AYB", "GYB"])],
                           "Can", date(2026, 11, 20))
        self.assertEqual(later["date"], date(2026, 12, 1))
        self.assertIn("11 gün sonra", next_shift_text(later, date(2026, 11, 20)))
        self.assertIsNone(next_shift([((2026, 11), nov, ["AYB", "GYB"])], "Can", date(2026, 11, 5)))
        plain = [{"Tarih": "05.11.2026 Per", "Nöbetçiler": "Can, Elif"}]
        self.assertEqual(next_shift([((2026, 11), plain, ["Nöbetçiler"])], "Can", today)["partners"], ["Elif"])
        self.assertEqual(next_shift_text(next_shift([((2026, 11), nov, ["AYB", "GYB"])], "Can", date(2026, 11, 4)),
                                         date(2026, 11, 4)).split(" · ")[1], "bugün")

    def test_paylasim_tarih_ve_gun_turu(self):
        from nobet_core import shared_date_label, shared_day_kinds
        rows = [{"Tarih": "04.11.2026 Çar"}, {"Tarih": "07.11.2026 Cmt"}, {"Tarih": "29.10.2026 Per"}, {"Tarih": "?"}]
        self.assertEqual(shared_date_label(rows[0], 2026, 11), "4 Kas Çar")
        self.assertEqual(shared_date_label(rows[3], 2026, 11), "?")
        self.assertEqual(shared_day_kinds(rows[:2], 2026, 11), ["", "weekend"])
        self.assertEqual(shared_day_kinds([rows[2]], 2026, 10), ["holiday"])

    def test_en_cok_birlikte_ve_simulasyon_ozeti(self):
        from nobet_core import most_paired, simulation_summary
        gd = build_gun_detaylari(2026, 11, 30, [])
        cols = list(gd)
        people = ["A", "B", "C", "D"]
        sched = pd.DataFrame(False, index=people, columns=cols)
        for i, c in enumerate(cols):
            pair = [("A", "B"), ("C", "D"), ("A", "C")][i % 3]
            for p in pair:
                sched.at[p, c] = True
        self.assertEqual(most_paired(sched, people, cols, limit=2), [("A", "B", 10), ("A", "C", 10)])
        text = simulation_summary(sched, people, cols, gd)
        self.assertTrue(text.startswith("Bu ay: hafta sonu farkı "))
        self.assertIn("toplam farkı 10", text)
        self.assertIn("en çok birlikte 10 nöbet", text)


class TestOfficialHolidaysAndReminder(unittest.TestCase):
    def test_resmi_tatiller(self):
        from nobet_core import official_holidays, official_holidays_summary, holidays_to_text, religious_holidays_known
        self.assertEqual(official_holidays(2026, 10), [(29, "Cumhuriyet Bayramı")])
        self.assertEqual([d for d, _ in official_holidays(2026, 3)], [20, 21, 22])
        self.assertEqual([d for d, _ in official_holidays(2026, 5)], [1, 19, 27, 28, 29, 30])
        # Ay sınırını aşan bayram iki aya bölünür (2025 Ramazan: 30 Mart - 1 Nisan)
        self.assertEqual([d for d, _ in official_holidays(2025, 3)], [30, 31])
        self.assertEqual([d for d, _ in official_holidays(2025, 4)], [1, 23])
        self.assertEqual(holidays_to_text([29, 20, 21, 22, 1]), "1, 20-22, 29")
        self.assertEqual(official_holidays_summary(2026, 5)[-1], "27-30 Mayıs Kurban Bayramı")
        self.assertTrue(religious_holidays_known(2027))
        self.assertEqual([d for d, _ in official_holidays(2028, 2)], [26, 27, 28])
        self.assertEqual([d for d, _ in official_holidays(2028, 5)], [1, 5, 6, 7, 8, 19])
        self.assertTrue(religious_holidays_known(2029))
        self.assertFalse(religious_holidays_known(2030))
        self.assertEqual(official_holidays(2030, 10), [(29, "Cumhuriyet Bayramı")])

    def test_takvimde_bir_gun_once_hatirlatici(self):
        rows = [{"Tarih": "03.10.2026 Cmt", "AYB": "Ali", "GYB": "Ayşe"}]
        ics = build_person_ics("Ali", rows, ["AYB", "GYB"], 2026, 10)
        lines = ics.replace("\r\n ", "").split("\r\n")
        self.assertIn("BEGIN:VALARM", lines)
        self.assertIn("TRIGGER:-PT6H", lines)
        self.assertIn("DESCRIPTION:Yarın nöbetin var — AYB", lines)
        self.assertLess(lines.index("BEGIN:VALARM"), lines.index("END:VEVENT"))
        for raw in ics.split("\r\n"):
            self.assertLessEqual(len(raw.encode("utf-8")), 75)

    def test_uygulama_resmi_tatilleri_varsayilan_yapar(self):
        source = _app_source()
        self.assertIn("value=holiday_text_for_period(_holidays_by_period, yil, ay),", source)
        self.assertIn("_holidays_by_period[_holiday_period] = holiday_input", source)
        self.assertIn("_holidays_by_period.pop(_holiday_period, None)", source)


class TestFairDistribution(unittest.TestCase):
    def test_toplam_farki_en_fazla_bir_ve_kurallar_bozulmaz(self):
        names = ["GÖRKEM", "FARUK", "EYLÜL", "EGE", "AYÇA", "ORCAN", "LEYLA", "AYBÜKE", "SENA"]
        days = build_gun_detaylari(2026, 10, 31, [])
        cols = list(days)
        unwanted = pd.DataFrame(False, index=names, columns=cols)
        unwanted.loc["EGE", cols[0:6]] = True
        pref = pd.DataFrame(0, index=names, columns=cols)
        for d in (2, 6, 10, 14, 18, 22, 26, 30):
            pref.loc["FARUK", cols[d - 1]] = 1
        for seed in range(3):
            random.seed(seed)
            schedule, _ = run_scheduling_core(names, cols, unwanted, days, 1, 2,
                                              df_preferred=pref, simulation_count=40)
            totals = schedule.sum(axis=1)
            self.assertLessEqual(int(totals.max() - totals.min()), 1, totals.to_dict())
            self.assertEqual(missing_shift_coverage(schedule, cols, 2), [])
            for person in names:
                worked = [days[c]["day_num"] for c in cols if schedule.at[person, c]]
                self.assertTrue(all(b - a > 1 for a, b in zip(worked, worked[1:])), person)
                self.assertFalse(any(schedule.at[person, c] and unwanted.at[person, c] for c in cols))

    def test_dengeleme_sabit_ve_limitlere_dokunmaz(self):
        names = ["Ali", "Ayşe", "Can", "Deniz"]
        days = build_gun_detaylari(2026, 11, 30, [])
        cols = list(days)
        unwanted = pd.DataFrame(False, index=names, columns=cols)
        fixed = {("Ali", cols[c]) for c in (0, 2, 4, 6, 8, 10, 12, 14, 16, 18)}
        random.seed(1)
        schedule, _ = run_scheduling_core(names, cols, unwanted, days, 1, 2, simulation_count=20,
                                          fixed_assignments=fixed,
                                          person_limits={"Can": {"min": 0, "max": 12}})
        self.assertTrue(all(schedule.at[p, c] for p, c in fixed))
        self.assertLessEqual(int(schedule.loc["Can"].sum()), 12)

    def test_isimler_ve_dugmeler_ayni_hizada(self):
        source = _app_source()
        self.assertIn(':is([data-testid="stElementContainer"], .element-container):has(.preference-state) {{\n    display: none !important;', source)
        self.assertIn('[data-testid="stHorizontalBlock"] {{\n    align-items: center !important;', source)

    def test_gun_basligi_bolunmez(self):
        source = _app_source()
        self.assertIn("f\"<div style='{num_style}white-space:nowrap;'>{day_num}</div>\"", source)
        self.assertNotIn('badge = "★"', source)


class TestFridayBalance(unittest.TestCase):
    def _fridays(self, schedule, days, cols):
        return {
            p: sum(1 for c in cols if schedule.at[p, c] and days[c]["weekday"] == 4 and not days[c]["holiday"])
            for p in schedule.index
        }

    def test_herkes_benzer_sayida_cuma_alir_hafta_sonu_dengesi_bozulmaz(self):
        # Kasım 2026: 4 Cuma × 2 kişi = 8 Cuma nöbeti, 8 kişi → herkes 1.
        names = [f"K{i}" for i in range(8)]
        days = build_gun_detaylari(2026, 11, 30, [])
        cols = list(days)
        unwanted = pd.DataFrame(False, index=names, columns=cols)
        special = [c for c in cols if days[c]["weekend"] or days[c]["holiday"]]
        for seed in range(5):
            random.seed(seed)
            schedule, _ = run_scheduling_core(names, cols, unwanted, days, 1, 2, simulation_count=30)
            fridays = self._fridays(schedule, days, cols)
            self.assertLessEqual(max(fridays.values()) - min(fridays.values()), 1, fridays)
            weekend = schedule[special].sum(axis=1)
            self.assertLessEqual(int(weekend.max() - weekend.min()), 1, weekend.to_dict())
            totals = schedule.sum(axis=1)
            self.assertLessEqual(int(totals.max() - totals.min()), 1, totals.to_dict())
            self.assertEqual(missing_shift_coverage(schedule, cols, 2), [])
            for person in names:
                worked = [days[c]["day_num"] for c in cols if schedule.at[person, c]]
                self.assertTrue(all(b - a > 1 for a, b in zip(worked, worked[1:])), person)

    def test_tatile_denk_gelen_cuma_cuma_sayilmaz(self):
        days = build_gun_detaylari(2026, 10, 31, [2])  # 2 Ekim 2026 Cuma
        self.assertTrue(days["2 Cum"]["holiday"])
        schedule = pd.DataFrame(False, index=["Ali"], columns=list(days))
        schedule.at["Ali", "2 Cum"] = True
        schedule.at["Ali", "9 Cum"] = True
        load, _, _ = build_schedule_analysis(schedule, [], [], ["Ali"], days, {}, 0, 1.0)
        self.assertEqual(int(load.at["Ali", "Cuma"]), 1)
        self.assertEqual(int(load.at["Ali", "Hafta sonu"]), 1)

    def test_cuma_takasi_sabit_musaitlik_ve_dinlenmeye_uyar(self):
        days = build_gun_detaylari(2026, 11, 30, [])
        cols = list(days)
        schedule = pd.DataFrame(False, index=["Ali", "Can"], columns=cols)
        # Ali üç Cuma (6, 13, 20), Can hiç; Can'ın hafta içi nöbetleri 3, 10, 17.
        for c in ("6 Cum", "13 Cum", "20 Cum"):
            schedule.at["Ali", c] = True
        for c in ("3 Sal", "10 Sal", "17 Sal"):
            schedule.at["Can", c] = True
        unwanted = pd.DataFrame(False, index=["Ali", "Can"], columns=cols)
        unwanted.at["Can", "13 Cum"] = True
        result = _balance_fridays(schedule.copy(), ["Ali", "Can"], cols, days, unwanted, None, 1,
                                  set(), {("Ali", "6 Cum")}, {})
        fridays = self._fridays(result, days, cols)
        self.assertEqual(fridays, {"Ali": 2, "Can": 1})
        self.assertTrue(result.at["Ali", "6 Cum"])  # sabit nöbet yerinde
        self.assertFalse(result.at["Can", "13 Cum"])  # müsait olmadığı Cuma verilmedi
        self.assertEqual(result.sum(axis=1).to_dict(), schedule.sum(axis=1).to_dict())


class TestMissingDayExplanation(unittest.TestCase):
    def test_her_kisinin_nedeni_gruplanir(self):
        days = build_gun_detaylari(2026, 11, 30, [])
        cols = list(days)
        names = ["Ali", "Ayşe", "Can", "Deniz", "Ece", "Fuat"]
        schedule = pd.DataFrame(False, index=names, columns=cols)
        unwanted = pd.DataFrame(False, index=names, columns=cols)
        col = "14 Cmt"
        schedule.at["Fuat", col] = True              # o gün nöbetçi (1/2)
        unwanted.at["Ali", col] = True               # müsait değil
        schedule.at["Ayşe", "13 Cum"] = True         # dinlenmede
        for c in ("2 Pzt", "5 Per"):
            schedule.at["Can", c] = True             # limitte (max 2)
        reasons = dict(explain_missing_day(
            schedule, col, days, unwanted, 1,
            person_limits={"Can": {"min": 0, "max": 2}},
            forbidden_pairs={("Deniz", "Fuat")},
        ))
        self.assertEqual(reasons["izinli / müsait değil"], ["Ali"])
        self.assertEqual(reasons["dinlenmede"], ["Ayşe (13. gün nöbetli)"])
        self.assertEqual(reasons["nöbet limitinde"], ["Can (en fazla 2)"])
        self.assertEqual(reasons["birlikte çalışamaz"], ["Deniz (Fuat ile)"])
        self.assertEqual(reasons["uygun, elle atanabilir"], ["Ece"])
        text = format_missing_day_reason(list(reasons.items()))
        self.assertIn("izinli / müsait değil: Ali", text)

    def test_gecen_ay_sonu_nobeti_dinlenme_sayilir(self):
        days = build_gun_detaylari(2026, 11, 30, [])
        cols = list(days)
        schedule = pd.DataFrame(False, index=["Ali"], columns=cols)
        unwanted = pd.DataFrame(False, index=["Ali"], columns=cols)
        reasons = explain_missing_day(schedule, "1 Paz", days, unwanted, 1, initial_last_shift={"Ali": 0})
        self.assertEqual(reasons, [("dinlenmede", ["Ali (geçen ayın sonunda nöbetli)"])])

    def test_kimse_kalmadiysa_acik_mesaj(self):
        self.assertIn("kimse yok", format_missing_day_reason([]))


class TestUiCleanup(unittest.TestCase):
    def setUp(self):
        self.source = _app_source()

    def test_bos_hucreler_acik_nobetliler_dolu(self):
        free = get_preference_grid_state(False, 0)
        assigned = get_preference_grid_state(True, 0)
        self.assertTrue(free.get("soft"))
        self.assertFalse(assigned.get("soft", False))
        self.assertTrue(free["css_class"].startswith("preference-free-"))
        self.assertTrue(assigned["css_class"].startswith("preference-state-"))
        for n in range(4):
            self.assertIn(f":has(.preference-free-{n}) .stButton > button", self.source)

    def test_lejant_iki_satir(self):
        self.assertIn("NÖBETLİ", self.source)
        self.assertIn("NÖBETSİZ", self.source)
        self.assertNotIn("legend_container.columns(", self.source)

    def test_disa_aktar_menusu_ve_sekmeler(self):
        self.assertIn('st.popover("📤 Dışa aktar"', self.source)
        self.assertNotIn("dl1, dl2, dl3, dl4, dl5, dl6", self.source)
        self.assertIn('st.subheader("📊 Analiz")', self.source)
        self.assertNotIn('"📋 Günlük Liste"', self.source)
        self.assertNotIn("Görev Yerlerini Düzenle", self.source)
        self.assertNotIn('st.header("📅 Nöbet Çizelgesi")', self.source)


class TestStaleModuleRecovery(unittest.TestCase):
    def test_eski_modul_kalirsa_yenilenip_tekrar_denenir(self):
        source = _app_source("nobet.py")
        recovery = source[source.index("for _attempt in range(2):"):source.index("from streamlit_local_storage")]
        self.assertRegex(
            recovery,
            r"except ImportError:\n        if _attempt:\n(            print\(.*\n)?            raise\n        _fresh_modules\(force=True\)",
        )
        # İçe aktarma listesi tek yerde (eskiden iki kopya elle eş tutuluyordu).
        self.assertEqual(recovery.count("from db import ("), 1)
        self.assertEqual(recovery.count("from nobet_core import ("), 1)
        # Arayüz bölümleri de her çalışmada yeniden okunur.
        self.assertIn("_load_ui_modules()", recovery)

    def test_degisen_cekirdek_moduller_her_calismada_tazelenir(self):
        source = _app_source("nobet.py")
        listed = re.search(r"_FRESH_MODULES = \((.*?)\)", source, flags=re.S).group(1)
        # nobet_iyilestir eskiyken İyileştir "ortools yüklü değil" diyordu.
        self.assertEqual(re.findall(r'"(\w+)"', listed), ["db", "nobet_kalicilik", "nobet_gunler", "nobet_ucret", "nobet_algoritma", "nobet_core", "nobet_iyilestir"])
        self.assertLess(source.index("\n_fresh_modules()\n"), source.index("for _attempt in range(2):"))
        # Tazeleme kilit altında nobet_tazele.py'de; davranışı test_tazele.py dener.
        self.assertIn("fresh_modules(_FRESH_MODULES, force=force)", source)
        for name in ("db.py", "nobet_kalicilik.py", "nobet_gunler.py", "nobet_ucret.py", "nobet_algoritma.py", "nobet_core.py",
                     "nobet_iyilestir.py"):
            self.assertTrue(_app_source(name).rstrip().endswith(
                "_nobet_loaded_stamp = os.stat(__file__).st_mtime_ns"), name)

    def test_gunler_ve_ucret_cekirdege_bagimli_degil(self):
        # Döngüsel içe aktarma olmasın: nobet_core bu dosyalardan okur, tersi olmaz.
        import ast
        def imported(name):
            tree = ast.parse(_app_source(name))
            return {
                (node.module if isinstance(node, ast.ImportFrom) else alias.name).split(".")[0]
                for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in (getattr(node, "names", None) or [None])
            }
        self.assertNotIn("nobet_core", imported("nobet_gunler.py"))
        self.assertNotIn("nobet_core", imported("nobet_ucret.py"))
        self.assertFalse({n for n in imported("nobet_gunler.py") if n.startswith(("nobet", "db"))})
        # Algoritma yalnız nobet_gunler'den okur; arayüz durumuna (session_state) dokunmaz.
        self.assertEqual({n for n in imported("nobet_algoritma.py") if n.startswith(("nobet", "db", "streamlit"))},
                         {"nobet_gunler"})
        self.assertNotIn("session_state", _app_source("nobet_algoritma.py"))

    def test_tazelenmeyen_yerel_modul_kalmaz(self):
        # Klasör taraması yerine elle tutulan liste: yeni bir .py eklenip listelere
        # yazılmazsa canlıda eski kopyası hafızada kalabilirdi; burada yakalanır.
        source = _app_source("nobet.py")
        listed = set()
        for key in ("_FRESH_MODULES", "_UI_MODULES"):
            listed |= set(re.findall(r'"(\w+)"', re.search(key + r" = \((.*?)\)", source, flags=re.S).group(1)))
        local = {
            path.stem for path in Path(__file__).parent.glob("*.py")
            if not path.name.startswith(("test_", "conftest"))
        }
        self.assertEqual(local - listed, {"nobet", "nobet_tazele"})

    def test_tum_arayuz_dosyalari_yeniden_okunur(self):
        source = _app_source("nobet.py")
        listed = re.search(r"_UI_MODULES = \((.*?)\)", source, flags=re.S).group(1)
        self.assertEqual(
            sorted(re.findall(r'"(\w+)"', listed)),
            sorted(name[:-3] for name in APP_FILES if name != "nobet.py"),
        )

    def test_yenileme_yeni_fonksiyonlari_getirir(self):
        import importlib, types
        stale = types.ModuleType("db")
        stale.init_db = lambda: True
        with patch.dict("sys.modules", {"db": stale}):
            with self.assertRaises(ImportError):
                exec("from db import resolve_share_token", {})
        fresh = importlib.import_module("db")
        importlib.reload(fresh)
        self.assertTrue(hasattr(fresh, "resolve_share_token"))


class TestAppModulesStatic(unittest.TestCase):
    def test_arayuz_dosyalarinda_tanimsiz_ad_yok(self):
        # Bölümler ayrı dosyada fonksiyon olduğundan unutulan bir ad (ör. eksik
        # parametre) sayfa açılmadan önce burada yakalanır.
        try:
            import ast
            from pyflakes import checker, messages
        except ImportError:
            self.skipTest("pyflakes yüklü değil")
        undefined = (messages.UndefinedName, messages.UndefinedLocal, messages.UndefinedExport)
        for name in APP_FILES:
            tree = ast.parse(_app_source(name), filename=name)
            found = [str(m) for m in checker.Checker(tree, filename=name).messages if isinstance(m, undefined)]
            self.assertEqual(found, [], name)


class TestSharedPageHelpers(unittest.TestCase):
    ROWS = [
        {"Tarih": "30.10.2026 Cum", "AYB": "Ali", "GYB": "Ayşe"},
        {"Tarih": "31.10.2026 Cmt", "AYB": "Can", "GYB": "-"},
        {"Tarih": "29.10.2026 Per", "AYB": "Deniz", "GYB": "Ece"},
    ]

    def test_ilk_acilan_ay(self):
        from datetime import date
        months = [(2026, 9), (2026, 10), (2026, 12)]
        self.assertEqual(shared_default_month(months, date(2026, 10, 17)), 1)
        # Bu ay kayıtlı değilse bugünden önceki son ay.
        self.assertEqual(shared_default_month(months, date(2026, 11, 5)), 1)
        self.assertEqual(shared_default_month(months, date(2026, 8, 1)), 0)
        self.assertEqual(shared_default_month(months, date(2027, 3, 1)), 2)
        self.assertIsNone(shared_default_month([], date(2026, 10, 17)))

    def test_kalici_linkte_ileri_aylar_gizlenir(self):
        from datetime import date
        months = [(2025, 12), (2026, 10), (2026, 11), (2026, 12), (2027, 1)]
        self.assertEqual(
            shared_visible_months(months, date(2026, 10, 17)),
            [(2025, 12), (2026, 10), (2026, 11)],
        )
        # Aralık'ta sonraki ay yeni yılın Ocak'ı.
        self.assertEqual(shared_visible_months(months, date(2026, 12, 31)), months)

    def test_gun_durumlari(self):
        from datetime import date
        self.assertEqual(
            shared_day_states(self.ROWS, 2026, 10, date(2026, 10, 30)),
            ["today", "future", "past"],
        )
        self.assertEqual(shared_day_states([{"Tarih": "?"}], 2026, 10, date(2026, 10, 30)), ["future"])

    def test_bugun_ve_yarin_nobetciler(self):
        from datetime import date
        self.assertEqual(
            on_duty_summary(self.ROWS, ["AYB", "GYB"], 2026, 10, date(2026, 10, 30)),
            [("Bugün", "Ali (AYB) · Ayşe (GYB)"), ("Yarın", "Can (AYB)")],
        )
        # Ayın son günü: yarın bu ayda değil, yalnızca bugün yazılır.
        self.assertEqual(
            on_duty_summary(self.ROWS, ["AYB", "GYB"], 2026, 10, date(2026, 10, 31)),
            [("Bugün", "Can (AYB)")],
        )
        # Başka ay açıkken kutu çıkmaz.
        self.assertEqual(on_duty_summary(self.ROWS, ["AYB", "GYB"], 2026, 10, date(2026, 11, 2)), [])
        plain = [{"Tarih": "05.11.2026 Per", "Nöbetçiler": "Ali, Can"}]
        self.assertEqual(on_duty_summary(plain, ["Nöbetçiler"], 2026, 11, date(2026, 11, 5)),
                         [("Bugün", "Ali, Can")])

    def test_son_guncelleme_metni(self):
        from datetime import datetime
        self.assertEqual(shared_updated_text(datetime(2026, 10, 17, 9, 5)), "17 Ekim 2026, 09:05")
        self.assertEqual(shared_updated_text(None), "")

    def test_paylas_dugmesi_tek_kalici_baglanti_kullanir(self):
        source = _app_source()
        share = source[source.index('with st.popover("🔗 Paylaş"'):source.index("with _snapshot_col:")]
        self.assertIn("create_share_link(current_owner, 0, 0)", share)
        self.assertNotIn("current_owner, yil, ay", share)
        self.assertEqual(share.count("delete_all_share_links(current_owner)"), 2)


class TestArifeMesai(unittest.TestCase):
    def _details(self, yil, ay, holiday_text=None):
        import calendar
        days = calendar.monthrange(yil, ay)[1]
        text = default_holiday_text(yil, ay) if holiday_text is None else holiday_text
        return build_gun_detaylari(yil, ay, days, parse_holiday_days(text, days)[0])

    def test_arifeler(self):
        self.assertEqual(official_half_days(2026, 10), [(28, "Cumhuriyet Bayramı arifesi")])
        self.assertEqual(official_half_days(2026, 3), [(19, "Ramazan Bayramı arifesi")])
        self.assertEqual(official_half_days(2026, 5), [(26, "Kurban Bayramı arifesi")])
        self.assertEqual(official_half_days(2026, 12), [])

    def test_ise_dusen_arife_yarim_gun_sayilir(self):
        # Ekim 2026: 21 iş günü, 28 Ekim Çarşamba yarım gün.
        hours, counted = required_work_hours(self._details(2026, 10), official_half_days(2026, 10))
        self.assertEqual(hours, 20 * 8 + 4)
        self.assertEqual(counted, [(28, "Cumhuriyet Bayramı arifesi")])

    def test_tatil_yapilan_ya_da_hafta_sonuna_dusen_arife_dusulmez(self):
        details = self._details(2026, 10, "28, 29")
        hours, counted = required_work_hours(details, official_half_days(2026, 10))
        self.assertEqual((hours, counted), (20 * 8, []))
        # 28 Ekim 2029 Pazar: arife hafta sonunda.
        hours, counted = required_work_hours(self._details(2029, 10), official_half_days(2029, 10))
        self.assertEqual(counted, [])

    def test_arifesiz_ay_eski_hesapla_ayni(self):
        details = self._details(2026, 11)
        working = sum(1 for d in details.values() if not (d["weekend"] or d["holiday"]))
        self.assertEqual(required_work_hours(details, official_half_days(2026, 11)), (working * 8, []))


class TestSecmeliAyarlar(unittest.TestCase):
    def test_capraz_tablo_ikiliyi_ekler_ve_cikarir(self):
        team = ["Ali", "Ayşe", "Can"]
        text = toggle_pair_text("", team, ("Can", "Ali"))
        self.assertEqual(text, "Ali-Can")
        text = toggle_pair_text(text, team, ("Ayşe", "Can"))
        self.assertEqual(text, "Ali-Can, Ayşe-Can")
        # Ters yönden tıklamak aynı ikiliyi kaldırır.
        self.assertEqual(toggle_pair_text(text, team, ("Ali", "Can")), "Ayşe-Can")

    def test_ad_degisikligi_ikililere_tasinir(self):
        text = "Ali-Ayşe-Nur, Ali-Can"
        renamed = rename_in_pairs_text(text, ["Ali", "Ayşe-Nur", "Can"], {"Ayşe-Nur": "Ayşe Nur"})
        self.assertEqual(renamed, "Ali-Ayşe Nur, Ali-Can")

    def test_ikili_metni_geri_okunur(self):
        team = ["Ayşe-Nur", "Ali", "Can"]
        pairs = {("Ali", "Ayşe-Nur"), ("Can", "Ali")}
        text = forbidden_pairs_to_text(pairs)
        self.assertEqual(text, "Ali-Ayşe-Nur, Ali-Can")
        self.assertEqual(parse_forbidden_pairs(text, team), {("Ali", "Ayşe-Nur"), ("Ali", "Can")})
        self.assertEqual(forbidden_pairs_to_text([]), "")

    def test_limit_metni_geri_okunur(self):
        limits = {"Ali": {"min": 2, "max": 5}, "Can": {"min": 1, "max": 999}}
        text = person_limits_to_text(limits)
        self.assertEqual(text, "Ali:2-5\nCan:1-999")
        self.assertEqual(parse_person_limits_with_issues(text), (limits, [], []))
        self.assertEqual(person_limits_to_text({}), "")

    def test_limit_metni_hafta_sonu_siniriyla(self):
        limits = {"Ali": {"min": 0, "max": 999, "max_special": 2}, "Can": {"min": 0, "max": 6}}
        text = person_limits_to_text(limits)
        self.assertEqual(text, "Ali:0-999:2\nCan:0-6")
        self.assertEqual(parse_person_limits_with_issues(text), (limits, [], []))
        self.assertEqual(special_cap(limits, "Ali"), 2)
        self.assertEqual(special_cap(limits, "Can"), 999)
        self.assertFalse(has_total_limit(limits["Ali"]))
        self.assertTrue(has_total_limit(limits["Can"]))
        # Bozuk hafta sonu değeri satırı geçersiz yapar.
        self.assertEqual(parse_person_limits_with_issues("Ali:0-5:x")[1], ["Ali:0-5:x"])
        self.assertEqual(len(parse_person_limits_with_issues("Ali:0-5:-1")[2]), 1)

    def test_ekip_tablosu_metinlere_donusur(self):
        rows = team_settings_rows(
            ["Ali", "Ayşe", "Can"],
            {"Ali": {"min": 2, "max": 8}, "Can": {"min": 0, "max": 999, "max_special": 1}},
        )
        # Tabloda ayrı kutu olmasa da kayıtlı "en az" durur.
        self.assertEqual(rows[0], {"name": "Ali", "min": 2, "max": 8, "max_special": None})
        self.assertEqual(rows[1], {"name": "Ayşe", "min": 0, "max": None, "max_special": None})
        self.assertEqual(rows[2]["max_special"], 1)
        names, limits = team_settings_texts(rows)
        self.assertEqual(names, "Ali, Ayşe, Can")
        self.assertEqual(limits, "Ali:2-8\nCan:0-999:1")
        # Ayşe:3-8 tabloya girip çıkınca aynı metin kalır.
        parsed = parse_person_limits("Ayşe:3-8")
        round_rows = compact_team_rows(team_settings_rows(["Ayşe"], parsed))
        _, again = team_settings_texts(round_rows)
        self.assertEqual(again, "Ayşe:3-8")

    def test_ekip_tablosu_bos_satir_ve_silme(self):
        rows = [
            {"name": "Ali", "min": 0, "max": 5, "max_special": None},
            {"name": "", "min": 0, "max": 3, "max_special": None},
            {"name": "Can", "min": 0, "max": None, "max_special": None},
        ]
        compacted = compact_team_rows(rows)
        # Adı silinen satır (limitiyle) gider, sonda tek boş satır kalır.
        self.assertEqual([r["name"] for r in compacted], ["Ali", "Can", ""])
        self.assertEqual(compacted[0]["max"], 5)
        self.assertEqual(compact_team_rows([])[0]["name"], "")
        # Virgülle yapıştırılan liste satırlara bölünür.
        pasted = compact_team_rows([{"name": "Ali, Ayşe ,Can", "min": 0, "max": 4, "max_special": None}])
        self.assertEqual([r["name"] for r in pasted], ["Ali", "Ayşe", "Can", ""])
        self.assertEqual([r["max"] for r in pasted], [4, None, None, None])
        # Virgül ismi bölmez; aynı isim tekrar yazılırsa limit ilk satırdan.
        names, limits = team_settings_texts([
            {"name": "Ali, Veli", "max": 4},
            {"name": "Ali, Veli", "max": 9},
        ])
        self.assertEqual(names, "Ali Veli, Ali Veli")
        self.assertEqual(limits, "Ali Veli:0-4")

    def test_ayarlar_tablo_ile_girilir(self):
        source = _app_source()
        self.assertNotIn('st.text_area(\n            "🚫 Birlikte', source)
        self.assertNotIn('st.text_area(\n            "👥 Ekip', source)
        self.assertNotIn('"📊 Kişisel Limitler:"', source)
        self.assertIn("render_team_table(draw=settings_expanded)", source)
        self.assertIn("render_pair_matrix(isimler)", source)


class TestTercihOzeti(unittest.TestCase):
    def test_kisi_basina_tercih_sonuclari(self):
        cols = ["1 Pzt", "2 Sal", "3 Çar", "4 Per"]
        schedule = pd.DataFrame(False, index=["Ali", "Ayşe"], columns=cols)
        schedule.loc["Ali", ["1 Pzt", "3 Çar"]] = True
        schedule.loc["Ayşe", ["2 Sal", "4 Per"]] = True
        prefs = pd.DataFrame(0, index=["Ali", "Ayşe"], columns=cols)
        prefs.loc["Ali", ["1 Pzt", "2 Sal"]] = 1
        prefs.at["Ali", "3 Çar"] = 2
        prefs.at["Ayşe", "4 Per"] = 3
        prefs.at["Ayşe", "2 Sal"] = 4
        summary = preference_summary(schedule, prefs, ["Ali", "Ayşe"])
        self.assertEqual(summary.loc["Ali"].tolist(), [2, 1, "%50", 1, 1, 0, 0, 0])
        self.assertEqual(summary.loc["Ayşe"].tolist(), [0, 0, "-", 0, 0, 1, 1, 1])

    def test_isaret_yoksa_ozet_yok(self):
        schedule = pd.DataFrame(True, index=["Ali"], columns=["1 Pzt"])
        prefs = pd.DataFrame(0, index=["Ali"], columns=["1 Pzt"])
        self.assertIsNone(preference_summary(schedule, prefs, ["Ali"]))
        # Tercih tablosunda olmayan kişi ya da gün sorun çıkarmaz.
        self.assertIsNone(preference_summary(schedule, prefs.iloc[0:0], ["Ali"]))


class TestTercihListesiPaylasimi(unittest.TestCase):
    def test_kisi_basina_gun_listesi_ve_whatsapp_metni(self):
        cols = [f"{d} X" for d in range(1, 15)]
        prefs = pd.DataFrame(0, index=["Görkem", "Faruk", "Can"], columns=cols)
        for d in (3, 5, 10, 11, 12):
            prefs.at["Görkem", f"{d} X"] = 1
        prefs.at["Görkem", "7 X"] = 2
        for d in (1, 2, 3):
            prefs.at["Faruk", f"{d} X"] = 3
        prefs.at["Faruk", "14 X"] = 4
        rows = preference_overview(prefs, ["Görkem", "Faruk", "Can", "Yeni"])
        self.assertEqual(rows[0], {"İsim": "Görkem", "Tercih": "3, 5, 10-12", "Kaçın": "7",
                                   "İzin": "-", "Müsait Değil": "-"})
        self.assertEqual(rows[1]["İzin"], "1-3")
        self.assertEqual(rows[1]["Müsait Değil"], "14")
        # İşareti olmayan ya da tabloda olmayan kişi satırda kalır (teyit için).
        self.assertEqual([r["İsim"] for r in rows], ["Görkem", "Faruk", "Can", "Yeni"])
        self.assertEqual(rows[3]["Tercih"], "-")
        text = build_preference_share_text(rows, 2026, 11)
        self.assertIn("Kasım 2026", text)
        self.assertIn("*Görkem* — Tercih: 3, 5, 10-12 · Kaçın: 7", text)
        self.assertIn("*Faruk* — İzin: 1-3 · Müsait Değil: 14", text)
        self.assertIn("*Can* — talep yok", text)

    def test_isaret_yoksa_bos(self):
        prefs = pd.DataFrame(0, index=["Ali"], columns=["1 Pzt"])
        self.assertEqual(preference_overview(prefs, ["Ali"]), [])
        self.assertEqual(preference_overview(pd.DataFrame(), ["Ali"]), [])


class TestShareLinksDb(unittest.TestCase):
    def test_baglanti_olusturulur_ve_cozumlenir(self):
        cursor = _FakeDbCursor()
        with patch.object(db.psycopg2, "connect", return_value=_FakeDbConnection(cursor)):
            token = db.create_share_link("ali@gmail.com", 2026, 10)
        self.assertTrue(token and len(token) >= 16)
        sql, params = cursor.executed[0]
        self.assertIn("ON CONFLICT (owner, year, month)", sql)
        self.assertEqual(params, (token, "ali@gmail.com", 2026, 10))

        lookup = _FakeDbCursor(schedule_row={"owner": "ali@gmail.com", "year": 2026, "month": 10})
        with patch.object(db.psycopg2, "connect", return_value=_FakeDbConnection(lookup)):
            self.assertEqual(db.resolve_share_token(token), ("ali@gmail.com", 2026, 10))

    def test_gecersiz_token_veritabanina_gitmez(self):
        with patch.object(db.psycopg2, "connect") as connect:
            self.assertIsNone(db.resolve_share_token(""))
            self.assertIsNone(db.resolve_share_token("x" * 65))
            self.assertIsNone(db.resolve_share_token(None))
        connect.assert_not_called()

    def test_paylasim_sayfasi_giristen_once_ve_salt_okunur(self):
        source = _app_source("nobet.py")
        self.assertLess(source.index('st.query_params.get("paylas")'),
                        source.index("AUTH_ENABLED = _auth_configured()"))
        shared = _app_source("nobet_paylasim.py")
        for forbidden in ("save_schedule", "save_app_settings", "st.button("):
            self.assertNotIn(forbidden, shared)


class TestSharedCoreFunctions(unittest.TestCase):
    def test_limit_sorunlari_kullaniciya_gosterilmek_uzere_ayrilir(self):
        limits, invalid, errors = parse_person_limits_with_issues(
            "Ali:2-5\nbozuk satır\nAyşe:-1-3\nMehmet:6-2\n:1-2\nFatma:x-3"
        )
        self.assertEqual(limits, {"Ali": {"min": 2, "max": 5}})
        self.assertEqual(invalid, ["bozuk satır", ":1-2", "Fatma:x-3"])
        self.assertEqual(len(errors), 2)
        self.assertIn("negatif", errors[0])
        self.assertIn("min (6) > max (2)", errors[1])

    def test_ekipte_olmayan_isimler(self):
        self.assertEqual(unknown_names(["Zeynep", "Ali", "Zeynep", "Can"], ["Ali"]), ["Can", "Zeynep"])

    def test_uygulama_kopya_fonksiyon_tanimlamaz(self):
        source = _app_source()
        for name in ("escape_html", "validate_inputs"):
            self.assertNotIn(f"\ndef {name}(", source, f"{name} nobet_core'dan alınmalı")
        self.assertIn("build_gun_detaylari(yil, ay, gun_sayisi, tatil_gunleri)", source)
        self.assertIn("parse_person_limits_with_issues(limits_text)", source)
        self.assertIn("parse_forbidden_pairs(forbidden_input, isimler)", source)


class TestAssignmentWarnings(unittest.TestCase):
    def setUp(self):
        self.names = ["Ali", "Ayşe", "Mehmet"]
        self.columns = ["1 Çar", "2 Per", "3 Cum", "4 Cmt"]
        self.days = {c: {"day_num": i + 1} for i, c in enumerate(self.columns)}
        self.schedule = pd.DataFrame(False, index=self.names, columns=self.columns)

    def _warn(self, column="2 Per", person="Ali", **kwargs):
        self.schedule.at[person, column] = True
        params = dict(min_rest=1)
        params.update(kwargs)
        return assignment_warnings(self.schedule, person, column, self.columns, self.days, **params)

    def test_atanmamis_hucre_icin_uyari_yok(self):
        self.assertEqual(
            assignment_warnings(self.schedule, "Ali", "2 Per", self.columns, self.days, 1), []
        )

    def test_onceki_ve_sonraki_gun_nobeti_uyarir(self):
        self.schedule.at["Ali", "1 Çar"] = True
        self.schedule.at["Ali", "3 Cum"] = True
        warnings = self._warn()
        self.assertEqual(len(warnings), 1)
        self.assertIn("1, 3. günde de nöbeti var", warnings[0])

    def test_dinlenme_sifirsa_bitisik_gun_serbest(self):
        self.schedule.at["Ali", "1 Çar"] = True
        self.assertEqual(self._warn(min_rest=0), [])

    def test_max_limit_asilinca_uyarir(self):
        self.schedule.at["Ali", "4 Cmt"] = True
        warnings = self._warn(column="1 Çar", person_limits={"Ali": {"min": 0, "max": 1}})
        self.assertTrue(any("en fazla 1" in w for w in warnings))

    def test_tercih_yasak_cift_ve_kapasite_uyarilari(self):
        prefs = pd.DataFrame(0, index=self.names, columns=self.columns)
        prefs.at["Ali", "2 Per"] = 3
        self.schedule.at["Ayşe", "2 Per"] = True
        self.schedule.at["Mehmet", "2 Per"] = True
        warnings = self._warn(
            preferences=prefs,
            forbidden_pairs={("Ali", "Ayşe")},
            people_per_shift=2,
        )
        joined = " | ".join(warnings)
        self.assertIn("izinli olarak işaretli", joined)
        self.assertIn("Ali ve Ayşe birlikte çalışamaz", joined)
        self.assertIn("3 kişi atandı, en fazla 2", joined)


class TestGoogleLoginSupport(unittest.TestCase):
    def test_eposta_tek_bicime_getirilir(self):
        self.assertEqual(normalize_owner_email("  Ali@Gmail.COM "), "ali@gmail.com")
        self.assertEqual(normalize_owner_email(None), "")

    def test_ayar_anahtari_kullaniciya_ozel(self):
        self.assertEqual(db.settings_key(""), "default")
        self.assertEqual(db.settings_key("ali@gmail.com"), "user:ali@gmail.com")

    def test_cizelge_sorgulari_kullaniciya_gore_filtrelenir(self):
        cursor = _FakeDbCursor()
        schedule = pd.DataFrame([[True]], index=["Ali"], columns=["1 Çar"])
        with patch.object(db.psycopg2, "connect", return_value=_FakeDbConnection(cursor)), \
                patch.object(db, "execute_values"):
            self.assertTrue(db.save_schedule("Ocak", 2099, 1, ["Ali"], schedule, owner="ali@gmail.com", expected_version=db.FORCE))
        insert_sql, insert_params = _writes(cursor)[0]
        self.assertIn("ON CONFLICT (owner, name, year, month)", insert_sql)
        self.assertEqual(insert_params[7], "ali@gmail.com")

        load_cursor = _FakeDbCursor()
        load_cursor.fetchone = lambda: None
        with patch.object(db.psycopg2, "connect", return_value=_FakeDbConnection(load_cursor)):
            db.load_schedule("Ocak", 2025, 1, owner="ali@gmail.com")
        self.assertEqual(load_cursor.executed[0][1][0], "ali@gmail.com")

    def test_eski_veri_kullaniciya_aktarilir(self):
        cursor = _FakeDbCursor()
        with patch.object(db.psycopg2, "connect", return_value=_FakeDbConnection(cursor)):
            self.assertTrue(db.claim_legacy_data("ali@gmail.com"))
        self.assertIn("WHERE legacy.owner = ''", cursor.executed[0][0])
        self.assertEqual(cursor.executed[1][1], ("user:ali@gmail.com",))
        self.assertFalse(db.claim_legacy_data(""))

    def test_uygulama_tum_kayitlari_kullaniciya_baglar(self):
        source = _app_source()
        calls = re.findall(r"(?<!def )\b(save_schedule|load_schedule|save_app_settings|load_app_settings)\((.*?)\)\n",
                           source, flags=re.S)
        self.assertGreaterEqual(len(calls), 6)
        for name, args in calls:
            # Paylaşım sayfası bağlantının sahibine ait listeyi okur (owner=_owner).
            self.assertTrue(
                "owner=current_owner" in args or "owner=_owner" in args,
                f"{name} kullanıcıya bağlı değil",
            )
        shared = _app_source("nobet_paylasim.py")
        # Liste (load_schedule), talep sekmesindeki ekip (load_app_settings) ve
        # kişinin ücreti (load_schedule + load_app_settings) ve yıl özeti
        # (load_auto_saved_assignments + load_app_settings).
        self.assertEqual(shared.count("owner=_owner"), 6)
        self.assertNotIn("owner=_owner", source.replace(shared, ""))
        # Giriş kontrolü ayarlar yüklenmeden önce yapılmalı.
        self.assertLess(source.index("AUTH_ENABLED = _auth_configured()"),
                        source.index("load_app_settings(owner=current_owner"))

    def test_indirme_dosyalari_tiklaninca_uretilir(self):
        source = _app_source()
        self.assertIn('st.download_button("🖼️ Resim (PNG)", _png_bytes,', source)
        self.assertNotIn("st.session_state[png_key]", source)


class TestUnvanSaatUcreti(unittest.TestCase):
    # 2026 Temmuz-Aralık resmi tablo (net, TL): normal, gece, riskli, riskli
    # gece, bayram, bayram gece, bayram riskli, bayram riskli gece.
    TABLO_2026_2 = {
        "Eğitim Görevlisi": (250.17, 275.19, 375.25, 412.78, 312.71, 343.98, 469.07, 515.97),
        "Uzman Tabip": (250.17, 275.19, 375.25, 412.78, 312.71, 343.98, 469.07, 515.97),
        "Tabip": (226.72, 249.39, 340.07, 374.08, 283.39, 311.73, 425.09, 467.60),
        "Tıpta uzmanlık tüzüğüne göre": (226.72, 249.39, 340.07, 374.08, 283.39, 311.73, 425.09, 467.60),
        "Diş Tabibi ve Eczacılar": (203.28, 223.59, 304.89, 335.38, 254.08, 279.49, 381.12, 419.23),
        "Sağlık Personeli (Lisans-Önlisans)": (156.36, 171.99, 234.55, 257.99, 195.44, 214.99, 293.17, 322.48),
        "Sağlık Personeli (Lise)": (132.90, 146.19, 199.35, 219.29, 166.13, 182.74, 249.19, 274.11),
        "Diğer Personel": (101.63, 111.78, 152.46, 167.68, 127.04, 139.74, 190.65, 209.61),
        "4/B Sağlık Personeli (Lisans-Önlisans)": (134.30, 147.73, 201.45, 221.59, 167.87, 184.66, 251.81, 276.99),
        "4/B Sağlık Personeli (Lise)": (114.15, 125.57, 171.25, 188.35, 142.69, 156.96, 214.04, 235.44),
        "4/B Diğer Personel": (87.28, 96.02, 130.84, 144.03, 109.12, 120.03, 163.58, 180.04),
    }

    def test_2026_temmuz_tablosuyla_kurusu_kurusuna_ayni(self):
        self.assertEqual(set(NOBET_UNVANLARI), set(self.TABLO_2026_2))
        for unvan, t in self.TABLO_2026_2.items():
            for riskli, beklenen in ((False, t[0:2] + t[4:6]), (True, t[2:4] + t[6:8])):
                info = unvan_saat_ucreti(unvan, 2026, 10, riskli)
                got = (info["gunduz"], info["gece"], info["tatil"], info["tatil_gece"])
                self.assertEqual(got, beklenen, (unvan, riskli))

    def test_formul_tabloya_yakin(self):
        # Tablosu olmayan dönemde kullanılan formül tabloya 10 kuruşa kadar yakın
        # (tablonun kendi yuvarlamaları ve 4/B karşılığı yüzünden tam değil).
        from nobet_core import NOBET_UCRET_TABLOSU
        saved = NOBET_UCRET_TABLOSU.pop((2026, 2))
        try:
            for unvan, t in self.TABLO_2026_2.items():
                for riskli, beklenen in ((False, t[0:2] + t[4:6]), (True, t[2:4] + t[6:8])):
                    info = unvan_saat_ucreti(unvan, 2026, 10, riskli)
                    got = (info["gunduz"], info["gece"], info["tatil"], info["tatil_gece"])
                    for g, e in zip(got, beklenen):
                        self.assertAlmostEqual(g, e, delta=0.1, msg=(unvan, riskli))
        finally:
            NOBET_UCRET_TABLOSU[(2026, 2)] = saved

    def test_ocak_haziran_sendika_tablosu(self):
        beklenen = {"Uzman Tabip": 220.37, "Tabip": 199.71,
                    "Sağlık Personeli (Lisans-Önlisans)": 137.73, "Sağlık Personeli (Lise)": 117.07}
        for unvan, tutar in beklenen.items():
            self.assertAlmostEqual(unvan_saat_ucreti(unvan, 2026, 3)["gunduz"], tutar, delta=0.02)

    def test_bilinmeyen_donem_son_katsayiyi_kullanir(self):
        self.assertEqual(katsayi_donemi(2026, 7), ((2026, 2), True))
        self.assertEqual(katsayi_donemi(2027, 1), ((2026, 2), False))
        self.assertEqual(katsayi_donemi(2025, 12), ((2026, 1), False))
        self.assertIsNone(unvan_saat_ucreti("Profesör", 2026, 10))
        # Tablo yalnız kendi döneminde; 2027'de formül (son katsayı) kullanılır.
        self.assertFalse(unvan_saat_ucreti("Tabip", 2027, 3)["tam"])

    def test_katsayi_uyarisi_yalniz_eksik_donemde(self):
        self.assertIsNone(katsayi_uyarisi(2026, 10))
        self.assertIsNone(katsayi_uyarisi(2026, 3))
        uyari = katsayi_uyarisi(2027, 1)
        self.assertIn("henüz eklenmedi", uyari)
        self.assertIn("Temmuz-Aralık 2026", uyari)

    UCRET = {"gunduz": 100.0, "gece": 110.0, "tatil": 125.0, "tatil_gece": 137.5}

    def test_gece_ucreti_fazla_mesai_saatiyle_sinirli(self):
        # 10 nöbet = 240 saat, 120 saati gece; 168 saat mesai -> 72 saat
        # fazla mesai. Gece ücreti 120 değil 72 saate ödenir.
        saat, fm, ucret = nobet_ucreti_hesapla(10, 0, 168, self.UCRET)
        self.assertEqual((saat, fm), (240, 72))
        self.assertAlmostEqual(ucret, 72 * 110.0)
        # Gece saati fazla mesaiden azsa kalan fazla mesai gündüz ücretiyle:
        # 12 nöbet = 288 saat, 144 gece; 120 saat mesai -> 168 fm.
        self.assertAlmostEqual(nobet_ucreti_hesapla(12, 0, 120, self.UCRET)[2], 144 * 110.0 + 24 * 100.0)
        # Fazla mesai yoksa ücret yok; tek sayı verilince her saat aynı ücret.
        self.assertEqual(nobet_ucreti_hesapla(6, 0, 160, self.UCRET)[2], 0)
        self.assertEqual(nobet_ucreti_hesapla(10, 1, 168, 100.0)[2], 7200.0)

    def test_bayram_nobeti_bayram_ucretiyle(self):
        # 8 nöbet, 1'i bayram; 160 saat mesai -> 32 fm. Bayramın 24 saati
        # (12 gündüz, 12 gece) fm'ye girer; kalan 8 saat gece ücretiyle.
        self.assertAlmostEqual(nobet_ucreti_hesapla(8, 1, 160, self.UCRET)[2],
                               12 * 125.0 + 12 * 137.5 + 8 * 110.0)
        # Bayram saati fazla mesaiyi aşamaz; önce gece saatleri: 7 nöbet,
        # 2 bayram, 8 fm -> 8 saat bayram gece.
        self.assertAlmostEqual(nobet_ucreti_hesapla(7, 2, 160, self.UCRET)[2], 8 * 137.5)

    def test_elle_gunduz_ucretinden_turetilir(self):
        self.assertEqual(saat_ucretleri(200.0, 0.25, 0.10),
                         {"gunduz": 200.0, "gece": 220.0, "tatil": 250.0, "tatil_gece": 275.0})

    def test_elle_eklenen_tatile_bayram_ucreti_yok(self):
        # 29 Ekim resmi tatil, 30 Ekim elle eklendi; zorunlu mesai 0.
        days = build_gun_detaylari(2026, 10, 31, [29, 30])
        schedule = pd.DataFrame(False, index=["Ali"], columns=list(days))
        schedule.at["Ali", "29 Per"] = True
        schedule.at["Ali", "30 Cum"] = True
        _, finance, _ = build_schedule_analysis(
            schedule, [], [], ["Ali"], days, {}, 0, self.UCRET, None, frozenset({29}))
        # 48 saat fm; 29 Ekim bayram ücretiyle, 30 Ekim normal ücretle.
        self.assertAlmostEqual(float(finance["Ücret (TL)"].iloc[0]),
                               12 * 125.0 + 12 * 137.5 + 12 * 100.0 + 12 * 110.0)

    def test_ucret_dokumu_toplami_hesapla_ile_ayni(self):
        from nobet_core import ucret_saatleri, UCRET_TURLERI
        for args in [(10, 0, 168), (12, 0, 120), (8, 1, 160), (7, 2, 160), (6, 0, 160), (15, 3, 100)]:
            saat, fm, satirlar = ucret_saatleri(*args, self.UCRET)
            self.assertEqual((saat, fm), nobet_ucreti_hesapla(*args, self.UCRET)[:2])
            self.assertEqual(sum(_t for _, _, _t in satirlar.values()), nobet_ucreti_hesapla(*args, self.UCRET)[2])
            self.assertEqual(sum(satirlar[_k][0] for _k in UCRET_TURLERI), fm)

    def test_ayarlardaki_saat_ucretleri_ana_sayfa_ile_ayni(self):
        from nobet_core import ayarlardaki_saat_ucretleri
        tablo = unvan_saat_ucreti("Tabip", 2026, 10, True)
        oran = ayarlardaki_saat_ucretleri({"title": "Tabip", "risky_unit": True}, 2026, 10)
        self.assertEqual(oran["ucretler"]["gunduz"], 340.07)
        self.assertEqual(oran["ucretler"]["tatil_gece"], tablo["tatil_gece"])
        self.assertFalse(oran["elle"])
        # Aynı dönem için elle değiştirilen gündüz ücretinden türetilir.
        elle = ayarlardaki_saat_ucretleri(
            {"title": "Tabip", "hourly_rate": 200.0, "hourly_rate_source": "Tabip|2026-2|0|g"}, 2026, 10)
        self.assertTrue(elle["elle"])
        self.assertEqual(elle["ucretler"], saat_ucretleri(200.0, 0.25, 0.10))
        # Başka dönemin elle ücreti bu aya taşınmaz (ana sayfa da tabloya döner).
        self.assertFalse(ayarlardaki_saat_ucretleri(
            {"title": "Tabip", "hourly_rate": 200.0, "hourly_rate_source": "Tabip|2026-1|0|g"}, 2026, 10)["elle"])
        self.assertEqual(ayarlardaki_saat_ucretleri({"title": "Elle gir", "hourly_rate": 150}, 2026, 10)["ucretler"], 150.0)
        # Ücret hiç girilmemişse gösterilmez.
        self.assertIsNone(ayarlardaki_saat_ucretleri({}, 2026, 10))
        self.assertIsNone(ayarlardaki_saat_ucretleri({"hourly_rate": 1.0}, 2026, 10))

    def test_kisi_ucret_dokumu_analiz_ucret_ozeti_ile_ayni(self):
        from nobet_core import kisi_ucret_dokumu, ayarlardaki_saat_ucretleri
        # Ekim 2026: 29 resmi bayram, 30 elle eklenmiş tatil, 28 arife; Ali 1 Ekim izinli.
        settings = {"title": "Tabip", "holidays_by_period": {"2026-10": "29,30"}}
        days = build_gun_detaylari(2026, 10, 31, [29, 30])
        cols = list(days)
        schedule = pd.DataFrame(False, index=["Ali"], columns=cols)
        nobet = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 30]
        for _d in nobet:
            schedule.at["Ali", cols[_d - 1]] = True
        prefs = pd.DataFrame(0, index=["Ali"], columns=cols)
        prefs.at["Ali", cols[0]] = 3
        half = official_half_days(2026, 10)
        zorunlu, _ = required_work_hours(days, half)
        mesai = personal_work_hours(days, half, prefs, ["Ali"])
        ucretler = ayarlardaki_saat_ucretleri(settings, 2026, 10)["ucretler"]
        _, finance, _ = build_schedule_analysis(
            schedule, [], [], ["Ali"], days, {}, zorunlu, ucretler, mesai, frozenset({29}))
        dokum = kisi_ucret_dokumu(nobet, 2026, 10, settings, prefs, "Ali")
        self.assertEqual(dokum["nobet"], 11)
        self.assertEqual(dokum["bayram_nobet"], 1)
        self.assertEqual(dokum["mesai"], zorunlu - 8)
        self.assertEqual(round(dokum["toplam"], 2), float(finance.loc["Ali", "Ücret (TL)"]))
        self.assertEqual(int(dokum["fazla_mesai"]), int(finance.loc["Ali", "Fazla mesai"]))
        self.assertGreater(dokum["satirlar"]["tatil_gece"][0], 0)
        self.assertIsNone(kisi_ucret_dokumu(nobet, 2026, 10, {}, None, "Ali"))

    def test_unvan_ayari_saklanir_gecersizi_atlanir(self):
        state = {"unvan_val": "Tabip", "riskli_birim_val": True,
                 "nobet_ucreti_kaynak": "Tabip|2026-2|1|g"}
        restored = {}
        apply_app_settings(restored, json.loads(json.dumps(collect_app_settings(state))))
        self.assertEqual(restored, state)
        # Eski sürümün ünvan adı yenisine çevrilir.
        restored = {}
        apply_app_settings(restored, {"title": "Pratisyen Tabip"})
        self.assertEqual(restored, {"unvan_val": "Tabip"})
        restored = {}
        apply_app_settings(restored, {"title": "Profesör", "risky_unit": "evet"})
        self.assertEqual(restored, {})

    def test_tutarlar_turkce_bicimde(self):
        from nobet_core import tr_sayi
        self.assertEqual(tr_sayi(5985.36), "5.985,36")
        self.assertEqual(tr_sayi(1234567.891), "1.234.567,89")
        self.assertEqual(tr_sayi(0), "0,00")
        self.assertEqual(tr_sayi(8139, 0), "8.139")


class TestAppSettingsPersistence(unittest.TestCase):
    def test_ayarlar_toplanir_ve_aynen_geri_yuklenir(self):
        state = {
            "isimler_text": "Ali, Ayşe", "yil_val": 2026, "ay_val": 10,
            "kisi_sayisi_val": 2, "min_bosluk_val": 1,
            "holidays_by_period": {"2026-10": "29", "2026-11": "10"},
            "nobet_ucreti_val": 1.5, "role_names_text": "AYB, GYB",
            "forbidden_pairs_text": "Ali-Ayşe", "limits_text": "Ali:3-8",
            "unrelated": "kept out",
        }
        settings = collect_app_settings(state)
        self.assertNotIn("unrelated", json.dumps(settings))

        restored = {}
        apply_app_settings(restored, json.loads(json.dumps(settings)))
        state.pop("unrelated")
        self.assertEqual(restored, state)

    def test_tatiller_aya_ozel_ve_kopya_bagimsiz(self):
        state = {"holidays_by_period": {"2026-10": "29"}}
        saved = collect_app_settings(state)
        state["holidays_by_period"]["2026-11"] = "10"
        # Kaydedilen kopya sonradan yapılan değişiklikten etkilenmemeli.
        self.assertEqual(saved["holidays_by_period"], {"2026-10": "29"})
        self.assertNotEqual(saved, collect_app_settings(state))

        restored = {}
        apply_app_settings(restored, {"holidays_by_period": {"2026-10": "29", "bad": "1", "2026-12": 5}})
        self.assertEqual(restored["holidays_by_period"], {"2026-10": "29"})

    def test_gecersiz_ayarlar_widget_hatasina_yol_acmaz(self):
        restored = {}
        apply_app_settings(restored, {
            "year": 1999, "month": 13, "people_per_shift": "2",
            "rest_days": True, "hourly_rate": -1, "team": ["Ali"],
        })
        self.assertEqual(restored, {})

    def test_db_ayarlari_kaydeder_ve_okur(self):
        settings = {"team": "Ali, Ayşe", "year": 2026, "month": 10}
        save_cursor = _FakeDbCursor()
        with patch.object(db.psycopg2, "connect", return_value=_FakeDbConnection(save_cursor)):
            self.assertTrue(db.save_app_settings(settings, expected_version=db.FORCE))
        stored_value = _writes(save_cursor)[0][1][1]

        load_cursor = _FakeDbCursor(schedule_row={"value": stored_value})
        with patch.object(db.psycopg2, "connect", return_value=_FakeDbConnection(load_cursor)):
            self.assertEqual(db.load_app_settings(), settings)

    def test_db_ayar_yoksa_none_hata_varsa_istisna(self):
        empty_cursor = _FakeDbCursor()
        empty_cursor.fetchone = lambda: None
        with patch.object(db.psycopg2, "connect", return_value=_FakeDbConnection(empty_cursor)):
            self.assertIsNone(db.load_app_settings())
        with patch.object(db.psycopg2, "connect", side_effect=RuntimeError("down")):
            with self.assertRaises(RuntimeError):
                db.load_app_settings(raise_on_error=True)

    def test_parmak_izi_her_degisikligi_ayirt_eder(self):
        schedule = pd.DataFrame([[False]], index=["Ali"], columns=["1 Çar"])
        prefs = pd.DataFrame([[0]], index=["Ali"], columns=["1 Çar"])
        rows = [{"Tarih": "01.10.2026 Çar", "Görev1": "-"}]
        base = persistence_fingerprint(schedule, prefs, rows)
        self.assertEqual(base, persistence_fingerprint(schedule.copy(), prefs.copy(), list(rows)))

        changed_schedule = schedule.copy(); changed_schedule.iloc[0, 0] = True
        changed_prefs = prefs.copy(); changed_prefs.iloc[0, 0] = 1
        changed_rows = [{"Tarih": "01.10.2026 Çar", "Görev1": "Ali"}]
        for args in ((changed_schedule, prefs, rows), (schedule, changed_prefs, rows),
                     (schedule, prefs, changed_rows)):
            self.assertNotEqual(base, persistence_fingerprint(*args))

    def test_uygulama_her_degisiklikte_kaydeder_ve_ayarlari_yukler(self):
        source = _app_source()
        self.assertNotIn("elapsed >= 30", source)
        self.assertIn("_persist_fp != st.session_state.get(\"last_saved_fingerprint\")", source)
        self.assertIn("load_app_settings(\n                owner=current_owner, raise_on_error=True,", source)
        self.assertIn("save_app_settings(\n            _current_settings, owner=current_owner,", source)
        # Ayarlar widget'lar çizilmeden önce yüklenmeli.
        self.assertLess(
            source.index("apply_app_settings(st.session_state"),
            source.index("render_team_table(draw=settings_expanded)"),
        )


class TestSwitchSchedulePeriod(unittest.TestCase):
    def _state(self):
        schedule = pd.DataFrame([[True]], index=["Ali"], columns=["1 Çar"])
        return {
            "loaded_period": (2026, 10),
            "schedule_bool": schedule,
            "pref_df": pd.DataFrame([[1]], index=["Ali"], columns=["1 Çar"]),
            "cached_rows_liste": [{"Tarih": "01.10.2026 Çar", "Görev1": "Ali"}],
            "undo_history": [{"schedule": schedule}],
            "isimler_text": "Ali",
        }

    def test_ay_degisince_eski_ay_kaydedilir_ve_state_temizlenir(self):
        state = self._state()
        saved = []

        switched = switch_schedule_period(
            state, (2026, 11), lambda *args: saved.append(args)
        )

        self.assertTrue(switched)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0][0], (2026, 10))
        self.assertTrue(saved[0][1].at["Ali", "1 Çar"])
        for key in ("schedule_bool", "pref_df", "cached_rows_liste", "undo_history", "loaded_period"):
            self.assertNotIn(key, state)
        # Döneme bağlı olmayan ayarlar korunur.
        self.assertEqual(state["isimler_text"], "Ali")

    def test_ayni_donemde_veya_ilk_yuklemede_hicbir_sey_degismez(self):
        state = self._state()
        self.assertFalse(switch_schedule_period(state, (2026, 10), lambda *a: self.fail()))
        self.assertIn("schedule_bool", state)

        first_load = {"schedule_bool": state["schedule_bool"]}
        self.assertFalse(switch_schedule_period(first_load, (2026, 11), lambda *a: self.fail()))
        self.assertIn("schedule_bool", first_load)

    def test_kayit_hatasi_donem_degisimini_engeller(self):
        state = self._state()

        def failing_save(*_):
            raise RuntimeError("database unavailable")

        # Çıkılan ayın kaydı yazılamadıysa ay değişmez ve durum silinmez (veri kaybolmasın).
        self.assertFalse(switch_schedule_period(state, (2027, 1), failing_save))
        self.assertIn("schedule_bool", state)
        self.assertEqual(state["period_switch_blocked"], (2026, 10))

    def test_kayit_false_donerse_donem_degisimi_durur(self):
        state = self._state()
        self.assertFalse(switch_schedule_period(state, (2027, 1), lambda *a: False))
        self.assertIn("schedule_bool", state)
        self.assertEqual(state["period_switch_blocked"], (2026, 10))
        # Kayıt başarılıysa (True) ya da değer dönmüyorsa (None) ay değişir.
        ok = self._state()
        self.assertTrue(switch_schedule_period(ok, (2027, 1), lambda *a: True))
        self.assertNotIn("period_switch_blocked", ok)

    def test_uygulama_donem_degisimini_otomatik_yuklemeden_once_yapar(self):
        source = _app_source()
        switch_pos = source.index("switch_schedule_period(st.session_state, (yil, ay)")
        restore_pos = source.index("if not restore_auto_saved_schedule(")
        self.assertLess(switch_pos, restore_pos)
        self.assertIn("st.session_state.loaded_period = (yil, ay)", source)


class TestDatabaseConnectionDiagnostics(unittest.TestCase):
    def test_adres_yoksa_sebep_aciklanir(self):
        with patch.object(db, "_database_url", return_value=None):
            self.assertFalse(db.init_db())
        self.assertIn("DATABASE_URL tanımlı değil", db.get_last_error())

    def test_baglanti_hatasi_sifresiz_gosterilir(self):
        error = RuntimeError(
            "could not connect to postgresql://neondb_owner:cokgizli@ep-x.neon.tech/neondb\nDETAIL: x"
        )
        with patch.object(db, "_database_url", return_value="postgresql://u:p@h/db"), \
                patch.object(db.psycopg2, "connect", side_effect=error):
            self.assertFalse(db.init_db())
        reason = db.get_last_error()
        self.assertNotIn("cokgizli", reason)
        self.assertIn("neondb_owner:***@ep-x.neon.tech", reason)
        self.assertNotIn("DETAIL", reason)

    def test_adres_her_baglantida_okunur(self):
        cursor = _FakeDbCursor()
        with patch.dict("os.environ", {"DATABASE_URL": "postgresql://yeni"}), \
                patch.object(db.psycopg2, "connect", return_value=_FakeDbConnection(cursor)) as connect:
            self.assertTrue(db.init_db())
        self.assertEqual(connect.call_args[0][0], "postgresql://yeni")
        self.assertIsNone(db.get_last_error())

    def test_basarisiz_baglanti_kalici_onbelleklenmez(self):
        source = _app_source()
        self.assertNotIn("def _init_db_once", source)
        self.assertIn("_db_state = _db_init_state(SCHEMA_VERSION)", source)
        self.assertIn("_db_ready = prepare_schema(\n    _db_state, st.session_state, SCHEMA_VERSION,", source)
        # Başarısızlık kalıcı değil: aynı oturum bir süre sonra yeniden dener.
        shared, session, calls = {"ready": False, "error": None}, {}, []
        fail = lambda: calls.append(1) or False  # noqa: E731
        self.assertFalse(db.prepare_schema(shared, session, "v", fail, lambda: "uyku", now=100))
        self.assertEqual(shared["error"], "uyku")
        self.assertFalse(db.prepare_schema(shared, session, "v", fail, lambda: "uyku", now=110))
        self.assertEqual(len(calls), 1)
        self.assertTrue(db.prepare_schema(shared, session, "v", lambda: True, lambda: None, now=140))


class TestSchedulePersistence(unittest.TestCase):
    def test_load_db_hatasi_istenirse_yukari_tasinir(self):
        with patch.object(
            db.psycopg2,
            "connect",
            side_effect=RuntimeError("database unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "database unavailable"):
                db.load_schedule("Ocak", 2025, 1, raise_on_error=True)

    def test_load_db_hatasi_legacy_cagriyi_bozmaz(self):
        with patch.object(
            db.psycopg2,
            "connect",
            side_effect=RuntimeError("database unavailable"),
        ):
            self.assertEqual(
                db.load_schedule("Ocak", 2025, 1),
                (None, None, None, None, None),
            )

    def test_save_db_hatasi_basarisizlik_dondurur(self):
        schedule = pd.DataFrame([[False]], index=["Ali"], columns=["1 Çar"])

        with patch.object(
            db.psycopg2,
            "connect",
            side_effect=db.psycopg2.OperationalError("database unavailable"),
        ):
            self.assertFalse(
                db.save_schedule("Ocak", 2025, 1, ["Ali"], schedule, expected_version=db.FORCE)
            )

    def test_save_serializes_role_rows_alongside_schedule(self):
        cursor = _FakeDbCursor()
        connection = _FakeDbConnection(cursor)
        schedule = pd.DataFrame(
            [[True, False], [False, True]],
            index=["Ali", "Ayşe"],
            columns=["1 Çar", "2 Per"],
        )
        rows = [
            {"Tarih": "01.01.2025 Çar", "Görev1": "Ali", "Görev2": "Ayşe"},
            {"Tarih": "02.01.2025 Per", "Görev1": "Ayşe", "Görev2": "Ali"},
        ]

        with patch.object(db.psycopg2, "connect", return_value=connection), \
                patch.object(db, "execute_values"):
            self.assertTrue(db.save_schedule(
                "Ocak", 2025, 1, ["Ali", "Ayşe"], schedule, rows_liste=rows, expected_version=db.FORCE
            ))

        insert_params = _writes(cursor)[0][1]
        self.assertEqual(json.loads(insert_params[6]), rows)
        self.assertIn("rows_liste_json", _writes(cursor)[0][0])

    def test_load_restores_role_rows_exactly(self):
        rows = [
            {"Tarih": "01.01.2025 Çar", "Görev1": "Ali", "Görev2": "Ayşe"},
            {"Tarih": "02.01.2025 Per", "Görev1": "Ayşe", "Görev2": "Ali"},
        ]
        cursor = _FakeDbCursor(
            schedule_row={
                "id": 7,
                "team_members": json.dumps(["Ali", "Ayşe"]),
                "rows_liste_json": json.dumps(rows, ensure_ascii=False),
            },
            data_rows=[
                {"person": "Ali", "day_col": "1 Çar", "assigned": True},
                {"person": "Ayşe", "day_col": "2 Per", "assigned": True},
            ],
        )
        connection = _FakeDbConnection(cursor)

        with patch.object(db.psycopg2, "connect", return_value=connection):
            loaded_team, loaded_schedule, _, _, loaded_rows = db.load_schedule("Ocak", 2025, 1)

        self.assertEqual(loaded_team, ["Ali", "Ayşe"])
        self.assertEqual(loaded_rows, rows)
        self.assertTrue(loaded_schedule.at["Ali", "1 Çar"])

    def test_tercihler_kaydedilip_aynen_geri_yuklenir(self):
        prefs = pd.DataFrame(
            [[1, 3], [2, 0]],
            index=["Ali", "Ayşe"],
            columns=["1 Çar", "2 Per"],
        )
        schedule = pd.DataFrame(False, index=prefs.index, columns=prefs.columns)
        save_cursor = _FakeDbCursor()
        with patch.object(db.psycopg2, "connect", return_value=_FakeDbConnection(save_cursor)), \
                patch.object(db, "execute_values"):
            self.assertTrue(db.save_schedule(
                "Ocak", 2025, 1, ["Ali", "Ayşe"], schedule, pref_df=prefs, expected_version=db.FORCE
            ))

        load_cursor = _FakeDbCursor(
            schedule_row={
                "id": 10,
                "team_members": json.dumps(["Ali", "Ayşe"]),
                "pref_json": _writes(save_cursor)[0][1][4],
            },
            data_rows=[{"person": "Ali", "day_col": "1 Çar", "assigned": False}],
        )
        with patch.object(db.psycopg2, "connect", return_value=_FakeDbConnection(load_cursor)):
            _, _, loaded_prefs, _, _ = db.load_schedule("Ocak", 2025, 1)

        self.assertIsNotNone(loaded_prefs, "Kaydedilen tercihler yüklenemedi")
        self.assertEqual(loaded_prefs.loc["Ali", "2 Per"], 3)
        self.assertEqual(loaded_prefs.loc["Ayşe", "1 Çar"], 2)

    def test_load_legacy_schedule_has_no_role_rows(self):
        cursor = _FakeDbCursor(
            schedule_row={
                "id": 8,
                "team_members": json.dumps(["Ali"]),
            },
            data_rows=[
                {"person": "Ali", "day_col": "1 Çar", "assigned": True},
            ],
        )
        connection = _FakeDbConnection(cursor)

        with patch.object(db.psycopg2, "connect", return_value=connection):
            loaded = db.load_schedule("Eski", 2025, 1)

        self.assertEqual(loaded[0], ["Ali"])
        self.assertIsNone(loaded[4])

    def test_schedule_edit_save_and_reload_uses_reconciled_role_rows(self):
        columns = ["1 Çar"]
        day_details = {"1 Çar": {"full_date": "01.01.2025 Çar"}}
        previous_schedule = pd.DataFrame(
            [[True], [True], [False]],
            index=["Ali", "Ayşe", "Mehmet"],
            columns=columns,
        )
        edited_schedule = pd.DataFrame(
            [[False], [True], [True]],
            index=["Ali", "Ayşe", "Mehmet"],
            columns=columns,
        )
        reconciled_rows = synchronize_changed_role_rows(
            [{"Tarih": "01.01.2025 Çar", "Görev1": "Ayşe", "Görev2": "Ali"}],
            previous_schedule,
            edited_schedule,
            ["Görev1", "Görev2"],
            columns,
            day_details,
        )
        expected_rows = [
            {"Tarih": "01.01.2025 Çar", "Görev1": "Ayşe", "Görev2": "Mehmet"},
        ]
        self.assertEqual(reconciled_rows, expected_rows)

        save_cursor = _FakeDbCursor()
        save_connection = _FakeDbConnection(save_cursor)
        with patch.object(db.psycopg2, "connect", return_value=save_connection), \
                patch.object(db, "execute_values"):
            self.assertTrue(db.save_schedule(
                "Düzenlenmiş", 2025, 1, ["Ali", "Ayşe", "Mehmet"],
                edited_schedule, rows_liste=reconciled_rows, expected_version=db.FORCE,
            ))

        load_cursor = _FakeDbCursor(
            schedule_row={
                "id": 9,
                "team_members": json.dumps(["Ali", "Ayşe", "Mehmet"]),
                "rows_liste_json": _writes(save_cursor)[0][1][6],
            },
            data_rows=[
                {"person": "Ayşe", "day_col": "1 Çar", "assigned": True},
                {"person": "Mehmet", "day_col": "1 Çar", "assigned": True},
            ],
        )
        with patch.object(
            db.psycopg2, "connect", return_value=_FakeDbConnection(load_cursor)
        ):
            _, loaded_schedule, _, _, loaded_rows = db.load_schedule("Düzenlenmiş", 2025, 1)

        self.assertEqual(loaded_rows, expected_rows)
        self.assertTrue(loaded_schedule.at["Ayşe", "1 Çar"])
        self.assertTrue(loaded_schedule.at["Mehmet", "1 Çar"])

class _FakeDbConnection:
    def __init__(self, cursor):
        self.fake_cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def cursor(self, **kwargs):
        return self.fake_cursor

    def close(self):
        pass


# ------------------------------------------------------------------------------
# İnceleme raporundaki düzeltmeler (Ekim 2026)
# ------------------------------------------------------------------------------
class TestFillOpenDays(unittest.TestCase):
    """Dengeleme takasından sonra doldurulabilir boş yer kalmamalı."""

    def _setup(self):
        from nobet_core import build_gun_detaylari
        gd = build_gun_detaylari(2027, 4, 30, [])
        cols = list(gd)
        names = ["A", "B", "C"]
        schedule = pd.DataFrame(False, index=names, columns=cols)
        unw = pd.DataFrame(False, index=names, columns=cols)
        return gd, cols, names, schedule, unw

    def test_bos_gun_kurala_uyan_en_az_nobetliye_verilir(self):
        from nobet_core import _fill_open_days
        gd, cols, names, schedule, unw = self._setup()
        # 10. gün boş; A 9'da nöbetçi (dinlenme 1 gün), B izinli, C uygun.
        for col in cols:
            if gd[col]["day_num"] != 10:
                schedule.at[names[gd[col]["day_num"] % 3], col] = True
        target = cols[9]
        schedule.loc[:, target] = False
        schedule.at["A", cols[8]] = True
        schedule.loc[["B", "C"], cols[8]] = False
        unw.at["B", target] = True
        schedule.loc[:, cols[10]] = False
        schedule.at["A", cols[10]] = True
        result = _fill_open_days(schedule, names, cols, gd, unw, None, 1, set(), {}, {}, 1)
        self.assertEqual(result.index[result[target]].tolist(), ["C"])

    def test_kural_disi_aday_yoksa_bos_kalir(self):
        from nobet_core import _fill_open_days
        gd, cols, names, schedule, unw = self._setup()
        target = cols[14]
        unw.at["A", target] = True                      # izinli
        schedule.at["B", cols[13]] = True                # dinlenme
        schedule.at["C", cols[0]] = True                 # max 1 dolu
        result = _fill_open_days(
            schedule.copy(), names, [target, cols[13], cols[0]], gd, unw, None, 1, set(),
            {"C": {"min": 0, "max": 1}}, {}, 1,
        )
        self.assertFalse(result[target].any())

    def test_yasak_cift_ve_kacinma_gunu(self):
        from nobet_core import _fill_open_days
        gd, cols, names, schedule, unw = self._setup()
        target = cols[4]
        schedule.at["A", target] = True
        pref = pd.DataFrame(0, index=names, columns=cols)
        pref.at["C", target] = 2
        # B, A ile çalışamaz; C kaçınıyor ama başka aday yok.
        result = _fill_open_days(
            schedule.copy(), names, cols, gd, unw, pref, 1, {("A", "B")}, {}, {}, 2,
        )
        self.assertEqual(sorted(result.index[result[target]]), ["A", "C"])


class TestReviewFixes(unittest.TestCase):

    def test_buyuk_harfli_turkce_ay_adlari_okunur(self):
        from nobet_core import parse_schedule_import_date
        expected = {"OCAK": 1, "ŞUBAT": 2, "MART": 3, "NİSAN": 4, "MAYIS": 5, "HAZİRAN": 6,
                    "TEMMUZ": 7, "AĞUSTOS": 8, "EYLÜL": 9, "EKİM": 10, "KASIM": 11, "ARALIK": 12,
                    "NISAN": 4, "Haziran": 6, "eki": 10}
        for name, month in expected.items():
            parsed = parse_schedule_import_date(f"5 {name} 2026")
            self.assertEqual((parsed["month"], parsed["year"]), (month, 2026), name)

    def test_gorev_adlari_benzersiz_ve_tarih_olamaz(self):
        from nobet_core import normalize_role_names
        self.assertEqual(normalize_role_names("AYB, AYB", 2)[0], ["AYB", "AYB 2"])
        self.assertEqual(normalize_role_names("Görev2", 2)[0], ["Görev2", "Görev1"])
        names, adjusted = normalize_role_names("Tarih, GYB, gyb", 3)
        self.assertEqual(names, ["Tarih 2", "GYB", "gyb 2"])
        self.assertEqual(len(adjusted), 2)
        self.assertEqual(normalize_role_names("", 2)[0], ["Görev1", "Görev2"])
        self.assertEqual(normalize_role_names("A, B, C", 2)[0], ["A", "B"])

    def test_isim_degisikligi_algilanir(self):
        from nobet_core import detect_renames, rename_people_in_rows
        self.assertEqual(detect_renames(["Ali", "Veli"], ["Ali", "Veli Can"]), {"Veli": "Veli Can"})
        self.assertEqual(detect_renames(["Ali", "Veli"], ["Veli", "Ali"]), {})
        self.assertEqual(detect_renames(["Ali", "Veli"], ["Ali", "Veli", "Can"]), {})
        self.assertEqual(detect_renames([], ["Ali"]), {})
        rows = [{"Tarih": "Veli", "AYB": "Veli", "GYB": "Ali"}]
        self.assertEqual(rename_people_in_rows(rows, {"Veli": "Veli Can"}),
                         [{"Tarih": "Veli", "AYB": "Veli Can", "GYB": "Ali"}])

    def test_ekipte_olmayan_isimler_satirlardan_silinir(self):
        from nobet_core import drop_unknown_people_from_rows
        rows = [{"Tarih": "01.10.2026 Per", "AYB": "Eski", "GYB": "Ali"}]
        self.assertEqual(drop_unknown_people_from_rows(rows, ["Ali"]),
                         [{"Tarih": "01.10.2026 Per", "AYB": "-", "GYB": "Ali"}])

    def test_tireli_isimler_yasak_ciftte_dogru_ayrilir(self):
        team = ["Ayşe-Nur", "Ali", "Can", "Ece"]
        self.assertEqual(parse_forbidden_pairs("Ayşe-Nur-Ali, Can-Ece", team),
                         {("Ali", "Ayşe-Nur"), ("Can", "Ece")})
        self.assertEqual(parse_forbidden_pairs("Ali-Can"), {("Ali", "Can")})

    def _import_args(self):
        isimler = ["Ali", "Ayşe"]
        gun = build_gun_detaylari(2026, 10, 31, [])
        return isimler, list(gun), gun

    def test_excel_tablosu_cizelgeye_cevrilir(self):
        from nobet_core import import_schedule_table
        isimler, sutunlar, gun = self._import_args()
        table = pd.DataFrame({
            "Tarih": ["01.10.2026 Per", "2 EKİM 2026", "x"],
            "AYB": ["Ali", "Ayşe", "Ali"],
            "GYB": ["Zeynep", "-", "-"],
        })
        result = import_schedule_table(table, isimler, sutunlar, gun, ["AYB", "GYB"], 2026, 10)
        self.assertTrue(result["format_ok"])
        self.assertIsNone(result["date_mismatch"])
        self.assertEqual(result["matched"], 2)
        self.assertEqual(result["unreadable_rows"], 1)
        self.assertEqual(result["skipped_names"], ["Zeynep"])
        self.assertTrue(result["schedule"].at["Ali", sutunlar[0]])
        self.assertTrue(result["schedule"].at["Ayşe", sutunlar[1]])
        self.assertEqual(len(result["rows"]), 31)
        self.assertEqual(result["rows"][0]["GYB"], "-")

    def test_excel_baska_ayin_tarihini_reddeder(self):
        from nobet_core import import_schedule_table
        isimler, sutunlar, gun = self._import_args()
        table = pd.DataFrame({"Tarih": ["5 NİSAN 2026"], "AYB": ["Ali"]})
        result = import_schedule_table(table, isimler, sutunlar, gun, ["AYB"], 2026, 10)
        self.assertEqual(result["date_mismatch"]["month"], 4)
        bad = import_schedule_table(pd.DataFrame({"Ad": ["Ali"]}), isimler, sutunlar, gun, ["AYB"], 2026, 10)
        self.assertFalse(bad["format_ok"])

    def test_geri_yuklemede_ekipten_cikan_kisi_listeden_duser(self):
        gun = build_gun_detaylari(2026, 10, 31, [])
        sutunlar = list(gun)
        saved = pd.DataFrame(False, index=["Ali", "Eski"], columns=sutunlar)
        saved.loc[:, sutunlar[0]] = True
        rows = [{"Tarih": gun[c]["full_date"], "AYB": "Ali" if i == 0 else "-",
                 "GYB": "Eski" if i == 0 else "-"} for i, c in enumerate(sutunlar)]
        state = {}
        ok = restore_auto_saved_schedule(
            state, lambda *_: (["Ali", "Eski"], saved, None, None, rows),
            "Otomatik_2026_10", 2026, 10, ["Ali", "Ayşe"], sutunlar, gun, ["AYB", "GYB"],
        )
        self.assertTrue(ok)
        self.assertEqual(state["cached_rows_liste"][0]["GYB"], "-")
        self.assertEqual(state["cached_rows_liste"][0]["AYB"], "Ali")

    def test_rakamdan_olusan_isimlerin_tercihleri_korunur(self):
        prefs = pd.DataFrame([[1, 3], [2, 0]], index=["1", "2"], columns=["1 Çar", "2 Per"])
        schedule = pd.DataFrame(False, index=prefs.index, columns=prefs.columns)
        save_cursor = _FakeDbCursor()
        with patch.object(db.psycopg2, "connect", return_value=_FakeDbConnection(save_cursor)), \
                patch.object(db, "execute_values"):
            self.assertTrue(db.save_schedule("Ocak", 2025, 1, ["1", "2"], schedule, pref_df=prefs, expected_version=db.FORCE))
        load_cursor = _FakeDbCursor(
            schedule_row={"id": 1, "team_members": json.dumps(["1", "2"]),
                          "pref_json": _writes(save_cursor)[0][1][4]},
            data_rows=[{"person": "1", "day_col": "1 Çar", "assigned": False}],
        )
        with patch.object(db.psycopg2, "connect", return_value=_FakeDbConnection(load_cursor)):
            _, _, loaded_prefs, _, _ = db.load_schedule("Ocak", 2025, 1)
        self.assertEqual(loaded_prefs.loc["1", "2 Per"], 3)

    def test_veritabani_sifresi_gunluge_yazilmaz(self):
        import io, contextlib
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            db._log_error("Test", RuntimeError("bad postgresql://u:cokgizli@host/db"))
        self.assertNotIn("cokgizli", out.getvalue())

    def test_yil_araligi_genis(self):
        from nobet_core import apply_app_settings, YEAR_MAX
        state = {}
        apply_app_settings(state, {"year": 2031})
        self.assertEqual(state["yil_val"], 2031)
        self.assertGreaterEqual(YEAR_MAX, 2031)

    def test_uygulama_kaynaginda_duzeltmeler_var(self):
        source = _app_source()
        # Excel her dosya için bir kez işlenir (sonsuz döngü düzeltmesi).
        self.assertIn('excel_import_done', source)
        # CSV Excel'de Türkçe karakterlerle açılır.
        self.assertIn("encode('utf-8-sig')", source)
        # Telefonda tüm düğmeleri küçülten kural kaldırıldı.
        self.assertNotIn("font-size: 9px", source)
        self.assertNotIn("2024, 2030", source)


class TestInceleme20261014(unittest.TestCase):
    """14 Ekim 2026 kod incelemesi bulguları."""

    def setUp(self):
        from nobet_core import build_gun_detaylari as _b
        self.days = _b(2026, 9, 30, [])
        self.cols = list(self.days)

    def _saved(self, team, shifts):
        schedule = pd.DataFrame(False, index=team, columns=self.cols)
        for person, count in shifts.items():
            schedule.loc[person, self.cols[:count]] = True
        return schedule

    def test_ad_degisikligi_eski_aylara_tasinir(self):
        saved = self._saved(["Ali", "Ayşe"], {"Ali": 5})
        rows = [{"Tarih": self.days[c]["full_date"], "AYB": "Ali" if i < 5 else "-", "GYB": "-"}
                for i, c in enumerate(self.cols)]
        settings = {"locked": [["Ali", self.cols[0]]]}
        state = {}
        restore_auto_saved_schedule(
            state, lambda *a: (["Ali", "Ayşe"], saved, None, settings, rows), "x", 2026, 9,
            ["Ali K.", "Ayşe"], self.cols, self.days, ["AYB", "GYB"], aliases={"Ali": "Ali K."},
        )
        self.assertEqual(int(state["schedule_bool"].loc["Ali K."].sum()), 5)
        self.assertEqual(state["cached_rows_liste"][0]["AYB"], "Ali K.")
        self.assertEqual(state["locked_cells"], {("Ali K.", self.cols[0])})
        self.assertEqual(state["hidden_people"], {})

    def test_eski_adi_alan_yeni_kisi_eslenmez(self):
        from nobet_core import resolve_alias
        self.assertEqual(resolve_alias("Ali", {"Ali": "Ali K."}, ["Ali", "Ali K."]), "Ali")
        self.assertEqual(resolve_alias("Can", {"Ali": "Ali K."}, ["Ali K."]), "Can")

    def test_ad_degisikligi_zinciri_ve_geri_donus(self):
        from nobet_core import record_renames, resolve_alias
        aliases = record_renames({}, {"A": "B"})
        aliases = record_renames(aliases, {"B": "C"})
        self.assertEqual(aliases, {"A": "C", "B": "C"})
        self.assertEqual(resolve_alias("A", aliases, ["C"]), "C")
        self.assertEqual(record_renames({"A": "B"}, {"B": "A"}), {"B": "A"})

    def test_satir_satir_siralama_ad_degisikligi_sanilmaz(self):
        # Maymun bulgusu: ekip tablosunda sıra satır satır değiştirilince bir
        # isim bir an listeden düşüp başka satıra yazılıyor; bu ad değişikliği
        # sanılıp nöbetler başka kişiye geçiyordu.
        from nobet_core import detect_renames, record_renames
        rows = ["Ali", "Ayşex", "Can", "Deniz", "Ece", "Fatihx", "Oya1"]
        target = ["Deniz", "Ali", "Oya1", "Ece", "Ayşex", "Can", "Fatihx"]
        prev, seen, aliases = list(rows), set(rows), {"Fatih": "Fatihx", "Ayşe": "Ayşex"}
        for index, name in enumerate(target):
            rows[index] = name
            team = list(dict.fromkeys(rows))
            renames = detect_renames(prev, team, known=seen, aliases=aliases)
            self.assertEqual(renames, {}, (index, team))
            aliases = record_renames(aliases, renames)
            seen.update(team)
            prev = team
        self.assertEqual(aliases, {"Fatih": "Fatihx", "Ayşe": "Ayşex"})

    def test_bilinen_isme_gecis_yalniz_geri_donusse_ad_degisikligi(self):
        from nobet_core import detect_renames
        self.assertEqual(detect_renames(["Ali", "Veli"], ["Ali", "Can"], known={"Ali", "Veli", "Can"}), {})
        self.assertEqual(detect_renames(["Ali", "Veli"], ["Ali", "Veli C."], known={"Ali", "Veli"}),
                         {"Veli": "Veli C."})
        # Yanlış yazılan ad geri düzeltilince eski adına döner.
        self.assertEqual(
            detect_renames(["Ali", "Veli C."], ["Ali", "Veli"], known={"Ali", "Veli", "Veli C."},
                           aliases={"Veli": "Veli C."}),
            {"Veli C.": "Veli"},
        )

    def test_sakli_nobet_eski_takma_adla_ekipteki_kisiye_gecmez(self):
        # "19 Kasım: Can": Fatihx çıkarılıp nöbeti saklandı; eski hatalı takma
        # ad (Fatihx → Can) yüzünden sayfa yenilenince nöbet Can'a geçiyordu.
        saved = self._saved(["Ayşe", "Can"], {"Ayşe": 1})
        settings = {"hidden": {"Fatihx": {"shifts": [19], "prefs": {}}}}
        state = {}
        restore_auto_saved_schedule(
            state, lambda *a: (["Ayşe", "Can"], saved, None, settings, None), "x", 2026, 9,
            ["Ayşe", "Can"], self.cols, self.days, aliases={"Fatihx": "Can"},
        )
        self.assertEqual(int(state["schedule_bool"].loc["Can"].sum()), 0)
        self.assertEqual(state["hidden_people"], {"Fatihx": {"shifts": [19], "prefs": {}}})

    def test_ad_kayitlari_ayarlarda_saklanir(self):
        state = {"name_aliases": {"Ali": "Ali K."}}
        self.assertEqual(collect_app_settings(state)["name_aliases"], {"Ali": "Ali K."})
        target = {}
        apply_app_settings(target, {"name_aliases": {"Ali": "Ali K.", "": "x", "a": 1}})
        self.assertEqual(target["name_aliases"], {"Ali": "Ali K."})

    def test_ekipten_cikan_saklanir_geri_yazilinca_doner(self):
        from nobet_core import stash_removed_people, take_back_people
        schedule = self._saved(["Ali", "Ayşe"], {"Ali": 3, "Ayşe": 2})
        prefs = pd.DataFrame(0, index=["Ali", "Ayşe"], columns=self.cols)
        prefs.loc["Ali", self.cols[10]] = 3
        hidden, stashed = stash_removed_people({}, schedule, prefs, ["Ayşe"])
        self.assertEqual(stashed, ["Ali"])
        self.assertEqual(hidden, {"Ali": {"shifts": [1, 2, 3], "prefs": {"11": 3}}})
        json.dumps(hidden)  # veritabanına yazılabilir olmalı

        new_schedule = pd.DataFrame(False, index=["Ayşe", "Ali"], columns=self.cols)
        new_prefs = pd.DataFrame(0, index=["Ayşe", "Ali"], columns=self.cols)
        hidden, returned = take_back_people(hidden, new_schedule, new_prefs, ["Ayşe", "Ali"])
        self.assertEqual(returned, ["Ali"])
        self.assertEqual(hidden, {})
        self.assertEqual(int(new_schedule.loc["Ali"].sum()), 3)
        self.assertEqual(int(new_prefs.at["Ali", self.cols[10]]), 3)

    def test_kayittaki_ekip_disi_kisi_silinmez(self):
        saved = self._saved(["Ali", "Ayşe"], {"Ali": 4, "Ayşe": 1})
        state = {}
        restore_auto_saved_schedule(
            state, lambda *a: (["Ali", "Ayşe"], saved, None, None, None), "x", 2026, 9,
            ["Ayşe"], self.cols, self.days,
        )
        self.assertEqual(state["hidden_people"]["Ali"]["shifts"], [1, 2, 3, 4])
        # Ali ekipte yokken kaydedilen ay tekrar açılınca (settings'te saklı) geri gelir.
        later = {}
        restore_auto_saved_schedule(
            later,
            lambda *a: (["Ayşe"], saved.loc[["Ayşe"]], None, {"hidden": state["hidden_people"]}, None),
            "x", 2026, 9, ["Ayşe", "Ali"], self.cols, self.days,
        )
        self.assertEqual(int(later["schedule_bool"].loc["Ali"].sum()), 4)
        self.assertEqual(later["hidden_people"], {})

    def test_saklilar_parmak_izine_girer(self):
        schedule = self._saved(["Ayşe"], {})
        self.assertNotEqual(
            persistence_fingerprint(schedule, None, [], None, {}),
            persistence_fingerprint(schedule, None, [], None, {"Ali": {"shifts": [1], "prefs": {}}}),
        )

    def test_fazla_atanan_kisi_ek_sutunda(self):
        from nobet_core import add_overflow_roles
        schedule = self._saved(["Ali", "Ayşe", "Can"], {"Ali": 1, "Ayşe": 1, "Can": 1})
        rows = [{"Tarih": self.days[c]["full_date"], "AYB": "-", "GYB": "-"} for c in self.cols]
        rows[0] = {"Tarih": rows[0]["Tarih"], "AYB": "Ali", "GYB": "Ayşe"}
        out, roles = add_overflow_roles(rows, ["AYB", "GYB"], schedule, self.cols)
        self.assertEqual(roles, ["AYB", "GYB", "Ek 1"])
        self.assertEqual(out[0]["Ek 1"], "Can")
        self.assertEqual(out[1]["Ek 1"], "-")
        self.assertNotIn("Ek 1", rows[0])
        self.assertIn("Ek 1: Can", build_share_text(out, roles, 2026, 9))
        self.assertIn("Ek 1", build_person_ics("Can", out, roles, 2026, 9))

    def test_fazla_kisi_yoksa_liste_aynen(self):
        from nobet_core import add_overflow_roles
        schedule = self._saved(["Ali", "Ayşe"], {"Ali": 1})
        rows = [{"Tarih": "x", "AYB": "Ali" if i == 0 else "-"} for i in range(len(self.cols))]
        out, roles = add_overflow_roles(rows, ["AYB"], schedule, self.cols)
        self.assertIs(out, rows)
        self.assertEqual(roles, ["AYB"])

    def test_turkiye_saati(self):
        from nobet_core import now_tr
        from datetime import timedelta
        self.assertEqual(now_tr().utcoffset(), timedelta(hours=3))

    def test_varsayilan_tatiller_saklanmaz(self):
        from nobet_core import holiday_text_for_period, prune_default_holidays
        self.assertEqual(holiday_text_for_period({}, 2026, 10), "29")
        self.assertEqual(holiday_text_for_period({"2026-10": ""}, 2026, 10), "")
        stored = {"2026-10": "29", "2026-11": "", "2026-12": "31", "2026-03": "20-22, 25"}
        prune_default_holidays(stored)
        self.assertEqual(stored, {"2026-12": "31", "2026-03": "20-22, 25"})

    def test_bos_hucreler_atanmamis_sayilir(self):
        from nobet_core import as_bool_frame
        frame = pd.DataFrame({"1 Sal": [True, np.nan], "2 Çar": [np.nan, False]}, index=["Ali", "Ayşe"])
        self.assertEqual(as_bool_frame(frame).values.tolist(), [[True, False], [False, False]])


class TestSecurityHardening(unittest.TestCase):
    def setUp(self):
        self.source = _app_source()

    def test_giris_ayari_yoksa_veritabanli_uygulama_kilitlenir(self):
        from streamlit.testing.v1 import AppTest
        env = {"DATABASE_URL": "postgresql://x@127.0.0.1:1/x", "GIRISSIZ_KULLANIM": ""}
        with patch.dict("os.environ", env):
            at = AppTest.from_file(str(Path(__file__).with_name("nobet.py")), default_timeout=60).run()
        self.assertTrue(any("kilitlendi" in e.value for e in at.error))
        # Kilit sayfası veritabanı hata nedenini (sunucu adı) göstermez.
        self.assertEqual(len(at.warning), 0)
        self.assertEqual(len(at.button), 0)

    def test_veritabani_hata_nedeni_giristen_sonra_gosterilir(self):
        reason = self.source.index('_db_reason = f" Sebep:')
        self.assertLess(self.source.index("_share_token = st.query_params"), reason)
        self.assertLess(self.source.index("current_owner = _current_user_email()"), reason)

    def test_excel_metni_formul_olarak_yazilmaz(self):
        from io import BytesIO
        from openpyxl import load_workbook
        self.assertIn('"strings_to_formulas": False', self.source)
        out = BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter",
                            engine_kwargs={"options": {"strings_to_formulas": False}}) as writer:
            pd.DataFrame({"Kişi": ["=1+1"]}).to_excel(writer, index=False)
        cell = load_workbook(BytesIO(out.getvalue()))["Sheet1"]["A2"]
        self.assertEqual(cell.data_type, "s")
        self.assertEqual(cell.value, "=1+1")

    def test_veritabani_ayari_algilanir(self):
        with patch.dict("os.environ", {"DATABASE_URL": "postgresql://a"}):
            self.assertTrue(db.is_database_configured())
        with patch.dict("os.environ", {"DATABASE_URL": ""}), \
                patch("streamlit.secrets", {}):
            self.assertFalse(db.is_database_configured())


class TestFullAppRender(unittest.TestCase):
    """Uygulama veritabanı taklidiyle baştan sona çizilir; her bölüm dosyası çalışır."""

    def test_tum_bolumler_hatasiz_cizilir(self):
        from streamlit.testing.v1 import AppTest
        fakes = dict(
            init_db=lambda: True, get_last_error=lambda: None, is_database_configured=lambda: False,
            load_app_settings=lambda **k: None, save_app_settings=lambda *a, **k: True,
            load_schedule=lambda *a, **k: None, save_schedule=lambda *a, **k: True,
            load_auto_saved_assignments=lambda *a, **k: ({}, {}), list_backup_snapshots=lambda **k: [],
            get_schedule_version=lambda *a, **k: None, get_settings_version=lambda *a, **k: None,
            get_share_link=lambda *a: None, ping_db=lambda: True,
        )
        with patch.dict("os.environ", {"DATABASE_URL": ""}), patch.multiple(db, **fakes):
            at = AppTest.from_file(str(Path(__file__).with_name("nobet.py")), default_timeout=120)
            at.session_state["isimler_text"] = "Ali, Ayşe, Can, Deniz, Ece"
            # Tarayıcı deposu bileşeni testte yanıt vermez; boş depo verilir.
            at.session_state["storage_init"] = {}
            at.session_state["yil_val"], at.session_state["ay_val"] = 2026, 10
            at.run()
            self.assertEqual(
                [t.label for t in at.tabs],
                ["🗂️ Tablo Görünümü", "📅 Takvim Görünümü"],
            )
            self.assertTrue(any(e.label.startswith("💾 Yedek") for e in at.expander))
            steps = [
                lambda: next(b for b in at.button if b.label == "⚡ Simülasyon").click(),
                lambda: at.session_state.__setitem__("cal_person_filter", ["Ali"]),
                lambda: next(b for b in at.button if b.label.startswith("🤖")).click(),
                lambda: next(b for b in at.button if b.label == "↩️ Geri").click(),
                lambda: at.radio(key="mode_radio").set_value("✏️ Nöbet Ata/Kaldır"),
                lambda: next(b for b in at.button if b.key and b.key.startswith("g_Ali_")).click(),
                lambda: next(b for b in at.button if b.key == "cal_swap_3").click(),
                lambda: next(b for b in at.button if b.label == "🔄 Sıfırla").click(),
            ]
            for step in steps:
                step()
                at.run()
                self.assertEqual([e.value for e in at.exception], [])
        # Taklit "veritabanı hazır" sonucu sonraki testlere kalmasın.
        import streamlit
        streamlit.cache_resource.clear()


class TestBekleyenTercihler(unittest.TestCase):
    """Kaydedilmemiş tercihler tarayıcıda tutulup sayfa açılınca geri gelir."""

    def _prefs(self, people=("Ali", "Ayşe"), columns=("1 Paz", "2 Pzt")):
        return pd.DataFrame(0, index=list(people), columns=list(columns))

    def test_ayni_surumde_geri_gelir(self):
        from nobet_core import apply_pending_prefs, pending_prefs_payload
        pending = self._prefs()
        pending.at["Ali", "2 Pzt"] = 3
        raw = pending_prefs_payload("", 2026, 11, "v1", pending)
        restored = apply_pending_prefs(raw, "", 2026, 11, "v1", self._prefs())
        self.assertEqual(int(restored.at["Ali", "2 Pzt"]), 3)
        self.assertEqual(int(restored.values.sum()), 3)

    def test_baska_yerde_kaydedildiyse_ya_da_baska_ay_ise_uygulanmaz(self):
        from nobet_core import apply_pending_prefs, is_pending_prefs_for, pending_prefs_payload
        pending = self._prefs()
        pending.at["Ali", "1 Paz"] = 1
        raw = pending_prefs_payload("a@b.c", 2026, 11, "v1", pending)
        self.assertIsNone(apply_pending_prefs(raw, "a@b.c", 2026, 11, "v2", self._prefs()))
        self.assertIsNone(apply_pending_prefs(raw, "a@b.c", 2026, 12, "v1", self._prefs()))
        self.assertIsNone(apply_pending_prefs(raw, "x@y.z", 2026, 11, "v1", self._prefs()))
        self.assertIsNone(apply_pending_prefs("bozuk", "a@b.c", 2026, 11, "v1", self._prefs()))
        self.assertTrue(is_pending_prefs_for(raw, "a@b.c", 2026, 11))
        self.assertFalse(is_pending_prefs_for(raw, "a@b.c", 2026, 12))

    def test_ekipten_cikan_ve_degismeyen_atlanir(self):
        from nobet_core import apply_pending_prefs, pending_prefs_payload
        pending = self._prefs(people=("Ali", "Can"))
        pending.at["Can", "1 Paz"] = 4
        raw = pending_prefs_payload("", 2026, 11, "v1", pending)
        # Can artık ekipte yok, Ali'de değişiklik yok: uygulanacak bir şey kalmaz.
        self.assertIsNone(apply_pending_prefs(raw, "", 2026, 11, "v1", self._prefs()))


class TestIzinVeMusaitDegil(unittest.TestCase):
    def test_izin_hucresi_i_harfiyle_gosterilir(self):
        from nobet_core import get_preference_grid_state
        self.assertEqual(get_preference_grid_state(False, 3)["symbol"], "İ")
        self.assertEqual(get_preference_grid_state(False, 4)["symbol"], "M")

    """İzin (3) mesaiden düşer; Müsait Değil (4) düşmez. İkisine de nöbet yazılmaz."""

    def setUp(self):
        # Ekim 2026: 28 Ekim arife (yarım gün), 29 Ekim tatil.
        self.days = build_gun_detaylari(2026, 10, 31, [29])
        self.cols = list(self.days)
        self.half = official_half_days(2026, 10)
        self.full, _ = required_work_hours(self.days, self.half)
        self.prefs = pd.DataFrame(0, index=["Ali", "Ayşe", "Can"], columns=self.cols)

    def _col(self, day):
        return next(c for c in self.cols if self.days[c]["day_num"] == day)

    def test_izin_is_gunu_mesaiden_duser_musait_degil_dusmez(self):
        self.prefs.at["Ali", self._col(1)] = 3   # Perşembe: 8 saat düşer
        self.prefs.at["Ali", self._col(28)] = 3  # arife: 4 saat düşer
        self.prefs.at["Ali", self._col(3)] = 3   # Cumartesi: düşmez
        self.prefs.at["Ali", self._col(29)] = 3  # tatil: düşmez
        self.prefs.at["Ayşe", self._col(1)] = 4
        self.prefs.at["Ayşe", self._col(2)] = 4
        hours = personal_work_hours(self.days, self.half, self.prefs, ["Ali", "Ayşe", "Can"])
        self.assertEqual(hours["Ali"], self.full - 12)
        self.assertEqual(hours["Ayşe"], self.full)
        self.assertEqual(hours["Can"], self.full)

    def test_ucret_ozeti_kisisel_mesaiyi_kullanir(self):
        schedule = pd.DataFrame(False, index=["Ali", "Ayşe"], columns=self.cols)
        schedule.loc["Ali", self.cols[0:20:2]] = True
        schedule.loc["Ayşe", self.cols[1:21:2]] = True
        self.prefs.at["Ali", self._col(1)] = 3
        mesai = personal_work_hours(self.days, self.half, self.prefs, ["Ali", "Ayşe"])
        _, finance, _ = build_schedule_analysis(
            schedule, [], [], ["Ali", "Ayşe"], self.days, {}, self.full, 10, mesai
        )
        finance = pd.DataFrame(finance).set_index("İsim") if not isinstance(finance, pd.DataFrame) else finance
        self.assertEqual(int(finance.loc["Ali", "Zorunlu mesai"]), self.full - 8)
        self.assertEqual(int(finance.loc["Ayşe", "Zorunlu mesai"]), self.full)
        self.assertEqual(int(finance.loc["Ali", "Fazla mesai"]) - int(finance.loc["Ayşe", "Fazla mesai"]), 8)

    def test_iki_kod_da_nobet_yazilmayan_gun_sayilir(self):
        self.assertTrue(is_blocked_preference(3))
        self.assertTrue(is_blocked_preference(4))
        self.assertFalse(is_blocked_preference(2))
        self.assertFalse(is_blocked_preference(None))
        counts = unavailable_day_counts('{"1 Per": {"Ali": 3, "Ayşe": 4}, "2 Cum": {"Ali": 4, "Can": 2}}')
        self.assertEqual(counts, {"Ali": 2, "Ayşe": 1})

    def test_algoritma_musait_degil_gune_nobet_yazmaz(self):
        team = ["Ali", "Ayşe", "Can", "Deniz"]
        prefs = pd.DataFrame(0, index=team, columns=self.cols)
        prefs.loc["Ali", self.cols[:10]] = 4
        prefs.loc["Ayşe", self.cols[10:20]] = 3
        unwanted = prefs.apply(lambda col: col.map(is_blocked_preference))
        schedule = run_scheduling_core(team, self.cols, unwanted, self.days, 1, 1)[0]
        self.assertFalse(schedule.loc["Ali", self.cols[:10]].any())
        self.assertFalse(schedule.loc["Ayşe", self.cols[10:20]].any())

    def test_yedek_ve_saklanan_kisi_4_kodunu_kabul_eder(self):
        from nobet_core import _valid_backup_preferences, _clean_hidden
        self.assertTrue(_valid_backup_preferences({"1 Per": {"Ali": 4}}))
        self.assertFalse(_valid_backup_preferences({"1 Per": {"Ali": 5}}))
        self.assertEqual(
            _clean_hidden({"Ali": {"shifts": [], "prefs": {"3": 4}}}),
            {"Ali": {"shifts": [], "prefs": {"3": 4}}},
        )


class TestHizliGiris(unittest.TestCase):
    def test_gun_araligi_tek_seferde_yazilir(self):
        from nobet_tablo import apply_quick_entry
        gun = build_gun_detaylari(2026, 11, 30, [])
        pref = pd.DataFrame(0, index=["Ali", "Ayşe"], columns=list(gun))
        days, invalid = parse_holiday_days("3-9, 15", 30)
        self.assertEqual(invalid, [])
        self.assertEqual(apply_quick_entry(pref, "Ali", days, 3, gun), 8)
        ali = {gun[c]["day_num"] for c in pref.columns if pref.at["Ali", c] == 3}
        self.assertEqual(ali, {3, 4, 5, 6, 7, 8, 9, 15})
        self.assertFalse(pref.loc["Ayşe"].any())
        # Aynı günlere tekrar yazmak değişiklik saymaz; Nötr siler.
        self.assertEqual(apply_quick_entry(pref, "Ali", days, 3, gun), 0)
        self.assertEqual(apply_quick_entry(pref, "Ali", {15}, 0, gun), 1)


class TestWhatsappBaglantisi(unittest.TestCase):
    def test_metin_kodlanir(self):
        from nobet_disa_aktar import whatsapp_url
        url = whatsapp_url("*1 Kasım Paz* — AYB: Ayşe\n2 Kasım & Can")
        self.assertTrue(url.startswith("https://wa.me/?text="))
        self.assertNotIn("\n", url)
        self.assertNotIn("&", url.split("text=", 1)[1])
        from urllib.parse import unquote
        self.assertEqual(unquote(url.split("text=", 1)[1]), "*1 Kasım Paz* — AYB: Ayşe\n2 Kasım & Can")


class TestGorevYeriEsitleme(unittest.TestCase):
    """Ayın bir kısmında nöbeti olmayan kişi sonradan gelince tek göreve yığılmamalı."""
    ROLES = ["AYB", "GYB"]
    NAMES = ["GÖRKEM", "FARUK", "EYLÜL", "EGE", "AYÇA", "ORCAN", "LEYLA", "AYBÜKE", "SENA"]

    def _month(self, seed):
        rng = random.Random(seed)
        days = []
        for day in range(30):
            pool = [n for n in self.NAMES if not (n == "AYBÜKE" and day < 15)]
            days.append(rng.sample(pool, 2))
        return days

    def test_gec_gelen_kisi_iki_goreve_de_dagilir(self):
        from nobet_core import distribute_roles
        for seed in range(30):
            days = self._month(seed)
            rows, _ = distribute_roles(list(range(30)), days, self.ROLES, rng=random.Random(seed))
            for name in self.NAMES:
                ayb = sum(row["AYB"] == name for row in rows)
                gyb = sum(row["GYB"] == name for row in rows)
                self.assertLessEqual(abs(ayb - gyb), 1, (seed, name, ayb, gyb))

    def test_nobetciler_degismez_ve_tek_kisi_ilk_goreve(self):
        from nobet_core import distribute_roles
        days = [["Ali", "Can"], ["Can"], [], ["Ali", "Can", "Ece"]]
        rows, counts = distribute_roles([1, 2, 3, 4], days, self.ROLES, rng=random.Random(3))
        self.assertEqual(sorted([rows[0]["AYB"], rows[0]["GYB"]]), ["Ali", "Can"])
        self.assertEqual((rows[1]["AYB"], rows[1]["GYB"]), ("Can", "-"))
        self.assertEqual((rows[2]["AYB"], rows[2]["GYB"]), ("-", "-"))
        self.assertEqual(sum(counts.values()), 3)


class TestSabitGorevYeri(unittest.TestCase):
    ROLES = ["AYB", "GYB"]

    def test_dagitimda_sabit_kisi_kendi_gorevinde(self):
        from nobet_core import distribute_roles
        days = [["Ali", "Can"], ["Can", "Ali"], ["Ali", "Ece"], ["Ece", "Can"]] * 5
        rows, _counts = distribute_roles(
            list(range(len(days))), days, self.ROLES, {"Ali": "GYB"}, rng=random.Random(1),
        )
        for row, people in zip(rows, days):
            self.assertEqual(sorted(row[r] for r in self.ROLES), sorted(people))
            if "Ali" in people:
                self.assertEqual(row["GYB"], "Ali")

    def test_ayni_goreve_iki_sabit_kisi_kaybolmaz(self):
        from nobet_core import distribute_roles, apply_fixed_roles
        fixed = {"Ali": "AYB", "Can": "AYB"}
        rows, _ = distribute_roles([1], [["Ali", "Can"]], self.ROLES, fixed, rng=random.Random(0))
        self.assertEqual(sorted(rows[0][r] for r in self.ROLES), ["Ali", "Can"])
        row = apply_fixed_roles({"Tarih": 1, "AYB": "Ali", "GYB": "Can"}, self.ROLES, fixed)
        self.assertEqual(row, {"Tarih": 1, "AYB": "Ali", "GYB": "Can"})

    def test_satirda_yer_degistirir(self):
        from nobet_core import apply_fixed_roles
        row = apply_fixed_roles({"Tarih": 1, "AYB": "Ali", "GYB": "Can"}, self.ROLES, {"Ali": "GYB"})
        self.assertEqual(row, {"Tarih": 1, "AYB": "Can", "GYB": "Ali"})
        row = apply_fixed_roles({"Tarih": 1, "AYB": "Ali", "GYB": "-"}, self.ROLES, {"Ali": "GYB"})
        self.assertEqual(row, {"Tarih": 1, "AYB": "-", "GYB": "Ali"})
        # Üç görevde zincirleme: taşınan kişi de kendi görevine gider.
        roles = ["A", "B", "C"]
        row = apply_fixed_roles({"A": "Ali", "B": "Can", "C": "Ece"}, roles, {"Ali": "C", "Ece": "B"})
        self.assertEqual(row["C"], "Ali")
        self.assertEqual(row["B"], "Ece")

    def test_degisen_gun_esitlenirken_sabit_gorev_korunur(self):
        from nobet_core import synchronize_changed_role_rows
        cols = ["d1", "d2"]
        details = {c: {"full_date": c} for c in cols}
        before = pd.DataFrame(False, index=["Ali", "Can", "Ece"], columns=cols)
        before.loc[["Can", "Ece"], "d1"] = True
        after = before.copy()
        after.loc["Ece", "d1"] = False
        after.loc["Ali", "d1"] = True
        rows = [{"Tarih": "d1", "AYB": "Can", "GYB": "Ece"}, {"Tarih": "d2", "AYB": "-", "GYB": "-"}]
        out = synchronize_changed_role_rows(rows, before, after, self.ROLES, cols, details, {"Ali": "AYB"})
        self.assertEqual(out[0], {"Tarih": "d1", "AYB": "Ali", "GYB": "Can"})

    def test_cakisma_ve_gecerli_kayitlar(self):
        from nobet_core import active_fixed_roles, fixed_role_conflicts, apply_app_settings
        schedule = pd.DataFrame({"d1": [True, True, False], "d2": [True, False, True]},
                                index=["Ali", "Can", "Ece"])
        fixed = {"Ali": "AYB", "Can": "AYB", "Ece": "GYB"}
        self.assertEqual(fixed_role_conflicts(schedule, ["d1", "d2"], fixed), [("d1", "AYB", ["Ali", "Can"])])
        self.assertEqual(active_fixed_roles({"Ali": "AYB", "Eski": "AYB", "Can": "Yok"}, ["Ali", "Can"], self.ROLES),
                         {"Ali": "AYB"})
        self.assertEqual(active_fixed_roles({"Ali": "AYB"}, ["Ali"], ["AYB"]), {})
        state = {}
        apply_app_settings(state, {"fixed_roles": {"Ali": "AYB", "": "GYB", "Can": 3}})
        self.assertEqual(state["fixed_roles"], {"Ali": "AYB"})


class TestYazdirmaDuzeni(unittest.TestCase):
    def test_yazdirma_css_yalniz_tablo_ve_takvimi_birakir(self):
        from nobet_stil import GLOBAL_CSS

        print_css = GLOBAL_CSS[GLOBAL_CSS.index("@media print"):]
        self.assertIn("size: A4 landscape", print_css)
        self.assertIn(".st-key-schedule-grid", print_css)
        self.assertIn(".st-key-cal-grid", print_css)
        self.assertIn(".st-key-bottom-actions", print_css)
        # Nöbet hücreleri düğme; yazdırmada gizlenmemeli.
        self.assertNotIn(".st-key-schedule-grid button", print_css)
        self.assertIn(".st-key-preference-grid", print_css)
        # Ayarlar paneli ve sekmelerin altındaki öğeler boş kutu / boş 2. sayfa bırakmasın.
        self.assertIn(".st-key-ayarlar-panel", print_css)
        self.assertIn('[data-testid="stTabs"] ~ *', print_css)


class TestGoogleGirisPaketleri(unittest.TestCase):
    def test_google_giris_paketleri_yuklu(self):
        # Streamlit 1.66 girişi Authlib'in Starlette istemcisiyle yapar; o da
        # httpx ister. Eksikse "Google ile giriş yap" sunucu hatası verir.
        from authlib.integrations import starlette_client  # noqa: F401
