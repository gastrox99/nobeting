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
    parse_holiday_days,
    validate_inputs,
    parse_forbidden_pairs,
    parse_person_limits,
    parse_person_limits_with_issues,
    unknown_names,
    apply_role_edits,
    previous_period,
    last_shift_offsets,
    special_shift_counts,
    history_special_adjustment,
    history_adjustment,
    holiday_shift_counts,
    schedule_totals,
    build_share_text,
    build_person_ics,
    restore_auto_saved_schedule,
    build_gun_detaylari,
    build_schedule_analysis,
    changed_schedule_columns,
    find_person_role,
    get_preference_grid_legend,
    get_preference_grid_state,
    normalize_preference_grid,
    run_scheduling_core,
    _balance_fridays,
    _balance_holidays,
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
    create_print_html,
    build_limit_violation_messages,
    parse_schedule_import_date,
    undo_history_state,
    redo_history_state,
    update_persistence_feedback,
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
            [[0, 1], [2, 3]],
            index=["Ali", "Ayşe"],
            columns=["G01", "G02"],
            dtype=object,
        )
        self.assertTrue(normalize_preference_grid(valid).equals(valid.astype(int)))

        for invalid_value in [None, 1.5, 4, -1]:
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
class TestCreatePrintHtml(unittest.TestCase):

    def _sample_dfs(self):
        df_liste = pd.DataFrame({
            "Tarih": ["1 Oca Çar", "2 Oca Per"],
            "Görev1": ["Ali", "Ayşe"],
            "Görev2": ["Mehmet", "Fatma"]
        })
        df_stats = pd.DataFrame({"Toplam": [3, 4]}, index=["Ali", "Ayşe"])
        return df_liste, df_stats

    def test_html_ciktisi_string(self):
        df_liste, df_stats = self._sample_dfs()
        html = create_print_html(df_liste, df_stats, 2025, 1)
        self.assertIsInstance(html, str)

    def test_html_baslik_iceriyor(self):
        df_liste, df_stats = self._sample_dfs()
        html = create_print_html(df_liste, df_stats, 2025, 1)
        self.assertIn("Ocak 2025", html)

    def test_html_isimler_iceriyor(self):
        df_liste, df_stats = self._sample_dfs()
        html = create_print_html(df_liste, df_stats, 2025, 1)
        self.assertIn("Ali", html)
        self.assertIn("Mehmet", html)

    def test_html_tablo_iceriyor(self):
        df_liste, df_stats = self._sample_dfs()
        html = create_print_html(df_liste, df_stats, 2025, 1)
        self.assertIn("<table>", html)
        self.assertIn("</table>", html)

    def test_html_ozel_karakterli_isimleri_escape_eder(self):
        df_liste = pd.DataFrame({
            "<Rol>": ["<script>alert('x')</script>"],
            "Tarih": ["1 Oca Çar"],
        })
        df_stats = pd.DataFrame({"<Toplam>": [1]}, index=["Ali & Ayşe"])

        html = create_print_html(df_liste, df_stats, 2025, 1)

        self.assertIn("&lt;Rol&gt;", html)
        self.assertIn("&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;", html)
        self.assertNotIn("<script>alert('x')</script>", html)

    def test_html_sayisal_hucreleri_yazdirir(self):
        df_liste = pd.DataFrame({"Tarih": ["1 Oca Çar"], "Görev1": [1]})
        df_stats = pd.DataFrame({"Toplam": [2]}, index=["Ali"])

        html = create_print_html(df_liste, df_stats, 2025, 1)

        self.assertIn("<td>1</td>", html)

    def test_tum_aylar_calisir(self):
        df_liste, df_stats = self._sample_dfs()
        ay_isimleri = {1:"Ocak",2:"Şubat",3:"Mart",4:"Nisan",5:"Mayıs",6:"Haziran",
                       7:"Temmuz",8:"Ağustos",9:"Eylül",10:"Ekim",11:"Kasım",12:"Aralık"}
        for ay, isim in ay_isimleri.items():
            html = create_print_html(df_liste, df_stats, 2025, ay)
            self.assertIn(isim, html, f"{ay}. ay için '{isim}' HTML'de bulunamadı")

class TestPreferenceGridAccessibility(unittest.TestCase):
    """Streamlit widget'larını hedefleyen mobil tercih stillerini korur."""

    @classmethod
    def setUpClass(cls):
        cls.app_source = Path(__file__).with_name("nobet.py").read_text(encoding="utf-8")

    def test_tercih_grid_gercek_anahtarli_container_kullanir(self):
        self.assertIn('grid_key = "preference-grid"', self.app_source)
        self.assertIn("grid_container = st.container(", self.app_source)
        self.assertIn("key=grid_key", self.app_source)
        self.assertIn('legend_container = st.container(key="preference-legend"', self.app_source)
        self.assertIn(".st-key-preference-grid", self.app_source)
        self.assertIn(".st-key-preference-legend", self.app_source)

    def test_mobil_hucreler_44px_ve_renk_anlamli_etiketler_tasir(self):
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
        self.assertIn("min-width: 44px !important", mobile_css)
        self.assertIn("min-height: 44px !important", mobile_css)
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
        self.assertEqual(len(legend), 9)

        for legend_state in legend:
            cell_state = get_preference_grid_state(
                legend_state["key"].startswith("assigned_"),
                {
                    "assigned_neutral": 0,
                    "assigned_preferred": 1,
                    "assigned_avoid": 2,
                    "assigned_unavailable": 3,
                    "conflict": 3,
                    "neutral": 0,
                    "preferred": 1,
                    "avoid": 2,
                    "unavailable": 3,
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
        self.assertIn('st.toast("En adil sonuç seçildi."', self.app_source)


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
        source = Path(__file__).with_name("nobet.py").read_text(encoding="utf-8")
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

    def test_gorev_duzenleme_yer_degistirir(self):
        row = {"Tarih": "01.11.2026 Paz", "AYB": "Ali", "Acil Servis": "Ayşe"}
        roles = ["AYB", "Acil Servis"]
        new_row, changed = apply_role_edits(row, {"AYB": "Ayşe", "Acil Servis": "Ayşe"}, roles, ["Ali", "Ayşe"])
        self.assertTrue(changed)
        self.assertEqual((new_row["AYB"], new_row["Acil Servis"]), ("Ayşe", "Ali"))
        # İki hücreyi birden çapraz yazmak da aynı sonucu verir.
        new_row, _ = apply_role_edits(row, {"AYB": "Ayşe", "Acil Servis": "Ali"}, roles, ["Ali", "Ayşe"])
        self.assertEqual((new_row["AYB"], new_row["Acil Servis"]), ("Ayşe", "Ali"))
        # O gün nöbeti olmayan isim yok sayılır.
        same, changed = apply_role_edits(row, {"AYB": "Can", "Acil Servis": "Ayşe"}, roles, ["Ali", "Ayşe"])
        self.assertFalse(changed)
        self.assertEqual(same, row)

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
        source = Path(__file__).with_name("nobet.py").read_text(encoding="utf-8")
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
        # 6 bayram nöbeti / 8 kişi: geçmişi fazla olan ikisi neredeyse hiç almaz.
        self.assertLessEqual(heavy, 1)

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

    def test_nobetler_aya_yayilir(self):
        # 8 kişi, günde 2: beklenen aralık 4 gün. Dinlenme 1 gün; yayma
        # kapalıyken günaşırı (aralık ≤2) nöbetler çok daha sık çıkar.
        import nobet_core
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
        with patch.object(nobet_core, "SPREAD_WEIGHT", 0), patch.object(nobet_core, "SPREAD_SCORE_WEIGHT", 0):
            unspread = short_gaps()
        self.assertLess(spread * 2, unspread)

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

    def test_analizde_bayram_sutunu(self):
        days = build_gun_detaylari(2026, 10, 31, [29])
        cols = list(days)
        schedule = pd.DataFrame(False, index=["Ali", "Ayşe"], columns=cols)
        schedule.at["Ali", "29 Per"] = True
        load, _, _ = build_schedule_analysis(schedule, [], [], ["Ali", "Ayşe"], days, {}, 0, 0)
        self.assertEqual(int(load.at["Ali", "Bayram"]), 1)
        self.assertEqual(int(load.at["Ayşe", "Bayram"]), 0)


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
        self.source = Path(__file__).with_name("nobet.py").read_text(encoding="utf-8")

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
        self.assertIn("<span style='flex-shrink:0;background:{badge_bg}", self.source)

    def test_gunler_alt_satira_kaymaz(self):
        self.assertIn(
            '.st-key-preference-grid [data-testid="stHorizontalBlock"] {{\n    flex-wrap: nowrap !important;',
            self.source,
        )


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
        source = Path(__file__).with_name("nobet.py").read_text(encoding="utf-8")
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
        source = Path(__file__).with_name("nobet.py").read_text(encoding="utf-8")
        self.assertIn(':is([data-testid="stElementContainer"], .element-container):has(.preference-state) {{\n    display: none !important;', source)
        self.assertIn('[data-testid="stHorizontalBlock"] {{\n    align-items: center !important;', source)

    def test_gun_basligi_bolunmez(self):
        source = Path(__file__).with_name("nobet.py").read_text(encoding="utf-8")
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
        self.assertEqual(int(load.at["Ali", "Özel"]), 1)

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
        self.assertEqual(reasons["müsait değil"], ["Ali"])
        self.assertEqual(reasons["dinlenmede"], ["Ayşe (13. gün nöbetli)"])
        self.assertEqual(reasons["nöbet limitinde"], ["Can (en fazla 2)"])
        self.assertEqual(reasons["birlikte çalışamaz"], ["Deniz (Fuat ile)"])
        self.assertEqual(reasons["uygun, elle atanabilir"], ["Ece"])
        text = format_missing_day_reason(list(reasons.items()))
        self.assertIn("müsait değil: Ali", text)

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
        self.source = Path(__file__).with_name("nobet.py").read_text(encoding="utf-8")

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
        self.assertIn('tab_liste, tab_analiz = st.tabs(["📋 Günlük Liste", "📊 Analiz"])', self.source)
        self.assertNotIn('st.header("📅 Nöbet Çizelgesi")', self.source)


class TestStaleModuleRecovery(unittest.TestCase):
    def test_eski_modul_kalirsa_yenilenip_tekrar_denenir(self):
        source = Path(__file__).with_name("nobet.py").read_text(encoding="utf-8")
        recovery = source[source.index("try:\n    from db import ("):source.index("from streamlit_local_storage")]
        self.assertIn("except ImportError:\n    _fresh_modules()", recovery)
        self.assertEqual(recovery.count("from db import ("), 2)
        self.assertEqual(recovery.count("from nobet_core import ("), 2)

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
        source = Path(__file__).with_name("nobet.py").read_text(encoding="utf-8")
        self.assertLess(source.index('st.query_params.get("paylas")'),
                        source.index("AUTH_ENABLED = _auth_configured()"))
        shared = source[source.index("def _render_shared_view"):source.index('_share_token = st.query_params')]
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

    def test_yazdirma_cizelge_ve_ozet_tablosunu_icerir(self):
        df_liste = pd.DataFrame({"Tarih": ["04.01.2025 Cmt"], "Görev1": ["Ali"]})
        df_stats = pd.DataFrame({"Toplam": [1]}, index=["Ali"])
        html = create_print_html(df_liste, df_stats, 2025, 1)
        self.assertIn("Nöbet Yükü Özeti", html)
        self.assertIn("class='weekend'", html)

    def test_uygulama_kopya_fonksiyon_tanimlamaz(self):
        source = Path(__file__).with_name("nobet.py").read_text(encoding="utf-8")
        for name in ("escape_html", "validate_inputs", "create_print_html"):
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
        self.assertIn("müsait değil", joined)
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
            self.assertTrue(db.save_schedule("Ocak", 2025, 1, ["Ali"], schedule, owner="ali@gmail.com"))
        insert_sql, insert_params = cursor.executed[0]
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
        source = Path(__file__).with_name("nobet.py").read_text(encoding="utf-8")
        calls = re.findall(r"(?<!def )\b(save_schedule|load_schedule|save_app_settings|load_app_settings)\((.*?)\)\n",
                           source, flags=re.S)
        self.assertGreaterEqual(len(calls), 6)
        for name, args in calls:
            # Paylaşım sayfası bağlantının sahibine ait listeyi okur (owner=_owner).
            self.assertTrue(
                "owner=current_owner" in args or "owner=_owner" in args,
                f"{name} kullanıcıya bağlı değil",
            )
        shared = source[source.index("def _render_shared_view"):source.index('_share_token = st.query_params')]
        self.assertEqual(shared.count("owner=_owner"), 1)
        self.assertNotIn("owner=_owner", source.replace(shared, ""))
        # Giriş kontrolü ayarlar yüklenmeden önce yapılmalı.
        self.assertLess(source.index("AUTH_ENABLED = _auth_configured()"),
                        source.index("load_app_settings(owner=current_owner"))

    def test_indirme_dosyalari_tiklaninca_uretilir(self):
        source = Path(__file__).with_name("nobet.py").read_text(encoding="utf-8")
        self.assertIn('st.download_button("🖼️ Resim (PNG)", _png_bytes,', source)
        self.assertNotIn("st.session_state[png_key]", source)


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
            self.assertTrue(db.save_app_settings(settings))
        stored_value = save_cursor.executed[0][1][1]

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
        source = Path(__file__).with_name("nobet.py").read_text(encoding="utf-8")
        self.assertNotIn("elapsed >= 30", source)
        self.assertIn("_persist_fp != st.session_state.get(\"last_saved_fingerprint\")", source)
        self.assertIn("load_app_settings(owner=current_owner, raise_on_error=True)", source)
        self.assertIn("save_app_settings(_current_settings, owner=current_owner)", source)
        # Ayarlar widget'lar çizilmeden önce yüklenmeli.
        self.assertLess(
            source.index("apply_app_settings(st.session_state"),
            source.index('st.text_area(\n            "👥 Ekip'),
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

    def test_kayit_hatasi_donem_degisimini_engellemez(self):
        state = self._state()

        def failing_save(*_):
            raise RuntimeError("database unavailable")

        self.assertTrue(switch_schedule_period(state, (2027, 1), failing_save))
        self.assertNotIn("schedule_bool", state)

    def test_uygulama_donem_degisimini_otomatik_yuklemeden_once_yapar(self):
        source = Path(__file__).with_name("nobet.py").read_text(encoding="utf-8")
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
        source = Path(__file__).with_name("nobet.py").read_text(encoding="utf-8")
        self.assertNotIn("def _init_db_once", source)
        self.assertIn('if not _db_state["ready"] and not st.session_state.get("_db_init_attempted")', source)


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
            side_effect=RuntimeError("database unavailable"),
        ):
            self.assertFalse(
                db.save_schedule("Ocak", 2025, 1, ["Ali"], schedule)
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
                "Ocak", 2025, 1, ["Ali", "Ayşe"], schedule, rows_liste=rows
            ))

        insert_params = cursor.executed[0][1]
        self.assertEqual(json.loads(insert_params[6]), rows)
        self.assertIn("rows_liste_json", cursor.executed[0][0])

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
                "Ocak", 2025, 1, ["Ali", "Ayşe"], schedule, pref_df=prefs
            ))

        load_cursor = _FakeDbCursor(
            schedule_row={
                "id": 10,
                "team_members": json.dumps(["Ali", "Ayşe"]),
                "pref_json": save_cursor.executed[0][1][4],
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
                edited_schedule, rows_liste=reconciled_rows,
            ))

        load_cursor = _FakeDbCursor(
            schedule_row={
                "id": 9,
                "team_members": json.dumps(["Ali", "Ayşe", "Mehmet"]),
                "rows_liste_json": save_cursor.executed[0][1][6],
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

    def test_uygulanamayan_gorev_duzenlemesi_bildirilir(self):
        from nobet_core import ignored_role_edits
        row = {"Tarih": "1", "AYB": "Ali", "GYB": "Ayşe"}
        self.assertEqual(ignored_role_edits(row, {"AYB": "Can", "GYB": "Ayşe"}, ["AYB", "GYB"], ["Ali", "Ayşe"]),
                         [("AYB", "Can")])
        self.assertEqual(ignored_role_edits(row, {"AYB": "", "GYB": "Ayşe"}, ["AYB", "GYB"], ["Ali", "Ayşe"]),
                         [("AYB", "")])
        self.assertEqual(ignored_role_edits(row, {"AYB": "Ayşe", "GYB": "Ayşe"}, ["AYB", "GYB"], ["Ali", "Ayşe"]), [])

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
            self.assertTrue(db.save_schedule("Ocak", 2025, 1, ["1", "2"], schedule, pref_df=prefs))
        load_cursor = _FakeDbCursor(
            schedule_row={"id": 1, "team_members": json.dumps(["1", "2"]),
                          "pref_json": save_cursor.executed[0][1][4]},
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
        source = Path(__file__).with_name("nobet.py").read_text(encoding="utf-8")
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
        self.source = Path(__file__).with_name("nobet.py").read_text(encoding="utf-8")

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
