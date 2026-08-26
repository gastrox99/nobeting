# -*- coding: utf-8 -*-
"""
test_nobet.py - Nöbet Yönetimi Uygulaması Unit Testleri
Çalıştırmak için: python -m pytest test_nobet.py -v
"""
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
import random
import time
from nobet_core import (
    parse_unwanted_days,
    parse_holiday_days,
    validate_inputs,
    parse_forbidden_pairs,
    parse_person_limits,
    restore_auto_saved_schedule,
    build_gun_detaylari,
    build_schedule_analysis,
    changed_schedule_columns,
    find_person_role,
    normalize_preference_grid,
    run_scheduling_core,
    schedule_fingerprint,
    use_compact_schedule_editor,
    create_print_html,
    build_limit_violation_messages,
    undo_history_state,
    redo_history_state,
)


# ==============================================================================
# 1. parse_unwanted_days testleri
# ==============================================================================
class TestParseUnwantedDays(unittest.TestCase):

    def test_bos_girdi(self):
        self.assertEqual(parse_unwanted_days("", 30), [])

    def test_none_girdi(self):
        self.assertEqual(parse_unwanted_days(None, 30), [])

    def test_tek_gun(self):
        self.assertEqual(parse_unwanted_days("5", 30), [5])

    def test_virgulle_ayrilmis_gunler(self):
        result = parse_unwanted_days("1,5,10", 30)
        self.assertEqual(sorted(result), [1, 5, 10])

    def test_aralik_girdi(self):
        result = parse_unwanted_days("3-7", 30)
        self.assertEqual(sorted(result), [3, 4, 5, 6, 7])

    def test_karisik_tek_ve_aralik(self):
        result = parse_unwanted_days("1,5-7,15", 30)
        self.assertEqual(sorted(result), [1, 5, 6, 7, 15])

    def test_max_day_siniri(self):
        result = parse_unwanted_days("28-35", 30)
        self.assertEqual(sorted(result), [28, 29, 30])

    def test_sinir_disi_gun(self):
        result = parse_unwanted_days("0,31,99", 30)
        self.assertEqual(result, [])

    def test_gecersiz_metin(self):
        result = parse_unwanted_days("abc,xyz", 30)
        self.assertEqual(result, [])

    def test_bosluklu_girdi(self):
        result = parse_unwanted_days("  3 , 5 , 10  ", 30)
        self.assertEqual(sorted(result), [3, 5, 10])


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


# ==============================================================================
# 2. validate_inputs testleri
# ==============================================================================
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

    def test_tekrarlanan_tatil_fizibiliteyi_iki_kez_etkilemez(self):
        is_valid, errors, warnings = validate_inputs(
            **self._base_params(
                isimler=["Ali", "Ayşe", "Mehmet", "Fatma", "Can"],
                gun_sayisi=3,
                tatil_gunleri=[1, 1],
            )
        )

        self.assertTrue(is_valid)
        self.assertEqual(errors, [])
        self.assertTrue(any("4 pozisyon" in warning for warning in warnings))

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


# ==============================================================================
# 3. parse_forbidden_pairs testleri
# ==============================================================================
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


# ==============================================================================
# 4. parse_person_limits testleri
# ==============================================================================
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


# ==============================================================================
# 5. build_gun_detaylari testleri
# ==============================================================================
class TestBuildGunDetaylari(unittest.TestCase):

    def test_ocak_2025_gun_sayisi(self):
        result = build_gun_detaylari(2025, 1, 31, [])
        self.assertEqual(len(result), 31)

    def test_tatil_gunleri_dahil_degil(self):
        result = build_gun_detaylari(2025, 1, 31, [1, 2, 3])
        self.assertEqual(len(result), 28)
        self.assertNotIn("G01", result)

    def test_hafta_sonu_tespiti(self):
        # 2025 Ocak 4 = Cumartesi
        result = build_gun_detaylari(2025, 1, 31, [])
        self.assertTrue(result["G04"]["weekend"])  # Cumartesi

    def test_hafta_ici_tespiti(self):
        # 2025 Ocak 6 = Pazartesi
        result = build_gun_detaylari(2025, 1, 31, [])
        self.assertFalse(result["G06"]["weekend"])  # Pazartesi

    def test_gun_numarasi_dogru(self):
        result = build_gun_detaylari(2025, 1, 31, [])
        self.assertEqual(result["G15"]["day_num"], 15)

    def test_tarih_string_formatli(self):
        result = build_gun_detaylari(2025, 1, 31, [])
        self.assertIn("Oca", result["G01"]["full_date"])

    def test_ayni_hafta_sonu_ayni_week(self):
        # Cmt ve Paz aynı hafta sonu -> aynı 'week' değeri olmalı
        result = build_gun_detaylari(2025, 1, 31, [])
        self.assertEqual(result["G04"]["week"], result["G05"]["week"])  # 4 Cmt, 5 Paz

    def test_ardisik_hafta_sonu_week_farki_bir(self):
        # Ardışık hafta sonları tam olarak 1 fark etmeli
        result = build_gun_detaylari(2025, 1, 31, [])
        self.assertEqual(result["G11"]["week"] - result["G04"]["week"], 1)  # 4 Cmt -> 11 Cmt

    def test_yil_siniri_week_monoton(self):
        # Ocak 2022: 1 Cmt (ISO hafta 52/2021), 8 Cmt (ISO hafta 1/2022)
        # ISO hafta numarası kullanılsaydı 52 -> 1 olur, +1 mantığı kırılırdı.
        result = build_gun_detaylari(2022, 1, 31, [])
        self.assertEqual(result["G08"]["week"] - result["G01"]["week"], 1)


# ==============================================================================
# 6. run_scheduling_core testleri (algoritma)
# ==============================================================================
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


# ==============================================================================
# 7. create_print_html testleri
# ==============================================================================
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
        self.assertIn('label = "T"', self.app_source)
        self.assertIn('"K" if pref_val == 2', self.app_source)
        self.assertIn('"Y" if pref_val == 3', self.app_source)


# ==============================================================================
# Çalıştır
# ==============================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
