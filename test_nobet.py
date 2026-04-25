# -*- coding: utf-8 -*-
"""
test_nobet.py - Nöbet Yönetimi Uygulaması Unit Testleri
Çalıştırmak için: python -m pytest test_nobet.py -v
"""
import unittest
import pandas as pd
import numpy as np
import random
from nobet_core import (
    parse_unwanted_days,
    validate_inputs,
    parse_forbidden_pairs,
    parse_person_limits,
    build_gun_detaylari,
    run_scheduling_core,
    create_print_html,
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

    def test_tum_aylar_calisir(self):
        df_liste, df_stats = self._sample_dfs()
        ay_isimleri = {1:"Ocak",2:"Şubat",3:"Mart",4:"Nisan",5:"Mayıs",6:"Haziran",
                       7:"Temmuz",8:"Ağustos",9:"Eylül",10:"Ekim",11:"Kasım",12:"Aralık"}
        for ay, isim in ay_isimleri.items():
            html = create_print_html(df_liste, df_stats, 2025, ay)
            self.assertIn(isim, html, f"{ay}. ay için '{isim}' HTML'de bulunamadı")


# ==============================================================================
# Çalıştır
# ==============================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
