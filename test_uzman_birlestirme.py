# -*- coding: utf-8 -*-
"""Uzman Excel'i ile ortak liste ve uzman idare şablonu."""
from io import BytesIO

import openpyxl
import pandas as pd

from nobet_disa_aktar import fill_idare_sablonu, ortak_liste_excel, read_uzman_excel
from nobet_gunler import build_gun_detaylari


def _uzman_excel(tarihler, baslik=("Tarih", "Nöbetçi 1", "Nöbetçi 2")):
    cikti = BytesIO()
    pd.DataFrame(
        [(t, f"A{i}", f"B{i}") for i, t in enumerate(tarihler, 1)], columns=list(baslik)
    ).to_excel(cikti, index=False)
    cikti.seek(0)
    return cikti


def test_uzman_excel_secili_ayin_gunlerini_okur_baska_ayi_atlar():
    dosya = _uzman_excel(["01.08.2026", "02.08.2026", "01.09.2026", "kötü"])
    gunler, atlanan = read_uzman_excel(dosya, 2026, 8)
    assert gunler == {1: ("A1", "B1"), 2: ("A2", "B2")}
    assert atlanan == 2


def test_uzman_excel_tarih_sutunu_yoksa_none():
    assert read_uzman_excel(_uzman_excel(["01.08.2026"], ("Gün", "N1", "N2")), 2026, 8) is None


def test_ortak_liste_uzman_ve_kendi_adlari_yan_yana_hafta_sonu_gri():
    detay = build_gun_detaylari(2026, 8, 31, [])
    rows = [{"AYB": f"K{g}", "GYB": f"L{g}"} for g in range(1, 32)]
    ws = openpyxl.load_workbook(BytesIO(
        ortak_liste_excel(rows, ["AYB", "GYB"], {1: ("=A", "B")}, 2026, 8, detay)
    )).active
    assert [c.value for c in ws[1][:5]] == ["TARİH", "UZMAN", None, "AYB", "GYB"]
    assert [c.value for c in ws[2][:5]] == ["01.08.2026 Cmt", "=A", "B", "K1", "L1"]
    assert ws["B2"].data_type == "s"
    assert ws["A2"].fill.fgColor.rgb.endswith("BFBFBF")      # 1 Ağustos Cumartesi
    assert ws["A4"].fill.fgColor.rgb != ws["A2"].fill.fgColor.rgb  # 3 Ağustos Pazartesi
    assert ws["B3"].value is None                             # uzman girilmemiş gün


def test_uzman_idare_sablonu_basligi_degisir():
    rows = [{"Uzman 1": "A", "Uzman 2": "B"}]
    ws = openpyxl.load_workbook(BytesIO(
        fill_idare_sablonu(rows, ["Uzman 1", "Uzman 2"], 2026, 8, baslik="UZMAN BAŞLIK")
    )).active
    assert ws["B1"].value == "UZMAN BAŞLIK"
    assert (ws["D5"].value, ws["E5"].value) == ("A", "B")
