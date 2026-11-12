# -*- coding: utf-8 -*-
"""İdare şablonuna aktarma: ikili çizelge şablonu seçilen ayın listesiyle dolar."""
import datetime
from io import BytesIO

import openpyxl

from nobet_disa_aktar import fill_idare_sablonu, tam_adlar


def _rows(gun, roller):
    return [{"Tarih": f"{g} Tem", roller[0]: f"A{g}", roller[1]: f"B{g}"} for g in range(1, gun + 1)]


def _ac(veri):
    return openpyxl.load_workbook(BytesIO(veri)).active


def test_sablon_ayin_gunleri_ve_adlariyla_dolar():
    roller = ["Yoğun Bakım", "Servis"]
    ws = _ac(fill_idare_sablonu(_rows(31, roller), roller, 2026, 7))
    assert ws["A3"].value == "AY/YIL: TEMMUZ/2026"
    assert ws["A5"].value == datetime.datetime(2026, 7, 1)
    assert ws["B5"].value == "ÇARŞAMBA"
    assert (ws["D5"].value, ws["E5"].value) == ("A1", "B1")
    assert ws["A35"].value == datetime.datetime(2026, 7, 31)
    assert (ws["D35"].value, ws["E35"].value) == ("A31", "B31")
    assert ws["B36"].value == "BİRİM SORUMLUSU"


def test_kisa_ayda_fazla_satirlar_bos_kalir_ve_formul_calismaz():
    roller = ["Yoğun Bakım", "Servis"]
    rows = _rows(28, roller)
    rows[0][roller[0]] = "=1+1"
    rows[1][roller[1]] = "-"
    ws = _ac(fill_idare_sablonu(rows, roller, 2026, 2))
    assert ws["A3"].value == "AY/YIL: ŞUBAT/2026"
    assert ws["D5"].value == "=1+1" and ws["D5"].data_type == "s"
    assert ws["E6"].value is None
    assert ws["A33"].value is None and ws["D34"].value is None


def test_tarih_gun_ay_yil_biciminde():
    roller = ["Yoğun Bakım", "Servis"]
    ws = _ac(fill_idare_sablonu(_rows(31, roller), roller, 2026, 7))
    assert ws["A5"].number_format == "DD.MM.YYYY" and ws["A35"].number_format == "DD.MM.YYYY"


def test_tam_ad_personel_listesinden_gelir():
    kisiler = [("Faruk", "Köser", "Pratisyen", ""), ("Murat", "Özgümüş", "Uzman", ""),
               ("Murat", "Aksoy", "Pratisyen", "")]
    sonuc = tam_adlar(kisiler, ["Faruk", "MURAT OZGUMUS", "Murat", "Bilinmeyen"])
    assert sonuc["Faruk"] == "Faruk Köser"
    assert sonuc["MURAT OZGUMUS"] == "Murat Özgümüş"
    assert sonuc["Murat"] == "Murat"  # iki Murat var: karışmasın
    assert sonuc["Bilinmeyen"] == "Bilinmeyen"


_IDARE_INDIRMELERI = []


def _idare_dugmeli_uygulama():
    import pandas as pd
    import streamlit as st
    import nobet_disa_aktar as m
    import test_idare_sablonu as t
    from nobet_gunler import build_gun_detaylari
    st.session_state.personel_text = "Faruk;Köser;Pratisyen\nLeyla;Dinçsoy;Pratisyen"
    _asil = m.st.download_button

    def _yakala(label, data, *a, **k):
        if label == "🏥 İdare şablonuna aktar":
            t._IDARE_INDIRMELERI.append(data)
        return _asil(label, data, *a, **k)
    m.st.download_button = _yakala
    try:
        bos = pd.DataFrame()
        m.render_toolbar(bos, bos, bos, [{"AYB": "Faruk", "GYB": "Leyla"}] * 30, ["AYB", "GYB"],
                         2026, 11, build_gun_detaylari(2026, 11, 30, []), ["Faruk", "Leyla"],
                         ["AYB", "GYB"], ["AYB", "GYB"], False, "", lambda: None)
    finally:
        m.st.download_button = _asil


def test_idare_dugmesi_tiklaninca_da_soyadi_yazar():
    # Streamlit, indirme düğmesine verilen işlevi tıklamada ayrı bir iş parçacığında çalıştırır;
    # orada st.session_state boş görünür. Personel listesi sayfa çizilirken alınmalı.
    from streamlit.testing.v1 import AppTest
    _IDARE_INDIRMELERI.clear()
    at = AppTest.from_function(_idare_dugmeli_uygulama, default_timeout=60)
    at.run()
    assert [e.value for e in at.exception] == []
    veri = _IDARE_INDIRMELERI[-1]
    ws = _ac(veri() if callable(veri) else veri)  # sayfa çalışması dışında, tıklamadaki gibi
    assert (ws["D5"].value, ws["E5"].value) == ("Faruk Köser", "Leyla Dinçsoy")
