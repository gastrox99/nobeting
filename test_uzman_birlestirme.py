# -*- coding: utf-8 -*-
"""Uzman Excel'i ile ortak liste ve uzman idare şablonu."""
from io import BytesIO

import openpyxl
import pandas as pd

from nobet_disa_aktar import (
    fill_idare_sablonu, ortak_liste_excel, ortak_liste_html, ortak_liste_metni, ortak_liste_png, read_uzman_excel,
)
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


from nobet_core import apply_app_settings, collect_app_settings
from nobet_disa_aktar import kisa_adlar, personel_oku, personel_yaz

_PERSONEL = [
    ("Murat", "Özgümüş", "Uzman"), ("Murat", "Alpaslan", "Uzman"), ("Doğuş", "Uyanık", "Uzman"),
    ("Serpil Canan", "Erbuyun", "Uzman"), ("Sinem", "Aksoy", "Uzman"), ("Ali", "Alagöz", "Pratisyen"),
]


def test_kisa_adlar_ilk_ad_ve_ayni_adda_soyad_bas_harfi():
    sonuc = kisa_adlar(_PERSONEL, ["MURAT OZGUMUS", "MURAT ALPASLAN", "DOGUS UYANIK",
                                   "SERPIL CANAN ERBUYUN", "SINEM A", "ALI ALAGOZ", "BILINMEYEN KISI"])
    assert sonuc == {
        "MURAT OZGUMUS": "MURAT Ö.", "MURAT ALPASLAN": "MURAT A.", "DOGUS UYANIK": "DOĞUŞ",
        "SERPIL CANAN ERBUYUN": "SERPİL", "SINEM A": "SİNEM", "ALI ALAGOZ": "ALİ",
        "BILINMEYEN KISI": "BILINMEYEN KISI",
    }


def test_belirsiz_ad_eslesmez():
    kisiler = [("Murat", "Özgümüş", ""), ("Murat", "Özgür", "")]
    assert kisa_adlar(kisiler, ["MURAT O"]) == {"MURAT O": "MURAT O"}


def test_personel_metni_gidip_gelir_ve_noktalı_virgul_temizlenir():
    metin = personel_yaz(_PERSONEL + [("Boş", "", ""), ("A;B", "C", "Uzman")])
    assert personel_oku(metin) == _PERSONEL + [("A B", "C", "Uzman")]


def test_personel_listesi_ayarlarla_kaydedilir():
    durum = {"personel_text": "Ali;Veli;Uzman"}
    ayarlar = collect_app_settings(durum)
    assert ayarlar["personnel"] == "Ali;Veli;Uzman"
    yeni = {}
    apply_app_settings(yeni, ayarlar)
    assert yeni["personel_text"] == "Ali;Veli;Uzman"


def test_kisa_adlar_yazim_farki_ve_eksik_sozcuk():
    kisiler = [("İsmahangül", "Özbayrak", ""), ("Murat", "Alparslan", ""), ("Murat", "Özgümüş", ""),
               ("Arda", "Balcan", ""), ("İrem", "Gür Türkan", ""), ("Sinem", "Akkaya", "")]
    sonuc = kisa_adlar(kisiler, ["ISMIHAN OZBAYRAK", "MURAT ALPASLAN", "MEHMET ARDA BALCAN", "IREM GUR"])
    assert sonuc == {"ISMIHAN OZBAYRAK": "İSMAHANGÜL", "MURAT ALPASLAN": "MURAT A.",
                     "MEHMET ARDA BALCAN": "ARDA", "IREM GUR": "İREM"}


def _ortak_girdi():
    detay = build_gun_detaylari(2026, 8, 31, [])
    rows = [{"AYB": f"K{g}", "GYB": f"L{g}"} for g in range(1, 32)]
    return rows, ["AYB", "GYB"], {1: ("A <b>", "B")}, detay


def test_ortak_liste_png_uretilir():
    assert ortak_liste_png(*_ortak_girdi()).startswith(b"\x89PNG")


def test_ortak_liste_whatsapp_metni_hafta_sonu_kalin():
    rows, roller, uzman, detay = _ortak_girdi()
    satirlar = ortak_liste_metni(rows, roller, uzman, detay, 2026, 8).splitlines()
    assert satirlar[0] == "📅 Ağustos 2026 Ortak Nöbet Listesi"
    assert satirlar[2] == "*01.08.2026 Cmt* — Uzman: A <b> / B · AYB: K1 · GYB: L1"
    assert satirlar[4] == "03.08.2026 Pzt — AYB: K3 · GYB: L3"


def test_ortak_liste_yazdirma_sayfasi_adlari_kacirir():
    html_ = ortak_liste_html(*_ortak_girdi())
    assert "A &lt;b&gt;" in html_ and "window.print()" in html_
    assert html_.count('<tr class="g">') == 10   # Ağustos 2026: 5 Cmt + 5 Paz

def test_uzman_excel_buyuk_harfli_tarih_basligi_okunur():
    gunler, _ = read_uzman_excel(_uzman_excel(["01.08.2026"], ("TARİH", "N1", "N2")), 2026, 8)
    assert gunler == {1: ("A1", "B1")}


def test_uzman_excel_ustte_baslik_satiri_olan_dosya_okunur():
    cikti = BytesIO()
    pd.DataFrame(
        [["UZMAN NÖBET ÇİZELGESİ", None, None], ["TARİH", "AD", "AD"], ["01.08.2026", "A1", "B1"]]
    ).to_excel(cikti, index=False, header=False)
    cikti.seek(0)
    gunler, _ = read_uzman_excel(cikti, 2026, 8)
    assert gunler == {1: ("A1", "B1")}


def test_uzman_excel_iki_ay_alt_alta_ise_ikinci_ay_da_okunur():
    tarihler = [f"{g:02d}.11.2026" for g in range(1, 31)] + [f"{g:02d}.12.2026" for g in range(1, 32)]
    gunler, _ = read_uzman_excel(_uzman_excel(tarihler), 2026, 12)
    assert len(gunler) == 31


def test_idare_sablonu_gun_adlari_turkce_harfli():
    ws = openpyxl.load_workbook(BytesIO(fill_idare_sablonu([], ["A", "B"], 2026, 8))).active
    assert (ws["B5"].value, ws["B6"].value) == ("CUMARTESİ", "PAZAR")  # 1 Ağustos 2026 Cumartesi
    assert ws["B7"].value == "PAZARTESİ"
