# -*- coding: utf-8 -*-
"""Açılınca çok büyüyen Excel dosyası (16 Kasım 2026 incelemesi).

Excel dosyası sıkıştırılmış bir pakettir. Uygulama yalnız sıkıştırılmış boyuta
(2 MB) bakıyordu: 400 KB'lık özel hazırlanmış bir dosya okunurken 2 GB bellek
tutup siteyi herkes için düşürebiliyordu.
"""
import zipfile
from io import BytesIO

import pandas as pd

import nobet_disa_aktar


class _Durum(dict):
    """st.session_state gibi: hem durum["x"] hem durum.x."""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class _Yuklenen(BytesIO):
    """st.file_uploader'ın verdiği dosya gibi: bellekte dosya + boyut + ad."""

    def __init__(self, veri, name="liste.xlsx"):
        super().__init__(veri)
        self.size = len(veri)
        self.name = name
        self.file_id = "deneme"


def _normal_excel():
    cikti = BytesIO()
    pd.DataFrame(
        [("01.12.2026", "Ali", "Ece")], columns=["Tarih", "AYB", "GYB"]
    ).to_excel(cikti, index=False, sheet_name="Günlük Liste")
    return cikti.getvalue()


def _sisen_excel(mb=30):
    """Sıkıştırılmış hâli küçük, içindeki tek hücre ``mb`` MB metin olan Excel."""
    kaynak = zipfile.ZipFile(BytesIO(_normal_excel()))
    cikti = BytesIO()
    with zipfile.ZipFile(cikti, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as hedef:
        for parca in kaynak.infolist():
            veri = kaynak.read(parca.filename)
            if b"<t>Ali</t>" in veri:  # hücre metni: sayfada ya da ortak metin tablosunda
                bas, son = veri.split(b"<t>Ali</t>")
                veri = bas + b"<t>" + b"A" * (mb * 1024 * 1024) + b"</t>" + son
            hedef.writestr(parca.filename, veri)
    return cikti.getvalue()


def test_acilinca_buyuyen_excel_cok_buyuk_sayilir():
    veri = _sisen_excel()
    assert len(veri) < 2 * 1024 * 1024  # eski denetimden (sıkıştırılmış 2 MB) geçiyordu
    assert nobet_disa_aktar.excel_cok_buyuk(_Yuklenen(veri)) is True


def test_uygulamanin_kendi_exceli_kabul_edilir_ve_bastan_okunur():
    dosya = _Yuklenen(_normal_excel())
    assert nobet_disa_aktar.excel_cok_buyuk(dosya) is False
    # Denetim dosyayı başa sarar: ardından okuma eksiksiz çalışır.
    assert list(pd.read_excel(dosya).columns) == ["Tarih", "AYB", "GYB"]


def test_sikistirilmis_hali_buyuk_dosya_cok_buyuk_sayilir():
    assert nobet_disa_aktar.excel_cok_buyuk(_Yuklenen(b"x" * (2 * 1024 * 1024 + 1))) is True


def test_excel_olmayan_dosya_boyut_denetimine_takilmaz():
    # "Dosya okunamadı" uyarısını okuma adımı verir; burada reddedilmez.
    assert nobet_disa_aktar.excel_cok_buyuk(_Yuklenen(b"excel degil")) is False


def test_uzman_exceli_sisen_dosyayi_okumadan_reddeder(monkeypatch):
    hatalar = []
    monkeypatch.setattr(nobet_disa_aktar.st, "error", hatalar.append)

    def _okunmamali(*args, **kwargs):
        raise AssertionError("şişen dosya okunmaya çalışıldı")

    monkeypatch.setattr(nobet_disa_aktar, "read_uzman_excel", _okunmamali)
    assert nobet_disa_aktar._yuklenen_uzman_gunleri(_Yuklenen(_sisen_excel()), 2026, 12, {}) is None
    assert hatalar and "çok büyük" in hatalar[0]


def test_excelden_yukle_sisen_dosyayi_okumadan_reddeder(monkeypatch):
    st = nobet_disa_aktar.st
    durum = _Durum()
    monkeypatch.setattr(st, "session_state", durum)
    monkeypatch.setattr(st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: _Yuklenen(_sisen_excel()))
    monkeypatch.setattr(st, "rerun", lambda: None)

    def _okunmamali(*args, **kwargs):
        raise AssertionError("şişen dosya okunmaya çalışıldı")

    monkeypatch.setattr(nobet_disa_aktar.pd, "ExcelFile", _okunmamali)
    nobet_disa_aktar.render_excel_import(2026, 12, {}, ["Ali", "Ece"], [], ["AYB", "GYB"], lambda: None)
    assert [duzey for duzey, _ in durum["excel_import_messages"]] == ["error"]
    assert "çok büyük" in durum["excel_import_messages"][0][1]
    assert "schedule_bool" not in durum

