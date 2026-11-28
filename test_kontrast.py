# -*- coding: utf-8 -*-
"""Okunurluk (sekiz rollü inceleme S7/S8): yazı/zemin kontrastı WCAG AA (4,5:1) ve rozette ⚠."""
import re
from pathlib import Path

from nobet_stil import contrast_css

ROOT = Path(__file__).parent


def _lum(hex_color):
    rgb = [int(hex_color.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _ratio(a, b):
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_ana_dugme_yazisi_okunur():
    config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    light = re.search(r'\[theme\.light\][^\[]*primaryColor = "(#[0-9a-f]{6})"', config).group(1)
    dark = re.search(r'\[theme\.dark\][^\[]*primaryColor = "(#[0-9a-f]{6})"', config).group(1)
    assert _ratio("#ffffff", light) >= 4.5
    # Koyu temada düğme yazısı koyu renge çevrilir.
    dark_text = re.search(r"stBaseButton-primary.*?color: (#[0-9a-f]{6})", contrast_css(True)).group(1)
    assert _ratio(dark_text, dark) >= 4.5


def test_aciklama_yazisi_acik_temada_okunur():
    caption = re.search(r"stCaptionContainer.*?color: (#[0-9a-f]{6})", contrast_css(False)).group(1)
    assert _ratio(caption, "#ffffff") >= 4.5
    assert "stCaptionContainer" not in contrast_css(True)  # koyu temada zaten okunuyor


def test_paylasimda_gecmis_gunler_okunur():
    source = (ROOT / "nobet_paylasim.py").read_text(encoding="utf-8")
    past = re.search(r'"past": "color: (#[0-9a-f]{6});"', source).group(1)
    assert _ratio(past, "#ffffff") >= 4.5


def test_kisi_rozeti_okunur_ve_sinir_disi_isaretli():
    source = (ROOT / "nobet_tablo.py").read_text(encoding="utf-8")
    colors = re.search(r'badge_bg = "(#[0-9a-f]{6})" if out_of_limits else "(#[0-9a-f]{6})"', source).groups()
    assert all(_ratio("#ffffff", c) >= 4.5 for c in colors)
    # Renk körlüğünde kırmızı/yeşil ayrılmıyor: sınır dışı rozet ⚠ ile de belli olur.
    assert "{' ⚠' if out_of_limits else ''}" in source
    assert "opacity:.85" not in source


def test_gri_yazi_streamlitin_soluklastirmasiyla_tekrar_solmaz():
    # Streamlit açıklamaya opacity:0.6 verir; renk koyulaştırılsa da bu kapatılmazsa 2,6:1'e düşer.
    assert "opacity: 1" in contrast_css(False)


def test_talebi_gonder_dugmesi_koyu_temada_okunur():
    # Form gönder düğmesi ayrı bir seçici kullanır; ana düğmeyle aynı koyu yazı rengini almalı.
    assert "stBaseButton-primaryFormSubmit" in contrast_css(True)


def test_yazdirmada_ay_adi_gorunur_ve_takvim_ilk_sayfada():
    stil = (ROOT / "nobet_stil.py").read_text(encoding="utf-8")
    assert ".st-key-cal-head { break-before: page; }" not in stil  # 1. sayfa boş kalıyordu
    assert ".st-key-month-bar," not in stil.split("@media print")[1].split("display: none")[0]


def test_analiz_uygun_sutunu_renksiz_ayirt_edilir():
    from nobet_core import build_schedule_analysis  # noqa: F401  ("Uygun" bu tablodan gelir)
    source = (ROOT / "nobet_core.py").read_text(encoding="utf-8")
    assert '"Uygun": "✓"' in source and "else \"⚠\"" in source
    assert '"🟢" if min_limit' not in source
