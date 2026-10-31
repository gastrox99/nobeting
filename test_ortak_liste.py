# -*- coding: utf-8 -*-
"""Ortak listenin PNG, WhatsApp metni ve yazdırma sayfası."""
from nobet_disa_aktar import ortak_liste_html, ortak_liste_metni, ortak_liste_png
from nobet_gunler import build_gun_detaylari

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
