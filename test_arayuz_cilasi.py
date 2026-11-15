# -*- coding: utf-8 -*-
"""Arayüz cilası (14 Kas 2026): PNG kırpma + başlık, tek bildirim, koyu tema hafta sonu."""
import re
from io import BytesIO

import pandas as pd
from PIL import Image

from nobet_core import simulation_toast
from nobet_disa_aktar import convert_df_to_png
from nobet_stil import ui_colors


def _liste(n=30):
    return pd.DataFrame({
        "Tarih": [f"{i:02d}.11.2026 {'Paz' if i % 7 == 1 else 'Pzt'}" for i in range(1, n + 1)],
        "Görev1": ["ALİ"] * n, "Görev2": ["AYŞE"] * n,
    })


def _icerik_kutusu(png):
    img = Image.open(BytesIO(png)).convert("L")
    # Beyaz olmayan piksellerin sınırları
    kutu = img.point(lambda v: 255 if v < 250 else 0).getbbox()
    return img.size, kutu


def test_png_icerige_kirpilir():
    (w, h), (x0, y0, x1, y1) = _icerik_kutusu(convert_df_to_png(_liste()))
    # Her kenardaki boşluk, genişliğin/yüksekliğin %8'ini geçmemeli.
    assert x0 < 0.08 * w and w - x1 < 0.08 * w, (w, x0, x1)
    assert y0 < 0.08 * h and h - y1 < 0.08 * h, (h, y0, y1)


def test_png_basligi_ustte_gorunur():
    df = _liste()
    (_, h0), _ = _icerik_kutusu(convert_df_to_png(df))
    (w, h1), (x0, y0, x1, y1) = _icerik_kutusu(convert_df_to_png(df, "Kasım 2026 Nöbet Listesi"))
    assert h1 > h0  # başlık için yer açıldı
    assert y0 < 0.08 * h1 and h1 - y1 < 0.08 * h1  # yine de sıkı kırpılmış


def test_simulasyon_tek_bildirim():
    ozet = "Bu ay: hafta sonu farkı 1."
    normal = simulation_toast(ozet, False)
    assert normal == "En adil sonuç seçildi. " + ozet
    iyilesmis = simulation_toast(ozet, True)
    assert "İyileştir" in iyilesmis and ozet in iyilesmis
    assert "\n" not in iyilesmis


def test_koyu_tema_hafta_sonu_biraz_belirgin():
    def alfa(dark):
        m = re.match(r"rgba\(\d+,\d+,\d+,([\d.]+)\)", ui_colors(dark)["weekend_bg"])
        return float(m.group(1)) if m else None

    assert 0.24 <= alfa(True) <= 0.35
    assert ui_colors(False)["weekend_bg"] == "#dbeafe"  # açık tema değişmedi
