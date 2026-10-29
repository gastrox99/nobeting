# -*- coding: utf-8 -*-
"""Simülasyondan önce: girilen izin ve limitlerle liste çıkabilir mi? (yalnız sayar, hiçbir şeyi değiştirmez)"""
import streamlit as st

from nobet_gunler import _column_day

_EN_FAZLA_GUN = 8


def onkontrol_uyarilari(df_unwanted, columns, kisi_sayisi, min_bosluk, person_limits=None):
    """Olası sorunların metin listesi; sorun yoksa boş.

    Dolaylı engeller (dinlenme zinciri, yasak çift) yakalanmaz; bu yüzden "olası" denir."""
    limits = person_limits or {}
    warnings = []
    short = []
    for column in columns:
        free = sum(1 for p in df_unwanted.index if not bool(df_unwanted.at[p, column]))
        if free < kisi_sayisi:
            short.append(f"{_column_day(column)}. gün ({free} müsait)")
    if short:
        shown = ", ".join(short[:_EN_FAZLA_GUN]) + (f" ve {len(short) - _EN_FAZLA_GUN} gün daha" if len(short) > _EN_FAZLA_GUN else "")
        warnings.append(f"Her güne {kisi_sayisi} kişi gerekiyor ama müsait kişi yetmiyor: {shown}.")
    # Bir kişi en fazla: limiti, müsait günü ve dinlenme aralığının izin verdiği kadar nöbet tutabilir.
    rest_cap = -(-len(columns) // (min_bosluk + 1))
    capacity = 0
    for person in df_unwanted.index:
        free_days = sum(1 for c in columns if not bool(df_unwanted.at[person, c]))
        capacity += min(limits.get(person, {}).get("max", 999), free_days, rest_cap)
    needed = kisi_sayisi * len(columns)
    if capacity < needed:
        warnings.append(
            f"Ekibin bu ayki toplam kapasitesi {capacity} nöbet, gereken {needed}; "
            f"en az {needed - capacity} nöbet açıkta kalır."
        )
    return warnings


def render_precheck(df_unwanted, columns, kisi_sayisi, min_bosluk, person_limits):
    """Sorun yoksa hiçbir şey göstermez."""
    warnings = onkontrol_uyarilari(df_unwanted, columns, kisi_sayisi, min_bosluk, person_limits)
    if warnings:
        st.warning("🔎 Ön kontrol (olası sorunlar; simülasyon çalıştırmadan görünür):\n\n"
                   + "\n\n".join(f"- {w}" for w in warnings))


def paylasim_kontrolleri(bekleyen, eksik_gunler, katsayi_metni):
    """Paylaşmadan önce gösterilecek kısa uyarılar; hepsi yeşilse boş liste."""
    items = []
    if bekleyen:
        items.append(f"{bekleyen} ekip talebi onay bekliyor.")
    if eksik_gunler:
        items.append(f"{len(eksik_gunler)} günün nöbetçisi eksik: {', '.join(str(d) for d in eksik_gunler[:_EN_FAZLA_GUN])}"
                     + (" …" if len(eksik_gunler) > _EN_FAZLA_GUN else "."))
    if katsayi_metni:
        items.append(katsayi_metni)
    return items


def render_share_check(items):
    """Paylaş / WhatsApp açılırken; sorun yoksa hiçbir şey göstermez."""
    if items:
        st.warning("🔎 Göndermeden önce:\n\n" + "\n\n".join(f"- {i}" for i in items))
