# -*- coding: utf-8 -*-
"""Simülasyondan önce: girilen izin ve limitlerle liste çıkabilir mi? (yalnız sayar, hiçbir şeyi değiştirmez)"""
import streamlit as st

from nobet_algoritma import _is_special_day, is_friday_column
from nobet_core import MISSING_DAY_FREE_LABEL, explain_missing_day
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


def _uygunluk_sirasi(schedule, col, gun_detaylari, kisiler):
    """Kişileri o gün için en uygundan başlayarak sıralar (hafta sonu/Cuma nöbeti az olan, sonra toplamı az olan)."""
    info = gun_detaylari[col]
    ayni_tur = _is_special_day if _is_special_day(info) else (is_friday_column if is_friday_column(info) else None)

    def yuk(person):
        n = 0 if ayni_tur is None else sum(
            1 for c in schedule.columns if c in gun_detaylari and ayni_tur(gun_detaylari[c]) and bool(schedule.at[person, c]))
        return (n, int(schedule.loc[person].sum()), list(schedule.index).index(person))

    return sorted(kisiler, key=yuk)


def gun_uygunlari(schedule, col, gun_detaylari, df_unwanted, min_bosluk,
                  person_limits=None, forbidden_pairs=None, initial_last_shift=None):
    """Takvimde bir güne basınca: o günün nöbetçileri, o güne yazılabilecek kişiler (en uygundan başlayarak)
    ve diğerlerinin neden uygun olmadığı. Çizelgeyi değiştirmez."""
    nobetciler = [p for p in schedule.index if bool(schedule.at[p, col])]
    nedenler = explain_missing_day(schedule, col, gun_detaylari, df_unwanted, min_bosluk,
                                   person_limits, forbidden_pairs, initial_last_shift)
    uygun = next((names for label, names in nedenler if label == MISSING_DAY_FREE_LABEL), [])
    return (nobetciler, _uygunluk_sirasi(schedule, col, gun_detaylari, uygun),
            [(label, names) for label, names in nedenler if label != MISSING_DAY_FREE_LABEL])


def render_day_availability(gun, sutun, gun_adi, schedule, gun_detaylari, df_unwanted, min_bosluk,
                            person_limits, forbidden_pairs, initial_last_shift):
    """Takvimde güne basılınca açılan pencere: o gün kim uygun, kim neden değil (yalnız gösterir)."""
    @st.dialog(f"{gun}. gün {gun_adi}: kim uygun?")
    def _pencere():
        nobetciler, uygun, nedenler = gun_uygunlari(schedule, sutun, gun_detaylari, df_unwanted, min_bosluk,
                                                    person_limits, forbidden_pairs, initial_last_shift)
        st.markdown("**Nöbetçi:** " + (", ".join(nobetciler) if nobetciler else "yok"))
        if uygun:
            st.markdown("**Yazılabilir (en uygundan başlayarak):** " + ", ".join(uygun))
        else:
            st.markdown("**Yazılabilir:** kimse yok")
        for label, names in nedenler:
            st.markdown(f"- {label}: {', '.join(names)}")
        st.caption("Şu anki listeye ve ayarlara göre; yalnız gösterir, hiçbir şeyi değiştirmez. "
                   "Birini yazmak için tabloyu kullanın.")
    _pencere()


def ilk_nobet_gunleri(frame):
    """Kaydedilmiş ayın kişi başına ilk nöbet günü (nöbeti olmayan yer almaz)."""
    first = {}
    for person in frame.index:
        days = [_column_day(c) for c in frame.columns if bool(frame.at[person, c]) and _column_day(c) is not None]
        if days:
            first[person] = min(days)
    return first


def ay_sonu_cakismalari(last_days, next_first, days_in_month, min_bosluk):
    """Bu ayın son nöbeti ile sonraki ayın ilk nöbeti arası dinlenme kuralına uymayanlar için uyarı metinleri."""
    messages = []
    for person, last in last_days.items():
        first = next_first.get(person)
        if last > 0 and first is not None and (days_in_month - last) + first <= min_bosluk:
            messages.append(f"⚠️ **{person}**: {last}. gün nöbeti ile sonraki ayın {first}. günü dinlenme kuralına uymuyor.")
    return messages


def dinlenme_uyarilari(ihlaller):
    """{kişi: [gün, ...]} için kişi başına tek uyarı satırı (16 benzer kutu yerine)."""
    return [
        f"⚠️ **{person}**: {', '.join(str(d) for d in days)}. "
        f"{'günler' if len(days) > 1 else 'gün'} dinlenme kuralına uymuyor."
        for person, days in ihlaller.items()
    ]
