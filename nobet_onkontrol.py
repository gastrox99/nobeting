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


def neden_yok_cevabi(reasons, person):
    """explain_missing_day sonucundan tek kişi için cevap; kişi o gün nöbetçiyse (listede yoksa) bunu söyler."""
    for label, names in reasons:
        for name in names:
            if name == person or name.startswith(f"{person} ("):
                detail = name[len(person):].strip()
                return f"{person}: {label} {detail}".strip()
    return f"{person} o gün zaten nöbetçi."


def render_why_not(isimler, sutunlar, gun_detaylari, reasons_for_day, schedule):
    """Analiz altı: bir gün ve kişi seçince o kişinin o güne neden yazılmadığını (şu anki listeye göre) söyler."""
    if not isimler:
        return
    with st.expander("❓ Neden o gün yok?", expanded=False):
        by_day = {gun_detaylari[c]["day_num"]: c for c in sutunlar}
        day = st.selectbox("Gün", list(by_day), key="neden_gun")
        person = st.selectbox("Kişi", isimler, key="neden_kisi")
        if day is None or not person:
            return
        column = by_day[day]
        st.write(neden_yok_cevabi(reasons_for_day(schedule, column), person))
        st.caption("Şu anki listeye ve ayarlara göre; dinlenme, limit, izin ve yasak çift kurallarına bakar.")


def yedek_adaylari(schedule, col, cikamayan, gun_detaylari, df_unwanted, min_bosluk,
                   person_limits=None, forbidden_pairs=None, initial_last_shift=None):
    """Nöbetçisi çıkamayan güne yazılabilecek kişiler (en uygundan başlayarak) ve diğerlerinin nedenleri.

    Çizelgeyi değiştirmez. Sıra: o gün hafta sonu/tatilse hafta sonu nöbeti, Cumaysa Cuma nöbeti
    en az olan önce; sonra toplam nöbeti az olan."""
    kalan = schedule.copy()
    kalan.at[cikamayan, col] = False
    nedenler = explain_missing_day(kalan, col, gun_detaylari, df_unwanted, min_bosluk,
                                   person_limits, forbidden_pairs, initial_last_shift)
    uygun = next((names for label, names in nedenler if label == MISSING_DAY_FREE_LABEL), [])
    uygun = [p for p in uygun if p != cikamayan]
    info = gun_detaylari[col]
    ayni_tur = _is_special_day if _is_special_day(info) else (is_friday_column if is_friday_column(info) else None)

    def yuk(person):
        n = 0 if ayni_tur is None else sum(
            1 for c in schedule.columns if c in gun_detaylari and ayni_tur(gun_detaylari[c]) and bool(schedule.at[person, c]))
        return (n, int(schedule.loc[person].sum()), list(schedule.index).index(person))

    return sorted(uygun, key=yuk), [(l, [n for n in names if n != cikamayan]) for l, names in nedenler
                                    if l != MISSING_DAY_FREE_LABEL]


def render_backup_suggestion(isimler, sutunlar, gun_detaylari, schedule, df_unwanted, min_bosluk,
                             person_limits, forbidden_pairs, initial_last_shift, assign_backup):
    """Analiz altı: çıkamayan kişinin gününe yazılabilecek yedekleri sıralar; seçileni o güne yazar."""
    if not isimler:
        return
    with st.expander("🆘 Boşalan güne yedek öner", expanded=False):
        by_day = {gun_detaylari[c]["day_num"]: c for c in sutunlar if bool(schedule[c].any())}
        day = st.selectbox("Gün", list(by_day), key="yedek_gun")
        if day is None:
            st.caption("Listede nöbetçisi olan gün yok.")
            return
        column = by_day[day]
        nobetciler = [p for p in schedule.index if bool(schedule.at[p, column])]
        out = st.selectbox("Çıkamayan kişi", nobetciler, key="yedek_kisi")
        if not out:
            return
        adaylar, nedenler = yedek_adaylari(schedule, column, out, gun_detaylari, df_unwanted, min_bosluk,
                                           person_limits, forbidden_pairs, initial_last_shift)
        if not adaylar:
            st.info("Uygun yedek yok. " + " · ".join(f"{label}: {', '.join(names)}" for label, names in nedenler))
            return
        st.caption("En uygundan başlayarak (hafta sonu/Cuma ve toplam nöbeti az olan önce). Ayın geri kalanı değişmez.")
        secilen = st.radio("Yedek", adaylar, key="yedek_secim")
        if st.button(f"{secilen} {day}. güne yazılsın, {out} çıkarılsın", key="yedek_yaz"):
            assign_backup(out, secilen, column)
            st.rerun()
        if nedenler:
            st.caption("Diğerleri: " + " · ".join(f"{label}: {', '.join(names)}" for label, names in nedenler))


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
