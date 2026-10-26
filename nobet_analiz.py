# -*- coding: utf-8 -*-
"""Analiz sekmesi: nöbet yükü, kişisel detay, ücret özeti, eşleşme ve tercih özeti."""
import pandas as pd
import streamlit as st

from nobet_core import (
    TR_AY_ADLARI, active_fixed_roles, build_person_ics, distribute_roles, escape_html,
    most_paired, preference_summary, tr_sayi,
)
from nobet_durum import save_undo_state


# Açık ve koyu temada okunur yarı saydam zeminler.
_EN_COK_ZEMIN = "background-color: rgba(245,158,11,0.28)"
_EN_AZ_ZEMIN = "background-color: rgba(59,130,246,0.22)"


def _highlight_extremes(column):
    """Sütunun en çoğunu ve en azını boyar; fark 2'den azsa (dengeli) boyamaz."""
    _high, _low = column.max(), column.min()
    if _high - _low < 2:
        return [""] * len(column)
    return [
        _EN_COK_ZEMIN if _v == _high else _EN_AZ_ZEMIN if _v == _low else ""
        for _v in column
    ]


def render_analysis_tab(isimler, role_names, prev_ctx, history_special, history_total, history_holiday,
                        df_stats_load, df_stats_finance, pair_display, df_liste, liste_rows, liste_roles,
                        yil, ay, zorunlu_saat, arife_gunleri, edited_df):
    # ── Analiz — yatay dağılım ───────────────────────────────────────────────────

    # Satır 1: Nöbet Yükü  |  Kişisel Detay
    an1, an2 = st.columns(2)

    with an1:
        st.markdown("**Nöbet Yükü**")
        if history_special:
            _hist_text = " · ".join(
                f"{_p} {prev_ctx['history_counts'].get(_p, 0)}" for _p in isimler
            )
            st.caption(
                f"⚖️ Önceki {prev_ctx['months']} ayın hafta sonu/tatil nöbetleri dengelemede "
                f"hesaba katılıyor: {_hist_text}"
            )
        if history_total:
            _total_text = " · ".join(
                f"{_p} {'+' if _v > 0 else ''}{_v:g}"
                for _p, _v in sorted(history_total.items(), key=lambda _item: -_item[1])
            )
            st.caption(
                f"⚖️ Önceki {prev_ctx['months']} ayın toplam nöbetleri (ortalamaya göre fark): "
                f"{_total_text}. Fazlası olan bu ay \"1 fazla\" alanlardan olmaz."
            )
        if history_holiday:
            _holiday_text = " · ".join(
                f"{_p} {prev_ctx['holiday_counts'].get(_p, 0)}" for _p in isimler
            )
            st.caption(
                f"🎉 Son 12 aydaki bayram/tatil nöbetleri: {_holiday_text}. "
                "Bu ayın tatil günleri azı olana verilir."
            )
        # Günde tek nöbetçi varsa görev sütunu Toplam'ın aynısıdır; gösterilmez.
        _single_role = len(role_names) < 2
        _load_table = df_stats_load.drop(
            columns=[r for r in role_names if r in df_stats_load.columns]) if _single_role else df_stats_load
        st.dataframe(
            _load_table.style.apply(
                _highlight_extremes,
                subset=["Toplam", "Hafta sonu", "Cuma", "Tatil"] + ([] if _single_role else list(role_names)),
            ),
            width="stretch"
        )
        st.caption("Hafta sonuna tatiller de dahildir. Renkli hücreler sütunun en çoğu (turuncu) "
                   "ve en azıdır (mavi); fark 2'den azsa renk yoktur.")
        # Düğme görev adlarını kullanır (ör. "AYB / GYB sayılarını eşitle").
        if not _single_role and st.button(f"🤖 {' / '.join(role_names)} sayılarını eşitle", width="stretch",
                     disabled=bool(st.session_state.get("ay_kilitli")),
                     help="Nöbetçileri değiştirmeden yalnızca kimin hangi görev yerinde olduğunu "
                          "herkese eşit dağıtır."):
            if 'cached_rows_liste' in st.session_state and st.session_state.cached_rows_liste:
                save_undo_state()
                rows_liste_to_redist = st.session_state.cached_rows_liste
                # Sabit görev yeri olanlar kendi görevinde kalır, diğerleri eşitlenir.
                new_rows_liste, first_role_counts_to_redist = distribute_roles(
                    [row["Tarih"] for row in rows_liste_to_redist],
                    [[row.get(rn, "-") for rn in role_names if row.get(rn, "-") in isimler]
                     for row in rows_liste_to_redist],
                    role_names,
                    active_fixed_roles(st.session_state.get("fixed_roles"), isimler, role_names),
                )
                st.session_state.cached_rows_liste = new_rows_liste
                st.session_state.cached_first_role_counts = first_role_counts_to_redist
                # Nöbetçiler değişmez; çizelge yeniden kurulmaz (görev sayısından
                # fazla atanmış kişiler sessizce siliniyordu).
                st.session_state.should_regenerate_assignments = False
                st.toast("Görev yerleri yeniden dağıtıldı!", icon="🤖")
                # Üstteki takvim yeni rolleri hemen göstersin.
                st.rerun()

    with an2:
        st.markdown("**Kişisel Detay**")
        kisi_sec = st.selectbox("Kişi:", isimler, label_visibility="collapsed")
        st.download_button(
            f"📅 {kisi_sec} — takvime ekle (.ics)",
            lambda _p=kisi_sec, _rows=list(liste_rows), _roles=list(liste_roles), _y=yil, _m=ay:
                build_person_ics(_p, _rows, _roles, _y, _m).encode("utf-8"),
            f"nobet_{yil}_{ay:02d}.ics", "text/calendar",
            on_click="ignore", width="stretch",
            help="Telefonda açınca nöbetler takvime eklenir; başlıkta görev yeri (birim) yazar.",
        )
        kisi_rows = []
        for _, row in df_liste.iterrows():
            roller = []
            partners_set = set()
            for role_idx, role_name in enumerate(liste_roles):
                if role_name in row and row.get(role_name) == kisi_sec:
                    roller.append(role_name)
                    for other_idx, other_role in enumerate(liste_roles):
                        if other_idx != role_idx and other_role in row and row.get(other_role) and row[other_role] != '-':
                            partners_set.add(row[other_role])
            if roller:
                partner_str = ', '.join(sorted(partners_set)) if partners_set else 'Tek'
                kisi_rows.append({"Tarih": row['Tarih'], "Partner": partner_str, "Rol": ', '.join(roller)})
        df_kisi = pd.DataFrame(kisi_rows)
        if not df_kisi.empty:
            h_kisi = min(len(df_kisi) * 35 + 38, 400)
            st.dataframe(df_kisi, height=h_kisi, width="stretch", hide_index=True)
        else:
            st.info("Nöbet yok.")

    st.divider()

    # Satır 2: Ücret Özeti  |  Eşleşme Matrisi
    an3, an4 = st.columns(2)

    with an3:
        st.markdown("**Ücret Özeti**")
        _mesai_text = f"Zorunlu mesai {zorunlu_saat} saat: hafta içi iş günü 8 saat"
        if arife_gunleri:
            _mesai_text += ", " + ", ".join(
                f"{_day} {TR_AY_ADLARI[ay]} {_name}" for _day, _name in arife_gunleri
            ) + " yarım gün (4 saat)"
        st.caption(_mesai_text + ". İzin işaretli iş günleri kişinin mesaisinden düşer; "
                   "Müsait Değil düşmez.")
        st.dataframe(
            df_stats_finance.style.format({"Ücret (TL)": lambda value: f"{tr_sayi(value)} ₺"}),
            width="stretch"
        )

    with an4:
        st.markdown("**En çok birlikte tutanlar**")
        _pairs = most_paired(st.session_state.schedule_bool, isimler, list(st.session_state.schedule_bool.columns))
        if _pairs and _pairs[0][2] <= 1:
            st.caption("Hiçbir ikili bu ay birden fazla birlikte nöbet tutmuyor.")
        elif _pairs:
            st.markdown(
                "".join(f"<div>{escape_html(_a)} – {escape_html(_b)}: <b>{_n}</b> nöbet</div>"
                        for _a, _b, _n in _pairs),
                unsafe_allow_html=True,
            )
        else:
            st.caption("Henüz birlikte nöbet yok.")
        with st.expander("Eşleşme Matrisi (herkes)"):
            st.dataframe(pair_display, width="stretch")

    # Satır 3: Ekip tercih özeti (tercih/kaçın/izin işaretlerinin sonucu)
    st.markdown("**Tercih Özeti**")
    _pref_summary = preference_summary(edited_df, st.session_state.pref_df, isimler)
    if _pref_summary is None:
        st.caption("Tabloda işaretli tercih, kaçınma, izin ya da müsait değil günü yok.")
    else:
        st.caption(
            "Her kişinin yeşil (tercih) günlerinin kaçında nöbet tuttuğu, sarı (kaçın) günlerine "
            "kaç nöbet düştüğü ve izinli ya da müsait olmadığı güne nöbet yazılıp yazılmadığı."
        )
        st.dataframe(_pref_summary, width="stretch")
