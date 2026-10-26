# -*- coding: utf-8 -*-
"""Tablo Görünümü sekmesi: tercih boyama ve elle nöbet atama tablosu."""
from datetime import date

import pandas as pd
import streamlit as st

from nobet_core import (
    active_locks,
    escape_html,
    get_preference_grid_legend,
    get_preference_grid_state,
    name_column_width,
    normalize_preference_grid,
    parse_holiday_days,
    redo_history_state,
    short_display_names,
    undo_history_state,
    update_locked_cells,
    use_compact_schedule_editor,
)
from nobet_durum import current_state_snapshot, restore_snapshot, save_undo_state
from nobet_stil import grid_css


def render_grid_tab(isimler, sutunlar, gun_detaylari, yil, ay, today_date, ui,
                    person_colors, conflict_cells, queue_assignment_warnings):
    if st.session_state.get("ay_kilitli"):
        st.info("🔒 Bu ay kilitli; yalnızca görüntülenir. Değiştirmek için ayın üstündeki 🔓 düğmesiyle kilidi açın.")
        st.dataframe(st.session_state.schedule_bool.replace({True: "●", False: ""}), width="stretch")
        return
    # Mode selector and paint color
    mode_cols = st.columns([2, 3])
    with mode_cols[0]:
        # Simülasyondan sonra düzeltme yapılacağı için tablo Nöbet Ata moduna geçer
        # (tercih modunda dokunulan hücre nöbet yerine tercih boyuyordu).
        if st.session_state.pop("switch_to_assign_mode", False):
            st.session_state.mode_radio = "✏️ Nöbet Ata/Kaldır"
        edit_mode = st.radio(
            "Düzenleme Modu:",
            ["🎨 Tercih Belirle", "✏️ Nöbet Ata/Kaldır"],
            horizontal=True,
            key="mode_radio"
        )
        st.session_state.edit_mode = "tercih" if "Tercih" in edit_mode else "atama"

    with mode_cols[1]:
        if st.session_state.edit_mode == "tercih":
            color_info = [
                (0, "⬜Nötr"),
                (1, "🟩Tercih"),
                (2, "🟨Kaçın"),
                (3, "🟥İzin"),
                (4, "🟪Müsait Değil"),
            ]
            if 'paint_color' not in st.session_state:
                st.session_state.paint_color = 0
            color_cols = st.container(key="paint-palette").columns(len(color_info))
            for i, (val, label) in enumerate(color_info):
                with color_cols[i]:
                    btn_type = "primary" if st.session_state.paint_color == val else "secondary"
                    if st.button(label, key=f"color_{val}", type=btn_type):
                        st.session_state.paint_color = val
                        st.rerun()
        else:
            st.caption("Hücreye dokununca nöbet eklenir ya da kalkar. Elle eklenen nöbetler "
                       "(sarı çerçeve) simülasyonda korunur.")

    # Quick actions row
    # Simülasyon/İyileştir geniş, Sıfırla/Geri/İleri dar; telefonda iki satıra
    # sığar (CSS: .st-key-grid-actions).
    action_cols = st.container(key="grid-actions").columns([2, 2, 1, 1, 1])
    with action_cols[0]:
        if st.button("⚡ Simülasyon", type="primary", width="stretch"):
            st.session_state.run_simulation = True
    with action_cols[1]:
        if st.button(
            "✨ İyileştir", width="stretch",
            disabled=not bool(st.session_state.schedule_bool.values.any()),
            help="Mevcut listeden başlayıp birkaç saniye içinde daha adilini arar. "
                 "Kurallar ve kilitli nöbetler aynen korunur; hafta sonu/tatil, Cuma ve "
                 "toplam dengesi kötüleşmez. Beğenmezseniz Geri ile dönebilirsiniz.",
        ):
            st.session_state.run_improve = True
    with action_cols[2]:
        if st.button("🔄 Sıfırla", width="stretch"):
            st.session_state.reset_pending = True
            st.rerun()
    with action_cols[3]:
        if st.button("↩️ Geri", width="stretch", disabled=len(st.session_state.undo_history)==0):
            _snap = undo_history_state(st.session_state, current_state_snapshot())
            if _snap is not None:
                restore_snapshot(_snap, sutunlar, gun_detaylari)
                st.rerun()
    with action_cols[4]:
        if st.button("↪️ İleri", width="stretch", disabled=len(st.session_state.redo_history)==0):
            _snap = redo_history_state(st.session_state, current_state_snapshot())
            if _snap is not None:
                restore_snapshot(_snap, sutunlar, gun_detaylari)
                st.rerun()

    if st.session_state.get("reset_pending", False):
        # Çoğu zaman istenen, izinler kalırken nöbetleri yeniden dağıtmaktır.
        st.warning("⚠️ Neyi silmek istiyorsunuz? Geri ile geri alınabilir.")
        _reset_cols = st.columns(3)
        with _reset_cols[0]:
            if st.button("🧹 Yalnız nöbetleri sil", type="primary", width="stretch"):
                save_undo_state()
                st.session_state.schedule_bool = pd.DataFrame(False, index=isimler, columns=sutunlar)
                st.session_state.locked_cells = set()
                st.session_state.cached_rows_liste = None
                st.session_state.should_regenerate_assignments = True
                st.session_state.reset_pending = False
                st.rerun()
        with _reset_cols[1]:
            if st.button("🗑️ Tercih ve izinleri de sil", width="stretch"):
                save_undo_state()
                st.session_state.pref_df = pd.DataFrame(0, index=isimler, columns=sutunlar)
                st.session_state.schedule_bool = pd.DataFrame(False, index=isimler, columns=sutunlar)
                st.session_state.locked_cells = set()
                st.session_state.hidden_people = {}
                st.session_state.cached_rows_liste = None
                st.session_state.should_regenerate_assignments = True
                st.session_state.reset_pending = False
                st.rerun()
        with _reset_cols[2]:
            if st.button("Vazgeç", width="stretch"):
                st.session_state.reset_pending = False
                st.rerun()

    if st.session_state.edit_mode == "tercih":
        _render_quick_entry(isimler, sutunlar, gun_detaylari)

    # Dynamic column background CSS
    col_bg_css_parts = []
    grid_css_selector = (
        ".st-key-preference-grid"
        if st.session_state.edit_mode == "tercih"
        else ".st-key-schedule-grid"
    )
    for _i, _col in enumerate(sutunlar):
        _idx = _i + 2
        _info = gun_detaylari[_col]
        _is_today = (date(yil, ay, _info['day_num']) == today_date)
        if _is_today:
            col_bg_css_parts.append(
                f"{grid_css_selector} [data-testid='stHorizontalBlock'] > :is([data-testid='column'],[data-testid='stColumn']):nth-child({_idx})"
                f"{{ background:{ui['today_bg']}!important; border-radius:6px; outline:2px solid #f59e0b; }}"
            )
        elif _info['holiday']:
            col_bg_css_parts.append(
                f"{grid_css_selector} [data-testid='stHorizontalBlock'] > :is([data-testid='column'],[data-testid='stColumn']):nth-child({_idx})"
                f"{{ background:{ui['holiday_bg']}!important; border-radius:6px; }}"
            )
        elif _info['weekend']:
            col_bg_css_parts.append(
                f"{grid_css_selector} [data-testid='stHorizontalBlock'] > :is([data-testid='column'],[data-testid='stColumn']):nth-child({_idx})"
                f"{{ background:{ui['weekend_bg']}!important; border-radius:6px; }}"
            )
    col_bg_css = "\n".join(col_bg_css_parts)
    # Rozet "6 · 2 HS" (toplam · hafta sonu/tatil) için yer ayrılır.
    name_col_px = name_column_width(isimler, maximum=240, badge_px=56)

    st.markdown(grid_css(ui, name_col_px, col_bg_css), unsafe_allow_html=True)

    grid_key = "preference-grid" if st.session_state.edit_mode == "tercih" else "schedule-grid"
    grid_container = st.container(key=grid_key, border=False)

    tr_gunler_short = {0:"Pzt", 1:"Sal", 2:"Çar", 3:"Per", 4:"Cum", 5:"Cmt", 6:"Paz"}
    _compact_editor = use_compact_schedule_editor(len(isimler), len(sutunlar))

    def _draw_day_header():
        header_cols = grid_container.columns([2] + [1] * len(sutunlar), gap=None)
        with header_cols[0]:
            st.markdown(
                f"<div class='grid-head' style='font-size:12px;font-weight:700;color:{ui['muted']};padding:2px 4px;'"
                " title='Kişi · toplam nöbet · hafta sonu/tatil (HS) nöbeti'>Kişi · Nöbet</div>",
                unsafe_allow_html=True,
            )
        for i, col in enumerate(sutunlar):
            with header_cols[i + 1]:
                day_num = gun_detaylari[col]['day_num']
                day_name = tr_gunler_short[date(yil, ay, day_num).weekday()]
                if date(yil, ay, day_num) == today_date:
                    num_style = f"color:{ui['today_text']};font-weight:800;font-size:15px;"
                    name_style = f"color:{ui['today_text']};font-size:11px;"
                elif gun_detaylari[col]['holiday']:
                    num_style = f"color:{ui['holiday_text']};font-weight:700;font-size:14px;"
                    name_style = f"color:{ui['holiday_text']};font-size:11px;"
                elif gun_detaylari[col]['weekend']:
                    num_style = f"color:{ui['weekend_text']};font-weight:700;font-size:14px;"
                    name_style = f"color:{ui['weekend_text']};font-size:11px;"
                else:
                    num_style = f"color:{ui['day_text']};font-size:14px;"
                    name_style = f"color:{ui['muted']};font-size:11px;"
                # Bugün sarı çerçeve, tatil kırmızı renkle belli; rozet eklenmez
                # (dar sütunda rakamı ikiye bölüyordu).
                st.markdown(
                    f"<div style='text-align:center;line-height:1.35;padding:2px 0;'>"
                    f"<div style='{num_style}white-space:nowrap;'>{day_num}</div>"
                    f"<div style='{name_style}'>{day_name}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # Hızlı düzenleme tablosunun kendi başlıkları var; gün başlığı yalnızca
    # düğmeli tabloda çizilir (eskiden tablonun üstünde hizasız duruyordu).
    if not _compact_editor:
        _draw_day_header()

    if _compact_editor:
        st.caption(
            "⚡ Büyük çizelge hızlı düzenleme modunda. Hücre başına ayrı buton yerine "
            "tek tablo kullanılır; bu nedenle düzenlemeler daha akıcı kalır."
        )
        if st.session_state.edit_mode == "tercih":
            st.caption("Tercih kodları: 0 = boş, 1 = tercih, 2 = kaçınma, 3 = izin (mesaiden düşer), 4 = müsait değil")
            _compact_pref = st.data_editor(
                st.session_state.pref_df,
                width="stretch",
                key="large_schedule_preference_editor",
                num_rows="fixed",
            )
            _normalized_pref = normalize_preference_grid(_compact_pref)
            if _normalized_pref is None:
                st.error("Tercih hücrelerinde yalnızca 0, 1, 2, 3 veya 4 tam sayıları kullanılabilir.")
            elif not _normalized_pref.equals(st.session_state.pref_df):
                save_undo_state()
                st.session_state.pref_df = _normalized_pref
                st.rerun()
        else:
            _compact_schedule = st.data_editor(
                st.session_state.schedule_bool,
                width="stretch",
                key="large_schedule_assignment_editor",
                num_rows="fixed",
            )
            if not _compact_schedule.equals(st.session_state.schedule_bool):
                save_undo_state()
                _previous_schedule = st.session_state.schedule_bool
                st.session_state.schedule_bool = _compact_schedule.astype(bool)
                _compact_cells = [(_p, _c) for _c in sutunlar for _p in isimler]
                st.session_state.locked_cells = update_locked_cells(
                    st.session_state.get("locked_cells"),
                    [x for x in _compact_cells
                     if bool(st.session_state.schedule_bool.at[x]) and not bool(_previous_schedule.at[x])],
                    [x for x in _compact_cells
                     if not bool(st.session_state.schedule_bool.at[x]) and bool(_previous_schedule.at[x])],
                )
                queue_assignment_warnings([
                    (_p, _c)
                    for _c in sutunlar
                    for _p in isimler
                    if bool(st.session_state.schedule_bool.at[_p, _c])
                    and not bool(_previous_schedule.at[_p, _c])
                ])
                st.rerun()
    else:
        _person_limits = st.session_state.get('person_limits', {})
        _short_names = short_display_names(isimler)
        _special_cols = [c for c in sutunlar if gun_detaylari[c]['weekend'] or gun_detaylari[c]['holiday']]
        _locked_now = active_locks(st.session_state.get("locked_cells"), st.session_state.schedule_bool)
        for person in isimler:
            row_cols = grid_container.columns([2] + [1] * len(sutunlar), gap=None)
            with row_cols[0]:
                count = int(st.session_state.schedule_bool.loc[person].sum()) if person in st.session_state.schedule_bool.index else 0
                special_count = int(st.session_state.schedule_bool.loc[person, _special_cols].sum()) if (
                    person in st.session_state.schedule_bool.index and _special_cols) else 0
                p_lim = _person_limits.get(person, {})
                min_l = p_lim.get('min', 0)
                max_l = p_lim.get('max', 999)
                badge_bg = "#dc2626" if (count < min_l or (max_l < 999 and count > max_l)) else "#16a34a"
                pc = person_colors.get(person, "#e2e8f0")
                # Telefonda kısa ad ("Ayşe Y.") ve rozet alt satırda (CSS: .grid-name);
                # rozet: toplam · hafta sonu/tatil nöbeti.
                st.markdown(
                    f"<div class='grid-name' style='font-size:13px;white-space:nowrap;padding:2px 4px;color:{ui['name_text']};"
                    f"display:flex;flex-wrap:nowrap;align-items:center;gap:4px;min-height:30px;overflow:hidden;'"
                    f" title='{escape_html(person)}: {count} nöbet, {special_count} hafta sonu/tatil'>"
                    f"<span class='nm-dot' style='flex-shrink:0;width:11px;height:11px;border-radius:50%;"
                    f"background:{pc};border:1px solid rgba(0,0,0,0.12);'></span>"
                    f"<b class='nm-full' style='min-width:0;overflow:hidden;text-overflow:ellipsis;'>{escape_html(person)}</b>"
                    f"<b class='nm-short' style='min-width:0;overflow:hidden;text-overflow:ellipsis;'>"
                    f"{escape_html(_short_names.get(person, person))}</b>"
                    f"<span class='nm-badge' style='flex-shrink:0;background:{badge_bg};color:#fff;border-radius:10px;"
                    f"padding:1px 7px;font-size:11px;font-weight:700;'>{count}"
                    f"<span style='font-weight:600;opacity:.85;'> · {special_count} HS</span></span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            for i, col in enumerate(sutunlar):
                with row_cols[i + 1]:
                    pref_val = st.session_state.pref_df.at[person, col] if person in st.session_state.pref_df.index else 0
                    is_assigned = st.session_state.schedule_bool.at[person, col] if person in st.session_state.schedule_bool.index else False
                    has_conflict = (person, col) in conflict_cells
                    cell_state = get_preference_grid_state(
                        is_assigned,
                        pref_val,
                        has_conflict,
                    )
                    st.markdown(
                        f'<span class="preference-state {cell_state["css_class"]}'
                        f'{" cell-locked" if (person, col) in _locked_now else ""}" aria-hidden="true"></span>',
                        unsafe_allow_html=True
                    )
                    if st.button(
                        cell_state["symbol"],
                        key=f"g_{person}_{col}",
                        width="stretch",
                    ):
                        save_undo_state()
                        if st.session_state.edit_mode == "tercih":
                            st.session_state.pref_df.at[person, col] = st.session_state.paint_color
                        else:
                            st.session_state.schedule_bool.at[person, col] = not st.session_state.schedule_bool.at[person, col]
                            _now_on = bool(st.session_state.schedule_bool.at[person, col])
                            st.session_state.locked_cells = update_locked_cells(
                                st.session_state.get("locked_cells"),
                                [(person, col)] if _now_on else [],
                                [] if _now_on else [(person, col)],
                            )
                            queue_assignment_warnings([(person, col)])
                        st.rerun()

    legend_container = st.container(key="preference-legend", border=False)
    _soft_style = {
        "neutral": (ui["free_bg"], ui["free_text"], f"1px solid {ui['free_border']}"),
        "preferred": (ui["soft_green_bg"], ui["soft_green_text"], "none"),
        "avoid": (ui["soft_amber_bg"], ui["soft_amber_text"], "none"),
        "unavailable": (ui["soft_red_bg"], ui["soft_red_text"], "none"),
        "not_available": (ui["soft_purple_bg"], ui["soft_purple_text"], "none"),
    }

    def _legend_item(_state, _label):
        if _state.get("soft"):
            _bg, _fg, _border = _soft_style[_state["key"]]
        elif _state["key"] == "assigned_neutral":
            _bg, _fg, _border = ui["assigned_bg"], ui["assigned_text"], "none"
        else:
            _bg, _fg, _border = _state["color"], "#ffffff", "none"
        return (
            f"<span style='display:inline-flex;align-items:center;gap:6px;margin:2px 12px 2px 0;'>"
            f"<span class='preference-legend-swatch swatch-{_state['key']}' style='background:{_bg};color:{_fg} !important;border:{_border};'>"
            f"{_state['symbol']}</span>"
            f"<span style='color:{ui['legend_text']};font-size:12px;'>{_label}</span></span>"
        )

    _legend_states = get_preference_grid_legend()
    _on = [x for x in _legend_states if not x.get("soft")]
    _off = [x for x in _legend_states if x.get("soft")]
    _row_label = f"style='color:{ui['muted']};font-size:11px;font-weight:700;letter-spacing:.04em;margin-right:10px;'"
    _legend_rows = (
        "<div style='display:flex;flex-direction:column;gap:4px;margin-top:6px;'>"
        f"<div style='display:flex;flex-wrap:wrap;align-items:center;'><span {_row_label}>NÖBETLİ</span>"
        + "".join(_legend_item(x, x["description"].replace("Atandı + ", "")) for x in _on)
        + "</div>"
        f"<div style='display:flex;flex-wrap:wrap;align-items:center;'><span {_row_label}>NÖBETSİZ</span>"
        + "".join(_legend_item(x, x["description"]) for x in _off)
        + "</div></div>"
    )
    # Masaüstünde açık; telefonda "🎨 Renkler" altında katlı (CSS: .legend-wide/.legend-fold).
    legend_container.markdown(
        f"<div class='legend-wide'>{_legend_rows}</div>"
        f"<details class='legend-fold'><summary style='cursor:pointer;color:{ui['legend_text']};font-size:13px;'>"
        f"🎨 Renkler</summary>{_legend_rows}</details>",
        unsafe_allow_html=True,
    )


# Hızlı giriş türleri: tablodaki boya düğmeleriyle aynı kodlar.
QUICK_ENTRY_TYPES = {"İzin": 3, "Müsait Değil": 4, "Tercih": 1, "Kaçın": 2, "Nötr (sil)": 0}


def apply_quick_entry(pref_df, person, days, value, gun_detaylari):
    """Kişinin verilen günlerine tercih kodunu yazar; değişen gün sayısını döner."""
    _by_day = {gun_detaylari[_c]["day_num"]: _c for _c in pref_df.columns if _c in gun_detaylari}
    _changed = 0
    for _day in sorted(days):
        _col = _by_day.get(_day)
        if _col is not None and int(pref_df.at[person, _col]) != value:
            pref_df.at[person, _col] = value
            _changed += 1
    return _changed


def _clear_quick_entry_days():
    # Kutuya yeni anahtar vermek, formda yazılıp gönderilmemiş metni de kesin siler.
    st.session_state.quick_entry_days_gen = st.session_state.get("quick_entry_days_gen", 0) + 1


def _render_quick_entry(isimler, sutunlar, gun_detaylari):
    """Kişi + gün aralığı ile tek seferde izin/tercih girişi (hücre hücre boyamaya ek)."""
    _notice = st.session_state.pop("quick_entry_notice", None)
    with st.expander("⚡ Hızlı giriş: kişi + günler", expanded=bool(_notice)):
        if _notice:
            getattr(st, _notice[0])(_notice[1])
        # Tek satır: etiketler gizli, kutuların içindeki yazı yol gösterir. Telefonda da
        # alt alta düşmez (nobet_stil .st-key-quick-entry).
        with st.container(key="quick-entry"):
            _cols = st.columns([3, 7], gap="small", vertical_alignment="center")
            # Kişi formun dışında: değişince önceki kişinin günleri temizlenir, yanlışlıkla
            # başka birine kaydedilmez. Tür seçimi bilerek korunur.
            _person = _cols[0].selectbox(
                "Kişi", isimler, key="quick_entry_person", on_change=_clear_quick_entry_days,
                label_visibility="collapsed",
            )
            # Form: yazarken sayfa yeniden çizilmez, yalnızca Uygula'da işlenir.
            with _cols[1].form("quick_entry_form", border=False):
                _fcols = st.columns([3, 3, 2], gap="small", vertical_alignment="center")
                _days_key = f"quick_entry_days_{st.session_state.get('quick_entry_days_gen', 0)}"
                _days_text = _fcols[0].text_input(
                    "Günler", placeholder="Gün: 3-9, 15", key=_days_key,
                    label_visibility="collapsed",
                )
                _kind = _fcols[1].selectbox(
                    "Tür", list(QUICK_ENTRY_TYPES), key="quick_entry_kind",
                    label_visibility="collapsed",
                )
                _submitted = _fcols[2].form_submit_button("Uygula", type="primary", width="stretch")
        if not _submitted:
            return
        _days, _invalid = parse_holiday_days(_days_text, len(sutunlar))
        if _invalid:
            st.session_state.quick_entry_notice = (
                "warning",
                f"⚠️ Şu günler anlaşılmadı: {', '.join(_invalid)}. Günleri 1-{len(sutunlar)} "
                "arasında tek gün (5) ya da aralık (3-9) olarak yazın; hiçbir şey değişmedi.",
            )
            st.rerun()
        if not _days or _person not in st.session_state.pref_df.index:
            st.session_state.quick_entry_notice = ("info", "Gün yazılmadı; hiçbir şey değişmedi.")
            st.rerun()
        save_undo_state()
        _changed = apply_quick_entry(
            st.session_state.pref_df, _person, _days, QUICK_ENTRY_TYPES[_kind], gun_detaylari,
        )
        _clear_quick_entry_days()
        st.session_state.quick_entry_notice = (
            "success",
            f"✅ {_person}: {_changed} gün {_kind.replace(' (sil)', '')} olarak işaretlendi. "
            "Beğenmezseniz ↩️ Geri ile alabilirsiniz.",
        )
        st.rerun()
