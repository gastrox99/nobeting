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
    redo_history_state,
    undo_history_state,
    update_locked_cells,
    use_compact_schedule_editor,
)
from nobet_durum import current_state_snapshot, restore_snapshot, save_undo_state
from nobet_stil import grid_css


def render_grid_tab(isimler, sutunlar, gun_detaylari, yil, ay, today_date, ui,
                    person_colors, conflict_cells, queue_assignment_warnings):
    # Mode selector and paint color
    mode_cols = st.columns([2, 3])
    with mode_cols[0]:
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
            color_cols = st.columns(len(color_info))
            for i, (val, label) in enumerate(color_info):
                with color_cols[i]:
                    btn_type = "primary" if st.session_state.paint_color == val else "secondary"
                    if st.button(label, key=f"color_{val}", type=btn_type):
                        st.session_state.paint_color = val
                        st.rerun()
        else:
            st.info("Tıklayarak nöbet ekle/kaldır. Elle atadığın nöbetler (sarı çerçeve) simülasyonda korunur.")

    # Quick actions row
    action_cols = st.columns(5)
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
        st.warning("⚠️ Çizelge ve tüm tercihler silinecek. Devam etmek istediğinizden emin misiniz?")
        _reset_cols = st.columns(2)
        with _reset_cols[0]:
            if st.button("🗑️ Evet, çizelgeyi sıfırla", type="primary", width="stretch"):
                save_undo_state()
                st.session_state.pref_df = pd.DataFrame(0, index=isimler, columns=sutunlar)
                st.session_state.schedule_bool = pd.DataFrame(False, index=isimler, columns=sutunlar)
                st.session_state.locked_cells = set()
                st.session_state.hidden_people = {}
                st.session_state.cached_rows_liste = None
                st.session_state.should_regenerate_assignments = True
                st.session_state.reset_pending = False
                st.rerun()
        with _reset_cols[1]:
            if st.button("Vazgeç", width="stretch"):
                st.session_state.reset_pending = False
                st.rerun()

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
    name_col_px = name_column_width(isimler)

    st.markdown(grid_css(ui, name_col_px, col_bg_css), unsafe_allow_html=True)

    grid_key = "preference-grid" if st.session_state.edit_mode == "tercih" else "schedule-grid"
    grid_container = st.container(key=grid_key, border=False)

    tr_gunler_short = {0:"Pzt", 1:"Sal", 2:"Çar", 3:"Per", 4:"Cum", 5:"Cmt", 6:"Paz"}
    _compact_editor = use_compact_schedule_editor(len(isimler), len(sutunlar))

    def _draw_day_header():
        header_cols = grid_container.columns([2] + [1] * len(sutunlar), gap=None)
        with header_cols[0]:
            st.markdown(f"<div style='font-size:13px;font-weight:700;color:{ui['muted']};padding:2px 4px;'>İSİM&nbsp;&nbsp;#</div>", unsafe_allow_html=True)
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
        _locked_now = active_locks(st.session_state.get("locked_cells"), st.session_state.schedule_bool)
        for person in isimler:
            row_cols = grid_container.columns([2] + [1] * len(sutunlar), gap=None)
            with row_cols[0]:
                count = int(st.session_state.schedule_bool.loc[person].sum()) if person in st.session_state.schedule_bool.index else 0
                p_lim = _person_limits.get(person, {})
                min_l = p_lim.get('min', 0)
                max_l = p_lim.get('max', 999)
                badge_bg = "#dc2626" if (count < min_l or (max_l < 999 and count > max_l)) else "#16a34a"
                pc = person_colors.get(person, "#e2e8f0")
                st.markdown(
                    f"<div style='font-size:13px;white-space:nowrap;padding:2px 4px;color:{ui['name_text']};"
                    f"display:flex;flex-wrap:nowrap;align-items:center;gap:4px;min-height:30px;overflow:hidden;'"
                    f" title='{escape_html(person)}'>"
                    f"<span style='flex-shrink:0;width:11px;height:11px;border-radius:50%;"
                    f"background:{pc};border:1px solid rgba(0,0,0,0.12);'></span>"
                    f"<b style='min-width:0;overflow:hidden;text-overflow:ellipsis;'>{escape_html(person)}</b>"
                    f"<span style='flex-shrink:0;background:{badge_bg};color:#fff;border-radius:10px;padding:1px 7px;font-size:11px;font-weight:700;'>{count}</span>"
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
        else:
            _bg, _fg, _border = _state["color"], "#ffffff", "none"
        return (
            f"<span style='display:inline-flex;align-items:center;gap:6px;margin:2px 12px 2px 0;'>"
            f"<span class='preference-legend-swatch' style='background:{_bg};color:{_fg} !important;border:{_border};'>"
            f"{_state['symbol']}</span>"
            f"<span style='color:{ui['legend_text']};font-size:12px;'>{_label}</span></span>"
        )

    _legend_states = get_preference_grid_legend()
    _on = [x for x in _legend_states if not x.get("soft")]
    _off = [x for x in _legend_states if x.get("soft")]
    _row_label = f"style='color:{ui['muted']};font-size:11px;font-weight:700;letter-spacing:.04em;margin-right:10px;'"
    legend_container.markdown(
        "<div style='display:flex;flex-direction:column;gap:4px;margin-top:6px;'>"
        f"<div style='display:flex;flex-wrap:wrap;align-items:center;'><span {_row_label}>NÖBETLİ</span>"
        + "".join(_legend_item(x, x["description"].replace("Atandı + ", "")) for x in _on)
        + "</div>"
        f"<div style='display:flex;flex-wrap:wrap;align-items:center;'><span {_row_label}>NÖBETSİZ</span>"
        + "".join(_legend_item(x, x["description"]) for x in _off)
        + "</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown('<p class="mobile-hint" style="color:#888;font-size:12px;margin:4px 0;">📱 Mobilde yana kaydırın →</p>', unsafe_allow_html=True)
