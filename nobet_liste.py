# -*- coding: utf-8 -*-
"""Günlük Liste sekmesi: görev yeri döndürme ve görev yerlerini elle düzenleme."""
import pandas as pd
import streamlit as st

from nobet_core import apply_role_edits, ignored_role_edits
from nobet_durum import save_undo_state


def render_list_tab(rows_liste, liste_rows, liste_roles, role_names, isimler, sutunlar, edited_df):
    # ── Günlük Liste — görev yeri değişimi ───────────────────────────────────────
    st.caption("Her satırdaki 🔄 butona tıklayarak o günün görev yerlerini döngüsel olarak değiştirebilirsiniz. Değişiklik Geri Al / İleri Al ile geri alınabilir.")
    if rows_liste:
        _gl_role_names = role_names
        _gl_cols_cfg = {"Tarih": st.column_config.TextColumn("Tarih", disabled=True)}
        for _glrn in liste_roles:
            _gl_cols_cfg[_glrn] = st.column_config.TextColumn(_glrn, disabled=True)
        _gl_cols_cfg["🔄"] = st.column_config.CheckboxColumn("Değiştir", default=False)
        _gl_display_rows = []
        for _glrow in liste_rows:
            _gl_display_rows.append({
                "Tarih": _glrow.get("Tarih", ""),
                **{rn: _glrow.get(rn, "-") for rn in liste_roles},
                "🔄": False,
            })
        _gl_df = pd.DataFrame(_gl_display_rows)
        _gl_edited = st.data_editor(
            _gl_df,
            column_config=_gl_cols_cfg,
            width="stretch",
            hide_index=True,
            key="gunluk_liste_swap_editor",
            num_rows="fixed",
        )
        if _gl_edited is not None:
            _gl_swap_rows = _gl_edited[_gl_edited["🔄"] == True]
            if not _gl_swap_rows.empty:
                _gl_swap_idx = _gl_swap_rows.index[0]
                _gl_swap_row = rows_liste[_gl_swap_idx]
                _gl_day_people = [_gl_swap_row.get(rn, "-") for rn in _gl_role_names]
                _gl_day_people = [p for p in _gl_day_people if p and p != "-" and p in isimler]
                if len(_gl_day_people) >= 2:
                    _gl_tarih_str = _gl_swap_row.get("Tarih", "")
                    try:
                        _gl_dn = int(str(_gl_tarih_str).split(".")[0])
                    except Exception:
                        _gl_dn = _gl_swap_idx + 1
                    st.session_state['liste_swap_day'] = _gl_dn
                    st.rerun()
    else:
        st.info("Önce simülasyon çalıştırın veya nöbet ataması yapın.")

    st.divider()

    # ── Görev Yeri Düzenleme ─────────────────────────────────────────────────────
    with st.expander("✏️ Görev Yerlerini Düzenle (Gelişmiş)", expanded=False):
        st.caption("Görev sütunlarındaki isimleri değiştirerek rol atamalarını düzenleyebilirsiniz. Sadece o güne atanmış kişiler geçerlidir.")
        if rows_liste:
            _edit_col_cfg = {"Tarih": st.column_config.TextColumn("Tarih", disabled=True)}
            for _rn in role_names:
                _edit_col_cfg[_rn] = st.column_config.TextColumn(_rn, max_chars=50)
            _role_edit_df = pd.DataFrame([
                {k: v for k, v in r.items() if k == "Tarih" or k in role_names}
                for r in rows_liste
            ])
            for _msg in st.session_state.pop("role_edit_notice", []):
                st.warning(_msg)
            _edited_roles = st.data_editor(
                _role_edit_df,
                column_config=_edit_col_cfg,
                width="stretch",
                hide_index=True,
                # Her düzenlemeden sonra sürüm artar; uygulanamayan yazım tabloda kalmaz.
                key=f"role_editor_table_{st.session_state.get('role_editor_version', 0)}",
                num_rows="fixed",
            )
            if _edited_roles is not None:
                _role_changed = False
                _role_ignored = []
                _new_rows_edit = []
                # to_dict: görev adında boşluk/tire olsa da sütun okunur (itertuples
                # bu adları değiştirdiği için düzenlemeler sessizce kayboluyordu).
                for _ri, (_erow, _orow) in enumerate(zip(_edited_roles.to_dict("records"), rows_liste)):
                    _day_col = sutunlar[_ri] if _ri < len(sutunlar) else None
                    _assigned_today = (
                        edited_df.index[edited_df[_day_col]].tolist()
                        if _day_col and _day_col in edited_df.columns else []
                    )
                    # Başka birimdeki kişiyi yazınca ikisi yer değiştirir (çift kayıt olmaz).
                    _new_r, _row_changed = apply_role_edits(_orow, _erow, role_names, _assigned_today)
                    _role_changed = _role_changed or _row_changed
                    _new_rows_edit.append(_new_r)
                    for _role, _value in ignored_role_edits(_orow, _erow, role_names, _assigned_today):
                        _day_label = _orow.get("Tarih", "")
                        _role_ignored.append(
                            f"{_day_label} · {_role}: **{_value}** o gün nöbetçi değil."
                            if _value else
                            f"{_day_label} · {_role}: görev boş bırakılamaz; kişiyi çıkarmak için tablodan nöbetini kaldırın."
                        )
                if _role_ignored:
                    st.session_state.role_edit_notice = ["⚠️ Uygulanmadı — " + _m for _m in _role_ignored[:5]]
                if _role_changed:
                    save_undo_state()
                    st.session_state.cached_rows_liste = _new_rows_edit
                    st.session_state.should_regenerate_assignments = False
                if _role_changed or _role_ignored:
                    # Tablo bir sonraki çizimde güncel satırlarla temiz başlasın.
                    st.session_state.role_editor_version = st.session_state.get("role_editor_version", 0) + 1
                    st.rerun()
        else:
            st.info("Önce simülasyon çalıştırın veya nöbet ataması yapın.")
