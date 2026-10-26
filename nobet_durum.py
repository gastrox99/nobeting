# -*- coding: utf-8 -*-
"""Oturum durumu yardımcıları: geri al/ileri al, görev yeri döndürme, ad değişikliği."""
import pandas as pd
import streamlit as st

from db import list_recent_errors
from nobet_core import align_rows_to_roles, record_renames, rename_people_in_rows


def current_state_snapshot():
    """Geri al/yinele için hem nöbet hem tercih hem de görev yeri atamalarını kaydeder."""
    return {
        'schedule': st.session_state.schedule_bool.copy() if 'schedule_bool' in st.session_state else None,
        'pref': st.session_state.pref_df.copy() if 'pref_df' in st.session_state else None,
        'rows_liste': list(st.session_state.cached_rows_liste) if st.session_state.get('cached_rows_liste') else None,
        'role_names': list(st.session_state.cached_role_names) if st.session_state.get('cached_role_names') else None,
        'locked': set(st.session_state.get('locked_cells', set())),
        'hidden': dict(st.session_state.get('hidden_people') or {}),
    }

def save_undo_state():
    """Geri al için mevcut durumu (çizelge, tercihler, görevler, kilitler) saklar."""
    if 'undo_history' not in st.session_state:
        st.session_state.undo_history = []
    if 'redo_history' not in st.session_state:
        st.session_state.redo_history = []
    
    # Limit history to 10 states
    if len(st.session_state.undo_history) >= 10:
        st.session_state.undo_history.pop(0)
    
    st.session_state.undo_history.append(current_state_snapshot())
    st.session_state.redo_history = []  # Clear redo on new action

def restore_snapshot(snap, sutunlar, gun_detaylari):
    """Geri al / ileri al ile gelen durumu uygular."""
    if not isinstance(snap, dict):
        st.session_state.schedule_bool = snap
        st.session_state.cached_rows_liste = None
        st.session_state.should_regenerate_assignments = True
        return
    if snap.get('schedule') is not None:
        st.session_state.schedule_bool = snap['schedule']
    if snap.get('pref') is not None:
        st.session_state.pref_df = snap['pref']
    if snap.get('locked') is not None:
        st.session_state.locked_cells = set(snap['locked'])
    if snap.get('hidden'):
        # Birleştir: geri alınan adımdan sonra ekipten çıkarılanlar da kaybolmasın.
        st.session_state.hidden_people = {
            **snap['hidden'], **(st.session_state.get('hidden_people') or {})
        }
    if snap.get('rows_liste') is not None:
        _rows, _roles = snap['rows_liste'], list(snap.get('role_names') or [])
        _current_roles = list(st.session_state.get('rol_isimleri') or _roles)
        if _roles != _current_roles:
            # Görev adları o zamandan beri değiştiyse satırları güncel düzene taşı.
            _rows = align_rows_to_roles(_rows, _roles, _current_roles, st.session_state.schedule_bool,
                                        sutunlar, gun_detaylari)
            _roles = _current_roles
        st.session_state.cached_rows_liste = _rows
        st.session_state.cached_role_names = _roles
        st.session_state.should_regenerate_assignments = False
    else:
        st.session_state.cached_rows_liste = None
        st.session_state.should_regenerate_assignments = True


def rotate_roles_for_day(day_number):
    """O günün nöbetçilerini görev yerleri arasında döngüsel kaydırır.

    Takvimdeki 🔄 düğmesi bunu kullanır."""
    _rows = st.session_state.get('cached_rows_liste')
    if not _rows or not 1 <= day_number <= len(_rows):
        return False
    _roles = st.session_state.get('cached_role_names') or st.session_state.get('rol_isimleri', [])
    _team = set(st.session_state.get('isimler_cache', []))
    _row = dict(_rows[day_number - 1])
    _slots = [_r for _r in _roles if _row.get(_r) not in (None, "", "-") and _row.get(_r) in _team]
    if len(_slots) < 2:
        return False
    save_undo_state()
    _people = [_row[_r] for _r in _slots]
    _people = [_people[-1]] + _people[:-1]
    for _r, _p in zip(_slots, _people):
        _row[_r] = _p
    st.session_state.cached_rows_liste[day_number - 1] = _row
    st.session_state.should_regenerate_assignments = False
    st.toast(f"{day_number}. gün görev yerleri değiştirildi", icon="🔄")
    return True


def apply_renames(renames):
    """Yeniden adlandırılan kişilerin verisini yeni adlarına taşır.

    Diğer aylar veritabanında eski adla durur; ad değişikliği ayarlara
    kaydedilir ve o aylar açılırken güncel ada çevrilir.
    """
    st.session_state.name_aliases = record_renames(st.session_state.get("name_aliases"), renames)
    for _key in ("schedule_bool", "pref_df", "cached_role_schedule"):
        _df = st.session_state.get(_key)
        if isinstance(_df, pd.DataFrame):
            st.session_state[_key] = _df.rename(index=renames)
    st.session_state.locked_cells = {
        (renames.get(_p, _p), _c) for _p, _c in st.session_state.get("locked_cells", set())
    }
    if st.session_state.get("cached_rows_liste"):
        st.session_state.cached_rows_liste = rename_people_in_rows(st.session_state.cached_rows_liste, renames)
    _key = st.session_state.get("cached_key")
    if isinstance(_key, tuple) and len(_key) == 5:
        st.session_state.cached_key = _key[:2] + (tuple(renames.get(_p, _p) for _p in _key[2]),) + _key[3:]
    _ctx = st.session_state.get("prev_month_context")
    if _ctx:
        for _field in ("last_shift", "history_counts", "holiday_counts"):
            _ctx[_field] = {renames.get(_p, _p): _v for _p, _v in _ctx.get(_field, {}).items()}
        for _field in ("monthly_totals", "monthly_available", "monthly_holidays"):
            _ctx[_field] = [
                {renames.get(_p, _p): _v for _p, _v in _month.items()} for _month in _ctx.get(_field, [])
            ]


def render_recent_errors():
    """Ayarlar'da «Son hatalar»: anahtar açılınca en yeni hataları (sürüm etiketiyle) listeler."""
    if not st.toggle("🩺 Son hataları göster", key="son_hatalar_ac"):
        return
    errors = list_recent_errors()
    if errors is None:
        st.warning("Hata kaydı okunamadı (veritabanına ulaşılamadı).")
    elif not errors:
        st.caption("Kayıtlı hata yok.")
    else:
        st.caption("En yeni 20 hata. Kayıt tutarken ya da yüklerken sorun yaşarsanız bu tabloyu yazılımcıya gösterin.")
        st.dataframe(
            [{"Zaman": e["created_tr"], "İşlem": e["action"], "Hata": e["message"], "Sürüm": e["version"]} for e in errors],
            hide_index=True, width="stretch",
        )
