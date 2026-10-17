# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import random
import calendar
import re
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import date
import time
import os
import zlib
# Streamlit Cloud yeni kodu çektiğinde db / nobet_core modüllerinin eski
# hâli hafızada kalabiliyor ve yeni fonksiyonlar "ImportError" veriyor.
# Böyle olursa modülleri tazeleyip yeniden dene (Reboot gerekmesin).
import importlib
import db as _db_module
import nobet_core as _core_module


def _fresh_modules():
    for _module in (_db_module, _core_module):
        importlib.reload(_module)


try:
    from db import (
        init_db, save_schedule, load_schedule,
        save_app_settings, load_app_settings, claim_legacy_data, get_last_error,
        is_database_configured,
        create_share_link, get_share_link, resolve_share_token,
        delete_all_share_links, list_shared_months,
        load_auto_saved_assignments, export_owner_data, restore_owner_data,
        save_backup_snapshot, list_backup_snapshots, load_backup_snapshot,
    )
    from nobet_core import (
        TR_AY_ADLARI,
        TR_GUN_KISA,
        build_person_ics,
        build_share_text,
        shared_default_month,
        shared_day_states,
        on_duty_summary,
        shared_updated_text,
        BACKUP_MAX_BYTES,
        EXCEL_IMPORT_MAX_BYTES,
        EXCEL_IMPORT_MAX_ROWS,
        SNAPSHOT_BEFORE_RESTORE,
        month_backup,
        snapshot_time_text,
        SNAPSHOT_MANUAL,
        snapshot_label,
        backup_reminder,
        backup_summary,
        backup_to_bytes,
        build_backup,
        parse_backup,
        history_adjustment,
        history_special_adjustment,
        holiday_shift_counts,
        last_shift_offsets,
        schedule_totals,
        unavailable_day_counts,
        previous_period,
        special_shift_counts,
        build_gun_detaylari,
        create_print_html,
        escape_html,
        parse_forbidden_pairs,
        parse_person_limits_with_issues,
        unknown_names,
        pair_label,
        forbidden_pair_choices,
        forbidden_pairs_to_text,
        person_limits_to_text,
        validate_inputs,
        normalize_owner_email,
        apply_app_settings,
        apply_role_edits,
        ignored_role_edits,
        normalize_role_names,
        detect_renames,
        align_rows_to_roles,
        rename_people_in_rows,
        import_schedule_table,
        YEAR_MIN,
        YEAR_MAX,
        official_holidays,
        official_holidays_summary,
        official_half_days,
        required_work_hours,
        holidays_to_text,
        religious_holidays_known,
        active_locks,
        locks_to_json,
        update_locked_cells,
        assignment_warnings,
        name_column_width,
        collect_app_settings,
        persistence_fingerprint,
        build_schedule_analysis,
        preference_summary,
        changed_schedule_columns,
        find_person_role,
        get_preference_grid_legend,
        get_preference_grid_state,
        normalize_preference_grid,
        parse_schedule_import_date,
        parse_holiday_days,
        period_key,
        restore_auto_saved_schedule,
        run_scheduling_core,
        schedule_fingerprint,
        missing_shift_coverage,
        explain_missing_day,
        MISSING_DAY_FREE_LABEL,
        format_missing_day_reason,
        synchronize_changed_role_rows,
        switch_schedule_period,
        use_compact_schedule_editor,
        build_limit_violation_messages,
        undo_history_state,
        redo_history_state,
        update_persistence_feedback,
        add_overflow_roles,
        as_bool_frame,
        holiday_text_for_period,
        default_holiday_text,
        prune_default_holidays,
        now_tr,
        record_renames,
        resolve_alias,
        stash_removed_people,
        take_back_people,
        drop_unknown_people_from_rows,
    )
except ImportError:
    _fresh_modules()
    from db import (
        init_db, save_schedule, load_schedule,
        save_app_settings, load_app_settings, claim_legacy_data, get_last_error,
        is_database_configured,
        create_share_link, get_share_link, resolve_share_token,
        delete_all_share_links, list_shared_months,
        load_auto_saved_assignments, export_owner_data, restore_owner_data,
        save_backup_snapshot, list_backup_snapshots, load_backup_snapshot,
    )
    from nobet_core import (
        TR_AY_ADLARI,
        TR_GUN_KISA,
        build_person_ics,
        build_share_text,
        shared_default_month,
        shared_day_states,
        on_duty_summary,
        shared_updated_text,
        BACKUP_MAX_BYTES,
        EXCEL_IMPORT_MAX_BYTES,
        EXCEL_IMPORT_MAX_ROWS,
        SNAPSHOT_BEFORE_RESTORE,
        month_backup,
        snapshot_time_text,
        SNAPSHOT_MANUAL,
        snapshot_label,
        backup_reminder,
        backup_summary,
        backup_to_bytes,
        build_backup,
        parse_backup,
        history_adjustment,
        history_special_adjustment,
        holiday_shift_counts,
        last_shift_offsets,
        schedule_totals,
        unavailable_day_counts,
        previous_period,
        special_shift_counts,
        build_gun_detaylari,
        create_print_html,
        escape_html,
        parse_forbidden_pairs,
        parse_person_limits_with_issues,
        unknown_names,
        pair_label,
        forbidden_pair_choices,
        forbidden_pairs_to_text,
        person_limits_to_text,
        validate_inputs,
        normalize_owner_email,
        apply_app_settings,
        apply_role_edits,
        ignored_role_edits,
        normalize_role_names,
        detect_renames,
        align_rows_to_roles,
        rename_people_in_rows,
        import_schedule_table,
        YEAR_MIN,
        YEAR_MAX,
        official_holidays,
        official_holidays_summary,
        official_half_days,
        required_work_hours,
        holidays_to_text,
        religious_holidays_known,
        active_locks,
        locks_to_json,
        update_locked_cells,
        assignment_warnings,
        name_column_width,
        collect_app_settings,
        persistence_fingerprint,
        build_schedule_analysis,
        preference_summary,
        changed_schedule_columns,
        find_person_role,
        get_preference_grid_legend,
        get_preference_grid_state,
        normalize_preference_grid,
        parse_schedule_import_date,
        parse_holiday_days,
        period_key,
        restore_auto_saved_schedule,
        run_scheduling_core,
        schedule_fingerprint,
        missing_shift_coverage,
        explain_missing_day,
        MISSING_DAY_FREE_LABEL,
        format_missing_day_reason,
        synchronize_changed_role_rows,
        switch_schedule_period,
        use_compact_schedule_editor,
        build_limit_violation_messages,
        undo_history_state,
        redo_history_state,
        update_persistence_feedback,
        add_overflow_roles,
        as_bool_frame,
        holiday_text_for_period,
        default_holiday_text,
        prune_default_holidays,
        now_tr,
        record_renames,
        resolve_alias,
        stash_removed_people,
        take_back_people,
        drop_unknown_people_from_rows,
    )
from streamlit_local_storage import LocalStorage

# Excel export
try:
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# Veritabanını sunucu başına bir kez hazırla; başarısızsa sonucu kalıcı
# saklama (eskiden "bağlanamadım" önbellekte kalıyor, Secrets'a adres
# eklense bile yeniden başlatmaya kadar uyarı sürüyordu). Her yeni
# oturumda bir kez yeniden denenir.
@st.cache_resource
def _db_init_state():
    return {"ready": False, "error": None}

_db_state = _db_init_state()
if not _db_state["ready"] and not st.session_state.get("_db_init_attempted"):
    st.session_state["_db_init_attempted"] = True
    _db_state["ready"] = init_db()
    _db_state["error"] = get_last_error()
_db_ready = _db_state["ready"]

# Sayfa Ayarları
st.set_page_config(page_title="Adil Nöbet", layout="wide")

# --- SALT OKUNUR PAYLAŞIM SAYFASI (?paylas=...) ---
# Ekip arkadaşları giriş yapmadan yalnızca listeyi görür.
def _rows_from_schedule(schedule, year, month):
    _rows = []
    _days = {}
    for _col in schedule.columns:
        try:
            _days[int(str(_col).split()[0])] = _col
        except (ValueError, IndexError):
            continue
    for _day in sorted(_days):
        _assigned = schedule.index[schedule[_days[_day]]].tolist()
        _weekday = TR_GUN_KISA[date(year, month, _day).weekday()]
        _rows.append({
            "Tarih": f"{_day:02d}.{month:02d}.{year} {_weekday}",
            "Nöbetçiler": ", ".join(_assigned) or "-",
        })
    return _rows


def _render_shared_view(token):
    st.markdown("<h1 style='text-align: center; font-size: 2.2rem;'>Nöbet Listesi</h1>", unsafe_allow_html=True)
    _target = resolve_share_token(token) if _db_ready else None
    if not _target:
        st.error("Bu paylaşım bağlantısı geçersiz ya da kapatılmış.")
        return
    _owner, _year, _month = _target
    _months = list_shared_months(_owner)
    _updated = {(_y, _m): _t for _y, _m, _t in _months}
    _today = now_tr().date()
    if (_year, _month) == (0, 0):
        # Kalıcı bağlantı: kayıtlı ayların hepsi gösterilir, bu ay önce açılır.
        _periods = list(_updated)
        if not _periods:
            st.info("Henüz paylaşılan bir nöbet listesi yok.")
            return
        _year, _month = st.selectbox(
            "Ay:", _periods,
            index=shared_default_month(_periods, _today),
            format_func=lambda _p: f"{TR_AY_ADLARI[_p[1]]} {_p[0]}",
        )
    else:
        st.subheader(f"{TR_AY_ADLARI[_month]} {_year}")
    _loaded = load_schedule(f"Otomatik_{_year}_{_month:02d}", _year, _month, owner=_owner)
    _rows = _loaded[4] if _loaded and len(_loaded) > 4 and _loaded[4] else None
    _schedule = _loaded[1] if _loaded else None
    if _schedule is not None:
        _schedule = as_bool_frame(_schedule)
    if _rows:
        _roles = []
        for _row in _rows:
            for _key in _row:
                if _key != "Tarih" and _key not in _roles:
                    _roles.append(_key)
        if _schedule is not None and not _schedule.empty:
            # Görev sayısından fazla atanan kişiler de listede görünsün.
            _day_columns = {}
            for _col in _schedule.columns:
                try:
                    _day_columns[int(str(_col).split()[0])] = _col
                except (ValueError, IndexError):
                    continue
            _row_columns = []
            for _row in _rows:
                _parsed = parse_schedule_import_date(_row.get("Tarih"))
                _row_columns.append(_day_columns.get(_parsed["day"]) if _parsed else None)
            _rows, _roles = add_overflow_roles(_rows, _roles, _schedule, _row_columns)
    elif _schedule is not None and not _schedule.empty:
        _rows = _rows_from_schedule(_schedule, _year, _month)
        _roles = ["Nöbetçiler"]
    else:
        st.info("Bu ay için henüz çizelge yok.")
        return
    _updated_text = shared_updated_text(_updated.get((_year, _month)))
    if _updated_text:
        st.caption(f"Son güncelleme: {_updated_text}")
    _on_duty = on_duty_summary(_rows, _roles, _year, _month, _today)
    if _on_duty:
        for _col, (_label, _text) in zip(st.columns(len(_on_duty)), _on_duty):
            with _col:
                st.markdown(
                    f"<div style='border:1px solid rgba(128,128,128,.35);border-radius:8px;padding:8px 12px;'>"
                    f"<div style='font-size:12px;opacity:.7;'>{escape_html(_label)} nöbetçi</div>"
                    f"<div style='font-weight:600;'>{escape_html(_text)}</div></div>",
                    unsafe_allow_html=True,
                )
        st.write("")
    _people = sorted({
        _row.get(_role) for _row in _rows for _role in _roles
        if _row.get(_role) not in (None, "", "-") and _role != "Nöbetçiler"
    })
    _selected = st.selectbox("Kendi nöbetlerini görmek için adını seç:", ["Tümü"] + _people) if _people else "Tümü"
    _states = shared_day_states(_rows, _year, _month, _today)
    _shown = [
        (_row, _state) for _row, _state in zip(_rows, _states)
        if _selected == "Tümü" or any(_row.get(_role) == _selected for _role in _roles)
    ]
    _table = pd.DataFrame(
        [{_k: _row.get(_k, "-") for _k in ["Tarih"] + _roles} for _row, _ in _shown],
        columns=["Tarih"] + _roles,
    )
    # Geçmiş günler soluk, bugün vurgulu görünür.
    _row_styles = {
        "past": "color: rgba(128,128,128,.55);",
        "today": "font-weight: 700; background-color: rgba(255,200,0,.18);",
        "future": "",
    }
    _shown_states = [_state for _, _state in _shown]
    st.dataframe(
        _table.style.apply(lambda _r: [_row_styles[_shown_states[_r.name]]] * len(_r), axis=1),
        hide_index=True, width="stretch",
    )
    if _selected != "Tümü":
        st.download_button(
            "📅 Nöbetlerimi takvime ekle",
            build_person_ics(_selected, _rows, _roles, _year, _month).encode("utf-8"),
            f"nobet_{_year}_{_month:02d}.ics", "text/calendar",
            on_click="ignore", width="stretch",
        )
    st.caption("Bu sayfa salt okunurdur ve liste sahibi değişiklik yaptıkça güncellenir.")


_share_token = st.query_params.get("paylas")
if _share_token:
    _render_shared_view(_share_token)
    st.stop()

# --- GOOGLE İLE GİRİŞ (isteğe bağlı) ---
# .streamlit/secrets.toml içinde [auth] ayarları varsa giriş zorunludur ve her
# kullanıcının verisi Google e-postasına bağlı saklanır. Yoksa uygulama eskisi
# gibi tek kullanıcılı çalışır.
def _auth_configured():
    try:
        _auth = st.secrets.get("auth")
    except Exception:
        return False
    return bool(
        _auth
        and _auth.get("client_id")
        and _auth.get("client_secret")
        and _auth.get("redirect_uri")
    )


def _current_user_email():
    try:
        return normalize_owner_email(st.user["email"])
    except Exception:
        return ""


AUTH_ENABLED = _auth_configured()
# Veritabanı varken [auth] ayarı silinir ya da bozulursa uygulama herkese açık
# tek kullanıcı moduna düşmesin. Girişsiz çalışmak bilerek istenirse
# GIRISSIZ_KULLANIM=1 ortam değişkeni verilir (ör. maymun testi).
if (
    not AUTH_ENABLED
    and is_database_configured()
    and os.getenv("GIRISSIZ_KULLANIM") != "1"
):
    st.markdown("<h1 style='text-align: center; font-size: 2.5rem;'>Nöbet Yönetimi</h1>", unsafe_allow_html=True)
    st.error(
        "🔒 Giriş ayarları eksik olduğu için uygulama kilitlendi. Yönetici "
        "Streamlit Secrets'taki [auth] bölümünü (client_id, client_secret, "
        "redirect_uri) kontrol etmeli."
    )
    st.stop()
if AUTH_ENABLED:
    if not st.user.is_logged_in:
        st.markdown("<h1 style='text-align: center; font-size: 2.5rem;'>Nöbet Yönetimi</h1>", unsafe_allow_html=True)
        st.info("Çizelgeleriniz Google hesabınıza bağlı olarak saklanır. Devam etmek için giriş yapın.")
        st.button("🔐 Google ile giriş yap", on_click=st.login, type="primary")
        st.stop()
    current_owner = _current_user_email()
    if not current_owner:
        st.error("Google hesabınızın e-posta adresi alınamadı. Çıkış yapıp tekrar deneyin.")
        st.button("Çıkış yap", on_click=st.logout)
        st.stop()
    # Girişten önce kaydedilmiş veriyi yönetici hesabına bir kez aktar.
    if (
        "legacy_claim_done" not in st.session_state
        and current_owner == normalize_owner_email(os.getenv("ADMIN_EMAIL", ""))
        and _db_ready
    ):
        st.session_state.legacy_claim_done = claim_legacy_data(current_owner)
else:
    current_owner = ""

# Bağlantı hatasının nedeni (sunucu adı içerebilir) yalnızca giriş yapmış
# kullanıcıya gösterilir; paylaşım ve giriş sayfalarında görünmez.
if not _db_ready:
    _db_reason = f" Sebep: {_db_state['error']}" if _db_state.get("error") else ""
    st.warning(
        "⚠️ Veritabanına bağlanılamadı. Kaydet/Yükle ve otomatik kayıt "
        f"özellikleri şu an çalışmayabilir.{_db_reason}"
    )

# --- Ekip ve ayarları veritabanından geri yükle (sayfa yenilemesinde) ---
if 'app_settings_loaded' not in st.session_state:
    st.session_state.app_settings_loaded = True
    _stored_settings = None
    # Okuma başarısızsa kayıtlı ayarların üzerine boş ayar yazılmasın.
    st.session_state.app_settings_load_failed = not _db_ready
    if _db_ready:
        try:
            _stored_settings = load_app_settings(owner=current_owner, raise_on_error=True)
        except Exception:
            st.session_state.app_settings_load_failed = True
            st.warning("⚠️ Kayıtlı ekip ve ayarlar okunamadı. Sayfayı yenileyin.")
    apply_app_settings(st.session_state, _stored_settings)
    # Resmi tatillerle aynı kayıtlar silinir; koddaki bayram düzeltmeleri yansısın.
    prune_default_holidays(st.session_state.get("holidays_by_period"))
    st.session_state.last_saved_settings = (
        collect_app_settings(st.session_state) if _stored_settings else None
    )

# --- localStorage for team list persistence (veritabanı yoksa yedek) ---
local_storage = LocalStorage()

# Load saved team from localStorage on first run
if 'localStorage_loaded' not in st.session_state:
    st.session_state.localStorage_loaded = True
    # Girişli modda tarayıcıdaki ekip başka bir kullanıcıya ait olabilir.
    saved_team = None if AUTH_ENABLED else local_storage.getItem("nobet_team")
    if saved_team and not st.session_state.get("isimler_text"):
        st.session_state.isimler_text = saved_team
    st.session_state.last_stored_team_text = saved_team or st.session_state.get("isimler_text", "")

# --- GLOBAL RESPONSIVE CSS ---
st.markdown("""
<style>
/* Keyed Streamlit containers keep day columns usable with horizontal scroll. */
.st-key-schedule-grid,
.st-key-preference-grid {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    padding-bottom: 8px;
}

/* Mobile hint - hidden on desktop by default */
.mobile-hint {
    display: none;
}

@media (max-width: 768px) {
    /* Show mobile hint only on mobile */
    .mobile-hint {
        display: block !important;
    }
    
    /* Settings panel columns - stack vertically */
    [data-testid="stExpander"] [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="stExpander"] :is([data-testid="column"],[data-testid="stColumn"]) {
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    
    /* Grid wrapper - enable scroll */
    .st-key-schedule-grid {
        max-width: 100vw;
        overflow-x: scroll !important;
    }
    .st-key-schedule-grid [data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        min-width: max-content;
    }
    .st-key-schedule-grid :is([data-testid="column"],[data-testid="stColumn"]) {
        flex-shrink: 0 !important;
        min-width: 44px !important;
    }
    .st-key-schedule-grid :is([data-testid="column"],[data-testid="stColumn"]):first-child {
        min-width: 112px !important;
    }
    
    /* Reduce text sizes */
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.4rem !important; }
    h3 { font-size: 1.2rem !important; }
    
    /* Make editor rows larger for better touch/readability */
    [data-testid="stDataEditor"] div {
        font-size: 14px !important;
    }
    
    /* Make expander title smaller */
    [data-testid="stExpander"] summary {
        font-size: 13px !important;
    }

    /* Keep preference cells at a comfortable touch size instead of
       shrinking them until the green/yellow/red states become unclear. */
    .st-key-preference-grid {
        max-width: 100vw;
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch;
        padding: 4px 0 12px;
    }
    .st-key-preference-grid [data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        min-width: max-content;
    }
    .st-key-preference-grid :is([data-testid="column"],[data-testid="stColumn"]) {
        flex-shrink: 0 !important;
        min-width: 44px !important;
    }
    .st-key-preference-grid :is([data-testid="column"],[data-testid="stColumn"]):first-child {
        min-width: 112px !important;
    }
}

/* Tablet adjustments */
@media (max-width: 1024px) and (min-width: 769px) {
    [data-testid="stExpander"] :is([data-testid="column"],[data-testid="stColumn"]) {
        flex: 1 1 45% !important;
        min-width: 45% !important;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-size: 2.5rem;'>Nöbet Yönetimi</h1>", unsafe_allow_html=True)
if AUTH_ENABLED:
    _user_cols = st.columns([5, 1])
    with _user_cols[0]:
        st.caption(f"👤 `{current_owner}`")
    with _user_cols[1]:
        st.button("Çıkış yap", on_click=st.logout, width="stretch")

# --- YARDIMCI FONKSİYONLAR ---
# escape_html, validate_inputs ve create_print_html nobet_core'dan gelir.
def convert_df_to_png(df):
    fig, ax = plt.subplots(figsize=(8, len(df) * 0.4 + 1))
    try:
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.2)

        if "Tarih" in df.columns:
            idx = df.columns.get_loc("Tarih")
            for i, row_data in enumerate(df['Tarih']):
                color = "white"
                if "Cmt" in row_data or "Paz" in row_data: color = "#dbeafe"
                for j in range(len(df.columns)):
                    table[i+1, j].set_facecolor(color)
                    if j == idx: table[i+1, j].set_text_props(ha='left')

        for j in range(len(df.columns)):
            table[0, j].set_facecolor("#dddddd")
            table[0, j].set_text_props(weight='bold')

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        return buf.getvalue()
    finally:
        plt.close(fig)

def convert_df_to_excel(df_liste, df_stats_load, df_stats_finance, yil=None):
    """Convert dataframes to Excel with multiple sheets.

    Include the year in exported schedule labels when it is known so a later
    import can reject a schedule from the same month in the wrong year.
    """
    output = BytesIO()
    _excel_list = df_liste.copy()
    if yil is not None and "Tarih" in _excel_list.columns:
        def _add_year_to_date_label(value):
            _parsed = parse_schedule_import_date(value)
            if not _parsed or _parsed["month"] is None or _parsed["year"] is not None:
                return value
            _match = re.match(r"^(\d{1,2}\s+[^\W\d_]+)(.*)$", str(value).strip())
            return f"{_match.group(1)} {yil}{_match.group(2)}" if _match else value

        _excel_list["Tarih"] = _excel_list["Tarih"].map(_add_year_to_date_label)

    # "=" ile başlayan isim/görev adı Excel'de formül olarak çalışmasın.
    with pd.ExcelWriter(output, engine='xlsxwriter',
                        engine_kwargs={"options": {"strings_to_formulas": False}}) as writer:
        _excel_list.to_excel(writer, sheet_name='Günlük Liste', index=False)
        df_stats_load.to_excel(writer, sheet_name='Nöbet Yükü')
        df_stats_finance.to_excel(writer, sheet_name='Ücret Özeti')
        
        def _xl_col(n):
            s = ""
            while n >= 0:
                s = chr(65 + n % 26) + s
                n = n // 26 - 1
            return s
        n_cols = max(
            df_liste.shape[1],
            df_stats_load.shape[1] + 1,
            df_stats_finance.shape[1] + 1,
        )
        last_col = _xl_col(n_cols + 2)
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column(f'A:{last_col}', 15)
    
    return output.getvalue()

def _current_state_snapshot():
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
    
    st.session_state.undo_history.append(_current_state_snapshot())
    st.session_state.redo_history = []  # Clear redo on new action

def _restore_snapshot(snap):
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


def _rotate_roles_for_day(day_number):
    """O günün nöbetçilerini görev yerleri arasında döngüsel kaydırır.

    Takvimdeki ve Günlük Liste'deki 🔄 düğmeleri bunu kullanır."""
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


# --- ANA ALGORİTMA: nobet_core.run_scheduling_core kullanır ---
def run_scheduling_algorithm_v98(isimler, sutunlar, df_unwanted_bool, gun_detaylari, min_bosluk, kisi_sayisi, forbidden_pairs=None, person_limits=None, df_preferred=None, initial_last_shift=None, history_special=None, fixed_assignments=None, history_total=None, history_holiday=None):
    progress_bar = st.progress(0)
    def _progress(n):
        if n % 10 == 0:
            progress_bar.progress(n)
    try:
        best_schedule, _ = run_scheduling_core(
            isimler, sutunlar, df_unwanted_bool, gun_detaylari,
            min_bosluk, kisi_sayisi, forbidden_pairs, person_limits, df_preferred,
            simulation_count=100, progress_callback=_progress,
            initial_last_shift=initial_last_shift, history_special=history_special,
            fixed_assignments=fixed_assignments,
            history_total=history_total, history_holiday=history_holiday,
        )
    except ValueError as exc:
        st.error(f"🚨 Simülasyon başlatılamadı: {exc}")
        return None
    finally:
        progress_bar.empty()

    if best_schedule is None:
        st.error("🚨 Simülasyon sonuç üretemedi. Mevcut çizelge değiştirilmedi.")
        return None

    return best_schedule


def _apply_renames(renames):
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


# --- AYARLAR (Ana Sayfada Açılır Panel) ---
# Check if team exists to determine if settings should be expanded
prev_isimler = st.session_state.get('isimler_cache', [])
settings_expanded = len(prev_isimler) == 0

with st.expander("⚙️ Ayarlar", expanded=settings_expanded):
    set_col1, set_col2, set_col3 = st.columns([2, 1, 1])
    
    with set_col1:
        # Ayar kutularının sabit bir "key"i var. Eskiden yalnızca value= ile
        # kurulduklarından her düzenlemede kutu yeniden yaratılıyor ve art arda
        # yapılan iki düzenlemeden ikincisi sessizce geri dönüyordu.
        isimler_input = st.text_area(
            "👥 Ekip (virgülle ayırın):",
            value=st.session_state.get("isimler_text", ""),
            key="w_isimler_text",
            height=100,
            placeholder="Ali, Ayşe, Mehmet, Fatma"
        )
        st.session_state.isimler_text = isimler_input
        _raw_isimler = [x.strip() for x in isimler_input.split(",") if x.strip()]
        # Aynı isim iki kez yazılırsa tablo çöküyordu; her ismi bir kez kullan.
        isimler = list(dict.fromkeys(_raw_isimler))
        if len(isimler) < len(_raw_isimler):
            _dupes = sorted({x for x in _raw_isimler if _raw_isimler.count(x) > 1})
            st.warning(f"⚠️ Birden fazla yazılan isim(ler) bir kez kullanıldı: **{', '.join(_dupes)}**")
        # Bir ismin yazımı düzeltilince o kişinin nöbetleri/tercihleri silinmesin.
        _renames = detect_renames(prev_isimler, isimler)
        if _renames:
            _apply_renames(_renames)
            st.toast("İsim değişikliği uygulandı: " + ", ".join(f"{a} → {b}" for a, b in _renames.items()), icon="✏️")
        st.session_state.isimler_cache = isimler
        
        # Tarayıcı deposuna yalnızca ekip listesi değiştiğinde yaz.
        if st.session_state.get("last_stored_team_text") != isimler_input:
            local_storage.setItem("nobet_team", isimler_input)
            st.session_state.last_stored_team_text = isimler_input
    
    with set_col2:
        _now = now_tr()
        if _now.month == 12:
            current_year = _now.year + 1
            current_month = 1
        else:
            current_year = _now.year
            current_month = _now.month + 1
        yil = st.number_input("📅 Yıl", YEAR_MIN, YEAR_MAX, value=st.session_state.get('yil_val', current_year), key='yil_input_direct')
        st.session_state.yil_val = yil
        
        ay_options = list(range(1, 13))
        ay = st.selectbox("📆 Ay", ay_options, index=ay_options.index(st.session_state.get('ay_val', current_month)), key='ay_input_direct')
        st.session_state.ay_val = ay
        
        gun_sayisi = calendar.monthrange(yil, ay)[1]
        kişi_sayısı = st.slider("👤 Nöbet Başına Kişi:", 1, 5, value=st.session_state.get('kisi_sayisi_val', 2), key='kisi_slider_direct')
        st.session_state.kisi_sayisi_val = kişi_sayısı
    
    with set_col3:
        min_bosluk = st.slider("⏸️ Dinlenme (gün):", 0, 3, st.session_state.get("min_bosluk_val", 1), key="w_min_bosluk_val")
        st.session_state.min_bosluk_val = min_bosluk
        # Tatiller her ay için ayrı saklanır ("2026-10": "29"). Hiç girilmemiş
        # bir ayda resmi tatiller kendiliğinden gelir; silinirse boş kalır.
        _holidays_by_period = st.session_state.setdefault("holidays_by_period", {})
        _holiday_period = period_key(yil, ay)
        _official = official_holidays(yil, ay)
        holiday_input = st.text_input(
            "🎉 Tatiller (bu ay):",
            value=holiday_text_for_period(_holidays_by_period, yil, ay),
            key=f"w_holiday_text_{_holiday_period}",
            placeholder="1,2,5-10,23",
        )
        # Yalnızca elle değiştirilen tatiller saklanır; varsayılan resmi
        # tatiller her açılışta güncel listeden gelir.
        if holiday_input == default_holiday_text(yil, ay):
            _holidays_by_period.pop(_holiday_period, None)
        else:
            _holidays_by_period[_holiday_period] = holiday_input
        if _official:
            st.caption("Resmi tatiller: " + ", ".join(official_holidays_summary(yil, ay)))
        if not religious_holidays_known(yil):
            st.caption(f"{yil} dini bayram tarihleri tanımlı değil; varsa elle ekleyin.")
        tatil_gunleri, invalid_holiday_parts = parse_holiday_days(holiday_input, gun_sayisi)
        if invalid_holiday_parts:
            invalid_text = ", ".join(f"'{part}'" for part in invalid_holiday_parts)
            st.warning(
                f"⚠️ Şu tatil girdileri uygulanmadı: {invalid_text}. "
                f"Geçerli günler uygulanmaya devam ediyor. Günleri 1-{gun_sayisi} "
                "arasında tek gün (5) veya aralık (5-10) biçiminde yazın."
            )
        nobet_ucreti = st.number_input(
            "💰 Saat Ücreti (TL):",
            value=float(st.session_state.get("nobet_ucreti_val", 1.0)),
            key="w_nobet_ucreti_val",
        )
        st.session_state.nobet_ucreti_val = nobet_ucreti
    
    # Additional settings row
    st.divider()
    adv_col1, adv_col2, adv_col3 = st.columns(3)
    
    with adv_col1:
        role_names_input = st.text_input(
            "🏷️ Görev İsimleri:",
            value=st.session_state.get("role_names_text", ""),
            key="w_role_names_text",
            placeholder="Örn: AYB, GYB"
        )
        st.session_state.role_names_text = role_names_input
        
        # Aynı ad iki kez ya da "Tarih" yazılırsa kişiler listeden kayboluyordu.
        role_names, _role_adjusted = normalize_role_names(role_names_input, kişi_sayısı)
        if _role_adjusted:
            st.caption("Görev adları benzersiz olmalı: " + ", ".join(f"{a} → {b}" for a, b in _role_adjusted))
        st.session_state.rol_isimleri = role_names
        st.session_state.balance_history = st.checkbox(
            "⚖️ Önceki aylarla dengele",
            value=st.session_state.get("balance_history", True),
            key="w_balance_history",
            help=(
                "Son 3 ayda daha çok hafta sonu/tatil ya da toplamda daha çok nöbet tutanlara "
                "bu ay biraz daha az verilir. Bayram ve diğer tatil günleri son 12 ayda "
                "az tutana verilir."
            ),
        )
    
    # Ayarlar metin olarak saklanır; seçim kutuları o metinden kurulur. Anahtar
    # metne bağlı olduğundan geri yükleme gibi dış değişiklikler de görünür.
    with adv_col2:
        _fp_text = st.session_state.get("forbidden_pairs_text", "")
        _fp_choices = forbidden_pair_choices(isimler, parse_forbidden_pairs(_fp_text, isimler))
        _fp_selected = st.multiselect(
            "🚫 Birlikte Çalışamayan:",
            list(_fp_choices),
            default=[pair_label(_pair) for _pair in parse_forbidden_pairs(_fp_text, isimler)],
            key=f"w_forbidden_pairs_{zlib.crc32(_fp_text.encode('utf-8'))}",
            placeholder="Aynı gün nöbet tutmayacak ikiliyi seçin",
        )
        _fp_new_text = forbidden_pairs_to_text(_fp_choices[_label] for _label in _fp_selected)
        if _fp_new_text != _fp_text:
            st.session_state.forbidden_pairs_text = _fp_new_text
            st.rerun()
        forbidden_input = _fp_text

    with adv_col3:
        limits_text = st.session_state.get("limits_text", "")
        _lim_key = zlib.crc32(limits_text.encode("utf-8"))
        _lim_current = parse_person_limits_with_issues(limits_text)[0]
        _lim_people = st.multiselect(
            "📊 Kişisel Limitler:",
            isimler + unknown_names(_lim_current, isimler),
            default=list(_lim_current),
            key=f"w_limit_people_{_lim_key}",
            placeholder="Limiti olacak kişiyi seçin",
        )
        _lim_new = {}
        for _person in _lim_people:
            _old = _lim_current.get(_person, {"min": 0, "max": 999})
            _min_col, _max_col = st.columns(2)
            with _min_col:
                _min = st.number_input(
                    f"{_person}: en az", min_value=0, max_value=31, step=1,
                    value=min(31, _old["min"]), key=f"w_limit_min_{_person}_{_lim_key}",
                )
            with _max_col:
                # En çok, en azdan küçük olamaz; boş = sınırsız.
                _max = st.number_input(
                    f"{_person}: en çok", min_value=int(_min), max_value=31, step=1,
                    value=None if _old["max"] >= 999 else min(31, max(int(_min), _old["max"])),
                    key=f"w_limit_max_{_person}_{_lim_key}", placeholder="sınırsız",
                )
            _lim_new[_person] = {"min": int(_min), "max": 999 if _max is None else int(_max)}
        _lim_new_text = person_limits_to_text(_lim_new)
        if _lim_new_text != limits_text:
            st.session_state.limits_text = _lim_new_text
            st.rerun()

    # Yasak çiftler ve kişisel limitler (ayrıştırma nobet_core'da, test edilir)
    forbidden_pairs = parse_forbidden_pairs(forbidden_input, isimler)
    _fp_unknown = unknown_names([name for pair in forbidden_pairs for name in pair], isimler)
    st.session_state.forbidden_pairs = forbidden_pairs
    st.session_state.fp_unknown_names = _fp_unknown
    if _fp_unknown:
        st.warning(f"⚠️ Yasak çiftte ekip listesinde olmayan isim(ler): **{', '.join(_fp_unknown)}** — kısıtlama uygulanmayacak.")

    person_limits, _lim_invalid_lines, _lim_errors = parse_person_limits_with_issues(limits_text)
    _lim_unknown = unknown_names(person_limits, isimler)
    st.session_state.person_limits = person_limits
    st.session_state.lim_unknown_names = _lim_unknown
    if _lim_errors:
        st.error("❌ Kişisel limit sorunları:\n" + "\n".join(_lim_errors))
    if _lim_invalid_lines:
        st.warning(f"⚠️ {len(_lim_invalid_lines)} satır geçersiz format (doğru format: `İsim:min-max`) — atlandı.")
    if _lim_unknown:
        st.warning(f"⚠️ Kişisel limitte ekip listesinde olmayan isim(ler): **{', '.join(_lim_unknown)}** — limit uygulanmayacak.")

# Ekip ve ayarlar değiştiyse veritabanına yaz (sayfa yenilenince geri gelir).
_current_settings = collect_app_settings(st.session_state)
_nothing_saved_yet = st.session_state.get("last_saved_settings") is None
if (
    _db_ready
    and not st.session_state.get("app_settings_load_failed", False)
    and _current_settings != st.session_state.get("last_saved_settings")
    # İlk açılışta boş ekip kaydedilmesin (localStorage'dan gelecek ekip ezilmesin).
    and not (_nothing_saved_yet and not st.session_state.get("isimler_text", "").strip())
):
    if save_app_settings(_current_settings, owner=current_owner):
        st.session_state.last_saved_settings = _current_settings
        # Hazırlanmış yedek artık eski; yeniden hazırlanması gerekir.
        st.session_state.pop("backup_bytes", None)

# --- YEDEK: tüm aylar ve ayarlar tek dosyada; aynı dosyadan geri yükleme ---
def _mark_backup_downloaded():
    st.session_state.last_backup_at = now_tr().strftime("%Y-%m-%d %H:%M")
    st.session_state.pop("backup_bytes", None)


def _take_snapshot(reason, period=None):
    """Veritabanındaki veriyi uygulamada saklar; özet döner, veri yoksa None.

    ``period`` (yıl, ay) verilirse yalnızca o ayın çizelgesi o ayın tek
    yedeği olarak saklanır; verilmezse tüm veri (geri yükleme öncesi kopya).
    """
    _settings_sn, _schedules_sn = export_owner_data(owner=current_owner)
    _bk = build_backup(_settings_sn, _schedules_sn)
    if period is not None:
        _bk = month_backup(_bk, *period)
    if not _bk["schedules"] and not _bk.get("settings"):
        return None
    _summary_sn = backup_summary(_bk)
    _year_sn, _month_sn = period if period is not None else (0, 0)
    save_backup_snapshot(
        current_owner, _year_sn, _month_sn, now_tr().strftime("%Y-%m-%d %H:%M"), reason,
        _summary_sn, backup_to_bytes(_bk).decode("utf-8"),
    )
    for _key in ("backup_snapshots", "snapshot_select", "backup_snapshot_data"):
        st.session_state.pop(_key, None)
    return _summary_sn


def _run_requested_snapshot():
    """Yedekle tuşu: otomatik kayıt bittikten sonra çağrılır."""
    if not st.session_state.pop("snapshot_requested", False):
        return
    if st.session_state.get("persistence_error"):
        st.error("❌ Son değişiklik kaydedilemediği için yedek alınmadı. Tekrar deneyin.")
        return
    _period_sn = (yil, ay)
    try:
        _summary_sn = _take_snapshot(SNAPSHOT_MANUAL, _period_sn)
    except Exception:
        st.error("❌ Yedek alınamadı (veritabanına ulaşılamadı). Tekrar deneyin.")
        return
    _month_text = f"{TR_AY_ADLARI[ay]} {yil}"
    st.session_state.snapshot_notice = (
        f"{_month_text} yedeklendi; bu ayın önceki yedeğinin yerine geçti."
        if _summary_sn else f"{_month_text} için henüz kaydedilmiş çizelge yok."
    )
    st.rerun()


def _restore_backup(parsed, summary):
    """Geri yüklemeden önce mevcut hâlin kopyasını alır, sonra yedeği yükler."""
    try:
        _take_snapshot(SNAPSHOT_BEFORE_RESTORE)
    except Exception:
        st.error("❌ Mevcut hâlin kopyası alınamadığı için geri yükleme yapılmadı. Tekrar deneyin.")
        return
    try:
        restore_owner_data(parsed, owner=current_owner)
    except Exception:
        st.error("❌ Geri yükleme başarısız oldu; hiçbir kayıt değişmedi. Tekrar deneyin.")
        return
    # Ekrandaki eski durum yedeğin üzerine otomatik kaydedilmesin:
    # her şeyi veritabanından baştan yükle.
    _kept_backup_at = st.session_state.get("last_backup_at")
    st.session_state.clear()
    st.session_state.backup_restored_notice = summary
    st.session_state.backup_restored_last_at = _kept_backup_at
    st.rerun()


if _db_ready:
    _restored_notice = st.session_state.pop("backup_restored_notice", None)
    # Yedekteki "son indirme" tarihi, bu yedeği indirmeden önceki tarihtir.
    _kept_backup_at = st.session_state.pop("backup_restored_last_at", None)
    if _kept_backup_at and _kept_backup_at > str(st.session_state.get("last_backup_at") or ""):
        st.session_state.last_backup_at = _kept_backup_at
    # Bildirimler her zaman var olan bir kabın içinde: bildirim bir sonraki
    # tıklamada kaybolunca Yedek bölümü yer değiştirip kapanmasın.
    with st.container():
        if _restored_notice:
            # Onay kutuları işaretli kalmasın; ikinci bir geri yükleme kazara yapılmasın.
            st.session_state.snapshot_restore_confirm = False
            st.session_state.backup_restore_confirm = False
            st.success(
                f"✅ Yedek geri yüklendi: {_restored_notice}. "
                "Önceki hâl Yedek bölümünde \"Son geri yüklemeden önceki hâl\" olarak saklandı."
            )
        _snapshot_notice = st.session_state.pop("snapshot_notice", None)
        if _snapshot_notice:
            st.success(f"✅ {_snapshot_notice}")
            # Tuş sayfanın aşağısında; bildirim oradan da görünsün.
            st.toast(f"✅ {_snapshot_notice}")
    _backup_hint = backup_reminder(st.session_state.get("last_backup_at"))
    _backup_label = "💾 Yedek" + (f" — {_backup_hint} İndirmeniz önerilir." if _backup_hint else "")
    with st.expander(_backup_label, expanded=False):
        st.markdown("**Uygulamadaki yedekler**")
        st.caption(
            "Çizelge her değişiklikte zaten kaydediliyor. Ayı bitirince listenin üstündeki "
            "💾 Yedekle düğmesine basın: o ayın hâli saklanır. Her ayın tek yedeği olur; tekrar basarsanız eskisinin yerine geçer."
        )
        if "backup_snapshots" not in st.session_state:
            try:
                st.session_state.backup_snapshots = list_backup_snapshots(owner=current_owner)
            except Exception:
                st.session_state.backup_snapshots = None
        _snaps = st.session_state.backup_snapshots
        _month_snaps = [_i for _i in (_snaps or []) if _i.get("reason") == SNAPSHOT_MANUAL]
        _undo_snap = next((_i for _i in (_snaps or []) if _i.get("reason") == SNAPSHOT_BEFORE_RESTORE), None)
        _this_month_snap = next(
            (_i for _i in _month_snaps if (_i.get("year"), _i.get("month")) == (yil, ay)), None,
        )
        if not isimler:
            # Ekip boşsa sayfa aşağıda duruyor ve listenin üstündeki tuş
            # görünmüyor; yedek tuşu burada kalır ve yedeği hemen alır.
            if st.button(f"💾 {TR_AY_ADLARI[ay]} {yil} ayını yedekle", type="primary", key="snapshot_take_btn"):
                st.session_state.snapshot_requested = True
                _run_requested_snapshot()
        if _this_month_snap:
            st.caption(f"Bu ayın son yedeği: {snapshot_time_text(_this_month_snap)}")
        if _snaps is None:
            st.warning("Uygulamadaki yedekler okunamadı (veritabanına ulaşılamadı).")
            if st.button("Tekrar dene", key="snapshot_list_retry"):
                st.session_state.pop("backup_snapshots", None)
                st.rerun()
        elif _month_snaps or _undo_snap:
            _snap_by_id = {_item["id"]: _item for _item in _month_snaps}
            if _undo_snap:
                _snap_by_id[_undo_snap["id"]] = _undo_snap
            _snap_id = st.selectbox(
                "Yedek seçin", list(_snap_by_id), key="snapshot_select",
                format_func=lambda _i: (
                    f"Son geri yüklemeden önceki hâl · {snapshot_time_text(_snap_by_id[_i])}"
                    if _snap_by_id[_i].get("reason") == SNAPSHOT_BEFORE_RESTORE
                    else snapshot_label(_snap_by_id[_i])
                ),
            )
            _cached = st.session_state.get("backup_snapshot_data")
            if not _cached or _cached[0] != _snap_id:
                try:
                    _cached = (_snap_id, load_backup_snapshot(_snap_id, owner=current_owner))
                except Exception:
                    _cached = (_snap_id, None)
                st.session_state.backup_snapshot_data = _cached
            _snap_data = _cached[1]
            if _snap_data is None:
                st.warning("Bu yedek okunamadı. Sayfayı yenileyip tekrar deneyin.")
            else:
                _snap_item = _snap_by_id[_snap_id]
                if _snap_item.get("reason") == SNAPSHOT_MANUAL:
                    _snap_file = f"nobet_yedek_{int(_snap_item['year'])}_{int(_snap_item['month']):02d}.json"
                else:
                    _snap_file = f"nobet_yedek_geri_yukleme_oncesi_{now_tr():%Y-%m-%d}.json"
                _sn_col1, _sn_col2 = st.columns(2)
                with _sn_col1:
                    st.download_button(
                        "⬇️ Bu yedeği indir (.json)", _snap_data.encode("utf-8"),
                        _snap_file, "application/json",
                        key="snapshot_download_btn", width="stretch",
                    )
                with _sn_col2:
                    _snap_confirm = st.checkbox(
                        "Mevcut kayıtların üzerine yazılsın", key="snapshot_restore_confirm",
                    )
                    if st.button(
                        "♻️ Bu yedeğe dön", disabled=not _snap_confirm,
                        key="snapshot_restore_btn", width="stretch",
                    ):
                        _parsed_snap, _snap_errors = parse_backup(_snap_data.encode("utf-8"))
                        if _snap_errors:
                            st.error("❌ Bu yedek geri yüklenemedi:\n" + "\n".join(f"- {_e}" for _e in _snap_errors))
                        else:
                            _restore_backup(_parsed_snap, backup_summary(_parsed_snap))
            if _undo_snap:
                st.caption(
                    "Geri yüklemeden önce mevcut hâlin kopyası otomatik alınır "
                    "(\"Son geri yüklemeden önceki hâl\"); yanlışlıkla geri yüklerseniz ona dönebilirsiniz."
                )
        else:
            st.caption("Henüz uygulamada yedek yok.")

        st.divider()
        _bk_col1, _bk_col2 = st.columns(2)
        with _bk_col1:
            st.markdown("**Bilgisayara indir**")
            st.caption(
                "Bütün aylar (nöbetler, tercihler, görev yerleri) ve ekip/ayarlar tek dosyada. "
                "Ayda bir indirip bilgisayarınızda ya da Drive'da saklayın."
            )
            if st.session_state.get("last_backup_at"):
                st.caption(f"Son indirme: {st.session_state.last_backup_at}")
            if st.session_state.get("backup_bytes") is None:
                if st.button("📦 Yedeği hazırla", width="stretch"):
                    try:
                        _settings_bk, _schedules_bk = export_owner_data(owner=current_owner)
                    except Exception:
                        st.error("❌ Yedek hazırlanamadı (veritabanına ulaşılamadı). Tekrar deneyin.")
                    else:
                        st.session_state.backup_bytes = backup_to_bytes(build_backup(_settings_bk, _schedules_bk))
                        st.session_state.backup_prepared_at = now_tr().strftime("%H:%M")
                        st.rerun()
            else:
                st.caption(
                    f"Hazırlandı: {st.session_state.get('backup_prepared_at', '')}. "
                    "Sonra bir değişiklik yaparsanız yedek yeniden hazırlanır."
                )
                st.download_button(
                    "⬇️ Yedeği indir (.json)", st.session_state.backup_bytes,
                    f"nobet_yedek_{now_tr():%Y-%m-%d}.json", "application/json",
                    on_click=_mark_backup_downloaded, width="stretch",
                )
        with _bk_col2:
            st.markdown("**Dosyadan geri yükle**")
            st.caption("Yedekteki aylar mevcut kayıtların yerine geçer; yedekte olmayan aylara dokunulmaz.")
            _backup_file = st.file_uploader(
                "Yedek dosyası (.json)", type=["json"], key="backup_uploader", label_visibility="collapsed",
            )
            if _backup_file is not None:
                if _backup_file.size > BACKUP_MAX_BYTES:
                    _parsed_backup, _backup_errors = None, ["Dosya çok büyük (en fazla 20 MB)."]
                else:
                    _parsed_backup, _backup_errors = parse_backup(_backup_file.getvalue())
                if _backup_errors:
                    st.error("❌ Yedek yüklenemedi:\n" + "\n".join(f"- {_e}" for _e in _backup_errors))
                else:
                    _summary = backup_summary(_parsed_backup)
                    _created = _parsed_backup.get("created_at")
                    st.info(f"Yedek: {_summary}" + (f" · oluşturulma {_created}" if _created else ""))
                    if _parsed_backup.get("skipped"):
                        st.caption(
                            f"Uygulamanın kullanmadığı {_parsed_backup['skipped']} eski kayıt "
                            "geri yüklenmeyecek."
                        )
                    _confirm = st.checkbox("Mevcut kayıtların üzerine yazılsın", key="backup_restore_confirm")
                    if st.button("♻️ Geri yükle", type="primary", disabled=not _confirm, width="stretch"):
                        _restore_backup(_parsed_backup, _summary)

if not isimler: st.stop()

# --- GÜN VERİLERİ ---
tr_gunler = TR_GUN_KISA
gun_detaylari = build_gun_detaylari(yil, ay, gun_sayisi, tatil_gunleri)
sutunlar = list(gun_detaylari)

zorunlu_saat, arife_gunleri = required_work_hours(gun_detaylari, official_half_days(yil, ay))

# --- SESSION ---
def _save_outgoing_period(period, schedule, preferences, rows):
    _old_yil, _old_ay = period
    # Okunamayan bir dönemin kaydının üzerine yazma; değişmediyse tekrar yazma.
    if st.session_state.get("autosave_blocked_period") == tuple(period):
        return
    _locks = active_locks(st.session_state.get("locked_cells"), schedule)
    _hidden = st.session_state.get("hidden_people") or {}
    if persistence_fingerprint(schedule, preferences, rows, _locks, _hidden) == st.session_state.get("last_saved_fingerprint"):
        return
    st.session_state.pop("backup_bytes", None)
    save_schedule(
        f"Otomatik_{_old_yil}_{_old_ay:02d}", _old_yil, _old_ay, list(schedule.index),
        schedule, pref_df=preferences, rows_liste=rows, owner=current_owner,
        settings_dict={"locked": locks_to_json(_locks), "hidden": _hidden},
    )

# Ay/yıl değişince eski ayın çizelgesini yeni aya taşımak yerine kaydet ve
# yeni ayın kendi kaydını yükle (aksi halde otomatik kayıt o ayın üzerine yazar).
switch_schedule_period(st.session_state, (yil, ay), _save_outgoing_period)

if 'schedule_bool' not in st.session_state:
    # Sayfa yenilenmesinde otomatik geri yükle
    _auto_name = f"Otomatik_{yil}_{ay:02d}"

    def _load_auto_schedule(name, year, month):
        # Keep "no saved schedule yet" distinct from a database failure so the
        # first visit does not show a misleading error.
        return load_schedule(name, year, month, raise_on_error=True, owner=current_owner)

    def _on_auto_load_error(_error):
        # Kayıt okunamadıysa boş çizelgeyi kaydedip gerçek verinin üzerine yazma.
        st.session_state.autosave_blocked_period = (yil, ay)
        update_persistence_feedback(
            st.session_state,
            False,
            "⚠️ Kayıtlı çizelge okunamadı. Veriyi korumak için otomatik kayıt "
            "durduruldu; veritabanı bağlantısını kontrol edip sayfayı yenileyin.",
        )

    st.session_state.autosave_blocked_period = None
    if not restore_auto_saved_schedule(
        st.session_state,
        _load_auto_schedule,
        _auto_name,
        yil,
        ay,
        isimler,
        sutunlar,
        gun_detaylari,
        role_names,
        on_error=_on_auto_load_error,
        aliases=st.session_state.get("name_aliases"),
    ):
        st.session_state.schedule_bool = pd.DataFrame(False, index=isimler, columns=sutunlar)
    # Yüklenen durum zaten veritabanındakiyle aynı; ilk kayıt ilk değişiklikte yapılır.
    st.session_state.autosave_baseline_pending = True
else:
    # Ensure schedule matches current team and columns
    current_schedule = st.session_state.schedule_bool
    needs_update = False
    
    # Check if columns changed (month/year change)
    if list(current_schedule.columns) != sutunlar:
        needs_update = True
    # Check if team changed
    elif list(current_schedule.index) != isimler:
        needs_update = True
    
    if needs_update:
        # Ekipten çıkan kişilerin nöbet/tercihleri silinmesin; isim geri
        # yazılınca geri gelir (eskiden hemen kaydedilip kalıcı kayboluyordu).
        st.session_state.hidden_people, _stashed = stash_removed_people(
            st.session_state.get("hidden_people"), current_schedule,
            st.session_state.get("pref_df"), isimler,
        )
        if _stashed:
            st.toast(
                f"{', '.join(_stashed)} ekipten çıkarıldı. Bu aydaki nöbet ve tercihleri "
                "saklandı; ismi ekibe geri yazarsanız geri gelir.",
                icon="🗂️",
            )
        # Mevcut veriyi mümkün olduğunca koru — gün numarasıyla eşleştir
        # (kolon adı yıl değiştiğinde farklı olabilir: "1 Pzt" vs "1 Çar")
        new_schedule = pd.DataFrame(False, index=isimler, columns=sutunlar)
        old_day_map = {}
        for oc in current_schedule.columns:
            try:
                old_day_map[int(str(oc).split()[0])] = oc
            except (ValueError, IndexError):
                pass
        for person in isimler:
            if person in current_schedule.index:
                for col in sutunlar:
                    dnum = gun_detaylari[col]['day_num']
                    oc = old_day_map.get(dnum)
                    if oc:
                        new_schedule.at[person, col] = current_schedule.at[person, col] if oc == col else current_schedule.at[person, oc]
        st.session_state.schedule_bool = new_schedule
        _cached_key = st.session_state.get("cached_key")
        if (
            list(current_schedule.columns) == sutunlar
            and st.session_state.get("cached_rows_liste")
            and isinstance(_cached_key, tuple) and len(_cached_key) == 5
        ):
            # Yalnızca ekip değişti: elle düzenlenen görev yerleri korunur,
            # çıkanlar listeden düşer, gelenler boş görevlere yerleşir
            # (eskiden bütün ayın görev yerleri yeniden karılıyordu).
            st.session_state.cached_rows_liste = drop_unknown_people_from_rows(
                st.session_state.cached_rows_liste, isimler
            )
            st.session_state.cached_role_schedule = None
            st.session_state.cached_key = _cached_key[:2] + (tuple(isimler),) + _cached_key[3:]
        else:
            st.session_state.cached_rows_liste = None  # Reset cache

st.session_state.loaded_period = (yil, ay)

if 'inputs' not in st.session_state: st.session_state.inputs = {i: "" for i in isimler}
if 'cached_rows_liste' not in st.session_state: st.session_state.cached_rows_liste = None
if 'cached_role_schedule' not in st.session_state: st.session_state.cached_role_schedule = None
if 'cached_analysis_key' not in st.session_state: st.session_state.cached_analysis_key = None
if 'cached_analysis_tables' not in st.session_state: st.session_state.cached_analysis_tables = None
if 'cached_ayb_counts' not in st.session_state: st.session_state.cached_ayb_counts = None
if 'last_edited_hash' not in st.session_state: st.session_state.last_edited_hash = None
if 'should_regenerate_assignments' not in st.session_state: st.session_state.should_regenerate_assignments = False
if 'forbidden_pairs_text' not in st.session_state: st.session_state.forbidden_pairs_text = ""
if 'forbidden_pairs' not in st.session_state: st.session_state.forbidden_pairs = set()
if 'locked_cells' not in st.session_state: st.session_state.locked_cells = set()
if 'undo_history' not in st.session_state: st.session_state.undo_history = []
if 'redo_history' not in st.session_state: st.session_state.redo_history = []
if 'last_auto_save' not in st.session_state: st.session_state.last_auto_save = time.time()
if 'auto_save_failed' not in st.session_state: st.session_state.auto_save_failed = False
if 'persistence_error' not in st.session_state: st.session_state.persistence_error = None
if 'reset_pending' not in st.session_state: st.session_state.reset_pending = False
if 'simulation_success_pending' not in st.session_state: st.session_state.simulation_success_pending = False
if 'incomplete_simulation' not in st.session_state: st.session_state.incomplete_simulation = None
if 'preferences' not in st.session_state: st.session_state.preferences = {}
if 'person_preferences' not in st.session_state: st.session_state.person_preferences = {}
for i in isimler:
    if i not in st.session_state.person_preferences:
        st.session_state.person_preferences[i] = {}
for i in isimler: 
    if i not in st.session_state.inputs: st.session_state.inputs[i] = ""
# Ekipten çıkarılan üyelerin eski verilerini temizle
for _stale in [k for k in list(st.session_state.inputs.keys()) if k not in isimler]:
    del st.session_state.inputs[_stale]
for _stale in [k for k in list(st.session_state.person_preferences.keys()) if k not in isimler]:
    del st.session_state.person_preferences[_stale]

if st.session_state.pop("simulation_success_pending", False):
    st.toast("En adil sonuç seçildi.", icon="🧠")


def _queue_assignment_warnings(cells):
    """Yeni atanan hücreler için uyarıları sayfa yenilenince gösterilmek üzere sıraya koy."""
    _queued = []
    for _person, _column in cells:
        _queued.extend(assignment_warnings(
            st.session_state.schedule_bool,
            _person,
            _column,
            sutunlar,
            gun_detaylari,
            min_bosluk,
            st.session_state.get('person_limits', {}),
            st.session_state.get('pref_df'),
            st.session_state.get('forbidden_pairs', set()),
            kişi_sayısı,
            prev_last_shift,
        ))
    if _queued:
        st.session_state.pending_assignment_warnings = _queued


for _warning in st.session_state.pop("pending_assignment_warnings", [])[:5]:
    st.toast(_warning, icon="⚠️")

# --- KİŞİ RENK PALETİ (20 pastel ton) ---
PASTEL_COLORS = [
    "#AED6F1", "#A9DFBF", "#F9E79F", "#F5CBA7", "#D2B4DE",
    "#F1948A", "#A3E4D7", "#FAD7A0", "#ABEBC6", "#D7BDE2",
    "#7FB3D3", "#82E0AA", "#F8C471", "#F0A987", "#C39BD3",
    "#7DCEA0", "#85C1E9", "#F7DC6F", "#BB8FCE", "#F0B27A",
]
person_colors = {isim: PASTEL_COLORS[i % len(PASTEL_COLORS)] for i, isim in enumerate(isimler)}
today_date = now_tr().date()

# --- BİRLEŞİK TABLO ---

# Initialize preference DataFrame if needed
if 'pref_df' not in st.session_state:
    st.session_state.pref_df = pd.DataFrame(0, index=isimler, columns=sutunlar)
else:
    current_pref = st.session_state.pref_df
    if list(current_pref.columns) != sutunlar or list(current_pref.index) != isimler:
        new_pref = pd.DataFrame(0, index=isimler, columns=sutunlar)
        for person in isimler:
            if person in current_pref.index:
                for col in sutunlar:
                    if col in current_pref.columns:
                        new_pref.at[person, col] = current_pref.at[person, col]
        st.session_state.pref_df = new_pref

# Ekibe geri yazılan kişinin saklanan nöbet ve tercihleri geri gelir.
if any(_p in (st.session_state.get("hidden_people") or {}) for _p in isimler):
    st.session_state.hidden_people, _returned = take_back_people(
        st.session_state.hidden_people, st.session_state.schedule_bool,
        st.session_state.pref_df, isimler,
    )
    if _returned:
        st.toast(f"{', '.join(_returned)} ekibe geri döndü; bu aydaki nöbetleri geri yüklendi.", icon="↩️")

# Initialize edit mode
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = "tercih"

if 'selected_cal_day' not in st.session_state:
    st.session_state.selected_cal_day = None

# --- ÖNCEKİ AYLAR (ay geçişi dinlenme kuralı + aylar arası denge) ---
# Son 3 ay: hafta sonu/tatil ve toplam nöbet dengesi. Son 12 ay: bayram/tatil
# nöbetleri (bayramlar arasında 3 aydan uzun süre olur).
def _load_previous_context(year, month):
    _ctx = {
        "period": (year, month), "last_shift": {}, "history_counts": {}, "months": 0,
        "monthly_totals": [], "monthly_available": [], "monthly_holidays": [],
        "holiday_counts": {}, "holiday_months": 0,
    }
    if not _db_ready:
        return _ctx
    _holiday_texts = st.session_state.get("holidays_by_period", {})
    _periods = [previous_period(year, month, _back) for _back in range(1, 13)]
    try:
        _loaded_all, _prefs_all = load_auto_saved_assignments(
            _periods, owner=current_owner, include_preferences=True,
        )
    except Exception:
        # Bir dakika sonra yeniden denenir; o zamana kadar uyarı gösterilir.
        _ctx["failed_at"] = time.monotonic()
        return _ctx
    _aliases = st.session_state.get("name_aliases")
    for _back, (_py, _pm) in enumerate(_periods, start=1):
        _prev = _loaded_all.get((_py, _pm))
        if _prev is None or _prev.empty:
            continue
        _prev = as_bool_frame(_prev)
        # Eski adla kaydedilmiş aylar güncel ada sayılsın.
        _prev = _prev.rename(index={
            _p: resolve_alias(_p, _aliases, isimler) for _p in _prev.index
        })
        if _prev.index.has_duplicates:
            _prev = _prev.groupby(level=0).any()
        _days_in_prev = calendar.monthrange(_py, _pm)[1]
        _prev_holidays, _ = parse_holiday_days(holiday_text_for_period(_holiday_texts, _py, _pm), _days_in_prev)
        _holiday_month = holiday_shift_counts(_prev, _prev_holidays)
        if _prev_holidays:
            _ctx["monthly_holidays"].append(_holiday_month)
            _ctx["holiday_months"] += 1
            for _person, _count in _holiday_month.items():
                _ctx["holiday_counts"][_person] = _ctx["holiday_counts"].get(_person, 0) + _count
        if _back > 3:
            continue
        if _back == 1:
            _ctx["last_shift"] = last_shift_offsets(_prev, _days_in_prev)
        for _person, _count in special_shift_counts(_prev, _py, _pm, _prev_holidays).items():
            _ctx["history_counts"][_person] = _ctx["history_counts"].get(_person, 0) + _count
        _ctx["monthly_totals"].append(schedule_totals(_prev))
        # İzinli/müsait olmadığı günler yüzünden az tutan "eksik" sayılmasın.
        _available = {_p: _days_in_prev for _p in _prev.index}
        for _p, _n in unavailable_day_counts(_prefs_all.get((_py, _pm))).items():
            _p = resolve_alias(_p, _aliases, isimler)
            if _p in _available:
                _available[_p] = max(0, _available[_p] - _n)
        _ctx["monthly_available"].append(_available)
        _ctx["months"] += 1
    return _ctx


_prev_ctx = st.session_state.get("prev_month_context")
if (
    not _prev_ctx
    or _prev_ctx.get("period") != (yil, ay)
    or "monthly_available" not in _prev_ctx
    or ("failed_at" in _prev_ctx and time.monotonic() - _prev_ctx["failed_at"] > 60)
):
    _prev_ctx = _load_previous_context(yil, ay)
    st.session_state.prev_month_context = _prev_ctx
if "failed_at" in _prev_ctx:
    st.warning(
        "⚠️ Önceki aylar okunamadı: ay geçişi dinlenme kuralı ve aylar arası denge "
        "şu an uygulanmıyor. Bir dakika sonra yeniden denenecek."
    )
prev_last_shift = {p: d for p, d in _prev_ctx["last_shift"].items() if p in isimler}
_balance_history_on = st.session_state.get("balance_history", True)
history_special = (
    history_special_adjustment(_prev_ctx["history_counts"], isimler)
    if _balance_history_on and _prev_ctx["months"]
    else {}
)
# Kişisel limiti olanların az/çok nöbeti bilinçli; toplam dengesine katılmaz.
_limited_people = set(st.session_state.get("person_limits", {}))
history_total = (
    {
        _p: _v for _p, _v in history_adjustment(
            _prev_ctx["monthly_totals"], [_p for _p in isimler if _p not in _limited_people],
            monthly_available=_prev_ctx.get("monthly_available"),
        ).items() if _v
    }
    if _balance_history_on and _prev_ctx["months"]
    else {}
)
history_holiday = (
    {_p: _v for _p, _v in history_adjustment(_prev_ctx["monthly_holidays"], isimler).items() if _v}
    if _balance_history_on and _prev_ctx["holiday_months"]
    else {}
)

# --- ÇAKIŞMA HÜCRELERİ (grid vurgulaması için önceden hesapla) ---
conflict_cells = set()  # (kişi, col) çiftleri
_last_cf = {i: prev_last_shift.get(i, -10) for i in isimler}
for _cf_col in sutunlar:
    _cf_gun = gun_detaylari[_cf_col]['day_num']
    _cf_assigned = (
        st.session_state.schedule_bool.index[st.session_state.schedule_bool[_cf_col]].tolist()
        if _cf_col in st.session_state.schedule_bool.columns else []
    )
    for _cf_k in _cf_assigned:
        _pref_v = (
            st.session_state.pref_df.at[_cf_k, _cf_col]
            if _cf_k in st.session_state.pref_df.index and _cf_col in st.session_state.pref_df.columns
            else 0
        )
        if _pref_v == 3:
            conflict_cells.add((_cf_k, _cf_col))
        if (_cf_gun - _last_cf.get(_cf_k, -10)) <= min_bosluk and _last_cf.get(_cf_k, -10) != -10:
            conflict_cells.add((_cf_k, _cf_col))
        _last_cf[_cf_k] = _cf_gun

tab_grid, tab_cal = st.tabs(["🗂️ Tablo Görünümü", "📅 Takvim Görünümü"])

# ═══════════════════════════════════════════════════════
# TAB 1: TABLO GÖRÜNÜMü
# ═══════════════════════════════════════════════════════
with tab_grid:
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
                (3, "🟥İzin")
            ]
            if 'paint_color' not in st.session_state:
                st.session_state.paint_color = 0
            color_cols = st.columns(4)
            for i, (val, label) in enumerate(color_info):
                with color_cols[i]:
                    btn_type = "primary" if st.session_state.paint_color == val else "secondary"
                    if st.button(label, key=f"color_{val}", type=btn_type):
                        st.session_state.paint_color = val
                        st.rerun()
            selected_label = color_info[st.session_state.paint_color][1]
        else:
            st.info("Tıklayarak nöbet ekle/kaldır. Elle atadığın nöbetler (sarı çerçeve) simülasyonda korunur.")
            selected_label = ""

    # Quick actions row
    action_cols = st.columns(4)
    with action_cols[0]:
        if st.button("⚡ Simülasyon", type="primary", width="stretch"):
            st.session_state.run_simulation = True
    with action_cols[1]:
        if st.button("🔄 Sıfırla", width="stretch"):
            st.session_state.reset_pending = True
            st.rerun()
    with action_cols[2]:
        if st.button("↩️ Geri", width="stretch", disabled=len(st.session_state.undo_history)==0):
            _snap = undo_history_state(st.session_state, _current_state_snapshot())
            if _snap is not None:
                _restore_snapshot(_snap)
                st.rerun()
    with action_cols[3]:
        if st.button("↪️ İleri", width="stretch", disabled=len(st.session_state.redo_history)==0):
            _snap = redo_history_state(st.session_state, _current_state_snapshot())
            if _snap is not None:
                _restore_snapshot(_snap)
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
    # Karanlık modda açık renkli yazılar/zeminler; eskiden isimler açık zemin
    # üzerinde açık renkte kalıp görünmüyordu.
    try:
        _dark_theme = st.context.theme.type == "dark"
    except Exception:
        _dark_theme = False
    ui = {
        "name_text": "#e5e7eb" if _dark_theme else "#1f2937",
        "name_bg": "#1e293b" if _dark_theme else "#f8fafc",
        "name_border": "#334155" if _dark_theme else "#e2e8f0",
        "muted": "#9ca3af",
        "day_text": "#e5e7eb" if _dark_theme else "#374151",
        "weekend_text": "#93c5fd" if _dark_theme else "#1d4ed8",
        "holiday_text": "#fca5a5" if _dark_theme else "#c62828",
        "today_text": "#fcd34d" if _dark_theme else "#b45309",
        "weekend_bg": "rgba(59,130,246,0.18)" if _dark_theme else "rgba(219,234,254,0.5)",
        "holiday_bg": "rgba(239,68,68,0.18)" if _dark_theme else "rgba(254,226,226,0.55)",
        "today_bg": "rgba(245,158,11,0.22)" if _dark_theme else "rgba(254,243,199,0.75)",
        "legend_text": "#d1d5db" if _dark_theme else "#374151",
        # Nöbetsiz hücreler: açık zemin + renkli yazı (nöbetliler dolu renk)
        "free_bg": "rgba(148,163,184,0.10)" if _dark_theme else "#ffffff",
        "free_border": "#334155" if _dark_theme else "#e2e8f0",
        "free_text": "#64748b",
        "soft_green_bg": "rgba(34,197,94,0.16)" if _dark_theme else "#dcfce7",
        "soft_green_text": "#86efac" if _dark_theme else "#166534",
        "soft_amber_bg": "rgba(245,158,11,0.16)" if _dark_theme else "#fef3c7",
        "soft_amber_text": "#fcd34d" if _dark_theme else "#92400e",
        "soft_red_bg": "rgba(239,68,68,0.18)" if _dark_theme else "#fee2e2",
        "soft_red_text": "#fca5a5" if _dark_theme else "#991b1b",
        # Takvim sekmesi ve kişi kartı
        "card_bg": "#1e293b" if _dark_theme else "#f8fafc",
        "stat_bg": "#0f172a" if _dark_theme else "#ffffff",
        "strong_text": "#e5e7eb" if _dark_theme else "#1f2937",
        "label_text": "#9ca3af" if _dark_theme else "#6b7280",
        "stat_blue": "#93c5fd" if _dark_theme else "#1d4ed8",
        "stat_gray": "#e5e7eb" if _dark_theme else "#374151",
        "stat_red": "#fca5a5" if _dark_theme else "#dc2626",
        "stat_green": "#4ade80" if _dark_theme else "#16a34a",
        "stat_purple": "#c084fc" if _dark_theme else "#9333ea",
        "cal_head_bg": "#1e293b" if _dark_theme else "#f1f5f9",
        "cal_head_we_bg": "rgba(59,130,246,0.25)" if _dark_theme else "#dbeafe",
        "cal_cell": ("#111827", "1px solid #334155") if _dark_theme else ("#f8fafc", "1px solid #e2e8f0"),
        "cal_cell_we": ("rgba(59,130,246,0.15)", "1px solid #1e3a8a") if _dark_theme else ("#dbeafe", "1px solid #bfdbfe"),
        "cal_cell_hol": ("rgba(239,68,68,0.15)", "1px solid #7f1d1d") if _dark_theme else ("#fee2e2", "1px solid #fca5a5"),
        "cal_cell_today": ("rgba(245,158,11,0.18)", "2px solid #f59e0b") if _dark_theme else ("#fef3c7", "2px solid #f59e0b"),
    }
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

    st.markdown(f"""
<style>
.st-key-schedule-grid [data-testid="stHorizontalBlock"] > :is([data-testid="column"],[data-testid="stColumn"]):nth-child(1),
.st-key-preference-grid [data-testid="stHorizontalBlock"] > :is([data-testid="column"],[data-testid="stColumn"]):nth-child(1) {{
    background: {ui['name_bg']} !important;
    border-right: 2px solid {ui['name_border']};
    min-width: 80px;
}}
/* Ayın tüm günleri tek satırda kalsın; sığmazsa tablo yana kayar
   (eskiden son günler alt satıra geniş şeritler olarak düşüyordu). */
.st-key-schedule-grid [data-testid="stHorizontalBlock"],
.st-key-preference-grid [data-testid="stHorizontalBlock"] {{
    flex-wrap: nowrap !important;
}}
@media (min-width: 769px) {{
    /* Masaüstü/tablet: hücreler küçülerek tek satıra sığar (mobil kuralı ayrı). */
    .st-key-schedule-grid [data-testid="stHorizontalBlock"] > :is([data-testid="column"],[data-testid="stColumn"]),
    .st-key-preference-grid [data-testid="stHorizontalBlock"] > :is([data-testid="column"],[data-testid="stColumn"]) {{
        min-width: 26px !important;
        flex-shrink: 1 !important;
    }}
    .st-key-schedule-grid [data-testid="stHorizontalBlock"] > :is([data-testid="column"],[data-testid="stColumn"]):nth-child(1),
    .st-key-preference-grid [data-testid="stHorizontalBlock"] > :is([data-testid="column"],[data-testid="stColumn"]):nth-child(1) {{
        min-width: 96px !important;
    }}
}}
/* İsim sütunu en uzun isme göre sabit genişlikte (nokta + isim + sayı rozeti);
   sayı alt satıra kayıp satırı uzatmaz. */
.st-key-schedule-grid [data-testid="stHorizontalBlock"] > :is([data-testid="column"],[data-testid="stColumn"]):nth-child(1),
.st-key-preference-grid [data-testid="stHorizontalBlock"] > :is([data-testid="column"],[data-testid="stColumn"]):nth-child(1) {{
    flex: 0 0 {name_col_px}px !important;
    min-width: {name_col_px}px !important;
    max-width: {name_col_px}px !important;
}}
{col_bg_css}
.st-key-schedule-grid :is([data-testid="column"],[data-testid="stColumn"]),
.st-key-preference-grid :is([data-testid="column"],[data-testid="stColumn"]) {{ padding: 0 1px !important; }}
.st-key-schedule-grid .stButton > button,
.st-key-preference-grid .stButton > button {{
    padding: 0px !important;
    min-height: 30px !important;
    font-size: 14px !important;
    border-radius: 5px !important;
    line-height: 1 !important;
    width: 100% !important;
    border: 1px solid #e5e7eb !important;
    transition: transform 0.1s, box-shadow 0.1s;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}}
.st-key-schedule-grid .stButton > button:hover,
.st-key-preference-grid .stButton > button:hover {{
    transform: scale(1.15);
    box-shadow: 0 2px 8px rgba(0,0,0,0.18);
    z-index: 20;
    position: relative;
}}
:is(.st-key-preference-grid, .st-key-schedule-grid) .preference-state {{
    display: none !important;
}}
/* Renk işaretinin kapsayıcısı da yer kaplamasın: düğmeleri aşağı itip
   isimlerle hizayı bozuyordu. (Gizli öğe DOM'da kalır; :has() eşleşmeye devam eder.) */
:is(.st-key-preference-grid, .st-key-schedule-grid) :is([data-testid="stElementContainer"], .element-container):has(.preference-state) {{
    display: none !important;
}}
/* İsim, sayı ve gün düğmeleri satırda dikeyde ortalı */
:is(.st-key-preference-grid, .st-key-schedule-grid) [data-testid="stHorizontalBlock"] {{
    align-items: center !important;
}}
:is(.st-key-preference-grid, .st-key-schedule-grid) .stButton > button {{
    color: #ffffff !important;
    border-width: 2px !important;
    font-weight: 800 !important;
}}
:is(.st-key-preference-grid, .st-key-schedule-grid) :is([data-testid="column"],[data-testid="stColumn"]):has(.preference-state-0) .stButton > button {{
    background: #475569 !important;
    border-color: #334155 !important;
}}
:is(.st-key-preference-grid, .st-key-schedule-grid) :is([data-testid="column"],[data-testid="stColumn"]):has(.preference-state-1) .stButton > button {{
    background: #166534 !important;
    border-color: #14532d !important;
}}
:is(.st-key-preference-grid, .st-key-schedule-grid) :is([data-testid="column"],[data-testid="stColumn"]):has(.preference-state-2) .stButton > button {{
    background: #92400e !important;
    border-color: #78350f !important;
}}
:is(.st-key-preference-grid, .st-key-schedule-grid) :is([data-testid="column"],[data-testid="stColumn"]):has(.preference-state-3) .stButton > button {{
    background: #991b1b !important;
    border-color: #7f1d1d !important;
}}
:is(.st-key-preference-grid, .st-key-schedule-grid) :is([data-testid="column"],[data-testid="stColumn"]):has(.preference-free-0) .stButton > button {{
    background: {ui['free_bg']} !important;
    border-color: {ui['free_border']} !important;
    border-width: 1px !important;
    color: {ui['free_text']} !important;
    font-weight: 600 !important;
}}
:is(.st-key-preference-grid, .st-key-schedule-grid) :is([data-testid="column"],[data-testid="stColumn"]):has(.preference-free-1) .stButton > button {{
    background: {ui['soft_green_bg']} !important;
    border-color: transparent !important;
    color: {ui['soft_green_text']} !important;
}}
:is(.st-key-preference-grid, .st-key-schedule-grid) :is([data-testid="column"],[data-testid="stColumn"]):has(.preference-free-2) .stButton > button {{
    background: {ui['soft_amber_bg']} !important;
    border-color: transparent !important;
    color: {ui['soft_amber_text']} !important;
}}
:is(.st-key-preference-grid, .st-key-schedule-grid) :is([data-testid="column"],[data-testid="stColumn"]):has(.preference-free-3) .stButton > button {{
    background: {ui['soft_red_bg']} !important;
    border-color: transparent !important;
    color: {ui['soft_red_text']} !important;
}}
:is(.st-key-preference-grid, .st-key-schedule-grid) :is([data-testid="column"],[data-testid="stColumn"]):has(.preference-state-conflict) .stButton > button {{
    background: #7f1d1d !important;
    border-color: #450a0a !important;
}}
/* Elle atanmış (sabit) nöbet: ince sarı iç çerçeve */
:is(.st-key-preference-grid, .st-key-schedule-grid) :is([data-testid="column"],[data-testid="stColumn"]):has(.cell-locked) .stButton > button {{
    box-shadow: inset 0 0 0 2px #fbbf24 !important;
}}
:is(.st-key-preference-grid, .st-key-schedule-grid) .stButton > button:focus-visible {{
    outline: 3px solid #0f172a !important;
    outline-offset: 2px;
}}
@media (max-width: 768px) {{
    /*
     * This must follow the shared grid-button rule above. Otherwise its
     * min-height:30px declaration wins in the mobile cascade.
     */
    :is(.st-key-preference-grid, .st-key-schedule-grid) .stButton > button {{
        min-width: 44px !important;
        min-height: 44px !important;
        width: 44px !important;
        padding: 0 !important;
        font-size: 16px !important;
        font-weight: 800 !important;
        border-radius: 7px !important;
    }}
    :is(.st-key-preference-grid, .st-key-schedule-grid) .stButton > button:hover {{
        transform: none !important;
    }}
}}
.st-key-preference-legend .preference-legend-swatch {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 5px;
    color: #ffffff !important;
    font-weight: 800;
    line-height: 1;
}}
</style>
""", unsafe_allow_html=True)

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
            st.caption("Tercih kodları: 0 = boş, 1 = tercih, 2 = kaçınma, 3 = izin")
            _compact_pref = st.data_editor(
                st.session_state.pref_df,
                width="stretch",
                key="large_schedule_preference_editor",
                num_rows="fixed",
            )
            _normalized_pref = normalize_preference_grid(_compact_pref)
            if _normalized_pref is None:
                st.error("Tercih hücrelerinde yalnızca 0, 1, 2 veya 3 tam sayıları kullanılabilir.")
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
                _queue_assignment_warnings([
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
                            _queue_assignment_warnings([(person, col)])
                        st.rerun()

    legend_container = st.container(key="preference-legend", border=False)
    _soft_style = {
        "neutral": (ui["free_bg"], ui["free_text"], f"1px solid {ui['free_border']}"),
        "preferred": (ui["soft_green_bg"], ui["soft_green_text"], "none"),
        "avoid": (ui["soft_amber_bg"], ui["soft_amber_text"], "none"),
        "unavailable": (ui["soft_red_bg"], ui["soft_red_text"], "none"),
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

# ═══════════════════════════════════════════════════════
# TAB 2: TAKVİM GÖRÜNÜMü
# ═══════════════════════════════════════════════════════
with tab_cal:
    # Rol isimleri (Görev1, Görev2 vb.)
    _cal_role_names = st.session_state.get('rol_isimleri', [f"Görev{i+1}" for i in range(kişi_sayısı)])

    # ── Takvimden gelen görev yeri değişim isteği ────────
    _swap_dn = st.session_state.pop('cal_swap_day', None)
    if _swap_dn is not None and _rotate_roles_for_day(_swap_dn):
        st.rerun()

    # ── Filtre + açıklama satırı ──────────────────────────
    cal_top1, cal_top2 = st.columns([3, 2])
    with cal_top1:
        cal_filter_persons = st.multiselect(
            "Kişiye göre filtrele:",
            options=isimler,
            default=[],
            key="cal_person_filter",
            placeholder="Tüm ekip gösteriliyor"
        )
    with cal_top2:
        st.markdown(
            f"<div style='padding-top:28px;font-size:12px;color:{ui['label_text']};'>"
            "🟢 Tercih &nbsp; 🟡 Kaçın &nbsp; 🔴 İzin &nbsp; 🔄 Görev Yeri Değiştir"
            "</div>",
            unsafe_allow_html=True
        )

    # ── KİŞİ DETAY KARTI (tek kişi seçilince) ────────────
    if len(cal_filter_persons) == 1:
        dp = cal_filter_persons[0]
        dp_color = person_colors.get(dp, "#e2e8f0")
        dp_shifts = [col for col in sutunlar if (
            dp in st.session_state.schedule_bool.index and
            bool(st.session_state.schedule_bool.at[dp, col])
        )]
        # Her nöbet tek yerde sayılır: hafta sonu, (hafta içine denk gelen) tatil
        # ya da hafta içi; eskiden hafta içi tatiller iki yerde sayılıyordu.
        dp_we = sum(1 for c in dp_shifts if gun_detaylari[c]['weekend'])
        dp_hol = sum(1 for c in dp_shifts if gun_detaylari[c]['holiday'] and not gun_detaylari[c]['weekend'])
        dp_weekday = len(dp_shifts) - dp_we - dp_hol
        dp_total_h = len(dp_shifts) * 24
        dp_fm_h = max(0, dp_total_h - zorunlu_saat)
        dp_pay = dp_fm_h * nobet_ucreti
        _dp_rows = st.session_state.get('cached_rows_liste') or []
        _dp_role_names = st.session_state.get('cached_role_names', _cal_role_names)
        # Tercih istatistikleri (sadece nöbet tutulan günler)
        dp_pref_on_shift = {0: 0, 1: 0, 2: 0, 3: 0}
        dp_pref_all = {0: 0, 1: 0, 2: 0, 3: 0}
        for col in sutunlar:
            pv = int(st.session_state.pref_df.at[dp, col]) if (
                dp in st.session_state.pref_df.index and col in st.session_state.pref_df.columns
            ) else 0
            dp_pref_all[pv] += 1
            if col in dp_shifts:
                dp_pref_on_shift[pv] += 1
        dp_green_ok = dp_pref_on_shift[1]
        dp_yellow_hit = dp_pref_on_shift[2]
        dp_red_hit = dp_pref_on_shift[3]
        dp_total_pref = dp_pref_all[1] + dp_pref_all[2]
        # Tercih başarısı: yeşil günlerde kaç atama yapıldı
        dp_pref_score = (
            int(dp_green_ok / dp_pref_all[1] * 100) if dp_pref_all[1] > 0 else None
        )

        st.markdown(
            f"<div style='background:linear-gradient(135deg,{dp_color}55 0%,{ui['card_bg']} 60%);"
            f"border:2px solid {dp_color};border-radius:14px;padding:16px 20px;margin-bottom:12px;'>"
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>"
            f"<span style='background:{dp_color};border-radius:50%;width:36px;height:36px;"
            f"display:inline-flex;align-items:center;justify-content:center;"
            f"font-size:16px;font-weight:800;color:#1f2937;border:2px solid rgba(0,0,0,0.1);'>"
            f"{escape_html(dp[0].upper())}</span>"
            f"<span style='font-size:20px;font-weight:800;color:{ui['strong_text']};'>{escape_html(dp)}</span>"
            f"<span style='font-size:13px;color:{ui['label_text']};margin-left:4px;'>"
            f"— {ay} / {yil}</span></div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:10px;'>"
            f"<div style='background:{ui['stat_bg']};border-radius:10px;padding:8px 14px;min-width:90px;text-align:center;"
            f"box-shadow:0 1px 4px rgba(0,0,0,0.08);'>"
            f"<div style='font-size:22px;font-weight:800;color:{ui['stat_blue']};'>{len(dp_shifts)}</div>"
            f"<div style='font-size:11px;color:{ui['label_text']};'>Toplam Nöbet</div></div>"
            f"<div style='background:{ui['stat_bg']};border-radius:10px;padding:8px 14px;min-width:90px;text-align:center;"
            f"box-shadow:0 1px 4px rgba(0,0,0,0.08);'>"
            f"<div style='font-size:22px;font-weight:800;color:{ui['stat_gray']};'>{dp_weekday}</div>"
            f"<div style='font-size:11px;color:{ui['label_text']};'>Hafta İçi</div></div>"
            f"<div style='background:{ui['stat_bg']};border-radius:10px;padding:8px 14px;min-width:90px;text-align:center;"
            f"box-shadow:0 1px 4px rgba(0,0,0,0.08);'>"
            f"<div style='font-size:22px;font-weight:800;color:{ui['stat_blue']};'>{dp_we}</div>"
            f"<div style='font-size:11px;color:{ui['label_text']};'>Hafta Sonu</div></div>"
            f"<div style='background:{ui['stat_bg']};border-radius:10px;padding:8px 14px;min-width:90px;text-align:center;"
            f"box-shadow:0 1px 4px rgba(0,0,0,0.08);'>"
            f"<div style='font-size:22px;font-weight:800;color:{ui['stat_red']};'>{dp_hol}</div>"
            f"<div style='font-size:11px;color:{ui['label_text']};'>Tatil</div></div>"
            f"<div style='background:{ui['stat_bg']};border-radius:10px;padding:8px 14px;min-width:90px;text-align:center;"
            f"box-shadow:0 1px 4px rgba(0,0,0,0.08);'>"
            f"<div style='font-size:22px;font-weight:800;color:{ui['stat_green']};'>{'%{}'.format(dp_pref_score) if dp_pref_score is not None else '—'}</div>"
            f"<div style='font-size:11px;color:{ui['label_text']};'>Tercih Başarısı</div></div>"
            f"<div style='background:{ui['stat_bg']};border-radius:10px;padding:8px 14px;min-width:90px;text-align:center;"
            f"box-shadow:0 1px 4px rgba(0,0,0,0.08);'>"
            f"<div style='font-size:22px;font-weight:800;color:{ui['stat_purple']};'>{dp_pay:.0f} ₺</div>"
            f"<div style='font-size:11px;color:{ui['label_text']};'>Tahmini Ödeme</div></div>"
            f"</div>"
            f"<div style='margin-top:10px;font-size:11px;color:{ui['label_text']};'>"
            f"🟢 {dp_green_ok} tercih tuttu &nbsp;·&nbsp; "
            f"🟡 {dp_yellow_hit} kaçın gününde atandı &nbsp;·&nbsp; "
            f"🔴 {dp_red_hit} izin gününde atandı</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Nöbet günleri satırı
        if dp_shifts:
            day_pills = ""
            for c in dp_shifts:
                inf = gun_detaylari[c]
                dn2 = inf['day_num']
                dn2_name = tr_gunler[date(yil, ay, dn2).weekday()]
                dp_role = find_person_role(_dp_rows, _dp_role_names, sutunlar.index(c), dp)
                dp_role_label = f" · {escape_html(dp_role)}" if dp_role else ""
                if inf['holiday']:
                    pill_bg = "#fee2e2"; pill_tc = "#dc2626"
                elif inf['weekend']:
                    pill_bg = "#dbeafe"; pill_tc = "#1d4ed8"
                else:
                    pill_bg = dp_color; pill_tc = "#1f2937"
                day_pills += (
                    f"<span style='display:inline-block;background:{pill_bg};color:{pill_tc};"
                    f"border-radius:20px;padding:2px 10px;font-size:11px;font-weight:600;"
                    f"margin:2px;white-space:nowrap;'>{dn2} {dn2_name}{dp_role_label}</span>"
                )
            st.markdown(
                f"<div style='margin:6px 0 12px 0;line-height:2;'>{day_pills}</div>",
                unsafe_allow_html=True
            )

    # ── Takvim başlık satırı (Pzt → Paz) ─────────────────
    cal_day_names = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
    cal_hdr = st.columns(7)
    for j, gn in enumerate(cal_day_names):
        with cal_hdr[j]:
            bg = ui["cal_head_we_bg"] if j >= 5 else ui["cal_head_bg"]
            tc = ui["weekend_text"] if j >= 5 else ui["day_text"]
            st.markdown(
                f"<div style='text-align:center;background:{bg};border-radius:8px;padding:6px 2px;"
                f"font-size:12px;font-weight:700;color:{tc};'>{gn[:3]}</div>",
                unsafe_allow_html=True
            )

    # ── Takvim hücreleri ──────────────────────────────────
    first_weekday = date(yil, ay, 1).weekday()
    cal_day_num = 1
    for week in range(6):
        if cal_day_num > gun_sayisi:
            break
        week_cols = st.columns(7)
        for j in range(7):
            with week_cols[j]:
                abs_pos = week * 7 + j
                if abs_pos < first_weekday or cal_day_num > gun_sayisi:
                    st.markdown("<div style='min-height:95px;'></div>", unsafe_allow_html=True)
                else:
                    dn = cal_day_num
                    ck = f"{dn} {tr_gunler[date(yil, ay, dn).weekday()]}"
                    _is_we = date(yil, ay, dn).weekday() >= 5
                    _is_hol = dn in tatil_gunleri
                    _is_tod = (date(yil, ay, dn) == today_date)
                    # Nöbetçi listesi (isimler sırasına göre = rol sırası)
                    all_nobetciler = (
                        st.session_state.schedule_bool.index[st.session_state.schedule_bool[ck]].tolist()
                        if ck in st.session_state.schedule_bool.columns else []
                    )

                    # ── Hücre arkaplanı: gün tipi renkleri korunur ──
                    if _is_tod:
                        (cell_bg, border), day_c = ui["cal_cell_today"], ui["today_text"]
                    elif _is_hol:
                        (cell_bg, border), day_c = ui["cal_cell_hol"], ui["holiday_text"]
                    elif _is_we:
                        (cell_bg, border), day_c = ui["cal_cell_we"], ui["weekend_text"]
                    else:
                        (cell_bg, border), day_c = ui["cal_cell"], ui["day_text"]

                    # Kişi renk bandı: nöbetçilerin renkleri hücre altında ince bant olarak
                    color_bar_html = ""
                    if all_nobetciler:
                        _default_pc = "#e2e8f0"
                        bar_parts = "".join(
                            "<div style='flex:1;background:{};'></div>".format(
                                person_colors.get(p, _default_pc)
                            )
                            for p in all_nobetciler
                        )
                        color_bar_html = (
                            "<div style='display:flex;height:5px;border-radius:0 0 7px 7px;"
                            "overflow:hidden;margin:-7px -7px 0 -7px;'>{}</div>".format(bar_parts)
                        )
                    # Filtre uygula (sadece görsel; boş filtre = hepsi)
                    visible = [p for p in all_nobetciler if (not cal_filter_persons or p in cal_filter_persons)]

                    chips = ""
                    _cal_rows = st.session_state.get('cached_rows_liste') or []
                    _cal_crnames = st.session_state.get('cached_role_names', _cal_role_names)
                    for role_idx, p in enumerate(all_nobetciler):
                        if cal_filter_persons and p not in cal_filter_persons:
                            continue
                        pc = person_colors.get(p, '#e2e8f0')
                        # cached_rows_liste'den kişinin gerçek görev yerini bul
                        rl = find_person_role(_cal_rows, _cal_crnames, dn - 1, p)
                        if rl is None:
                            rl = _cal_role_names[role_idx] if role_idx < len(_cal_role_names) else f"G{role_idx+1}"
                        # Müsaitlik rengi kenarlık olarak
                        pv = int(st.session_state.pref_df.at[p, ck]) if (
                            p in st.session_state.pref_df.index and ck in st.session_state.pref_df.columns
                        ) else 0
                        outline = {1: "2px solid #16a34a", 2: "2px solid #d97706", 3: "2px solid #dc2626"}.get(pv, "none")
                        chips += (
                            f"<div style='display:flex;align-items:center;gap:3px;margin:2px 0;'>"
                            f"<span style='background:{pc};border-radius:8px;padding:2px 8px;"
                            f"font-size:12px;font-weight:700;color:#1f2937;white-space:nowrap;"
                            f"outline:{outline};'>{escape_html(p)}</span>"
                            f"<span style='font-size:10px;color:{ui['label_text']};white-space:nowrap;'>{escape_html(rl)}</span>"
                            f"</div>"
                        )

                    day_icon = "🎉" if _is_hol else ("★" if _is_tod else "")
                    # Filtreli görünümde bu günde seçili kişi var mı?
                    has_filtered = bool(visible) if cal_filter_persons else bool(all_nobetciler)
                    empty_msg = "" if chips else "<span style='color:#9ca3af;font-size:10px;'>Boş</span>"
                    # Filtre varsa ama bu günde o kişi yoksa soluk göster
                    cell_opacity = "1" if (not cal_filter_persons or has_filtered) else "0.35"

                    st.markdown(
                        f"<div style='background:{cell_bg};border:{border};border-radius:10px;"
                        f"padding:7px;min-height:95px;opacity:{cell_opacity};overflow:hidden;'>"
                        f"<div style='font-size:15px;font-weight:700;color:{day_c};'>{day_icon}{dn}</div>"
                        f"<div style='margin-top:3px;'>{chips}{empty_msg}</div>"
                        f"{color_bar_html}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    if len(all_nobetciler) >= 2:
                        if st.button("🔄", key=f"cal_swap_{dn}", help="Görev yerlerini değiştir", width="stretch"):
                            st.session_state['cal_swap_day'] = dn
                            st.rerun()
                    cal_day_num += 1

# Build algorithm inputs from pref_df
df_unwanted = pd.DataFrame(False, index=isimler, columns=sutunlar)
df_preferred = pd.DataFrame(0, index=isimler, columns=sutunlar)

for person in isimler:
    for col in sutunlar:
        status = st.session_state.pref_df.at[person, col] if person in st.session_state.pref_df.index and col in st.session_state.pref_df.columns else 0
        if status == 3:
            df_unwanted.at[person, col] = True
        if status == 1:
            df_preferred.at[person, col] = 1
        elif status == 2:
            df_preferred.at[person, col] = 2

def _missing_day_reasons(schedule, col):
    return explain_missing_day(
        schedule, col, gun_detaylari, df_unwanted, min_bosluk,
        st.session_state.get("person_limits", {}),
        st.session_state.get("forbidden_pairs", []),
        prev_last_shift,
    )


def _missing_day_reason_text(schedule, col):
    """Eksik nöbetçili gün için kimin neden atanamadığını tek satırda anlatır."""
    return format_missing_day_reason(_missing_day_reasons(schedule, col))


def _simulation_context():
    """Simülasyon girdilerinin özeti: değişirse bekleyen eksik sonuç geçersiz olur."""
    return (
        yil, ay, tuple(isimler), tuple(sutunlar), kişi_sayısı, min_bosluk,
        schedule_fingerprint(df_unwanted), schedule_fingerprint(df_preferred),
        tuple(sorted(st.session_state.forbidden_pairs)),
        tuple(sorted((p, tuple(sorted(limits.items())))
                     for p, limits in st.session_state.get('person_limits', {}).items())),
        tuple(sorted(active_locks(st.session_state.get("locked_cells"), st.session_state.schedule_bool))),
        tuple(sorted(prev_last_shift.items())),
        tuple(sorted(history_special.items())),
        tuple(sorted(history_total.items())),
        tuple(sorted(history_holiday.items())),
    )


# Handle simulation trigger
sim_clicked = st.session_state.get('run_simulation', False)
st.session_state.run_simulation = False

if sim_clicked:
    st.session_state.incomplete_simulation = None
    is_valid, errors, warnings = validate_inputs(isimler, yil, ay, gun_sayisi, tatil_gunleri, nobet_ucreti, min_bosluk, kişi_sayısı)

    # Min limit fizibilite kontrolü
    _sim_person_limits = st.session_state.get('person_limits', {})
    if _sim_person_limits:
        for _p, _lim in _sim_person_limits.items():
            if _p in isimler:
                _min_req = _lim.get('min', 0)
                if _min_req > 0:
                    _avail = sum(1 for _c in sutunlar if not df_unwanted.at[_p, _c])
                    _rest_cap = -(-gun_sayisi // (min_bosluk + 1))
                    if _min_req > _avail:
                        warnings.append(f"⚠️ **{_p}**: Min {_min_req} nöbet gerekli ama yalnızca {_avail} müsait gün var — limit karşılanamayabilir")
                    elif _min_req > _rest_cap:
                        warnings.append(
                            f"⚠️ **{_p}**: Min {_min_req} nöbet gerekli ama {min_bosluk} gün dinlenmeyle "
                            f"en fazla {_rest_cap} nöbet tutulabilir — limit karşılanamayabilir"
                        )

    if errors:
        st.error("🚨 Hata(lar) düzeltilmeli:")
        for err in errors:
            st.error(err)
    else:
        if warnings:
            st.warning("⚠️ Uyarı(lar):")
            for warn in warnings:
                st.warning(warn)
        
        _new_schedule = run_scheduling_algorithm_v98(
            isimler, sutunlar, df_unwanted, gun_detaylari, min_bosluk, 
            kişi_sayısı,
            st.session_state.forbidden_pairs,
            st.session_state.get('person_limits', {}),
            df_preferred,
            initial_last_shift=prev_last_shift,
            history_special=history_special,
            fixed_assignments=active_locks(st.session_state.get("locked_cells"), st.session_state.schedule_bool),
            history_total=history_total,
            history_holiday=history_holiday,
        )
        if _new_schedule is not None:
            _missing = missing_shift_coverage(_new_schedule, sutunlar, kişi_sayısı)
            if _missing:
                st.session_state.incomplete_simulation = {
                    "schedule": _new_schedule,
                    "missing": _missing,
                    "context": _simulation_context(),
                    "previous_fingerprint": schedule_fingerprint(st.session_state.schedule_bool),
                }
            else:
                save_undo_state()
                st.session_state.schedule_bool = _new_schedule
                st.session_state.should_regenerate_assignments = True
                st.session_state.simulation_success_pending = True
                st.rerun()

_incomplete = st.session_state.incomplete_simulation
if _incomplete is not None:
    if (_incomplete["context"] != _simulation_context()
            or _incomplete["previous_fingerprint"] != schedule_fingerprint(st.session_state.schedule_bool)):
        st.session_state.incomplete_simulation = None
        st.info("Çizelge veya ekip ayarları değiştiği için eksik simülasyon sonucu iptal edildi. Yeniden simülasyon çalıştırın.")
    else:
        _missing = _incomplete["missing"]
        _slots = sum(kişi_sayısı - assigned for _, assigned in _missing)
        _days = ", ".join(str(gun_detaylari[col]["day_num"]) for col, _ in _missing)
        st.warning(
            f"⚠️ Simülasyon tamamlandı ancak {_days}. gün(ler) için toplam {_slots} nöbetçi eksik "
            f"({len(_missing)} gün). Sonuç kullanıma hazır değil; mevcut çizelge korunuyor. "
            "Müsaitlik, dinlenme aralığı, yasak çift ve kişi limitlerini gözden geçirip yeniden deneyin."
        )
        for _col, _assigned in _missing:
            st.caption(
                f"**{gun_detaylari[_col]['day_num']}. gün** ({_assigned}/{kişi_sayısı}) — "
                + _missing_day_reason_text(_incomplete["schedule"], _col)
            )
        _decision_cols = st.columns(2)
        with _decision_cols[0]:
            if st.button("Mevcut çizelgeyi koru", key="keep_incomplete_simulation"):
                st.session_state.incomplete_simulation = None
                st.rerun()
        with _decision_cols[1]:
            if st.button("Eksik sonuçla devam et", key="accept_incomplete_simulation"):
                save_undo_state()
                st.session_state.schedule_bool = _incomplete["schedule"]
                st.session_state.should_regenerate_assignments = True
                st.session_state.incomplete_simulation = None
                st.rerun()

# Use the latest schedule state for every calculation below.
# Grid buttons update schedule_bool and trigger a rerun. Taking this snapshot
# after those interactions prevents analysis from reading a cached list of
# assignments instead of the schedule currently being edited.
# REMOVED: Redundant Müsaitlik editor as requested
edited_df = st.session_state.schedule_bool.copy(deep=True)

# --- HATA KONTROL ---
violations = []
conflict_msg = []
forbidden_msg = []
max_person_msg = []
min_person_msg = []
last_shift_check = {i: prev_last_shift.get(i, -10) for i in isimler}

for col in sutunlar:
    gun_no = gun_detaylari[col]['day_num']
    nobetciler = edited_df.index[edited_df[col]].tolist()
    
    if len(nobetciler) > kişi_sayısı:
        max_person_msg.append(f"🔴 **{gun_no}. Gün**: {len(nobetciler)} kişi atanmış! (Max {kişi_sayısı})")
        
    for k in nobetciler:
        if df_unwanted.at[k, col]:
            conflict_msg.append(f"❌ **{k}**: {gun_no}. gün izinli.")
        
        if (gun_no - last_shift_check[k]) <= min_bosluk:
            violations.append(f"⚠️ **{k}**: {gun_no}. gün dinlenme kuralına uymuyor.")
        last_shift_check[k] = gun_no
    
    # Check for forbidden pairs
    if st.session_state.forbidden_pairs and len(nobetciler) >= 2:
        for i in range(len(nobetciler)):
            for j in range(i+1, len(nobetciler)):
                pair = tuple(sorted((nobetciler[i], nobetciler[j])))
                if pair in st.session_state.forbidden_pairs:
                    forbidden_msg.append(f"🚫 **{gun_no}. Gün**: {nobetciler[i]} ve {nobetciler[j]} birlikte çalışamaz!")

for col, assigned in missing_shift_coverage(edited_df, sutunlar, kişi_sayısı):
    _msg = f"🟠 **{gun_detaylari[col]['day_num']}. Gün**: Eksik nöbetçi! ({assigned}/{kişi_sayısı} kişi)"
    _reasons = _missing_day_reasons(edited_df, col)
    # Herkes uygunsa (ör. henüz boş çizelge) neden satırı gürültü olur.
    if not _reasons or any(_label != MISSING_DAY_FREE_LABEL for _label, _ in _reasons):
        _msg += f"  \nNeden: {format_missing_day_reason(_reasons)}"
    min_person_msg.append(_msg)

# Kişisel limit ihlali kontrolü
_pl = st.session_state.get('person_limits', {})
limit_msg = build_limit_violation_messages(edited_df, isimler, _pl)

# Giriş doğrulama uyarıları (yasak çift / limit — bilinmeyen isimler)
_fp_unk = st.session_state.get('fp_unknown_names', [])
_lim_unk = st.session_state.get('lim_unknown_names', [])
input_warn_msg = []
if _fp_unk:
    input_warn_msg.append(f"⚠️ Yasak çiftte ekip dışı isim(ler): **{', '.join(_fp_unk)}** — bu kısıtlama etkisiz.")
if _lim_unk:
    input_warn_msg.append(f"⚠️ Kişisel limitte ekip dışı isim(ler): **{', '.join(_lim_unk)}** — bu limit etkisiz.")

if max_person_msg or min_person_msg or conflict_msg or forbidden_msg or violations or limit_msg or input_warn_msg:
    with st.expander("🚨 HATA RAPORU (Tıklayıp Açın)", expanded=True):
        for m in max_person_msg: st.error(m)
        for m in min_person_msg: st.warning(m)
        for c in conflict_msg: st.error(c)
        for f in forbidden_msg: st.error(f)
        for v in violations: st.info(v)
        for m in limit_msg: st.warning(m)
        for m in input_warn_msg: st.warning(m)
else:
    st.success(f"✅ Kurallar uygun (Her gün {kişi_sayısı} kişi, çakışma yok).")

# ── Günlük Listeden gelen görev yeri değişim isteği ──────────────────────────
_liste_swap_dn = st.session_state.pop('liste_swap_day', None)
if _liste_swap_dn is not None and _rotate_roles_for_day(_liste_swap_dn):
    st.rerun()

# --- VERİ HAZIRLIĞI ---
# Only regenerate assignments when AI button is clicked, not on manual edits.
# Rol isimleri/sırası da satır yapısının parçasıdır.
_cache_key = (yil, ay, tuple(isimler), tuple(sutunlar), tuple(role_names))
if (st.session_state.should_regenerate_assignments
        or st.session_state.cached_rows_liste is None
        or st.session_state.get('cached_key') != _cache_key):
    rows_liste = []
    first_role_counts = {i: 0 for i in isimler}
    for col in sutunlar:
        nobetciler = edited_df.index[edited_df[col]].tolist()
        random.shuffle(nobetciler) 
        nobetciler.sort(key=lambda x: first_role_counts[x]) 
        
        # Build row with all assigned people using custom role names
        row_data = {"Tarih": gun_detaylari[col]['full_date']}
        
        for idx, role_name in enumerate(role_names):
            if len(nobetciler) > idx:
                row_data[role_name] = nobetciler[idx]
                if idx == 0:
                    first_role_counts[nobetciler[idx]] += 1
            else:
                row_data[role_name] = "-"
        
        rows_liste.append(row_data)
    st.session_state.cached_rows_liste = rows_liste
    st.session_state.cached_first_role_counts = first_role_counts
    st.session_state.cached_role_names = role_names
    st.session_state.cached_key = _cache_key
    st.session_state.cached_role_schedule = edited_df.copy(deep=True)
    st.session_state.should_regenerate_assignments = False
else:
    rows_liste = st.session_state.cached_rows_liste
    _cached_roles = list(st.session_state.get('cached_role_names') or role_names)
    if _cached_roles != list(role_names):
        # Satırlar eski görev düzenindeyse güncel görevlere taşı (yedek önlem).
        rows_liste = align_rows_to_roles(rows_liste, _cached_roles, role_names, edited_df, sutunlar, gun_detaylari)
        st.session_state.cached_rows_liste = rows_liste
        st.session_state.cached_role_names = list(role_names)
        st.session_state.cached_role_schedule = edited_df.copy(deep=True)
    if changed_schedule_columns(
        st.session_state.get("cached_role_schedule"),
        edited_df,
    ):
        # Tek hücre düzenlemesinde yalnızca ilgili günün rol satırını eşle.
        rows_liste = synchronize_changed_role_rows(
            rows_liste,
            st.session_state.get("cached_role_schedule"),
            edited_df,
            role_names,
            sutunlar,
            gun_detaylari,
        )
        st.session_state.cached_rows_liste = rows_liste
        st.session_state.cached_role_schedule = edited_df.copy(deep=True)

# Display and allow manual editing of ROLES (swapping people)
# REMOVED: Manual list editing as requested. Using autonomous redistribution instead.

# Re-distribution logic moved to Analiz section as requested.

# Görev sayısından fazla atanan kişiler listelerde "Ek 1", "Ek 2" sütununda görünür.
liste_rows, liste_roles = add_overflow_roles(rows_liste, role_names, edited_df, sutunlar)
df_liste = pd.DataFrame(liste_rows)

# Auto-save only after role rows have been generated or reconciled with the
# current schedule, so both database payloads describe the same snapshot.
# Her değişiklik hemen kaydedilir (eskiden 30 sn bekleniyordu ve son
# değişiklikler sayfa yenilenince kayboluyordu).
_active_locks = active_locks(st.session_state.get("locked_cells"), st.session_state.schedule_bool)
_hidden_people = st.session_state.get("hidden_people") or {}
_persist_fp = persistence_fingerprint(
    st.session_state.schedule_bool, st.session_state.get('pref_df'), rows_liste, _active_locks,
    _hidden_people,
)
if st.session_state.pop("autosave_baseline_pending", False):
    st.session_state.last_saved_fingerprint = _persist_fp
elif (
    st.session_state.get("autosave_blocked_period") != (yil, ay)
    and _persist_fp != st.session_state.get("last_saved_fingerprint")
):
    auto_name = f"Otomatik_{yil}_{ay:02d}"
    _saved_ok = False
    try:
        _saved_ok = save_schedule(auto_name, yil, ay, isimler, st.session_state.schedule_bool,
                                  pref_df=st.session_state.get('pref_df'),
                                  rows_liste=rows_liste, owner=current_owner,
                                  settings_dict={"locked": locks_to_json(_active_locks),
                                                 "hidden": _hidden_people})
    except Exception:
        _saved_ok = False
    if _saved_ok:
        st.session_state.last_saved_fingerprint = _persist_fp
        st.session_state.last_auto_save = time.time()
        update_persistence_feedback(st.session_state, True)
        # Yedek bölümü bu çalışmada eski dosyayla çizildi; yeniden çiz.
        if st.session_state.pop("backup_bytes", None) is not None:
            st.rerun()
    else:
        update_persistence_feedback(
            st.session_state,
            False,
            "❌ Çizelge kaydedilemedi — otomatik kayıt veritabanına yazılamıyor! "
            "Sayfayı yenilemeden önce Excel/PNG olarak indirin.",
        )

if st.session_state.get("persistence_error"):
    st.error(st.session_state.persistence_error)

# Yedekle tuşu: son değişiklik kaydedildikten sonra yedek alınır.
_run_requested_snapshot()

_rows_key = tuple(tuple(row.get(role, "-") for role in role_names) for row in rows_liste)
_analysis_key = (
    schedule_fingerprint(edited_df),
    _rows_key,
    tuple(role_names),
    tuple(sorted((person, limits.get("min", 0), limits.get("max", 999))
                 for person, limits in st.session_state.get("person_limits", {}).items())),
    zorunlu_saat,
    nobet_ucreti,
    tuple((column, gun_detaylari[column]["weekend"], gun_detaylari[column]["holiday"]) for column in sutunlar),
)
if (
    st.session_state.get("cached_analysis_key") != _analysis_key
    or st.session_state.get("cached_analysis_tables") is None
):
    st.session_state.cached_analysis_tables = build_schedule_analysis(
        edited_df,
        rows_liste,
        role_names,
        isimler,
        gun_detaylari,
        st.session_state.get("person_limits", {}),
        zorunlu_saat,
        nobet_ucreti,
    )
    st.session_state.cached_analysis_key = _analysis_key

df_stats_load, df_stats_finance, pair_display = st.session_state.cached_analysis_tables

# --- GÖRÜNÜM ---
st.divider()

# ── İndirme butonları ince yatay satır ──────────────────────────────────────
# PNG/Excel/HTML yalnızca butona tıklanınca üretilir (eskiden her nöbet
# işaretlemesinde matplotlib ile yeniden çiziliyordu). Varsayılan argümanlar
# o anki tabloları sabitler; on_click="ignore" indirmede sayfayı yeniden çizmez.
def _png_bytes(_liste=df_liste.copy()):
    return convert_df_to_png(_liste)

def _excel_bytes(_liste=df_liste.copy(), _load=df_stats_load.copy(),
                 _finance=df_stats_finance.copy(), _yil=yil):
    return convert_df_to_excel(_liste, _load, _finance, yil=_yil)

def _html_bytes(_liste=df_liste.copy(), _load=df_stats_load.copy(), _yil=yil, _ay=ay):
    return create_print_html(_liste, _load, _yil, _ay).encode('utf-8')

def _app_base_url():
    try:
        _url = st.context.url
    except Exception:
        _url = None
    if _url:
        return _url.split("?")[0].split("#")[0].rstrip("/")
    try:
        return st.secrets["auth"]["redirect_uri"].rsplit("/oauth2callback", 1)[0]
    except Exception:
        return ""


_share_cache = st.session_state.setdefault("share_token_cache", {})
_export_col, _share_col, _import_col, _snapshot_col, _ = st.columns([1, 1, 1, 1, 2])
with _export_col:
    with st.popover("📤 Dışa aktar", width="stretch"):
        st.download_button("📊 Excel", _excel_bytes, f"nobet_{yil}_{ay:02d}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           on_click="ignore", width="stretch", disabled=not EXCEL_AVAILABLE)
        st.download_button("🖨️ Yazdırılabilir sayfa", _html_bytes, f"nobet_{yil}_{ay:02d}.html", "text/html",
                           on_click="ignore", width="stretch")
        st.download_button("🖼️ Resim (PNG)", _png_bytes, f"nobet_{yil}_{ay:02d}.png", "image/png",
                           on_click="ignore", width="stretch")
        # BOM'lu UTF-8: Excel'de Ş, ğ, ı gibi harfler bozulmadan açılır.
        st.download_button("📄 CSV", df_liste.to_csv(index=False).encode('utf-8-sig'),
                           f"nobet_{yil}_{ay:02d}.csv", "text/csv",
                           on_click="ignore", width="stretch")
        st.markdown("**💬 WhatsApp metni**")
        st.caption("Sağ üstteki simgeyle kopyalayıp yapıştırın. Hafta sonu ve tatiller kalın görünür.")
        st.code(
            build_share_text(
                liste_rows, liste_roles, yil, ay,
                special_days=[d["day_num"] for d in gun_detaylari.values() if d["weekend"] or d["holiday"]],
            ),
            language=None,
        )
with _share_col:
    with st.popover("🔗 Paylaş", width="stretch"):
        if not _db_ready:
            st.caption("Paylaşım için veritabanı bağlantısı gerekli.")
        else:
            # Tek kalıcı bağlantı (yıl 0, ay 0): bütün ayları gösterir.
            if "all" not in _share_cache:
                _share_cache["all"] = get_share_link(current_owner, 0, 0)
            _token = _share_cache["all"]
            st.caption(
                "Ekip arkadaşlarınız bu tek bağlantıyla giriş yapmadan bütün ayların listesini görebilir; "
                "her ay yeni bağlantı göndermeniz gerekmez. Kendi adlarını seçip nöbetlerini takvimlerine ekleyebilirler."
            )
            if _token:
                st.code(f"{_app_base_url()}?paylas={_token}", language=None)
                if st.button("🔄 Yeni bağlantı oluştur (eskileri çalışmaz)", width="stretch"):
                    # Eski aylık bağlantılar da kapanır.
                    if delete_all_share_links(current_owner):
                        _share_cache["all"] = create_share_link(current_owner, 0, 0)
                    st.rerun()
                if st.button("🚫 Paylaşımı kapat", width="stretch"):
                    if delete_all_share_links(current_owner):
                        _share_cache["all"] = None
                    st.rerun()
            elif st.button("🔗 Bağlantı oluştur", type="primary", width="stretch"):
                _share_cache["all"] = create_share_link(current_owner, 0, 0)
                if not _share_cache["all"]:
                    st.error("Bağlantı oluşturulamadı.")
                else:
                    st.rerun()

with _import_col:
    with st.popover("📂 Excel Yükle", width="stretch"):
        st.caption("Daha önce indirilen Excel dosyasını veya aynı formatta hazırladığınız listeyi yükleyin.")
        for _level, _msg in st.session_state.pop("excel_import_messages", []):
            getattr(st, _level)(_msg)
        _uploaded = st.file_uploader("Excel dosyası seçin (.xlsx)", type=["xlsx"], key="excel_import_uploader", label_visibility="collapsed")
        # Dosya kutuda kaldıkça sayfa her yenilendiğinde yeniden içe aktarılıp
        # sonsuz döngüye giriyordu (elle yapılan değişiklikler de eziliyordu).
        # Her yüklenen dosya yalnızca bir kez işlenir.
        _upload_id = (
            getattr(_uploaded, "file_id", None) or (_uploaded.name, _uploaded.size)
            if _uploaded is not None else None
        )
        if _uploaded is not None and _upload_id != st.session_state.get("excel_import_done"):
            st.session_state.excel_import_done = _upload_id
            _msgs = []
            _imp = None
            if _uploaded.size > EXCEL_IMPORT_MAX_BYTES:
                # Uygulamanın kendi Excel'i birkaç KB; büyük dosya sunucuyu yorar.
                _msgs.append(("error", "Dosya çok büyük (en fazla 2 MB). Uygulamadan indirilen Excel'i yükleyin."))
            else:
                try:
                    _xl = pd.ExcelFile(_uploaded)
                    _sheet = "Günlük Liste" if "Günlük Liste" in _xl.sheet_names else _xl.sheet_names[0]
                    _imp = import_schedule_table(
                        # Bir ay en fazla 31 gün; fazlası okunmaz.
                        pd.read_excel(_xl, sheet_name=_sheet, nrows=EXCEL_IMPORT_MAX_ROWS),
                        isimler, sutunlar, gun_detaylari, role_names, yil, ay,
                    )
                except Exception:
                    _msgs.append(("error",
                        "Dosya okunamadı. Lütfen uygulamadan indirilen Excel formatını "
                        "('Tarih' + görev sütunları) kullandığınızdan emin olun."))
            if _imp is None:
                pass
            elif not _imp["format_ok"]:
                _msgs.append(("error", "Dosya formatı uyumsuz: 'Tarih' sütunu ve en az bir görev sütunu gerekli."))
            elif _imp["date_mismatch"]:
                _msgs.append(("warning",
                    f"⚠️ Excel tarihleri seçili dönemle uyuşmuyor ({ay:02d}/{yil}). "
                    f"{_imp['date_mismatch']['day']} günü için farklı ay/yıl veya geçersiz gün "
                    "bulundu; yükleme iptal edildi."))
            elif _imp["matched"] == 0:
                _msgs.append(("warning",
                    "Eşleşen gün bulunamadı; Excel tarihleri okunamadı. Tarih sütununda "
                    "gün/ay/yıl bilgilerinin bulunduğunu kontrol edin."
                    if _imp["unreadable_rows"] else
                    f"Eşleşen gün bulunamadı. Excel tarihleri seçili dönemle ({ay:02d}/{yil}) "
                    "uyumlu değil; yükleme iptal edildi."))
            else:
                save_undo_state()
                st.session_state.schedule_bool = _imp["schedule"]
                st.session_state.locked_cells = set()
                st.session_state.cached_rows_liste = _imp["rows"]
                st.session_state.cached_role_names = role_names
                st.session_state.cached_key = (yil, ay, tuple(isimler), tuple(sutunlar), tuple(role_names))
                st.session_state.cached_role_schedule = _imp["schedule"].copy(deep=True)
                st.session_state.should_regenerate_assignments = False
                # Kayıt, sayfa yenilenince otomatik kayıtla yapılır (hata olursa orada görünür).
                _msgs.append(("success", f"{_imp['matched']}/{len(sutunlar)} gün yüklendi."))
                if _imp["skipped_names"]:
                    _msgs.append(("warning",
                        f"⚠️ {len(_imp['skipped_names'])} isim mevcut ekipte bulunamadığı için "
                        f"atlandı: {', '.join(_imp['skipped_names'])}"))
                if _imp["matched"] < len(sutunlar):
                    _msgs.append(("warning",
                        f"⚠️ Excel'de {len(sutunlar)} günden {_imp['matched']} gün eşleşti; "
                        f"{len(sutunlar) - _imp['matched']} gün boş bırakıldı."))
                if _imp["unreadable_rows"]:
                    _msgs.append(("warning", f"⚠️ {_imp['unreadable_rows']} satırın tarihi okunamadığı için atlandı."))
            st.session_state.excel_import_messages = _msgs
            st.rerun()

with _snapshot_col:
    if _db_ready:
        # Tıklama sayfa yeniden çizilmeden önce işlenir; yedek, bu çalışmada
        # son değişiklik kaydedildikten sonra alınır (_run_requested_snapshot).
        st.button(
            "💾 Yedekle", width="stretch", key="snapshot_take_btn",
            help=f"{TR_AY_ADLARI[ay]} {yil} ayını yedekler. Her ayın tek yedeği olur; tekrar basarsanız eskisinin yerine geçer.",
            on_click=lambda: st.session_state.update(snapshot_requested=True),
        )

st.divider()

tab_liste, tab_analiz = st.tabs(["📋 Günlük Liste", "📊 Analiz"])

with tab_liste:
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

with tab_analiz:
    # ── Analiz — yatay dağılım ───────────────────────────────────────────────────

    # Satır 1: Nöbet Yükü  |  Kişisel Detay
    an1, an2 = st.columns(2)

    with an1:
        st.markdown("**Nöbet Yükü**")
        if history_special:
            _hist_text = " · ".join(
                f"{_p} {_prev_ctx['history_counts'].get(_p, 0)}" for _p in isimler
            )
            st.caption(
                f"⚖️ Önceki {_prev_ctx['months']} ayın hafta sonu/tatil nöbetleri dengelemede "
                f"hesaba katılıyor: {_hist_text}"
            )
        if history_total:
            _total_text = " · ".join(
                f"{_p} {'+' if _v > 0 else ''}{_v:g}"
                for _p, _v in sorted(history_total.items(), key=lambda _item: -_item[1])
            )
            st.caption(
                f"⚖️ Önceki {_prev_ctx['months']} ayın toplam nöbetleri (ortalamaya göre fark): "
                f"{_total_text}. Fazlası olan bu ay \"1 fazla\" alanlardan olmaz."
            )
        if history_holiday:
            _holiday_text = " · ".join(
                f"{_p} {_prev_ctx['holiday_counts'].get(_p, 0)}" for _p in isimler
            )
            st.caption(
                f"🎉 Son 12 aydaki bayram/tatil nöbetleri: {_holiday_text}. "
                "Bu ayın tatil günleri azı olana verilir."
            )
        st.dataframe(
            df_stats_load.style.background_gradient(cmap="Blues", subset=["Toplam"] + list(role_names))
                               .background_gradient(cmap="Oranges", subset=["Özel", "Cuma", "Tatil"]),
            width="stretch"
        )
        if st.button("🤖 Görev Yerlerini Otomatik Dağıt", width="stretch", help="Mevcut nöbetçileri değiştirmeden sadece görev yerlerini (Rollerini) otomatik olarak dengeler."):
            if 'cached_rows_liste' in st.session_state and st.session_state.cached_rows_liste:
                save_undo_state()
                rows_liste_to_redist = st.session_state.cached_rows_liste
                first_role_counts_to_redist = {i: 0 for i in isimler}
                new_rows_liste = []
                for row in rows_liste_to_redist:
                    nobetciler = []
                    for rn in role_names:
                        p = row.get(rn, "-")
                        if p != "-" and p in isimler: nobetciler.append(p)
                    random.shuffle(nobetciler)
                    nobetciler.sort(key=lambda x: first_role_counts_to_redist.get(x, 0))
                    new_row = {"Tarih": row["Tarih"]}
                    for idx, rn in enumerate(role_names):
                        if len(nobetciler) > idx:
                            p = nobetciler[idx]
                            new_row[rn] = p
                            if idx == 0:
                                first_role_counts_to_redist[p] = first_role_counts_to_redist.get(p, 0) + 1
                        else:
                            new_row[rn] = "-"
                    new_rows_liste.append(new_row)
                st.session_state.cached_rows_liste = new_rows_liste
                st.session_state.cached_first_role_counts = first_role_counts_to_redist
                # Nöbetçiler değişmez; çizelge yeniden kurulmaz (görev sayısından
                # fazla atanmış kişiler sessizce siliniyordu).
                st.session_state.should_regenerate_assignments = False
                st.toast("Görev yerleri yeniden dağıtıldı!", icon="🤖")
                # Üstteki Günlük Liste ve takvim yeni rolleri hemen göstersin.
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
        st.caption(_mesai_text + ".")
        st.dataframe(
            df_stats_finance.style.background_gradient(cmap="Reds", subset=["FM", "Ücret (TL)"])
                                  .format({"Ücret (TL)": "₺ {:,.2f}"}),
            width="stretch"
        )

    with an4:
        st.markdown("**Eşleşme Matrisi**")
        st.dataframe(pair_display, width="stretch")

    # Satır 3: Ekip tercih özeti (tercih/kaçın/izin işaretlerinin sonucu)
    st.markdown("**Tercih Özeti**")
    _pref_summary = preference_summary(edited_df, st.session_state.pref_df, isimler)
    if _pref_summary is None:
        st.caption("Tabloda işaretli tercih, kaçınma ya da izin günü yok.")
    else:
        st.caption(
            "Her kişinin yeşil (tercih) günlerinin kaçında nöbet tuttuğu, sarı (kaçın) günlerine "
            "kaç nöbet düştüğü ve izinli gününe nöbet yazılıp yazılmadığı."
        )
        st.dataframe(_pref_summary, width="stretch")
