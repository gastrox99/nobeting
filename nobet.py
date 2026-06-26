# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import random
import calendar
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import date, datetime
import numpy as np
import time
import json
from db import init_db, save_schedule, load_schedule, list_schedules, delete_schedule
from streamlit_local_storage import LocalStorage
from nobet_core import run_scheduling_core

# Excel export
try:
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# Initialize database on app start
_db_ready = init_db()

# Sayfa Ayarları
st.set_page_config(page_title="Adil Nöbet v98 (AI Simulation)", layout="wide")

if not _db_ready:
    st.warning("⚠️ Veritabanına bağlanılamadı. Kaydet/Yükle ve otomatik kayıt özellikleri şu an çalışmayabilir.")

# --- localStorage for team list persistence ---
local_storage = LocalStorage()

# Load saved team from localStorage on first run
if 'localStorage_loaded' not in st.session_state:
    st.session_state.localStorage_loaded = False
    saved_team = local_storage.getItem("nobet_team")
    if saved_team:
        st.session_state.isimler_text = saved_team

# --- GLOBAL RESPONSIVE CSS ---
st.markdown("""
<style>
/* Schedule grid wrapper - horizontal scroll on mobile */
.schedule-grid-wrapper {
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
    [data-testid="stExpander"] [data-testid="column"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    
    /* Grid wrapper - enable scroll */
    .schedule-grid-wrapper {
        max-width: 100vw;
        overflow-x: scroll !important;
    }
    .schedule-grid-wrapper [data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        min-width: max-content;
    }
    .schedule-grid-wrapper [data-testid="column"] {
        flex-shrink: 0 !important;
        min-width: 32px !important;
    }
    
    /* Make buttons smaller on mobile */
    .stButton > button {
        padding: 2px 4px !important;
        min-height: 24px !important;
        font-size: 9px !important;
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
}

/* Tablet adjustments */
@media (max-width: 1024px) and (min-width: 769px) {
    [data-testid="stExpander"] [data-testid="column"] {
        flex: 1 1 45% !important;
        min-width: 45% !important;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-size: 2.5rem;'>Nöbet Yönetimi</h1>", unsafe_allow_html=True)

# --- YARDIMCI FONKSİYONLAR ---
def parse_unwanted_days(text_input, max_day):
    if not text_input or pd.isna(text_input): return []
    days = set()
    parts = str(text_input).split(',')
    for part in parts:
        part = part.strip()
        if not part: continue
        try:
            if '-' in part:
                start, end = map(int, part.split('-'))
                start = max(1, start); end = min(max_day, end)
                if start <= end: days.update(range(start, end + 1))
            else:
                d = int(part)
                if 1 <= d <= max_day: days.add(d)
        except ValueError: continue
    return list(days)

# --- VALIDATION ---
def validate_inputs(isimler, yil, ay, gun_sayisi, tatil_gunleri, nobet_ucreti, min_bosluk, kisi_sayisi=2):
    """Validate all inputs and return (is_valid, errors, warnings)"""
    errors = []
    warnings = []
    
    # Team validation
    if not isimler or len(isimler) == 0:
        errors.append("❌ En az 1 kişi ekleyin")
    elif len(isimler) > 50:
        errors.append("❌ Maksimum 50 kişi ekleyebilirsiniz")
    
    # Check for duplicate names
    if len(isimler) != len(set(isimler)):
        errors.append("❌ Aynı isimde 2 kişi olamaz")
    
    # Pay validation
    if nobet_ucreti < 0:
        errors.append("❌ Saatlik ücret negatif olamaz")
    elif nobet_ucreti == 0:
        warnings.append("⚠️ Saatlik ücret 0 TL")
    
    # Holiday validation
    invalid_holidays = [h for h in tatil_gunleri if h < 1 or h > gun_sayisi]
    if invalid_holidays:
        errors.append(f"❌ Geçersiz tatil günleri: {invalid_holidays}")
    
    # Rest period validation
    if min_bosluk < 0 or min_bosluk > 7:
        errors.append("❌ Dinlenme süresi 0-7 gün arasında olmalı")
    
    # Feasibility warnings
    working_days = gun_sayisi - len(tatil_gunleri)
    total_positions_needed = working_days * kisi_sayisi
    team_size = len(isimler)
    
    if team_size < kisi_sayisi:
        errors.append(f"❌ {kisi_sayisi} kişi nöbet için en az {kisi_sayisi} kişi gerekli")
    elif team_size > 0 and total_positions_needed > team_size * 30:
        avg_per_person = total_positions_needed / team_size
        warnings.append(f"⚠️ Her kişiye ortalama {avg_per_person:.1f} nöbet düşecek (çok fazla)")
    elif team_size > 0 and total_positions_needed < team_size:
        warnings.append(f"⚠️ Nöbetleri dağıtmak için çok fazla kişi var ({team_size} kişi, {total_positions_needed} pozisyon)")
    
    return len(errors) == 0, errors, warnings

def convert_df_to_png(df):
    fig, ax = plt.subplots(figsize=(8, len(df) * 0.4 + 1))
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
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    return buf.getvalue()

def convert_df_to_excel(df_liste, df_stats_load, df_stats_finance):
    """Convert dataframes to Excel with multiple sheets"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_liste.to_excel(writer, sheet_name='Günlük Liste', index=False)
        df_stats_load.to_excel(writer, sheet_name='Nöbet Yükü')
        df_stats_finance.to_excel(writer, sheet_name='Ücret Özeti')
        
        # Format worksheets
        workbook = writer.book
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

def create_print_html(df_liste, df_stats_load, yil, ay):
    """Create print-friendly HTML"""
    ay_isimleri = {1:"Ocak", 2:"Şubat", 3:"Mart", 4:"Nisan", 5:"Mayıs", 6:"Haziran",
                   7:"Temmuz", 8:"Ağustos", 9:"Eylül", 10:"Ekim", 11:"Kasım", 12:"Aralık"}
    
    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Nöbet Listesi - {ay_isimleri[ay]} {yil}</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            h1 {{ text-align: center; color: #333; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .weekend {{ background-color: #e3f2fd !important; }}
            @media print {{
                body {{ margin: 0; }}
                button {{ display: none; }}
            }}
        </style>
    </head>
    <body>
        <h1>Nöbet Listesi - {ay_isimleri[ay]} {yil}</h1>
        <table>
            <tr>{''.join(f'<th>{col}</th>' for col in df_liste.columns)}</tr>
    """
    
    for _, row in df_liste.iterrows():
        css_class = 'weekend' if 'Cmt' in str(row.get('Tarih', '')) or 'Paz' in str(row.get('Tarih', '')) else ''
        html += f"<tr class='{css_class}'>{''.join(f'<td>{val}</td>' for val in row)}</tr>"
    
    html += """
        </table>
        <h2>Nöbet Yükü Özeti</h2>
        <table>
            <tr><th>İsim</th>""" + ''.join(f'<th>{col}</th>' for col in df_stats_load.columns) + "</tr>"
    
    for idx, row in df_stats_load.iterrows():
        html += f"<tr><td><strong>{idx}</strong></td>{''.join(f'<td>{val}</td>' for val in row)}</tr>"
    
    html += """
        </table>
        <p style="text-align: center; color: #666; margin-top: 30px;">
            Oluşturulma: """ + datetime.now().strftime("%d.%m.%Y %H:%M") + """
        </p>
    </body>
    </html>
    """
    return html

def _current_state_snapshot():
    """Geri al/yinele için hem nöbet hem tercih hem de görev yeri atamalarını kaydeder."""
    return {
        'schedule': st.session_state.schedule_bool.copy() if 'schedule_bool' in st.session_state else None,
        'pref': st.session_state.pref_df.copy() if 'pref_df' in st.session_state else None,
        'rows_liste': list(st.session_state.cached_rows_liste) if st.session_state.get('cached_rows_liste') else None,
        'role_names': list(st.session_state.cached_role_names) if st.session_state.get('cached_role_names') else None,
        'forbidden_pairs_text': st.session_state.get('forbidden_pairs_text', ''),
        'limits_text': st.session_state.get('limits_text', ''),
        'rol_isimleri': list(st.session_state.get('rol_isimleri', [])),
    }

def save_undo_state(schedule_df=None):
    """Save current state (schedule + preferences) for undo"""
    if 'undo_history' not in st.session_state:
        st.session_state.undo_history = []
    if 'redo_history' not in st.session_state:
        st.session_state.redo_history = []
    
    # Limit history to 10 states
    if len(st.session_state.undo_history) >= 10:
        st.session_state.undo_history.pop(0)
    
    st.session_state.undo_history.append(_current_state_snapshot())
    st.session_state.redo_history = []  # Clear redo on new action

# --- ANA ALGORİTMA: nobet_core.run_scheduling_core kullanır ---
def run_scheduling_algorithm_v98(isimler, sutunlar, df_unwanted_bool, gun_detaylari, min_bosluk, kisi_sayisi, forbidden_pairs=None, person_limits=None, df_preferred=None):
    progress_bar = st.progress(0)
    def _progress(n):
        if n % 10 == 0:
            progress_bar.progress(n)
    best_schedule, _ = run_scheduling_core(
        isimler, sutunlar, df_unwanted_bool, gun_detaylari,
        min_bosluk, kisi_sayisi, forbidden_pairs, person_limits, df_preferred,
        simulation_count=100, progress_callback=_progress
    )
    progress_bar.empty()
    if best_schedule is not None:
        st.session_state.schedule_bool = best_schedule
    st.toast("100 Simülasyon yapıldı. En adil sonuç seçildi!", icon="🧠")


# --- AYARLAR (Ana Sayfada Açılır Panel) ---
# Check if team exists to determine if settings should be expanded
prev_isimler = st.session_state.get('isimler_cache', [])
settings_expanded = len(prev_isimler) == 0

with st.expander("⚙️ Ayarlar", expanded=settings_expanded):
    set_col1, set_col2, set_col3 = st.columns([2, 1, 1])
    
    with set_col1:
        isimler_input = st.text_area(
            "👥 Ekip (virgülle ayırın):",
            value=st.session_state.get("isimler_text", ""),
            height=100,
            placeholder="Ali, Ayşe, Mehmet, Fatma"
        )
        st.session_state.isimler_text = isimler_input
        isimler = [x.strip() for x in isimler_input.split(",") if x.strip()]
        st.session_state.isimler_cache = isimler
        
        # Save team to localStorage (always sync, even when cleared)
        local_storage.setItem("nobet_team", isimler_input)
    
    with set_col2:
        _now = datetime.now()
        if _now.month == 12:
            current_year = _now.year + 1
            current_month = 1
        else:
            current_year = _now.year
            current_month = _now.month + 1
        yil = st.number_input("📅 Yıl", 2024, 2030, value=st.session_state.get('yil_val', current_year), key='yil_input_direct')
        st.session_state.yil_val = yil
        
        ay_options = list(range(1, 13))
        ay = st.selectbox("📆 Ay", ay_options, index=ay_options.index(st.session_state.get('ay_val', current_month)), key='ay_input_direct')
        st.session_state.ay_val = ay
        
        gun_sayisi = calendar.monthrange(yil, ay)[1]
        kişi_sayısı = st.slider("👤 Nöbet Başına Kişi:", 1, 5, value=st.session_state.get('kisi_sayisi_val', 2), key='kisi_slider_direct')
        st.session_state.kisi_sayisi_val = kişi_sayısı
    
    with set_col3:
        min_bosluk = st.slider("⏸️ Dinlenme (gün):", 0, 3, 1)
        tatil_gunleri = parse_unwanted_days(st.text_input("🎉 Tatiller:", placeholder="1,2,5-10,23"), gun_sayisi)
        nobet_ucreti = st.number_input("💰 Saat Ücreti (TL):", value=1.0)
    
    # Additional settings row
    st.divider()
    adv_col1, adv_col2, adv_col3 = st.columns(3)
    
    with adv_col1:
        role_names_input = st.text_input(
            "🏷️ Görev İsimleri:",
            value=st.session_state.get("role_names_text", ""),
            placeholder=f"Örn: AYB, GYB"
        )
        st.session_state.role_names_text = role_names_input
        
        if role_names_input.strip():
            role_names = [r.strip() for r in role_names_input.split(",") if r.strip()]
        else:
            role_names = []
        while len(role_names) < kişi_sayısı:
            role_names.append(f"Görev{len(role_names)+1}")
        role_names = role_names[:kişi_sayısı]
        st.session_state.rol_isimleri = role_names
    
    with adv_col2:
        forbidden_input = st.text_area(
            "🚫 Birlikte Çalışamayan:",
            value=st.session_state.get("forbidden_pairs_text", ""),
            height=68,
            placeholder="Ali-Ayşe, Mehmet-Fatma"
        )
        st.session_state.forbidden_pairs_text = forbidden_input
    
    with adv_col3:
        limits_text = st.text_area(
            "📊 Kişisel Limitler:",
            value=st.session_state.get("limits_text", ""),
            height=68,
            placeholder="Ali:5-10\nAyşe:3-8"
        )
        st.session_state.limits_text = limits_text
    
    # Parse forbidden pairs
    forbidden_pairs = set()
    _fp_unknown = []
    if forbidden_input.strip():
        all_pairs = []
        for line in forbidden_input.strip().split('\n'):
            all_pairs.extend(line.split(','))
        for pair_str in all_pairs:
            pair_str = pair_str.strip()
            if '-' in pair_str:
                parts = pair_str.split('-', 1)
                if len(parts) == 2:
                    p1, p2 = parts[0].strip(), parts[1].strip()
                    if p1 and p2:
                        forbidden_pairs.add(tuple(sorted((p1, p2))))
                        for _n in (p1, p2):
                            if _n and _n not in isimler:
                                _fp_unknown.append(_n)
    st.session_state.forbidden_pairs = forbidden_pairs
    st.session_state.fp_unknown_names = sorted(set(_fp_unknown))
    if _fp_unknown:
        st.warning(f"⚠️ Yasak çiftte ekip listesinde olmayan isim(ler): **{', '.join(sorted(set(_fp_unknown)))}** — kısıtlama uygulanmayacak.")

    # Parse limits
    person_limits = {}
    _lim_unknown = []
    if limits_text.strip():
        for line in limits_text.strip().split('\n'):
            if ':' in line:
                parts = line.split(':')
                name = parts[0].strip()
                if len(parts) == 2 and '-' in parts[1]:
                    try:
                        min_val, max_val = map(int, parts[1].split('-'))
                        person_limits[name] = {'min': min_val, 'max': max_val}
                        if name not in isimler:
                            _lim_unknown.append(name)
                    except ValueError:
                        pass
    st.session_state.person_limits = person_limits
    st.session_state.lim_unknown_names = sorted(set(_lim_unknown))
    if _lim_unknown:
        st.warning(f"⚠️ Kişisel limitte ekip listesinde olmayan isim(ler): **{', '.join(sorted(set(_lim_unknown)))}** — limit uygulanmayacak.")
    

if not isimler: st.stop()

# --- GÜN VERİLERİ ---
sutunlar = [] 
gun_detaylari = {} 
ozel_gun_sayisi = 0
tr_gunler = {0:"Pzt", 1:"Sal", 2:"Çar", 3:"Per", 4:"Cum", 5:"Cmt", 6:"Paz"}
sutunlar_display = []

for g in range(1, gun_sayisi + 1):
    dt = date(yil, ay, g)
    baslik = f"{g} {tr_gunler[dt.weekday()]}"
    sutunlar.append(baslik)
    is_we = dt.weekday() >= 5
    is_hol = g in tatil_gunleri
    if is_we or is_hol: ozel_gun_sayisi += 1
    full_d = dt.strftime("%d.%m.%Y") + " " + tr_gunler[dt.weekday()]
    week_val = (dt.toordinal() - dt.weekday()) // 7
    gun_detaylari[baslik] = {"weekend": is_we, "holiday": is_hol, "day_num": g, "full_date": full_d, "week": week_val}
    disp = str(g)
    if is_hol: disp = f"🚨 {g}"
    elif is_we: disp = f"🏖️ {g}"
    sutunlar_display.append(disp)

calisma_gunu = gun_sayisi - ozel_gun_sayisi
zorunlu_saat = calisma_gunu * 8

# --- SESSION ---
if 'schedule_bool' not in st.session_state:
    # Sayfa yenilenmesinde otomatik geri yükle
    _auto_name = f"Otomatik_{yil}_{ay:02d}"
    try:
        _, _saved_df, _saved_pref = load_schedule(_auto_name, yil, ay)
    except Exception:
        _saved_df = None
        _saved_pref = None
    if _saved_df is not None and not _saved_df.empty:
        _new_sched = pd.DataFrame(False, index=isimler, columns=sutunlar)
        # Match by day number (column adı yıla göre değişse de gün no sabit)
        _old_day_map = {}
        for _oc in _saved_df.columns:
            try:
                _old_day_map[int(str(_oc).split()[0])] = _oc
            except (ValueError, IndexError):
                pass
        for _p in isimler:
            if _p in _saved_df.index:
                for _c in sutunlar:
                    _dnum = gun_detaylari[_c]['day_num']
                    _oc = _old_day_map.get(_dnum)
                    if _oc and _oc in _saved_df.columns:
                        try:
                            _new_sched.at[_p, _c] = bool(_saved_df.at[_p, _oc])
                        except Exception:
                            pass
        st.session_state.schedule_bool = _new_sched
        st.session_state.should_regenerate_assignments = True
        # Restore preferences if available
        if _saved_pref is not None:
            try:
                _new_pref = pd.DataFrame(0, index=isimler, columns=sutunlar)
                for _p in isimler:
                    if _p in _saved_pref.index:
                        for _c in sutunlar:
                            _dnum = gun_detaylari[_c]['day_num']
                            _oc = _old_day_map.get(_dnum)
                            if _oc and _oc in _saved_pref.columns:
                                try:
                                    _new_pref.at[_p, _c] = int(_saved_pref.at[_p, _oc])
                                except Exception:
                                    pass
                st.session_state.pref_df = _new_pref
            except Exception:
                pass
    else:
        st.session_state.schedule_bool = pd.DataFrame(False, index=isimler, columns=sutunlar)
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
        st.session_state.cached_rows_liste = None  # Reset cache

if 'inputs' not in st.session_state: st.session_state.inputs = {i: "" for i in isimler}
if 'cached_rows_liste' not in st.session_state: st.session_state.cached_rows_liste = None
if 'cached_ayb_counts' not in st.session_state: st.session_state.cached_ayb_counts = None
if 'last_edited_hash' not in st.session_state: st.session_state.last_edited_hash = None
if 'should_regenerate_assignments' not in st.session_state: st.session_state.should_regenerate_assignments = False
if 'forbidden_pairs_text' not in st.session_state: st.session_state.forbidden_pairs_text = ""
if 'forbidden_pairs' not in st.session_state: st.session_state.forbidden_pairs = set()
if 'undo_history' not in st.session_state: st.session_state.undo_history = []
if 'redo_history' not in st.session_state: st.session_state.redo_history = []
if 'last_auto_save' not in st.session_state: st.session_state.last_auto_save = time.time()
if 'auto_save_failed' not in st.session_state: st.session_state.auto_save_failed = False
if 'sim_pending_confirm' not in st.session_state: st.session_state.sim_pending_confirm = False
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

# --- KİŞİ RENK PALETİ (20 pastel ton) ---
PASTEL_COLORS = [
    "#AED6F1", "#A9DFBF", "#F9E79F", "#F5CBA7", "#D2B4DE",
    "#F1948A", "#A3E4D7", "#FAD7A0", "#ABEBC6", "#D7BDE2",
    "#7FB3D3", "#82E0AA", "#F8C471", "#F0A987", "#C39BD3",
    "#7DCEA0", "#85C1E9", "#F7DC6F", "#BB8FCE", "#F0B27A",
]
person_colors = {isim: PASTEL_COLORS[i % len(PASTEL_COLORS)] for i, isim in enumerate(isimler)}
today_date = date.today()

# --- BİRLEŞİK TABLO ---
st.header("📅 Nöbet Çizelgesi")

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

# Initialize edit mode
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = "tercih"

if 'selected_cal_day' not in st.session_state:
    st.session_state.selected_cal_day = None

# --- ÇAKIŞMA HÜCRELERİ (grid vurgulaması için önceden hesapla) ---
conflict_cells = set()  # (kişi, col) çiftleri
_last_cf = {i: -10 for i in isimler}
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
                (3, "🟥Yok")
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
            st.info("Tıklayarak nöbet ekle/kaldır ✓")
            selected_label = ""

    # Quick actions row
    action_cols = st.columns(4)
    with action_cols[0]:
        if st.button("⚡ Simülasyon", type="primary", use_container_width=True):
            _has_edits = bool(st.session_state.schedule_bool.values.sum())
            if _has_edits:
                st.session_state.sim_pending_confirm = True
                st.rerun()
            else:
                st.session_state.run_simulation = True
    with action_cols[1]:
        if st.button("🔄 Sıfırla", use_container_width=True):
            st.session_state.pref_df = pd.DataFrame(0, index=isimler, columns=sutunlar)
            st.session_state.schedule_bool = pd.DataFrame(False, index=isimler, columns=sutunlar)
            st.rerun()
    with action_cols[2]:
        if st.button("↩️ Geri", use_container_width=True, disabled=len(st.session_state.undo_history)==0):
            if st.session_state.undo_history:
                st.session_state.redo_history.append(_current_state_snapshot())
                _snap = st.session_state.undo_history.pop()
                if isinstance(_snap, dict):
                    if _snap.get('schedule') is not None:
                        st.session_state.schedule_bool = _snap['schedule']
                    if _snap.get('pref') is not None:
                        st.session_state.pref_df = _snap['pref']
                    if _snap.get('rows_liste') is not None:
                        st.session_state.cached_rows_liste = _snap['rows_liste']
                        st.session_state.cached_role_names = _snap.get('role_names')
                        st.session_state.should_regenerate_assignments = False
                    else:
                        st.session_state.cached_rows_liste = None
                        st.session_state.should_regenerate_assignments = True
                else:
                    st.session_state.schedule_bool = _snap
                    st.session_state.cached_rows_liste = None
                    st.session_state.should_regenerate_assignments = True
                st.rerun()
    with action_cols[3]:
        if st.button("↪️ İleri", use_container_width=True, disabled=len(st.session_state.redo_history)==0):
            if st.session_state.redo_history:
                st.session_state.undo_history.append(_current_state_snapshot())
                _snap = st.session_state.redo_history.pop()
                if isinstance(_snap, dict):
                    if _snap.get('schedule') is not None:
                        st.session_state.schedule_bool = _snap['schedule']
                    if _snap.get('pref') is not None:
                        st.session_state.pref_df = _snap['pref']
                    if _snap.get('rows_liste') is not None:
                        st.session_state.cached_rows_liste = _snap['rows_liste']
                        st.session_state.cached_role_names = _snap.get('role_names')
                        st.session_state.should_regenerate_assignments = False
                    else:
                        st.session_state.cached_rows_liste = None
                        st.session_state.should_regenerate_assignments = True
                else:
                    st.session_state.schedule_bool = _snap
                    st.session_state.cached_rows_liste = None
                    st.session_state.should_regenerate_assignments = True
                st.rerun()

    # Simülasyon onay dialogu
    if st.session_state.sim_pending_confirm:
        st.warning("⚠️ Mevcut çizelgede el ile yapılan değişiklikler var. Simülasyon bunları silip yeniden hesaplayacak.")
        _conf_cols = st.columns(2)
        with _conf_cols[0]:
            if st.button("✅ Devam Et — Simülasyonu Çalıştır", type="primary", use_container_width=True):
                st.session_state.sim_pending_confirm = False
                st.session_state.run_simulation = True
                st.rerun()
        with _conf_cols[1]:
            if st.button("❌ İptal", use_container_width=True):
                st.session_state.sim_pending_confirm = False
                st.rerun()

    # Dynamic column background CSS
    col_bg_css_parts = []
    for _i, _col in enumerate(sutunlar):
        _idx = _i + 2
        _info = gun_detaylari[_col]
        _is_today = (date(yil, ay, _info['day_num']) == today_date)
        if _is_today:
            col_bg_css_parts.append(
                f".schedule-grid [data-testid='stHorizontalBlock'] > [data-testid='column']:nth-child({_idx})"
                f"{{ background:rgba(254,243,199,0.75)!important; border-radius:6px; outline:2px solid #f59e0b; }}"
            )
        elif _info['holiday']:
            col_bg_css_parts.append(
                f".schedule-grid [data-testid='stHorizontalBlock'] > [data-testid='column']:nth-child({_idx})"
                f"{{ background:rgba(254,226,226,0.55)!important; border-radius:6px; }}"
            )
        elif _info['weekend']:
            col_bg_css_parts.append(
                f".schedule-grid [data-testid='stHorizontalBlock'] > [data-testid='column']:nth-child({_idx})"
                f"{{ background:rgba(219,234,254,0.5)!important; border-radius:6px; }}"
            )
    col_bg_css = "\n".join(col_bg_css_parts)

    st.markdown(f"""
<style>
.schedule-grid [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(1) {{
    background: #f8fafc !important;
    border-right: 2px solid #e2e8f0;
    min-width: 80px;
}}
{col_bg_css}
.schedule-grid div[data-testid="column"] {{ padding: 0 1px !important; }}
.schedule-grid .stButton > button {{
    padding: 0px !important;
    min-height: 30px !important;
    font-size: 14px !important;
    border-radius: 5px !important;
    line-height: 1 !important;
    width: 100% !important;
    border: 1px solid #e5e7eb !important;
    transition: transform 0.1s, box-shadow 0.1s;
}}
.schedule-grid .stButton > button:hover {{
    transform: scale(1.15);
    box-shadow: 0 2px 8px rgba(0,0,0,0.18);
    z-index: 20;
    position: relative;
}}
</style>
""", unsafe_allow_html=True)

    st.markdown('<div class="schedule-grid-wrapper"><div class="schedule-grid">', unsafe_allow_html=True)

    tr_gunler_short = {0:"Pzt", 1:"Sal", 2:"Çar", 3:"Per", 4:"Cum", 5:"Cmt", 6:"Paz"}
    header_cols = st.columns([2] + [1] * len(sutunlar))
    with header_cols[0]:
        st.markdown("<div style='font-size:13px;font-weight:700;color:#6b7280;padding:2px 4px;'>İSİM&nbsp;&nbsp;#</div>", unsafe_allow_html=True)
    for i, col in enumerate(sutunlar):
        with header_cols[i + 1]:
            _info = gun_detaylari[col]
            day_num = _info['day_num']
            is_we = _info['weekend']
            is_hol = _info['holiday']
            _is_today = (date(yil, ay, day_num) == today_date)
            day_name = tr_gunler_short[date(yil, ay, day_num).weekday()]
            if _is_today:
                num_style = "color:#b45309;font-weight:800;font-size:15px;"
                name_style = "color:#b45309;font-size:11px;"
                badge = "★"
            elif is_hol:
                num_style = "color:#c62828;font-weight:700;font-size:14px;"
                name_style = "color:#c62828;font-size:11px;"
                badge = "🎉"
            elif is_we:
                num_style = "color:#1d4ed8;font-weight:700;font-size:14px;"
                name_style = "color:#1d4ed8;font-size:11px;"
                badge = ""
            else:
                num_style = "color:#374151;font-size:14px;"
                name_style = "color:#9ca3af;font-size:11px;"
                badge = ""
            st.markdown(
                f"<div style='text-align:center;line-height:1.35;padding:2px 0;'>"
                f"<div style='{num_style}'>{badge}{day_num}</div>"
                f"<div style='{name_style}'>{day_name}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    _person_limits = st.session_state.get('person_limits', {})
    for person in isimler:
        row_cols = st.columns([2] + [1] * len(sutunlar))
        with row_cols[0]:
            count = int(st.session_state.schedule_bool.loc[person].sum()) if person in st.session_state.schedule_bool.index else 0
            p_lim = _person_limits.get(person, {})
            min_l = p_lim.get('min', 0)
            max_l = p_lim.get('max', 999)
            badge_bg = "#dc2626" if (count < min_l or (max_l < 999 and count > max_l)) else "#16a34a"
            pc = person_colors.get(person, "#e2e8f0")
            st.markdown(
                f"<div style='font-size:13px;white-space:nowrap;padding:2px 4px;line-height:1.6;'>"
                f"<span style='display:inline-block;width:11px;height:11px;border-radius:50%;"
                f"background:{pc};margin-right:4px;vertical-align:middle;border:1px solid rgba(0,0,0,0.12);'></span>"
                f"<b>{person}</b>&nbsp;"
                f"<span style='background:{badge_bg};color:#fff;border-radius:10px;padding:1px 7px;font-size:11px;font-weight:700;'>{count}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        for i, col in enumerate(sutunlar):
            with row_cols[i + 1]:
                pref_val = st.session_state.pref_df.at[person, col] if person in st.session_state.pref_df.index else 0
                is_assigned = st.session_state.schedule_bool.at[person, col] if person in st.session_state.schedule_bool.index else False
                has_conflict = (person, col) in conflict_cells
                if is_assigned:
                    if has_conflict:
                        label = "🔺"
                    elif pref_val == 1:
                        label = "✅"
                    elif pref_val == 2:
                        label = "⚠️"
                    elif pref_val == 3:
                        label = "🚫"
                    else:
                        label = "●"
                else:
                    label = "🟢" if pref_val == 1 else ("🟡" if pref_val == 2 else ("🔴" if pref_val == 3 else "○"))
                if st.button(label, key=f"g_{person}_{col}", use_container_width=True):
                    save_undo_state()
                    if st.session_state.edit_mode == "tercih":
                        st.session_state.pref_df.at[person, col] = st.session_state.paint_color
                    else:
                        st.session_state.schedule_bool.at[person, col] = not st.session_state.schedule_bool.at[person, col]
                    st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)

    legend_cols = st.columns(8)
    legends = [
        ("●", "#374151", "Atandı"),
        ("✅", "#16a34a", "Atandı+İstedi"),
        ("⚠️", "#d97706", "Atandı+Kaçın"),
        ("🚫", "#dc2626", "Atandı+Yasak"),
        ("🔺", "#dc2626", "Çakışma"),
        ("🟢", "#16a34a", "Tercih"),
        ("🟡", "#d97706", "Kaçınma"),
        ("🔴", "#dc2626", "Yasak"),
    ]
    for lc, (icon, color, desc) in zip(legend_cols, legends):
        with lc:
            st.markdown(f"<div style='text-align:center;font-size:16px;color:{color};'>{icon}<br/><span style='color:#6b7280;font-size:12px;font-weight:500;'>{desc}</span></div>", unsafe_allow_html=True)

    st.markdown('<p class="mobile-hint" style="color:#888;font-size:12px;margin:4px 0;">📱 Mobilde yana kaydırın →</p>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# TAB 2: TAKVİM GÖRÜNÜMü
# ═══════════════════════════════════════════════════════
with tab_cal:
    # Rol isimleri (Görev1, Görev2 vb.)
    _cal_role_names = st.session_state.get('rol_isimleri', [f"Görev{i+1}" for i in range(kişi_sayısı)])

    # ── Takvimden gelen görev yeri değişim isteği ────────
    _swap_dn = st.session_state.get('cal_swap_day', None)
    if _swap_dn is not None:
        if 'cal_swap_day' in st.session_state:
            del st.session_state['cal_swap_day']
        # cached_rows_liste içindeki rol isimlerini kullan (schedule ile tutarlı)
        _swap_role_names = st.session_state.get('cached_role_names', _cal_role_names)
        if st.session_state.get('cached_rows_liste'):
            _row_idx = _swap_dn - 1
            if 0 <= _row_idx < len(st.session_state.cached_rows_liste):
                _srow = dict(st.session_state.cached_rows_liste[_row_idx])
                _valid = [_srow.get(rn, "-") for rn in _swap_role_names]
                _valid = [p for p in _valid if p and p != "-" and p in isimler]
                if len(_valid) >= 2:
                    save_undo_state()
                    _valid = [_valid[-1]] + _valid[:-1]  # döngüsel kaydırma
                    _vi = 0
                    for _rn in _swap_role_names:
                        _old = _srow.get(_rn, "-")
                        if _old and _old != "-" and _old in isimler:
                            _srow[_rn] = _valid[_vi]; _vi += 1
                    st.session_state.cached_rows_liste[_row_idx] = _srow
                    st.session_state.should_regenerate_assignments = False
                    st.toast(f"{_swap_dn}. gün görev yerleri değiştirildi", icon="🔄")
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
            "<div style='padding-top:28px;font-size:12px;color:#6b7280;'>"
            "🟢 Tercih &nbsp; 🟡 Kaçın &nbsp; 🔴 Müsait Değil &nbsp; 🔄 Görev Yeri Değiştir"
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
        dp_we = sum(1 for c in dp_shifts if gun_detaylari[c]['weekend'])
        dp_hol = sum(1 for c in dp_shifts if gun_detaylari[c]['holiday'])
        dp_weekday = len(dp_shifts) - dp_we
        dp_total_h = len(dp_shifts) * 24
        dp_fm_h = max(0, dp_total_h - zorunlu_saat)
        dp_pay = dp_fm_h * nobet_ucreti
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
            f"<div style='background:linear-gradient(135deg,{dp_color}55 0%,#f8fafc 60%);"
            f"border:2px solid {dp_color};border-radius:14px;padding:16px 20px;margin-bottom:12px;'>"
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>"
            f"<span style='background:{dp_color};border-radius:50%;width:36px;height:36px;"
            f"display:inline-flex;align-items:center;justify-content:center;"
            f"font-size:16px;font-weight:800;color:#1f2937;border:2px solid rgba(0,0,0,0.1);'>"
            f"{dp[0].upper()}</span>"
            f"<span style='font-size:20px;font-weight:800;color:#1f2937;'>{dp}</span>"
            f"<span style='font-size:13px;color:#6b7280;margin-left:4px;'>"
            f"— {ay} / {yil}</span></div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:10px;'>"
            f"<div style='background:white;border-radius:10px;padding:8px 14px;min-width:90px;text-align:center;"
            f"box-shadow:0 1px 4px rgba(0,0,0,0.08);'>"
            f"<div style='font-size:22px;font-weight:800;color:#1d4ed8;'>{len(dp_shifts)}</div>"
            f"<div style='font-size:11px;color:#6b7280;'>Toplam Nöbet</div></div>"
            f"<div style='background:white;border-radius:10px;padding:8px 14px;min-width:90px;text-align:center;"
            f"box-shadow:0 1px 4px rgba(0,0,0,0.08);'>"
            f"<div style='font-size:22px;font-weight:800;color:#374151;'>{dp_weekday}</div>"
            f"<div style='font-size:11px;color:#6b7280;'>Hafta İçi</div></div>"
            f"<div style='background:white;border-radius:10px;padding:8px 14px;min-width:90px;text-align:center;"
            f"box-shadow:0 1px 4px rgba(0,0,0,0.08);'>"
            f"<div style='font-size:22px;font-weight:800;color:#1d4ed8;'>{dp_we}</div>"
            f"<div style='font-size:11px;color:#6b7280;'>Hafta Sonu</div></div>"
            f"<div style='background:white;border-radius:10px;padding:8px 14px;min-width:90px;text-align:center;"
            f"box-shadow:0 1px 4px rgba(0,0,0,0.08);'>"
            f"<div style='font-size:22px;font-weight:800;color:#dc2626;'>{dp_hol}</div>"
            f"<div style='font-size:11px;color:#6b7280;'>Tatil</div></div>"
            f"<div style='background:white;border-radius:10px;padding:8px 14px;min-width:90px;text-align:center;"
            f"box-shadow:0 1px 4px rgba(0,0,0,0.08);'>"
            f"<div style='font-size:22px;font-weight:800;color:#16a34a;'>{'%{}'.format(dp_pref_score) if dp_pref_score is not None else '—'}</div>"
            f"<div style='font-size:11px;color:#6b7280;'>Tercih Başarısı</div></div>"
            f"<div style='background:white;border-radius:10px;padding:8px 14px;min-width:90px;text-align:center;"
            f"box-shadow:0 1px 4px rgba(0,0,0,0.08);'>"
            f"<div style='font-size:22px;font-weight:800;color:#9333ea;'>{dp_pay:.0f} ₺</div>"
            f"<div style='font-size:11px;color:#6b7280;'>Tahmini Ödeme</div></div>"
            f"</div>"
            f"<div style='margin-top:10px;font-size:11px;color:#6b7280;'>"
            f"🟢 {dp_green_ok} tercih tuttu &nbsp;·&nbsp; "
            f"🟡 {dp_yellow_hit} kaçın gününde atandı &nbsp;·&nbsp; "
            f"🔴 {dp_red_hit} yasak günde atandı</div>"
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
                if inf['holiday']:
                    pill_bg = "#fee2e2"; pill_tc = "#dc2626"
                elif inf['weekend']:
                    pill_bg = "#dbeafe"; pill_tc = "#1d4ed8"
                else:
                    pill_bg = dp_color; pill_tc = "#1f2937"
                day_pills += (
                    f"<span style='display:inline-block;background:{pill_bg};color:{pill_tc};"
                    f"border-radius:20px;padding:2px 10px;font-size:11px;font-weight:600;"
                    f"margin:2px;white-space:nowrap;'>{dn2} {dn2_name}</span>"
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
            bg = "#dbeafe" if j >= 5 else "#f1f5f9"
            tc = "#1d4ed8" if j >= 5 else "#374151"
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
                        cell_bg = "#fef3c7"; border = "2px solid #f59e0b"; day_c = "#b45309"
                    elif _is_hol:
                        cell_bg = "#fee2e2"; border = "1px solid #fca5a5"; day_c = "#dc2626"
                    elif _is_we:
                        cell_bg = "#dbeafe"; border = "1px solid #bfdbfe"; day_c = "#1d4ed8"
                    else:
                        cell_bg = "#f8fafc"; border = "1px solid #e2e8f0"; day_c = "#374151"

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
                    for role_idx, p in enumerate(all_nobetciler):
                        if cal_filter_persons and p not in cal_filter_persons:
                            continue
                        pc = person_colors.get(p, '#e2e8f0')
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
                            f"outline:{outline};'>{p}</span>"
                            f"<span style='font-size:10px;color:#6b7280;white-space:nowrap;'>{rl}</span>"
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
                        if st.button("🔄", key=f"cal_swap_{dn}", help="Görev yerlerini değiştir", use_container_width=True):
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

# Handle simulation trigger
sim_clicked = st.session_state.get('run_simulation', False)
st.session_state.run_simulation = False

if sim_clicked:
    is_valid, errors, warnings = validate_inputs(isimler, yil, ay, gun_sayisi, tatil_gunleri, nobet_ucreti, min_bosluk, kişi_sayısı)

    # Min limit fizibilite kontrolü
    _sim_person_limits = st.session_state.get('person_limits', {})
    if _sim_person_limits:
        for _p, _lim in _sim_person_limits.items():
            if _p in isimler:
                _min_req = _lim.get('min', 0)
                if _min_req > 0:
                    _avail = sum(1 for _c in sutunlar if not df_unwanted.at[_p, _c])
                    if _min_req > _avail:
                        warnings.append(f"⚠️ **{_p}**: Min {_min_req} nöbet gerekli ama yalnızca {_avail} müsait gün var — limit karşılanamayabilir")

    if errors:
        st.error("🚨 Hata(lar) düzeltilmeli:")
        for err in errors:
            st.error(err)
    else:
        if warnings:
            st.warning("⚠️ Uyarı(lar):")
            for warn in warnings:
                st.warning(warn)
        
        save_undo_state(st.session_state.schedule_bool)
        
        run_scheduling_algorithm_v98(
            isimler, sutunlar, df_unwanted, gun_detaylari, min_bosluk, 
            kişi_sayısı,
            st.session_state.forbidden_pairs,
            st.session_state.get('person_limits', {}),
            df_preferred
        )
        st.session_state.should_regenerate_assignments = True
        st.rerun()

# Auto-save functionality
elapsed = time.time() - st.session_state.last_auto_save
if elapsed >= 30:
    auto_name = f"Otomatik_{yil}_{ay:02d}"
    _saved_ok = False
    try:
        _saved_ok = save_schedule(auto_name, yil, ay, isimler, st.session_state.schedule_bool,
                                  pref_df=st.session_state.get('pref_df'))
    except Exception:
        _saved_ok = False
    if _saved_ok:
        st.session_state.last_auto_save = time.time()
        st.session_state.auto_save_failed = False
    else:
        st.session_state.auto_save_failed = True

if st.session_state.auto_save_failed:
    st.error("🔴 Otomatik kayıt başarısız oldu — çizelgeniz veritabanına kaydedilemiyor! Sayfayı yenilemeden önce Excel/PNG olarak indirin.")

# Use the current schedule for all calculations
# In Fast Mode, we allow direct editing of the main grid
# REMOVED: Redundant Müsaitlik editor as requested
edited = st.session_state.schedule_bool.copy()

# --- HATA KONTROL ---
violations = []
conflict_msg = []
forbidden_msg = []
max_person_msg = []
min_person_msg = []
last_shift_check = {i: -10 for i in isimler}

for col in sutunlar:
    gun_no = gun_detaylari[col]['day_num']
    nobetciler = edited.index[edited[col]].tolist()
    
    if len(nobetciler) > kişi_sayısı:
        max_person_msg.append(f"🔴 **{gun_no}. Gün**: {len(nobetciler)} kişi atanmış! (Max {kişi_sayısı})")
    elif len(nobetciler) < kişi_sayısı:
        min_person_msg.append(f"🟠 **{gun_no}. Gün**: Eksik nöbetçi! ({len(nobetciler)}/{kişi_sayısı} kişi)")
        
    for k in nobetciler:
        if df_unwanted.at[k, col]:
            conflict_msg.append(f"❌ **{k}**: {gun_no}. gün müsait değilim demişti.")
        
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

# Kişisel limit ihlali kontrolü
limit_msg = []
_pl = st.session_state.get('person_limits', {})
for isim in isimler:
    _lim = _pl.get(isim, {})
    _min_l = _lim.get('min', 0)
    _max_l = _lim.get('max', 999)
    _tot = int(edited.loc[isim].sum())
    if _min_l > 0 and _tot < _min_l:
        limit_msg.append(f"🔻 **{isim}**: Min {_min_l} nöbet gerekli, şu an {_tot} atanmış.")
    if _max_l < 999 and _tot > _max_l:
        limit_msg.append(f"🔺 **{isim}**: Max {_max_l} nöbet aşıldı, şu an {_tot} atanmış.")

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
if _liste_swap_dn is not None and st.session_state.get('cached_rows_liste'):
    _ls_role_names = st.session_state.get('cached_role_names', [f"Görev{i+1}" for i in range(kişi_sayısı)])
    _ls_row_idx = _liste_swap_dn - 1
    if 0 <= _ls_row_idx < len(st.session_state.cached_rows_liste):
        _ls_row = dict(st.session_state.cached_rows_liste[_ls_row_idx])
        _ls_valid = [_ls_row.get(rn, "-") for rn in _ls_role_names]
        _ls_valid = [p for p in _ls_valid if p and p != "-" and p in isimler]
        if len(_ls_valid) >= 2:
            save_undo_state()
            _ls_valid = [_ls_valid[-1]] + _ls_valid[:-1]  # döngüsel kaydırma
            _lsvi = 0
            for _lsrn in _ls_role_names:
                _lsold = _ls_row.get(_lsrn, "-")
                if _lsold and _lsold != "-" and _lsold in isimler:
                    _ls_row[_lsrn] = _ls_valid[_lsvi]; _lsvi += 1
            st.session_state.cached_rows_liste[_ls_row_idx] = _ls_row
            st.session_state.should_regenerate_assignments = False
            st.toast(f"{_liste_swap_dn}. gün görev yerleri değiştirildi", icon="🔄")
            st.rerun()

# --- VERİ HAZIRLIĞI ---
# Only regenerate assignments when AI button is clicked, not on manual edits
_cache_key = (yil, ay, tuple(isimler), tuple(sutunlar))
if (st.session_state.should_regenerate_assignments
        or st.session_state.cached_rows_liste is None
        or st.session_state.get('cached_key') != _cache_key):
    rows_liste = []
    first_role_counts = {i: 0 for i in isimler}
    for col in sutunlar:
        nobetciler = edited.index[edited[col]].tolist()
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
    st.session_state.should_regenerate_assignments = False
else:
    # If we are not regenerating (manual edits happened), we need to SYNC rows_liste from the edited dataframe
    # to allow editing roles (swapping people between Görev1, Görev2, etc.)
    rows_liste = st.session_state.cached_rows_liste
    first_role_counts = {i: 0 for i in isimler}
    
    # We update rows_liste based on current edited state, but keep the role structure
    new_rows_liste = []
    for i, col in enumerate(sutunlar):
        current_row = rows_liste[i]
        nobetciler_in_df = edited.index[edited[col]].tolist()
        
        # Update names in the row based on what's in the boolean dataframe
        # If someone was removed, they should be '-' in all role columns
        # If someone was added, they should be in the first available '-' slot
        new_row = {"Tarih": current_row["Tarih"]}
        
        # Track who is already in the row to avoid duplicates
        people_already_assigned = []
        for role_name in role_names:
            p = current_row.get(role_name, "-")
            if p in nobetciler_in_df and p != "-":
                new_row[role_name] = p
                people_already_assigned.append(p)
                if role_name == role_names[0]:
                    first_role_counts[p] += 1
            else:
                new_row[role_name] = "-"
        
        # Add people who are in nobetciler_in_df but not yet in new_row
        for p in nobetciler_in_df:
            if p not in people_already_assigned:
                # Find first '-' slot
                for role_name in role_names:
                    if new_row[role_name] == "-":
                        new_row[role_name] = p
                        if role_name == role_names[0]:
                            first_role_counts[p] += 1
                        break
        
        new_rows_liste.append(new_row)
    
    rows_liste = new_rows_liste
    st.session_state.cached_rows_liste = rows_liste
    st.session_state.cached_first_role_counts = first_role_counts
    role_names = st.session_state.get('cached_role_names', role_names)

# Display and allow manual editing of ROLES (swapping people)
# REMOVED: Manual list editing as requested. Using autonomous redistribution instead.

# Re-distribution logic moved to Analiz section as requested.

df_liste = pd.DataFrame(rows_liste)

# Tüm roller için kişi başı sayım tablosu
all_role_counts = {rn: {isim: 0 for isim in isimler} for rn in role_names}
for _row in rows_liste:
    for rn in role_names:
        _p = _row.get(rn, '-')
        if _p in isimler:
            all_role_counts[rn][_p] += 1

stats_load = []
stats_finance = []
pair_matrix = pd.DataFrame(0, index=isimler, columns=isimler, dtype=int)

for isim in isimler:
    toplam = edited.loc[isim].sum()
    haftasonu = 0
    ozel_gun = 0
    for col in sutunlar:
        if edited.at[isim, col]:
            if gun_detaylari[col]['weekend']: haftasonu += 1
            if gun_detaylari[col]['weekend'] or gun_detaylari[col]['holiday']: ozel_gun += 1
    
    saat = toplam * 24
    fm_saat = max(0, saat - zorunlu_saat)
    ucret = fm_saat * nobet_ucreti
    
    _p_lim_s = st.session_state.get('person_limits', {}).get(isim, {})
    _min_s = _p_lim_s.get('min', 0)
    _max_s = _p_lim_s.get('max', 999)
    _limit_str = (
        f"{_min_s}-{_max_s}" if _min_s or _max_s < 999
        else "-"
    )
    _limit_ok = (int(toplam) >= _min_s) and (int(toplam) <= _max_s)
    stats_load.append({
        "İsim": isim,
        "Toplam": int(toplam),
        "Özel": int(ozel_gun),
        **{rn: all_role_counts[rn].get(isim, 0) for rn in role_names},
        "Limit": _limit_str,
        "✓": "🟢" if _limit_ok else "🔴",
    })
    stats_finance.append({
        "İsim": isim,
        "Nöbet": int(saat),
        "Mesai": int(zorunlu_saat),
        "FM": int(fm_saat),
        "Ücret (TL)": round(ucret, 2)
    })

for col in sutunlar:
    n = edited.index[edited[col]].tolist()
    if len(n) >= 2:
        for i in range(len(n)):
            for j in range(i+1, len(n)):
                pair_matrix.loc[n[i], n[j]] = pair_matrix.loc[n[i], n[j]] + 1
                pair_matrix.loc[n[j], n[i]] = pair_matrix.loc[n[j], n[i]] + 1

# Convert pair_matrix to strings for clean display (self-pairs as "-")
pair_display = pair_matrix.astype(str)
for i in isimler: pair_display.loc[i,i] = "-" 

df_stats_load = pd.DataFrame(stats_load).set_index("İsim")
df_stats_finance = pd.DataFrame(stats_finance).set_index("İsim")

# --- GÖRÜNÜM ---
st.divider()

# ── İndirme butonları ince yatay satır ──────────────────────────────────────
data_hash = hash(df_liste.to_csv() + df_stats_load.to_csv() + df_stats_finance.to_csv())
excel_key = f"excel_{yil}_{ay}"
hash_key = f"excel_hash_{yil}_{ay}"
if EXCEL_AVAILABLE:
    if excel_key not in st.session_state or st.session_state.get(hash_key) != data_hash:
        st.session_state[excel_key] = convert_df_to_excel(df_liste, df_stats_load, df_stats_finance)
        st.session_state[hash_key] = data_hash

print_html = create_print_html(df_liste, df_stats_load, yil, ay)

dl1, dl2, dl3, dl4, dl5, dl6 = st.columns(6)
with dl1:
    st.download_button("📥 CSV", df_liste.to_csv(index=False).encode('utf-8'), "liste.csv", "text/csv", use_container_width=True)
with dl2:
    st.download_button("🖼️ PNG", convert_df_to_png(df_liste), "liste.png", "image/png", use_container_width=True)
with dl3:
    if EXCEL_AVAILABLE:
        st.download_button("📊 Excel", st.session_state[excel_key], f"nobet_{yil}_{ay:02d}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    else:
        st.caption("Excel yok")
with dl4:
    with st.popover("💬 Metin", use_container_width=True):
        st.text_area("Kopyala:", value=df_liste.to_markdown(index=False), height=200)
with dl5:
    st.download_button("🖨️ Yazdır", print_html.encode('utf-8'), f"nobet_{yil}_{ay:02d}.html", "text/html", use_container_width=True)
with dl6:
    with st.popover("📂 Excel Yükle", use_container_width=True):
        st.caption("Daha önce indirilen Excel dosyasını veya aynı formatta hazırladığınız listeyi yükleyin.")
        _uploaded = st.file_uploader("Excel dosyası seçin (.xlsx)", type=["xlsx"], key="excel_import_uploader", label_visibility="collapsed")
        if _uploaded is not None:
            try:
                _xl = pd.ExcelFile(_uploaded)
                _sheet = "Günlük Liste" if "Günlük Liste" in _xl.sheet_names else _xl.sheet_names[0]
                _df_imp = pd.read_excel(_xl, sheet_name=_sheet)
                # Tarih sütunundan gün numarasını çıkar (örn: "01.06.2026 Paz" → 1)
                _imp_role_cols = [c for c in _df_imp.columns if c != "Tarih"]
                if "Tarih" not in _df_imp.columns or not _imp_role_cols:
                    st.error("Dosya formatı uyumsuz: 'Tarih' sütunu ve en az bir görev sütunu gerekli.")
                else:
                    _new_sched = pd.DataFrame(False, index=isimler, columns=sutunlar)
                    _new_rows = []
                    _matched = 0
                    for _, _irow in _df_imp.iterrows():
                        _tarih_str = str(_irow.get("Tarih", ""))
                        try:
                            # dd.mm.yyyy veya "1 Oca Pzt" gibi formatları dene
                            _ts = _tarih_str.strip()
                            if "." in _ts:
                                _day_num = int(_ts.split(".")[0])
                            elif "/" in _ts:
                                _day_num = int(_ts.split("/")[0])
                            elif "-" in _ts:
                                # yyyy-mm-dd
                                _day_num = int(_ts.split("-")[2].split()[0])
                            else:
                                # Sadece sayı (Excel integer tarihi) veya "15 Haz Pzt" formatı
                                _day_num = int(float(_ts.split()[0]))
                        except Exception:
                            continue
                        # Sutunlar içinde eşleşen kolonu bul
                        _match_col = None
                        for _sc in sutunlar:
                            if gun_detaylari[_sc]['day_num'] == _day_num:
                                _match_col = _sc; break
                        if _match_col is None:
                            continue
                        _row_data = {"Tarih": gun_detaylari[_match_col]['full_date']}
                        for _ri, _rn in enumerate(role_names):
                            _src_col = _imp_role_cols[_ri] if _ri < len(_imp_role_cols) else None
                            _person = str(_irow.get(_src_col, "-")).strip() if _src_col else "-"
                            if _person and _person != "-" and _person != "nan" and _person in isimler:
                                _new_sched.at[_person, _match_col] = True
                                _row_data[_rn] = _person
                            else:
                                _row_data[_rn] = "-"
                        _new_rows.append(_row_data)
                        _matched += 1
                    if _matched > 0:
                        st.session_state.schedule_bool = _new_sched
                        st.session_state.cached_rows_liste = _new_rows
                        st.session_state.cached_role_names = role_names
                        st.session_state.should_regenerate_assignments = False
                        _xl_save_ok = save_schedule(f"Otomatik_{yil}_{ay:02d}", yil, ay, isimler, _new_sched,
                                                    pref_df=st.session_state.get('pref_df'))
                        if not _xl_save_ok:
                            st.warning("⚠️ Excel verisi yüklendi ancak otomatik kayıt başarısız oldu.")
                        st.success(f"{_matched} gün yüklendi!")
                        st.rerun()
                    else:
                        st.warning("Eşleşen gün bulunamadı. Ekip isimleri ve ay/yıl ayarlarının uyumlu olduğunu kontrol edin.")
            except Exception as _e:
                st.error(
                    "Dosya okunamadı. Lütfen uygulamadan indirilen Excel formatını "
                    f"('Tarih' + görev sütunları) kullandığınızdan emin olun.\n\nDetay: {_e}"
                )

st.divider()

# ── Günlük Liste — görev yeri değişimi ───────────────────────────────────────
st.subheader("📋 Günlük Liste")
st.caption("Her satırdaki 🔄 butona tıklayarak o günün görev yerlerini döngüsel olarak değiştirebilirsiniz. Değişiklik Geri Al / İleri Al ile geri alınabilir.")
if rows_liste:
    _gl_role_names = role_names
    _gl_cols_cfg = {"Tarih": st.column_config.TextColumn("Tarih", disabled=True)}
    for _glrn in _gl_role_names:
        _gl_cols_cfg[_glrn] = st.column_config.TextColumn(_glrn, disabled=True)
    _gl_cols_cfg["🔄"] = st.column_config.CheckboxColumn("Değiştir", default=False)
    _gl_display_rows = []
    for _glrow in rows_liste:
        _gl_persons = [_glrow.get(rn, "-") for rn in _gl_role_names]
        _gl_persons = [p for p in _gl_persons if p and p != "-" and p in isimler]
        _gl_display_rows.append({
            "Tarih": _glrow.get("Tarih", ""),
            **{rn: _glrow.get(rn, "-") for rn in _gl_role_names},
            "🔄": False,
        })
    _gl_df = pd.DataFrame(_gl_display_rows)
    _gl_edited = st.data_editor(
        _gl_df,
        column_config=_gl_cols_cfg,
        use_container_width=True,
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
        _edited_roles = st.data_editor(
            _role_edit_df,
            column_config=_edit_col_cfg,
            use_container_width=True,
            hide_index=True,
            key="role_editor_table",
            num_rows="fixed",
        )
        if _edited_roles is not None:
            _role_changed = False
            _new_rows_edit = []
            for _ri, (_erow, _orow) in enumerate(zip(_edited_roles.itertuples(index=False), rows_liste)):
                _new_r = dict(_orow)
                _day_col = sutunlar[_ri] if _ri < len(sutunlar) else None
                _assigned_today = (
                    edited.index[edited[_day_col]].tolist()
                    if _day_col and _day_col in edited.columns else []
                )
                for _rn in role_names:
                    _val = str(getattr(_erow, _rn, "-")).strip()
                    if _val and _val not in ("nan", "-") and _val in _assigned_today:
                        if _new_r.get(_rn) != _val:
                            _new_r[_rn] = _val
                            _role_changed = True
                _new_rows_edit.append(_new_r)
            if _role_changed:
                save_undo_state()
                st.session_state.cached_rows_liste = _new_rows_edit
                st.session_state.should_regenerate_assignments = False
                st.rerun()
    else:
        st.info("Önce simülasyon çalıştırın veya nöbet ataması yapın.")

# ── Analiz — yatay dağılım ───────────────────────────────────────────────────
st.header("📊 Analiz")

# Satır 1: Nöbet Yükü  |  Kişisel Detay
an1, an2 = st.columns(2)

with an1:
    st.markdown("**Nöbet Yükü**")
    st.dataframe(
        df_stats_load.style.background_gradient(cmap="Blues", subset=["Toplam"] + list(role_names))
                           .background_gradient(cmap="Oranges", subset=["Özel"]),
        use_container_width=True
    )
    if st.button("🤖 Görev Yerlerini Otomatik Dağıt", use_container_width=True, help="Mevcut nöbetçileri değiştirmeden sadece görev yerlerini (Rollerini) otomatik olarak dengeler."):
        if 'cached_rows_liste' in st.session_state and st.session_state.cached_rows_liste:
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
            new_schedule = pd.DataFrame(False, index=isimler, columns=sutunlar)
            for i, col in enumerate(sutunlar):
                row = new_rows_liste[i]
                for rn in role_names:
                    p = row.get(rn)
                    if p in isimler:
                        new_schedule.at[p, col] = True
            st.session_state.schedule_bool = new_schedule
            st.session_state.should_regenerate_assignments = False
            st.toast("Görev yerleri yeniden dağıtıldı!", icon="🤖")

with an2:
    st.markdown("**Kişisel Detay**")
    kisi_sec = st.selectbox("Kişi:", isimler, label_visibility="collapsed")
    kisi_rows = []
    for index, row in df_liste.iterrows():
        roller = []
        partners_set = set()
        for role_idx, role_name in enumerate(role_names):
            if role_name in row and row.get(role_name) == kisi_sec:
                roller.append(role_name)
                for other_idx, other_role in enumerate(role_names):
                    if other_idx != role_idx and other_role in row and row.get(other_role) and row[other_role] != '-':
                        partners_set.add(row[other_role])
        if roller:
            partner_str = ', '.join(sorted(partners_set)) if partners_set else 'Tek'
            kisi_rows.append({"Tarih": row['Tarih'], "Partner": partner_str, "Rol": ', '.join(roller)})
    df_kisi = pd.DataFrame(kisi_rows)
    if not df_kisi.empty:
        h_kisi = min(len(df_kisi) * 35 + 38, 400)
        st.dataframe(df_kisi, height=h_kisi, use_container_width=True, hide_index=True)
    else:
        st.info("Nöbet yok.")

st.divider()

# Satır 2: Ücret Özeti  |  Eşleşme Matrisi
an3, an4 = st.columns(2)

with an3:
    st.markdown("**Ücret Özeti**")
    st.dataframe(
        df_stats_finance.style.background_gradient(cmap="Reds", subset=["FM", "Ücret (TL)"])
                              .format({"Ücret (TL)": "₺ {:,.2f}"}),
        use_container_width=True
    )

with an4:
    st.markdown("**Eşleşme Matrisi**")
    st.dataframe(pair_display, use_container_width=True)