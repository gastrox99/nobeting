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
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
from solver import solve_schedule, SolverResult

# Excel export
try:
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# Initialize database on app start
init_db()

# Sayfa Ayarları
st.set_page_config(page_title="Adil Nöbet v98 (AI Simulation)", layout="wide")
st.title("🤝 Nöbet Yönetimi")

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
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column('A:Z', 15)
    
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

def save_undo_state(schedule_df):
    """Save current state for undo"""
    if 'undo_history' not in st.session_state:
        st.session_state.undo_history = []
    if 'redo_history' not in st.session_state:
        st.session_state.redo_history = []
    
    # Limit history to 10 states
    if len(st.session_state.undo_history) >= 10:
        st.session_state.undo_history.pop(0)
    
    st.session_state.undo_history.append(schedule_df.copy())
    st.session_state.redo_history = []  # Clear redo on new action

# --- ANA ALGORİTMA (V98: BEST-OF-N SIMULATION) ---
def run_scheduling_algorithm_v98(isimler, sutunlar, df_unwanted_bool, gun_detaylari, min_bosluk, forbidden_pairs=None, person_limits=None, df_preferred=None):
    
    best_schedule = None
    best_score = float('inf') # Daha düşük puan daha iyi (Ceza puanı mantığı)
    
    # 100 Deneme Yap, En İyisini Seç
    SIMULATION_COUNT = 100
    
    progress_bar = st.progress(0)
    
    for attempt in range(SIMULATION_COUNT):
        # İlerleme çubuğunu güncelle
        if attempt % 10 == 0: progress_bar.progress(attempt + 1)
        
        # --- TEKİL DENEME BAŞLANGICI ---
        stat_total = {i: 0 for i in isimler}
        stat_special = {i: 0 for i in isimler} 
        stat_consecutive_weekend = {i: 0 for i in isimler}  # Weekend balance
        last_weekend_shift = {i: -10 for i in isimler}  # Track last weekend
        pair_history = {} 
        last_shift_day = {i: -10 for i in isimler}
        
        temp_schedule = pd.DataFrame({col: [False]*len(isimler) for col in sutunlar}, index=isimler)
        
        # Score Calculation (Local decision)
        def get_decision_score(p, is_sp, col, p1=None):
            total = stat_total[p] + (random.random() * 0.5) # Küçük rastgelelik tie-breaker
            sp_count = stat_special[p]
            penalty = pair_history.get(tuple(sorted((p1, p))) if p1 else None, 0)
            
            # Weekend balance penalty - avoid consecutive weekends
            consecutive_penalty = stat_consecutive_weekend[p] * 200
            
            # Preference bonus (negative = preferred, positive = avoid)
            pref_bonus = 0
            if df_preferred is not None and p in df_preferred.index and col in df_preferred.columns:
                pref_val = df_preferred.at[p, col]
                if pref_val == 1:  # Preferred
                    pref_bonus = -50
                elif pref_val == 2:  # Avoid
                    pref_bonus = 100
            
            # Min/max limits penalty
            limit_penalty = 0
            if person_limits and p in person_limits:
                max_limit = person_limits[p].get('max', 999)
                if stat_total[p] >= max_limit:
                    limit_penalty = 50000  # Very high to prevent assignment
            
            # Hafta sonuysa, önce hafta sonu dengesine bak
            if is_sp:
                return (sp_count * 100) + (total * 10) + penalty + consecutive_penalty + pref_bonus + limit_penalty
            else:
                return (total * 10) + (sp_count * 1) + penalty + pref_bonus + limit_penalty

        # Lineer İşleme (1..30) - Dağılım dengesi için şart
        empty_shifts = 0
        limit_violations = 0
        
        for col in sutunlar:
            info = gun_detaylari[col]
            gun_no = info['day_num']
            is_sp = info['weekend'] or info['holiday']
            is_weekend = info['weekend']
            
            # Calculate which weekend number this is (for consecutive tracking)
            weekend_num = (gun_no - 1) // 7
            
            # Adayları bul - also check max limits
            adaylar = []
            for k in isimler:
                if df_unwanted_bool.at[k, col]:
                    continue
                if (gun_no - last_shift_day[k]) <= min_bosluk:
                    continue
                # Check max limit
                if person_limits and k in person_limits:
                    max_limit = person_limits[k].get('max', 999)
                    if stat_total[k] >= max_limit:
                        continue
                adaylar.append(k)
            
            random.shuffle(adaylar) # Şans faktörü
            
            # Adayları o anki duruma göre sırala
            adaylar.sort(key=lambda x: get_decision_score(x, is_sp, col))
            
            if len(adaylar) >= kişi_sayısı:
                # Check for forbidden pairs and skip if found
                secilenler = []
                for p in adaylar:
                    valid = True
                    if forbidden_pairs:
                        for selected in secilenler:
                            pair = tuple(sorted((p, selected)))
                            if pair in forbidden_pairs:
                                valid = False
                                break
                    if valid:
                        secilenler.append(p)
                        if len(secilenler) >= kişi_sayısı:
                            break
                
                if len(secilenler) >= kişi_sayısı:
                    # Pair history tracking for main 2
                    if kişi_sayısı >= 2:
                        pair = tuple(sorted((secilenler[0], secilenler[1])))
                        pair_history[pair] = pair_history.get(pair, 0) + 1
                    
                    for k in secilenler:
                        temp_schedule.at[k, col] = True
                        stat_total[k] += 1
                        if is_sp: stat_special[k] += 1
                        last_shift_day[k] = gun_no
                        
                        # Track consecutive weekends
                        if is_weekend:
                            if last_weekend_shift[k] >= 0 and weekend_num == last_weekend_shift[k] + 1:
                                stat_consecutive_weekend[k] += 1
                            elif last_weekend_shift[k] >= 0 and weekend_num > last_weekend_shift[k] + 1:
                                # Gap in weekends - reset consecutive counter
                                stat_consecutive_weekend[k] = 0
                            last_weekend_shift[k] = weekend_num
                else:
                    empty_shifts += 1
            else:
                empty_shifts += 1 # Ceza: Yetersiz aday

        # Check min limits violations
        if person_limits:
            for p, limits in person_limits.items():
                min_limit = limits.get('min', 0)
                if stat_total.get(p, 0) < min_limit:
                    limit_violations += 1

        # --- DENEME SONUCU PUANLAMA (GLOBAL SCORE) ---
        # Amaç: Standart sapmayı (farkları) minimize etmek
        totals = list(stat_total.values())
        specials = list(stat_special.values())
        consecutive_weekends = sum(stat_consecutive_weekend.values())
        
        std_dev_total = np.std(totals)
        std_dev_special = np.std(specials)
        range_total = max(totals) - min(totals)
        
        # Puan Fonksiyonu: Ne kadar düşükse o kadar iyi
        current_sim_score = (
            (empty_shifts * 10000) + 
            (limit_violations * 5000) +
            (consecutive_weekends * 500) +  # Weekend balance
            (range_total * 100) + 
            (std_dev_total * 10) + 
            (std_dev_special * 5)
        )
        
        if current_sim_score < best_score:
            best_score = current_sim_score
            best_schedule = temp_schedule.copy()
    
    progress_bar.empty()
    st.session_state.schedule_bool = best_schedule
    st.toast(f"100 Simülasyon yapıldı. En adil sonuç seçildi!", icon="🧠")

# --- AYARLAR ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    isimler = [x.strip() for x in st.text_area("Ekip:", "").split(",") if x.strip()]
    st.divider()
    yil = st.number_input("Yıl", 2024, 2030, 2025)
    ay = st.selectbox("Ay", range(1, 13), index=0)
    gun_sayisi = calendar.monthrange(yil, ay)[1]
    tatil_gunleri = [int(x) for x in st.text_input("Tatiller:", "").split(",") if x.strip().isdigit()]
    min_bosluk = st.slider("Dinlenme", 0, 3, 1)
    kişi_sayısı = st.slider("Nöbet Başına Kişi:", 1, 5, 2)
    
    # --- CUSTOM ROLE NAMES ---
    default_roles = ", ".join([f"Rol{i+1}" for i in range(kişi_sayısı)])
    role_names_input = st.text_input(
        "Görev İsimleri:",
        value=st.session_state.get("role_names_text", ""),
        placeholder=f"Örn: AYB, GYB veya Nöbetçi1, Nöbetçi2"
    )
    st.session_state.role_names_text = role_names_input
    
    # Parse role names
    if role_names_input.strip():
        role_names = [r.strip() for r in role_names_input.split(",") if r.strip()]
    else:
        role_names = []
    
    # Ensure we have enough role names (pad with defaults if needed)
    while len(role_names) < kişi_sayısı:
        role_names.append(f"Kişi{len(role_names)+1}")
    role_names = role_names[:kişi_sayısı]  # Trim excess
    st.session_state.rol_isimleri = role_names  # Store for solver
    
    # --- FORBIDDEN PAIRS ---
    st.markdown("🚫 **Birlikte Çalışamayan Kişiler**")
    forbidden_input = st.text_area(
        "Kimin kimin ile çalışamayacağını yazın:",
        value=st.session_state.get("forbidden_pairs_text", ""),
        height=80,
        placeholder="Örn: Ali-Ayşe, Mehmet-Fatma\n(virgülü ve tire kullanın)",
        key="forbidden_pairs_input"
    )
    st.session_state.forbidden_pairs_text = forbidden_input
    
    # Parse forbidden pairs (supports both comma and newline separators)
    forbidden_pairs = set()
    if forbidden_input.strip():
        # First split by newlines, then by commas
        all_pairs = []
        for line in forbidden_input.strip().split('\n'):
            all_pairs.extend(line.split(','))
        
        for pair_str in all_pairs:
            pair_str = pair_str.strip()
            if '-' in pair_str:
                parts = pair_str.split('-', 1)  # Only split on first hyphen
                if len(parts) == 2:
                    p1, p2 = parts[0].strip(), parts[1].strip()
                    if p1 and p2:
                        forbidden_pairs.add(tuple(sorted((p1, p2))))
    st.session_state.forbidden_pairs = forbidden_pairs
    
    # --- MIN/MAX LIMITS ---
    st.divider()
    st.markdown("📊 **Kişisel Limitler**")
    with st.expander("Min/Max Nöbet Sayısı Ayarla"):
        if 'person_limits' not in st.session_state:
            st.session_state.person_limits = {}
        
        limits_text = st.text_area(
            "Her satıra: İsim:min-max",
            value=st.session_state.get("limits_text", ""),
            height=80,
            placeholder="Örn:\nAli:5-10\nAyşe:3-8",
            key="limits_input"
        )
        st.session_state.limits_text = limits_text
        
        # Parse limits
        person_limits = {}
        if limits_text.strip():
            for line in limits_text.strip().split('\n'):
                if ':' in line:
                    parts = line.split(':')
                    name = parts[0].strip()
                    if len(parts) == 2 and '-' in parts[1]:
                        try:
                            min_val, max_val = map(int, parts[1].split('-'))
                            person_limits[name] = {'min': min_val, 'max': max_val}
                        except ValueError:
                            pass
        st.session_state.person_limits = person_limits
        
        if person_limits:
            st.caption(f"Aktif limitler: {len(person_limits)} kişi")
    
    nobet_ucreti = st.number_input("Saatlik FM Ücreti (TL)", value=252.59)
    
    # --- SAVE/LOAD ---
    st.divider()
    st.header("💾 Kaydet/Yükle")
    
    tab1, tab2, tab3 = st.tabs(["Kaydet", "Yükle", "Listele"])
    
    with tab1:
        save_name = st.text_input("Takvim Adı:", f"Nöbet_{yil}_{ay:02d}")
        if st.button("💾 Kaydet", type="primary", key="save_btn"):
            if save_schedule(save_name, yil, ay, isimler, st.session_state.schedule_bool):
                st.success(f"✅ '{save_name}' kaydedildi!")
            else:
                st.error("❌ Kaydetme başarısız")
    
    with tab2:
        schedules = list_schedules()
        if schedules:
            schedule_options = [f"{s['name']} ({s['year']}-{s['month']:02d})" for s in schedules]
            selected = st.selectbox("Kaydedilmiş Takvim:", schedule_options)
            
            if st.button("📂 Yükle", key="load_btn"):
                # Parse selection
                selected_name = selected.split(" (")[0]
                for s in schedules:
                    if s['name'] == selected_name:
                        team, df = load_schedule(s['name'], s['year'], s['month'])
                        if df is not None:
                            st.session_state.schedule_bool = df
                            st.session_state.should_regenerate_assignments = True
                            st.success(f"✅ '{s['name']}' yüklendi!")
                            st.rerun()
                        break
        else:
            st.info("Henüz kayıtlı takvim yok")
    
    with tab3:
        schedules = list_schedules()
        if schedules:
            st.write("📋 Kayıtlı Takvimler:")
            for s in schedules:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{s['name']}** - {s['year']}-{s['month']:02d}")
                    updated = s['updated_at'].strftime('%Y-%m-%d') if hasattr(s['updated_at'], 'strftime') else str(s['updated_at'])[:10]
                    st.caption(f"Güncellendi: {updated}")
                with col2:
                    if st.button("🗑️", key=f"del_{s['name']}_{s['year']}_{s['month']}"):
                        delete_schedule(s['name'], s['year'], s['month'])
                        st.rerun()
        else:
            st.info("Henüz kayıtlı takvim yok")

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
    gun_detaylari[baslik] = {"weekend": is_we, "holiday": is_hol, "day_num": g, "full_date": full_d}
    disp = str(g)
    if is_hol: disp = f"🚨 {g}"
    elif is_we: disp = f"🏖️ {g}"
    sutunlar_display.append(disp)

calisma_gunu = gun_sayisi - ozel_gun_sayisi
zorunlu_saat = calisma_gunu * 8

# --- SESSION ---
if 'schedule_bool' not in st.session_state:
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
        # Preserve existing data where possible
        new_schedule = pd.DataFrame(False, index=isimler, columns=sutunlar)
        for person in isimler:
            if person in current_schedule.index:
                for col in sutunlar:
                    if col in current_schedule.columns:
                        new_schedule.at[person, col] = current_schedule.at[person, col]
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
if 'preferences' not in st.session_state: st.session_state.preferences = {}
if 'person_preferences' not in st.session_state: st.session_state.person_preferences = {}
for i in isimler:
    if i not in st.session_state.person_preferences:
        st.session_state.person_preferences[i] = {}
for i in isimler: 
    if i not in st.session_state.inputs: st.session_state.inputs[i] = ""

# --- ADIM 1: KARTLI GİRİŞ ---
st.header("⬇️ 1. ADIM: Müsaitlik ve Tercihler")

# Initialize preference DataFrame if needed
if 'pref_df' not in st.session_state:
    st.session_state.pref_df = pd.DataFrame(0, index=isimler, columns=sutunlar)
else:
    # Sync with current team/month
    current_pref = st.session_state.pref_df
    if list(current_pref.columns) != sutunlar or list(current_pref.index) != isimler:
        new_pref = pd.DataFrame(0, index=isimler, columns=sutunlar)
        for person in isimler:
            if person in current_pref.index:
                for col in sutunlar:
                    if col in current_pref.columns:
                        new_pref.at[person, col] = current_pref.at[person, col]
        st.session_state.pref_df = new_pref

# Initialize paint color
if 'paint_color' not in st.session_state:
    st.session_state.paint_color = 0

# Color palette - click to select paint color
st.markdown("### 🎨 Renk Seç (tıkla), sonra tabloda hücrelere tıklayarak boya")
color_cols = st.columns(4)
color_info = [
    (0, "⬜ Nötr", "#ffffff"),
    (1, "🟩 Tercih", "#c8e6c9"),
    (2, "🟨 Kaçın", "#fff9c4"),
    (3, "🟥 Müsait Değil", "#ffcdd2")
]

for i, (val, label, bg) in enumerate(color_info):
    with color_cols[i]:
        btn_type = "primary" if st.session_state.paint_color == val else "secondary"
        if st.button(label, key=f"color_{val}", use_container_width=True, type=btn_type):
            st.session_state.paint_color = val
            st.rerun()

selected_label = color_info[st.session_state.paint_color][1]
st.info(f"✏️ Seçili: **{selected_label}** - Tablodaki hücrelere çift tıklayıp 0-3 arası değer girin veya direkt düzenleyin")

# Quick fill buttons
qf_cols = st.columns(4)
with qf_cols[0]:
    if st.button("🔄 Tümünü Sıfırla", use_container_width=True):
        st.session_state.pref_df = pd.DataFrame(0, index=isimler, columns=sutunlar)
        st.rerun()
with qf_cols[1]:
    if st.button("🏖️ HS → Kaçın", use_container_width=True):
        for col in sutunlar:
            if gun_detaylari[col]['weekend']:
                st.session_state.pref_df[col] = 2
        st.rerun()
with qf_cols[2]:
    if st.button("🏖️ HS → Müsait Değil", use_container_width=True):
        for col in sutunlar:
            if gun_detaylari[col]['weekend']:
                st.session_state.pref_df[col] = 3
        st.rerun()
with qf_cols[3]:
    if st.button(f"📅 Tümü → {selected_label}", use_container_width=True):
        st.session_state.pref_df[:] = st.session_state.paint_color
        st.rerun()

# Prepare DataFrame for AgGrid
grid_df = st.session_state.pref_df.copy()
grid_df.insert(0, 'İsim', grid_df.index)
grid_df = grid_df.reset_index(drop=True)

# Rename columns to show just day numbers
col_rename = {'İsim': 'İsim'}
for col in sutunlar:
    day_num = gun_detaylari[col]['day_num']
    is_we = gun_detaylari[col]['weekend']
    is_hol = gun_detaylari[col]['holiday']
    if is_hol:
        col_rename[col] = f"🚨{day_num}"
    elif is_we:
        col_rename[col] = f"🏖{day_num}"
    else:
        col_rename[col] = str(day_num)

grid_df = grid_df.rename(columns=col_rename)

# Build AgGrid options with cell selection
gb = GridOptionsBuilder.from_dataframe(grid_df)
gb.configure_default_column(editable=False, minWidth=40, maxWidth=55)
gb.configure_column('İsim', pinned='left', minWidth=100, maxWidth=150)

# Cell style based on value (0-3)
cell_style_jscode = JsCode("""
function(params) {
    if (params.colDef.field === 'İsim') return {'fontWeight': 'bold'};
    var val = params.value;
    var style = {'cursor': 'pointer', 'textAlign': 'center'};
    if (val === 1) { style['backgroundColor'] = '#c8e6c9'; }
    else if (val === 2) { style['backgroundColor'] = '#fff9c4'; }
    else if (val === 3) { style['backgroundColor'] = '#ffcdd2'; }
    else { style['backgroundColor'] = '#ffffff'; }
    return style;
}
""")

# Cell renderer to show emoji
cell_renderer = JsCode("""
function(params) {
    if (params.colDef.field === 'İsim') return params.value;
    var val = params.value;
    if (val === 1) return '✓';
    if (val === 2) return '~';
    if (val === 3) return '✗';
    return '·';
}
""")

# Apply cell style to all data columns
for col in grid_df.columns:
    if col != 'İsim':
        gb.configure_column(col, cellStyle=cell_style_jscode, cellRenderer=cell_renderer)

# Enable cell selection
gb.configure_selection(selection_mode='single', use_checkbox=False)
gb.configure_grid_options(
    suppressRowClickSelection=True,
    enableRangeSelection=True,
    suppressCellSelection=False
)

grid_options = gb.build()
grid_options['rowHeight'] = 35

# Display AgGrid with selection tracking
st.markdown(f"**📅 Tercih Tablosu** - Hücreye tıklayın → **{selected_label}** uygulanır")
grid_response = AgGrid(
    grid_df,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    fit_columns_on_grid_load=True,
    height=min(400, len(isimler) * 40 + 60),
    allow_unsafe_jscode=True,
    theme='streamlit',
    enable_enterprise_modules=False
)

# Person selector for painting
st.markdown("---")
paint_cols = st.columns([2, 3])
with paint_cols[0]:
    selected_person = st.selectbox("👤 Kişi Seç:", isimler, key="paint_person")
with paint_cols[1]:
    st.markdown(f"**Seçili renk:** {selected_label}")

# Day buttons for selected person - single click to paint
if selected_person:
    st.markdown(f"**{selected_person}** için günlere tıklayın:")
    
    # Display in rows of 10 days
    days_per_row = 10
    for row_start in range(0, len(sutunlar), days_per_row):
        row_cols = sutunlar[row_start:row_start + days_per_row]
        cols = st.columns(len(row_cols))
        for i, col in enumerate(row_cols):
            with cols[i]:
                val = st.session_state.pref_df.at[selected_person, col]
                day_num = gun_detaylari[col]['day_num']
                is_we = gun_detaylari[col]['weekend']
                is_hol = gun_detaylari[col]['holiday']
                
                # Choose emoji based on value
                if val == 1:
                    emoji = "🟩"
                elif val == 2:
                    emoji = "🟨"
                elif val == 3:
                    emoji = "🟥"
                else:
                    emoji = "⬜"
                
                # Add weekend/holiday indicator
                day_label = f"{emoji}{day_num}"
                if is_hol:
                    day_label = f"🚨{day_num}"
                elif is_we:
                    day_label = f"🏖{day_num}"
                else:
                    day_label = f"{day_num}"
                
                btn_label = f"{emoji}"
                if st.button(btn_label, key=f"day_{selected_person}_{col}", use_container_width=True, help=f"Gün {day_num}"):
                    st.session_state.pref_df.at[selected_person, col] = st.session_state.paint_color
                    st.rerun()

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

st.markdown("### 🎯 Nöbet Dağıtımı")
btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    sim_clicked = st.button("⚡ AI Simülasyon (100 deneme)", type="secondary", use_container_width=True)

with btn_col2:
    opt_clicked = st.button("🧠 AI Optimizasyon (OR-Tools)", type="primary", use_container_width=True)

if sim_clicked:
    is_valid, errors, warnings = validate_inputs(isimler, yil, ay, gun_sayisi, tatil_gunleri, nobet_ucreti, min_bosluk, kişi_sayısı)
    
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
            st.session_state.forbidden_pairs,
            st.session_state.get('person_limits', {}),
            df_preferred
        )
        st.session_state.should_regenerate_assignments = True
        st.rerun()

if opt_clicked:
    is_valid, errors, warnings = validate_inputs(isimler, yil, ay, gun_sayisi, tatil_gunleri, nobet_ucreti, min_bosluk, kişi_sayısı)
    
    if errors:
        st.error("🚨 Hata(lar) düzeltilmeli:")
        for err in errors:
            st.error(err)
    else:
        if warnings:
            st.warning("⚠️ Uyarı(lar):")
            for warn in warnings:
                st.warning(warn)
        
        limits_dict = st.session_state.get('person_limits', {})
        limits_tuples = {}
        for name, lim in limits_dict.items():
            if isinstance(lim, dict):
                limits_tuples[name] = (lim.get('min', 0), lim.get('max', 999))
            elif isinstance(lim, tuple):
                limits_tuples[name] = lim
        
        pref_for_solver = df_preferred.copy()
        for person in pref_for_solver.index:
            for col in pref_for_solver.columns:
                val = pref_for_solver.at[person, col]
                if val == 1:
                    pref_for_solver.at[person, col] = 10
                elif val == 2:
                    pref_for_solver.at[person, col] = -10
                else:
                    pref_for_solver.at[person, col] = 0
        
        with st.spinner("🧠 OR-Tools optimizasyonu çalışıyor... (max 30s)"):
            result = solve_schedule(
                isimler=isimler,
                sutunlar=sutunlar,
                gun_detaylari=gun_detaylari,
                kisi_sayisi=kişi_sayısı,
                min_bosluk=min_bosluk,
                df_unwanted=df_unwanted,
                df_preferred=pref_for_solver,
                forbidden_pairs=st.session_state.forbidden_pairs,
                person_limits=limits_tuples,
                rol_isimleri=st.session_state.get('rol_isimleri', ['AYB', 'GYB', 'Rol3', 'Rol4', 'Rol5'][:kişi_sayısı]),
                timeout_seconds=30
            )
        
        st.markdown(result.message)
        
        if result.success:
            save_undo_state(st.session_state.schedule_bool)
            
            new_schedule = pd.DataFrame(False, index=isimler, columns=sutunlar)
            for day in sutunlar:
                if day in result.schedule.columns:
                    assigned = result.schedule[day].tolist()
                    for person in assigned:
                        if person in isimler:
                            new_schedule.at[person, day] = True
            
            st.session_state.schedule_bool = new_schedule
            st.session_state.should_regenerate_assignments = True
            
            if 'stats' in result.stats:
                with st.expander("📊 Optimizasyon Detayları"):
                    st.write(f"**Çözüm tipi:** {result.stats.get('status', 'N/A')}")
                    st.write(f"**Süre:** {result.solve_time:.2f}s")
                    st.write(f"**Dağılım farkı:** {result.stats.get('spread', 'N/A')}")
                    st.write(f"**Standart sapma:** {result.stats.get('std_dev', 0):.2f}")
                    if 'shift_counts' in result.stats:
                        st.write("**Kişi başı nöbet:**")
                        for person, count in result.stats['shift_counts'].items():
                            st.write(f"  - {person}: {count}")
            
            st.rerun()
        else:
            st.warning("💡 Simülasyon yöntemini deneyin veya kısıtları gevşetin.")

# --- ADIM 2: EDİTÖR ---
st.divider()
st.subheader("📝 2. ADIM: Kontrol & Düzenleme")

# Undo/Redo buttons
undo_col, redo_col, auto_col = st.columns([1, 1, 2])
with undo_col:
    undo_disabled = len(st.session_state.undo_history) == 0
    if st.button("↩️ Geri Al", disabled=undo_disabled, key="undo_btn"):
        if st.session_state.undo_history:
            # Save current state to redo
            st.session_state.redo_history.append(st.session_state.schedule_bool.copy())
            # Restore previous state
            st.session_state.schedule_bool = st.session_state.undo_history.pop()
            st.session_state.cached_rows_liste = None
            st.rerun()

with redo_col:
    redo_disabled = len(st.session_state.redo_history) == 0
    if st.button("↪️ Yinele", disabled=redo_disabled, key="redo_btn"):
        if st.session_state.redo_history:
            # Save current state to undo
            st.session_state.undo_history.append(st.session_state.schedule_bool.copy())
            # Restore redo state
            st.session_state.schedule_bool = st.session_state.redo_history.pop()
            st.session_state.cached_rows_liste = None
            st.rerun()

with auto_col:
    # Auto-save indicator with auto-refresh
    elapsed = time.time() - st.session_state.last_auto_save
    if elapsed >= 30:
        # Auto-save to database
        auto_name = f"Otomatik_{yil}_{ay:02d}"
        try:
            if save_schedule(auto_name, yil, ay, isimler, st.session_state.schedule_bool):
                st.session_state.last_auto_save = time.time()
                st.caption("💾 Otomatik kaydedildi")
            else:
                st.caption("⚠️ Kayıt başarısız")
        except Exception:
            st.caption("⚠️ Kayıt hatası")
    else:
        remaining = int(30 - elapsed)
        st.caption(f"⏱️ Sonraki kayıt: {remaining}s")
    
    # Auto-refresh trigger using JavaScript (refreshes every 30 seconds)
    st.markdown("""
    <script>
    setTimeout(function() {
        window.parent.postMessage({type: 'streamlit:rerun'}, '*');
    }, 30000);
    </script>
    """, unsafe_allow_html=True)

# Prepare data with fresh index/columns to avoid Arrow serialization
df_for_editor = st.session_state.schedule_bool.copy()
df_for_editor.index.name = "İsim"

# Use st.form to batch edits and prevent continuous reruns
with st.form(key="schedule_form", clear_on_submit=False):
    form_edited = st.data_editor(
        df_for_editor,
        key="schedule_data_editor",
        column_config={col: st.column_config.CheckboxColumn(display, width="small") 
                      for col, display in zip(sutunlar, sutunlar_display)}
    )
    
    submitted = st.form_submit_button("💾 Düzenlemeleri Kayıt Et")
    
    if submitted:
        # Save undo state before editing
        save_undo_state(st.session_state.schedule_bool)
        # Update schedule
        edited_clean = form_edited.astype(bool)
        st.session_state.schedule_bool = edited_clean
        st.success("✅ Düzenlemeler kaydedildi!")

# Use the current schedule for all calculations (either from form or session state)
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

if max_person_msg or min_person_msg or conflict_msg or forbidden_msg or violations:
    with st.expander("🚨 HATA RAPORU (Tıklayıp Açın)", expanded=True):
        for m in max_person_msg: st.error(m)
        for m in min_person_msg: st.warning(m)
        for c in conflict_msg: st.error(c)
        for f in forbidden_msg: st.error(f)
        for v in violations: st.info(v)
else:
    st.success(f"✅ Kurallar uygun (Her gün {kişi_sayısı} kişi, çakışma yok).")

# --- VERİ HAZIRLIĞI ---
# Only regenerate assignments when AI button is clicked, not on manual edits
if st.session_state.should_regenerate_assignments or st.session_state.cached_rows_liste is None:
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
    st.session_state.should_regenerate_assignments = False
else:
    rows_liste = st.session_state.cached_rows_liste
    first_role_counts = st.session_state.get('cached_first_role_counts', {i: 0 for i in isimler})
    role_names = st.session_state.get('cached_role_names', role_names)
df_liste = pd.DataFrame(rows_liste)

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
    
    stats_load.append({
        "İsim": isim,
        "Toplam": int(toplam),
        "Özel": int(ozel_gun),
        "HS": int(haftasonu),
        role_names[0]: first_role_counts.get(isim, 0)
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
col_left, col_right = st.columns([1.3, 1])

# === SOL SÜTUN: LİSTE ===
with col_left:
    st.subheader("📅 Günlük Liste")
    
    # Export buttons row 1
    c1, c2, c3 = st.columns(3)
    with c1: st.download_button("📥 CSV", df_liste.to_csv(index=False).encode('utf-8'), "liste.csv", "text/csv", type="primary")
    with c2: st.download_button("🖼️ PNG", convert_df_to_png(df_liste), "liste.png", "image/png")
    with c3: 
        if EXCEL_AVAILABLE:
            excel_data = convert_df_to_excel(df_liste, df_stats_load, df_stats_finance)
            st.download_button("📊 Excel", excel_data, f"nobet_{yil}_{ay:02d}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.caption("Excel yok")
    
    # Export buttons row 2
    c4, c5, c6 = st.columns(3)
    with c4:
        with st.popover("💬 Metin"):
            st.text_area("Kopyala:", value=df_liste.to_markdown(index=False), height=200)
    with c5:
        # Print-friendly HTML
        print_html = create_print_html(df_liste, df_stats_load, yil, ay)
        st.download_button("🖨️ Yazdır", print_html.encode('utf-8'), f"nobet_{yil}_{ay:02d}.html", "text/html")

    def highlight_list(row):
        c = "white"
        if "Cmt" in row['Tarih'] or "Paz" in row['Tarih']: c = "#eff6ff"
        return [f'background-color: {c}' for _ in row]
    
    h_liste = len(df_liste) * 35 + 38
    st.dataframe(df_liste.style.apply(highlight_list, axis=1), height=h_liste, use_container_width=True)

# === SAĞ SÜTUN: ANALİZ ===
with col_right:
    st.header("📊 Analiz")
    
    # 1. Yük
    st.markdown("**1. Nöbet Yükü (Dengeli)**")
    st.dataframe(
        df_stats_load.style.background_gradient(cmap="Blues", subset=["Toplam", role_names[0]])
                           .background_gradient(cmap="Oranges", subset=["Özel", "HS"]),
        use_container_width=True
    )
    st.divider()

    # 2. Kişisel Detay
    st.markdown("**2. Kişisel Detay 🔍**")
    kisi_sec = st.selectbox("Kişi:", isimler, label_visibility="collapsed")
    kisi_rows = []
    for index, row in df_liste.iterrows():
        # Check all role columns for this person
        partners = []
        rol = None
        
        for role_idx, role_name in enumerate(role_names):
            if role_name in row and row.get(role_name) == kisi_sec:
                rol = role_name
                # Collect all other assigned people as partners
                for other_idx, other_role in enumerate(role_names):
                    if other_idx != role_idx and other_role in row and row.get(other_role) and row[other_role] != '-':
                        partners.append(row[other_role])
                break
        
        if rol:
            partner_str = ', '.join(partners) if partners else 'Tek'
            kisi_rows.append({"Tarih": row['Tarih'], "Partner": partner_str, "Rol": rol})
    
    df_kisi = pd.DataFrame(kisi_rows)
    if not df_kisi.empty:
        h_kisi = len(df_kisi) * 35 + 38
        st.dataframe(df_kisi, height=h_kisi, use_container_width=True, hide_index=True)
    else:
        st.info("Nöbet yok.")

    st.divider()

    # 3. Matris
    st.markdown("**3. Eşleşme Matrisi**")
    st.dataframe(pair_display, use_container_width=True)

    st.divider()

    # 4. Ücret
    st.markdown("**4. Ücret Özeti**")
    st.dataframe(
        df_stats_finance.style.background_gradient(cmap="Reds", subset=["FM", "Ücret (TL)"])
                              .format({"Ücret (TL)": "₺ {:,.2f}"}),
        use_container_width=True
    )