# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import random
import calendar
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import date
import numpy as np
from db import init_db, save_schedule, load_schedule, list_schedules, delete_schedule

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

# --- ANA ALGORİTMA (V98: BEST-OF-N SIMULATION) ---
def run_scheduling_algorithm_v98(isimler, sutunlar, df_unwanted_bool, gun_detaylari, min_bosluk):
    
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
        pair_history = {} 
        last_shift_day = {i: -10 for i in isimler}
        
        temp_schedule = pd.DataFrame({col: [False]*len(isimler) for col in sutunlar}, index=isimler)
        
        # Score Calculation (Local decision)
        def get_decision_score(p, is_sp, p1=None):
            total = stat_total[p] + (random.random() * 0.5) # Küçük rastgelelik tie-breaker
            sp_count = stat_special[p]
            penalty = pair_history.get(tuple(sorted((p1, p))) if p1 else None, 0)
            
            # Hafta sonuysa, önce hafta sonu dengesine bak
            if is_sp:
                return (sp_count * 100) + (total * 10) + penalty
            else:
                return (total * 10) + (sp_count * 1) + penalty

        # Lineer İşleme (1..30) - Dağılım dengesi için şart
        empty_shifts = 0
        for col in sutunlar:
            info = gun_detaylari[col]
            gun_no = info['day_num']
            is_sp = info['weekend'] or info['holiday']
            
            # Adayları bul
            adaylar = [k for k in isimler if not df_unwanted_bool.at[k, col] and (gun_no - last_shift_day[k]) > min_bosluk]
            random.shuffle(adaylar) # Şans faktörü
            
            # Adayları o anki duruma göre sırala
            adaylar.sort(key=lambda x: get_decision_score(x, is_sp))
            
            if len(adaylar) >= 2:
                p1 = adaylar[0]
                
                # P2 seçimi (P1 ile uyumlu)
                others = [k for k in adaylar if k != p1]
                others.sort(key=lambda x: get_decision_score(x, is_sp, p1))
                p2 = others[0]
                
                secilenler = [p1, p2]
                pair = tuple(sorted((p1, p2)))
                pair_history[pair] = pair_history.get(pair, 0) + 1
                
                for k in secilenler:
                    temp_schedule.at[k, col] = True
                    stat_total[k] += 1
                    if is_sp: stat_special[k] += 1
                    last_shift_day[k] = gun_no
            else:
                empty_shifts += 1 # Ceza: Yetersiz aday

        # --- DENEME SONUCU PUANLAMA (GLOBAL SCORE) ---
        # Amaç: Standart sapmayı (farkları) minimize etmek
        totals = list(stat_total.values())
        specials = list(stat_special.values())
        
        std_dev_total = np.std(totals)
        std_dev_special = np.std(specials)
        range_total = max(totals) - min(totals)
        
        # Puan Fonksiyonu: Ne kadar düşükse o kadar iyi
        # Öncelik 1: Boş gün olmasın (empty_shifts * 1000)
        # Öncelik 2: Toplam nöbet farkı az olsun (range_total * 50)
        # Öncelik 3: Standart sapma düşük olsun
        current_sim_score = (empty_shifts * 10000) + (range_total * 100) + (std_dev_total * 10) + (std_dev_special * 5)
        
        if current_sim_score < best_score:
            best_score = current_sim_score
            best_schedule = temp_schedule.copy()
    
    progress_bar.empty()
    st.session_state.schedule_bool = best_schedule
    st.toast(f"100 Simülasyon yapıldı. En adil sonuç seçildi!", icon="🧠")

# --- AYARLAR ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    isimler = [x.strip() for x in st.text_area("Ekip:", "Görkem, Eylül, Faruk, Ege, Ayça, Mizgin, Taha").split(",") if x.strip()]
    st.divider()
    yil = st.number_input("Yıl", 2024, 2030, 2025)
    ay = st.selectbox("Ay", range(1, 13), index=0)
    gun_sayisi = calendar.monthrange(yil, ay)[1]
    tatil_gunleri = [int(x) for x in st.text_input("Tatiller:", "").split(",") if x.strip().isdigit()]
    min_bosluk = st.slider("Dinlenme", 0, 3, 1)
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
                    st.caption(f"Güncellendi: {s['updated_at'][:10]}")
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
if 'inputs' not in st.session_state: st.session_state.inputs = {i: "" for i in isimler}
if 'cached_rows_liste' not in st.session_state: st.session_state.cached_rows_liste = None
if 'cached_ayb_counts' not in st.session_state: st.session_state.cached_ayb_counts = None
if 'last_edited_hash' not in st.session_state: st.session_state.last_edited_hash = None
if 'should_regenerate_assignments' not in st.session_state: st.session_state.should_regenerate_assignments = False
for i in isimler: 
    if i not in st.session_state.inputs: st.session_state.inputs[i] = ""

# --- ADIM 1: KARTLI GİRİŞ ---
st.header("⬇️ 1. ADIM: Müsaitlik")
st.info("Müsait olunmayan günleri yazın (Örn: `3-5, 12`).")
cols = st.columns(3)
input_data = {}
for i, isim in enumerate(isimler):
    with cols[i % 3]:
        val = st.text_input(f"👤 {isim}", st.session_state.inputs[isim], key=f"t_{isim}")
        st.session_state.inputs[isim] = val
        input_data[isim] = val

df_unwanted = pd.DataFrame(False, index=isimler, columns=sutunlar)
for i, t in input_data.items():
    for d in parse_unwanted_days(t, gun_sayisi):
        if 1 <= d <= len(sutunlar): df_unwanted.at[i, sutunlar[d-1]] = True

if st.button("⚡ Nöbetleri Dağıt (AI Simülasyon)", type="primary"):
    run_scheduling_algorithm_v98(isimler, sutunlar, df_unwanted, gun_detaylari, min_bosluk)
    st.session_state.should_regenerate_assignments = True
    st.rerun()

# --- ADIM 2: EDİTÖR ---
st.divider()
st.subheader("📝 2. ADIM: Kontrol & Düzenleme")

edited = st.data_editor(
    st.session_state.schedule_bool.copy(),
    use_container_width=False, 
    key="schedule_editor",
    column_config={c: st.column_config.CheckboxColumn(l, width="small") for c, l in zip(sutunlar, sutunlar_display)}
)
st.session_state.schedule_bool = edited

# --- HATA KONTROL ---
violations = []
conflict_msg = []
max_person_msg = []
min_person_msg = []
last_shift_check = {i: -10 for i in isimler}

for col in sutunlar:
    gun_no = gun_detaylari[col]['day_num']
    nobetciler = edited.index[edited[col]].tolist()
    
    if len(nobetciler) > 2:
        max_person_msg.append(f"🔴 **{gun_no}. Gün**: {len(nobetciler)} kişi atanmış! (Max 2)")
    elif len(nobetciler) < 2:
        min_person_msg.append(f"🟠 **{gun_no}. Gün**: Eksik nöbetçi! ({len(nobetciler)} kişi var)")
        
    for k in nobetciler:
        if df_unwanted.at[k, col]:
            conflict_msg.append(f"❌ **{k}**: {gun_no}. gün müsait değilim demişti.")
        
        if (gun_no - last_shift_check[k]) <= min_bosluk:
            violations.append(f"⚠️ **{k}**: {gun_no}. gün dinlenme kuralına uymuyor.")
        last_shift_check[k] = gun_no

if max_person_msg or min_person_msg or conflict_msg or violations:
    with st.expander("🚨 HATA RAPORU (Tıklayıp Açın)", expanded=True):
        for m in max_person_msg: st.error(m)
        for m in min_person_msg: st.warning(m)
        for c in conflict_msg: st.error(c)
        for v in violations: st.info(v)
else:
    st.success("✅ Kurallar uygun (Her gün 2 kişi, çakışma yok).")

# --- VERİ HAZIRLIĞI ---
# Only regenerate assignments when AI button is clicked, not on manual edits
if st.session_state.should_regenerate_assignments or st.session_state.cached_rows_liste is None:
    rows_liste = []
    ayb_counts = {i: 0 for i in isimler}
    for col in sutunlar:
        nobetciler = edited.index[edited[col]].tolist()
        random.shuffle(nobetciler) 
        nobetciler.sort(key=lambda x: ayb_counts[x]) 
        p1, p2 = "-", "-"
        if len(nobetciler) > 0:
            p1 = nobetciler[0]; ayb_counts[p1] += 1
            if len(nobetciler) > 1: p2 = nobetciler[1]
        rows_liste.append({"Tarih": gun_detaylari[col]['full_date'], "AYB": p1, "GYB": p2})
    st.session_state.cached_rows_liste = rows_liste
    st.session_state.cached_ayb_counts = ayb_counts
    st.session_state.should_regenerate_assignments = False
else:
    rows_liste = st.session_state.cached_rows_liste
    ayb_counts = st.session_state.cached_ayb_counts
df_liste = pd.DataFrame(rows_liste)

stats_load = []
stats_finance = []
pair_matrix = pd.DataFrame('', index=isimler, columns=isimler, dtype=object)

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
        "AYB": ayb_counts[isim]
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
    if len(n) == 2:
        pair_matrix.loc[n[0], n[1]] = int(pair_matrix.loc[n[0], n[1]] or 0) + 1
        pair_matrix.loc[n[1], n[0]] = int(pair_matrix.loc[n[1], n[0]] or 0) + 1
for i in isimler: pair_matrix.loc[i,i] = "-" 

df_stats_load = pd.DataFrame(stats_load).set_index("İsim")
df_stats_finance = pd.DataFrame(stats_finance).set_index("İsim")

# --- GÖRÜNÜM ---
st.divider()
col_left, col_right = st.columns([1.3, 1])

# === SOL SÜTUN: LİSTE ===
with col_left:
    st.subheader("📅 Günlük Liste")
    
    c1, c2, c3 = st.columns(3)
    with c1: st.download_button("📥 CSV", df_liste.to_csv(index=False).encode('utf-8'), "liste.csv", "text/csv", type="primary")
    with c2: st.download_button("🖼️ PNG", convert_df_to_png(df_liste), "liste.png", "image/png")
    with c3: 
        with st.popover("💬 Metin"):
            st.text_area("Kopyala:", value=df_liste.to_markdown(index=False), height=200)

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
        df_stats_load.style.background_gradient(cmap="Blues", subset=["Toplam", "AYB"])
                           .background_gradient(cmap="Oranges", subset=["Özel", "HS"]),
        use_container_width=True
    )
    st.divider()

    # 2. Kişisel Detay
    st.markdown("**2. Kişisel Detay 🔍**")
    kisi_sec = st.selectbox("Kişi:", isimler, label_visibility="collapsed")
    kisi_rows = []
    for index, row in df_liste.iterrows():
        if row['AYB'] == kisi_sec:
            kisi_rows.append({"Tarih": row['Tarih'], "Partner": row['GYB'] if row['GYB'] != '-' else 'Tek', "Rol": "AYB"})
        elif row['GYB'] == kisi_sec:
            kisi_rows.append({"Tarih": row['Tarih'], "Partner": row['AYB'], "Rol": "GYB"})
    
    df_kisi = pd.DataFrame(kisi_rows)
    if not df_kisi.empty:
        h_kisi = len(df_kisi) * 35 + 38
        st.dataframe(df_kisi, height=h_kisi, use_container_width=True, hide_index=True)
    else:
        st.info("Nöbet yok.")

    st.divider()

    # 3. Matris
    st.markdown("**3. Eşleşme Matrisi**")
    st.dataframe(pair_matrix.style.background_gradient(cmap="Greys"), use_container_width=True)

    st.divider()

    # 4. Ücret
    st.markdown("**4. Ücret Özeti**")
    st.dataframe(
        df_stats_finance.style.background_gradient(cmap="Reds", subset=["FM", "Ücret (TL)"])
                              .format({"Ücret (TL)": "₺ {:,.2f}"}),
        use_container_width=True
    )