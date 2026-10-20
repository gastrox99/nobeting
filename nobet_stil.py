# -*- coding: utf-8 -*-
"""Sayfanın CSS'i ve açık/koyu tema renkleri."""

# --- GLOBAL RESPONSIVE CSS ---
GLOBAL_CSS = """
<style>
/* Keyed Streamlit containers keep day columns usable with horizontal scroll. */
.st-key-schedule-grid,
.st-key-preference-grid {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    padding-bottom: 8px;
}

@media (max-width: 768px) {
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
        min-width: 36px !important;
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
/* Ayarlar: ekip tablosu ve birlikte çalışamayanlar çapraz tablosu.
   Satırlar dar ekranda da yan yana kalır; çapraz tablo yana kayar. */
[data-testid="stExpander"] .st-key-ekip-tablo [data-testid="stHorizontalBlock"],
[data-testid="stExpander"] .st-key-yasak-tablo [data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap !important;
    gap: 6px;
}
[data-testid="stExpander"] .st-key-ekip-tablo :is([data-testid="column"],[data-testid="stColumn"]) {
    min-width: 0 !important;
    flex-basis: auto !important;
}
[data-testid="stExpander"] .st-key-ekip-tablo :is([data-testid="column"],[data-testid="stColumn"]):first-child {
    flex: 3 1 0 !important;
}
[data-testid="stExpander"] .st-key-ekip-tablo :is([data-testid="column"],[data-testid="stColumn"]):not(:first-child) {
    flex: 2 1 0 !important;
}
[data-testid="stExpander"] .st-key-ekip-tablo [data-testid="stVerticalBlock"] { gap: 4px; }
[data-testid="stExpander"] .st-key-ekip-tablo [data-testid="stNumberInputStepDown"],
[data-testid="stExpander"] .st-key-ekip-tablo [data-testid="stNumberInputStepUp"] { display: none; }
[data-testid="stExpander"] .st-key-ekip-tablo [data-testid="stCaptionContainer"] { line-height: 1.2; }
[data-testid="stExpander"] .st-key-yasak-tablo {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}
[data-testid="stExpander"] .st-key-yasak-tablo [data-testid="stVerticalBlock"] { gap: 0; }
[data-testid="stExpander"] .st-key-yasak-tablo [data-testid="stHorizontalBlock"] {
    min-width: max-content;
    gap: 0;
}
[data-testid="stExpander"] .st-key-yasak-tablo :is([data-testid="column"],[data-testid="stColumn"]) {
    flex: 0 0 34px !important;
    width: 34px !important;
    min-width: 34px !important;
    display: flex;
    align-items: center;
    justify-content: center;
}
[data-testid="stExpander"] .st-key-yasak-tablo :is([data-testid="column"],[data-testid="stColumn"]):first-child {
    flex: 0 0 110px !important;
    width: 110px !important;
    min-width: 110px !important;
    justify-content: flex-start;
}
[data-testid="stExpander"] .st-key-yasak-tablo [data-testid="stCheckbox"] { margin: 0 auto; min-height: 0; }
[data-testid="stExpander"] .st-key-yasak-tablo [data-testid="stCheckbox"] label { margin: 0; padding: 0; }
.yasak-baslik {
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    white-space: nowrap;
    max-height: 96px;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 12px;
    margin: 0 auto;
}
.yasak-isim {
    font-size: 13px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 106px;
}
.yasak-kendi { opacity: 0.35; text-align: center; }

/* Tablo üstü: boya renkleri bitişik, yazı genişliğinde düğmeler. */
.st-key-paint-palette [data-testid="stHorizontalBlock"] {
    flex-wrap: wrap !important;
    gap: 6px !important;
    justify-content: flex-end;
}
.st-key-paint-palette :is([data-testid="column"],[data-testid="stColumn"]) {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
}
.st-key-paint-palette .stButton > button { padding: 4px 12px; min-height: 36px; }
@media (max-width: 768px) {
    .st-key-paint-palette [data-testid="stHorizontalBlock"] { justify-content: flex-start; }
    /* İşlem düğmeleri alt alta 5 satır yerine iki satır: Simülasyon + İyileştir,
       altında Sıfırla + Geri + İleri. */
    .st-key-grid-actions [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 6px !important;
    }
    .st-key-grid-actions :is([data-testid="column"],[data-testid="stColumn"]) {
        flex: 1 1 calc(33% - 6px) !important;
        min-width: calc(33% - 6px) !important;
        width: auto !important;
    }
    .st-key-grid-actions :is([data-testid="column"],[data-testid="stColumn"]):nth-child(-n+2) {
        flex: 1 1 calc(50% - 6px) !important;
        min-width: calc(50% - 6px) !important;
    }
}

/* Üstteki ay çubuğu: ‹ Kasım 2026 › telefonda da tek satır. */
.st-key-month-bar [data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap !important;
    gap: 8px !important;
    align-items: center;
}
.st-key-month-bar :is([data-testid="column"],[data-testid="stColumn"]) {
    min-width: 0 !important;
    width: auto !important;
}
.st-key-month-bar :is([data-testid="column"],[data-testid="stColumn"]):is(:first-child, :last-child) {
    flex: 0 0 56px !important;
}
.st-key-month-bar :is([data-testid="column"],[data-testid="stColumn"]):nth-child(2) {
    flex: 1 1 auto !important;
}
.st-key-month-bar .stButton > button { font-size: 1.5rem; line-height: 1; min-height: 44px; }
.month-bar-title {
    text-align: center;
    font-size: 1.75rem;
    font-weight: 700;
    white-space: nowrap;
}
@media (max-width: 768px) {
    .month-bar-title { font-size: 1.4rem; }
}

/* Dosya yükleme kutusunun hazır İngilizce yazıları Türkçe. */
[data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-child { font-size: 0; }
[data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-child::after {
    content: "Dosyayı buraya sürükleyin";
    font-size: 0.875rem;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span:nth-child(2) { display: none; }
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] { font-size: 0; }
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"]::after {
    content: "Dosya seç";
    font-size: 0.875rem;
}

/* Takvim: geniş ekranda gün adı başlık satırında; telefonda sütunlar alt alta
   dizildiği için başlık ve boş kutular gizlenir, gün adı kartta yazar. */
.cal-gun-adi { display: none; font-weight: 600; }
/* Görev yeri değiştir (🔄): kartın altında ayrı satır yerine sağ üst köşede küçük simge. */
.st-key-cal-grid :is([data-testid="column"],[data-testid="stColumn"]) > [data-testid="stVerticalBlock"] {
    position: relative;
}
.st-key-cal-grid [class*="st-key-cal_swap_"] {
    position: absolute !important;
    top: 4px;
    right: 4px;
    width: auto !important;
    z-index: 5;
}
.st-key-cal-grid [class*="st-key-cal_swap_"] button {
    min-height: 26px !important;
    height: 26px !important;
    width: 30px !important;
    padding: 0 !important;
    font-size: 13px !important;
    background: transparent !important;
    border: 1px solid rgba(128,128,128,.3) !important;
    border-radius: 7px !important;
}
.st-key-cal-grid [class*="st-key-cal_swap_"] button:hover {
    background: rgba(128,128,128,.15) !important;
}
@media (max-width: 640px) {
    .st-key-cal-head { display: none !important; }
    .st-key-cal-grid :is([data-testid="column"],[data-testid="stColumn"]):has(.cal-empty) { display: none !important; }
    .cal-gun-adi { display: inline; }
}

</style>
"""


def ui_colors(_dark_theme):
    """Tablo, takvim ve kişi kartı renkleri; karanlık modda açık renkli yazılar."""
    return {
        "name_text": "#e5e7eb" if _dark_theme else "#1f2937",
        "name_bg": "#1e293b" if _dark_theme else "#f8fafc",
        "name_border": "#334155" if _dark_theme else "#e2e8f0",
        "muted": "#9ca3af",
        "day_text": "#e5e7eb" if _dark_theme else "#374151",
        "weekend_text": "#93c5fd" if _dark_theme else "#1d4ed8",
        "holiday_text": "#fca5a5" if _dark_theme else "#c62828",
        "today_text": "#fcd34d" if _dark_theme else "#b45309",
        "weekend_bg": "rgba(59,130,246,0.18)" if _dark_theme else "#dbeafe",
        "holiday_bg": "rgba(239,68,68,0.18)" if _dark_theme else "#fee2e2",
        "today_bg": "rgba(245,158,11,0.22)" if _dark_theme else "rgba(254,243,199,0.75)",
        "legend_text": "#d1d5db" if _dark_theme else "#374151",
        # Nöbetsiz hücreler: açık zemin + renkli yazı (nöbetliler dolu renk)
        # Açık temada yarı saydam: hafta sonu/tatil sütun zemini hücrelerden görünür.
        "free_bg": "rgba(148,163,184,0.10)" if _dark_theme else "rgba(255,255,255,0.6)",
        # Nöbet yazılmış (tercihsiz) hücre. Koyu temada açık zemin + koyu ✓:
        # koyu gri boş hücrelerle karışıyordu.
        "assigned_bg": "#cbd5e1" if _dark_theme else "#475569",
        "assigned_border": "#e2e8f0" if _dark_theme else "#334155",
        "assigned_text": "#0f172a" if _dark_theme else "#ffffff",
        "free_border": "#334155" if _dark_theme else "#e2e8f0",
        "free_text": "#64748b",
        "soft_green_bg": "rgba(34,197,94,0.16)" if _dark_theme else "#dcfce7",
        "soft_green_text": "#86efac" if _dark_theme else "#166534",
        "soft_amber_bg": "rgba(245,158,11,0.16)" if _dark_theme else "#fef3c7",
        "soft_amber_text": "#fcd34d" if _dark_theme else "#92400e",
        "soft_red_bg": "rgba(239,68,68,0.18)" if _dark_theme else "#fee2e2",
        "soft_red_text": "#fca5a5" if _dark_theme else "#991b1b",
        "soft_purple_bg": "rgba(168,85,247,0.18)" if _dark_theme else "#f3e8ff",
        "soft_purple_text": "#d8b4fe" if _dark_theme else "#6b21a8",
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


def grid_css(ui, name_col_px, col_bg_css):
    """Tercih/nöbet tablosunun CSS'i (gün sütunu renkleri ``col_bg_css`` ile)."""
    return f"""
<style>
.st-key-schedule-grid [data-testid="stHorizontalBlock"] > :is([data-testid="column"],[data-testid="stColumn"]):nth-child(1),
.st-key-preference-grid [data-testid="stHorizontalBlock"] > :is([data-testid="column"],[data-testid="stColumn"]):nth-child(1) {{
    background: {ui['name_bg']} !important;
    border-right: 2px solid {ui['name_border']};
    min-width: 80px;
}}
/* Yana kaydırınca isim sütunu solda sabit kalır (telefonda satırın kime ait
   olduğu görünsün); hücrelerin üzerinde durması için opak zemin ve üst katman. */
.st-key-schedule-grid [data-testid="stHorizontalBlock"] > :is([data-testid="column"],[data-testid="stColumn"]):nth-child(1),
.st-key-preference-grid [data-testid="stHorizontalBlock"] > :is([data-testid="column"],[data-testid="stColumn"]):nth-child(1) {{
    position: sticky;
    left: 0;
    z-index: 30;
    /* Satır boyu kaplasın (hücreler isimden uzun); soldaki kenar boşluğu da örtülsün. */
    align-self: stretch !important;
    display: flex;
    align-items: center;
    box-shadow: -16px 0 0 {ui['name_bg']};
}}
/* Başlık satırında gün adları satırın altına taşar; isim başlığı onları da örter. */
:is(.st-key-schedule-grid, .st-key-preference-grid) [data-testid="stHorizontalBlock"]:has(.grid-head) > :is([data-testid="column"],[data-testid="stColumn"]):nth-child(1) {{
    box-shadow: -16px 0 0 {ui['name_bg']}, -16px 18px 0 {ui['name_bg']}, 0 18px 0 {ui['name_bg']};
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
    background: {ui['assigned_bg']} !important;
    border-color: {ui['assigned_border']} !important;
    color: {ui['assigned_text']} !important;
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
:is(.st-key-preference-grid, .st-key-schedule-grid) :is([data-testid="column"],[data-testid="stColumn"]):has(.preference-state-4) .stButton > button {{
    background: #6b21a8 !important;
    border-color: #581c87 !important;
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
:is(.st-key-preference-grid, .st-key-schedule-grid) :is([data-testid="column"],[data-testid="stColumn"]):has(.preference-free-4) .stButton > button {{
    background: {ui['soft_purple_bg']} !important;
    border-color: transparent !important;
    color: {ui['soft_purple_text']} !important;
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
    /* Telefonda bir hafta (7 gün) ekrana sığsın: hücre 36 px, isim sütunu
       88 px ve kısa ad ("Ayşe Y."), rozet adın altında. */
    :is(.st-key-schedule-grid, .st-key-preference-grid) [data-testid="stHorizontalBlock"] > :is([data-testid="column"],[data-testid="stColumn"]) {{
        min-width: 36px !important;
        flex-shrink: 0 !important;
    }}
    :is(.st-key-schedule-grid, .st-key-preference-grid) [data-testid="stHorizontalBlock"] > :is([data-testid="column"],[data-testid="stColumn"]):nth-child(1) {{
        flex: 0 0 88px !important;
        min-width: 88px !important;
        max-width: 88px !important;
    }}
    .grid-name {{
        flex-wrap: wrap !important;
        row-gap: 1px !important;
        font-size: 12px !important;
        line-height: 1.2;
        /* Düğme (40 px) boyunda; iki satır (ad + rozet) dikeyde ortalı. */
        min-height: 40px !important;
        align-content: center;
    }}
    .grid-name .nm-full {{ display: none !important; }}
    .grid-name .nm-short {{ display: inline !important; max-width: 68px; }}
    .grid-name .nm-dot {{ width: 8px !important; height: 8px !important; }}
    .grid-name .nm-badge {{ font-size: 10px !important; padding: 0 5px !important; }}
    .st-key-preference-legend .legend-wide {{ display: none !important; }}
    .st-key-preference-legend .legend-fold {{ display: block !important; }}
    :is(.st-key-preference-grid, .st-key-schedule-grid) .stButton > button {{
        min-width: 34px !important;
        min-height: 40px !important;
        width: 34px !important;
        padding: 0 !important;
        font-size: 16px !important;
        font-weight: 800 !important;
        border-radius: 7px !important;
    }}
    :is(.st-key-preference-grid, .st-key-schedule-grid) .stButton > button:hover {{
        transform: none !important;
    }}
}}
.grid-name .nm-short {{ display: none; }}
.st-key-preference-legend .legend-fold {{ display: none; }}
.st-key-preference-legend .legend-fold summary {{ margin: 6px 0 2px; }}
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
.st-key-preference-legend .preference-legend-swatch.swatch-assigned_neutral {{
    color: {ui['assigned_text']} !important;
}}
</style>
"""
