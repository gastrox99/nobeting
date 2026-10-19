# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import random
import calendar
import time
import os
import zlib
# Streamlit Cloud yeni kodu çektiğinde db / nobet_core / arayüz modüllerinin
# eski hâli hafızada kalabiliyor ve yeni fonksiyonlar "ImportError" veriyor.
# Böyle olursa modülleri tazeleyip yeniden dene (Reboot gerekmesin).
import importlib
import db as _db_module
import nobet_core as _core_module

# Arayüz bölümleri ayrı dosyalarda. Küçük oldukları için her çalışmada
# yeniden okunurlar; böylece sunucuda eski bir kopya kalıp yeni nobet.py ile
# uyumsuz çalışamaz.
_UI_MODULES = (
    "nobet_stil", "nobet_durum", "nobet_paylasim", "nobet_disa_aktar",
    "nobet_yedek", "nobet_tablo", "nobet_takvim", "nobet_liste", "nobet_analiz",
)


def _fresh_modules():
    for _module in (_db_module, _core_module):
        importlib.reload(_module)


def _load_ui_modules():
    for _name in _UI_MODULES:
        importlib.reload(importlib.import_module(_name))


for _attempt in range(2):
    try:
        from db import (
            init_db,
            save_schedule,
            load_schedule,
            save_app_settings,
            load_app_settings,
            claim_legacy_data,
            get_last_error,
            is_database_configured,
            load_auto_saved_assignments,
        )
        from nobet_core import (
            history_adjustment,
            history_special_adjustment,
            holiday_shift_counts,
            last_shift_offsets,
            schedule_totals,
            unavailable_day_counts,
            previous_period,
            special_shift_counts,
            build_gun_detaylari,
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
            normalize_role_names,
            detect_renames,
            align_rows_to_roles,
            YEAR_MIN,
            YEAR_MAX,
            official_holidays,
            official_holidays_summary,
            official_half_days,
            required_work_hours,
            NOBET_UNVANLARI,
            ELLE_UNVAN,
            unvan_saat_ucreti,
            donem_adi,
            religious_holidays_known,
            active_locks,
            locks_to_json,
            assignment_warnings,
            collect_app_settings,
            persistence_fingerprint,
            build_schedule_analysis,
            changed_schedule_columns,
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
            build_limit_violation_messages,
            update_persistence_feedback,
            add_overflow_roles,
            as_bool_frame,
            holiday_text_for_period,
            default_holiday_text,
            prune_default_holidays,
            now_tr,
            resolve_alias,
            stash_removed_people,
            take_back_people,
            drop_unknown_people_from_rows,
        )
        _load_ui_modules()
        break
    except ImportError:
        if _attempt:
            raise
        _fresh_modules()
from nobet_stil import GLOBAL_CSS, ui_colors
from nobet_durum import (
    apply_renames, rotate_roles_for_day, save_undo_state,
)
from nobet_paylasim import render_shared_view
from nobet_disa_aktar import render_toolbar
from nobet_yedek import render_backup_section, run_requested_snapshot
from nobet_tablo import render_grid_tab
from nobet_takvim import render_calendar_tab
from nobet_liste import render_list_tab
from nobet_analiz import render_analysis_tab
from streamlit_local_storage import LocalStorage

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
# Ekip arkadaşları giriş yapmadan yalnızca listeyi görür (nobet_paylasim.py).
_share_token = st.query_params.get("paylas")
if _share_token:
    render_shared_view(_share_token, _db_ready)
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

# --- GLOBAL RESPONSIVE CSS (nobet_stil.py) ---
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-size: 2.5rem;'>Nöbet Yönetimi</h1>", unsafe_allow_html=True)
if AUTH_ENABLED:
    _user_cols = st.columns([5, 1])
    with _user_cols[0]:
        st.caption(f"👤 `{current_owner}`")
    with _user_cols[1]:
        st.button("Çıkış yap", on_click=st.logout, width="stretch")

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
            apply_renames(_renames)
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
        # Ünvan seçilince saat ücreti resmi tablodan (gösterge × katsayı)
        # gelir; elle değiştirilen ücret, ünvan/dönem/birim değişene kadar kalır.
        _unvan_options = list(NOBET_UNVANLARI) + [ELLE_UNVAN]
        _unvan_saved = st.session_state.get("unvan_val", ELLE_UNVAN)
        unvan = st.selectbox(
            "🩺 Ünvan:",
            _unvan_options,
            index=_unvan_options.index(_unvan_saved) if _unvan_saved in _unvan_options else len(_unvan_options) - 1,
            key="w_unvan_val",
        )
        st.session_state.unvan_val = unvan
        _rate_info = None
        if unvan != ELLE_UNVAN:
            riskli = st.checkbox(
                "Acil / yoğun bakım / ameliyathane (%50 artırımlı)",
                value=st.session_state.get("riskli_birim_val", False),
                key="w_riskli_birim_val",
            )
            st.session_state.riskli_birim_val = riskli
            _rate_info = unvan_saat_ucreti(unvan, yil, ay, riskli)
            _rate_source = f"{unvan}|{_rate_info['donem'][0]}-{_rate_info['donem'][1]}|{int(riskli)}"
            if st.session_state.get("nobet_ucreti_kaynak") != _rate_source:
                st.session_state.nobet_ucreti_kaynak = _rate_source
                st.session_state.nobet_ucreti_val = _rate_info["saatlik"]
                st.session_state["w_nobet_ucreti_val"] = float(_rate_info["saatlik"])
        else:
            st.session_state.nobet_ucreti_kaynak = None
        if "w_nobet_ucreti_val" not in st.session_state:
            st.session_state["w_nobet_ucreti_val"] = float(st.session_state.get("nobet_ucreti_val", 1.0))
        nobet_ucreti = st.number_input("💰 Saat Ücreti (TL):", key="w_nobet_ucreti_val")
        st.session_state.nobet_ucreti_val = nobet_ucreti
        if _rate_info:
            _rate_note = (
                f"Gündüz {_rate_info['gunduz']:.2f} TL × 12 sa + gece (20-08, %10) "
                f"{_rate_info['gece']:.2f} TL × 12 sa → {_rate_info['saatlik']:.2f} TL/sa · "
                f"{donem_adi(_rate_info['donem'])} katsayısı, damga vergisi düşülmüş."
            )
            if not _rate_info["tam"]:
                _rate_note += " Bu dönemin katsayısı henüz eklenmedi, son bilinen kullanıldı."
            if abs(nobet_ucreti - _rate_info["saatlik"]) > 0.005:
                _rate_note += " Ücret elle değiştirildi."
            st.caption(_rate_note)
    
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
            # Yalnızca en fazla nöbet girilir; boş = sınırsız.
            _old_max = _lim_current.get(_person, {}).get("max", 999)
            _max = st.number_input(
                f"{_person}: en fazla nöbet", min_value=0, max_value=31, step=1,
                value=None if _old_max >= 999 else min(31, _old_max),
                key=f"w_limit_max_{_person}_{_lim_key}", placeholder="sınırsız",
            )
            _lim_new[_person] = {"min": 0, "max": 999 if _max is None else int(_max)}
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
    # "Sınırsız" bırakılan kişi (0-999) limitli sayılmaz; yoksa aylar arası
    # toplam dengesinden ve İyileştir'in toplam dengesinden habersizce çıkıyordu.
    person_limits = {
        _p: _v for _p, _v in person_limits.items() if _v["min"] > 0 or _v["max"] < 999
    }
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

# --- YEDEK: tüm aylar ve ayarlar tek dosyada; aynı dosyadan geri yükleme (nobet_yedek.py) ---
if _db_ready:
    render_backup_section(current_owner, yil, ay, isimler)

if not isimler: st.stop()

# --- GÜN VERİLERİ ---
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

# Karanlık modda açık renkli yazılar/zeminler; eskiden isimler açık zemin
# üzerinde açık renkte kalıp görünmüyordu.
try:
    _dark_theme = st.context.theme.type == "dark"
except Exception:
    _dark_theme = False
ui = ui_colors(_dark_theme)

tab_grid, tab_cal = st.tabs(["🗂️ Tablo Görünümü", "📅 Takvim Görünümü"])

# ═══════════════════════════════════════════════════════
# TAB 1: TABLO GÖRÜNÜMü (nobet_tablo.py)
# ═══════════════════════════════════════════════════════
with tab_grid:
    render_grid_tab(isimler, sutunlar, gun_detaylari, yil, ay, today_date, ui,
                    person_colors, conflict_cells, _queue_assignment_warnings)

# ═══════════════════════════════════════════════════════
# TAB 2: TAKVİM GÖRÜNÜMü (nobet_takvim.py)
# ═══════════════════════════════════════════════════════
with tab_cal:
    render_calendar_tab(isimler, sutunlar, gun_detaylari, yil, ay, gun_sayisi, tatil_gunleri,
                        today_date, ui, person_colors, kişi_sayısı, zorunlu_saat, nobet_ucreti)

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

# İyileştir: mevcut listeden OR-Tools ile daha adilini arar (nobet_iyilestir).
if st.session_state.pop("run_improve", False):
    st.session_state.incomplete_simulation = None
    try:
        from nobet_iyilestir import improve_schedule, improvement_lines, rule_breaking_shifts
    except ImportError:
        improve_schedule = None
    if improve_schedule is None:
        st.error("İyileştir için gereken çözücü (ortools) yüklü değil.")
    else:
        with st.spinner("Daha adil liste aranıyor (birkaç saniye)..."):
            try:
                _improved, _before, _after = improve_schedule(
                    st.session_state.schedule_bool, isimler, sutunlar, df_unwanted, gun_detaylari,
                    min_bosluk, kişi_sayısı,
                    forbidden_pairs=st.session_state.forbidden_pairs,
                    person_limits=st.session_state.get('person_limits', {}),
                    df_preferred=df_preferred,
                    fixed_assignments=active_locks(st.session_state.get("locked_cells"), st.session_state.schedule_bool),
                    initial_last_shift=prev_last_shift,
                    history_special=history_special,
                    history_total=history_total,
                    history_holiday=history_holiday,
                )
            except Exception as _exc:
                _improved = None
                st.error(f"İyileştirme yapılamadı: {_exc}")
            else:
                if _improved is None:
                    _breaking = rule_breaking_shifts(
                        st.session_state.schedule_bool, isimler, sutunlar, df_unwanted, gun_detaylari,
                        min_bosluk, forbidden_pairs=st.session_state.forbidden_pairs,
                        fixed_assignments=active_locks(st.session_state.get("locked_cells"), st.session_state.schedule_bool),
                        initial_last_shift=prev_last_shift,
                    )
                    if _breaking:
                        st.session_state.improve_notice = ("info", [
                            f"Listede kurala uymayan {_breaking} nöbet var (izin günü, dinlenme ya da "
                            "birlikte çalışamayan ikili). İyileştir bu yüzden çözüm bulamadı; önce bu "
                            "nöbetleri düzeltin ya da Simülasyon'u çalıştırın. Mevcut liste korundu."
                        ])
                    else:
                        st.session_state.improve_notice = ("info", ["Daha iyi bir liste bulunamadı; mevcut liste korundu."])
                else:
                    save_undo_state()
                    st.session_state.schedule_bool = _improved
                    # Görev yerleri yalnızca değişen günlerde güncellenir (elle
                    # yapılan görev değişiklikleri korunur).
                    st.session_state.improve_notice = ("success", improvement_lines(_before, _after))
                st.rerun()

_improve_notice = st.session_state.pop("improve_notice", None)
if _improve_notice:
    _kind, _lines = _improve_notice
    if _kind == "success":
        st.success(
            "✨ Liste iyileştirildi: " + ", ".join(_lines) + ". Beğenmezseniz ↩️ Geri ile dönebilirsiniz."
        )
    else:
        st.info(_lines[0])

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
if _liste_swap_dn is not None and rotate_roles_for_day(_liste_swap_dn):
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
run_requested_snapshot(current_owner, yil, ay)

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

# ── İndirme/paylaşım/yükleme/yedek düğmeleri ince yatay satır (nobet_disa_aktar.py) ──
render_toolbar(df_liste, df_stats_load, df_stats_finance, liste_rows, liste_roles,
               yil, ay, gun_detaylari, isimler, sutunlar, role_names,
               _db_ready, current_owner, save_undo_state)

st.divider()

tab_liste, tab_analiz = st.tabs(["📋 Günlük Liste", "📊 Analiz"])

with tab_liste:
    render_list_tab(rows_liste, liste_rows, liste_roles, role_names, isimler, sutunlar, edited_df)

with tab_analiz:
    render_analysis_tab(isimler, role_names, _prev_ctx, history_special, history_total, history_holiday,
                        df_stats_load, df_stats_finance, pair_display, df_liste, liste_rows, liste_roles,
                        yil, ay, zorunlu_saat, arife_gunleri, edited_df)
