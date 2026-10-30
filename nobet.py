# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import calendar
import contextlib
import time
import os
import random
# Doğrulama koşusu aynı listeyi tekrar üretebilsin diye. Değişken boşsa
# davranış aynı kalır; her sayfa çiziminde tohum başa alınır.
_nobet_seed = os.environ.get("NOBET_SEED")
if _nobet_seed:
    random.seed(int(_nobet_seed))
# Streamlit Cloud yeni kodu çektiğinde db / nobet_core / nobet_iyilestir'in
# eski hâli hafızada kalabiliyor: yeni fonksiyonlar "ImportError" veriyor ya da
# eski kod sessizce çalışıyordu (İyileştir "ortools yüklü değil" diyordu).
# Her çalışmada dosyası değişen modül (ve ona bağlı olanlar) kilit altında
# yeniden yüklenir (nobet_tazele.py; o dosyanın kendisi yeniden yüklenmez).
import importlib
import importlib.util
from html import escape as _escape
import db as _db_module  # noqa: F401
import nobet_core as _core_module  # noqa: F401
from nobet_tazele import code_version, fresh_modules

# Arayüz bölümleri ayrı dosyalarda. Küçük oldukları için her çalışmada
# yeniden okunurlar; böylece sunucuda eski bir kopya kalıp yeni nobet.py ile
# uyumsuz çalışamaz. Sıra önemli: bir modülün içe aktardığı arayüz modülü ondan önce yeniden yüklenir.
_UI_MODULES = (
    "nobet_stil", "nobet_durum", "nobet_yil", "nobet_onkontrol", "nobet_paylasim", "nobet_disa_aktar",
    "nobet_yedek", "nobet_tablo", "nobet_takvim", "nobet_analiz",
    "nobet_ekip", "nobet_talepler",
)


# Sıra önemli: db ve nobet_core, nobet_kalicilik'ten; nobet_core, nobet_gunler/nobet_ucret/nobet_algoritma'dan;
# nobet_iyilestir nobet_core'dan isim alır.
_FRESH_MODULES = ("nobet_kalicilik", "db", "nobet_gunler", "nobet_ucret", "nobet_algoritma", "nobet_core", "nobet_iyilestir")


def _fresh_modules(force=False):
    """Dosyası diskte değişmiş modülleri kilit altında yeniden yükler (nobet_tazele.py).

    Yükleme yarıda kalırsa damga yazılmaz, sonraki çalışma yeniden dener; aynı anda
    gelen başka oturum yüklemenin bitmesini bekler.
    """
    fresh_modules(_FRESH_MODULES, force=force)


def _render_version_footer():
    """Sayfanın en altında küçük, soluk sürüm satırı: sunucudaki kodun sürümü görünür."""
    st.markdown(
        '<div class="surum-etiketi" style="text-align:center;font-size:0.7rem;'
        f'color:#9ca3af;margin-top:2rem;">sürüm {_escape(code_version())}</div>',
        unsafe_allow_html=True,
    )


_fresh_modules()


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
            SaveConflict,
            MonthLocked,
            month_is_locked,
            VersionUnknown,
            log_unexpected,
            prepare_schema,
            ping_db,
            SCHEMA_VERSION,
        )
        from nobet_kalicilik import kaydi_durdur, kayit_durdu_mu, engeli_kaldir, engelleri_sifirla, otomatik_kayit_adi
        from nobet_core import (
            history_adjustment,
            history_special_adjustment,
            holiday_shift_counts,
            last_shift_offsets,
            schedule_totals,
            unavailable_day_counts,
            is_blocked_preference,
            personal_work_hours,
            previous_period,
            special_shift_counts,
            build_gun_detaylari,
            parse_forbidden_pairs,
            parse_together_pairs,
            parse_person_limits_with_issues,
            unknown_names,
            has_total_limit,
            special_capped_people,
            rename_in_pairs_text,
            validate_inputs,
            normalize_owner_email,
            apply_app_settings,
            normalize_role_names,
            detect_renames,
            align_rows_to_roles,
            YEAR_MIN,
            YEAR_MAX,
            TR_AY_ADLARI,
            official_holidays,
            official_holidays_summary,
            official_half_days,
            required_work_hours,
            NOBET_UNVANLARI,
            ELLE_UNVAN,
            katsayi_uyarisi,
            unvan_saat_ucreti,
            donem_adi,
            tr_sayi,
            PENDING_PREFS_KEY,
            pending_prefs_payload,
            apply_pending_prefs,
            is_pending_prefs_for,
            TATIL_ARTIRIMI,
            GECE_FARKI,
            UCRET_TURLERI,
            saat_ucretleri,
            religious_holidays_known,
            active_locks,
            locks_to_json,
            assignment_warnings,
            collect_app_settings,
            PERIOD_STATE_KEYS,
            merge_app_settings,
            settings_difference_labels,
            writer_tag,
            conflict_writer_text,
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
            update_locked_cells,
            MISSING_DAY_FREE_LABEL,
            format_missing_day_reason,
            synchronize_changed_role_rows,
            active_fixed_roles,
            apply_fixed_roles,
            distribute_roles,
            fixed_role_conflicts,
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
            simulation_summary,
        )
        _load_ui_modules()
        break
    except ImportError:
        if _attempt:
            print(f"[nobet] Yenilemeden sonra da içe aktarılamadı (sürüm {code_version()})", flush=True)
            raise
        _fresh_modules(force=True)
from nobet_stil import GLOBAL_CSS, ui_colors
from nobet_durum import apply_renames, render_recent_errors, save_undo_state
from nobet_paylasim import render_shared_view
from nobet_disa_aktar import render_excel_import, render_toolbar
from nobet_yedek import render_backup_section, run_requested_snapshot, take_daily_snapshot
from nobet_tablo import render_grid_tab
from nobet_takvim import render_calendar_tab
from nobet_analiz import render_analysis_tab
from nobet_yil import render_year_summary
from nobet_onkontrol import (
    ay_sonu_cakismalari, ilk_nobet_gunleri, paylasim_kontrolleri, render_backup_suggestion, render_precheck,
    render_why_not,
)
from nobet_ekip import render_team_table, render_pair_matrix, render_fixed_roles
from nobet_talepler import (
    finish_approved_requests, pending_count, render_pending_badge, render_request_status, render_team_requests,
)
from streamlit_local_storage import LocalStorage

# Veritabanını sunucu başına bir kez hazırla; başarısızsa sonucu kalıcı
# saklama (eskiden "bağlanamadım" önbellekte kalıyor, Secrets'a adres
# eklense bile yeniden başlatmaya kadar uyarı sürüyordu). Her yeni
# oturumda bir kez yeniden denenir. Şema sürümü değişince (yeni tablo)
# sunucu yeniden başlamasa da hazırlık yeniden çalışır.
@st.cache_resource
def _db_init_state(schema_version):
    return {"ready": False, "error": None}

_db_state = _db_init_state(SCHEMA_VERSION)
# Başarı bayrağı yalnızca True yönünde değişir; başarısız oturum 30 sn sonra yeniden dener.
# Şema hazırsa yeni oturum tek bir uyandırma bağlantısı açar (Neon uyuyor olabilir);
# ulaşılamazsa bu çalışmadaki diğer okumalar atlanır, her biri ayrı ayrı beklemez.
_db_ready = prepare_schema(
    _db_state, st.session_state, SCHEMA_VERSION, init_fn=init_db, error_fn=get_last_error, ping_fn=ping_db,
)

# Sayfa Ayarları
_APP_ICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "ikon-64.png")
st.set_page_config(page_title="Adil Nöbet", page_icon=_APP_ICON, layout="wide")
# Safari "Ana Ekrana Ekle" simgesi (Streamlit kendisi eklemiyor; yoksa sayfa
# görüntüsü kullanılıyordu). Dosya config.toml enableStaticServing ile sunulur.
st.html(
    "<script>if(!document.querySelector('link[rel=apple-touch-icon]')){"
    "const l=document.createElement('link');l.rel='apple-touch-icon';"
    "l.href='app/static/ikon-180.png';document.head.appendChild(l);}</script>",
    unsafe_allow_javascript=True,
)

# --- SALT OKUNUR PAYLAŞIM SAYFASI (?paylas=...) ---
# Ekip arkadaşları giriş yapmadan yalnızca listeyi görür (nobet_paylasim.py).
_share_token = st.query_params.get("paylas")
if _share_token:
    render_shared_view(_share_token, _db_ready)
    _render_version_footer()
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
if not _db_ready and st.session_state.get("_db_unreachable"):
    st.warning(
        "⏳ Veritabanına şu an ulaşılamıyor (uykudan uyanıyor olabilir). Kayıtlı ekip ve "
        "çizelge yüklenmedi; bu sırada yapılan değişiklikler kaydedilmez. Biraz sonra "
        "sayfayı yenileyin."
    )
elif not _db_ready:
    _db_reason = f" Sebep: {_db_state['error']}" if _db_state.get("error") else ""
    st.warning(
        "⚠️ Veritabanına bağlanılamadı. Kaydet/Yükle ve otomatik kayıt "
        f"özellikleri şu an çalışmayabilir.{_db_reason}"
    )

def _version_box(key):
    """Kaydın son bilinen sürümünü tutan, oturumda saklı değiştirilebilir sözlük.

    Kayıt yeni sürümü bu sözlüğe yerinde yazar (``version_out``). Sürüm kayıttan
    sonra ``st.session_state``'e ayrıca atansaydı, telefonda hızlı dokunuşların
    başlattığı yeni çalıştırma betiği tam o atamada kesebiliyordu: veri yazılmış
    ama sürüm eski kalmış olurdu ve sonraki kayıt kendi değişikliğini "başka
    yerde değiştirilmiş" sanıp duruyordu.
    """
    box = st.session_state.get(key)
    if not isinstance(box, dict):
        box = {"version": box}
        st.session_state[key] = box
    return box


# Okunamayan (bozuk) kayıt bölümleri: kullanıcının gördüğü adlar.
CORRUPT_PART_LABELS = {
    "pref_json": "Tercihler",
    "settings_json": "Kilitler ve saklı kişiler",
    "rows_liste_json": "Görev yerleri",
    "app_settings": "Ekip ve ayarlar",
}
CORRUPT_RECOVER_LABEL = "Okunamayan bölümü boşaltıp devam et"
_BACKUP_HINT = (
    "Bu ayın yedeği varsa sayfanın altındaki «Yedek ve Excel yükleme» bölümünden "
    "geri yükleyebilirsiniz"
)


def _corrupt_labels(fields):
    return ", ".join(dict.fromkeys(CORRUPT_PART_LABELS.get(_f, _f) for _f in fields))


def _render_corrupt_notice(kind, message, labels, on_confirm):
    """Bozuk kayıt uyarısı ve açık onaylı "boşaltıp devam et" düğmesi."""
    st.warning(message)
    _confirmed = st.checkbox(
        f"Okunamayan bölüm ({labels}) boş kabul edilsin; bu bölümdeki kayıtlı bilgi silinir.",
        key=f"corrupt_confirm_{kind}",
    )
    st.button(
        CORRUPT_RECOVER_LABEL, key=f"corrupt_recover_{kind}",
        disabled=not _confirmed, on_click=on_confirm,
    )


def _own_writer_tag():
    """Bu oturumun cihaz/oturum etiketi; kayıtla birlikte saklanır, çakışmada gösterilir."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        _session = get_script_run_ctx().session_id
    except Exception:
        _session = None
    try:
        _agent = st.context.headers.get("User-Agent")
    except Exception:
        _agent = None
    return writer_tag(_agent, _session)


_writer = _own_writer_tag()


# --- Veritabanı açılışta yoktu, şimdi var: kayıtlı veri bu oturumda yüklenir ---
# Ekip/ayarlar ve ay, veritabanına ulaşılamadığı için yüklenmeden açıldıysa ve o
# zamandan beri bir şey değiştirilmediyse şimdi yüklenir. Değiştirildiyse üzerine
# yazılmaz; kayıt kapalı kalır ve yenileme istenir.
_offline = st.session_state.get("loaded_without_db")
if _db_ready and _offline:
    _offline_clean = _offline.get("schedule_clean", True) and (
        not _offline.get("settings_skipped")
        or collect_app_settings(st.session_state) == _offline.get("settings")
    )
    if _offline_clean:
        if _offline.get("settings_skipped"):
            st.session_state.pop("app_settings_loaded", None)
        if _offline.get("period") is not None:
            for _key in PERIOD_STATE_KEYS:
                st.session_state.pop(_key, None)
        st.session_state.loaded_without_db = None
    else:
        st.warning(
            "⚠️ Veritabanına yeniden ulaşıldı, ancak bağlantı yokken değişiklik yapıldığı için "
            "kayıtlı ekip ve çizelge yüklenmedi; kayıt yapılmıyor. Değişiklikleri "
            "kaybetmek istemiyorsanız önce Excel olarak indirin, sonra sayfayı yenileyin."
        )

# --- Ekip ve ayarları veritabanından geri yükle (sayfa yenilemesinde) ---
if 'app_settings_loaded' not in st.session_state:
    _stored_settings = None
    # Okuma başarısızsa kayıtlı ayarların üzerine boş ayar yazılmasın.
    st.session_state.app_settings_load_failed = not _db_ready
    if not _db_ready:
        # Veritabanı gelince yüklenecek; o zamana kadar değişiklik olursa yüklenmez.
        st.session_state.loaded_without_db = {
            **(st.session_state.get("loaded_without_db") or {}),
            "settings_skipped": True, "settings": None,
        }
    if _db_ready:
        try:
            # Sürüm, ayarlarla aynı okumadan gelir (ayrı okunup hata verince None
            # kalıyor, çakışma denetimi kapanıyordu). Sürüm değiştirilebilir bir
            # kutuda tutulur: kayıt bu kutuyu yerinde günceller (bkz. _version_box).
            st.session_state.settings_version = {"version": None}
            st.session_state.pop("settings_corrupt", None)
            _settings_corrupt = []
            _stored_settings = load_app_settings(
                owner=current_owner, raise_on_error=True, version_out=st.session_state.settings_version,
                corrupt_out=_settings_corrupt,
            )
            if _settings_corrupt:
                # Bozuk kayıt boş sayılıp üzerine yazılmasın (sessiz veri kaybı);
                # kullanıcı onaylayınca bilinen sürümle yeniden yazılabilir.
                st.session_state.app_settings_load_failed = True
                st.session_state.settings_corrupt = _settings_corrupt
        except Exception as _error:
            log_unexpected("Ayarları okuma", _error)
            st.session_state.app_settings_load_failed = True
            st.warning("⚠️ Kayıtlı ekip ve ayarlar okunamadı. Sayfayı yenileyin.")
    apply_app_settings(st.session_state, _stored_settings)
    # Resmi tatillerle aynı kayıtlar silinir; koddaki bayram düzeltmeleri yansısın.
    prune_default_holidays(st.session_state.get("holidays_by_period"))
    st.session_state.last_saved_settings = (
        collect_app_settings(st.session_state) if _stored_settings else None
    )
    # En sonda: telefonda dokunuş bu çalışmayı yarıda keserse yükleme yeniden
    # yapılır (yarım yüklenmiş ayarlar kayıtlıların üzerine yazılmasın).
    st.session_state.app_settings_loaded = True

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

if AUTH_ENABLED:
    _user_cols = st.columns([5, 1])
    with _user_cols[0]:
        st.caption(f"👤 `{current_owner}`")
    with _user_cols[1]:
        st.button("Çıkış yap", on_click=st.logout, width="stretch")

# --- ANA ALGORİTMA: nobet_core.run_scheduling_core kullanır ---
def run_scheduling_algorithm_v98(isimler, sutunlar, df_unwanted_bool, gun_detaylari, min_bosluk, kisi_sayisi, forbidden_pairs=None, person_limits=None, df_preferred=None, initial_last_shift=None, history_special=None, fixed_assignments=None, history_total=None, history_holiday=None, together_pairs=None):
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
            together_pairs=together_pairs,
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
def _set_period_state(year, month):
    """Dönemi yazar; ay/yıl kutularının anahtarlarına yalnızca Ayarlar açıkken yazılır.

    Kapalı panelde çizilmeyen kutunun anahtarına kodla yazılan değer Streamlit'te
    saklı kalır ve kutu sonra başka değer alsa da geri gelirdi; kapalıyken kutu
    anahtarı silinir, kalıcı değer yil_val/ay_val'dır.
    """
    st.session_state.yil_val, st.session_state.ay_val = year, month
    if st.session_state.get("settings_expanded", True):
        st.session_state.yil_input_direct, st.session_state.ay_input_direct = year, month
    else:
        st.session_state.pop("yil_input_direct", None)
        st.session_state.pop("ay_input_direct", None)


_period_revert = st.session_state.pop("period_revert", None)
if _period_revert:
    # Ay/yıl kutuları çizilmeden önce eski değere döndürülür (kutular çizildikten sonra değiştirilemez).
    _set_period_state(*_period_revert)
    st.session_state.period_revert_notice = st.session_state.pop("period_revert_reason", "db")
prev_isimler = st.session_state.get('isimler_cache', [])

# Varsayılan dönem: gelecek ay (nöbet listesi genelde bir ay önceden hazırlanır).
_now = now_tr()
if _now.month == 12:
    current_year = _now.year + 1
    current_month = 1
else:
    current_year = _now.year
    current_month = _now.month + 1


def _shift_period(delta):
    """Üstteki ‹ › oku: ayı bir öne/geri alır (Ayarlar'daki kutular da güncellenir)."""
    _y = st.session_state.get("yil_input_direct", st.session_state.get("yil_val", current_year))
    _m = st.session_state.get("ay_input_direct", st.session_state.get("ay_val", current_month))
    _index = _y * 12 + (_m - 1) + delta
    _y, _m = divmod(_index, 12)
    _m += 1
    if YEAR_MIN <= _y <= YEAR_MAX:
        _set_period_state(_y, _m)


# Ay ve yıl her zaman en üstte görünür (Ayarlar kapalıyken de).
_bar_year = st.session_state.get("yil_input_direct", st.session_state.get("yil_val", current_year))
_bar_month = st.session_state.get("ay_input_direct", st.session_state.get("ay_val", current_month))
_bar_cols = st.container(key="month-bar").columns([1, 4, 1], vertical_alignment="center")
_bar_cols[0].button(
    "‹", key="month_prev", help="Önceki ay", width="stretch",
    on_click=_shift_period, args=(-1,), disabled=(_bar_year, _bar_month) <= (YEAR_MIN, 1),
)
_bar_cols[1].markdown(
    f"<div class='month-bar-title'>{TR_AY_ADLARI[_bar_month]} {_bar_year}</div>",
    unsafe_allow_html=True,
)
_bar_cols[2].button(
    "›", key="month_next", help="Sonraki ay", width="stretch",
    on_click=_shift_period, args=(1,), disabled=(_bar_year, _bar_month) >= (YEAR_MAX, 12),
)
# Biten ay 5 gün sonra kilitlenir; kilidi bu düğmeyle açıp kapatırsınız.
_bar_key = period_key(_bar_year, _bar_month)
_bar_locked = month_is_locked(_bar_year, _bar_month, st.session_state.get("open_months"))
st.session_state.ay_kilitli = _bar_locked


def _toggle_month_lock(key, unlock):
    _others = [m for m in (st.session_state.get("open_months") or []) if m != key]
    st.session_state.open_months = sorted(_others + [key]) if unlock else _others
    # Kilitli ayda durdurulmuş otomatik kayıt, kilit açılınca yeniden başlasın;
    # başka nedenle (ör. okunamayan bölüm) durdurulduysa durdurma sürer.
    engeli_kaldir(st.session_state, (_bar_year, _bar_month), "kilitli")


if month_is_locked(_bar_year, _bar_month, []):
    _lock_cols = st.container(key="month-lock").columns([4, 1], vertical_alignment="center")
    _lock_cols[0].caption(
        "🔒 Bu ay kilitli (bittikten 5 gün sonra kilitlenir); değişiklik yapılamaz."
        if _bar_locked else "🔓 Bu ay düzenlemeye açık."
    )
    _lock_cols[1].button(
        "🔓 Kilidi aç" if _bar_locked else "🔒 Kilitle", key="month_lock_toggle", width="stretch",
        on_click=_toggle_month_lock, args=(_bar_key, _bar_locked),
    )
if _db_ready:
    render_pending_badge(current_owner)
# Ekip yoksa Ayarlar açık başlar. Değer oturum boyunca sabit kalır; yoksa
# ilk isim yazılınca panel kapanıyordu (ekip tablosu satır satır doldurulur).
if "settings_expanded" not in st.session_state:
    st.session_state.settings_expanded = not st.session_state.get("isimler_text", "").strip()
settings_expanded = st.session_state.settings_expanded


def _toggle_settings():
    _open = not st.session_state.get("settings_expanded", False)
    st.session_state.settings_expanded = _open
    # Kapanma çalışmasında, Enter/Tab'a basılmadan yazılıp başlığa tıklanınca gelen
    # kutu değerleri (ekip tablosu, çapraz tablolar) çizimsiz işlenir.
    st.session_state._ayarlar_kapaniyor = not _open


def _ayar_kutusu(widget, key, closed_value, *args, **kwargs):
    """Ayarlar açıksa kutuyu çizer; kapalıysa çizmeden değeri döner ve kutu durumunu siler.

    Kapalıyken kutunun durumu tamamen silinir (pop): Streamlit çizilmeyen kutunun
    son değerini (kutu kimliğiyle saklanan) atar ama aynı anahtara kodla daha önce
    yazılmış eski değer (ör. ok tuşunun yazdığı ay, varsayılan ücret) kalır ve
    sonraki çalışmada geri gelirdi. Kalıcı değer ``closed_value`` (``*_val``) içinde;
    bu çalışmada kutudan ya da koddan gelen değer varsa o döner.
    """
    if settings_expanded:
        return widget(*args, key=key, **kwargs)
    if key in st.session_state:
        return st.session_state.pop(key)
    return closed_value


# st.expander kapalıyken de içini her çalışmada çizer (14 kişide ~360 öğe, her
# tıklamada ~160 ms). Bu yüzden başlık bir düğme; açık/kapalı oturumda saklanır
# ve kapalıyken yalnızca hesaplar yapılır, kutular çizilmez.
_settings_panel = st.container(border=True, key="ayarlar-panel")
_settings_panel.button(
    "⚙️ Ayarlar", key="ayarlar_ac_kapa", on_click=_toggle_settings, type="tertiary",
    width="stretch", icon=":material/expand_more:" if settings_expanded else ":material/chevron_right:",
)
# Bayrak panel işlendikten sonra silinir: yarıda kesilen çalışma sonraki çalışmada tekrarlanır.
_settings_closing = bool(st.session_state.get("_ayarlar_kapaniyor", False)) and not settings_expanded

with (_settings_panel.container(key="ayarlar-icerik") if settings_expanded else _settings_panel):
    # Sol: ekip tablosu. Sağ: dönem, tatil, ücret ve görev ayarları alt alta.
    if settings_expanded:
        set_col1, set_col2 = st.columns([3, 2])
    else:
        set_col1 = set_col2 = contextlib.nullcontext()
    
    with set_col1:
        # Ekip tablosu (isim + limitler) ayarlardaki metinleri düzenler (nobet_ekip.py).
        if settings_expanded or _settings_closing:
            render_team_table(draw=settings_expanded)
        if not settings_expanded:
            # Açılınca ekip tablosu kutuları ayarlardaki metinden yeniden kurulsun
            # (kapalıyken silinen kutu durumu boş isim sanılmasın).
            st.session_state.pop("_ekip_source", None)
        isimler_input = st.session_state.get("isimler_text", "")
        _raw_isimler = [x.strip() for x in isimler_input.split(",") if x.strip()]
        # Aynı isim iki kez yazılırsa tablo çöküyordu; her ismi bir kez kullan.
        isimler = list(dict.fromkeys(_raw_isimler))
        if settings_expanded and len(isimler) < len(_raw_isimler):
            _dupes = sorted({x for x in _raw_isimler if _raw_isimler.count(x) > 1})
            st.warning(f"⚠️ Birden fazla yazılan isim(ler) bir kez kullanıldı: **{', '.join(_dupes)}**")
        # Bir ismin yazımı düzeltilince o kişinin nöbetleri/tercihleri silinmesin.
        _seen_names = st.session_state.setdefault("seen_team_names", set(prev_isimler))
        _renames = detect_renames(
            prev_isimler, isimler,
            known=_seen_names | set(st.session_state.get("hidden_people") or {}),
            aliases=st.session_state.get("name_aliases"),
        )
        if _renames:
            apply_renames(_renames)
            for _pairs_key in ("forbidden_pairs_text", "together_pairs_text"):
                st.session_state[_pairs_key] = rename_in_pairs_text(
                    st.session_state.get(_pairs_key, ""), prev_isimler, _renames
                )
            if st.session_state.get("fixed_roles"):
                st.session_state.fixed_roles = {
                    _renames.get(_p, _p): _r for _p, _r in st.session_state.fixed_roles.items()
                }
            st.toast("İsim değişikliği uygulandı: " + ", ".join(f"{a} → {b}" for a, b in _renames.items()), icon="✏️")
        st.session_state.isimler_cache = isimler
        _seen_names.update(isimler)
        
        # Tarayıcı deposuna yalnızca ekip listesi değiştiğinde yaz.
        if st.session_state.get("last_stored_team_text") != isimler_input:
            local_storage.setItem("nobet_team", isimler_input)
            st.session_state.last_stored_team_text = isimler_input
    
    with set_col2:
        yil = _ayar_kutusu(st.number_input, 'yil_input_direct', st.session_state.get('yil_val', current_year),
                           "📅 Yıl", YEAR_MIN, YEAR_MAX, value=st.session_state.get('yil_val', current_year))
        st.session_state.yil_val = yil
        
        ay_options = list(range(1, 13))
        ay = _ayar_kutusu(st.selectbox, 'ay_input_direct', st.session_state.get('ay_val', current_month),
                          "📆 Ay", ay_options, index=ay_options.index(st.session_state.get('ay_val', current_month)),
                          format_func=lambda m: TR_AY_ADLARI[m])
        st.session_state.ay_val = ay
        # Tarayıcı sekmesinde/Ana Ekranda hangi ayın açık olduğu görünsün.
        st.set_page_config(page_title=f"{TR_AY_ADLARI[ay]} {yil} · Adil Nöbet")
        
        gun_sayisi = calendar.monthrange(yil, ay)[1]
        kişi_sayısı = _ayar_kutusu(st.slider, 'kisi_slider_direct', st.session_state.get('kisi_sayisi_val', 2),
                                   "👤 Nöbet Başına Kişi:", 1, 5, value=st.session_state.get('kisi_sayisi_val', 2))
        st.session_state.kisi_sayisi_val = kişi_sayısı
    
        # Dinlenme ayarı kaldırıldı: nöbetten sonra her zaman 1 gün boş kalır.
        # Eski kayıtlarda farklı değer olsa da 1 kullanılır (kayda da 1 yazılır).
        min_bosluk = 1
        st.session_state.min_bosluk_val = min_bosluk
        # Tatiller her ay için ayrı saklanır ("2026-10": "29"). Hiç girilmemiş
        # bir ayda resmi tatiller kendiliğinden gelir; silinirse boş kalır.
        _holidays_by_period = st.session_state.setdefault("holidays_by_period", {})
        _holiday_period = period_key(yil, ay)
        _official = official_holidays(yil, ay)
        # %25 tatil artırımı yalnız resmi/dini bayramlara; elle eklenenlere değil.
        resmi_tatil_gunleri = frozenset(day for day, _ in _official)
        holiday_input = _ayar_kutusu(
            st.text_input, f"w_holiday_text_{_holiday_period}",
            holiday_text_for_period(_holidays_by_period, yil, ay),
            "🎉 Tatiller (bu ay):",
            value=holiday_text_for_period(_holidays_by_period, yil, ay),
            placeholder="1,2,5-10,23",
        )
        # Yalnızca elle değiştirilen tatiller saklanır; varsayılan resmi
        # tatiller her açılışta güncel listeden gelir.
        if holiday_input == default_holiday_text(yil, ay):
            _holidays_by_period.pop(_holiday_period, None)
        else:
            _holidays_by_period[_holiday_period] = holiday_input
        if settings_expanded and _official:
            st.caption("Resmi tatiller: " + ", ".join(official_holidays_summary(yil, ay)))
        if settings_expanded and not religious_holidays_known(yil):
            st.caption(f"{yil} dini bayram tarihleri tanımlı değil; varsa elle ekleyin.")
        tatil_gunleri, invalid_holiday_parts = parse_holiday_days(holiday_input, gun_sayisi)
        if settings_expanded and invalid_holiday_parts:
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
        unvan = _ayar_kutusu(
            st.selectbox, "w_unvan_val", _unvan_saved if _unvan_saved in _unvan_options else ELLE_UNVAN,
            "🩺 Ünvan:",
            _unvan_options,
            index=_unvan_options.index(_unvan_saved) if _unvan_saved in _unvan_options else len(_unvan_options) - 1,
        )
        st.session_state.unvan_val = unvan
        _rate_info = None
        # Tatil (bayram) nöbetlerinin %25 fazlası ve gece farkı (%10) yalnızca
        # ünvan seçiliyken eklenir; saat ücreti o zaman gündüz ücretidir.
        tatil_artirimi = TATIL_ARTIRIMI if unvan != ELLE_UNVAN else 0.0
        gece_artirimi = GECE_FARKI if unvan != ELLE_UNVAN else 0.0
        if unvan != ELLE_UNVAN:
            riskli = _ayar_kutusu(
                st.checkbox, "w_riskli_birim_val", st.session_state.get("riskli_birim_val", False),
                "Riskli birim (%50 artırımlı)",
                help="Acil, yoğun bakım, ameliyathane, diyaliz, 112 gibi birimler.",
                value=st.session_state.get("riskli_birim_val", False),
            )
            st.session_state.riskli_birim_val = riskli
            _rate_info = unvan_saat_ucreti(unvan, yil, ay, riskli)
            # "|g": ücret artık gündüz ücreti; eski kayıtlardaki ortalama yenilenir.
            _rate_source = f"{unvan}|{_rate_info['donem'][0]}-{_rate_info['donem'][1]}|{int(riskli)}|g"
            if st.session_state.get("nobet_ucreti_kaynak") != _rate_source:
                st.session_state.nobet_ucreti_kaynak = _rate_source
                st.session_state.nobet_ucreti_val = _rate_info["gunduz"]
                if settings_expanded:
                    st.session_state["w_nobet_ucreti_val"] = float(_rate_info["gunduz"])
                else:
                    st.session_state.pop("w_nobet_ucreti_val", None)
        else:
            st.session_state.nobet_ucreti_kaynak = None
        if settings_expanded and "w_nobet_ucreti_val" not in st.session_state:
            st.session_state["w_nobet_ucreti_val"] = float(st.session_state.get("nobet_ucreti_val", 1.0))
        nobet_ucreti = _ayar_kutusu(
            st.number_input, "w_nobet_ucreti_val", float(st.session_state.get("nobet_ucreti_val", 1.0)),
            "💰 Saat Ücreti (TL):",
            # Uzun açıklama "?" içinde; altta yalnızca ücretler görünür. Metin sabit
            # kalır (değişen yardım metni kutunun değerini sıfırlayabilir).
            help=(
                "Yalnız fazla mesai ödenir. Ünvan seçiliyse gece ücreti, ayın gece saati ile "
                "fazla mesai saatinden küçük olanına (her nöbet 12 sa gece), kalan fazla "
                "mesaiye gündüz ücreti uygulanır. Bayram ücreti yalnız resmi ve dini "
                "bayramlara (elle eklenen tatillere değil). Tutarlardan damga vergisi düşülmüştür."
            ),
        )
        st.session_state.nobet_ucreti_val = nobet_ucreti
        # Ünvan tablosundaki ücretler; gündüz ücreti elle değişirse gece ve
        # bayram ücretleri ondan türetilir. Elle girişte her saat aynı ücret.
        if _rate_info and abs(nobet_ucreti - _rate_info["gunduz"]) <= 0.005:
            saat_ucretleri_secili = {tur: _rate_info[tur] for tur in UCRET_TURLERI}
        elif unvan != ELLE_UNVAN:
            saat_ucretleri_secili = saat_ucretleri(nobet_ucreti, tatil_artirimi, gece_artirimi)
        else:
            saat_ucretleri_secili = nobet_ucreti
        if settings_expanded and _rate_info:
            _rate_note = (
                f"Gündüz {tr_sayi(_rate_info['gunduz'])} · gece (20-08) {tr_sayi(_rate_info['gece'])} · "
                f"bayram {tr_sayi(_rate_info['tatil'])} · bayram gece {tr_sayi(_rate_info['tatil_gece'])} TL/sa "
                f"({donem_adi(_rate_info['donem'])}, damga vergisi düşülmüş)."
            )
            if not _rate_info["tam"]:
                _rate_note += " Bu dönemin katsayısı henüz eklenmedi, son bilinen kullanıldı."
            if abs(nobet_ucreti - _rate_info["gunduz"]) > 0.005:
                _rate_note += " Ücret elle değiştirildi."
            st.caption(_rate_note)
    
        role_names_input = _ayar_kutusu(
            st.text_input, "w_role_names_text", st.session_state.get("role_names_text", ""),
            "🏷️ Görev İsimleri:",
            value=st.session_state.get("role_names_text", ""),
            placeholder="Örn: AYB, GYB"
        )
        st.session_state.role_names_text = role_names_input
        
        # Aynı ad iki kez ya da "Tarih" yazılırsa kişiler listeden kayboluyordu.
        role_names, _role_adjusted = normalize_role_names(role_names_input, kişi_sayısı)
        if settings_expanded and _role_adjusted:
            st.caption("Görev adları benzersiz olmalı: " + ", ".join(f"{a} → {b}" for a, b in _role_adjusted))
        st.session_state.rol_isimleri = role_names
        st.session_state.balance_history = _ayar_kutusu(
            st.checkbox, "w_balance_history", st.session_state.get("balance_history", True),
            "⚖️ Önceki aylarla dengele",
            value=st.session_state.get("balance_history", True),
            help=(
                "Son 3 ayda daha çok hafta sonu/tatil ya da toplamda daha çok nöbet tutanlara "
                "bu ay biraz daha az verilir. Bayram ve diğer tatil günleri son 12 ayda "
                "az tutana verilir."
            ),
        )

    if settings_expanded:
        st.divider()
        # Kişi kuralları yan yana: birlikte çalışamayan, birlikte çalışmak isteyen,
        # sabit görev yeri (görev 2 ve üstüyse).
        _yasak_col, _birlikte_col, _sabit_col = st.columns([2, 2, 1.3], gap="large")
        with _yasak_col:
            render_pair_matrix(isimler)
        with _birlikte_col:
            render_pair_matrix(isimler, "birlikte")
        with _sabit_col:
            render_fixed_roles(isimler, role_names)
    elif _settings_closing:
        render_pair_matrix(isimler, draw=False)
        render_pair_matrix(isimler, "birlikte", draw=False)
        render_fixed_roles(isimler, role_names, draw=False)
    forbidden_input = st.session_state.get("forbidden_pairs_text", "")
    limits_text = st.session_state.get("limits_text", "")

    # Yasak çiftler ve kişisel limitler (ayrıştırma nobet_core'da, test edilir)
    forbidden_pairs = parse_forbidden_pairs(forbidden_input, isimler)
    _fp_unknown = unknown_names([name for pair in forbidden_pairs for name in pair], isimler)
    st.session_state.forbidden_pairs = forbidden_pairs
    st.session_state.fp_unknown_names = _fp_unknown
    if settings_expanded and _fp_unknown:
        st.warning(f"⚠️ Yasak çiftte ekip listesinde olmayan isim(ler): **{', '.join(_fp_unknown)}** — kısıtlama uygulanmayacak.")
    # Birlikte nöbet isteyenler; aynı ikili yasaksa yasak geçerlidir.
    st.session_state.together_pairs = {
        _pair for _pair in parse_together_pairs(
            st.session_state.get("together_pairs_text", ""), isimler, forbidden_pairs
        ) if not unknown_names(_pair, isimler)
    }

    person_limits, _lim_invalid_lines, _lim_errors = parse_person_limits_with_issues(limits_text)
    _lim_unknown = unknown_names(person_limits, isimler)
    # "Sınırsız" bırakılan kişi (0-999) limitli sayılmaz; yoksa aylar arası
    # toplam dengesinden ve İyileştir'in toplam dengesinden habersizce çıkıyordu.
    person_limits = {
        _p: _v for _p, _v in person_limits.items()
        if has_total_limit(_v) or _v.get("max_special", 999) < 999
    }
    st.session_state.person_limits = person_limits
    st.session_state.lim_unknown_names = _lim_unknown
    if settings_expanded and _lim_errors:
        st.error("❌ Kişisel limit sorunları:\n" + "\n".join(_lim_errors))
    if settings_expanded and _lim_invalid_lines:
        st.warning(f"⚠️ {len(_lim_invalid_lines)} satır geçersiz format (doğru format: `İsim:min-max`) — atlandı.")
    if settings_expanded and _lim_unknown:
        st.warning(f"⚠️ Kişisel limitte ekip listesinde olmayan isim(ler): **{', '.join(_lim_unknown)}** — limit uygulanmayacak.")
    # Hata metinleri başka kullanıcıların adını içerebilir: yalnız yöneticiye.
    if settings_expanded and _db_ready and (
        not AUTH_ENABLED or current_owner == normalize_owner_email(os.getenv("ADMIN_EMAIL", ""))
    ):
        render_recent_errors()

st.session_state.pop("_ayarlar_kapaniyor", None)

_revert_notice = st.session_state.pop("period_revert_notice", None)
if _revert_notice == "version_unknown":
    st.error(
        "⚠️ Önceki ayın kayıtlı sürümü okunamadığı için (başka bir yerde yapılmış "
        "değişikliklerin üzerine yazmamak için) bu sekmedeki değişiklikler kaydedilmedi ve "
        "ay değiştirilmedi. Önce Excel olarak indirebilirsiniz. Ayı yine de değiştirirseniz "
        "bu sekmedeki değişiklikler atılır; sayfayı yenilemek güncel hâli getirir."
    )
elif _revert_notice == "conflict":
    st.error(
        "⚠️ Önceki ay başka bir yerde (telefon ya da başka sekme) değiştirilmiş; bu sekmedeki "
        "değişiklikler kaydedilmedi ve ay değiştirilmedi. Önce Excel olarak indirebilirsiniz. "
        "Ayı yine de değiştirirseniz bu sekmedeki değişiklikler atılır; sayfayı yenilemek "
        "güncel hâli getirir."
    )
elif _revert_notice:
    st.error(
        "❌ Önceki ayın son değişiklikleri kaydedilemediği için ay değiştirilmedi. "
        "Veritabanı bağlantısını kontrol edin ya da sayfayı yenileyip tekrar deneyin."
    )

# Ekip ve ayarlar değiştiyse veritabanına yaz (sayfa yenilenince geri gelir).
_current_settings = collect_app_settings(st.session_state)
_offline = st.session_state.get("loaded_without_db")
if _offline and _offline.get("settings_skipped") and _offline.get("settings") is None:
    # Yüklenmeden açılan ayarların hâli: veritabanı gelince değişmiş mi diye bakılır.
    _offline["settings"] = _current_settings
_nothing_saved_yet = st.session_state.get("last_saved_settings") is None
if (
    _db_ready
    and not st.session_state.get("app_settings_load_failed", False)
    and _current_settings != st.session_state.get("last_saved_settings")
    # İlk açılışta boş ekip kaydedilmesin (localStorage'dan gelecek ekip ezilmesin).
    and not (_nothing_saved_yet and not st.session_state.get("isimler_text", "").strip())
):
    _settings_version = _version_box("settings_version")
    try:
        _settings_saved = save_app_settings(
            _current_settings, owner=current_owner,
            expected_version=_settings_version.get("version"), version_out=_settings_version,
            writer=_writer,
        )
    except VersionUnknown:
        # Kayıtlı ayarların sürümü bilinmiyor: başka yerdeki değişikliğin üzerine yazma.
        _settings_saved = False
        st.session_state.app_settings_load_failed = True
        st.session_state.settings_version_unknown = True
    except SaveConflict as _conflict:
        # Başka bir yerde (ör. telefonda) ayarlar değişmiş. Öbür yerdeki fark
        # yalnızca bakılan ay ya da bu sekmenin de aynen yaptığı değişiklikse
        # birleştirilip kaydedilir; başka bir ayar değiştiyse onun üzerine yazılmaz.
        _settings_saved = False
        _their_settings = None
        try:
            _their_box = {"version": None}
            _their_settings = load_app_settings(owner=current_owner, raise_on_error=True, version_out=_their_box)
            _their_version = _their_box["version"]
            _merge = merge_app_settings(
                st.session_state.get("last_saved_settings"), _current_settings, _their_settings,
            )
            if _merge is not None and not _merge[1] and _their_version is not None:
                _current_settings = _merge[0]
                _settings_version["version"] = _their_version
                _settings_saved = save_app_settings(
                    _current_settings, owner=current_owner,
                    expected_version=_their_version, version_out=_settings_version,
                    writer=_writer,
                )
        except SaveConflict:
            _settings_saved = False
        except Exception as _merge_error:
            log_unexpected("Ayar birleştirme", _merge_error)
            _settings_saved = False
        if not _settings_saved:
            st.session_state.app_settings_load_failed = True
            st.session_state.settings_conflict = (
                (settings_difference_labels(_current_settings, _their_settings) or True) if _their_settings else True
            )
            st.session_state.settings_conflict_by = conflict_writer_text(
                _conflict.writer, _conflict.saved_at, _writer,
            )
    if _settings_saved:
        st.session_state.last_saved_settings = _current_settings
        # Hazırlanmış yedek artık eski; yeniden hazırlanması gerekir.
        st.session_state.pop("backup_bytes", None)
if st.session_state.get("settings_version_unknown"):
    st.warning(
        "⚠️ Kayıtlı ekip ve ayarların sürümü okunamadı. Başka bir yerde yapılmış "
        "değişikliklerin üzerine yazmamak için ayarlar kaydedilmiyor. Sayfayı yenileyin."
    )


def _accept_corrupt_settings():
    # Okunamayan ayar kaydı boş kabul edilir; sürüm yüklemeden bilindiği için
    # sonraki kayıt (ekip girilince) çakışma denetimiyle yazılır.
    st.session_state.pop("settings_corrupt", None)
    st.session_state.app_settings_load_failed = False


if st.session_state.get("settings_corrupt"):
    _render_corrupt_notice(
        "settings",
        f"⚠️ Kayıtlı ayarlarda okunamayan bir bölüm var: {_corrupt_labels(st.session_state.settings_corrupt)}. "
        "Ekip ve ayarlar boş gösteriliyor; kayıtlı hâlin üzerine yazılmaması için ayarlar "
        "kaydedilmiyor. Bilgisayara indirdiğiniz tam yedek dosyası varsa sayfanın altındaki "
        "«Yedek ve Excel yükleme» › «Dosyadan geri yükle» ile geri yükleyebilirsiniz (uygulama "
        "içindeki aylık yedekler ekip ve ayarları içermez); yoksa aşağıdan okunamayan bölümü boşaltıp ekibi yeniden girebilirsiniz.",
        _corrupt_labels(st.session_state.settings_corrupt),
        _accept_corrupt_settings,
    )
_settings_conflict = st.session_state.get("settings_conflict")
if _settings_conflict:
    _conflict_fields = (
        f" Farklı olan: {', '.join(_settings_conflict)}." if isinstance(_settings_conflict, list) and _settings_conflict else ""
    )
    st.error(
        "⚠️ Ekip ve ayarlar başka bir yerde (telefon ya da başka sekme) değiştirilmiş. "
        f"Yeni değişikliklerin üzerine yazmamak için ayarlar kaydedilmiyor. Sayfayı yenileyin.{_conflict_fields}"
        f"{st.session_state.get('settings_conflict_by', '')}"
    )

# --- YEDEK: tüm aylar ve ayarlar tek dosyada; aynı dosyadan geri yükleme (nobet_yedek.py) ---
# Ekip varken bölüm sayfanın altında, Yedekle tuşunun yanında çizilir.
if not isimler:
    if _db_ready:
        render_backup_section(current_owner, yil, ay, isimler)
    _render_version_footer()
    st.stop()

# --- GÜN VERİLERİ ---
gun_detaylari = build_gun_detaylari(yil, ay, gun_sayisi, tatil_gunleri)
sutunlar = list(gun_detaylari)

zorunlu_saat, arife_gunleri = required_work_hours(gun_detaylari, official_half_days(yil, ay))

# --- SESSION ---
def _save_period(period, team, schedule, preferences, rows, locks, hidden, error_label):
    """Ayın otomatik kaydı; iki kayıt yerinin (ay değişimi ve her çalışma) ortak kısmı.

    ``(sonuç, hata)`` döner. Sonuç: "tamam", "yazilamadi" (veritabanı hatası),
    "kilitli", "cakisma", "surum_bilinmiyor" ya da "hata" (kod hatası, iz günlüğe).
    Çakışma ve bilinmeyen sürümde bu ayın kaydı durdurulur; kilitli ayda karar çağıranda.
    """
    _year, _month = period
    _box = _version_box("schedule_version")
    try:
        _ok = save_schedule(
            otomatik_kayit_adi(_year, _month), _year, _month, team, schedule,
            pref_df=preferences, rows_liste=rows, owner=current_owner,
            settings_dict={"locked": locks_to_json(locks), "hidden": hidden},
            expected_version=_box.get("version"), version_out=_box, writer=_writer,
        )
    except MonthLocked as _error:
        return "kilitli", _error
    except SaveConflict as _error:
        kaydi_durdur(st.session_state, period, "cakisma")
        return "cakisma", _error
    except VersionUnknown as _error:
        kaydi_durdur(st.session_state, period, "surum_bilinmiyor")
        return "surum_bilinmiyor", _error
    except Exception as _error:
        log_unexpected(error_label, _error)
        return "hata", _error
    return ("tamam" if _ok else "yazilamadi"), None


def _save_outgoing_period(period, schedule, preferences, rows):
    _old_yil, _old_ay = period
    # Okunamayan bir dönemin kaydının üzerine yazma; değişmediyse tekrar yazma.
    if kayit_durdu_mu(st.session_state, period):
        # Başka yerde değişmiş ay: bu sekmedeki değişiklikler atılacaksa önce bir kez uyar.
        if (
            st.session_state.get("autosave_conflict_period") == tuple(period)
            and st.session_state.get("conflict_leave_ack") != tuple(period)
        ):
            st.session_state.conflict_leave_ack = tuple(period)
            st.session_state.period_revert_reason = st.session_state.get("autosave_conflict_kind") or "conflict"
            return False
        return True
    _locks = active_locks(st.session_state.get("locked_cells"), schedule)
    _hidden = st.session_state.get("hidden_people") or {}
    if persistence_fingerprint(schedule, preferences, rows, _locks, _hidden) == st.session_state.get("last_saved_fingerprint"):
        return True
    st.session_state.pop("backup_bytes", None)
    # False dönerse ay değişimi durdurulur: son değişiklikler kaybolmasın.
    _result, _ = _save_period(period, list(schedule.index), schedule, preferences, rows, _locks, _hidden,
                              "Ay değişiminde kayıt")
    if _result in ("cakisma", "surum_bilinmiyor"):
        # Çakışma ya da bilinmeyen sürüm: bir kez uyarılır, ikinci denemede
        # bu sekmedeki değişiklikler atılarak ay değişir.
        st.session_state.conflict_leave_ack = tuple(period)
        st.session_state.period_revert_reason = "conflict" if _result == "cakisma" else "version_unknown"
    # Kilitli ayda değişiklik yapılamaz; kaydedilecek bir şey yok, ay değişebilir.
    # Kod ya da veritabanı hatasında ay değişmez (son değişiklikler kaybolmasın).
    return _result in ("tamam", "kilitli")

# Ay/yıl değişince eski ayın çizelgesini yeni aya taşımak yerine kaydet ve
# yeni ayın kendi kaydını yükle (aksi halde otomatik kayıt o ayın üzerine yazar).
switch_schedule_period(st.session_state, (yil, ay), _save_outgoing_period)
if st.session_state.get("period_switch_blocked"):
    # Çıkılan ayın son değişiklikleri kaydedilemedi: ay değişmesin, uyarı gösterilsin.
    st.session_state.period_revert = st.session_state.pop("period_switch_blocked")
    st.rerun()

if 'schedule_bool' not in st.session_state:
    # Sayfa yenilenmesinde otomatik geri yükle
    _auto_name = otomatik_kayit_adi(yil, ay)

    # Sürüm, çizelgeyle aynı okumadan gelir; okuma başarısızsa None kalır ve
    # kayıt durdurulur (VersionUnknown). Eskiden sürüm ayrı okunuyordu: o okuma
    # hata verip çizelge yine yüklenince çakışma denetimi kapanıyordu.
    st.session_state.schedule_version = {"version": None}
    # Kaydın okunamayan bölümleri (ör. bozuk tercih JSON'u): sağlam kısımlar yüklenir,
    # bozuk bölüm boş gösterilir ama üzerine yazılmaz (bkz. aşağıdaki uyarı).
    st.session_state.pop("schedule_corrupt", None)
    _corrupt_fields = []

    def _load_auto_schedule(name, year, month):
        # Keep "no saved schedule yet" distinct from a database failure so the
        # first visit does not show a misleading error.
        return load_schedule(
            name, year, month, raise_on_error=True, owner=current_owner,
            version_out=st.session_state.schedule_version, corrupt_out=_corrupt_fields,
        )

    def _on_auto_load_error(_error):
        # Kayıt okunamadıysa boş çizelgeyi kaydedip gerçek verinin üzerine yazma.
        kaydi_durdur(st.session_state, (yil, ay), "okuma")
        update_persistence_feedback(
            st.session_state,
            False,
            "⚠️ Kayıtlı çizelge okunamadı. Veriyi korumak için otomatik kayıt "
            "durduruldu; veritabanı bağlantısını kontrol edip sayfayı yenileyin.",
        )

    engelleri_sifirla(st.session_state)
    # Önceki ayın kayıt uyarısı ("bu ayın ...") yeni ayda gösterilmesin.
    st.session_state.persistence_error = None
    if not _db_ready:
        # Veritabanına ulaşılamıyor: okuma denenmez (yeniden zaman aşımı beklenmesin)
        # ve boş çizelge kayıtlı olanın üzerine yazılmasın. Veritabanı gelince yüklenir.
        kaydi_durdur(st.session_state, (yil, ay), "cevrimdisi")
        st.session_state.loaded_without_db = {
            **(st.session_state.get("loaded_without_db") or {}), "period": (yil, ay), "schedule_clean": True,
        }
        st.session_state.schedule_bool = pd.DataFrame(False, index=isimler, columns=sutunlar)
    elif not restore_auto_saved_schedule(
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
    if _corrupt_fields:
        # Okunamayan bölüm boş sayılıp ilk değişiklikte kalıcı olarak silinmesin.
        kaydi_durdur(st.session_state, (yil, ay), "bozuk")
        st.session_state.schedule_corrupt = {"period": (yil, ay), "fields": list(_corrupt_fields)}
    # Yüklenen durum zaten veritabanındakiyle aynı; ilk kayıt ilk değişiklikte yapılır.
    st.session_state.autosave_baseline_pending = True
    # Tercih Belirle'de kaydedilmeden kalan tercihler (sayfa yenilendi ya da
    # kapandı) tarayıcıda durur; ay o zamandan beri başka yerde kaydedilmediyse
    # geri getirilip hemen kaydedilir.
    _pending_raw = local_storage.getItem(PENDING_PREFS_KEY)
    if _pending_raw and not kayit_durdu_mu(st.session_state, (yil, ay)):
        _base_pref = st.session_state.get("pref_df")
        if _base_pref is None:
            _base_pref = pd.DataFrame(0, index=isimler, columns=sutunlar)
        _restored_pref = apply_pending_prefs(
            _pending_raw, current_owner, yil, ay,
            st.session_state.schedule_version.get("version"), _base_pref,
        )
        if _restored_pref is not None:
            st.session_state.pref_df = _restored_pref
            st.session_state.autosave_baseline_pending = False
            st.session_state.save_preferences_now = True
            st.toast("📝 Kaydedilmemiş tercihler geri getirildi.")
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


def _accept_corrupt_schedule():
    # Okunamayan bölüm boş kabul edilir; kayıt yeniden açılır ve bu çalışmada
    # yüklemedeki sürümle (başka yerde değiştiyse çakışma uyarısı) temiz yazılır.
    _state = st.session_state.pop("schedule_corrupt", None) or {}
    if _state.get("period"):
        engeli_kaldir(st.session_state, _state["period"], "bozuk")
    # İki parmak izi de sıfırlanır: Tercih modunda da kayıt ertelenmez.
    st.session_state.last_saved_fingerprint = None
    st.session_state.last_saved_core_fingerprint = None
    st.session_state.persistence_error = None


_schedule_corrupt = st.session_state.get("schedule_corrupt")
if _schedule_corrupt and _schedule_corrupt.get("period") == (yil, ay):
    _render_corrupt_notice(
        "schedule",
        f"⚠️ Bu ayın kaydında okunamayan bir bölüm var: {_corrupt_labels(_schedule_corrupt['fields'])}. "
        "Nöbetler gösteriliyor. Okunamayan bölümün üzerine yazılmaması için bu ayın otomatik "
        "kaydı durduruldu; bu sırada yapılan değişiklikler kaydedilmez. "
        f"{_BACKUP_HINT}; yoksa aşağıdan okunamayan bölümü boşaltıp devam edebilirsiniz.",
        _corrupt_labels(_schedule_corrupt["fields"]),
        _accept_corrupt_schedule,
    )

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
if 'together_pairs_text' not in st.session_state: st.session_state.together_pairs_text = ""
if 'together_pairs' not in st.session_state: st.session_state.together_pairs = set()
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
    st.toast("En adil sonuç seçildi. " + simulation_summary(
        st.session_state.schedule_bool, isimler, sutunlar, gun_detaylari,
        st.session_state.get("person_limits", {}),
    ), icon="🧠")
    if st.session_state.pop("auto_improved_pending", False):
        st.toast("Eksik ya da toplam farkı vardı; ✨ İyileştir çözücüsü düzeltti.", icon="✨")


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
    except Exception as _error:
        log_unexpected("Önceki aylar bağlamı", _error)
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


def _load_next_first_shift(year, month):
    """Sonraki ayın kayıtlı listesinde kişi başına ilk nöbet günü; kayıt yoksa ya da okunamazsa boş."""
    if not _db_ready:
        return {}
    _next = previous_period(year, month, -1)
    try:
        _loaded, _ = load_auto_saved_assignments([_next], owner=current_owner, include_preferences=True)
    except Exception as _error:
        log_unexpected("Sonraki ay ilk nöbetler", _error)
        return {}
    _frame = _loaded.get(_next)
    if _frame is None or _frame.empty:
        return {}
    _frame = as_bool_frame(_frame)
    _aliases = st.session_state.get("name_aliases")
    _frame = _frame.rename(index={_p: resolve_alias(_p, _aliases, isimler) for _p in _frame.index})
    if _frame.index.has_duplicates:
        _frame = _frame.groupby(level=0).any()
    return ilk_nobet_gunleri(_frame)


# Dönem değişince bir kez okunur (ay sonu dinlenme kontrolü için).
if st.session_state.get("next_month_first_shift_period") != (yil, ay):
    st.session_state.next_month_first_shift = _load_next_first_shift(yil, ay)
    st.session_state.next_month_first_shift_period = (yil, ay)
_balance_history_on = st.session_state.get("balance_history", True)
# Hafta sonu sınırı olanlar hafta sonu dengesine (geçmiş dahil) katılmaz.
_special_capped = special_capped_people(st.session_state.get("person_limits", {}))
history_special = (
    history_special_adjustment(
        _prev_ctx["history_counts"], [_p for _p in isimler if _p not in _special_capped]
    )
    if _balance_history_on and _prev_ctx["months"]
    else {}
)
# Kişisel limiti olanların az/çok nöbeti bilinçli; toplam dengesine katılmaz.
_limited_people = {
    _p for _p, _v in st.session_state.get("person_limits", {}).items() if has_total_limit(_v)
}
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
        if is_blocked_preference(_pref_v):
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

# Kişisel zorunlu mesai: İzin işaretli iş günleri düşülür (Müsait Değil düşmez).
mesai_saatleri = personal_work_hours(
    gun_detaylari, official_half_days(yil, ay), st.session_state.pref_df, isimler
)

# Paylaşım linkinden gelen tercih/izin talepleri onay bekler (nobet_talepler.py).
if _db_ready:
    render_team_requests(current_owner, yil, ay, isimler, save_undo_state)
    render_request_status(current_owner, yil, ay, isimler)

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
                        today_date, ui, person_colors, kişi_sayısı, mesai_saatleri, saat_ucretleri_secili,
                        resmi_tatil_gunleri)

# Build algorithm inputs from pref_df
df_unwanted = pd.DataFrame(False, index=isimler, columns=sutunlar)
df_preferred = pd.DataFrame(0, index=isimler, columns=sutunlar)

for person in isimler:
    for col in sutunlar:
        status = st.session_state.pref_df.at[person, col] if person in st.session_state.pref_df.index and col in st.session_state.pref_df.columns else 0
        if is_blocked_preference(status):
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
        tuple(sorted(st.session_state.get("together_pairs", set()))),
        tuple(sorted((p, tuple(sorted(limits.items())))
                     for p, limits in st.session_state.get('person_limits', {}).items())),
        tuple(sorted(active_locks(st.session_state.get("locked_cells"), st.session_state.schedule_bool))),
        tuple(sorted(prev_last_shift.items())),
        tuple(sorted(history_special.items())),
        tuple(sorted(history_total.items())),
        tuple(sorted(history_holiday.items())),
    )


# Elle İyileştir 5 sn; kendiliğinden çalışanı her simülasyonda beklenmesin diye kısa.
AUTO_IMPROVE_SECONDS = 2.5


def _auto_improve(schedule):
    """Boş nöbet, fark ya da karşılanmayan tercih varsa İyileştir'in çözücüsünü
    (kısa süreli) bir kez dener: (çizelge, düzeldi_mi)."""
    try:
        from nobet_iyilestir import improve_if_needed
    except ImportError:
        return schedule, False
    return improve_if_needed(
        schedule, isimler, sutunlar, df_unwanted, gun_detaylari, min_bosluk, kişi_sayısı,
        forbidden_pairs=st.session_state.forbidden_pairs,
        person_limits=st.session_state.get('person_limits', {}), df_preferred=df_preferred,
        fixed_assignments=active_locks(st.session_state.get("locked_cells"), st.session_state.schedule_bool),
        initial_last_shift=prev_last_shift, history_special=history_special,
        history_total=history_total, history_holiday=history_holiday,
        together_pairs=st.session_state.get("together_pairs", set()),
        time_limit=AUTO_IMPROVE_SECONDS,
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
            together_pairs=st.session_state.get("together_pairs", set()),
        )
        _auto_improved = False
        if _new_schedule is not None:
            with st.spinner("Eksik ya da toplam farkı için çözücü deneniyor..."):
                _new_schedule, _auto_improved = _auto_improve(_new_schedule)
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
                st.session_state.switch_to_assign_mode = True
                st.session_state.auto_improved_pending = _auto_improved
                st.rerun()

# İyileştir: mevcut listeden OR-Tools ile daha adilini arar (nobet_iyilestir).
if st.session_state.pop("run_improve", False):
    st.session_state.incomplete_simulation = None
    try:
        from nobet_iyilestir import improve_schedule, improvement_lines, rule_breaking_shifts
    except ImportError as _exc:
        improve_schedule = None
        _import_error = _exc
    if improve_schedule is None:
        if importlib.util.find_spec("ortools") is None:
            st.error("İyileştir için gereken çözücü (ortools) yüklü değil.")
        else:
            # Çözücü yüklü; sorun uygulamanın kendi dosyalarında (ör. yarım güncelleme).
            st.error(f"İyileştir açılamadı: {_import_error}. Sayfayı yenileyip tekrar deneyin.")
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
                    together_pairs=st.session_state.get("together_pairs", set()),
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

render_precheck(df_unwanted, sutunlar, kişi_sayısı, min_bosluk, st.session_state.get('person_limits', {}))

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
                st.session_state.switch_to_assign_mode = True
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
            conflict_msg.append(f"❌ **{k}**: {gun_no}. gün izinli ya da müsait değil.")
        
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

violations += ay_sonu_cakismalari(
    {p: d for p, d in last_shift_check.items() if p in st.session_state.next_month_first_shift},
    st.session_state.next_month_first_shift, gun_sayisi, min_bosluk,
)

# Eksik günler tek satırda toplanır; kuralların engellediği günlerin nedeni altta küçük yazılır.
_schedule_empty = not bool(edited_df.values.any())
_missing_days = [] if _schedule_empty else missing_shift_coverage(edited_df, sutunlar, kişi_sayısı)
if _missing_days:
    min_person_msg.append(
        f"🟠 **Eksik nöbetçi** ({len(_missing_days)} gün): "
        + ", ".join(f"{gun_detaylari[col]['day_num']} ({assigned}/{kişi_sayısı})" for col, assigned in _missing_days)
    )
missing_reason_msg = []
for col, assigned in _missing_days:
    _reasons = _missing_day_reasons(edited_df, col)
    # Herkes uygunsa neden satırı gürültü olur (elle atanabilir).
    if not _reasons or any(_label != MISSING_DAY_FREE_LABEL for _label, _ in _reasons):
        missing_reason_msg.append(
            f"**{gun_detaylari[col]['day_num']}. gün** — {format_missing_day_reason(_reasons)}"
        )

# Kişisel limit ihlali kontrolü
_pl = st.session_state.get('person_limits', {})
limit_msg = build_limit_violation_messages(
    edited_df, isimler, _pl,
    [c for c in sutunlar if gun_detaylari[c]["weekend"] or gun_detaylari[c]["holiday"]],
)

# Giriş doğrulama uyarıları (yasak çift / limit — bilinmeyen isimler)
_fp_unk = st.session_state.get('fp_unknown_names', [])
_lim_unk = st.session_state.get('lim_unknown_names', [])
input_warn_msg = []
if _fp_unk:
    input_warn_msg.append(f"⚠️ Yasak çiftte ekip dışı isim(ler): **{', '.join(_fp_unk)}** — bu kısıtlama etkisiz.")
if _lim_unk:
    input_warn_msg.append(f"⚠️ Kişisel limitte ekip dışı isim(ler): **{', '.join(_lim_unk)}** — bu limit etkisiz.")
_fixed_roles = active_fixed_roles(st.session_state.get("fixed_roles"), isimler, role_names)
for _col, _role, _people in fixed_role_conflicts(edited_df, sutunlar, _fixed_roles):
    input_warn_msg.append(
        f"📌 {gun_detaylari[_col]['day_num']}. gün: **{', '.join(_people)}** ikisi de {_role} görevine sabit; "
        f"yalnız biri {_role}'de. Olmasın isterseniz Birlikte Çalışamayan'da işaretleyin."
    )

_issue_count = (
    len(max_person_msg) + len(_missing_days) + len(conflict_msg) + len(forbidden_msg)
    + len(violations) + len(limit_msg) + len(input_warn_msg)
)
if _schedule_empty:
    # Henüz hiç nöbet yokken her gün için "eksik" uyarısı yerine ne yapılacağını söyle.
    st.info("Liste boş. İzin ve tercihleri işaretleyin, sonra ⚡ Simülasyon'a basın.")
    for m in input_warn_msg: st.warning(m)
elif _issue_count:
    with st.expander(f"⚠️ Kontrol: {_issue_count} sorun", expanded=True):
        for m in max_person_msg: st.error(m)
        for m in min_person_msg: st.warning(m)
        for m in missing_reason_msg: st.caption(m)
        for c in conflict_msg: st.error(c)
        for f in forbidden_msg: st.error(f)
        for v in violations: st.info(v)
        for m in limit_msg: st.warning(m)
        for m in input_warn_msg: st.warning(m)
else:
    st.caption(f"✅ Kurallar uygun: her gün {kişi_sayısı} kişi, çakışma yok.")

# --- VERİ HAZIRLIĞI ---
# Only regenerate assignments when AI button is clicked, not on manual edits.
# Rol isimleri/sırası da satır yapısının parçasıdır.
_cache_key = (yil, ay, tuple(isimler), tuple(sutunlar), tuple(role_names))
if (st.session_state.should_regenerate_assignments
        or st.session_state.cached_rows_liste is None
        or st.session_state.get('cached_key') != _cache_key):
    # Sabit görev yeri olan kişi kendi görevine yerleşir (Ayarlar > Görev İsimleri).
    rows_liste, first_role_counts = distribute_roles(
        [gun_detaylari[col]['full_date'] for col in sutunlar],
        [edited_df.index[edited_df[col]].tolist() for col in sutunlar],
        role_names,
        _fixed_roles,
    )
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
            _fixed_roles,
        )
        st.session_state.cached_rows_liste = rows_liste
        st.session_state.cached_role_schedule = edited_df.copy(deep=True)
    if st.session_state.pop("fixed_roles_changed", False):
        # Sabit görev yeri değişince bu ayın satırları yeni kurala göre düzenlenir;
        # açılan eski aylar ise olduğu gibi kalır.
        rows_liste = [apply_fixed_roles(row, role_names, _fixed_roles) for row in rows_liste]
        st.session_state.cached_rows_liste = rows_liste

# Display and allow manual editing of ROLES (swapping people)
# REMOVED: Manual list editing as requested. Using autonomous redistribution instead.

# Re-distribution logic moved to Analiz section as requested.

# Görev sayısından fazla atanan kişiler listelerde "Ek 1", "Ek 2" sütununda görünür.
liste_rows, liste_roles = add_overflow_roles(rows_liste, role_names, edited_df, sutunlar)
df_liste = pd.DataFrame(liste_rows)

# Auto-save only after role rows have been generated or reconciled with the
# current schedule, so both database payloads describe the same snapshot.
# Çizelge değişiklikleri hemen kaydedilir (eskiden 30 sn bekleniyordu ve son
# değişiklikler sayfa yenilenince kayboluyordu). "Tercih Belirle" modunda
# yalnızca tercihler değiştiyse her dokunuşta yazılmaz: mod değişince, çizelge
# değişince (Simülasyon vb.), ay değişince, Yedekle ya da Kaydet'e basınca
# hepsi birlikte kaydedilir.
_active_locks = active_locks(st.session_state.get("locked_cells"), st.session_state.schedule_bool)
_hidden_people = st.session_state.get("hidden_people") or {}
_persist_fp = persistence_fingerprint(
    st.session_state.schedule_bool, st.session_state.get('pref_df'), rows_liste, _active_locks,
    _hidden_people,
)
_core_fp = persistence_fingerprint(
    st.session_state.schedule_bool, None, rows_liste, _active_locks, _hidden_people,
)
_save_now = st.session_state.pop("save_preferences_now", False)
_defer_save = (
    st.session_state.get("edit_mode") == "tercih"
    and not _save_now
    and not st.session_state.get("snapshot_requested")
    and _core_fp == st.session_state.get("last_saved_core_fingerprint")
)
if st.session_state.pop("autosave_baseline_pending", False):
    st.session_state.last_saved_fingerprint = _persist_fp
    st.session_state.last_saved_core_fingerprint = _core_fp
elif (
    not kayit_durdu_mu(st.session_state, (yil, ay))
    and _persist_fp != st.session_state.get("last_saved_fingerprint")
    and _defer_save
):
    st.caption("📝 Tercih değişiklikleri henüz kaydedilmedi; mod değişince ya da Simülasyon'da kaydedilir.")
    st.button("💾 Tercihleri Kaydet", key="save_preferences_button",
              on_click=lambda: st.session_state.update(save_preferences_now=True))
    # Sayfa kaydetmeden yenilenir ya da kapanırsa kaybolmasın: tarayıcıda da tut.
    _pending_payload = pending_prefs_payload(
        current_owner, yil, ay, _version_box("schedule_version").get("version"), st.session_state.pref_df,
    )
    if local_storage.getItem(PENDING_PREFS_KEY) != _pending_payload:
        local_storage.setItem(PENDING_PREFS_KEY, _pending_payload, key="pending_prefs_set")
elif (
    not kayit_durdu_mu(st.session_state, (yil, ay))
    and _persist_fp != st.session_state.get("last_saved_fingerprint")
):
    _result, _error = _save_period((yil, ay), isimler, st.session_state.schedule_bool,
                                   st.session_state.get('pref_df'), rows_liste, _active_locks,
                                   _hidden_people, "Otomatik kayıt")
    _saved_ok = _result == "tamam"
    _save_conflict = _result not in ("tamam", "yazilamadi")
    if _result == "cakisma":
        # Bu ay başka bir yerde (ör. telefonda) değişmiş; onların üzerine yazma.
        update_persistence_feedback(
            st.session_state,
            False,
            "⚠️ Bu ayın çizelgesi başka bir yerde (telefon ya da başka sekme) değiştirilmiş. "
            "Yeni değişikliklerin üzerine yazmamak için kayıt durduruldu. Sayfayı yenileyin; "
            "bu sekmedeki değişiklikleri kaybetmek istemiyorsanız önce Excel olarak indirin."
            + conflict_writer_text(_error.writer, _error.saved_at, _writer),
        )
    elif _result == "kilitli":
        kaydi_durdur(st.session_state, (yil, ay), "kilitli")
        update_persistence_feedback(
            st.session_state,
            False,
            "🔒 Bu ay kilitli; değişiklikler kaydedilmedi. Ayın üstündeki 🔓 düğmesiyle kilidi açın.",
        )
    elif _result == "surum_bilinmiyor":
        # Çakışmadaki gibi: kayıt durur, ay değiştirilirse bir kez uyarılıp
        # ikinci denemede bu sekmedeki değişiklikler atılarak geçilir.
        update_persistence_feedback(
            st.session_state,
            False,
            "⚠️ Bu ayın kayıtlı sürümü okunamadı. Başka bir yerde yapılmış değişikliklerin "
            "üzerine yazmamak için kayıt yapılmadı. Sayfayı yenileyin; bu sekmedeki "
            "değişiklikleri kaybetmek istemiyorsanız önce Excel olarak indirin.",
        )
    elif _result == "hata":
        # Veritabanı değil, kod hatası: "bağlantı" diye gösterme, iz günlükte.
        update_persistence_feedback(
            st.session_state,
            False,
            f"❌ Çizelge kaydedilemedi: beklenmeyen bir program hatası ({type(_error).__name__}). "
            "Veritabanı bağlantısında sorun yok; sayfayı yenilemeden önce Excel/PNG olarak indirin.",
        )
    if _save_conflict:
        pass
    elif _saved_ok:
        st.session_state.last_saved_fingerprint = _persist_fp
        st.session_state.last_saved_core_fingerprint = _core_fp
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

_offline = st.session_state.get("loaded_without_db")
if _offline and _offline.get("period") == (yil, ay):
    # Veritabanı yokken açılan ay: değiştirilmediyse veritabanı gelince yüklenir.
    _offline["schedule_clean"] = _persist_fp == st.session_state.get("last_saved_fingerprint")

# Onaylanan ekip talepleri ancak ay kaydedildiyse silinir.
if _db_ready:
    finish_approved_requests(
        current_owner, yil, ay, _persist_fp == st.session_state.get("last_saved_fingerprint"),
    )

# Tercihler kaydedildiyse tarayıcıdaki bekleyen kopya artık gereksiz.
if (
    is_pending_prefs_for(local_storage.getItem(PENDING_PREFS_KEY), current_owner, yil, ay)
    and _persist_fp == st.session_state.get("last_saved_fingerprint")
):
    local_storage.deleteItem(PENDING_PREFS_KEY, key="pending_prefs_delete")

if st.session_state.get("persistence_error"):
    st.error(st.session_state.persistence_error)

# Yedekle tuşu: son değişiklik kaydedildikten sonra yedek alınır.
run_requested_snapshot(current_owner, yil, ay)
# Günlük otomatik kopya: veritabanının ücretsiz planında geri dönüş yalnızca
# son birkaç saate yetiyor; günün ilk açılışında tüm veri saklanır
# (testlerdeki sahte veritabanında atlanır).
if _db_ready and is_database_configured():
    take_daily_snapshot(current_owner)

_rows_key = tuple(tuple(row.get(role, "-") for role in role_names) for row in rows_liste)
_analysis_key = (
    schedule_fingerprint(edited_df),
    _rows_key,
    tuple(role_names),
    tuple(sorted((person, tuple(sorted(limits.items())))
                 for person, limits in st.session_state.get("person_limits", {}).items())),
    zorunlu_saat,
    tuple(sorted(mesai_saatleri.items())),
    tuple(sorted(saat_ucretleri_secili.items())) if isinstance(saat_ucretleri_secili, dict) else saat_ucretleri_secili,
    tuple(sorted(resmi_tatil_gunleri)),
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
        saat_ucretleri_secili,
        mesai_saatleri,
        resmi_tatil_gunleri,
    )
    st.session_state.cached_analysis_key = _analysis_key

df_stats_load, df_stats_finance, pair_display = st.session_state.cached_analysis_tables

# --- GÖRÜNÜM ---
st.divider()

# ── İndirme/paylaşım/yükleme/yedek düğmeleri ince yatay satır (nobet_disa_aktar.py) ──
_share_check = paylasim_kontrolleri(
    pending_count(current_owner) if _db_ready else 0,
    [gun_detaylari[col]["day_num"] for col, _ in _missing_days],
    katsayi_uyarisi(yil, ay) if st.session_state.get("unvan_val", ELLE_UNVAN) != ELLE_UNVAN else None,
)
render_toolbar(df_liste, df_stats_load, df_stats_finance, liste_rows, liste_roles,
               yil, ay, gun_detaylari, isimler, sutunlar, role_names,
               _db_ready, current_owner, save_undo_state, share_check=_share_check)
render_backup_section(
    current_owner, yil, ay, isimler, _db_ready,
    excel_import=lambda: render_excel_import(
        yil, ay, gun_detaylari, isimler, sutunlar, role_names, save_undo_state,
    ),
)

st.divider()

# Görev yeri değişimi takvim görünümündeki 🔄 ile yapılır; altta doğrudan Analiz.
st.subheader("📊 Analiz")
render_analysis_tab(isimler, role_names, _prev_ctx, history_special, history_total, history_holiday,
                    df_stats_load, df_stats_finance, pair_display, df_liste, liste_rows, liste_roles,
                    yil, ay, zorunlu_saat, arife_gunleri, edited_df)
render_why_not(isimler, sutunlar, gun_detaylari, _missing_day_reasons, edited_df)


def _assign_backup(out, backup, column):
    """Çıkamayan kişiyi o günden alıp yedeği yazar (tabloda iki hücreye dokunmakla aynı)."""
    save_undo_state()
    st.session_state.schedule_bool.at[out, column] = False
    st.session_state.schedule_bool.at[backup, column] = True
    st.session_state.locked_cells = update_locked_cells(
        st.session_state.get("locked_cells"), [(backup, column)], [(out, column)])
    _queue_assignment_warnings([(backup, column)])


render_backup_suggestion(
    isimler, sutunlar, gun_detaylari, edited_df, df_unwanted, min_bosluk,
    st.session_state.get("person_limits", {}), st.session_state.get("forbidden_pairs", []),
    prev_last_shift, _assign_backup)
if _db_ready:
    render_year_summary(current_owner, yil, isimler, st.session_state.get("holidays_by_period", {}))

_render_version_footer()
