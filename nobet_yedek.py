# -*- coding: utf-8 -*-
"""Yedek: tüm aylar ve ayarlar tek dosyada; aynı dosyadan geri yükleme.

Uygulamadaki yedekler her ay için tektir; geri yüklemeden önce mevcut hâlin
kopyası alınır.
"""
import streamlit as st

from db import (
    export_owner_data,
    list_backup_snapshots,
    load_backup_snapshot,
    restore_owner_data,
    save_backup_snapshot,
)
from nobet_core import (
    BACKUP_MAX_BYTES,
    SNAPSHOT_BEFORE_RESTORE,
    SNAPSHOT_MANUAL,
    TR_AY_ADLARI,
    backup_reminder,
    backup_summary,
    backup_to_bytes,
    build_backup,
    month_backup,
    now_tr,
    parse_backup,
    snapshot_label,
    snapshot_time_text,
)


def mark_backup_downloaded():
    st.session_state.last_backup_at = now_tr().strftime("%Y-%m-%d %H:%M")
    st.session_state.pop("backup_bytes", None)


def take_snapshot(reason, current_owner, period=None):
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


def run_requested_snapshot(current_owner, yil, ay):
    """Yedekle tuşu: otomatik kayıt bittikten sonra çağrılır."""
    if not st.session_state.pop("snapshot_requested", False):
        return
    if st.session_state.get("persistence_error"):
        st.error("❌ Son değişiklik kaydedilemediği için yedek alınmadı. Tekrar deneyin.")
        return
    _period_sn = (yil, ay)
    try:
        _summary_sn = take_snapshot(SNAPSHOT_MANUAL, current_owner, _period_sn)
    except Exception:
        st.error("❌ Yedek alınamadı (veritabanına ulaşılamadı). Tekrar deneyin.")
        return
    _month_text = f"{TR_AY_ADLARI[ay]} {yil}"
    st.session_state.snapshot_notice = (
        f"{_month_text} yedeklendi; bu ayın önceki yedeğinin yerine geçti."
        if _summary_sn else f"{_month_text} için henüz kaydedilmiş çizelge yok."
    )
    st.rerun()


def restore_backup(parsed, summary, current_owner):
    """Geri yüklemeden önce mevcut hâlin kopyasını alır, sonra yedeği yükler."""
    try:
        take_snapshot(SNAPSHOT_BEFORE_RESTORE, current_owner)
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


def render_backup_section(current_owner, yil, ay, isimler, db_ready=True, excel_import=None):
    """Yedek bölümü; ``excel_import`` verilirse Excel'den yükleme de burada çizilir.

    Veritabanı yoksa yalnızca Excel'den yükleme gösterilir.
    """
    if not db_ready:
        if excel_import is not None:
            with st.expander("💾 Yedek ve Excel yükleme", expanded=False):
                st.markdown("**📂 Excel'den yükle**")
                excel_import()
        return
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
            # Bölüm Yedekle tuşunun hemen altında; ayrıca bildirim gerekmez.
            st.success(f"✅ {_snapshot_notice}")
    _backup_hint = backup_reminder(st.session_state.get("last_backup_at"))
    _backup_label = "💾 Yedek ve Excel yükleme" + (
        f" — {_backup_hint} Bilgisayara indirmeniz önerilir." if _backup_hint else ""
    )
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
                run_requested_snapshot(current_owner, yil, ay)
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
                            restore_backup(_parsed_snap, backup_summary(_parsed_snap), current_owner)
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
                    on_click=mark_backup_downloaded, width="stretch",
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
                        restore_backup(_parsed_backup, _summary, current_owner)

        if excel_import is not None:
            st.divider()
            st.markdown("**📂 Excel'den yükle**")
            excel_import()
