# -*- coding: utf-8 -*-
"""Takvim Görünümü sekmesi: aylık takvim ve tek kişi seçilince kişi kartı."""
from datetime import date

import streamlit as st

from nobet_core import TR_GUN_KISA, escape_html, find_person_role, nobet_ucreti_hesapla, tr_sayi
from nobet_durum import rotate_roles_for_day


def render_calendar_tab(isimler, sutunlar, gun_detaylari, yil, ay, gun_sayisi, tatil_gunleri,
                        today_date, ui, person_colors, kişi_sayısı, mesai_saatleri, nobet_ucreti, resmi_tatil_gunleri=None):
    tr_gunler = TR_GUN_KISA
    # Rol isimleri (Görev1, Görev2 vb.)
    _cal_role_names = st.session_state.get('rol_isimleri', [f"Görev{i+1}" for i in range(kişi_sayısı)])

    # ── Takvimden gelen görev yeri değişim isteği ────────
    _swap_dn = st.session_state.pop('cal_swap_day', None)
    if _swap_dn is not None and rotate_roles_for_day(_swap_dn):
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
            "🟢 Tercih &nbsp; 🟡 Kaçın &nbsp; 🔴 İzin &nbsp; 🟣 Müsait Değil &nbsp; 🔄 Görev Yeri Değiştir"
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
        dp_total_h, dp_fm_h, dp_pay = nobet_ucreti_hesapla(
            len(dp_shifts),
            sum(1 for c in dp_shifts if gun_detaylari[c]['holiday'] and (
                resmi_tatil_gunleri is None or gun_detaylari[c]['day_num'] in resmi_tatil_gunleri)),
            mesai_saatleri.get(dp, 0), nobet_ucreti,
        )
        _dp_rows = st.session_state.get('cached_rows_liste') or []
        _dp_role_names = st.session_state.get('cached_role_names', _cal_role_names)
        # Tercih istatistikleri (sadece nöbet tutulan günler)
        dp_pref_on_shift = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        dp_pref_all = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
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
        dp_purple_hit = dp_pref_on_shift[4]
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
            f"<div style='font-size:22px;font-weight:800;color:{ui['stat_purple']};'>{tr_sayi(dp_pay, 0)} ₺</div>"
            f"<div style='font-size:11px;color:{ui['label_text']};'>Tahmini Ödeme</div></div>"
            f"</div>"
            f"<div style='margin-top:10px;font-size:11px;color:{ui['label_text']};'>"
            f"🟢 {dp_green_ok} tercih tuttu &nbsp;·&nbsp; "
            f"🟡 {dp_yellow_hit} kaçın gününde atandı &nbsp;·&nbsp; "
            f"🔴 {dp_red_hit} izin gününde atandı &nbsp;·&nbsp; "
            f"🟣 {dp_purple_hit} müsait olmadığı gün atandı</div>"
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
    # Kısa adlar elle yazılır: uzun adın ilk 3 harfi "Pazartesi"yi "Paz",
    # "Cumartesi"yi "Cum" yapıp Pazar ve Cuma ile karıştırıyordu.
    cal_day_names = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
    # Telefonda 7 sütun alt alta dizilir: başlık satırı ve ay başındaki boş
    # kutular gizlenir, her günün kartında gün adı yazar (CSS: nobet_stil.py).
    cal_hdr = st.container(key="cal-head").columns(7)
    for j, gn in enumerate(cal_day_names):
        with cal_hdr[j]:
            bg = ui["cal_head_we_bg"] if j >= 5 else ui["cal_head_bg"]
            tc = ui["weekend_text"] if j >= 5 else ui["day_text"]
            st.markdown(
                f"<div style='text-align:center;background:{bg};border-radius:8px;padding:6px 2px;"
                f"font-size:12px;font-weight:700;color:{tc};'>{gn}</div>",
                unsafe_allow_html=True
            )

    # ── Takvim hücreleri ──────────────────────────────────
    first_weekday = date(yil, ay, 1).weekday()
    cal_day_num = 1
    cal_grid = st.container(key="cal-grid")
    for week in range(6):
        if cal_day_num > gun_sayisi:
            break
        week_cols = cal_grid.columns(7)
        for j in range(7):
            with week_cols[j]:
                abs_pos = week * 7 + j
                if abs_pos < first_weekday or cal_day_num > gun_sayisi:
                    st.markdown("<div class='cal-empty' style='min-height:95px;'></div>", unsafe_allow_html=True)
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
                        outline = {1: "2px solid #16a34a", 2: "2px solid #d97706", 3: "2px solid #dc2626", 4: "2px solid #9333ea"}.get(pv, "none")
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
                        f"<div style='font-size:15px;font-weight:700;color:{day_c};'>{day_icon}{dn}"
                        f"<span class='cal-gun-adi'> {tr_gunler[date(yil, ay, dn).weekday()]}</span></div>"
                        f"<div style='margin-top:3px;'>{chips}{empty_msg}</div>"
                        f"{color_bar_html}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    if len(all_nobetciler) >= 2:
                        if st.button("🔄", key=f"cal_swap_{dn}", help="Görev yerlerini değiştir"):
                            st.session_state['cal_swap_day'] = dn
                            st.rerun()
                    cal_day_num += 1
