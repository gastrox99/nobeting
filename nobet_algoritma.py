# -*- coding: utf-8 -*-
"""nobet_algoritma.py - Nöbet listesini kuran simülasyon ve denge takasları."""
import os
import random
from datetime import date, timedelta
from itertools import combinations as _combinations
from numbers import Integral

import numpy as np
import pandas as pd

from nobet_gunler import _column_day


def together_partners(together_pairs):
    """{kişi: {birlikte tutmak istediği kişiler}}."""
    partners = {}
    for a, b in together_pairs or ():
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
    return partners


def together_split_count(schedule, together_pairs, columns):
    """Birlikte ikilisinden yalnız birinin nöbetçi olduğu gün sayısı (az = iyi)."""
    split = 0
    for a, b in together_pairs or ():
        if a not in schedule.index or b not in schedule.index:
            continue
        for c in columns:
            if c in schedule.columns and bool(schedule.at[a, c]) != bool(schedule.at[b, c]):
                split += 1
    return split


def _together_count(people, together_pairs):
    """Aynı gündeki kişilerden kaç birlikte ikilisi oluşur."""
    people = set(people)
    return sum(1 for a, b in together_pairs or () if a in people and b in people)


def special_cap(person_limits, person):
    """Kişinin bu ay en fazla hafta sonu/tatil nöbeti (999 = sınırsız)."""
    return (person_limits or {}).get(person, {}).get("max_special", 999)


def has_total_limit(limits):
    """Toplam nöbet limiti (en az / en fazla) var mı? Yalnız hafta sonu sınırı sayılmaz."""
    return limits.get("min", 0) > 0 or limits.get("max", 999) < 999


def special_capped_people(person_limits):
    """Hafta sonu/tatil sınırı olan kişiler; bu kişiler hafta sonu dengesine katılmaz."""
    return {p for p, v in (person_limits or {}).items() if v.get("max_special", 999) < 999}


def _balance_fridays(schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
                     df_preferred, min_bosluk, forbidden_pairs, fixed_assignments,
                     initial_last_shift, max_moves=200, together_pairs=None):
    """Cuma sayılarındaki farkı takasla azaltır; toplamlar değişmez.

    Çok Cuması olan kişinin bir Cuma nöbeti, az Cuması olan kişinin bir hafta
    içi nöbetiyle yer değiştirir. Hafta sonu/tatil nöbetlerine dokunulmaz.
    """
    friday_cols = [c for c in sutunlar if is_friday_column(gun_detaylari[c])]
    weekday_cols = [
        c for c in sutunlar
        if not gun_detaylari[c]["weekend"] and not gun_detaylari[c]["holiday"] and c not in friday_cols
    ]
    return _balance_by_swaps(
        schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool, df_preferred,
        min_bosluk, forbidden_pairs, fixed_assignments, initial_last_shift,
        friday_cols, weekday_cols, max_moves=max_moves, together_pairs=together_pairs,
    )


def _balance_holidays(schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
                      df_preferred, min_bosluk, forbidden_pairs, fixed_assignments,
                      initial_last_shift, history_holiday, max_moves=200, together_pairs=None):
    """Bayram/tatil sayılarındaki farkı (varsa son 12 ay dahil) takasla azaltır.

    Çok tatil nöbeti olanın bir tatil nöbeti, az olanın tatil olmayan bir
    hafta sonu nöbetiyle yer değiştirir. Toplam, hafta sonu/tatil ve Cuma
    sayıları değişmez; yalnızca hangi özel günün kimde olduğu değişir.
    """
    holiday_cols = [c for c in sutunlar if gun_detaylari[c]["holiday"]]
    weekend_cols = [
        c for c in sutunlar if gun_detaylari[c]["weekend"] and not gun_detaylari[c]["holiday"]
    ]
    return _balance_by_swaps(
        schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool, df_preferred,
        min_bosluk, forbidden_pairs, fixed_assignments, initial_last_shift,
        holiday_cols, weekend_cols, extra_counts=history_holiday, max_moves=max_moves,
        swap_ok=_no_new_back_to_back_weekends(schedule, sutunlar, gun_detaylari),
        together_pairs=together_pairs,
    )


def _balance_specials(schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
                      df_preferred, min_bosluk, forbidden_pairs, fixed_assignments,
                      initial_last_shift, history_special, max_moves=200, capped=None,
                      together_pairs=None, prior_weeks=None):
    """Hafta sonu/tatil sayılarındaki farkı (önceki aylar dahil) takasla azaltır.

    Simülasyon seçimi hafta sonu dengesini aya yayma ve bayram takibiyle
    birlikte tartar; bu yüzden fark bazen 2 kalabiliyordu. Çok hafta
    sonu/tatil nöbeti olanın bir özel gün nöbeti, az olanın Cuma olmayan bir
    hafta içi nöbetiyle yer değiştirir. Toplam ve Cuma sayıları değişmez.
    Önce art arda hafta sonu oluşturmayan takaslar denenir; fark yine 2 ya
    da daha fazla kalırsa denge önce gelir (art arda hafta sonu olabilir).
    Hafta sonu sınırı olan kişiler (``capped``) bu takaslara katılmaz.
    """
    isimler = [p for p in isimler if p not in (capped or ())]
    special_cols = [c for c in sutunlar if gun_detaylari[c]["weekend"] or gun_detaylari[c]["holiday"]]
    weekday_cols = [
        c for c in sutunlar
        if c not in special_cols and not is_friday_column(gun_detaylari[c])
    ]
    schedule = _balance_by_swaps(
        schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool, df_preferred,
        min_bosluk, forbidden_pairs, fixed_assignments, initial_last_shift,
        special_cols, weekday_cols, extra_counts=history_special, max_moves=max_moves,
        swap_ok=_no_new_back_to_back_weekends(schedule, sutunlar, gun_detaylari, prior_weeks),
        together_pairs=together_pairs,
    )
    return _balance_by_swaps(
        schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool, df_preferred,
        min_bosluk, forbidden_pairs, fixed_assignments, initial_last_shift,
        special_cols, weekday_cols, extra_counts=history_special, max_moves=max_moves,
        together_pairs=together_pairs,
    )


def prior_weekend_weeks(gun_detaylari, initial_last_shift):
    """Önceki ayın son nöbeti hafta sonuna denk geliyorsa {kişi: hafta numarası}.

    Hafta numarası bu ayın günleriyle aynı sayıdır. 0 = önceki ayın son günü.
    """
    if not gun_detaylari or not initial_last_shift:
        return {}
    info = next(item for item in gun_detaylari.values() if item["day_num"] == 1)
    raw = info["full_date"].split()[0]
    day_s, month_s, year_s = raw.split(".")
    day1 = date(int(year_s), int(month_s), int(day_s))
    found = {}
    for person, offset in initial_last_shift.items():
        if offset is None:
            continue
        try:
            offset = int(offset)
        except (TypeError, ValueError):
            continue
        shift = day1 + timedelta(days=offset - 1)
        if shift.weekday() >= 5:
            found[person] = (shift.toordinal() - shift.weekday()) // 7
    return found


def _no_new_back_to_back_weekends(schedule, sutunlar, gun_detaylari, prior_weeks=None):
    """Takas art arda hafta sonu nöbeti oluşturmasın (simülasyon bundan kaçınır)."""
    weekend_days = [c for c in sutunlar if gun_detaylari[c]["weekend"]]
    prior_weeks = prior_weeks or {}

    def back_to_back(person):
        weeks = {
            gun_detaylari[c].get("week", (gun_detaylari[c]["day_num"] - 1) // 7)
            for c in weekend_days if bool(schedule.at[person, c])
        }
        if person in prior_weeks:
            weeks.add(prior_weeks[person])
        weeks = sorted(weeks)
        return sum(1 for a, b in zip(weeks, weeks[1:]) if b == a + 1)

    def swap_ok(giver, receiver, apply_swap, undo_swap):
        before = back_to_back(giver) + back_to_back(receiver)
        apply_swap()
        after = back_to_back(giver) + back_to_back(receiver)
        undo_swap()
        return after <= before

    return swap_ok


def _balance_by_swaps(schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
                      df_preferred, min_bosluk, forbidden_pairs, fixed_assignments,
                      initial_last_shift, target_cols, partner_cols, extra_counts=None,
                      max_moves=200, swap_ok=None, together_pairs=None):
    """``target_cols`` sayılarındaki farkı takasla azaltır; toplamlar değişmez.

    Çok sayısı olan kişinin bir hedef gün nöbeti, az olanın bir eş gün
    (``partner_cols``) nöbetiyle yer değiştirir. ``extra_counts`` geçmiş
    aylardan gelen farktır. İkisi de yeni gününde müsait olmalı, kaçınma
    günü olmamalı, dinlenme kuralına ve yasak çiftlere uymalı; sabit
    nöbetler yerinden oynamaz.

    ``together_pairs`` verilirse önce birlikte ikilileri ayırmayan takaslar
    denenir; fark yine 2 ya da daha fazla kalırsa denge önce gelir.
    """
    if together_pairs:
        def _keeps_together(giver, receiver, apply_swap, undo_swap):
            cols = [c for c in sutunlar if bool(schedule.at[giver, c]) != bool(schedule.at[receiver, c])]

            def together():
                return sum(_together_count(schedule.index[schedule[c]], together_pairs) for c in cols)

            before = together()
            apply_swap()
            after = together()
            undo_swap()
            return after >= before and (swap_ok is None or swap_ok(giver, receiver, apply_swap, undo_swap))

        schedule = _balance_by_swaps(
            schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool, df_preferred,
            min_bosluk, forbidden_pairs, fixed_assignments, initial_last_shift,
            target_cols, partner_cols, extra_counts=extra_counts, max_moves=max_moves,
            swap_ok=_keeps_together,
        )
    fixed = set(fixed_assignments or ())
    initial = initial_last_shift or {}
    extra = extra_counts or {}
    day_of = {col: gun_detaylari[col]["day_num"] for col in sutunlar}
    target_cols = list(target_cols)
    partner_cols = list(partner_cols)
    if not target_cols or not partner_cols:
        return schedule

    def avoided(person, col):
        if df_preferred is None or person not in df_preferred.index or col not in df_preferred.columns:
            return False
        try:
            return int(df_preferred.at[person, col]) == 2
        except (TypeError, ValueError):
            return False

    def can_take(person, col, dropped):
        if bool(df_unwanted_bool.at[person, col]) or avoided(person, col):
            return False
        day = day_of[col]
        previous = initial.get(person)
        if previous is not None and day - previous <= min_bosluk:
            return False
        for other in sutunlar:
            if other in (col, dropped) or not bool(schedule.at[person, other]):
                continue
            if abs(day_of[other] - day) <= min_bosluk:
                return False
        if forbidden_pairs:
            for mate in schedule.index[schedule[col]]:
                if tuple(sorted((person, mate))) in forbidden_pairs:
                    return False
        return True

    for _ in range(max_moves):
        counts = {
            p: sum(1 for c in target_cols if bool(schedule.at[p, c])) + extra.get(p, 0)
            for p in isimler
        }
        move = None
        for giver in sorted(isimler, key=lambda p: -counts[p]):
            for receiver in sorted(isimler, key=lambda p: counts[p]):
                if counts[giver] - counts[receiver] <= 1:
                    break
                for tcol in target_cols:
                    if not bool(schedule.at[giver, tcol]) or bool(schedule.at[receiver, tcol]):
                        continue
                    if (giver, tcol) in fixed:
                        continue
                    for pcol in partner_cols:
                        if not bool(schedule.at[receiver, pcol]) or bool(schedule.at[giver, pcol]):
                            continue
                        if (receiver, pcol) in fixed:
                            continue
                        # Takası önce uygula, iki tarafı da yeni hâliyle kontrol et.
                        schedule.at[giver, tcol] = False
                        schedule.at[receiver, pcol] = False
                        ok = can_take(receiver, tcol, pcol) and can_take(giver, pcol, tcol)
                        schedule.at[giver, tcol] = True
                        schedule.at[receiver, pcol] = True
                        if ok and swap_ok is not None:
                            def _apply(g=giver, r=receiver, t=tcol, q=pcol):
                                schedule.at[g, t] = False
                                schedule.at[g, q] = True
                                schedule.at[r, q] = False
                                schedule.at[r, t] = True

                            def _undo(g=giver, r=receiver, t=tcol, q=pcol):
                                schedule.at[g, t] = True
                                schedule.at[g, q] = False
                                schedule.at[r, q] = True
                                schedule.at[r, t] = False

                            ok = swap_ok(giver, receiver, _apply, _undo)
                        if ok:
                            move = (giver, receiver, tcol, pcol)
                            break
                    if move:
                        break
                if move:
                    break
            if move:
                break
        if move is None:
            break
        giver, receiver, tcol, pcol = move
        schedule.at[giver, tcol] = False
        schedule.at[giver, pcol] = True
        schedule.at[receiver, pcol] = False
        schedule.at[receiver, tcol] = True
    return schedule


def _fill_open_days(schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
                    df_preferred, min_bosluk, forbidden_pairs, person_limits,
                    initial_last_shift, kisi_sayisi, together_pairs=None):
    """Dengeleme takaslarından sonra doldurulabilir hâle gelen boş yerleri doldurur.

    Takas birini bir günden çekince o kişi, daha önce kimsenin alamadığı
    bir güne uygun hâle gelebilir. Aday izinli olmamalı, dinlenme kuralına,
    yasak çiftlere ve max limitine uymalı. Kaçınma günü olmayan ve en az
    nöbeti olan önce gelir; kaçınma günü yalnızca başka aday yoksa verilir.
    O gün nöbetçi olanla birlikte tutmak isteyen (``together_pairs``) öne alınır.
    """
    initial = initial_last_shift or {}
    limits = person_limits or {}
    partners = together_partners(together_pairs)
    day_of = {col: gun_detaylari[col]["day_num"] for col in sutunlar}

    def avoided(person, col):
        if df_preferred is None or person not in df_preferred.index or col not in df_preferred.columns:
            return False
        try:
            return int(df_preferred.at[person, col]) == 2
        except (TypeError, ValueError):
            return False

    def special(col):
        info = gun_detaylari[col]
        return bool(info.get("weekend") or info.get("holiday"))

    def eligible(person, col):
        if bool(schedule.at[person, col]) or bool(df_unwanted_bool.at[person, col]):
            return False
        day = day_of[col]
        previous = initial.get(person)
        if previous is not None and day - previous <= min_bosluk:
            return False
        for other in sutunlar:
            if other != col and bool(schedule.at[person, other]) and abs(day_of[other] - day) <= min_bosluk:
                return False
        if forbidden_pairs:
            for mate in schedule.index[schedule[col]]:
                if tuple(sorted((person, mate))) in forbidden_pairs:
                    return False
        max_limit = limits.get(person, {}).get("max")
        if max_limit is not None and int(schedule.loc[person].sum()) >= max_limit:
            return False
        if special(col) and sum(
            1 for c in sutunlar if special(c) and bool(schedule.at[person, c])
        ) >= special_cap(limits, person):
            return False
        return True

    for col in sutunlar:
        while int(schedule[col].sum()) < kisi_sayisi:
            candidates = [p for p in isimler if eligible(p, col)]
            if not candidates:
                break
            sp = special(col)
            on_duty = set(schedule.index[schedule[col]])
            best = min(candidates, key=lambda p: (
                avoided(p, col),
                -len(partners.get(p, set()) & on_duty),
                int(schedule.loc[p].sum()),
                sum(1 for c in sutunlar if special(c) and bool(schedule.at[p, c])) if sp else 0,
                isimler.index(p),
            ))
            schedule.at[best, col] = True
    return schedule


def _is_special_day(info):
    """Hafta sonu ya da tatil günü mü?"""
    return bool(info.get("weekend") or info.get("holiday"))


def is_friday_column(info):
    """Tatil olmayan Cuma mı? (Tatile denk gelen Cuma zaten özel gün sayılır.)"""
    return info.get("weekday") == 4 and not info.get("weekend") and not info.get("holiday")


def _find_valid_group(adaylar, kisi_sayisi, forbidden_pairs):
    """Sıralı listeden forbidden_pairs'e uymayan kisi_sayisi boyutunda grup bulur.
    Önce greedy (sıra korunur), başarısız olursa tam kombinasyon araması yapar."""
    secilenler = []
    for p in adaylar:
        valid = True
        if forbidden_pairs:
            for s in secilenler:
                if tuple(sorted((p, s))) in forbidden_pairs:
                    valid = False
                    break
        if valid:
            secilenler.append(p)
            if len(secilenler) >= kisi_sayisi:
                return secilenler
    # Greedy yetmedi; tüm kombinasyonları dene (geçerli varsa mutlaka bulur)
    if len(secilenler) < kisi_sayisi and forbidden_pairs and len(adaylar) >= kisi_sayisi:
        for combo in _combinations(adaylar, kisi_sayisi):
            ok = True
            for i in range(len(combo)):
                for j in range(i + 1, len(combo)):
                    if tuple(sorted((combo[i], combo[j]))) in forbidden_pairs:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return list(combo)
    return secilenler


def _pick_varied_group(adaylar, needed, scores, sabitler, pair_count, forbidden_pairs, partners=None):
    """Sıralı adaylardan ``needed`` kişi seçer; aynı gün nöbet tutanlarla daha
    önce çok birlikte tutmuş adayın puanı PAIR_WEIGHT kadar kötüleşir (aynı
    ikili her nöbette yan yana düşmesin). Yasak ikililer seçilmez; seçim
    tıkanırsa eski yöntem (_find_valid_group) kullanılır.

    ``partners`` birlikte nöbet isteyenlerdir: eşi o gün nöbetçi olan aday
    TOGETHER_WEIGHT kadar öne alınır; eşi o güne gelemeyecek aday aynı kadar
    geri itilir. Birlikte ikililere tekrar cezası uygulanmaz."""
    partners = partners or {}
    secilenler = []
    havuz = list(adaylar)
    while len(secilenler) < needed:
        yanindakiler = list(sabitler) + secilenler
        kalan_yer = needed - len(secilenler) - 1
        en_iyi, en_iyi_puan = None, None
        for aday in havuz:
            if forbidden_pairs and any(tuple(sorted((aday, y))) in forbidden_pairs for y in yanindakiler):
                continue
            esler = partners.get(aday, ())
            puan = scores[aday] + PAIR_WEIGHT * sum(
                pair_count.get(tuple(sorted((aday, y))), 0) for y in yanindakiler if y not in esler
            )
            if esler:
                gelen = sum(1 for y in yanindakiler if y in esler)
                if gelen:
                    puan -= TOGETHER_WEIGHT * gelen
                elif not (kalan_yer > 0 and any(e in havuz and e != aday for e in esler)):
                    puan += TOGETHER_WEIGHT
            if en_iyi_puan is None or puan < en_iyi_puan:
                en_iyi, en_iyi_puan = aday, puan
        if en_iyi is None:
            break
        secilenler.append(en_iyi)
        havuz.remove(en_iyi)
    if len(secilenler) < needed:
        eski = _find_valid_group(adaylar, needed, forbidden_pairs)[:needed]
        if len(eski) > len(secilenler):
            return eski
    return secilenler


def pair_repeat_excess(schedule, people, columns, kisi_sayisi, exempt=None):
    """Aynı ikilinin beklenenden fazla birlikte nöbet tutma sayısı.

    Her ikiliye düşen ortak nöbetin yukarı yuvarlanmış ortalaması (en az 1)
    "normal" sayılır; bunu aşan her ortak nöbet 1 puandır (az = iyi).
    ``exempt`` (birlikte nöbet isteyen ikililer) sayılmaz.
    """
    exempt = {tuple(sorted(p)) for p in (exempt or ())}
    people = list(people)
    if len(people) < 2 or kisi_sayisi < 2:
        return 0
    allowed = pair_repeat_allowed(len(people), len(columns), kisi_sayisi)
    counts = {}
    for c in columns:
        who = [p for p in people if p in schedule.index and c in schedule.columns and schedule.at[p, c]]
        for a, b in _combinations(sorted(who), 2):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    return sum(max(0, v - allowed) for pair, v in counts.items() if pair not in exempt)


def pair_repeat_allowed(n_people, n_days, kisi_sayisi):
    """Bir ikilinin ayda "normal" sayılan ortak nöbet sayısı."""
    pairs = n_people * (n_people - 1) / 2
    per_day = kisi_sayisi * (kisi_sayisi - 1) / 2
    return max(1, int(np.ceil(n_days * per_day / max(1, pairs) - 1e-9)))


def spread_targets(isimler, n_days, kisi_sayisi, person_limits=None):
    """Kişi başı beklenen nöbet aralığı ve "nöbet hızı" çarpanı.

    Herkes eşit alırsa iki nöbet arası kişi sayısı / günlük nöbetçi gün olur.
    En fazla nöbeti bundan az olan kişinin aralığı kendi limitine göredir
    (30 günde en fazla 3 → 10 günde bir); hızı da buna göre ölçülür, yoksa
    az nöbeti olduğu için hep öne alınır, limitini ayın başında doldurur.
    Dönen değer ({kişi: aralık}, {kişi: çarpan}).
    """
    team_gap = len(isimler) / max(1, kisi_sayisi)
    avg_shifts = n_days * kisi_sayisi / max(1, len(isimler))
    gaps, pace = {}, {}
    for p in isimler:
        max_limit = (person_limits or {}).get(p, {}).get('max', 999)
        if 0 < max_limit < avg_shifts:
            gaps[p] = n_days / max_limit
            pace[p] = avg_shifts / max_limit
        else:
            gaps[p] = team_gap
            pace[p] = 1.0
    return gaps, pace


def _balance_totals(schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
                    df_preferred, min_bosluk, forbidden_pairs, person_limits,
                    fixed_assignments, initial_last_shift, history_special,
                    max_moves=500, history_total=None, history_holiday=None,
                    together_pairs=None, prior_weeks=None, allow_avoid=False):
    """Toplam nöbet farkını kurallara uyarak en fazla 1'e indirir.

    Gün gün ilerleyen seçim ay sonunda toplamlarda kayma bırakabiliyordu
    (ör. 62 nöbet / 9 kişide biri 6, biri 8). En çok nöbeti olandan en az
    olana, aynı gün içinde, şu koşullarla bir nöbet aktarılır:
    alan kişi o gün müsait, dinlenme kuralına ve yasak
    çiftlere uyuyor, max limitini aşmıyor; veren kişinin nöbeti sabit değil
    ve min limitinin altına düşmüyor; hafta sonu/tatil dengesi bozulmuyor.
    Tercih edilen günler mümkün olduğunca korunur. Kaçınılan gün, toplam
    farkı 1'in üstündeyken ve başka gün kalmadıysa kullanılabilir.

    Toplam farkı zaten 1 iken önceki aylar (``history_total``) hesaba katılır:
    geçmişiyle birlikte 2 ya da daha fazla önde olan kişinin fazla nöbeti,
    geçmişi az olana geçer. Bu ayın farkı yine 1'i aşmaz.
    Tatil günü aktarılırken son 12 ayın tatil sayısı (``history_holiday``)
    bozulmaz. Birlikte nöbet isteyen ikiliyi (``together_pairs``) ayıran
    aktarım en sona kalır; yalnızca geçmiş yüzünden yapılacaksa hiç yapılmaz.
    """
    limits = person_limits or {}
    history_total = history_total or {}
    history_holiday = history_holiday or {}
    prior_weeks = prior_weeks or {}
    weekend_cols = [c for c in sutunlar if gun_detaylari[c]["weekend"]]
    fixed = set(fixed_assignments or ())
    history = history_special or {}
    initial = initial_last_shift or {}
    day_of = {col: gun_detaylari[col]["day_num"] for col in sutunlar}
    special_cols = {col for col in sutunlar if gun_detaylari[col]["weekend"] or gun_detaylari[col]["holiday"]}
    friday_cols = {col for col in sutunlar if is_friday_column(gun_detaylari[col])}
    holiday_cols = {col for col in sutunlar if gun_detaylari[col]["holiday"]}

    def pref(person, col):
        if df_preferred is None or person not in df_preferred.index or col not in df_preferred.columns:
            return 0
        try:
            return int(df_preferred.at[person, col])
        except (TypeError, ValueError):
            return 0

    def rest_ok(person, col):
        day = day_of[col]
        previous = initial.get(person)
        if previous is not None and day - previous <= min_bosluk:
            return False
        for other in sutunlar:
            if other != col and bool(schedule.at[person, other]) and abs(day_of[other] - day) <= min_bosluk:
                return False
        return True

    for _ in range(max_moves):
        totals = {p: int(schedule.loc[p].sum()) for p in isimler}
        specials = {
            p: sum(1 for c in special_cols if bool(schedule.at[p, c])) + history.get(p, 0)
            for p in isimler
        }
        month_specials = {p: sum(1 for c in special_cols if bool(schedule.at[p, c])) for p in isimler}
        fridays = {p: sum(1 for c in friday_cols if bool(schedule.at[p, c])) for p in isimler}
        holidays = {
            p: sum(1 for c in holiday_cols if bool(schedule.at[p, c])) + history_holiday.get(p, 0)
            for p in isimler
        }
        overall = {p: totals[p] + history_total.get(p, 0) for p in isimler}
        best_move = None
        for giver in sorted(isimler, key=lambda p: (-totals[p], -overall[p])):
            if totals[giver] - 1 < limits.get(giver, {}).get("min", 0):
                continue
            for receiver in sorted(isimler, key=lambda p: (totals[p], overall[p])):
                gap = totals[giver] - totals[receiver]
                if gap <= 0:
                    break
                # Fark 1 ise yalnızca geçmişiyle birlikte belirgin önde olan verir.
                if gap == 1 and overall[giver] - overall[receiver] < 2:
                    continue
                if totals[receiver] + 1 > limits.get(receiver, {}).get("max", 999):
                    continue
                for col in sutunlar:
                    if not bool(schedule.at[giver, col]) or bool(schedule.at[receiver, col]):
                        continue
                    if (giver, col) in fixed or bool(df_unwanted_bool.at[receiver, col]):
                        continue
                    on_avoid = pref(receiver, col) == 2
                    # Fark 1 iken kaçınılan güne geçilmez. Fark daha büyükse
                    # önce diğer günler; kaçınılan gün yalnız onlar yetmezse.
                    if on_avoid and (not allow_avoid or gap <= 1):
                        continue
                    # Yalnızca geçmiş yüzünden yapılan aktarım verenin tercih ettiği günü almaz.
                    if gap == 1 and pref(giver, col) == 1:
                        continue
                    if col in special_cols and specials[giver] <= specials[receiver]:
                        continue
                    if col in special_cols and month_specials[receiver] >= special_cap(limits, receiver):
                        continue
                    if col in friday_cols and fridays[giver] <= fridays[receiver]:
                        continue
                    if col in holiday_cols and holidays[giver] <= holidays[receiver]:
                        continue
                    others = [p for p in schedule.index[schedule[col]] if p not in (giver, receiver)]
                    if forbidden_pairs and any(tuple(sorted((receiver, o))) in forbidden_pairs for o in others):
                        continue
                    if not rest_ok(receiver, col):
                        continue
                    breaks = 0
                    if together_pairs:
                        breaks = (_together_count(others + [giver], together_pairs)
                                  - _together_count(others + [receiver], together_pairs))
                        if breaks > 0 and gap == 1:
                            continue
                    # Önceki ayın hafta sonuna bitişik hafta sonu, başka gün varken seçilmesin.
                    owned = {
                        gun_detaylari[c]["week"] for c in weekend_cols if bool(schedule.at[receiver, c])
                    }
                    if receiver in prior_weeks:
                        owned.add(prior_weeks[receiver])
                    week = gun_detaylari[col]["week"]
                    links_weekend = (
                        col in weekend_cols and week not in owned
                        and ((week - 1) in owned or (week + 1) in owned)
                    )
                    # Önce hafta içi; verenin tercih ettiği gün en son, alanın tercih ettiği gün önce.
                    score = (
                        (1000 if on_avoid else 0)
                        + (80 if links_weekend else 0)
                        + (50 * max(0, breaks))
                        + (10 if col in special_cols else 0)
                        + (4 if col in friday_cols else 0)
                        + (5 if pref(giver, col) == 1 else 0)
                        - (3 if pref(receiver, col) == 1 else 0)
                    )
                    if best_move is None or score < best_move[0]:
                        best_move = (score, giver, receiver, col)
                if best_move is not None:
                    break
            if best_move is not None:
                break
        if best_move is None:
            break
        _, giver, receiver, col = best_move
        schedule.at[giver, col] = False
        schedule.at[receiver, col] = True
    if not allow_avoid:
        totals_now = [int(schedule.loc[p].sum()) for p in isimler]
        if totals_now and max(totals_now) - min(totals_now) > 1:
            return _balance_totals(
                schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
                df_preferred, min_bosluk, forbidden_pairs, person_limits,
                fixed_assignments, initial_last_shift, history_special,
                max_moves=max_moves, history_total=history_total,
                history_holiday=history_holiday, together_pairs=together_pairs,
                prior_weeks=prior_weeks, allow_avoid=True,
            )
    return schedule


# Aya yayma ağırlıkları: beklenen aralıktan her eksik gün için seçim cezası
# ve simülasyonlar arasında seçimde kısa aralık başına ceza. Denge
# kurallarından (toplam ×10, Cuma ×40, hafta sonu ×100) bilerek hafiftir.
SPREAD_WEIGHT = 3
SPREAD_SCORE_WEIGHT = 4
# Uzun süre nöbet tutmayan öne alınır: beklenen aralıktan sonraki her boş gün
# için seçimde öncelik; simülasyonlar arasında seçimde en uzun boşluğu
# beklenenin LONG_GAP_LIMIT katını aşan kişi başına, aşan her gün için ceza.
LONG_GAP_WEIGHT = 3
LONG_GAP_SCORE_WEIGHT = 10
LONG_GAP_LIMIT = 1.5
# Aynı ikiliyi tekrar tekrar aynı güne koymama: seçimde önceki her ortak
# nöbet başına ceza ve simülasyonlar arasında seçimde beklenenden fazla ortak
# nöbet başına ceza. Hafta sonu/Cuma dengesinden hafiftir.
PAIR_WEIGHT = 12
PAIR_SCORE_WEIGHT = 15
# Birlikte nöbet isteyenler: eşi o gün nöbetçiyse seçimde öncelik (tercih
# edilen gün -500'den hafif, hafta sonu dengesindeki 1 farktan ağır) ve
# simülasyonlar arasında seçimde ikiliden yalnız birinin nöbetçi olduğu her
# gün için ceza. Kesin kural değildir; denge ve kurallar önce gelir.
TOGETHER_WEIGHT = 400
TOGETHER_SCORE_WEIGHT = 80
# Gün aşırı zincir: iki nöbet arası tam 2 gün (nöbet-boş-nöbet). İki nöbet
# normaldir; üçüncüsü (ve sonrakiler) hem seçimde hem simülasyonlar arasında
# seçimde cezalanır. Hafta sonu dengesinden (×100) hafif, toplamdan ağırdır.
ALT_CHAIN_WEIGHT = 40
ALT_CHAIN_SCORE_WEIGHT = 60


def alternate_day_chains(schedule, people, columns, day_of=None):
    """Gün aşırı zincirde ikiyi aşan nöbet sayısı (d, d+2, d+4 üçlüsü başına 1).

    3 nöbetlik zincir 1, 4 nöbetlik 2 sayılır.
    """
    excess = 0
    for p in people:
        if p not in schedule.index:
            continue
        days = {
            (day_of[c] if day_of else _column_day(c)) for c in columns
            if c in schedule.columns and bool(schedule.at[p, c])
        }
        excess += sum(1 for d in days if d + 2 in days and d + 4 in days)
    return excess


def weekend_pair_count(schedule, columns, gun_detaylari):
    """Bu aydaki art arda hafta sonu çiftleri. Aradaki boş hafta sayacı sıfırlamaz.

    Önceki ayın hafta sonu bu sayıya girmez; o, seçim anındaki ayrı cezadadır.
    """
    total = 0
    for person in schedule.index:
        worked = {
            gun_detaylari[c]["week"] for c in columns
            if gun_detaylari[c]["weekend"] and c in schedule.columns and bool(schedule.at[person, c])
        }
        total += sum(1 for week in worked if week + 1 in worked)
    return total


def run_scheduling_core(isimler, sutunlar, df_unwanted_bool, gun_detaylari,
                        min_bosluk, kisi_sayisi, forbidden_pairs=None,
                        person_limits=None, df_preferred=None,
                        simulation_count=10, progress_callback=None,
                        initial_last_shift=None, history_special=None,
                        fixed_assignments=None, history_total=None,
                        history_holiday=None, together_pairs=None):
    """
    Saf zamanlama algoritması (Streamlit bağımsız).
    progress_callback(int) isteğe bağlı ilerleme bildirimi için kullanılır.

    simulation_count pozitif bir tamsayı olmalıdır; aksi halde ValueError yükseltilir.

    initial_last_shift: {kişi: gün} önceki ayın son nöbeti, bu aya göre
        (0 = önceki ayın son günü, -1 = ondan önceki gün). Ay geçişinde
        dinlenme kuralı için kullanılır.
    history_special: {kişi: fark} önceki aylardaki hafta sonu/tatil
        nöbetlerinin ekip ortalamasından farkı; dengelemede hesaba katılır.
    fixed_assignments: {(kişi, sütun)} elle sabitlenmiş nöbetler. Aynen
        korunur; dinlenme kuralı, max limit ve yasak çiftlerde hesaba katılır,
        simülasyon yalnızca kalan boşlukları doldurur.
    history_total: {kişi: fark} önceki aylardaki toplam nöbetlerin ay
        ortalamasından farkı. Bu ayın toplam farkı yine en fazla 1 kalır;
        geçmişi fazla olan, bu ayın "1 fazla" alanlarından olmaz.
    history_holiday: {kişi: fark} son 12 aydaki tatil/bayram nöbetlerinin
        ekip ortalamasından farkı; tatil günleri azı olana verilir.

    Nöbetler aya yayılır: kişinin son nöbetinden bu yana geçen gün,
    beklenen aralıktan (spread_targets) kısaysa hafif ceza alır, uzunsa
    her fazla gün için öne alınır. Bunlar denge kurallarından hafiftir.

    together_pairs: {(kişi, kişi)} birlikte nöbet tutmak isteyenler; mümkün
        olduğunca aynı güne konur (güçlü tercih, kesin kural değil).
    """
    together_pairs = {tuple(sorted(p)) for p in (together_pairs or ()) if p[0] in isimler and p[1] in isimler}
    if forbidden_pairs:
        together_pairs -= set(forbidden_pairs)
    if kisi_sayisi < 2:
        together_pairs = set()
    partners = together_partners(together_pairs)
    initial_last_shift = initial_last_shift or {}
    prior_weeks = prior_weekend_weeks(gun_detaylari, initial_last_shift)
    history_special = history_special or {}
    history_total = history_total or {}
    history_holiday = history_holiday or {}
    person_gap, pace = spread_targets(isimler, len(sutunlar), kisi_sayisi, person_limits)
    team_gap = {p: len(isimler) / max(1, kisi_sayisi) for p in isimler}
    fixed_by_col = {}
    fixed_days = {}
    fixed_special = {}
    for person, column in (fixed_assignments or ()):
        if person in isimler and column in gun_detaylari:
            fixed_by_col.setdefault(column, []).append(person)
            fixed_days.setdefault(person, set()).add(gun_detaylari[column]["day_num"])
            if _is_special_day(gun_detaylari[column]):
                fixed_special[person] = fixed_special.get(person, 0) + 1
    # Hafta sonu sınırı olanlar sınıra kadar alır; hafta sonu dengesine katılmaz.
    sp_caps = {p: special_cap(person_limits, p) for p in isimler}
    capped = {p for p in isimler if sp_caps[p] < 999}
    balanced = [p for p in isimler if p not in capped] or list(isimler)
    for column in fixed_by_col:
        fixed_by_col[column].sort(key=isimler.index)
    if isinstance(simulation_count, bool) or not isinstance(simulation_count, Integral) or simulation_count <= 0:
        raise ValueError("Simülasyon sayısı pozitif bir tamsayı olmalıdır.")
    # Arayüzdeki Simülasyon da aynı tohumu kullansın (NOBET_SEED boşsa dokunulmaz).
    _nobet_seed = os.environ.get("NOBET_SEED")
    if _nobet_seed:
        random.seed(int(_nobet_seed))

    # Tercih ve izin tabloları denemeler boyunca değişmez; iç döngüde pandas'tan
    # hücre hücre okumak yerine bir kez sözlük/kümeye çevrilir (aynı karşılaştırmalar).
    pref_bonus_of = {}
    if df_preferred is not None:
        pref_cols = [c for c in sutunlar if c in df_preferred.columns]
        for kisi in isimler:
            if kisi not in df_preferred.index:
                continue
            for c in pref_cols:
                pref_val = df_preferred.at[kisi, c]
                if pref_val == 1:
                    pref_bonus_of[(kisi, c)] = -500
                elif pref_val == 2:
                    pref_bonus_of[(kisi, c)] = 300
    unwanted_cells = {(k, c) for k in isimler for c in sutunlar if df_unwanted_bool.at[k, c]}

    best_schedule = None
    best_score = float('inf')
    best_empty = 0

    attempts = int(simulation_count)
    attempt = -1
    while attempt + 1 < attempts:
        attempt += 1
        # Her dört simülasyondan biri eski, sade yaymayla kurulur. Çok sıkışık
        # ekiplerde (az kişi, sık nöbet) yeni öncelikler boş gün bırakabiliyor;
        # öyleyse aşağıda aynı sayıda sade simülasyon daha denenir.
        plain = attempt % 4 == 3 or attempt >= simulation_count
        gap_target = team_gap if plain else person_gap
        long_weight = 0 if plain else LONG_GAP_WEIGHT
        if progress_callback:
            progress_callback(min(attempt + 1, simulation_count))

        stat_total = {i: 0 for i in isimler}
        stat_special = {i: 0 for i in isimler}
        stat_friday = {i: 0 for i in isimler}
        stat_holiday = {i: 0 for i in isimler}
        short_gaps = 0
        month_last = {i: 0 for i in isimler}
        # Aralık bu aya göre ölçülür; önceki ay bilinmiyorsa son nöbet yarım
        # aralık önce sayılır (ay başındaki herkes birden "uzun boş" olmasın).
        spread_from = {i: initial_last_shift.get(i, -10 if plain else -gap_target[i] / 2) for i in isimler}
        worst_gap = {i: 0 for i in isimler}
        stat_consecutive_weekend = {i: 0 for i in isimler}
        pair_count = {}
        last_weekend_shift = {i: prior_weeks.get(i, -10) for i in isimler}
        last_shift_day = {i: initial_last_shift.get(i, -10) for i in isimler}
        prev_shift_day = {i: -10 for i in isimler}
        alt_chains = 0

        temp_schedule = pd.DataFrame(
            {col: [False] * len(isimler) for col in sutunlar}, index=isimler
        )
        future_fixed = {i: len(fixed_days.get(i, ())) for i in isimler}
        future_fixed_special = {i: fixed_special.get(i, 0) for i in isimler}

        def get_decision_score(p, is_sp, col, is_friday=False, is_holiday=False, day=None):
            # Limitli kişi limitine göre ne kadar ilerlediğiyle karşılaştırılır;
            # sabit nöbetleri de baştan sayılır.
            done = stat_total[p] if plain or pace[p] == 1.0 else (stat_total[p] + future_fixed[p]) * pace[p]
            total = done + history_total.get(p, 0) + (random.random() * 0.5)
            sp_count = stat_special[p] + history_special.get(p, 0)
            consecutive_penalty = stat_consecutive_weekend[p] * 200
            # Önceki ayın son hafta sonunun hemen ertesi hafta: seçilirken de ceza.
            # Sayaç ancak atamadan sonra artar; ceza yalnız ona bakarsa ilk çift kaçardı.
            if (
                gun_detaylari[col].get("weekend")
                and prior_weeks.get(p) is not None
                and gun_detaylari[col]["week"] == prior_weeks[p] + 1
            ):
                consecutive_penalty += 200
            pref_bonus = pref_bonus_of.get((p, col), 0)
            limit_penalty = 0
            min_bonus = 0
            if person_limits and p in person_limits:
                max_limit = person_limits[p].get('max', 999)
                if stat_total[p] >= max_limit:
                    limit_penalty = 50000
                # Minimumuna ulaşmamış kişi öne alınır (çoğunlukla hafta içi).
                if stat_total[p] < person_limits[p].get('min', 0):
                    min_bonus = -250
            # Son nöbete çok yakın günler hafifçe geri itilir, uzun süredir
            # nöbet tutmayan öne alınır (aya yayma).
            spread_penalty = 0
            if day is not None:
                gap = day - spread_from[p]
                if gap < gap_target[p]:
                    spread_penalty = (gap_target[p] - gap) * SPREAD_WEIGHT
                else:
                    spread_penalty = -(gap - gap_target[p]) * long_weight
                # Üçüncü gün aşırı nöbet (nöbet-boş-nöbet-boş-nöbet) geri itilir.
                if day - last_shift_day[p] == 2 and last_shift_day[p] - prev_shift_day[p] == 2:
                    spread_penalty += ALT_CHAIN_WEIGHT
            if is_sp:
                # Bayram/tatil: son 12 ayda az tutan öne; hafta sonu dengesinden (×100) hafif.
                holiday_penalty = (
                    (stat_holiday[p] + history_holiday.get(p, 0)) * 60 if is_holiday else 0
                )
                return ((sp_count * 100) + (total * 10) + consecutive_penalty + pref_bonus
                        + limit_penalty + min_bonus + holiday_penalty + spread_penalty)
            # Cuma da dengelenir; hafta sonu/tatil dengesinden (×100) daha hafif.
            friday_penalty = stat_friday[p] * 40 if is_friday else 0
            return ((total * 10) + (sp_count * 1) + friday_penalty + pref_bonus
                    + limit_penalty + min_bonus + spread_penalty)

        empty_shifts = 0
        limit_violations = 0

        for col in sutunlar:
            info = gun_detaylari[col]
            gun_no = info['day_num']
            is_sp = info['weekend'] or info['holiday']
            is_weekend = info['weekend']
            is_friday = is_friday_column(info)
            is_holiday = bool(info['holiday'])
            weekend_num = info.get('week', (gun_no - 1) // 7)

            # Sabit nöbetler önce ve aynen yerleşir.
            sabitler = fixed_by_col.get(col, [])
            for k in sabitler:
                future_fixed[k] -= 1
                if is_sp:
                    future_fixed_special[k] -= 1
            needed = max(0, kisi_sayisi - len(sabitler))

            adaylar = []
            for k in isimler:
                if k in sabitler:
                    continue
                if (k, col) in unwanted_cells:
                    continue
                if (gun_no - last_shift_day[k]) <= min_bosluk:
                    continue
                # Yaklaşan sabit nöbetinin dinlenme aralığına girme.
                if any(0 < f - gun_no <= min_bosluk for f in fixed_days.get(k, ())):
                    continue
                if person_limits and k in person_limits:
                    max_limit = person_limits[k].get('max', 999)
                    # Kalan sabit nöbetleri için yer bırak.
                    if stat_total[k] + future_fixed[k] >= max_limit:
                        continue
                # Hafta sonu sınırı: kalan sabit hafta sonları için de yer bırak.
                if is_sp and stat_special[k] + future_fixed_special[k] >= sp_caps[k]:
                    continue
                if forbidden_pairs and any(
                    tuple(sorted((k, f))) in forbidden_pairs for f in sabitler
                ):
                    continue
                adaylar.append(k)

            random.shuffle(adaylar)
            scores = {x: get_decision_score(x, is_sp, col, is_friday, is_holiday, gun_no) for x in adaylar}
            adaylar.sort(key=scores.get)

            # Yeterli aday yoksa günü tamamen boş bırakma: uyan adayları ata,
            # eksik kalan her nöbetçiyi say (eskiden 1 kişi müsaitken gün boştu).
            secilenler = (
                _pick_varied_group(adaylar, needed, scores, sabitler, pair_count, forbidden_pairs, partners)
                if adaylar and needed else []
            )
            empty_shifts += needed - len(secilenler)
            gunun_ekibi = sorted(list(sabitler) + secilenler)
            for a, b in _combinations(gunun_ekibi, 2):
                pair_count[(a, b)] = pair_count.get((a, b), 0) + 1
            for k in list(sabitler) + secilenler:
                temp_schedule.at[k, col] = True
                stat_total[k] += 1
                if is_sp:
                    stat_special[k] += 1
                if is_friday:
                    stat_friday[k] += 1
                if is_holiday:
                    stat_holiday[k] += 1
                if gun_no - last_shift_day[k] < person_gap[k] * 0.6:
                    short_gaps += 1
                if gun_no - last_shift_day[k] == 2 and last_shift_day[k] - prev_shift_day[k] == 2:
                    alt_chains += 1
                prev_shift_day[k] = last_shift_day[k]
                last_shift_day[k] = gun_no
                spread_from[k] = gun_no
                worst_gap[k] = max(worst_gap[k], gun_no - month_last[k])
                month_last[k] = gun_no
                if is_weekend:
                    if last_weekend_shift[k] >= 0 and weekend_num == last_weekend_shift[k] + 1:
                        stat_consecutive_weekend[k] += 1
                    elif last_weekend_shift[k] >= 0 and weekend_num > last_weekend_shift[k] + 1:
                        stat_consecutive_weekend[k] = 0
                    last_weekend_shift[k] = weekend_num

        if person_limits:
            for p, limits in person_limits.items():
                if stat_total.get(p, 0) < limits.get('min', 0):
                    limit_violations += 1

        totals = list(stat_total.values())
        specials = [stat_special[i] + history_special.get(i, 0) for i in balanced]
        # Seçim bu aydaki her çifti sayar. Döngüdeki sayaç aradan sonra sıfırlanır;
        # o yalnız uzayan seriyi, seçildiği anda cezalandırır.
        consecutive_weekends = weekend_pair_count(temp_schedule, sutunlar, gun_detaylari)
        std_dev_total = np.std(totals)
        std_dev_special = np.std(specials)
        range_total = max(totals) - min(totals)
        # Hafta sonu/tatil farkı Cuma farkından daha ağır cezalandırılır.
        range_special = max(specials) - min(specials)
        fridays = list(stat_friday.values())
        range_friday = max(fridays) - min(fridays)
        holidays = [stat_holiday[i] + history_holiday.get(i, 0) for i in isimler]
        range_holiday = max(holidays) - min(holidays)

        allowed_pairs = pair_repeat_allowed(len(isimler), len(sutunlar), kisi_sayisi)
        long_gaps = sum(
            max(0, max(worst_gap[p], len(sutunlar) + 1 - month_last[p]) - LONG_GAP_LIMIT * person_gap[p])
            for p in isimler if month_last[p]
        )
        pair_excess = sum(
            max(0, v - allowed_pairs) for pair, v in pair_count.items() if pair not in together_pairs
        )
        together_split = together_split_count(temp_schedule, together_pairs, sutunlar)

        score = (
            (empty_shifts * 10000) +
            (limit_violations * 5000) +
            (consecutive_weekends * 500) +
            (range_total * 100) +
            (range_special * 60) +
            (range_friday * 30) +
            (range_holiday * 20) +
            (short_gaps * SPREAD_SCORE_WEIGHT) +
            (long_gaps * LONG_GAP_SCORE_WEIGHT) +
            (pair_excess * PAIR_SCORE_WEIGHT) +
            (alt_chains * ALT_CHAIN_SCORE_WEIGHT) +
            (together_split * TOGETHER_SCORE_WEIGHT) +
            (std_dev_total * 10) +
            (std_dev_special * 5)
        )

        if score < best_score:
            best_score = score
            best_schedule = temp_schedule.copy()
            best_empty = empty_shifts
        if attempt + 1 == simulation_count and best_empty:
            attempts = 2 * simulation_count

    if best_schedule is not None:
        best_schedule = _balance_totals(
            best_schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
            df_preferred, min_bosluk, forbidden_pairs, person_limits,
            {(p, c) for c, people in fixed_by_col.items() for p in people},
            initial_last_shift, history_special,
                    history_total=history_total, history_holiday=history_holiday,
            together_pairs=together_pairs, prior_weeks=prior_weeks,
        )
        best_schedule = _balance_specials(
            best_schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
            df_preferred, min_bosluk, forbidden_pairs,
            {(p, c) for c, people in fixed_by_col.items() for p in people},
            initial_last_shift, history_special, capped=capped, together_pairs=together_pairs,
            prior_weeks=prior_weeks,
        )
        best_schedule = _balance_fridays(
            best_schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
            df_preferred, min_bosluk, forbidden_pairs,
            {(p, c) for c, people in fixed_by_col.items() for p in people},
            initial_last_shift, together_pairs=together_pairs,
        )
        if any(gun_detaylari[c]["holiday"] for c in sutunlar):
            best_schedule = _balance_holidays(
                best_schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
                df_preferred, min_bosluk, forbidden_pairs,
                {(p, c) for c, people in fixed_by_col.items() for p in people},
                initial_last_shift, history_holiday, together_pairs=together_pairs,
            )
        best_schedule = _fill_open_days(
            best_schedule, isimler, sutunlar, gun_detaylari, df_unwanted_bool,
            df_preferred, min_bosluk, forbidden_pairs, person_limits,
            initial_last_shift, kisi_sayisi, together_pairs=together_pairs,
        )
    return best_schedule, best_score


# Damga en sonda: yükleme yarıda kalırsa sonraki çalışma yeniden dener (nobet_tazele.py).
_nobet_loaded_stamp = os.stat(__file__).st_mtime_ns
