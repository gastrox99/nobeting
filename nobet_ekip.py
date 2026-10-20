# -*- coding: utf-8 -*-
"""Ayarlar: ekip tablosu (isim + kişisel limitler), sabit görev yerleri ve birlikte çalışamayanlar çapraz tablosu.

İkisi de ayarlarda saklanan metinleri (``isimler_text``, ``limits_text``,
``forbidden_pairs_text``) düzenler; kayıt, yedek ve Excel aynı metinleri
kullanmaya devam eder.
"""
import streamlit as st

from nobet_core import (
    compact_team_rows,
    escape_html,
    forbidden_pairs_to_text,
    parse_forbidden_pairs,
    parse_person_limits,
    team_settings_rows,
    team_settings_texts,
    toggle_pair_text,
    unknown_names,
)

_NAME = "w_ekip_ad_{}"
_MAX = "w_ekip_max_{}"
_SPECIAL = "w_ekip_hs_{}"


def _team_names(text):
    return [part.strip() for part in (text or "").split(",") if part.strip()]


def _write_rows(rows):
    """Satırları kutulara yazar (kutular çizilmeden önce çağrılmalı)."""
    state = st.session_state
    for i, row in enumerate(rows):
        state[_NAME.format(i)] = row["name"]
        state[_MAX.format(i)] = row["max"]
        state[_SPECIAL.format(i)] = row["max_special"]
    for i in range(len(rows), state.get("_ekip_rows", 0)):
        for key in (_NAME, _MAX, _SPECIAL):
            state.pop(key.format(i), None)
    state["_ekip_rows"] = len(rows)
    state["_ekip_mins"] = [row.get("min", 0) for row in rows]


def render_team_table():
    """Her satır bir kişi: isim, en fazla nöbet, en fazla hafta sonu (boş = sınırsız).

    Sondaki boş satıra isim yazılınca altına yeni boş satır açılır; ismi
    silinen satır kaldırılır. Bir satırdaki ismi değiştirmek ad değişikliğidir.
    """
    state = st.session_state
    source = (state.get("isimler_text", ""), state.get("limits_text", ""))
    if state.get("_ekip_source") != source:
        # İlk açılış ya da dışarıdan değişiklik (kayıt yükleme, geri yükleme).
        _names = _team_names(source[0])
        _write_rows(compact_team_rows(team_settings_rows(_names, parse_person_limits(source[1]))))

    mins = state.get("_ekip_mins", [])
    rows = [
        {
            "name": str(state.get(_NAME.format(i)) or ""),
            "min": mins[i] if i < len(mins) else 0,
            "max": state.get(_MAX.format(i)),
            "max_special": state.get(_SPECIAL.format(i)),
        }
        for i in range(state.get("_ekip_rows", 0))
    ]
    compacted = compact_team_rows(rows)
    if [r["name"] for r in compacted] != [r["name"] for r in rows]:
        _write_rows(compacted)
    names_text, limits_text = team_settings_texts(compacted)
    if names_text != source[0] or limits_text != source[1]:
        # Eski kayıtlarda "Ali,Ayşe" gibi biçimler aynı ekibi gösterir; yeniden yazma.
        if _team_names(names_text) != _team_names(source[0]) or limits_text != source[1]:
            state.isimler_text = names_text
            state.limits_text = limits_text
    state["_ekip_source"] = (state.get("isimler_text", ""), state.get("limits_text", ""))

    with st.container(key="ekip-tablo"):
        st.markdown("**👥 Ekip**")
        widths = [3, 2, 2]
        head = st.columns(widths, gap="small", vertical_alignment="bottom")
        head[0].caption("İsim")
        head[1].caption("En fazla nöbet")
        head[2].caption("En fazla hafta sonu")
        for i in range(state["_ekip_rows"]):
            last = i == state["_ekip_rows"] - 1
            cols = st.columns(widths, gap="small")
            cols[0].text_input(
                f"{i + 1}. kişi", key=_NAME.format(i), label_visibility="collapsed",
                placeholder="Yeni kişi ekle…" if last else "",
            )
            cols[1].number_input(
                f"{i + 1}. kişi en fazla nöbet", min_value=0, max_value=31, step=1, value=None,
                key=_MAX.format(i), label_visibility="collapsed", placeholder="—", disabled=last,
            )
            cols[2].number_input(
                f"{i + 1}. kişi en fazla hafta sonu", min_value=0, max_value=31, step=1, value=None,
                key=_SPECIAL.format(i), label_visibility="collapsed", placeholder="—", disabled=last,
            )
        st.caption("Boş bırakılan sayı sınırsızdır. Hafta sonuna tatiller de dahildir. "
                   "Kişiyi çıkarmak için ismini silin.")


_FIXED = "w_sabit_gorev::{}"


def render_fixed_roles(isimler, role_names):
    """Görev başına çoklu seçim: seçilen kişinin nöbetleri hep o görevde olur.

    Ayarlarda ``fixed_roles`` ({kişi: görev}) olarak saklanır. Bir kişi tek
    göreve sabitlenebilir; başka bir görevde seçilince oraya taşınır.
    """
    state = st.session_state
    if len(role_names) < 2:
        return
    fixed = dict(state.get("fixed_roles") or {})
    shown = state.get("_sabit_shown") or {}
    changed = False
    # Son çizimden bu yana değişen kutuyu bul ve sabit görevlere uygula.
    for role in role_names:
        key = _FIXED.format(role)
        if key not in state or role not in shown or list(state[key]) == shown[role]:
            continue
        chosen, before = set(state[key]), set(shown[role])
        for person in before - chosen:
            if fixed.get(person) == role:
                fixed.pop(person)
        for person in chosen - before:
            fixed[person] = role
        changed = True
        break
    if changed:
        state.fixed_roles = fixed
        state.fixed_roles_changed = True
    shown = {}
    for role in role_names:
        shown[role] = [p for p in isimler if fixed.get(p) == role]
        state[_FIXED.format(role)] = list(shown[role])
    state["_sabit_shown"] = shown

    with st.container(key="sabit-gorev"):
        st.markdown("**📌 Sabit görev yeri**")
        for role in role_names:
            st.multiselect(
                role, isimler, key=_FIXED.format(role), placeholder="Kimse sabit değil",
            )
        st.caption("Seçilen kişinin nöbetleri hep bu görevde olur. Kimin nöbet tutacağını "
                   "değiştirmez, yalnız görev yerini belirler.")


def _pair_key(a, b):
    return f"w_yasak::{a}::{b}"


# Çapraz tablo kişi sayısının karesi kadar kutu çizer; büyük ekipte tek bir seçim listesi kullanılır.
PAIR_MATRIX_MAX_PEOPLE = 25
_PAIR_LIST = "w_yasak_liste"


def _pair_label(pair):
    return f"{pair[0]} + {pair[1]}"


def _render_pair_list(isimler, text):
    """Büyük ekip için ikililer tek çoklu seçim kutusunda işaretlenir; yeni metni döner."""
    state = st.session_state
    options = {
        _pair_label(tuple(sorted((a, b)))): tuple(sorted((a, b)))
        for i, a in enumerate(isimler) for b in isimler[i + 1:]
    }
    shown = state.get("_yasak_liste_shown")
    if shown is not None and _PAIR_LIST in state and set(state[_PAIR_LIST]) != set(shown):
        chosen = {options[label] for label in state[_PAIR_LIST] if label in options}
        outside = {p for p in parse_forbidden_pairs(text, isimler) if unknown_names(p, isimler)}
        text = forbidden_pairs_to_text(chosen | outside)
    pairs = parse_forbidden_pairs(text, isimler)
    current = [_pair_label(p) for p in sorted(pairs) if p in options.values()]
    state[_PAIR_LIST] = current
    state["_yasak_liste_shown"] = list(current)
    return text, options


def render_pair_matrix(isimler):
    """Birlikte çalışamayanlar: kişi × kişi çapraz tablo, bir tıkla işaretlenir."""
    state = st.session_state
    text = state.get("forbidden_pairs_text", "")
    if len(isimler) > PAIR_MATRIX_MAX_PEOPLE:
        text, options = _render_pair_list(isimler, text)
        if text != state.get("forbidden_pairs_text", ""):
            state.forbidden_pairs_text = text
        with st.container(key="yasak-tablo"):
            st.markdown("**🚫 Birlikte Çalışamayan**")
            st.multiselect(
                "Aynı gün nöbet tutamayan ikililer", list(options), key=_PAIR_LIST,
                placeholder="İkili seçin (ör. Ali + Ayşe)",
            )
            st.caption(f"Ekip {PAIR_MATRIX_MAX_PEOPLE} kişiden büyük olduğu için tablo yerine liste kullanılıyor.")
        return
    shown = state.get("_yasak_shown", set())
    # Tablo simetrik olduğu için her ikili bir kez (üst üçgende) gösterilir.
    _ikililer = [(a, b) for i, a in enumerate(isimler) for b in isimler[i + 1:]]
    # Son çizimden bu yana tıklanan kutuyu bul ve metne uygula.
    for a, b in _ikililer:
        key = _pair_key(a, b)
        if key in state and bool(state[key]) != (tuple(sorted((a, b))) in shown):
            text = toggle_pair_text(text, isimler, (a, b))
            break
    if text != state.get("forbidden_pairs_text", ""):
        state.forbidden_pairs_text = text
    pairs = parse_forbidden_pairs(text, isimler)
    for a, b in _ikililer:
        state[_pair_key(a, b)] = tuple(sorted((a, b))) in pairs
    state["_yasak_shown"] = {pair for pair in pairs}

    with st.container(key="yasak-tablo"):
        st.markdown("**🚫 Birlikte Çalışamayan**")
        if len(isimler) < 2:
            st.caption("Ekipte en az iki kişi olunca burada işaretlenebilir.")
        else:
            # Satırlar ilk kişiden sondan bir öncekine, sütunlar ikinciden sona.
            _sutun_kisiler = isimler[1:]
            widths = [3] + [1] * len(_sutun_kisiler)
            head = st.columns(widths, gap=None, vertical_alignment="bottom")
            for j, b in enumerate(_sutun_kisiler):
                head[j + 1].markdown(
                    f"<div class='yasak-baslik' title='{escape_html(b)}'>{escape_html(b)}</div>",
                    unsafe_allow_html=True,
                )
            for i, a in enumerate(isimler[:-1]):
                cols = st.columns(widths, gap=None, vertical_alignment="center")
                cols[0].markdown(
                    f"<div class='yasak-isim' title='{escape_html(a)}'>{escape_html(a)}</div>",
                    unsafe_allow_html=True,
                )
                for j, b in enumerate(_sutun_kisiler):
                    if j < i:
                        cols[j + 1].markdown("<div class='yasak-kendi'></div>", unsafe_allow_html=True)
                    else:
                        cols[j + 1].checkbox(f"{a} ile {b}", key=_pair_key(a, b), label_visibility="collapsed")
            st.caption("İşaretlenen ikili aynı gün nöbet tutmaz.")
        _outside = [p for p in parse_forbidden_pairs(text, isimler)
                    if unknown_names(p, isimler)]
        if _outside and st.button("Ekipte olmayanların ikililerini sil", key="w_yasak_temizle"):
            state.forbidden_pairs_text = forbidden_pairs_to_text(
                p for p in parse_forbidden_pairs(text, isimler) if p not in _outside
            )
            st.rerun()
