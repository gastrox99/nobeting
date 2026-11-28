# -*- coding: utf-8 -*-
"""Ayarlar: ekip tablosu (isim + kişisel limitler), birlikte çalışamayanlar ve
birlikte çalışmak isteyenler çapraz tabloları, sabit görev yerleri.

Hepsi ayarlarda saklanan metinleri (``isimler_text``, ``limits_text``,
``forbidden_pairs_text``, ``together_pairs_text``) düzenler; kayıt, yedek ve Excel aynı metinleri
kullanmaya devam eder.
"""
import calendar
import copy
from datetime import date

import streamlit as st

from nobet_core import (
    compact_team_rows,
    escape_html,
    forbidden_pairs_to_text,
    parse_forbidden_pairs,
    parse_person_limits,
    rename_in_pairs_text,
    team_settings_rows,
    team_settings_texts,
    toggle_pair_text,
    unknown_names,
)

_NAME = "w_ekip_ad_{}"
_MAX = "w_ekip_max_{}"
_SPECIAL = "w_ekip_hs_{}"


def _forget_boxes(prefix, *extra):
    """Çizilmeyen kutuların anahtarlarını tamamen siler (Ayarlar kapanırken).

    Kapalı panelde kutu anahtarına kodla yazılmaz: çizilmeyen kutuya yazılan değer
    Streamlit'te saklı kalır, panel açılınca kullanıcı tıklaması sanılırdı. Silinen
    kutular açılışta ayarlardaki metinden yeniden kurulur.
    """
    state = st.session_state
    for key in [k for k in list(state.keys()) if str(k).startswith(prefix)]:
        state.pop(key, None)
    for key in extra:
        state.pop(key, None)


# *_by_period içinde aya bağlı olmayan "eski aylar için varsayılan" kaydın anahtarı.
_ESKI_AYLAR = "0000-00"

# Aya özel ayarlar: (oturum anahtarı, ayarlardaki aylık anahtar, boş değer).
AYLIK_AYARLAR = (
    ("limits_text", "limits_by_period", ""),
    ("forbidden_pairs_text", "forbidden_by_period", ""),
    ("together_pairs_text", "together_by_period", ""),
    ("fixed_roles", "fixed_roles_by_period", {}),
)


def _kopya(value):
    return dict(value) if isinstance(value, dict) else value


def constraints_for_period(ss, period):
    """Kişi başı sınırlar, yasak çiftler, birlikte çalışacaklar ve sabit görev yerleri aya özeldir.

    Her biri ``*_by_period`` ayarında aya göre saklanır. Ay değişince çıkılan ayınki
    kendi anahtarında kalır, girilen ayın kayıtlısı yüklenir. Kaydı olmayan ay:
    ileri gidilen yeni aya o anki değerler kopyalanır; geriye gidilen eski aya ise
    uygulama ilk kez bu özellikle açıldığı andaki (eski tek) değerler gelir, böylece
    bu ay için yazılan kısıt önceki aya yansımaz. Her çalışmada çağrılır; kısıt
    yazılınca ayın kaydı da güncellenir.
    """
    previous = ss.get("limits_period")
    for state_key, by_key, empty in AYLIK_AYARLAR:
        by_period = ss.get(by_key)
        if not isinstance(by_period, dict):
            by_period = ss[by_key] = {}
        current = ss.get(state_key) or _kopya(empty)
        by_period.setdefault(_ESKI_AYLAR, _kopya(current))
        if previous is not None and previous != period:
            by_period[previous] = _kopya(current)
            if period in by_period:
                ss[state_key] = _kopya(by_period[period])
            elif period < previous:
                ss[state_key] = _kopya(by_period[_ESKI_AYLAR])
        elif previous is None and period in by_period:
            ss[state_key] = _kopya(by_period[period])
        by_period[period] = _kopya(ss.get(state_key) or empty)
    ss["limits_period"] = period


def adopt_period_records(ss, merged):
    """Ayar birleştirmesinden gelen aylık kayıtları (öbür cihazın baktığı aylar dahil) oturuma alır.

    Alınmazsa bu oturumun sonraki kaydı öbür cihazın aylarını silerdi.
    """
    for state_key, by_key, _empty in AYLIK_AYARLAR:
        if by_key in merged:
            ss[by_key] = copy.deepcopy(merged[by_key])
            # Bakılan ayın üst düzey kopyası da eşitlenir; yoksa sonraki çalışma eski kopyayı
            # ayın kaydına yazıp öbür cihazın yazdığını silerdi.
            if ss.get("limits_period") in merged[by_key]:
                ss[state_key] = _kopya(ss[by_key][ss["limits_period"]])


def _sinir_metninde_adlari_degistir(text, renames):
    lines = []
    for line in (text or "").split("\n"):
        name, sep, rest = line.partition(":")
        lines.append(f"{renames.get(name.strip(), name.strip())}{sep}{rest}" if sep else line)
    return "\n".join(lines)


def rename_in_other_periods(ss, old_team, renames, period):
    """Ad değişikliğini diğer ayların kısıtlarına da uygular (bu ayınki zaten güncellendi)."""
    for state_key, by_key, _empty in AYLIK_AYARLAR:
        for key, value in (ss.get(by_key) or {}).items():
            if key == period:
                continue
            if state_key == "limits_text":
                ss[by_key][key] = _sinir_metninde_adlari_degistir(value, renames)
            elif state_key == "fixed_roles":
                ss[by_key][key] = {renames.get(p, p): r for p, r in value.items()}
            else:
                ss[by_key][key] = rename_in_pairs_text(value, old_team, renames)


def _team_names(text):
    return [part.strip() for part in (text or "").split(",") if part.strip()]


def _tam_dagilim(kalan, kisi):
    """Kalanı tam sayılarla böler: 46/5 → "4 kişi 9, 1 kişi 10"; tam bölünürse "kişi başı 9"."""
    taban, fazla = divmod(kalan, kisi)
    if not fazla:
        return f"kişi başı {taban}"
    return f"{kisi - fazla} kişi {taban}, {fazla} kişi {taban + 1}" if taban else f"{fazla} kişi 1, {kisi - fazla} kişi 0"


def ekip_kalan_ozeti(isimler, limits, yil, ay, tatil_gunleri, kisi_sayisi):
    """Sınırı boş bırakılan kişilere kalan nöbet ve hafta sonu/tatil sayısının özeti.

    Ayın toplam ihtiyacından sınırı yazılanların toplamı düşülür, kalan sınırsız
    kişilere bölünür. Basit tahmin: izin ve yasak günleri hesaba katılmaz.
    """
    if not isimler:
        return None
    gun_sayisi = calendar.monthrange(yil, ay)[1]
    ozel_gun = sum(1 for d in range(1, gun_sayisi + 1)
                   if date(yil, ay, d).weekday() >= 5 or d in set(tatil_gunleri))
    satirlar = []
    for baslik, anahtar, ihtiyac in (
        ("Nöbet", "max", gun_sayisi * kisi_sayisi),
        ("Hafta sonu", "max_special", ozel_gun * kisi_sayisi),
    ):
        sinirlar = [limits.get(n, {}).get(anahtar, 999) for n in isimler]
        sinirli = [v for v in sinirlar if v < 999]
        sinirsiz = len(sinirlar) - len(sinirli)
        kalan = max(0, ihtiyac - sum(sinirli))
        if sinirsiz:
            satirlar.append(f"{baslik}: sınırsız {sinirsiz} kişiye kalan {kalan} ({_tam_dagilim(kalan, sinirsiz)})")
        else:
            uyari = " ⚠️ ihtiyaçtan az" if sum(sinirli) < ihtiyac else ""
            satirlar.append(f"{baslik}: herkesin sınırı var (toplam {sum(sinirli)}){uyari}")
    return (f"Ay ihtiyacı: {gun_sayisi * kisi_sayisi} nöbet ({gun_sayisi} gün × {kisi_sayisi} kişi), "
            f"{ozel_gun * kisi_sayisi} hafta sonu/tatil.\n\n" + "\n\n".join(satirlar)
            + "\n\nBasit tahmin: izin ve yasak günleri hesaba katılmadı.")


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


def render_team_table(draw=True):
    """Her satır bir kişi: isim, en fazla nöbet, en fazla hafta sonu (boş = sınırsız).

    Sondaki boş satıra isim yazılınca altına yeni boş satır açılır; ismi
    silinen satır kaldırılır. Bir satırdaki ismi değiştirmek ad değişikliğidir.
    ``draw=False``: kutulardan gelen değerler metne yazılır, tablo çizilmez
    (Ayarlar kapanırken).
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
    if draw and [r["name"] for r in compacted] != [r["name"] for r in rows]:
        _write_rows(compacted)
    names_text, limits_text = team_settings_texts(compacted)
    if names_text != source[0] or limits_text != source[1]:
        # Eski kayıtlarda "Ali,Ayşe" gibi biçimler aynı ekibi gösterir; yeniden yazma.
        if _team_names(names_text) != _team_names(source[0]) or limits_text != source[1]:
            state.isimler_text = names_text
            state.limits_text = limits_text
    state["_ekip_source"] = (state.get("isimler_text", ""), state.get("limits_text", ""))
    if not draw:
        _forget_boxes("w_ekip_", "_ekip_source")
        return

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


def render_fixed_roles(isimler, role_names, draw=True):
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
    if not draw:
        _forget_boxes("w_sabit_gorev::", "_sabit_shown")
        return
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


# İki çapraz tablo aynı kodla çizilir: birlikte çalışamayanlar ve birlikte
# nöbet tutmak isteyenler. Her biri kendi metnini ve kutu anahtarlarını kullanır.
PAIR_TABLES = {
    "yasak": {
        "text": "forbidden_pairs_text",
        "title": "**🚫 Birlikte Çalışamayan**",
        "list_label": "Aynı gün nöbet tutamayan ikililer",
        "caption": "İşaretlenen ikili aynı gün nöbet tutmaz.",
        "box": "{} ile {}",
    },
    "birlikte": {
        "text": "together_pairs_text",
        "title": "**🤝 Birlikte Çalışmak İsteyen**",
        "list_label": "Mümkünse aynı gün nöbet tutacak ikililer",
        "caption": ("İşaretlenen ikili mümkün olduğunca aynı gün nöbet tutar. Kesin kural "
                    "değildir: izin, dinlenme ve hafta sonu/toplam dengesi önce gelir. "
                    "Birlikte çalışamayan olarak da işaretli ikili için yasak geçerlidir."),
        "box": "{} ile {} birlikte",
    },
}


def _pair_key(a, b, kind="yasak"):
    return f"w_{kind}::{a}::{b}"


# Çapraz tablo kişi sayısının karesi kadar kutu çizer; büyük ekipte tek bir seçim listesi kullanılır.
PAIR_MATRIX_MAX_PEOPLE = 25


def _pair_list_key(kind):
    return f"w_{kind}_liste"


def _pair_label(pair):
    return f"{pair[0]} + {pair[1]}"


def _render_pair_list(isimler, text, kind="yasak", draw=True):
    """Büyük ekip için ikililer tek çoklu seçim kutusunda işaretlenir; yeni metni döner."""
    state = st.session_state
    list_key = _pair_list_key(kind)
    shown_key = f"_{kind}_liste_shown"
    options = {
        _pair_label(tuple(sorted((a, b)))): tuple(sorted((a, b)))
        for i, a in enumerate(isimler) for b in isimler[i + 1:]
    }
    shown = state.get(shown_key)
    if shown is not None and list_key in state and set(state[list_key]) != set(shown):
        chosen = {options[label] for label in state[list_key] if label in options}
        outside = {p for p in parse_forbidden_pairs(text, isimler) if unknown_names(p, isimler)}
        text = forbidden_pairs_to_text(chosen | outside)
    if not draw:
        _forget_boxes(list_key, shown_key)
        return text, options
    pairs = parse_forbidden_pairs(text, isimler)
    current = [_pair_label(p) for p in sorted(pairs) if p in options.values()]
    state[list_key] = current
    state[shown_key] = list(current)
    return text, options


def render_pair_matrix(isimler, kind="yasak", draw=True):
    """Kişi × kişi çapraz tablo, bir tıkla işaretlenir (``kind``: yasak / birlikte).

    ``draw=False``: son tıklama metne işlenir, tablo çizilmez (Ayarlar kapanırken).
    """
    state = st.session_state
    table = PAIR_TABLES[kind]
    text_key = table["text"]
    text = state.get(text_key, "")
    if len(isimler) > PAIR_MATRIX_MAX_PEOPLE:
        text, options = _render_pair_list(isimler, text, kind, draw)
        if text != state.get(text_key, ""):
            state[text_key] = text
        if not draw:
            return
        with st.container(key=f"{kind}-tablo"):
            st.markdown(table["title"])
            st.multiselect(
                table["list_label"], list(options), key=_pair_list_key(kind),
                placeholder="İkili seçin (ör. Ali + Ayşe)",
            )
            st.caption(f"Ekip {PAIR_MATRIX_MAX_PEOPLE} kişiden büyük olduğu için tablo yerine liste kullanılıyor.")
        return
    shown_key = f"_{kind}_shown"
    shown = state.get(shown_key, set())
    # Tablo simetrik olduğu için her ikili bir kez (üst üçgende) gösterilir.
    _ikililer = [(a, b) for i, a in enumerate(isimler) for b in isimler[i + 1:]]
    # Son çizimden bu yana tıklanan kutuyu bul ve metne uygula.
    for a, b in _ikililer:
        key = _pair_key(a, b, kind)
        if shown_key in state and key in state and bool(state[key]) != (tuple(sorted((a, b))) in shown):
            text = toggle_pair_text(text, isimler, (a, b))
            break
    if text != state.get(text_key, ""):
        state[text_key] = text
    if not draw:
        _forget_boxes(f"w_{kind}::", shown_key)
        return
    pairs = parse_forbidden_pairs(text, isimler)
    for a, b in _ikililer:
        state[_pair_key(a, b, kind)] = tuple(sorted((a, b))) in pairs
    state[shown_key] = {pair for pair in pairs}

    with st.container(key=f"{kind}-tablo"):
        st.markdown(table["title"])
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
                        cols[j + 1].checkbox(table["box"].format(a, b), key=_pair_key(a, b, kind), label_visibility="collapsed")
            st.caption(table["caption"])
        _outside = [p for p in parse_forbidden_pairs(text, isimler)
                    if unknown_names(p, isimler)]
        if _outside and st.button("Ekipte olmayanların ikililerini sil", key=f"w_{kind}_temizle"):
            state[text_key] = forbidden_pairs_to_text(
                p for p in parse_forbidden_pairs(text, isimler) if p not in _outside
            )
            st.rerun()
