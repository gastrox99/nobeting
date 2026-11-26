"""Listeyi yayınla: yayınlanmamış ay paylaşım linkinde görünmez (26 Kasım 2026)."""
from pathlib import Path
from unittest.mock import patch

import pytest

import db
from nobet_core import now_tr
from test_hiz import _fakes

OWNER = "sahip@example.com"


def _satir(bugun):
    return {"Tarih": f"15.{bugun.month:02d}.{bugun.year} Pzt", "Nöbetçi": "Ali"}


def _paylasim(yayinli, aylar_extra=()):
    """Paylaşım sayfasını açar, kişiyi seçer; (at, okunan ay listesi) döner."""
    import streamlit
    from streamlit.testing.v1 import AppTest
    bugun = now_tr().date()
    okunan = []

    def schedule(name, year, month, **k):
        okunan.append((year, month))
        return ["Ali", "Ayşe"], None, None, None, [_satir(bugun)]

    streamlit.cache_data.clear()
    streamlit.cache_resource.clear()
    fakes = _fakes(
        resolve_share_token=lambda t, raise_on_error=False: (OWNER, 0, 0),
        load_app_settings=lambda **k: {"team": "Ali, Ayşe"}, load_schedule=schedule,
        list_shared_months=lambda owner, **k: [(bugun.year, bugun.month, None, yayinli), *aylar_extra],
    )
    with patch.dict("os.environ", {"DATABASE_URL": ""}), patch.multiple(db, create=True, **fakes):
        at = AppTest.from_file(str(Path(__file__).with_name("nobet.py")), default_timeout=120)
        at.query_params["paylas"] = "abc123"
        at.run()
        at.selectbox(key="paylasim_kisi").select("Ali").run()
    streamlit.cache_data.clear()
    streamlit.cache_resource.clear()
    assert [e.value for e in at.exception] == []
    return at, okunan


def test_yayinlanmamis_ayda_liste_ve_ucret_gizli_talep_acik():
    at, okunan = _paylasim(False)
    assert any("Bu ayın listesi henüz yayınlanmadı" in i.value for i in at.info)
    assert len(at.dataframe) == 0
    assert not any("Tahmini nöbet ücretim" in e.label for e in at.expander)
    assert not any("Sıradaki nöbetin" in m.value for m in at.markdown)
    assert okunan == []                       # ayın satırları hiç okunmaz
    assert len(at.tabs) == 2                  # Talep Gönder sekmesi çalışmaya devam eder


def test_yayinlanan_ayda_liste_gorunur():
    at, okunan = _paylasim(True)
    assert not any("henüz yayınlanmadı" in i.value for i in at.info)
    assert len(at.dataframe) >= 1
    assert okunan


def test_yayinlanmamis_ay_yanindaki_yayinli_ay_gorunur_kalir():
    bugun = now_tr().date()
    onceki = (bugun.year - 1, 12) if bugun.month == 1 else (bugun.year, bugun.month - 1)
    at, _ = _paylasim(False, [(*onceki, None, True)])
    assert any("Bu ayın listesi henüz yayınlanmadı" in i.value for i in at.info)
    assert len(at.dataframe) >= 1


def test_yil_ozeti_yayinlanmamis_ayi_saymaz(monkeypatch):
    import nobet_paylasim
    istenen = []
    monkeypatch.setattr(nobet_paylasim, "load_auto_saved_assignments",
                        lambda periods, **k: istenen.append(list(periods)) or ({}, {}))
    monkeypatch.setattr(nobet_paylasim, "load_app_settings", lambda **k: {})
    monkeypatch.setattr(nobet_paylasim, "_team_of", lambda owner: ["Ali"])
    nobet_paylasim._my_year(OWNER, 2026, "Ali", ((2026, 9),))
    nobet_paylasim._my_year(OWNER, 2026, "Ali", ())
    assert istenen == [[(2026, 9)], []]


@pytest.mark.parametrize("yayinli", [False, True])
def test_yonetici_yayinla_dugmesi(yayinli):
    from streamlit.testing.v1 import AppTest
    cagrilar = []
    fakes = _fakes(
        is_database_configured=lambda: True,
        get_share_link=lambda owner, y, m, out=None: (out.update({(2026, 10)}) if yayinli and out is not None else None) or "tok",
        set_month_published=lambda *a: cagrilar.append(a) or True,
    )
    with patch.dict("os.environ", {"DATABASE_URL": "", "GIRISSIZ_KULLANIM": "1"}), patch.multiple(db, create=True, **fakes):
        at = AppTest.from_file(str(Path(__file__).with_name("nobet.py")), default_timeout=120)
        at.session_state["isimler_text"] = "Ali, Ayşe, Can, Deniz"
        at.session_state["storage_init"] = {}
        at.session_state["yil_val"], at.session_state["ay_val"] = 2026, 10
        at.run()
        assert [e.value for e in at.exception] == []
        if yayinli:
            at.button(key="yayindan_kaldir").click().run()
            assert cagrilar == [("", 2026, 10, False)]
            assert any("henüz yayınlanmadı" in w.value for w in at.warning)
        else:
            at.button(key="listeyi_yayinla").click().run()
            assert cagrilar == [("", 2026, 10, True)]
            assert any("yayında" in s.value for s in at.success)
