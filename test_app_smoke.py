"""Uygulamanın ana akışlarını gerçek Streamlit ile hata vermeden çalıştırır.

Run with: python -m pytest test_app_smoke.py -v
"""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "nobet.py"), default_timeout=120)
    at.run()
    assert not at.exception
    return at


def _set_team(at, names):
    at.text_area[0].set_value(", ".join(names)).run()
    assert not at.exception, at.exception


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_simulasyon_takvim_ve_analiz_hatasiz(app):
    _set_team(app, ["Ali", "Ayşe", "Can", "Deniz", "Ece"])
    _button(app, "⚡ Simülasyon").click().run()
    assert not app.exception, app.exception
    # Takvimde tek kişi seçilince kişi kartı çizilir.
    app.multiselect(key="cal_person_filter").set_value(["Ali"]).run()
    assert not app.exception, app.exception


def test_ayni_gorev_adi_numaralanir(app):
    _set_team(app, ["Ali", "Ayşe", "Can"])
    role_box = next(t for t in app.text_input if t.label == "🏷️ Görev İsimleri:")
    role_box.set_value("AYB, AYB").run()
    assert not app.exception, app.exception
    assert any("AYB → AYB 2" in c.value for c in app.caption)


def test_buyuk_ekip_hizli_duzenleme_modu(app):
    _set_team(app, [f"Kişi {i}" for i in range(20)])
    assert any("hızlı düzenleme" in c.value for c in app.caption)


def test_uzak_yil_secilebilir(app):
    _set_team(app, ["Ali", "Ayşe", "Can"])
    app.number_input(key="yil_input_direct").set_value(2031).run()
    assert not app.exception, app.exception
