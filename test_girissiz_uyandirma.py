# -*- coding: utf-8 -*-
"""Giriş yapmamış ziyaret veritabanına bağlantı açtırmasın (16 Kasım 2026 incelemesi).

Veritabanı hazırlığı sayfanın en başında, giriş denetiminden önce yapılıyordu:
giriş sayfasını açan herkes (otomatik programlar dahil) uyuyan veritabanını
uyandırıyor, ücretsiz plandaki aylık kullanım hakkını tüketebiliyordu.
"""
from unittest.mock import patch

from test_bozuk_kayit import _app, _stop

_AUTH = {"client_id": "x", "client_secret": "y", "redirect_uri": "http://localhost:8501/oauth2callback"}


def _girisli_site(giris_yapildi, **extra):
    """Google girişi açık site; ``giris_yapildi`` ziyaretçinin giriş yapıp yapmadığı."""
    calls = []
    at, patches = _app(
        lambda *a, **k: (None, None, None, None, None), lambda *a, **k: True,
        init_db=lambda: calls.append("hazirlik") or True,
        ping_db=lambda: calls.append("uyandirma") or True,
        **extra,
    )
    at.secrets["auth"] = _AUTH
    user = patch("streamlit.user_info._get_user_info",
                 return_value={"is_logged_in": giris_yapildi, "email": "ali@example.com"})
    user.start()
    patches.append(user)
    return at, patches, calls


def test_giris_yapmamis_ziyaret_veritabanina_dokunmaz():
    at, patches, calls = _girisli_site(False)
    try:
        at.run()
        assert [e.value for e in at.exception] == []
        dugmeler = [b.label for b in at.button]
    finally:
        _stop(patches)
    assert dugmeler == ["🔐 Google ile giriş yap"]
    assert calls == []


def test_giris_yapan_kullanicida_veritabani_hazirlanir():
    at, patches, calls = _girisli_site(True)
    try:
        at.run()
        assert [e.value for e in at.exception] == []
        uyarilar = [w.value for w in at.warning]
    finally:
        _stop(patches)
    assert calls == ["hazirlik"]
    assert not any("Veritabanına" in u for u in uyarilar)


def test_paylasim_baglantisi_girissiz_de_veritabanini_hazirlar():
    sorulan = []
    at, patches, calls = _girisli_site(
        False, resolve_share_token=lambda token, raise_on_error=False: sorulan.append(token),
    )
    at.query_params["paylas"] = "ornek-baglanti"
    try:
        at.run()
        assert [e.value for e in at.exception] == []
        hatalar = [e.value for e in at.error]
    finally:
        _stop(patches)
    assert calls == ["hazirlik"]
    assert sorulan == ["ornek-baglanti"]
    assert hatalar == ["Bu paylaşım bağlantısı geçersiz ya da kapatılmış."]
