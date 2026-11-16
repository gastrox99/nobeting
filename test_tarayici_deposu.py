# -*- coding: utf-8 -*-
"""Girişli kullanımda ekip adları tarayıcıda kalmasın (16 Kasım 2026 incelemesi).

Ekip adları her değişiklikte tarayıcı deposuna yazılıyordu. Girişli kullanımda bu
kayıt hiç okunmuyor (ekip veritabanında); çıkış yapılınca da silinmediği için
hastanedeki ortak bilgisayarda adlar tarayıcıda kalıyordu.
"""
from unittest.mock import patch

from test_bozuk_kayit import _app, _stop

TEAM = "Ali, Ayşe, Can"
_AUTH = {"client_id": "x", "client_secret": "y", "redirect_uri": "http://localhost:8501/oauth2callback"}


def _site(girisli, depo=None):
    at, patches = _app(lambda *a, **k: (None, None, None, None, None), lambda *a, **k: True, team=TEAM)
    if depo is not None:
        at.session_state["storage_init"] = depo
    if girisli:
        at.secrets["auth"] = _AUTH
        user = patch("streamlit.user_info._get_user_info",
                     return_value={"is_logged_in": True, "email": "ali@example.com"})
        user.start()
        patches.append(user)
    return at, patches


def _calistir(at, patches, ekibe_ekle=None):
    """Sayfayı açar; ``ekibe_ekle`` verilirse ekibe o kişiyi ekleyip yeniden çalıştırır.

    Dönen değer: (tarayıcı deposu, ekip metni)."""
    try:
        at.run()
        assert [e.value for e in at.exception] == []
        if ekibe_ekle:
            at.session_state["isimler_text"] = f"{TEAM}, {ekibe_ekle}"
            at.run()
            assert [e.value for e in at.exception] == []
        return dict(at.session_state["storage_init"]), at.session_state["isimler_text"]
    finally:
        _stop(patches)


def test_girisli_kullanimda_ekip_degisince_adlar_tarayiciya_yazilmaz():
    depo, ekip = _calistir(*_site(girisli=True), ekibe_ekle="Ece")
    assert "nobet_team" not in depo
    assert ekip == f"{TEAM}, Ece"


def test_girisli_kullanimda_onceden_yazilmis_ekip_adlari_silinir():
    depo, ekip = _calistir(*_site(girisli=True, depo={"nobet_team": "Eski, Ekip", "baska": "kalir"}))
    assert depo == {"baska": "kalir"}
    # Tarayıcıdaki eski ekip, giriş yapan kullanıcının ekibinin yerine geçmez.
    assert ekip == TEAM


def test_girissiz_kullanimda_ekip_eskisi_gibi_tarayiciya_yazilir():
    depo, ekip = _calistir(*_site(girisli=False), ekibe_ekle="Ece")
    assert depo.get("nobet_team") == f"{TEAM}, Ece"
    assert ekip == f"{TEAM}, Ece"
