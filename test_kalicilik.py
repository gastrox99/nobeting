# -*- coding: utf-8 -*-
"""Otomatik kaydın durdurulup yeniden açılması (nobet_kalicilik.py) ve çakışma akışı."""
import db
from nobet_kalicilik import (
    engeli_kaldir,
    engelleri_sifirla,
    kaydi_durdur,
    kayit_durdu_mu,
)
from test_bozuk_kayit import _app, _flip_cell, _october_columns, _stop

EKIM = (2026, 10)


def test_durdurma_nedeniyle_yazilir_yalniz_ayni_nedenle_kalkar():
    ss = {}
    kaydi_durdur(ss, [2026, 10], "bozuk")
    assert kayit_durdu_mu(ss, EKIM) and not kayit_durdu_mu(ss, (2026, 11))
    assert ss["autosave_block_reason"] == "bozuk"
    assert "autosave_conflict_period" not in ss          # bozuk ay için ay değişiminde uyarı yok
    assert engeli_kaldir(ss, EKIM, "kilitli") is False    # kilit düğmesi bozuk ayı açamaz
    assert kayit_durdu_mu(ss, EKIM)
    assert engeli_kaldir(ss, EKIM, "bozuk") is True
    assert not kayit_durdu_mu(ss, EKIM) and ss["autosave_block_reason"] is None


def test_cakisma_ve_surum_bilinmiyor_ay_degisiminde_uyari_turunu_yazar():
    for neden, tur in (("cakisma", "conflict"), ("surum_bilinmiyor", "version_unknown")):
        ss = {}
        kaydi_durdur(ss, EKIM, neden)
        assert ss["autosave_conflict_period"] == EKIM and ss["autosave_conflict_kind"] == tur
        engelleri_sifirla(ss)
        assert all(value is None for value in ss.values())


def test_uygulama_otomatik_kayitta_cakisma_uyarir_ve_bir_daha_yazmaz():
    calls = []

    def save(*args, **kwargs):
        calls.append(kwargs.get("expected_version"))
        raise db.SaveConflict("x")

    def load(name, year, month, version_out=None, **kwargs):
        if version_out is not None:
            version_out["version"] = "v1"
        return None, None, None, None, None

    at, patches = _app(load, save, team="Ali, Ayşe")
    try:
        at.run()
        columns = list(at.session_state["schedule_bool"].columns)
        assert columns == _october_columns()
        _flip_cell(at)
        at.run()
        assert [e.value for e in at.exception] == []
        assert at.session_state["autosave_blocked_period"] == EKIM
        assert at.session_state["autosave_block_reason"] == "cakisma"
        assert at.session_state["autosave_conflict_kind"] == "conflict"
        assert any("başka bir yerde" in m.value for m in list(at.warning) + list(at.error))
        _flip_cell(at, col=2)
        at.run()
    finally:
        _stop(patches)
    assert calls == ["v1"]          # durdurulduktan sonra ikinci kez yazmayı denemedi


def test_uygulama_kilitli_ayda_duran_kayit_kilit_acilinca_yeniden_yazilir(monkeypatch):
    from datetime import date
    import sys
    monkeypatch.setattr(sys.modules["db"], "_today", lambda: date(2026, 10, 27))
    calls = []

    def save(*args, **kwargs):
        calls.append(args[1:3])
        if len(calls) == 1:
            raise db.MonthLocked(2026, 8)
        return True

    def load(name, year, month, version_out=None, **kwargs):
        if version_out is not None:
            version_out["version"] = "v1"
        return None, None, None, None, None

    at, patches = _app(load, save, team="Ali, Ayşe")
    at.session_state["yil_val"], at.session_state["ay_val"] = 2026, 8
    try:
        at.run()
        _flip_cell(at)
        at.run()
        assert at.session_state["autosave_block_reason"] == "kilitli"
        next(b for b in at.button if b.key == "month_lock_toggle").click().run()
        assert at.session_state["autosave_blocked_period"] is None
        _flip_cell(at, col=2)
        at.run()
        assert [e.value for e in at.exception] == []
    finally:
        _stop(patches)
    # İlki kilit yüzünden reddedildi; kilit açılınca bekleyen değişiklik ve sonraki yazıldı.
    assert calls == [(2026, 8)] * 3
