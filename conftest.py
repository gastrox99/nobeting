# -*- coding: utf-8 -*-
from datetime import date

import pytest

import db


def _sabit_tarih():
    return date(2000, 1, 1)


@pytest.fixture(autouse=True, scope="session")
def _gecmis_ay_kilidi_kapali_oturum():
    """Modül kapsamlı fixture'lar (tarayıcı testlerinin tohum verisi) de kilitsiz yazsın."""
    gercek = db._today
    db._today = _sabit_tarih
    yield
    db._today = gercek


@pytest.fixture(autouse=True)
def _gecmis_ay_kilidi_kapali(monkeypatch):
    """Sabit tarihli testler gerçek tarihe göre kilitlenmesin; kilit testleri kendi tarihini verir.

    db modülü bir testte yeniden yüklenirse (test_tazele) sabitleme de yenilenir.
    """
    import sys
    monkeypatch.setattr(sys.modules["db"], "_today", _sabit_tarih)
