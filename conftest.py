# -*- coding: utf-8 -*-
from datetime import date

import pytest


@pytest.fixture(autouse=True)
def _gecmis_ay_kilidi_kapali(monkeypatch):
    """Sabit tarihli testler gerçek tarihe göre kilitlenmesin; kilit testleri kendi tarihini verir."""
    import db
    monkeypatch.setattr(db, "_today", lambda: date(2000, 1, 1))
