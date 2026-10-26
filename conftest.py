# -*- coding: utf-8 -*-
from datetime import date

import pytest


@pytest.fixture(autouse=True, scope="session")
def _gecmis_ay_kilidi_kapali():
    """Sabit tarihli testler gerçek tarihe göre kilitlenmesin; kilit testleri kendi tarihini verir.

    Oturum kapsamlı: modül kapsamlı fixture'lar (tarayıcı testlerinin tohum verisi) de kilitsiz yazsın.
    """
    import db
    gercek = db._today
    db._today = lambda: date(2000, 1, 1)
    yield
    db._today = gercek
