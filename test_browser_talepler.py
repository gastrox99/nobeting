"""Ekip talepleri tarayıcı testi: onay tercihlere işlenir ve kaydedilir, ret siler.

TEST_DATABASE_URL tanımlı değilse atlanır. Uygulama girişsiz çalışır ve
girişsiz kullanıcının ("") kayıtlarını siler; gerçek veritabanının adresini
buraya vermeyin.
"""

import os
import time

import pytest

TEST_DB = os.environ.get("TEST_DATABASE_URL")
if not TEST_DB:
    pytest.skip("TEST_DATABASE_URL tanımlı değil", allow_module_level=True)

from playwright.sync_api import expect, sync_playwright  # noqa: E402

from nobet_core import now_tr  # noqa: E402
from test_browser_nobet import _no_exception, _wait_idle  # noqa: E402
from test_browser_yedek import _open_app, app_url  # noqa: E402,F401  (modül fikstürü)

ALI, AYSE = "Talep Ali", "Talep Ayşe"


def _with_db(action):
    env_backup = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = TEST_DB
    try:
        import db
        assert db.init_db(), db.get_last_error()
        return action(db)
    finally:
        if env_backup is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = env_backup


def _clear_requests(db):
    conn = db._connect()
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM team_requests WHERE owner = ''")
    conn.close()


def _saved_pref(db, year, month, person):
    loaded = db.load_schedule(f"Otomatik_{year}_{month:02d}", year, month, owner="")
    if not loaded or loaded[2] is None or person not in loaded[2].index:
        return None
    return {int(str(col).split()[0]): int(v) for col, v in loaded[2].loc[person].items()}


def test_talep_onaylanir_ve_reddedilir(app_url):  # noqa: F811
    today = now_tr().date()
    # Uygulama gelecek ayla açılır.
    year, month = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
    _with_db(_clear_requests)
    _with_db(lambda db: db.submit_team_request("", year, month, ALI, {"tercih": [3], "izin": [5, 6]}))
    _with_db(lambda db: db.submit_team_request("", year, month, AYSE, {"kacin": [9]}))
    try:
        with sync_playwright() as p:
            browser, page = _open_app(p, app_url, f"{ALI}, {AYSE}")
            try:
                _wait_idle(page)
                expect(page.get_by_text("📥 Ekip talepleri (2)")).to_be_visible()
                expect(page.get_by_text("Tercih: 3 · İzin: 5-6")).to_be_visible()
                expect(page.get_by_text("🔔 2 ekip talebi onay bekliyor")).to_be_visible()
                page.get_by_role("button", name="✅ Onayla").first.click()
                _wait_idle(page)
                _no_exception(page)
                expect(page.get_by_text("📥 Ekip talepleri (1)")).to_be_visible()
                expect(page.get_by_text("🔔 1 ekip talebi onay bekliyor")).to_be_visible()
                deadline = time.monotonic() + 10
                pref = None
                while time.monotonic() < deadline:
                    pref = _with_db(lambda db: _saved_pref(db, year, month, ALI))
                    if pref and pref.get(3) == 1:
                        break
                    time.sleep(0.5)
                assert pref and (pref[3], pref[5], pref[6], pref[4]) == (1, 3, 3, 0)
                assert [r["person"] for r in _with_db(lambda db: db.list_team_requests(""))] == [AYSE]

                page.get_by_role("button", name="❌ Reddet").click()
                _wait_idle(page)
                _no_exception(page)
                expect(page.get_by_text("📥 Ekip talepleri")).to_have_count(0)
                expect(page.get_by_text("🔔")).to_have_count(0)
                assert _with_db(lambda db: db.list_team_requests("")) == []
                # Reddedilen talep tercihlere işlenmez.
                assert _with_db(lambda db: _saved_pref(db, year, month, AYSE)).get(9) == 0
            finally:
                browser.close()
    finally:
        _with_db(_clear_requests)


def _bump_schedule(db, year, month):
    """Ayı başka bir yerden (ör. telefondan) kaydedilmiş gibi yapar."""
    import pandas as pd
    frame = pd.DataFrame(False, index=[ALI, AYSE], columns=["1 Per"])
    assert db.save_schedule(f"Otomatik_{year}_{month:02d}", year, month, [ALI, AYSE], frame, owner="", expected_version=db.FORCE)


def test_kaydedilemeyen_onayda_talep_kaybolmaz(app_url):  # noqa: F811
    today = now_tr().date()
    year, month = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
    _with_db(_clear_requests)
    _with_db(lambda db: db.submit_team_request("", year, month, ALI, {"izin": [7]}))
    try:
        with sync_playwright() as p:
            browser, page = _open_app(p, app_url, f"{ALI}, {AYSE}")
            try:
                _wait_idle(page)
                expect(page.get_by_text("📥 Ekip talepleri (1)")).to_be_visible()
                # Sayfa açıkken ay başka yerde kaydedilir; onay kaydı çakışmaya düşer.
                _with_db(lambda db: _bump_schedule(db, year, month))
                page.get_by_role("button", name="✅ Onayla").click()
                _wait_idle(page)
                _no_exception(page)
                expect(page.get_by_text("başka bir yerde", exact=False).first).to_be_visible()
                # Talep silinmedi; listede yeniden görünür.
                assert [r["person"] for r in _with_db(lambda db: db.list_team_requests(""))] == [ALI]
                expect(page.get_by_text("📥 Ekip talepleri (1)")).to_be_visible()
            finally:
                browser.close()
    finally:
        _with_db(_clear_requests)
