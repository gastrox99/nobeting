"""Ekran kayması tarayıcı testi: talepler yüklenince ve tercih notu çıkınca tablo/düğmeler yer değiştirmez.

TEST_DATABASE_URL tanımlı değilse atlanır (bkz. test_browser_yedek.py).
"""

import time

import pytest

from test_browser_yedek import TEST_DB  # noqa: F401  (tanımsızsa modül atlanır)

from playwright.sync_api import expect, sync_playwright  # noqa: E402

from nobet_core import now_tr  # noqa: E402
from test_browser_nobet import _no_exception, _wait_idle  # noqa: E402
from test_browser_talepler import _clear_requests, _with_db  # noqa: E402
from test_browser_yedek import _open_app, app_url  # noqa: E402,F401  (modül fikstürü)

_TABLO_Y = "(document.querySelector('[role=\"tablist\"]')||{getBoundingClientRect:()=>({top:-1})}).getBoundingClientRect().top"


def test_talepler_yuklenince_tablo_kaymaz(app_url):  # noqa: F811
    today = now_tr().date()
    year, month = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
    _with_db(_clear_requests)
    try:
        with sync_playwright() as p:
            browser, page = _open_app(p, app_url, "Kayma Ali, Kayma Ayşe")
            try:
                _wait_idle(page)
                page.reload()
                page.evaluate("window.__y=[];(function f(){const y=" + _TABLO_Y + ";if(y>0)window.__y.push([y,[...document.querySelectorAll('summary')].some(e=>e.textContent.includes('Talep durumu'))]);requestAnimationFrame(f)})()")
                page.get_by_text("📋 Talep durumu").wait_for()
                _wait_idle(page)
                ys = page.evaluate("window.__y")
                i = next(k for k, (_, var) in enumerate(ys) if var)
                assert i > 0, "talep durumu tablodan önce geldi; ölçülemedi"
                assert abs(ys[i][0] - ys[i - 1][0]) <= 6, (ys[i - 1][0], ys[i][0])
            finally:
                browser.close()
    finally:
        _with_db(_clear_requests)


def test_tercih_notu_cikinca_alt_dugmeler_kaymaz(app_url):  # noqa: F811
    with sync_playwright() as p:
        browser, page = _open_app(p, app_url, "Not Ali, Not Ayşe")
        try:
            _wait_idle(page)
            page.get_by_text("🎨 Tercih Belirle", exact=True).click()
            _wait_idle(page, 1.0)
            page.get_by_role("button", name="🟥İzin").click()
            _wait_idle(page, 1.0)
            alt = page.locator(".st-key-bottom-actions")
            tabs = page.locator(".st-key-kayit-notu")
            once = alt.bounding_box()["y"] - tabs.bounding_box()["y"]
            row = page.locator(".st-key-preference-grid [data-testid='stHorizontalBlock']").filter(
                has=page.locator("b", has_text="Not Ayşe")).first
            button = row.locator(":scope > [data-testid='stColumn']").nth(1).get_by_role("button")
            button.click()
            expect(page.get_by_text("Tercih değişiklikleri henüz kaydedilmedi")).to_be_visible()
            sonra = alt.bounding_box()["y"] - tabs.bounding_box()["y"]
            assert abs(sonra - once) <= 4, (once, sonra)
            _no_exception(page)
        finally:
            browser.close()
