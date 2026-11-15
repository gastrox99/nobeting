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

# Tablo üst çizgisinin talep kapsayıcısının üstünden uzaklığı: talepler dolarken değişmemeli.
_TABLO_Y = "(()=>{const c=document.querySelector('.st-key-talep-yer'),t=document.querySelector('[role=tablist]');return c&&t&&getComputedStyle(c).minHeight==='56px'?t.getBoundingClientRect().top-c.getBoundingClientRect().top:-1})()"


def test_talepler_yuklenince_tablo_kaymaz(app_url):  # noqa: F811
    today = now_tr().date()
    year, month = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
    _with_db(_clear_requests)
    try:
        with sync_playwright() as p:
            browser, page = _open_app(p, app_url, "Kayma Ali, Kayma Ayşe")
            try:
                _wait_idle(page)
                # GitHub'daki yavaş makineyi taklit et: ara durumlar uzar, kayma yakalanır.
                page.context.new_cdp_session(page).send("Emulation.setCPUThrottlingRate", {"rate": 6})
                page.reload()
                page.evaluate("window.__y=[];(function f(){const y=" + _TABLO_Y + ";if(y>=0)window.__y.push([y,[...document.querySelectorAll('summary')].some(e=>e.textContent.includes('Talep durumu'))]);requestAnimationFrame(f)})()")
                page.get_by_text("📋 Talep durumu").wait_for()
                _wait_idle(page)
                ys = page.evaluate("window.__y")
                assert any(v for _, v in ys), "talep durumu hiç görünmedi"
                sonra = [y for y, _ in ys]
                assert max(sonra) - min(sonra) <= 6, (min(sonra), max(sonra))
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
