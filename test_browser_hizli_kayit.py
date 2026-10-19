"""Hızlı art arda tıklamada yanlış "başka yerde değiştirilmiş" uyarısı çıkmamalı.

Gerçek Streamlit, Chromium ve PostgreSQL ile çalışır; TEST_DATABASE_URL
tanımlı değilse atlanır. Telefonda hücrelere hızlıca dokunmak, kaydın hemen
ardından gelen yeniden çalıştırmayla betiği kesiyordu: veri yazılıyor ama
oturumdaki sürüm güncellenmiyor, sonraki kayıt kendi kaydını "başka yerde
değişmiş" sanıyordu.
"""

import time

# TEST_DATABASE_URL yoksa test_browser_yedek içe aktarılırken bu dosya da atlanır.
from test_browser_yedek import _cell, _open_app, _toggle, app_url  # noqa: F401  (fixture)
from test_browser_nobet import _day_button, _no_exception, _wait_idle

TEAM = "Hızlı Ali, Hızlı Ayşe, Hızlı Can"


def test_hizli_tiklamada_kayit_cakismasi_cikmaz(app_url):  # noqa: F811
    from playwright.sync_api import sync_playwright
    with sync_playwright() as playwright:
        browser, page = _open_app(playwright, app_url, TEAM)
        try:
            _wait_idle(page)
            # Önceki çalışma bitmeden yeni dokunuşlar: her biri betiği yarıda keser.
            for round_ in range(4):
                for day in range(10):
                    _day_button(page, "Hızlı Ali", day).click(no_wait_after=True)
                    time.sleep(0.05 * (round_ + 1))
            _wait_idle(page)
            # Ardından tek tek yapılan değişikliklerin her biri de kaydedilmeli.
            _toggle(page, "Hızlı Ayşe", 0, "✓")
            _toggle(page, "Hızlı Can", 1, "✓")
            _toggle(page, "Hızlı Ayşe", 0, "·")
            text = page.locator("body").inner_text()
            assert "başka bir yerde" not in text
            assert "kaydedilemedi" not in text
            _no_exception(page)
            ali = [_cell(page, "Hızlı Ali", day) for day in range(10)]
            # Sayfa yenilenince son hâl veritabanından aynen gelmeli.
            page.reload()
            _wait_idle(page)
            page.get_by_text("✏️ Nöbet Ata/Kaldır", exact=True).click()
            _wait_idle(page, 1.0)
            assert [_cell(page, "Hızlı Ali", day) for day in range(10)] == ali
            assert _cell(page, "Hızlı Ayşe", 0) == "·"
            assert _cell(page, "Hızlı Can", 1) == "✓"
        finally:
            browser.close()


def _saved_preferences():
    """Girişsiz kullanıcının en son kaydedilen tercih tablosu (kişi -> gün -> kod)."""
    import json
    import os

    from test_browser_yedek import TEST_DB
    env_backup = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = TEST_DB
    try:
        import db
        conn = db._connect()
        with conn, conn.cursor() as cur:
            cur.execute("SELECT pref_json FROM schedules WHERE owner = '' ORDER BY updated_at DESC LIMIT 1")
            row = cur.fetchone()
        conn.close()
    finally:
        if env_backup is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = env_backup
    return json.loads(row[0]) if row and row[0] else {}


def _pref_button(page, person, day_index):
    row = page.locator(".st-key-preference-grid [data-testid='stHorizontalBlock']").filter(
        has=page.locator("b", has_text=person)
    ).first
    return row.locator(":scope > [data-testid='stColumn']").nth(day_index + 1).get_by_role("button")


def _pref_code(person, day_index):
    prefs = _saved_preferences()
    days = list(prefs)
    return prefs[days[day_index]].get(person, 0) if days else 0


def test_tercihler_toplu_kaydedilir(app_url):  # noqa: F811
    from playwright.sync_api import expect, sync_playwright
    person = "Tercih Ali"
    with sync_playwright() as playwright:
        browser, page = _open_app(playwright, app_url, f"{person}, Tercih Ayşe, Tercih Can")
        try:
            _wait_idle(page)
            # Kayıtlı bir başlangıç: çizelgede bir değişiklik hemen kaydedilir.
            _toggle(page, "Tercih Ayşe", 5, "✓")
            page.get_by_text("🎨 Tercih Belirle", exact=True).click()
            _wait_idle(page, 1.0)
            page.get_by_role("button", name="🟩Tercih").click()
            _wait_idle(page, 1.0)
            for day in (0, 1):
                _pref_button(page, person, day).click()
                expect(_pref_button(page, person, day)).to_have_text("T")
            _wait_idle(page, 1.0)
            # Her dokunuşta yazılmaz; kaydedilmedi notu ve Kaydet tuşu görünür.
            expect(page.get_by_text("Tercih değişiklikleri henüz kaydedilmedi")).to_be_visible()
            assert _pref_code(person, 0) == 0
            page.get_by_role("button", name="💾 Tercihleri Kaydet").click()
            _wait_idle(page, 1.0)
            expect(page.get_by_text("Tercih değişiklikleri henüz kaydedilmedi")).to_have_count(0)
            assert (_pref_code(person, 0), _pref_code(person, 1)) == (1, 1)
            # Nöbet Ata moduna geçmek bekleyen tercihleri kaydeder.
            _pref_button(page, person, 2).click()
            expect(_pref_button(page, person, 2)).to_have_text("T")
            _wait_idle(page, 1.0)
            assert _pref_code(person, 2) == 0
            page.get_by_text("✏️ Nöbet Ata/Kaldır", exact=True).click()
            _wait_idle(page, 1.0)
            assert _pref_code(person, 2) == 1
            text = page.locator("body").inner_text()
            assert "başka bir yerde" not in text and "kaydedilemedi" not in text
            _no_exception(page)
        finally:
            browser.close()


def test_kaydedilmemis_tercihler_yenilemede_kaybolmaz(app_url):  # noqa: F811
    from playwright.sync_api import expect, sync_playwright
    person = "Bekleyen Ali"
    with sync_playwright() as playwright:
        browser, page = _open_app(playwright, app_url, f"{person}, Bekleyen Ayşe, Bekleyen Can")
        try:
            _wait_idle(page)
            _toggle(page, "Bekleyen Ayşe", 5, "✓")
            page.get_by_text("🎨 Tercih Belirle", exact=True).click()
            _wait_idle(page, 1.0)
            page.get_by_role("button", name="🟥İzin").click()
            _wait_idle(page, 1.0)
            for day in (0, 1):
                _pref_button(page, person, day).click()
                expect(_pref_button(page, person, day)).to_have_text("İ")
            _wait_idle(page, 1.5)
            expect(page.get_by_text("Tercih değişiklikleri henüz kaydedilmedi")).to_be_visible()
            assert _pref_code(person, 0) == 0
            # Kaydetmeden yenile: işaretler tarayıcıdan geri gelir ve kaydedilir.
            page.reload()
            _wait_idle(page)
            assert (_pref_code(person, 0), _pref_code(person, 1)) == (3, 3)
            page.get_by_text("🎨 Tercih Belirle", exact=True).click()
            _wait_idle(page, 1.0)
            expect(_pref_button(page, person, 0)).to_have_text("İ")
            expect(page.get_by_text("Tercih değişiklikleri henüz kaydedilmedi")).to_have_count(0)
            # Kaydedildikten sonra yenilemek aynı tercihleri bir daha uygulamaz.
            page.get_by_role("button", name="⬜Nötr").click()
            _wait_idle(page, 1.0)
            _pref_button(page, person, 0).click()
            expect(_pref_button(page, person, 0)).to_have_text("·")
            page.get_by_role("button", name="💾 Tercihleri Kaydet").click()
            _wait_idle(page, 1.5)
            page.reload()
            _wait_idle(page)
            assert _pref_code(person, 0) == 0
            text = page.locator("body").inner_text()
            assert "başka bir yerde" not in text and "kaydedilemedi" not in text
            _no_exception(page)
        finally:
            browser.close()
