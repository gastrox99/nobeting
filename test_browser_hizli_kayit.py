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
