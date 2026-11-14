"""Hızlı art arda tıklamada yanlış "başka yerde değiştirilmiş" uyarısı çıkmamalı.

Gerçek Streamlit, Chromium ve PostgreSQL ile çalışır; TEST_DATABASE_URL
tanımlı değilse atlanır. Telefonda hücrelere hızlıca dokunmak, kaydın hemen
ardından gelen yeniden çalıştırmayla betiği kesiyordu: veri yazılıyor ama
oturumdaki sürüm güncellenmiyor, sonraki kayıt kendi kaydını "başka yerde
değişmiş" sanıyordu.
"""

import os
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import pytest

# TEST_DATABASE_URL yoksa test_browser_yedek içe aktarılırken bu dosya da atlanır.
from test_browser_yedek import _cell, _open_app, _toggle, app_url  # noqa: F401  (fixture)
from test_browser_nobet import APP_DIR, _day_button, _no_exception, _wait_idle

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


def _start_slow_db_proxy(target_url, one_way_ms):
    """Veritabanına giden her pakete gecikme ekleyen yerel aracı; yeni adresi döner.

    Canlıda veritabanı uzakta (Frankfurt); kayıt yaklaşık bir saniye sürüyor.
    Yereldeki anlık veritabanında hızlı dokunuş kaydın ortasına denk gelmiyordu.
    """
    import asyncio
    import threading
    from urllib.parse import urlsplit

    target = urlsplit(target_url)
    delay = one_way_ms / 1000
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    async def pipe(reader, writer):
        queue = asyncio.Queue()

        async def read():
            while True:
                data = await reader.read(65536)
                await queue.put((time.monotonic() + delay, data))
                if not data:
                    return

        async def write():
            while True:
                due, data = await queue.get()
                await asyncio.sleep(max(0, due - time.monotonic()))
                if not data:
                    writer.close()
                    return
                writer.write(data)
                await writer.drain()

        await asyncio.gather(read(), write(), return_exceptions=True)

    async def handle(client_reader, client_writer):
        server_reader, server_writer = await asyncio.open_connection(target.hostname, target.port or 5432)
        await asyncio.gather(pipe(client_reader, server_writer), pipe(server_reader, client_writer),
                             return_exceptions=True)

    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(asyncio.start_server(handle, "127.0.0.1", port))
        ready.set()
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    ready.wait(5)
    return target._replace(netloc=f"{target.netloc.rsplit('@', 1)[0]}@127.0.0.1:{port}").geturl()


@pytest.fixture(scope="module")
def slow_db_app_url():
    from test_browser_yedek import TEST_DB, _clear_anonymous_data
    _clear_anonymous_data()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "DATABASE_URL": _start_slow_db_proxy(TEST_DB, 75), "GIRISSIZ_KULLANIM": "1"}
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "nobet.py", "--server.port", str(port),
         "--server.address", "127.0.0.1", "--server.headless", "true", "--browser.gatherUsageStats", "false"],
        cwd=APP_DIR, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                with urlopen(f"{url}/_stcore/health", timeout=1):
                    break
            except (URLError, TimeoutError, ConnectionError):
                time.sleep(0.2)
        else:
            raise TimeoutError("Streamlit did not become ready")
        yield url
    finally:
        server.terminate()
        server.wait(timeout=10)
        _clear_anonymous_data()


def test_kayit_surerken_dokunmak_cakisma_uyarisi_vermez(slow_db_app_url):
    """Kayıt sürerken gelen dokunuş yeni bir çalıştırma başlatır; ikisi aynı sürümle yazıp çakışmamalı.

    Streamlit'in "hızlı yeniden çalıştırma" ayarı açıkken yeni dokunuş, önceki
    çalıştırma veritabanına yazarken ikinci bir çalıştırmayı hemen başlatıyordu.
    İkincisi ilkinin henüz güncellemediği sürümle yazıp kendi kaydını "başka bir
    yerde değiştirilmiş" sanıyordu (uyarıdaki iki oturum numarası aynıydı).
    """
    from playwright.sync_api import sync_playwright
    with sync_playwright() as playwright:
        browser, page = _open_app(playwright, slow_db_app_url, "Yarış Ali, Yarış Ayşe, Yarış Can")
        try:
            _wait_idle(page)
            _toggle(page, "Yarış Ayşe", 5, "✓")
            for day in range(3):
                _day_button(page, "Yarış Ali", day).click(no_wait_after=True)
                time.sleep(0.3)  # ilk kayıt veritabanında sürerken
                _day_button(page, "Yarış Can", day).click(no_wait_after=True)
                _wait_idle(page, 1.0)
                text = page.locator("body").inner_text()
                assert "başka bir yerde" not in text, f"{day}. turda sahte çakışma"
            assert "kaydedilemedi" not in text
            _no_exception(page)
            page.reload()
            _wait_idle(page)
            page.get_by_text("✏️ Nöbet Ata/Kaldır", exact=True).click()
            _wait_idle(page, 1.0)
            # Hepsi kaydedilmiş olmalı (ardışık günler kural işaretiyle görünebilir).
            for person in ("Yarış Ali", "Yarış Can"):
                assert "·" not in [_cell(page, person, day) for day in range(3)]
        finally:
            browser.close()
