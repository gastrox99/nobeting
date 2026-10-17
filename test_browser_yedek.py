"""Yedek bölümü tarayıcı testi: gerçek Streamlit, Chromium ve PostgreSQL ile.

TEST_DATABASE_URL tanımlı değilse atlanır (GitHub'daki "Testler" iş akışı
geçici bir PostgreSQL açıp bu değişkeni verir). Uygulama girişsiz çalışır ve
girişsiz kullanıcının ("") kayıtlarını siler; gerçek veritabanının adresini
buraya vermeyin.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import pytest

TEST_DB = os.environ.get("TEST_DATABASE_URL")
if not TEST_DB:
    pytest.skip("TEST_DATABASE_URL tanımlı değil", allow_module_level=True)

from playwright.sync_api import expect, sync_playwright  # noqa: E402

from test_browser_nobet import APP_DIR, _day_button, _no_exception, _wait_idle  # noqa: E402

ALI, AYSE = "Yedek Ali", "Yedek Ayşe"


def _clear_anonymous_data():
    env_backup = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = TEST_DB
    try:
        import db
        assert db.init_db(), db.get_last_error()
        conn = db._connect()
        with conn, conn.cursor() as cur:
            cur.execute("DELETE FROM schedules WHERE owner = ''")
            cur.execute("DELETE FROM backup_snapshots WHERE owner = ''")
            cur.execute("DELETE FROM app_settings WHERE key = %s", (db.settings_key(""),))
            cur.execute("DELETE FROM share_links WHERE owner = ''")
        conn.close()
    finally:
        if env_backup is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = env_backup


@pytest.fixture(scope="module")
def app_url():
    _clear_anonymous_data()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "DATABASE_URL": TEST_DB, "GIRISSIZ_KULLANIM": "1"}
    server = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "nobet.py",
            "--server.port", str(port), "--server.address", "127.0.0.1",
            "--server.headless", "true", "--browser.gatherUsageStats", "false",
        ],
        cwd=APP_DIR, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"Streamlit exited: {server.stderr.read().decode()[-3000:]}")
            try:
                with urlopen(f"{url}/_stcore/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError, ConnectionError):
                time.sleep(0.2)
        else:
            raise TimeoutError("Streamlit did not become ready")
        yield url
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
        server.stderr.close()
        _clear_anonymous_data()


def _launch(playwright):
    return playwright.chromium.launch(
        headless=True, executable_path=os.environ.get("CHROMIUM_EXECUTABLE") or shutil.which("chromium"),
        args=["--no-sandbox"],
    )


def _open_app(playwright, app_url, team):
    """Sayfa tamamen yüklendikten sonra ekibi girer.

    Ayarlar paneli sayfa yüklenirken henüz görünmeyebilir; o anda başlığa
    tıklamak, açık gelecek paneli kapatırdı.
    """
    browser = _launch(playwright)
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.goto(app_url)
    page.get_by_text("⚙️ Ayarlar").wait_for()
    _wait_idle(page)
    box = page.get_by_role("textbox", name="👥 Ekip (virgülle ayırın):")
    if not box.is_visible():
        page.get_by_text("⚙️ Ayarlar").click()
    box.fill(team)
    box.press("Tab")
    page.get_by_text("✏️ Nöbet Ata/Kaldır", exact=True).click()
    return browser, page


def _cell(page, person, day_index):
    return _day_button(page, person, day_index).inner_text().strip()


def _toggle(page, person, day_index, expected):
    _day_button(page, person, day_index).click()
    expect(_day_button(page, person, day_index)).to_have_text(expected)
    _wait_idle(page, 1.5)


def _open_backup_section(page):
    """Yedek bölümünü açar (başlığa hatırlatma metni eklenebilir)."""
    section = page.locator("[data-testid='stExpander']").filter(
        has=page.locator("summary", has_text="💾 Yedek")
    ).first
    section.scroll_into_view_if_needed()
    if section.locator("details").get_attribute("open") is None:
        section.locator("summary").click()
    section.get_by_text("Uygulamadaki yedekler").wait_for()
    return section


def _restore_selected(page, section, option_text):
    select = section.get_by_role("combobox", name="Yedek seçin")
    select.click()
    page.get_by_role("option").filter(has_text=option_text).first.click()
    _wait_idle(page, 1.0)
    # Seçim sayfayı yeniden çizer; bölüm kapanmış olabilir.
    section = _open_backup_section(page)
    section.get_by_text("Mevcut kayıtların üzerine yazılsın").first.click()
    restore = section.get_by_role("button", name="♻️ Bu yedeğe dön")
    expect(restore).to_be_enabled()
    restore.click()
    expect(page.get_by_text("✅ Yedek geri yüklendi")).to_be_visible(timeout=20000)
    _wait_idle(page)
    page.get_by_text("✏️ Nöbet Ata/Kaldır", exact=True).click()
    _wait_idle(page, 1.0)


def test_yedekle_ve_geri_yukle(app_url, tmp_path):
    with sync_playwright() as playwright:
        browser, page = _open_app(playwright, app_url, f"{ALI}, {AYSE}")
        try:
            _wait_idle(page)
            assert _cell(page, ALI, 0) == "·"
            _toggle(page, ALI, 0, "✓")

            # 1) Listenin üstündeki 💾 Yedekle: bu ayın tek yedeği alınır.
            page.get_by_role("button", name="💾 Yedekle").click()
            expect(page.get_by_text("yedeklendi; bu ayın önceki yedeğinin yerine geçti").first).to_be_visible()
            _wait_idle(page)
            _no_exception(page)

            # Yedekten sonra çizelge değişir.
            _toggle(page, ALI, 0, "·")
            _toggle(page, AYSE, 1, "✓")

            # 2) Ay yedeğine dön: yedek anındaki hâl geri gelir.
            section = _open_backup_section(page)
            expect(section.get_by_text("Bu ayın son yedeği:")).to_be_visible()
            _restore_selected(page, section, "yedeklendi")
            _no_exception(page)
            assert _cell(page, ALI, 0) == "✓"
            assert _cell(page, AYSE, 1) == "·"

            # Sayfa yenilenince de aynı kalır (geri yükleme kaydedildi).
            page.reload()
            _wait_idle(page)
            page.get_by_text("✏️ Nöbet Ata/Kaldır", exact=True).click()
            _wait_idle(page, 1.0)
            assert _cell(page, ALI, 0) == "✓"
            assert _cell(page, AYSE, 1) == "·"

            # 3) "Son geri yüklemeden önceki hâl": yanlış geri yükleme geri alınır.
            section = _open_backup_section(page)
            _restore_selected(page, section, "Son geri yüklemeden önceki hâl")
            assert _cell(page, ALI, 0) == "·"
            assert _cell(page, AYSE, 1) == "✓"

            # 4) Tüm veriyi bilgisayara indir, değiştir, dosyadan geri yükle.
            section = _open_backup_section(page)
            section.get_by_role("button", name="📦 Yedeği hazırla").click()
            _wait_idle(page, 1.0)
            section = _open_backup_section(page)
            download_button = section.get_by_role("button", name="⬇️ Yedeği indir (.json)")
            download_button.wait_for()
            with page.expect_download() as download_info:
                download_button.click()
            path = tmp_path / download_info.value.suggested_filename
            download_info.value.save_as(path)
            assert path.name.startswith("nobet_yedek_") and path.suffix == ".json"
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["format"] == "nobeting-yedek"
            assert data["settings"] is not None
            (month_item,) = data["schedules"]
            assert month_item["team"] == [ALI, AYSE]
            assert [ALI, month_item["days"][0]] not in month_item["assigned"]
            assert [AYSE, month_item["days"][1]] in month_item["assigned"]
            _wait_idle(page)
            # İndirme tarihi hatırlatmayı kapatır.
            section = _open_backup_section(page)
            expect(section.get_by_text("Son indirme:")).to_be_visible()

            page.get_by_text("✏️ Nöbet Ata/Kaldır", exact=True).click()
            _toggle(page, AYSE, 1, "·")
            _toggle(page, ALI, 2, "✓")

            section = _open_backup_section(page)
            section.locator("input[type='file']").set_input_files(str(path))
            expect(section.get_by_text("Yedek: 1 ay")).to_be_visible()
            section.get_by_text("Mevcut kayıtların üzerine yazılsın").last.click()
            restore = section.get_by_role("button", name="♻️ Geri yükle")
            expect(restore).to_be_enabled()
            restore.click()
            expect(page.get_by_text("✅ Yedek geri yüklendi")).to_be_visible(timeout=20000)
            _wait_idle(page)
            page.get_by_text("✏️ Nöbet Ata/Kaldır", exact=True).click()
            _wait_idle(page, 1.0)
            assert _cell(page, ALI, 0) == "·"
            assert _cell(page, AYSE, 1) == "✓"
            assert _cell(page, ALI, 2) == "·"
            _no_exception(page)
        finally:
            browser.close()


def test_bozuk_yedek_dosyasi_reddedilir(app_url, tmp_path):
    bad = tmp_path / "bozuk.json"
    bad.write_text('{"format": "baska"}', encoding="utf-8")
    with sync_playwright() as playwright:
        browser = _launch(playwright)
        try:
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.goto(app_url)
            _wait_idle(page)
            section = _open_backup_section(page)
            section.locator("input[type='file']").set_input_files(str(bad))
            expect(section.get_by_text("Yedek yüklenemedi")).to_be_visible()
            expect(section.get_by_role("button", name="♻️ Geri yükle")).to_have_count(0)
            _no_exception(page)
        finally:
            browser.close()
