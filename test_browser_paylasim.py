"""Paylaşım sayfası tarayıcı testi: gerçek Streamlit, Chromium ve PostgreSQL ile.

TEST_DATABASE_URL tanımlı değilse atlanır (GitHub'daki "Testler" iş akışı
geçici bir PostgreSQL açıp bu değişkeni verir). Gerçek veritabanının adresini
buraya vermeyin: test kendi sahip adıyla kayıt yazar ve siler.
"""

import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from datetime import timedelta
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
import pytest

TEST_DB = os.environ.get("TEST_DATABASE_URL")
if not TEST_DB:
    pytest.skip("TEST_DATABASE_URL tanımlı değil", allow_module_level=True)

from playwright.sync_api import expect, sync_playwright  # noqa: E402

from nobet_core import TR_AY_ADLARI, TR_GUN_KISA, kisi_ucret_dokumu, now_tr, previous_period, tr_sayi  # noqa: E402
from test_browser_nobet import APP_DIR, _no_exception, _wait_idle  # noqa: E402

OWNER = f"paylasim-tarayici-{uuid.uuid4().hex[:8]}@example.com"


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


def _save_month(db, year, month, days):
    """``days``: {gün: (AYB, GYB)}."""
    columns = [f"{d} {TR_GUN_KISA[pd.Timestamp(year, month, d).weekday()]}" for d in sorted(days)]
    team = sorted({name for pair in days.values() for name in pair})
    schedule = pd.DataFrame(False, index=team, columns=columns)
    rows = []
    for column, day in zip(columns, sorted(days)):
        for name in days[day]:
            schedule.at[name, column] = True
        weekday = column.split()[1]
        rows.append({"Tarih": f"{day:02d}.{month:02d}.{year} {weekday}", "AYB": days[day][0], "GYB": days[day][1]})
    assert db.save_schedule(f"Otomatik_{year}_{month:02d}", year, month, team, schedule,
                            rows_liste=rows, owner=OWNER)


def _seed(db):
    today = now_tr().date()
    tomorrow = today + timedelta(days=1)
    days = {today.day: ("Bugün Ali", "Bugün Ayşe")}
    if tomorrow.month == today.month:
        days[tomorrow.day] = ("Yarın Can", "Yarın Deniz")
    if today.day > 1:
        days[1] = ("Geçmiş Ece", "Geçmiş Fatma")
    _save_month(db, today.year, today.month, days)
    prev_year, prev_month = previous_period(today.year, today.month)
    _save_month(db, prev_year, prev_month, {
        3: ("Önceki Gül", "Önceki Hasan"),
        **{_d: ("Ücret Ali", "Ücret Veli") for _d in range(5, 28, 2)},
    })
    assert db.save_app_settings(
        {"team": "Bugün Ali, Talep Ali, Ücret Ali", "title": "Tabip", "risky_unit": True}, owner=OWNER)
    return {
        "today": today, "tomorrow_in_month": tomorrow.month == today.month,
        "previous": (prev_year, prev_month),
        "permanent": db.create_share_link(OWNER, 0, 0),
        "monthly": db.create_share_link(OWNER, prev_year, prev_month),
    }


def _cleanup(db):
    conn = db._connect()
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM schedules WHERE owner = %s", (OWNER,))
        cur.execute("DELETE FROM share_links WHERE owner = %s", (OWNER,))
        cur.execute("DELETE FROM team_requests WHERE owner = %s", (OWNER,))
        cur.execute("DELETE FROM app_settings WHERE key = %s", (db.settings_key(OWNER),))
    conn.close()


@pytest.fixture(scope="module")
def shared():
    data = _with_db(_seed)
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
        yield url, data
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
        server.stderr.close()
        _with_db(_cleanup)


def _open(playwright, url):
    browser = playwright.chromium.launch(
        headless=True, executable_path=os.environ.get("CHROMIUM_EXECUTABLE") or shutil.which("chromium"),
        args=["--no-sandbox"],
    )
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.goto(url)
    page.get_by_text("Nöbet Listesi", exact=True).wait_for()
    _wait_idle(page)
    return browser, page


def _choose(page, name, all_rows=True):
    """Açılışta sorulan adı seçer; istenirse tüm listeyi açar."""
    page.get_by_text("Adını seç", exact=True).click()
    page.get_by_role("option", name=name, exact=True).click()
    _wait_idle(page)
    if all_rows:
        page.get_by_text("Tüm listeyi göster", exact=True).click()
        _wait_idle(page)


def test_once_ad_sorulur_sonra_kendi_nobetleri(shared):
    url, data = shared
    with sync_playwright() as p:
        browser, page = _open(p, f"{url}/?paylas={data['permanent']}")
        try:
            _no_exception(page)
            expect(page.get_by_text("Nöbetlerini görmek ve talep göndermek için adını seç.")).to_be_visible()
            expect(page.locator("[data-testid='stDataFrame']")).to_have_count(0)
            _choose(page, "Bugün Ali", all_rows=False)
            _no_exception(page)
            grid = page.locator("[data-testid='stDataFrame']")
            expect(grid).to_contain_text("Bugün Ali")
            if data["tomorrow_in_month"]:
                expect(grid).not_to_contain_text("Yarın Can")
        finally:
            browser.close()


def test_kalici_baglanti_bu_ayi_acar_ve_ay_degistirilir(shared):
    url, data = shared
    today = data["today"]
    with sync_playwright() as p:
        browser, page = _open(p, f"{url}/?paylas={data['permanent']}")
        try:
            _choose(page, "Bugün Ali")
            _no_exception(page)
            expect(page.get_by_text("Son güncelleme:")).to_be_visible()
            expect(page.get_by_text("Bugün nöbetçi")).to_be_visible()
            expect(page.get_by_text("Bugün Ali (AYB) · Bugün Ayşe (GYB)")).to_be_visible()
            if data["tomorrow_in_month"]:
                expect(page.get_by_text("Yarın Can (AYB) · Yarın Deniz (GYB)")).to_be_visible()
            grid = page.locator("[data-testid='stDataFrame']")
            expect(grid).to_contain_text("Bugün Ali")
            expect(grid).not_to_contain_text("Önceki Gül")

            prev_year, prev_month = data["previous"]
            page.get_by_text(f"{TR_AY_ADLARI[today.month]} {today.year}", exact=True).click()
            page.get_by_role("option", name=f"{TR_AY_ADLARI[prev_month]} {prev_year}").click()
            _wait_idle(page)
            _no_exception(page)
            expect(grid).to_contain_text("Önceki Gül")
            # Geçmiş ayda bugün/yarın kutusu çıkmaz.
            expect(page.get_by_text("Bugün nöbetçi")).to_have_count(0)
        finally:
            browser.close()


def test_eski_aylik_baglanti_calismaya_devam_eder(shared):
    url, data = shared
    prev_year, prev_month = data["previous"]
    with sync_playwright() as p:
        browser, page = _open(p, f"{url}/?paylas={data['monthly']}")
        try:
            _choose(page, "Bugün Ali")
            _no_exception(page)
            expect(page.get_by_text(f"{TR_AY_ADLARI[prev_month]} {prev_year}")).to_be_visible()
            expect(page.locator("[data-testid='stDataFrame']")).to_contain_text("Önceki Gül")
            # Aylık bağlantıda ay seçimi yok.
            expect(page.get_by_text("Ay:", exact=True)).to_have_count(0)
        finally:
            browser.close()


def test_talep_gonderilir_ve_onay_bekler(shared):
    url, data = shared
    today = data["today"]
    next_year, next_month = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
    with sync_playwright() as p:
        browser, page = _open(p, f"{url}/?paylas={data['permanent']}")
        try:
            _choose(page, "Talep Ali", all_rows=False)
            page.get_by_role("tab", name="📝 Talep Gönder").click()
            # Varsayılan ay sonraki ay.
            expect(page.get_by_text(f"{TR_AY_ADLARI[next_month]} {next_year}", exact=True)).to_be_visible()
            page.get_by_label("👍 Nöbet istediğim günler (Tercih)").fill("3,5")
            page.get_by_label("🏖️ İzinli olduğum günler (İzin)").fill("5-7,abc")
            page.get_by_role("button", name="📨 Talebi gönder").click()
            _wait_idle(page)
            _no_exception(page)
            expect(page.get_by_text("Talep gönderilmedi:")).to_be_visible()
            assert _with_db(lambda db: db.list_team_requests(OWNER)) == []

            page.get_by_label("🏖️ İzinli olduğum günler (İzin)").fill("10-12")
            page.get_by_role("button", name="📨 Talebi gönder").click()
            _wait_idle(page)
            _no_exception(page)
            expect(page.get_by_text("Talebin iletildi")).to_be_visible()
            requests = _with_db(lambda db: db.list_team_requests(OWNER))
            assert [(r["person"], r["year"], r["month"], r["days"]) for r in requests] == [
                ("Talep Ali", next_year, next_month, {"tercih": [3, 5], "izin": [10, 11, 12]}),
            ]
            # Liste sekmesi yerinde durur.
            page.get_by_role("tab", name="📋 Nöbet Listesi").click()
            expect(page.get_by_text("listesinde nöbetin yok")).to_be_visible()

            # Sayfa yeniden açılınca ad telefonda saklı olduğu için hazır gelir;
            # bekleyen talep kutularda durur ve güncellenince yerine geçer.
            page.reload()
            page.get_by_role("tab", name="📝 Talep Gönder").wait_for()
            _wait_idle(page)
            _no_exception(page)
            page.get_by_role("tab", name="📝 Talep Gönder").click()
            expect(page.get_by_label("👍 Nöbet istediğim günler (Tercih)")).to_have_value("3,5")
            expect(page.get_by_label("🏖️ İzinli olduğum günler (İzin)")).to_have_value("10-12")
            expect(page.get_by_role("button", name="📨 Talebi gönder")).to_have_count(0)
            page.get_by_label("🏖️ İzinli olduğum günler (İzin)").fill("20")
            page.get_by_role("button", name="🔄 Talebi güncelle").click()
            _wait_idle(page)
            _no_exception(page)
            expect(page.get_by_text("Talebin iletildi")).to_be_visible()
            requests = _with_db(lambda db: db.list_team_requests(OWNER))
            assert [(r["person"], r["days"]) for r in requests] == [
                ("Talep Ali", {"tercih": [3, 5], "izin": [20]}),
            ]
        finally:
            browser.close()


def test_kisi_yalniz_kendi_ucret_dokumunu_gorur(shared):
    url, data = shared
    today = data["today"]
    prev_year, prev_month = data["previous"]
    expected = kisi_ucret_dokumu(
        range(5, 28, 2), prev_year, prev_month, {"title": "Tabip", "risky_unit": True}, None, "Ücret Ali")
    assert expected["toplam"] > 0
    with sync_playwright() as p:
        browser, page = _open(p, f"{url}/?paylas={data['permanent']}")
        try:
            _choose(page, "Ücret Ali", all_rows=False)
            page.get_by_text(f"{TR_AY_ADLARI[today.month]} {today.year}", exact=True).click()
            page.get_by_role("option", name=f"{TR_AY_ADLARI[prev_month]} {prev_year}").click()
            _wait_idle(page)
            _no_exception(page)
            title = f"💰 Tahmini nöbet ücretim: {tr_sayi(expected['toplam'])} ₺"
            page.get_by_text(title).click()
            _wait_idle(page)
            expect(page.get_by_text("12 nöbet = 288 saat")).to_be_visible()
            expect(page.get_by_text("Tabip, riskli birim (%50)")).to_be_visible()
            pay_table = page.locator("[data-testid='stExpander'] [data-testid='stDataFrame']")
            expect(pay_table).to_contain_text("Gece (20-08)")
            expect(pay_table).to_contain_text(f"{tr_sayi(expected['satirlar']['gece'][1])} ₺")
            # Tüm liste açılsa da yalnız kendi ücreti görünür.
            page.get_by_text("Tüm listeyi göster", exact=True).click()
            _wait_idle(page)
            expect(page.get_by_text("💰 Tahmini nöbet ücretim")).to_have_count(1)
        finally:
            browser.close()
