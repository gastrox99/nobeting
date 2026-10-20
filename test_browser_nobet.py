"""Browser regression for manual assignment totals.

Run with: python -m pytest test_browser_nobet.py -v
Requires Playwright and Chromium (set CHROMIUM_EXECUTABLE if not on PATH).
"""

import csv
import io
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from playwright.sync_api import expect, sync_playwright


APP_DIR = Path(__file__).resolve().parent
PERSON = "Browser Test Ali"


@pytest.fixture(scope="module")
def app_url():
    # Run an independent server with no database so existing saved schedules
    # cannot change the initial count or overwrite a real user's data.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "DATABASE_URL": ""}
    server = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "nobet.py",
            "--server.port", str(port), "--server.address", "127.0.0.1",
            "--server.headless", "true", "--browser.gatherUsageStats", "false",
        ],
        cwd=APP_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
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
            except (URLError, TimeoutError):
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


def displayed_total(page):
    """Copy the visible Nöbet Yükü grid and read its Toplam column."""
    heading = page.get_by_text("Nöbet Yükü", exact=True)
    table = heading.locator("xpath=following::div[@data-testid='stDataFrame'][1]")
    table.scroll_into_view_if_needed()
    # Glide renders the table to a canvas, not HTML cells. Its scroll surface
    # supports select-all/copy of the displayed cells as tab-separated text.
    table.locator(".dvn-scroller").click(position={"x": 120, "y": 55})
    table.locator("[data-testid='data-grid-canvas']").focus()
    page.keyboard.press("ControlOrMeta+A")
    page.keyboard.press("ControlOrMeta+C")
    text = page.evaluate("navigator.clipboard.readText()")
    rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
    # The copied grid has no header; column 0 is the index and column 1 is
    # Toplam (see the Nöbet Yükü dataframe in nobet.py).
    person_row = next(row for row in rows if row and row[0] == PERSON)
    assert len(person_row) > 1, f"Grid copy did not include Toplam: {text!r}"
    return int(person_row[1])


def wait_for_total(page, expected):
    deadline = time.monotonic() + 10
    actual = None
    while time.monotonic() < deadline:
        actual = displayed_total(page)
        if actual == expected:
            return
        time.sleep(0.2)
    assert actual == expected, f"Nöbet Yükü Toplam remained {actual}, expected {expected}"


def test_manual_assignment_updates_visible_total_without_simulation(app_url):
    browser_path = os.environ.get("CHROMIUM_EXECUTABLE") or shutil.which("chromium")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, executable_path=browser_path,
            args=["--no-sandbox"],
        )
        try:
            context = browser.new_context(
                accept_downloads=True, viewport={"width": 1600, "height": 1000},
                permissions=["clipboard-read", "clipboard-write"],
            )
            page = context.new_page()
            page.goto(app_url)
            _set_team(page, f"{PERSON}, Browser Test Ayşe")
            # Streamlit hides the native radio input; click its visible label.
            page.get_by_text("✏️ Nöbet Ata/Kaldır", exact=True).click()

            grid = page.locator(".st-key-schedule-grid")
            row = grid.locator("[data-testid='stHorizontalBlock']").filter(
                has=page.locator("b", has_text=PERSON)
            ).first
            first_day = row.locator(":scope > [data-testid='stColumn']").nth(1).get_by_role("button")
            baseline = displayed_total(page)
            assert baseline == 0
            assert first_day.inner_text() == "·"
            first_day.click()
            first_day.get_by_text("✓").wait_for()
            wait_for_total(page, baseline + 1)
            first_day.click()
            first_day.get_by_text("·").wait_for()
            wait_for_total(page, baseline)
        finally:
            browser.close()

def _open_app(playwright, app_url, team, color_scheme="light"):
    browser_path = os.environ.get("CHROMIUM_EXECUTABLE") or shutil.which("chromium")
    browser = playwright.chromium.launch(headless=True, executable_path=browser_path, args=["--no-sandbox"])
    page = browser.new_context(viewport={"width": 1600, "height": 1000}, color_scheme=color_scheme).new_page()
    page.goto(app_url)
    _set_team(page, team)
    page.get_by_text("✏️ Nöbet Ata/Kaldır", exact=True).click()
    return browser, page


def _settings_box(page, role, name):
    """Ayarlar paneli ekip girilince kapanır; gerekirse açıp alanı döndür."""
    box = page.get_by_role(role, name=name)
    if not box.is_visible():
        page.get_by_text("⚙️ Ayarlar").click()
    box.wait_for()
    return box


def _team_box(page, row):
    return page.get_by_role("textbox", name=f"{row}. kişi", exact=True)


def _team_values(page):
    _settings_box(page, "textbox", "1. kişi")
    values = []
    while _team_box(page, len(values) + 1).count():
        values.append(_team_box(page, len(values) + 1).input_value())
    return [v for v in values if v]


def _set_team(page, team):
    """Ekip tablosunu verilen listeye getirir: boşsa virgüllü liste son satıra
    yapıştırılır, doluysa satırlar tek tek düzeltilir, fazla satırlar silinir."""
    names = [n.strip() for n in team.split(",") if n.strip()]
    current = _team_values(page)
    if not current:
        box = _team_box(page, 1)
        box.fill(team)
        box.press("Tab")
        _wait_idle(page, 1.5)
        return
    for i, name in enumerate(names, start=1):
        box = _team_box(page, i)
        if box.input_value() != name:
            box.fill(name)
            box.press("Tab")
            _wait_idle(page, 1.5)
    while _team_box(page, len(names) + 1).count() and _team_box(page, len(names) + 1).input_value():
        box = _team_box(page, len(names) + 1)
        box.fill("")
        box.press("Tab")
        _wait_idle(page, 1.5)


def _no_exception(page):
    assert page.locator("[data-testid='stException']").count() == 0, \
        page.locator("[data-testid='stException']").first.inner_text()


def _wait_idle(page, seconds=3.0):
    """Sayfa yeniden çizimi bitene kadar bekle; sürekli çalışıyorsa hata ver."""
    status = page.locator("[data-testid='stStatusWidget']")
    deadline = time.monotonic() + 20
    quiet_since = None
    while time.monotonic() < deadline:
        if status.count() and status.first.is_visible():
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since >= seconds:
            return
        time.sleep(0.1)
    raise AssertionError("Sayfa durmadan yeniden çiziliyor (sonsuz döngü?)")


def _day_button(page, person, day_index):
    row = page.locator(".st-key-schedule-grid [data-testid='stHorizontalBlock']").filter(
        has=page.locator("b", has_text=person)
    ).first
    return row.locator(":scope > [data-testid='stColumn']").nth(day_index + 1).get_by_role("button")


def test_excel_yukleme_bir_kez_islenir_ve_dongu_olmaz(app_url, tmp_path):
    import pandas as pd
    from datetime import datetime
    now = datetime.now()
    year, month = (now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1)
    path = tmp_path / "liste.xlsx"
    pd.DataFrame({
        "Tarih": [f"01.{month:02d}.{year}"],
        "Görev1": [PERSON],
        "Görev2": ["-"],
    }).to_excel(path, sheet_name="Günlük Liste", index=False)

    with sync_playwright() as playwright:
        browser, page = _open_app(playwright, app_url, f"{PERSON}, Browser Test Ayşe")
        try:
            assert page.locator("[data-testid='stException']").count() == 0
            # Excel yükleme sayfanın altındaki Yedek bölümünde.
            page.locator("summary", has_text="💾 Yedek").click()
            page.locator("[data-testid='stFileUploader'] input[type=file]").set_input_files(str(path))
            page.get_by_text("gün yüklendi").wait_for(timeout=20000)
            _wait_idle(page)
            assert _day_button(page, PERSON, 0).inner_text() == "✓"
            # Dosya kutuda dururken elle yapılan değişiklik Excel'le ezilmemeli.
            page.keyboard.press("Escape")
            second = _day_button(page, "Browser Test Ayşe", 1)
            second.click()
            second.get_by_text("✓").wait_for()
            _wait_idle(page)
            assert _day_button(page, "Browser Test Ayşe", 1).inner_text() == "✓"
            assert page.locator("[data-testid='stException']").count() == 0
        finally:
            browser.close()


def test_isim_duzeltilince_nobetler_korunur(app_url):
    with sync_playwright() as playwright:
        browser, page = _open_app(playwright, app_url, "Rename Ali, Rename Ayşe")
        try:
            first = _day_button(page, "Rename Ali", 0)
            first.click()
            first.get_by_text("✓").wait_for()
            _wait_idle(page)
            _set_team(page, "Rename Ali Can, Rename Ayşe")
            _wait_idle(page)
            _no_exception(page)
            assert _day_button(page, "Rename Ali Can", 0).inner_text() == "✓"
        finally:
            browser.close()


def test_ekip_satir_satir_siralaninca_nobetler_yer_degistirmez(app_url):
    # Maymun bulgusu: satırlar tek tek yeniden yazılırken ara adımlar ad
    # değişikliği sanılıyor, nöbetler başka kişiye geçiyordu.
    team = ["Sıra Ali", "Sıra Ayşe", "Sıra Can", "Sıra Deniz"]
    with sync_playwright() as playwright:
        browser, page = _open_app(playwright, app_url, ", ".join(team))
        try:
            for day, person in enumerate(team):
                button = _day_button(page, person, day)
                button.click()
                button.get_by_text("✓").wait_for()
                _wait_idle(page, 1.5)
            _set_team(page, "Sıra Deniz, Sıra Ali, Sıra Ayşe, Sıra Can")
            _no_exception(page)
            for day, person in enumerate(team):
                cells = [_day_button(page, person, d).inner_text() for d in range(len(team))]
                assert cells == ["✓" if d == day else "·" for d in range(len(team))], (person, cells)
        finally:
            browser.close()


def test_simulasyon_takvim_ve_analiz_karanlik_modda_hatasiz(app_url):
    with sync_playwright() as playwright:
        browser, page = _open_app(playwright, app_url, "Sim Ali, Sim Ayşe, Sim Can, Sim Deniz, Sim Ece",
                                  color_scheme="dark")
        try:
            page.get_by_role("button", name="⚡ Simülasyon").click()
            _wait_idle(page)
            _no_exception(page)
            page.get_by_role("tab", name="📅 Takvim Görünümü").click()
            page.get_by_text("Tüm ekip gösteriliyor").click()
            page.get_by_role("option", name="Sim Ali").click()
            page.get_by_text("Tercih Başarısı").wait_for()
            _wait_idle(page)
            _no_exception(page)
            page.get_by_role("heading", name="📊 Analiz").scroll_into_view_if_needed()
            _no_exception(page)
            # Günlük Liste sekmesi kaldırıldı; görev yeri takvimde değişir.
            assert page.get_by_role("tab", name="📋 Günlük Liste").count() == 0
        finally:
            browser.close()


def test_ayni_gorev_adi_numaralanir_ve_uzak_yil_secilir(app_url):
    with sync_playwright() as playwright:
        browser, page = _open_app(playwright, app_url, "Rol Ali, Rol Ayşe, Rol Can")
        try:
            # Ekip girişinin yeniden çizimi bitmeden yazılan görev adı CI'da kayboluyordu.
            _wait_idle(page)
            roles = _settings_box(page, "textbox", "🏷️ Görev İsimleri:")
            roles.fill("AYB, AYB")
            roles.press("Enter")
            page.get_by_text("AYB → AYB 2").wait_for()
            year = _settings_box(page, "spinbutton", "📅 Yıl")
            year.fill("2031")
            year.press("Enter")
            _wait_idle(page)
            _no_exception(page)
        finally:
            browser.close()


def test_buyuk_ekip_hizli_duzenleme_modunda_acilir(app_url):
    team = ", ".join(f"Büyük {i}" for i in range(20))
    with sync_playwright() as playwright:
        browser, page = _open_app(playwright, app_url, team)
        try:
            page.get_by_text("hızlı düzenleme modunda").wait_for()
            _wait_idle(page)
            _no_exception(page)
        finally:
            browser.close()


def _whatsapp_first_day(page):
    page.get_by_text("📤 Dışa aktar").click()
    code = page.locator("[data-testid='stPopoverBody'] [data-testid='stCode']").first
    code.wait_for()
    line = code.inner_text().splitlines()[2]
    page.keyboard.press("Escape")
    return line


def test_ekip_degisince_nobet_ve_gorev_yerleri_korunur(app_url):
    with sync_playwright() as playwright:
        browser, page = _open_app(playwright, app_url, "Ek Ali, Ek Ayşe, Ek Can")
        try:
            for person in ("Ek Ali", "Ek Ayşe", "Ek Can"):
                cell = _day_button(page, person, 0)
                cell.click()
                cell.get_by_text("✓").wait_for()
                _wait_idle(page)
            # Görev sayısından fazla atanan kişi listede "Ek 1" olarak görünür.
            assert "Ek 1: Ek Can" in _whatsapp_first_day(page)

            # Ekipten çıkan kişinin nöbeti saklanır, geri yazılınca döner.
            _set_team(page, "Ek Ali, Ek Ayşe")
            _wait_idle(page)
            _set_team(page, "Ek Ali, Ek Ayşe, Ek Can")
            _wait_idle(page)
            _no_exception(page)
            assert _day_button(page, "Ek Can", 0).inner_text() == "✓"

            # Elle değiştirilen görev yerleri, ekibe biri eklenince korunur.
            cell = _day_button(page, "Ek Can", 0)
            cell.click()
            cell.get_by_text("·").wait_for()
            _wait_idle(page)
            page.get_by_role("tab", name="📅 Takvim Görünümü").click()
            page.locator(".st-key-cal_swap_1 button, [class*='st-key-cal_swap_1'] button").first.click()
            _wait_idle(page)
            before = _whatsapp_first_day(page)
            _set_team(page, "Ek Ali, Ek Ayşe, Ek Can, Ek Deniz")
            _wait_idle(page)
            _no_exception(page)
            assert _whatsapp_first_day(page) == before
        finally:
            browser.close()


def test_ayar_kutulari_art_arda_duzenlenebilir(app_url):
    # Eskiden art arda yapılan iki düzenlemeden ikincisi sessizce geri dönüyordu.
    with sync_playwright() as playwright:
        browser, page = _open_app(playwright, app_url, "Kutu 1")
        try:
            for team in ("Kutu 1, Kutu 2", "Kutu 1, Kutu 2, Kutu 3", "Kutu 1"):
                _set_team(page, team)
                _wait_idle(page)
                assert ", ".join(_team_values(page)) == team
            for roles in ("A, B", "C, D"):
                box = _settings_box(page, "textbox", "🏷️ Görev İsimleri:")
                box.fill(roles)
                box.press("Enter")
                _wait_idle(page)
                assert _settings_box(page, "textbox", "🏷️ Görev İsimleri:").input_value() == roles
            _no_exception(page)
        finally:
            browser.close()


def test_unvan_secilince_saat_ucreti_gelir_ve_elle_degisir(app_url):
    with sync_playwright() as playwright:
        browser, page = _open_app(playwright, app_url, "Ücret Ali, Ücret Ayşe")
        try:
            _settings_box(page, "combobox", "🩺 Ünvan:").click()
            page.get_by_role("option", name="Tabip", exact=True).click()
            _wait_idle(page)
            rate = lambda: float(_settings_box(page, "spinbutton", "💰 Saat Ücreti (TL):").input_value())
            auto = rate()
            assert 200 < auto < 400
            expect(page.get_by_text("damga vergisi düşülmüş", exact=False)).to_be_visible()
            # Elle yazılan ücret sonraki yeniden çizimlerde korunur.
            box = _settings_box(page, "spinbutton", "💰 Saat Ücreti (TL):")
            box.fill("300")
            box.press("Tab")
            _wait_idle(page)
            _set_team(page, "Ücret Ali, Ücret Ayşe, Ücret Can")
            _wait_idle(page)
            assert rate() == 300
            expect(page.get_by_text("Ücret elle değiştirildi", exact=False)).to_be_visible()
            # Riskli birim seçilince ücret yeniden otomatik hesaplanır (%50).
            page.get_by_text("Riskli birim (%50 artırımlı)").click()
            _wait_idle(page)
            assert abs(rate() - auto * 1.5) < 0.05
            _no_exception(page)
        finally:
            browser.close()


def _pair_checkbox(page, first, second):
    box = page.get_by_role("checkbox", name=f"{first} ile {second}", exact=True)
    return box, page.locator("[data-testid='stCheckbox']").filter(has=box).first.locator("label")


def test_yasak_ikili_capraz_tablodan_ve_limit_ekip_tablosundan_girilir(app_url):
    with sync_playwright() as playwright:
        browser, page = _open_app(playwright, app_url, "Ayar Ali, Ayar Ayşe, Ayar Can, Ayar Deniz")
        try:
            _settings_box(page, "textbox", "1. kişi")
            box, label = _pair_checkbox(page, "Ayar Ali", "Ayar Ayşe")
            label.click()
            _wait_idle(page, 1.5)
            # Her ikili bir kez (ekip sırasıyla) gösterilir; tekrar tıklayınca kalkar.
            expect(box).to_be_checked()
            assert _pair_checkbox(page, "Ayar Ayşe", "Ayar Ali")[0].count() == 0
            _pair_checkbox(page, "Ayar Can", "Ayar Deniz")[1].click()
            _wait_idle(page, 1.5)
            _pair_checkbox(page, "Ayar Can", "Ayar Deniz")[1].click()
            _wait_idle(page, 1.5)
            expect(_pair_checkbox(page, "Ayar Can", "Ayar Deniz")[0]).not_to_be_checked()
            expect(_pair_checkbox(page, "Ayar Ali", "Ayar Ayşe")[0]).to_be_checked()

            most = page.get_by_role("spinbutton", name="3. kişi en fazla nöbet", exact=True)
            assert most.input_value() == ""
            most.fill("2")
            most.press("Tab")
            _wait_idle(page, 1.5)
            weekend = page.get_by_role("spinbutton", name="3. kişi en fazla hafta sonu", exact=True)
            weekend.fill("0")
            weekend.press("Tab")
            _wait_idle(page, 1.5)
            assert page.get_by_role("spinbutton", name="3. kişi en fazla nöbet", exact=True).input_value() == "2"
            assert weekend.input_value() == "0"
            # Son satır boş durur; isim yazılınca altına yeni satır açılır.
            _team_box(page, 5).fill("Ayar Ece")
            _team_box(page, 5).press("Tab")
            _wait_idle(page, 1.5)
            expect(_team_box(page, 6)).to_have_value("")
            # İsim değiştirmek limitleri ve yasak ikiliyi de taşır.
            _team_box(page, 1).fill("Ayar Alican")
            _team_box(page, 1).press("Tab")
            _wait_idle(page, 1.5)
            expect(_pair_checkbox(page, "Ayar Alican", "Ayar Ayşe")[0]).to_be_checked()
            page.get_by_role("button", name="⚡ Simülasyon").click()
            _wait_idle(page)
            _no_exception(page)
            # Yasak ikili ve limit hiçbir uyarı olmadan uygulanır.
            expect(page.get_by_text("ekip listesinde olmayan")).to_have_count(0)
            expect(page.get_by_text("Kişisel limit sorunları")).to_have_count(0)
        finally:
            browser.close()


def test_iyilestir_dugmesi_listeyi_iyilestirir_ve_geri_alinir(app_url):
    with sync_playwright() as playwright:
        browser, page = _open_app(playwright, app_url, "İyi Ali, İyi Ayşe, İyi Can, İyi Deniz, İyi Ece, İyi Fatma")
        try:
            improve = page.get_by_role("button", name="✨ İyileştir")
            # Boş listede iyileştirilecek bir şey yok.
            expect(improve).to_be_disabled()
            page.get_by_role("button", name="⚡ Simülasyon").click()
            _wait_idle(page)
            expect(improve).to_be_enabled()
            improve.click()
            result = page.get_by_text("Liste iyileştirildi").or_(page.get_by_text("Daha iyi bir liste bulunamadı"))
            expect(result.first).to_be_visible(timeout=30000)
            _wait_idle(page)
            _no_exception(page)
            if page.get_by_text("Liste iyileştirildi").count():
                expect(page.get_by_role("button", name="↩️ Geri")).to_be_enabled()
                page.get_by_role("button", name="↩️ Geri").click()
                _wait_idle(page)
                _no_exception(page)
        finally:
            browser.close()


def test_tercihler_toplu_tablo_ve_whatsapp_metni(app_url):
    team = "Toplu Ali, Toplu Ayşe, Toplu Can"
    with sync_playwright() as playwright:
        browser, page = _open_app(playwright, app_url, team)
        try:
            _wait_idle(page)
            page.get_by_text("📝 Tercihler").click()
            expect(page.locator("[data-testid='stPopoverBody']").get_by_text("işaretli tercih, kaçınma")).to_be_visible()
            page.keyboard.press("Escape")
            page.get_by_text("🎨 Tercih Belirle", exact=True).click()
            _wait_idle(page, 1.0)
            page.get_by_role("button", name="🟥İzin").click()
            _wait_idle(page, 1.0)
            pref_row = page.locator(".st-key-preference-grid [data-testid='stHorizontalBlock']").filter(
                has=page.locator("b", has_text="Toplu Ayşe")
            ).first
            for day in (1, 2):
                button = pref_row.locator(":scope > [data-testid='stColumn']").nth(day).get_by_role("button")
                button.click()
                expect(button).to_have_text("İ")
            _wait_idle(page, 1.0)
            page.get_by_text("📝 Tercihler").click()
            code = page.locator("[data-testid='stPopoverBody'] [data-testid='stCode']").first
            code.wait_for()
            text = code.inner_text()
            assert "*Toplu Ayşe* — İzin: 1-2" in text, text
            assert "*Toplu Ali* — talep yok" in text, text
            expect(page.get_by_role("link", name="💬 WhatsApp'ta gönder")).to_be_visible()
            _no_exception(page)
        finally:
            browser.close()
