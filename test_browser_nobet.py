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
from playwright.sync_api import sync_playwright


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
            team = page.get_by_role("textbox", name="👥 Ekip (virgülle ayırın):")
            team.fill(f"{PERSON}, Browser Test Ayşe")
            team.press("Tab")
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
            first_day.get_by_text("•").wait_for()
            wait_for_total(page, baseline + 1)
            first_day.click()
            first_day.get_by_text("·").wait_for()
            wait_for_total(page, baseline)
        finally:
            browser.close()