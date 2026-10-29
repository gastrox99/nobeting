"""Canlı siteyi açar; uykudaysa uyandırır, açılmazsa hata verir (iş akışı kırmızı olur)."""
import sys

from playwright.sync_api import sync_playwright

URL = "https://nobeting-n3t8zgwcucrj9fgmo5eswn.streamlit.app"
UYANDIR = "get this app back up"
HATA = ("Internal server error", "Error running app", "This app has encountered an error")

with sync_playwright() as p:
    sayfa = p.chromium.launch().new_page()
    sayfa.goto(URL, timeout=120_000)
    # Uyku ekranı ana belgede ya da Streamlit'in iç çerçevesinde olabilir.
    sayfa.wait_for_timeout(10_000)
    for hedef in [sayfa, *sayfa.frames]:
        dugme = hedef.get_by_role("button", name=UYANDIR, exact=False)
        if dugme.count():
            print("Site uykudaydı, uyandırılıyor.")
            dugme.first.click()
            break
    try:
        sayfa.frame_locator("iframe").first.locator("[data-testid='stApp']").wait_for(timeout=240_000)
    except Exception:
        sayfa.locator("[data-testid='stApp']").wait_for(timeout=60_000)
    govde = sayfa.inner_text("body")
    if any(h in govde for h in HATA):
        print("Sayfada hata metni var.")
        sys.exit(1)
    print("Site açıldı.")
