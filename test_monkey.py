"""Maymun testi: uygulamaya rastgele işlemler yaptırıp tutarlılığı denetler.

Yavaş olduğu için yalnızca RUN_MONKEY=1 iken çalışır (GitHub'da "Maymun testi"
iş akışı elle başlatılır). Bulguları monkey_findings.txt dosyasına yazar; test
kendisi başarısız olmaz.
"""
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

if os.environ.get("RUN_MONKEY") != "1":
    pytest.skip("Maymun testi yalnızca RUN_MONKEY=1 ile çalışır", allow_module_level=True)

from playwright.sync_api import sync_playwright  # noqa: E402

from nobet_core import TR_AY_ADLARI  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
FINDINGS = APP_DIR / "monkey_findings.txt"
ASSIGNED = {"✓", "!", "×", "🔺"}
FREE = {"·", "T", "K", "İ", "M"}
ROUNDS = int(os.environ.get("MONKEY_ROUNDS", "3"))
STEPS = int(os.environ.get("MONKEY_STEPS", "120"))

findings = []


def note(kind, detail, history):
    entry = f"[{kind}] {detail}\n  son işlemler: {' → '.join(history[-8:])}"
    findings.append(entry)
    print(entry, flush=True)


@pytest.fixture(scope="module")
def app_url():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ}
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "nobet.py", "--server.port", str(port),
         "--server.address", "127.0.0.1", "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=APP_DIR, env=env, stdout=subprocess.DEVNULL, stderr=open(APP_DIR / "server.log", "w"),
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{url}/_stcore/health", timeout=1) as response:
                if response.status == 200:
                    break
        except (URLError, TimeoutError, ConnectionError):
            time.sleep(0.3)
    yield url
    server.terminate()
    server.wait(timeout=10)


GRID_JS = """() => {
  const rows = [...document.querySelectorAll(":is(.st-key-schedule-grid,.st-key-preference-grid) [data-testid='stHorizontalBlock']")];
  return rows.map(r => {
    const cols = [...r.querySelectorAll(":scope > [data-testid='stColumn']")];
    const b = cols[0] && cols[0].querySelector('b');
    if (!b) return null;
    const spans = cols[0].querySelectorAll('span');
    return {
      name: b.innerText.trim(),
      badge: parseInt(spans[spans.length - 1].innerText),
      cells: cols.slice(1).map(c => {
        const btn = c.querySelector('button');
        return btn ? btn.innerText.trim() : null;
      }),
      locked: cols.slice(1).map(c => !!c.querySelector('.cell-locked')),
    };
  }).filter(Boolean);
}"""


def wait_idle(page, quiet=1.5, limit=40):
    status = page.locator("[data-testid='stStatusWidget']")
    deadline = time.monotonic() + limit
    quiet_since = None
    while time.monotonic() < deadline:
        busy = status.count() and status.first.is_visible()
        if busy:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since >= quiet:
            return True
        time.sleep(0.1)
    return False


def grid(page):
    return page.evaluate(GRID_JS)


def ensure_settings(page, role, name):
    box = page.get_by_role(role, name=name, exact=True)
    if not box.is_visible():
        page.get_by_text("⚙️ Ayarlar", exact=True).click()
        box.wait_for(timeout=5000)
    return box


def team_box(page, row):
    return page.get_by_role("textbox", name=f"{row}. kişi", exact=True)


def set_team(page, names):
    """Ekip tablosunu listeye getirir: satırları düzeltir, fazlasını siler."""
    ensure_settings(page, "textbox", "1. kişi")
    for i, name in enumerate(names, start=1):
        box = team_box(page, i)
        if box.input_value() != name:
            box.fill(name)
            box.press("Tab")
            wait_idle(page)
    while team_box(page, len(names) + 1).count() and team_box(page, len(names) + 1).input_value():
        box = team_box(page, len(names) + 1)
        box.fill("")
        box.press("Tab")
        wait_idle(page)


def share_text(page):
    page.get_by_text("📤 Dışa aktar").click()
    code = page.locator("[data-testid='stPopoverBody'] [data-testid='stCode'] code").first
    code.wait_for(timeout=5000)
    text = code.inner_text()
    page.keyboard.press("Escape")
    time.sleep(0.2)
    return text


def parse_share(text):
    days = {}
    for line in text.splitlines():
        m = re.match(r"^\*?(\d+) .*? — (.*)$", line)
        if not m:
            continue
        people = []
        if m.group(2) != "boş":
            for part in m.group(2).split(" · "):
                if ": " in part:
                    people.append(part.split(": ", 1)[1].rstrip("*"))
        days[int(m.group(1))] = people
    return days


def check_invariants(page, history, per_shift):
    if page.locator("[data-testid='stException']").count():
        note("HATA", page.locator("[data-testid='stException']").first.inner_text()[:600], history)
        return None
    rows = grid(page)
    if not rows:
        return None
    for row in rows:
        assigned = sum(1 for c in row["cells"] if c in ASSIGNED)
        unknown = [c for c in row["cells"] if c not in ASSIGNED | FREE]
        if unknown:
            note("BİLİNMEYEN HÜCRE", f"{row['name']}: {unknown[:3]}", history)
        if assigned != row["badge"]:
            note("SAYI TUTARSIZ", f"{row['name']}: rozet {row['badge']} ama {assigned} nöbetli hücre", history)
        for i, (cell, locked) in enumerate(zip(row["cells"], row["locked"])):
            if locked and cell not in ASSIGNED:
                note("KİLİT TUTARSIZ", f"{row['name']} gün {i+1}: kilitli ama nöbetsiz", history)
    try:
        days = parse_share(share_text(page))
    except Exception as exc:
        note("PAYLAŞIM METNİ OKUNAMADI", repr(exc)[:200], history)
        return rows
    for day_index in range(len(rows[0]["cells"])):
        on_grid = {r["name"] for r in rows if r["cells"][day_index] in ASSIGNED}
        listed = days.get(day_index + 1, [])
        if len(listed) != len(set(listed)):
            note("LİSTEDE ÇİFT İSİM", f"gün {day_index+1}: {listed}", history)
        if not set(listed) <= on_grid:
            note("LİSTE ÇİZELGEDEN FARKLI", f"gün {day_index+1}: listede {listed}, çizelgede {sorted(on_grid)}", history)
        elif len(on_grid) <= per_shift and set(listed) != on_grid:
            note("LİSTEDE EKSİK", f"gün {day_index+1}: listede {listed}, çizelgede {sorted(on_grid)}", history)
    return rows


def snapshot(page):
    rows = grid(page)
    return json.dumps(rows, ensure_ascii=False, sort_keys=True)


def assigned_cells(page):
    """Ekrandaki nöbetler {(kişi, gün)}; tercih/kilit işaretleri sayılmaz."""
    return {(r["name"], i) for r in grid(page) for i, c in enumerate(r["cells"]) if c in ASSIGNED}


def open_backup_section(page):
    section = page.locator("[data-testid='stExpander']").filter(
        has=page.locator("summary", has_text="💾 Yedek")).first
    section.scroll_into_view_if_needed()
    if section.locator("details").get_attribute("open") is None:
        section.locator("summary").click()
    section.get_by_text("Uygulamadaki yedekler").wait_for(timeout=10000)
    return section


def run_round(page, rnd, seed):
    history = []
    team = ["Ali", "Ayşe", "Can", "Deniz", "Ece", "Fatih"]
    per_shift = 2
    box = ensure_settings(page, "textbox", "1. kişi")
    box.fill(", ".join(team))
    box.press("Tab")
    wait_idle(page)

    def click_mode(label):
        page.get_by_text(label, exact=True).click()

    def act_cell():
        click_mode("✏️ Nöbet Ata/Kaldır")
        wait_idle(page)
        rows = grid(page)
        r = rnd.choice(rows)
        d = rnd.randrange(len(r["cells"]))
        history.append(f"hücre({r['name']},{d+1})")
        row = page.locator(".st-key-schedule-grid [data-testid='stHorizontalBlock']").filter(
            has=page.locator("b", has_text=re.compile(f"^{re.escape(r['name'])}$"))).first
        row.locator(":scope > [data-testid='stColumn']").nth(d + 1).get_by_role("button").click()

    def act_pref():
        click_mode("🎨 Tercih Belirle")
        wait_idle(page)
        color = rnd.choice(["⬜Nötr", "🟩Tercih", "🟨Kaçın", "🟥İzin", "🟪Müsait Değil"])
        page.get_by_role("button", name=color).click()
        wait_idle(page)
        rows = grid(page)
        r = rnd.choice(rows)
        d = rnd.randrange(len(r["cells"]))
        history.append(f"tercih({color},{r['name']},{d+1})")
        row = page.locator(".st-key-preference-grid [data-testid='stHorizontalBlock']").filter(
            has=page.locator("b", has_text=re.compile(f"^{re.escape(r['name'])}$"))).first
        row.locator(":scope > [data-testid='stColumn']").nth(d + 1).get_by_role("button").click()

    def act_undo():
        label = rnd.choice(["↩️ Geri", "↪️ İleri"])
        btn = page.get_by_role("button", name=label)
        if btn.is_enabled():
            history.append(label)
            btn.click()

    def act_sim():
        history.append("simülasyon")
        page.get_by_role("button", name="⚡ Simülasyon").click()
        wait_idle(page, limit=90)
        if page.get_by_role("button", name="Eksik sonuçla devam et").count():
            choice = rnd.choice(["Eksik sonuçla devam et", "Mevcut çizelgeyi koru"])
            history.append(choice)
            page.get_by_role("button", name=choice).click()

    def act_improve():
        button = page.get_by_role("button", name="✨ İyileştir")
        if not button.count() or not button.is_enabled():
            return
        history.append("iyileştir")
        button.click()
        wait_idle(page, limit=90)

    def act_cal_swap():
        page.get_by_role("tab", name="📅 Takvim Görünümü").click()
        buttons = page.get_by_role("button", name="🔄", exact=True)
        n = buttons.count()
        if n:
            i = rnd.randrange(n)
            history.append(f"takvim-değiştir({i})")
            buttons.nth(i).click()
            wait_idle(page)
        page.get_by_role("tab", name="🗂️ Tablo Görünümü").click()

    def act_redistribute():
        history.append("otomatik-dağıt")
        page.get_by_role("button", name=re.compile(r"^🤖")).first.click()

    def act_team():
        nonlocal team
        choice = rnd.choice(["ekle", "çıkar", "yeniden-adlandır", "sırala"])
        new = list(team)
        if choice == "ekle" and len(new) < 9:
            new.append(rnd.choice(["Gül", "Hakan", "İlker", "Oya", "Şule"]) + str(rnd.randint(1, 9)))
        elif choice == "çıkar" and len(new) > 4:
            new.pop(rnd.randrange(len(new)))
        elif choice == "yeniden-adlandır":
            i = rnd.randrange(len(new))
            new[i] = new[i] + "x"
        else:
            rnd.shuffle(new)
        new = list(dict.fromkeys(new))
        history.append(f"ekip-{choice}({', '.join(new)})")
        set_team(page, new)
        team = new

    def act_text(label, values, tag):
        value = rnd.choice(values)
        history.append(f"{tag}={value!r}")
        role = "textbox"
        box = ensure_settings(page, role, label)
        box.fill(value)
        box.press("Tab")

    def act_holidays():
        act_text("🎉 Tatiller (bu ay):", ["", "1", "5-7", "29", "1, 15, 31", "abc", "10-3"], "tatil")

    def act_roles():
        act_text("🏷️ Görev İsimleri:", ["", "AYB, GYB", "AYB, AYB", "Tarih", "A, B, C"], "görev")

    def act_limits():
        names = [n for n in team if n]
        if not names:
            return
        row = rnd.randrange(min(2, len(names))) + 1
        kind = rnd.choice(["en fazla nöbet", "en fazla hafta sonu"])
        box = ensure_settings(page, "spinbutton", f"{row}. kişi {kind}")
        value = rnd.choice(["", "1", "2", "4", "8"])
        history.append(f"limit({names[row - 1]},{kind})={value!r}")
        box.fill(value)
        box.press("Tab")

    def act_pairs():
        names = [n for n in team[:4] if n]
        if len(names) < 2:
            return
        # Birlikte Çalışamayan tablosu her ikiliyi bir kez (ekip sırasıyla) gösterir.
        first, second = sorted(rnd.sample(names, 2), key=names.index)
        history.append(f"yasak={first!r}-{second!r}")
        ensure_settings(page, "textbox", "1. kişi")
        box = page.get_by_role("checkbox", name=f"{first} ile {second}", exact=True)
        if box.count():
            page.locator("[data-testid='stCheckbox']").filter(has=box).first.locator("label").click()

    def act_month():
        box = ensure_settings(page, "combobox", "📆 Ay")
        month = rnd.randint(1, 12)
        history.append(f"ay={month}")
        state["month"] = None
        box.click()
        page.get_by_role("option", name=TR_AY_ADLARI[month], exact=True).click()

    def act_people_per_shift():
        nonlocal per_shift
        ensure_settings(page, "textbox", "1. kişi")
        slider = page.get_by_role("slider").first
        key = rnd.choice(["ArrowRight", "ArrowLeft"])
        slider.focus()
        slider.press(key)
        time.sleep(0.5)
        try:
            per_shift = int(slider.get_attribute("aria-valuenow"))
        except (TypeError, ValueError):
            pass
        history.append(f"kişi/nöbet={per_shift}")

    def act_reset():
        history.append("sıfırla")
        page.get_by_role("button", name="🔄 Sıfırla").click()
        wait_idle(page)
        r = rnd.random()
        if r < 0.35:
            page.get_by_role("button", name="🧹 Yalnız nöbetleri sil").click()
        elif r < 0.7:
            page.get_by_role("button", name="🗑️ Tercih ve izinleri de sil").click()
        else:
            page.get_by_role("button", name="Vazgeç").click()

    # Yedek bölümü: ay yedeği (💾 Yedekle) ve geri yükleme. Beklenen nöbetler
    # yalnızca ekip ve ay aynıyken karşılaştırılır (ay yedeği ekibi içermez).
    backups = {}
    state = {"month": None, "undo": None}

    def act_backup():
        history.append("yedekle")
        page.get_by_role("button", name="💾 Yedekle").click()
        notice = page.get_by_text(re.compile(r"\S+ \d{4} (yedeklendi;|için henüz)")).first
        notice.wait_for(timeout=20000)
        m = re.search(r"(\S+ \d{4}) (yedeklendi;|için henüz)", notice.inner_text())
        wait_idle(page)
        state["month"] = m.group(1)
        if m.group(2) == "yedeklendi;":
            click_mode("✏️ Nöbet Ata/Kaldır")
            wait_idle(page)
            backups[m.group(1)] = (list(team), assigned_cells(page))

    def restore(option_text, expected, tag):
        before = assigned_cells(page)
        section = open_backup_section(page)
        section.get_by_role("combobox", name="Yedek seçin").click()
        page.get_by_role("option").filter(has_text=option_text).first.click()
        wait_idle(page)
        # Seçim sayfayı yeniden çizer; bölüm kapanmış olabilir.
        section = open_backup_section(page)
        section.get_by_text("Mevcut kayıtların üzerine yazılsın").first.click()
        wait_idle(page)
        history.append(tag)
        section.get_by_role("button", name="♻️ Bu yedeğe dön").click()
        page.get_by_text("✅ Yedek geri yüklendi").wait_for(timeout=30000)
        wait_idle(page)
        click_mode("✏️ Nöbet Ata/Kaldır")
        wait_idle(page)
        after = assigned_cells(page)
        if expected is not None and after != expected[1]:
            note("YEDEKTEN DÖNÜŞ FARKLI",
                 f"{tag}: beklenen {sorted(expected[1])[:12]} ama {sorted(after)[:12]}", history)
        state["undo"] = (state["month"], (list(team), before))

    def act_restore_backup():
        month = state["month"]
        if month not in backups:
            return
        saved_team, cells = backups[month]
        restore(f"{month} ·", (saved_team, cells) if saved_team == team else None, f"ay-yedeğine-dön({month})")

    def act_undo_restore():
        if not state["undo"] or state["undo"][0] != state["month"]:
            return
        _, (saved_team, cells) = state["undo"]
        restore("Son geri yüklemeden önceki hâl", (saved_team, cells) if saved_team == team else None,
                "geri-yükleme-öncesine-dön")

    def act_reload():
        before = snapshot(page)
        before_share = share_text(page)
        history.append("sayfa-yenile")
        page.reload()
        wait_idle(page, limit=60)
        page.get_by_text("✏️ Nöbet Ata/Kaldır", exact=True).click()
        wait_idle(page)
        after = snapshot(page)
        if after != before:
            note("YENİLEMEDE KAYIP", f"önce {before[:300]} ... sonra {after[:300]}", history)
        try:
            after_share = share_text(page)
            if after_share != before_share:
                note("YENİLEMEDE LİSTE DEĞİŞTİ", f"önce:\n{before_share[:500]}\nsonra:\n{after_share[:500]}", history)
        except Exception as exc:
            note("PAYLAŞIM METNİ OKUNAMADI", repr(exc)[:200], history)

    actions = [
        (act_cell, 30), (act_pref, 10), (act_undo, 10), (act_sim, 6), (act_improve, 3), (act_cal_swap, 6),
        (act_redistribute, 4), (act_team, 5), (act_holidays, 4), (act_roles, 3), (act_limits, 3),
        (act_pairs, 3), (act_month, 3), (act_people_per_shift, 3), (act_reset, 2), (act_reload, 6),
        (act_backup, 3), (act_restore_backup, 3), (act_undo_restore, 2),
    ]
    funcs = [a for a, _ in actions]
    weights = [w for _, w in actions]
    for step in range(STEPS):
        action = rnd.choices(funcs, weights)[0]
        try:
            action()
        except Exception as exc:
            history.append(f"({action.__name__} uygulanamadı: {type(exc).__name__})")
        if not wait_idle(page, limit=60):
            note("SONSUZ ÇİZİM", "sayfa 60 sn boyunca durmadı", history)
        check_invariants(page, history, per_shift)
    return history


def test_maymun(app_url):
    browser_path = os.environ.get("CHROMIUM_EXECUTABLE") or shutil.which("chromium")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=browser_path, args=["--no-sandbox"])
        for round_no in range(ROUNDS):
            seed = 1000 + round_no
            rnd = random.Random(seed)
            context = browser.new_context(viewport={"width": 1700, "height": 1100},
                                          color_scheme=rnd.choice(["light", "dark"]))
            page = context.new_page()
            page.on("console", lambda m: m.type == "error" and "favicon" not in m.text and
                    findings.append(f"[KONSOL] {m.text[:300]}"))
            page.on("pageerror", lambda e: findings.append(f"[SAYFA HATASI] {str(e)[:300]}"))
            page.goto(app_url)
            wait_idle(page, limit=60)
            try:
                history = run_round(page, rnd, seed)
                findings.append(f"--- tur {round_no} (seed {seed}) bitti: {len(history)} işlem")
            except Exception:
                findings.append(f"[TUR ÇÖKTÜ] tur {round_no}: {traceback.format_exc()[-800:]}")
            context.close()
            write_findings()
        browser.close()
    log = (APP_DIR / "server.log").read_text(errors="replace") if (APP_DIR / "server.log").exists() else ""
    errors = [line for line in log.splitlines() if "Error" in line or "Traceback" in line or "error:" in line]
    if errors:
        findings.append("[SUNUCU GÜNLÜĞÜ]\n" + "\n".join(errors[:40]))
    write_findings()


def write_findings():
    """Bulguları tekrarları birleştirerek dosyaya yazar (her turdan sonra)."""
    counts, unique = {}, []
    for entry in findings:
        key = entry.split("\n")[0][:120]
        counts[key] = counts.get(key, 0) + 1
        if counts[key] == 1:
            unique.append(entry)
    out = []
    for entry in unique:
        n = counts[entry.split("\n")[0][:120]]
        out.append(entry + (f"\n  (×{n})" if n > 1 else ""))
    FINDINGS.write_text("\n\n".join(out) or "Bulgu yok", encoding="utf-8")
