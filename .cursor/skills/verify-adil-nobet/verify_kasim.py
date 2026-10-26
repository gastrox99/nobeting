#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adil Nöbet doğrulama yardımcısı.

Kasım 2026 örnek ekibini kullanıcı gibi sürer (ui) ya da aynı kısıtlarla
motoru çalıştırır (engine). Çıktı kanıt klasörüne yazılır; temizlik o
klasörü silmez.

  python .cursor/skills/verify-adil-nobet/verify_kasim.py ui
  python .cursor/skills/verify-adil-nobet/verify_kasim.py engine --repo .
  python .cursor/skills/verify-adil-nobet/verify_kasim.py doctor --port 8501 --pid 123
  python .cursor/skills/verify-adil-nobet/verify_kasim.py cleanup --pid 123
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

NAMES = [
    "Ayşe Demir", "Mehmet Kaya", "Zeynep Arslan", "Can Öztürk", "Elif Şahin",
    "Burak Yıldız", "Selin Acar", "Emre Koç", "Deniz Aksoy", "Işık Polat",
]
YEAR, MONTH = 2026, 11
SEED = 202611
EVIDENCE_DEFAULT = Path("/tmp/verify-adil-nobet-evidence")
SCRATCH_DEFAULT = Path("/tmp/verify-adil-nobet-run")

# İnceleme raporundaki kısıtlar. Önceki ay (31 Ekim) veritabanı ister;
# arayüz yolu onu uygulayamaz, motor yolu --onceki-ay ile uygular.
QUICK_ENTRIES = [
    ("Ayşe Demir", "İzin", "16-22", 7),
    ("Mehmet Kaya", "Müsait Değil", "7-8", 2),
    ("Zeynep Arslan", "Tercih", "6,13,20,27", 4),
    ("Zeynep Arslan", "Kaçın", "28-29", 2),
    ("Can Öztürk", "Kaçın", "2,9,16", 3),
]
FORBIDDEN_LABEL = "Burak Yıldız ile Selin Acar"
TOGETHER_LABEL = "Emre Koç ile Deniz Aksoy birlikte"


def _repo_on_path(repo):
    root = str(Path(repo).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _scenario(repo, onceki_ay):
    _repo_on_path(repo)
    import pandas as pd
    from nobet_core import build_gun_detaylari, official_holidays

    holidays = [day for day, _ in official_holidays(YEAR, MONTH)]
    gun = build_gun_detaylari(YEAR, MONTH, 30, holidays)
    cols = list(gun)
    by_day = {gun[c]["day_num"]: c for c in cols}
    pref = pd.DataFrame(0, index=NAMES, columns=cols)
    unwanted = pd.DataFrame(False, index=NAMES, columns=cols)

    def mark(person, days, code, block):
        for day in days:
            pref.at[person, by_day[day]] = code
            if block:
                unwanted.at[person, by_day[day]] = True

    mark("Ayşe Demir", range(16, 23), 3, True)
    mark("Mehmet Kaya", (7, 8), 4, True)
    mark("Zeynep Arslan", (6, 13, 20, 27), 1, False)
    mark("Zeynep Arslan", (28, 29), 2, False)
    mark("Can Öztürk", (2, 9, 16), 2, False)
    limits = {"Işık Polat": {"min": 0, "max": 4, "max_special": 1}}
    initial = {"Ayşe Demir": 0} if onceki_ay else {}
    return {
        "gun": gun, "cols": cols, "pref": pref, "unwanted": unwanted,
        "limits": limits, "initial": initial,
        "forbidden": {("Burak Yıldız", "Selin Acar")},
        "together": {("Emre Koç", "Deniz Aksoy")},
    }


def _days_of(schedule, person, cols, gun):
    return sorted(
        gun[c]["day_num"] for c in cols
        if person in schedule.index and bool(schedule.at[person, c])
    )


def measure(schedule, scenario):
    """Kişi başı sayılar ve inceleme raporundaki adalet ölçüleri."""
    import pandas as pd
    from nobet_core import has_total_limit, is_friday_column, special_cap
    from nobet_iyilestir import schedule_metrics

    gun, cols = scenario["gun"], scenario["cols"]
    people = list(schedule.index)
    limits = scenario["limits"]

    def special(person):
        return sum(
            1 for c in cols
            if bool(schedule.at[person, c]) and (gun[c]["weekend"] or gun[c]["holiday"])
        )

    def friday(person):
        return sum(1 for c in cols if bool(schedule.at[person, c]) and is_friday_column(gun[c]))

    def weekends(person):
        return sum(1 for c in cols if bool(schedule.at[person, c]) and gun[c]["weekend"])

    kisiler = {}
    for person in people:
        kisiler[person] = {
            "toplam": int(schedule.loc[person].sum()),
            "hafta_sonu": weekends(person),
            "ozel": special(person),
            "cuma": friday(person),
            "gunler": _days_of(schedule, person, cols, gun),
        }

    def spread(values):
        return (max(values) - min(values)) if values else 0

    free = [p for p in people if not has_total_limit(limits.get(p, {}))]
    uncapped = [p for p in people if special_cap(limits, p) >= 999]
    totals = [kisiler[p]["toplam"] for p in people]
    free_totals = [kisiler[p]["toplam"] for p in free]
    specials = [kisiler[p]["ozel"] for p in people]
    uncapped_specials = [kisiler[p]["ozel"] for p in uncapped]
    fridays = [kisiler[p]["cuma"] for p in people]

    rest = []
    initial = scenario["initial"]
    for person in people:
        days = kisiler[person]["gunler"]
        for a, b in zip(days, days[1:]):
            if b - a <= 1:
                rest.append({"kisi": person, "gunler": [a, b]})
        prev = initial.get(person)
        if prev is not None and days and days[0] - prev <= 1:
            rest.append({"kisi": person, "onceki": prev, "gun": days[0]})

    avoid = []
    pref = scenario["pref"]
    for person in people:
        for col in cols:
            if bool(schedule.at[person, col]) and int(pref.at[person, col]) == 2:
                avoid.append({"kisi": person, "gun": gun[col]["day_num"]})

    weeks = sorted({gun[c]["week"] for c in cols if gun[c]["weekend"]})
    consecutive = 0
    consecutive_people = {}
    for person in people:
        worked = {
            w for w in weeks
            if any(bool(schedule.at[person, c]) for c in cols if gun[c]["weekend"] and gun[c]["week"] == w)
        }
        links = sum(1 for w in worked if w + 1 in worked)
        consecutive += links
        if links:
            consecutive_people[person] = links

    forbidden_hits = []
    for col in cols:
        who = [p for p in people if bool(schedule.at[p, col])]
        for a, b in scenario["forbidden"]:
            if a in who and b in who:
                forbidden_hits.append({"gun": gun[col]["day_num"], "ikili": [a, b]})

    together_split = []
    for col in cols:
        who = {p for p in people if bool(schedule.at[p, col])}
        for a, b in scenario["together"]:
            if (a in who) != (b in who):
                together_split.append(gun[col]["day_num"])

    missing = [
        gun[c]["day_num"] for c in cols
        if sum(bool(schedule.at[p, c]) for p in people) != 2
    ]
    blocked = []
    for person in people:
        for col in cols:
            if bool(schedule.at[person, col]) and bool(scenario["unwanted"].at[person, col]):
                blocked.append({"kisi": person, "gun": gun[col]["day_num"]})

    engine = schedule_metrics(
        schedule, people, cols, gun, 2, 1,
        person_limits=limits, df_preferred=pref,
        together_pairs=scenario["together"],
    )
    ozet = {
        "toplam_fark_sinirli_haric": spread(free_totals),
        "toplam_fark_herkes": spread(totals),
        "hafta_sonu_fark_sinirli_haric": spread(uncapped_specials),
        "hafta_sonu_fark_herkes": spread(specials),
        "cuma_fark": spread(fridays),
        "kacinma_ihlali": len(avoid),
        "dinlenme_ihlali": len(rest),
        "art_arda_hafta_sonu": consecutive,
        "yasak_ikili": len(forbidden_hits),
        "birlikte_ayrilan_gun": len(together_split),
        "bos_ya_da_eksik_gun": missing,
        "engel_ihlali": blocked,
        "motor_olculeri": {k: engine[k] for k in (
            "unfilled", "total", "special", "friday", "avoid", "consecutive_weekends",
            "together_split", "missed_preference",
        )},
    }
    sert = {
        "izin_ve_musait_degil_bos": not blocked,
        "dinlenme_temiz": not rest,
        "yasak_ikili_ayri": not forbidden_hits,
        "her_gun_iki_kisi": not missing,
        "isik_toplam_en_fazla_4": kisiler.get("Işık Polat", {}).get("toplam", 99) <= 4,
        "isik_hafta_sonu_en_fazla_1": kisiler.get("Işık Polat", {}).get("hafta_sonu", 99) <= 1,
    }
    return {
        "kisiler": kisiler,
        "ozet": ozet,
        "sert_kurallar": sert,
        "kacinma": avoid,
        "dinlenme": rest,
        "art_arda_hafta_sonu_kisi": consecutive_people,
        "birlikte_ayrilan_gunler": together_split,
    }


def _write_roster_csv(path, schedule, scenario):
    import pandas as pd
    gun, cols = scenario["gun"], scenario["cols"]
    rows = []
    for col in cols:
        who = [p for p in schedule.index if bool(schedule.at[p, col])]
        rows.append({
            "Tarih": gun[col]["full_date"],
            "Servis": who[0] if who else "-",
            "Acil": who[1] if len(who) > 1 else "-",
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def run_engine(repo, evidence, seed, sims, onceki_ay):
    _repo_on_path(repo)
    os.environ["NOBET_SEED"] = str(seed)
    import random
    random.seed(seed)
    from nobet_core import run_scheduling_core
    from nobet_iyilestir import improve_schedule

    scenario = _scenario(repo, onceki_ay)
    started = time.time()
    schedule, score = run_scheduling_core(
        NAMES, scenario["cols"], scenario["unwanted"], scenario["gun"],
        1, 2,
        forbidden_pairs=scenario["forbidden"],
        person_limits=scenario["limits"],
        df_preferred=scenario["pref"],
        simulation_count=sims,
        initial_last_shift=scenario["initial"] or None,
        together_pairs=scenario["together"],
    )
    sim = measure(schedule, scenario)
    improved, before, after = improve_schedule(
        schedule, NAMES, scenario["cols"], scenario["unwanted"], scenario["gun"],
        1, 2,
        forbidden_pairs=scenario["forbidden"],
        person_limits=scenario["limits"],
        df_preferred=scenario["pref"],
        initial_last_shift=scenario["initial"] or None,
        together_pairs=scenario["together"],
        time_limit=8,
    )
    result = improved if improved is not None else schedule
    iyilestir = measure(result, scenario)
    evidence.mkdir(parents=True, exist_ok=True)
    _write_roster_csv(evidence / "engine-simulasyon.csv", schedule, scenario)
    _write_roster_csv(evidence / "engine-iyilestir.csv", result, scenario)
    payload = {
        "mod": "engine",
        "seed": seed,
        "simulasyon_sayisi": sims,
        "repo": str(Path(repo).resolve()),
        "sure_saniye": round(time.time() - started, 1),
        "skor": score,
        "onceki_ay_uygulandi": bool(onceki_ay),
        "onceki_ay_neden": None if onceki_ay else "Bayrak kapalı; 31 Ekim taşınmadı.",
        "iyilestir_degisti": improved is not None,
        "iyilestir_motor_oncesi": None if before is None else {k: before[k] for k in ("total", "special", "friday")},
        "iyilestir_motor_sonrasi": None if after is None else {k: after[k] for k in ("total", "special", "friday")},
        "simulasyon": sim,
        "iyilestir": iyilestir,
    }
    (evidence / "engine.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "kanit": str(evidence / "engine.json"),
        "isik_hafta_sonu": [
            sim["kisiler"]["Işık Polat"]["hafta_sonu"],
            iyilestir["kisiler"]["Işık Polat"]["hafta_sonu"],
        ],
        "toplam_fark_herkes": [sim["ozet"]["toplam_fark_herkes"], iyilestir["ozet"]["toplam_fark_herkes"]],
        "hafta_sonu_fark_herkes": [
            sim["ozet"]["hafta_sonu_fark_herkes"], iyilestir["ozet"]["hafta_sonu_fark_herkes"],
        ],
        "hafta_sonu_fark_sinirli_haric": [
            sim["ozet"]["hafta_sonu_fark_sinirli_haric"],
            iyilestir["ozet"]["hafta_sonu_fark_sinirli_haric"],
        ],
        "cuma_fark": [sim["ozet"]["cuma_fark"], iyilestir["ozet"]["cuma_fark"]],
        "kacinma": [sim["ozet"]["kacinma_ihlali"], iyilestir["ozet"]["kacinma_ihlali"]],
        "dinlenme": [sim["ozet"]["dinlenme_ihlali"], iyilestir["ozet"]["dinlenme_ihlali"]],
        "art_arda": [sim["ozet"]["art_arda_hafta_sonu"], iyilestir["ozet"]["art_arda_hafta_sonu"]],
        "sert": iyilestir["sert_kurallar"],
    }, ensure_ascii=False, indent=2))
    hard_ok = all(sim["sert_kurallar"].values()) and all(iyilestir["sert_kurallar"].values())
    return 0 if hard_ok else 1


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_http(port, deadline):
    url = f"http://127.0.0.1:{port}/_stcore/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def launch(repo, evidence, scratch, seed):
    port = _free_port()
    scratch.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    log_path = scratch / "streamlit.log"
    log = open(log_path, "w", encoding="utf-8")
    env = os.environ.copy()
    env["NOBET_SEED"] = str(seed)
    env.pop("DATABASE_URL", None)
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["PATH"] = os.path.expanduser("~/.local/bin") + os.pathsep + env.get("PATH", "")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "nobet.py",
            "--server.port", str(port),
            "--server.address", "127.0.0.1",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=str(Path(repo).resolve()),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (evidence / "streamlit.pid").write_text(str(proc.pid), encoding="utf-8")
    (evidence / "streamlit.port").write_text(str(port), encoding="utf-8")
    ready = _wait_http(port, time.time() + 40)
    if not ready:
        cleanup(proc.pid)
        raise SystemExit(f"Streamlit açılmadı. Günlük: {log_path}")
    print(json.dumps({"port": port, "pid": proc.pid, "log": str(log_path)}))
    return proc.pid, port


def doctor(port, pid):
    url = f"http://127.0.0.1:{port}/_stcore/health"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            body = resp.read(200).decode("utf-8", "replace")
            ok = resp.status == 200
    except Exception as exc:
        print(json.dumps({"ok": False, "neden": str(exc), "port": port, "pid": pid}))
        return 1
    alive = True
    if pid:
        try:
            os.kill(pid, 0)
        except OSError:
            alive = False
    owns = False
    if pid:
        try:
            out = subprocess.check_output(["ss", "-ltnp"], text=True)
            owns = f"pid={pid}" in out and f":{port}" in out
        except Exception:
            owns = alive
    report = {"ok": ok and alive, "port": port, "pid": pid, "canli": alive, "port_bizim": owns, "saglik": body.strip()}
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["ok"] else 1


def cleanup(pid):
    """Yalnızca bu koşunun başlattığı süreci kapatır. Kanıt klasörüne dokunmaz."""
    if not pid:
        print(json.dumps({"temizlik": "pid yok"}))
        return 0
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(json.dumps({"temizlik": "zaten kapalı", "pid": pid}))
        return 0
    except PermissionError:
        os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 8
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            print(json.dumps({"temizlik": "kapatıldı", "pid": pid}))
            return 0
        time.sleep(0.2)
    try:
        os.killpg(pid, signal.SIGKILL)
    except Exception:
        os.kill(pid, signal.SIGKILL)
    print(json.dumps({"temizlik": "zorla kapatıldı", "pid": pid}))
    return 0


def _wait_idle(page, seconds=1.5, deadline=25):
    status = page.locator("[data-testid='stStatusWidget']")
    end = time.time() + deadline
    quiet = None
    while time.time() < end:
        running = status.count() and status.first.is_visible()
        if running:
            quiet = None
        elif quiet is None:
            quiet = time.time()
        elif time.time() - quiet >= seconds:
            return
        time.sleep(0.1)
    raise AssertionError("Sayfa durmadan yeniden çiziliyor")


def _ensure_quick(page):
    if page.get_by_text("⚡ Hızlı giriş: kişi + günler").count() == 0:
        page.get_by_text("🎨 Tercih Belirle", exact=True).click()
        _wait_idle(page, 0.8)
    if not page.get_by_role("textbox", name="Günler").is_visible():
        page.get_by_text("⚡ Hızlı giriş: kişi + günler").click()
        _wait_idle(page, 0.8)


def _pick(page, root, option):
    """Streamlit seçim kutusu. Menü açılınca seçenek tıklanır; kutunun kendi
    tıklaması menü yüzünden yarım kalabilir."""
    select = page.locator(root).locator("[data-baseweb='select']")
    select.scroll_into_view_if_needed()
    option_node = page.get_by_role("option", name=option, exact=True)
    try:
        select.click(timeout=4000)
    except Exception:
        if not option_node.count():
            raise
    option_node.first.wait_for(timeout=8000)
    option_node.first.click()
    _wait_idle(page, 0.8, deadline=30)


def _quick(page, person, kind, days, changed):
    _ensure_quick(page)
    _pick(page, ".st-key-quick_entry_person", person)
    # Kişi değişince panel kapanabiliyor; tür kutusu yeniden açılır.
    _ensure_quick(page)
    _pick(page, ".st-key-quick_entry_kind", kind)
    page.get_by_role("textbox", name="Günler").fill(days)
    page.get_by_role("button", name="Uygula").click()
    page.get_by_text(f"{person}: {changed} gün").wait_for(timeout=20000)
    _wait_idle(page, 0.8)


def _check(page, label):
    box = page.get_by_role("checkbox", name=label, exact=True)
    label_node = page.locator("[data-testid='stCheckbox']").filter(has=box).first.locator("label")
    label_node.scroll_into_view_if_needed()
    label_node.click()
    _wait_idle(page, 1.2, deadline=30)
    if not box.is_checked():
        raise AssertionError(f"İşaretlenmedi: {label}")


def _download_csv(page, dest):
    page.get_by_role("button", name="📤 Dışa aktar").click()
    button = page.get_by_role("button", name="📄 CSV")
    button.wait_for()
    with page.expect_download(timeout=20000) as download:
        button.click()
    download.value.save_as(str(dest))
    page.keyboard.press("Escape")
    _wait_idle(page, 0.4, deadline=15)


def _schedule_from_csv(path, scenario):
    import pandas as pd
    table = pd.read_csv(path)
    gun, cols = scenario["gun"], scenario["cols"]
    by_day = {gun[c]["day_num"]: c for c in cols}
    schedule = pd.DataFrame(False, index=NAMES, columns=cols)
    role_cols = [c for c in table.columns if c != "Tarih"]
    for _, row in table.iterrows():
        text = str(row["Tarih"])
        day = int(text.split(".")[0])
        col = by_day[day]
        for role in role_cols:
            name = str(row[role]).strip()
            if name and name not in ("-", "nan"):
                if name not in schedule.index:
                    raise AssertionError(f"CSV'de tanımsız isim: {name}")
                schedule.at[name, col] = True
    return schedule


def run_ui(repo, evidence, scratch, seed):
    pid, port = launch(repo, evidence, scratch, seed)
    code = 0
    try:
        if doctor(port, pid) != 0:
            raise SystemExit("Doktor başarısız")
        scenario = _scenario(repo, onceki_ay=False)
        from playwright.sync_api import expect, sync_playwright
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True, channel="chrome", args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            page = browser.new_context(
                viewport={"width": 1600, "height": 1000}, accept_downloads=True,
            ).new_page()
            try:
                page.goto(f"http://127.0.0.1:{port}", wait_until="domcontentloaded")
                page.get_by_text("⚙️ Ayarlar").wait_for(timeout=30000)
                _wait_idle(page, 2.0, deadline=40)
                if "Kasım 2026" not in page.locator("body").inner_text():
                    raise AssertionError("Açılış ayı Kasım 2026 değil: " + page.title())
                box = page.get_by_role("textbox", name="1. kişi", exact=True)
                box.fill(", ".join(NAMES))
                box.press("Tab")
                _wait_idle(page, 1.5, deadline=40)
                expect(page.get_by_role("textbox", name="10. kişi", exact=True)).to_have_value("Işık Polat")
                most = page.get_by_role("spinbutton", name="10. kişi en fazla nöbet", exact=True)
                most.fill("4")
                most.press("Tab")
                _wait_idle(page, 1.2, deadline=30)
                weekend = page.get_by_role("spinbutton", name="10. kişi en fazla hafta sonu", exact=True)
                weekend.fill("1")
                weekend.press("Tab")
                _wait_idle(page, 1.2, deadline=30)
                roles = page.get_by_role("textbox", name="🏷️ Görev İsimleri:")
                roles.fill("Servis, Acil")
                roles.press("Enter")
                _wait_idle(page, 1.2, deadline=30)
                for person, kind, days, changed in QUICK_ENTRIES:
                    _quick(page, person, kind, days, changed)
                _check(page, FORBIDDEN_LABEL)
                _check(page, TOGETHER_LABEL)
                page.screenshot(path=str(evidence / "01-girdiler.png"), full_page=True)
                page.get_by_role("button", name="⚡ Simülasyon").click()
                _wait_idle(page, 2.0, deadline=180)
                page.get_by_text("Kurallar uygun", exact=False).wait_for(timeout=20000)
                page.screenshot(path=str(evidence / "02-simulasyon.png"), full_page=True)
                _download_csv(page, evidence / "ui-simulasyon.csv")
                improve = page.get_by_role("button", name="✨ İyileştir")
                expect(improve).to_be_enabled()
                improve.click()
                page.get_by_text("Liste iyileştirildi").or_(
                    page.get_by_text("Daha iyi bir liste bulunamadı")
                ).first.wait_for(timeout=60000)
                _wait_idle(page, 1.5, deadline=40)
                notice = page.get_by_text("Liste iyileştirildi").or_(
                    page.get_by_text("Daha iyi bir liste bulunamadı")
                ).first.inner_text()
                page.screenshot(path=str(evidence / "03-iyilestir.png"), full_page=True)
                _download_csv(page, evidence / "ui-iyilestir.csv")
                (evidence / "03-iyilestir.txt").write_text(notice, encoding="utf-8")
            except Exception:
                try:
                    page.screenshot(path=str(evidence / "hata.png"), full_page=True)
                except Exception:
                    pass
                raise
            finally:
                browser.close()
        sim_df = _schedule_from_csv(evidence / "ui-simulasyon.csv", scenario)
        imp_df = _schedule_from_csv(evidence / "ui-iyilestir.csv", scenario)
        sim = measure(sim_df, scenario)
        iyilestir = measure(imp_df, scenario)
        payload = {
            "mod": "ui",
            "seed": seed,
            "port": port,
            "onceki_ay_uygulandi": False,
            "onceki_ay_neden": (
                "Veritabanı yok. 31 Ekim Cumartesi nöbeti arayüzden bu aya taşınamadı; "
                "yol zayıflatılmadı, önkoşul karşılanmadı."
            ),
            "iyilestir_bildirimi": (evidence / "03-iyilestir.txt").read_text(encoding="utf-8"),
            "simulasyon": sim,
            "iyilestir": iyilestir,
        }
        (evidence / "ui.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        hard_ok = all(sim["sert_kurallar"].values()) and all(iyilestir["sert_kurallar"].values())
        print(json.dumps({
            "kanit": str(evidence / "ui.json"),
            "isik_hafta_sonu": [
                sim["kisiler"]["Işık Polat"]["hafta_sonu"],
                iyilestir["kisiler"]["Işık Polat"]["hafta_sonu"],
            ],
            "toplamlar": {p: sim["kisiler"][p]["toplam"] for p in NAMES},
            "sert": {"simulasyon": sim["sert_kurallar"], "iyilestir": iyilestir["sert_kurallar"]},
            "bildirim": payload["iyilestir_bildirimi"],
        }, ensure_ascii=False, indent=2))
        code = 0 if hard_ok else 1
    except Exception:
        code = 1
        raise
    finally:
        cleanup(pid)
        if evidence.exists():
            print(json.dumps({"kanit_duruyor": sorted(p.name for p in evidence.iterdir())}))
    return code


def main():
    parser = argparse.ArgumentParser(description="Adil Nöbet Kasım 2026 doğrulaması")
    parser.add_argument("komut", choices=["ui", "engine", "doctor", "cleanup", "launch"])
    parser.add_argument("--repo", default=".")
    parser.add_argument("--evidence", default=str(EVIDENCE_DEFAULT))
    parser.add_argument("--scratch", default=str(SCRATCH_DEFAULT))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--onceki-ay", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()
    evidence = Path(args.evidence)
    if args.komut == "engine":
        return run_engine(args.repo, evidence, args.seed, args.sims, args.onceki_ay)
    if args.komut == "ui":
        return run_ui(args.repo, evidence, Path(args.scratch), args.seed)
    if args.komut == "launch":
        launch(args.repo, evidence, Path(args.scratch), args.seed)
        return 0
    if args.komut == "doctor":
        return doctor(args.port, args.pid)
    if args.komut == "cleanup":
        return cleanup(args.pid)
    return 2


if __name__ == "__main__":
    sys.exit(main())
