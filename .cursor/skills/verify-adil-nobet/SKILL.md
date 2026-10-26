---
name: verify-adil-nobet
description: "Adil Nöbet Streamlit arayüzünü kullanıcı gibi sürer: Kasım 2026 örnek ekibini yükler, Simülasyon ve İyileştir'i çalıştırır, CSV ve adalet metriklerini üretir. Arayüz veya nöbet dağıtımı değişince doğrulamak için kullan."
---

# Adil Nöbet doğrulaması

Birincil yüzey Streamlit web arayüzüdür (`streamlit run nobet.py`). Motor (`nobet_core.run_scheduling_core`) aynı senaryoyu tarayıcısız tekrarlar; arayüzün yerini tutmaz. Veritabanı bu doğrulamada yoktur: önceki ay taşınamaz, yol bu yüzden kısaltılmaz.

Kanıt klasörü: `/tmp/verify-adil-nobet-evidence`. Temizlik bu klasörü silmez.
Çalışma artığı: `/tmp/verify-adil-nobet-run` (Streamlit günlüğü).

## Launch

Depo kökünden, boş oturum ve sabit tohumla:

```bash
python .cursor/skills/verify-adil-nobet/verify_kasim.py launch --repo . --seed 202611
```

Komut boş bir port seçer, `NOBET_SEED=202611` ile `python -m streamlit run nobet.py` başlatır, `DATABASE_URL` vermez. Hazır olma: `http://127.0.0.1:<port>/_stcore/health` 200 döner ve günlük `/tmp/verify-adil-nobet-run/streamlit.log` içinde hata yoktur. Port ve pid kanıt klasöründeki `streamlit.port` ve `streamlit.pid` dosyalarına yazılır.

Tek komutla uçtan uca (başlat, doktor, sür, kanıt, kapat):

```bash
python .cursor/skills/verify-adil-nobet/verify_kasim.py ui --repo . --seed 202611
```

Başka bir doğrulama koşusuyla aynı portu paylaşma. Her `launch` / `ui` kendi portunu ve kendi pid'sini kullanır.

## Doctor

```bash
python .cursor/skills/verify-adil-nobet/verify_kasim.py doctor --port <port> --pid <pid>
```

Şunların hepsi doğruysa çıkış 0: sağlık ucu 200, pid canlı, port o pid'ye ait. Başka birinin açık bıraktığı Streamlit'i sürme.

## Drive

Tarayıcı: Playwright + sistemdeki Chrome (`channel=chrome`, `--no-sandbox`). Seçiciler erişilebilir addır.

Kasım 2026 örneği (inceleme raporundaki ekip):

- İsimler, sırayla: Ayşe Demir, Mehmet Kaya, Zeynep Arslan, Can Öztürk, Elif Şahin, Burak Yıldız, Selin Acar, Emre Koç, Deniz Aksoy, Işık Polat.
- Görev adları: `Servis, Acil`. Nöbet başına 2 kişi. Dinlenme arayüzde her zaman 1 boş gün.
- Işık Polat: en fazla 4 nöbet, en fazla 1 hafta sonu.
- Ayşe Demir izin 16–22. Mehmet Kaya müsait değil 7–8. Zeynep Arslan Cuma günlerini tercih (6, 13, 20, 27), 28–29 kaçın. Can Öztürk 2, 9 ve 16 kaçın.
- Yasak ikili: Burak Yıldız ile Selin Acar. Birlikte: Emre Koç ile Deniz Aksoy.
- 31 Ekim Cumartesi (Ayşe) bu koşuda uygulanamaz; aşağıdaki Gotchas.

Adımlar `features/kasim-simulasyon.md` içindedir. `ui` komutu bu feature'ı bir kez sürer: ekibi yapıştırır, limit ve tercihleri girer, Simülasyon'a basar, CSV indirir, İyileştir'e basar, CSV'yi tekrar indirir.

Motor karşılaştırması (aynı kısıt, tarayıcı yok; arayüz kanıtının yerine geçmez):

```bash
python .cursor/skills/verify-adil-nobet/verify_kasim.py engine --repo . --seed 202611 --sims 100
python .cursor/skills/verify-adil-nobet/verify_kasim.py engine --repo . --seed 202611 --onceki-ay
```

`--onceki-ay` Ayşe'nin 31 Ekim nöbetini motora verir. Arayüz yolu bunu yapmaz.

## Evidence

`ui` şunları `/tmp/verify-adil-nobet-evidence` altına yazar:

- `01-girdiler.png` — Simülasyon'dan hemen önce, kısıtlar ekranda.
- `02-simulasyon.png` — liste geldikten sonra.
- `03-iyilestir.png` ve `03-iyilestir.txt` — İyileştir bildirimi.
- `ui-simulasyon.csv`, `ui-iyilestir.csv` — kullanıcının indirdiği CSV (Tarih, Servis, Acil).
- `ui.json` — kişi başı toplam, hafta sonu, Cuma; kaçınma, dinlenme, art arda hafta sonu; sınırlı kişiler dahil ve hariç toplam/hafta sonu farkı; sert kural sonucu; `onceki_ay_uygulandi: false`.

`engine` aynı ölçüleri `engine.json`, `engine-simulasyon.csv`, `engine-iyilestir.csv` olarak yazar. Çıkış 0, ancak her iki listede de izin/müsait değil, dinlenme, yasak ikili, günde iki kişi ve Işık'ın tavanları bozulmamışsa.

## Cleanup

```bash
python .cursor/skills/verify-adil-nobet/verify_kasim.py cleanup --pid <pid>
```

Yalnızca bu pid'nin süreç grubunu kapatır. Süreç adına göre arama yapmaz. Kanıt klasörünü silmez. `ui` komutu bitince temizliği kendisi çağırır; kanıtın durduğu son satırda `kanit_duruyor` listelenir.

## Helpers

Yardımcı: `.cursor/skills/verify-adil-nobet/verify_kasim.py` (çalıştırılabilir).

```bash
python .cursor/skills/verify-adil-nobet/verify_kasim.py ui --repo .
python .cursor/skills/verify-adil-nobet/verify_kasim.py engine --repo . --seed 202611
python .cursor/skills/verify-adil-nobet/verify_kasim.py doctor --port 8501 --pid 123
python .cursor/skills/verify-adil-nobet/verify_kasim.py cleanup --pid 123
```

`--repo` ölçülen kodun köküdür. Başka bir dalı ölçerken script'i bu daldan, `--repo` değerini ölçülen çalışma kopyasından ver.

## Bakım

Harita ile uygulama ayrılırsa `/maintain-verification-skill` ile güncelle. Feature dosyaları: `features/README.md`.
