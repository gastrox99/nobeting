# nobeting için çalışma kuralları

Bu dosya, bu depoda kod yazan yapay zekâ asistanı (Claude) içindir.
Genel ilkeler Andrej Karpathy'nin yapay zekâyla kodlamadaki sık hatalara dair
gözlemlerinden uyarlanmıştır (kaynak: github.com/multica-ai/andrej-karpathy-skills, MIT).

## Proje

- Nöbet listesi hazırlayan Streamlit uygulaması ("Adil Nöbet"). Python 3.12.
- Canlı site Streamlit Community Cloud'da, `main` dalından otomatik yayınlanır:
  `main`'e giren her şey birkaç dakika içinde kullanıcıya ulaşır.
- Veritabanı Neon PostgreSQL; bağlantı `DATABASE_URL` ile (Streamlit Secrets).
  Şifreler ve `.streamlit/secrets.toml` depoya asla girmez.
- Replit artık kullanılmıyor; Replit'e özgü dosya veya ayar eklenmez.
- Kullanıcı yazılımcı değil, hekim. Açıklamalar sade Türkçe; teknik terim
  gerekiyorsa tek cümleyle açıklanır.

## Korunacak davranışlar

- En fazla 8 kişilik listeler; nöbetler iki ayrı birimde (iki rol sütunu) tutulur,
  birim ayrımı bozulmamalı.
- Nöbetler arası en az aralık "aradaki nöbetsiz gün" sayısı olarak yorumlanır;
  bu yorum değiştirilmez.
- Algoritma (`nobet_core.py`, `nobet_iyilestir.py`) ve veri kaydetme (`db.py`)
  en hassas yerlerdir; buradaki değişiklikler en küçük hâliyle yapılır.

## 1. Kodlamadan önce düşün

- Varsayımlarını açıkça yaz. Emin değilsen tahmin yürütme, sor.
- İstek birden fazla şekilde anlaşılabiliyorsa seçenekleri göster, sessizce birini seçme.
- Daha basit bir yol varsa söyle; gerekiyorsa itiraz et.
- Bir şey anlaşılmıyorsa dur, neyin belirsiz olduğunu söyle.

## 2. Önce sadelik

- Sorunu çözen en az kod. İstenmeyen özellik, ayar veya "ileride lazım olur" esnekliği ekleme.
- Tek yerde kullanılan kod için soyutlama kurma.
- Olamayacak durumlar için hata yakalama yazma.
- 200 satır 50'ye inebiliyorsa kısalt. Ölçüt: deneyimli bir yazılımcı buna
  "gereğinden karmaşık" der mi?

## 3. Cerrahi değişiklik

- Yalnızca istenen yere dokun. Komşu kodu, yorumları veya biçimi "iyileştirme".
- Bozuk olmayanı yeniden düzenleme. Mevcut yazım tarzına uy.
- Alakasız ölü kod görürsen silme, raporda belirt.
- Kendi değişikliğinin kullanılmaz bıraktığı import/değişken/fonksiyonu temizle;
  önceden var olanlara dokunma.
- Ölçüt: değişen her satır doğrudan kullanıcının isteğine bağlanabilmeli.

## 4. Hedefe dayalı çalışma

- İşi doğrulanabilir hedefe çevir: "hatayı düzelt" yerine
  "hatayı gösteren testi yaz, başarısız olduğunu gör, sonra geçir".
- Çok adımlı işlerde kısa bir plan yaz: `adım → nasıl doğrulanacak`.
- Değişiklikten önce ve sonra ilgili testler geçmeli.

## Değişiklik süreci

1. **Önce teşhis:** İlk adımda kodu değiştirme. Bulguları önem sırasıyla raporla
   (sorun ne, kullanıcı ekranda nasıl fark eder, düzeltilmezse ne olur).
   Neyin düzeltileceğine kullanıcı karar verir.
2. **Kanıt:** Hatayı bir test ya da çalıştırmayla göster. Üretilemiyorsa düzeltme;
   "şüphe, kanıtlanamadı" diye işaretle.
3. **Test:** Önce hatayı yakalayan test (başarısız), sonra düzeltme; yeni test ve
   tüm paket geçmeli. Testler `test_*.py`; tarayıcı testleri `test_browser_*.py`
   (Playwright + Chromium). Yerelde: `python -m pytest -p no:cacheprovider`.
4. **Ayrı dal, ayrı PR:** `main`'e doğrudan yazılmaz. Her değişiklik kendi dalında,
   küçük bir PR. CI ("Testler" iş akışı) geçince PR'ı Claude birleştirir;
   CI kırmızıysa birleştirilmez.
5. **Kapanış:** Ne düzeldi (kullanıcının göreceği şekilde), davranışta ne değişti,
   toplam test sayısı ve canlı sitede bir kez denenecek adımlar. Algoritma veya
   veri kaydetmeyi etkileyen büyük değişikliklerden sonra elle başlatılan
   "Maymun testi" iş akışını (yaklaşık 50 dk) öner.

## Denge

Bu kurallar hızdan çok dikkati öne alır. Yazım hatası düzeltmek gibi önemsiz
işlerde sağduyu kullan; amaç pahalı hataları önlemek, basit işleri yavaşlatmak değil.
