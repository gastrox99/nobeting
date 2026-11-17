# Drive'a otomatik yedek

Her gece 05:43 (TR) GitHub Actions, veritabanındaki en son "Günlük otomatik" yedeği
Drive klasörüne yazar (kullanıcı başına son 30 dosya). Uygulamanın açılışına etkisi yok.
Dosyalar uygulamanın "Yedek → yükle" bölümüyle geri yüklenebilir.

## Kurulum (bir kez)
1. Drive'da bir klasör açın (ör. "Nobet yedek"). Adres çubuğundaki `folders/` sonrası uzun kodu kopyalayın = KLASOR.
2. script.google.com → Yeni proje → `drive_yedek/Kod.gs` içeriğini yapıştırın.
3. Proje ayarları (dişli) → Komut dosyası özellikleri → ekleyin: `ANAHTAR` = uzun rastgele bir parola, `KLASOR` = 1. adımdaki kod.
4. Dağıt → Yeni dağıtım → Web uygulaması → Yürüten: Ben; Erişim: Herkes → Dağıt. İzinleri onaylayın. Çıkan `.../exec` adresini kopyalayın.
5. GitHub → depo → Settings → Secrets and variables → Actions → New repository secret: `DATABASE_URL` (Neon bağlantı adresi, Streamlit Secrets'takiyle aynı), `DRIVE_YEDEK_URL` (4. adım adresi), `DRIVE_YEDEK_ANAHTAR` (3. adımdaki parola).
6. Actions → "Drive yedeği" → Run workflow. Yeşil olunca klasörde `nobet-yedek-...json` görünür.

Yedek bulunamazsa ya da Drive reddederse iş kırmızı olur, GitHub e-posta atar.
