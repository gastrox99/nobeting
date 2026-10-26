# Kasım simülasyonu

Kullanıcı Kasım 2026 ekibini, izin ve tercihlerini girer, Simülasyon'a basar ve oluşan listenin CSV'sini indirir.

## Sub-features

- `kasim-ekip` on kişilik örnek ekibi ve Servis/Acil görev adlarını yazar.
- `kasim-kisit` Işık'ın tavanını, hızlı giriş kısıtlarını ve iki ikiliyi işaretler.
- `kasim-simulasyon` Simülasyon'u çalıştırır ve kuralların uygun göründüğünü okur.
- `kasim-csv` kullanıcının indirdiği CSV'den kişi başı ölçüleri çıkarır.

## How to get to it (user POV)

- Uygulamayı aç. Üstte `Kasım 2026` yazar.
- `⚙️ Ayarlar` içindeki `1. kişi` kutusuna ekibi yapıştır.
- `10. kişi en fazla nöbet` ve `10. kişi en fazla hafta sonu` kutularını doldur.
- `🏷️ Görev İsimleri:` alanına görev adlarını yaz.
- `🎨 Tercih Belirle` altındaki `⚡ Hızlı giriş: kişi + günler` ile günleri işaretle.
- `Burak Yıldız ile Selin Acar` ve `Emre Koç ile Deniz Aksoy birlikte` kutularını işaretle.
- `⚡ Simülasyon` düğmesine bas.
- `📤 Dışa aktar` içinden `📄 CSV` indir.

## Driving it with verify_kasim.py

Preconditions:

- `verify_kasim.py doctor` çıkış 0. Sağlık ucu bu koşunun pid'sine ait.
- Ekip boş. Ay `Kasım 2026`.
- `DATABASE_URL` yok. 31 Ekim nöbeti bu feature'da uygulanamaz; bu bir başarısız önkoşuldur, adım listeden çıkarılmaz.

- **Ekibi yaz.** `1. kişi` kutusuna on ismi virgülle yapıştır ve Tab'a bas. `10. kişi` değeri `Işık Polat` olur.
- **Tavan.** `10. kişi en fazla nöbet` = `4`, `10. kişi en fazla hafta sonu` = `1`. Kutular bu değerleri gösterir.
- **Görev.** `🏷️ Görev İsimleri:` = `Servis, Acil`.
- **Kısıtlar.** Hızlı giriş ve ikili feature'larındaki adımlar. Her `Uygula` ilgili kişinin gün sayısını söyler. İki kutu işaretli kalır.
- **Simülasyon.** `⚡ Simülasyon` düğmesine bas. Sayfa durunca `Kurallar uygun` görünür. Kanıt: `02-simulasyon.png`.
- **CSV.** `📤 Dışa aktar`, sonra `📄 CSV`. Dosya `ui-simulasyon.csv`. İçinde 30 gün ve Servis/Acil sütunları vardır.
- **Kanıt.** `ui` komutu `ui.json` yazar. Simülasyon bloğunda kişi başı toplam, hafta sonu, Cuma, kaçınma, dinlenme, art arda hafta sonu ve her iki fark (sınırlı hariç / herkes) bulunur.

Tek komut:

```bash
python .cursor/skills/verify-adil-nobet/verify_kasim.py ui --repo . --seed 202611
```

## Gotchas

- 31 Ekim Cumartesi nöbeti ancak kayıtlı önceki aydan gelir. Veritabanı yokken ay değiştirmek listeyi saklamaz. `ui.json` içinde `onceki_ay_uygulandi` false kalır. Motor ölçüsü için ayrı komut vardır: `engine --onceki-ay`. O komut arayüz kanıtı değildir.
- Simülasyon yaklaşık yüz deneme sürer. Boşta bekleme 20 saniyede kesilirse liste henüz gelmemiştir; süre en az 180 saniye olsun.
- `NOBET_SEED` boşsa aynı tıklama başka liste üretir. Karşılaştırma için tohum şart.
- İndirilen CSV görev yerlerini gösterir. Adalet hesabı kişi ve gün üzerinden yapılır; Servis/Acil yer değişimi toplamı değiştirmez.
