# İyileştir

Kullanıcı dolu bir listenin ardından İyileştir'e basar, bildirimi okur ve yeni CSV'yi indirir.

## Sub-features

- `iyilestir-kapali` boş listede düğme kapalıdır.
- `iyilestir-calistir` dolu listede düğme açılır, basılınca bildirim gelir.
- `iyilestir-csv` oluşan listenin CSV'si indirilir.

## How to get to it (user POV)

- Önce Kasım simülasyonunu bitir. Liste dolu olsun.
- `✨ İyileştir` düğmesine bas.
- Yeşil `Liste iyileştirildi` ya da `Daha iyi bir liste bulunamadı` yazısını oku.
- `📤 Dışa aktar` içinden `📄 CSV` indir.

## Driving it with verify_kasim.py

Preconditions:

- Kasım simülasyonu feature'ı bu oturumda bitmiş, `Kurallar uygun` görünür.
- `✨ İyileştir` etkindir.

- **Boş liste.** Ekip yeni yazılmış ve Simülasyon'a basılmamışken `✨ İyileştir` kapalıdır. `ui` komutu bunu ayrıca tıklamaz; düğme ancak listeden sonra kullanılır.
- **Çalıştır.** `✨ İyileştir` düğmesine bas. `Liste iyileştirildi` ya da `Daha iyi bir liste bulunamadı` görünür. Metin `03-iyilestir.txt` dosyasındadır. Ekran `03-iyilestir.png`.
- **CSV.** `📄 CSV` inince dosya `ui-iyilestir.csv` olur. `ui.json` içindeki `iyilestir` bloğu simülasyon bloğuyla aynı ölçüleri taşır.

`ui` komutu bu adımları simülasyondan hemen sonra yapar.

## Gotchas

- Çözücü birkaç saniye ile bir dakika arasında sürer. Bildirim gelmeden CSV indirilirse eski liste kaydolur.
- Bildirimdeki `hafta sonu/tatil farkı`, hafta sonu sınırı olan kişiyi dışarıda bırakabilir. Ekrandaki tablonun en çok ve en az hafta sonu farkıyla aynı sandan önce `ui.json` içindeki `hafta_sonu_fark_herkes` değerine bak.
- Geri düğmesi listeyi eski haline alır. Karşılaştırma CSV'si Geri'ye basılmadan indirilir.
