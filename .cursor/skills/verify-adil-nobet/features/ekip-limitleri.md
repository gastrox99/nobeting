# Ekip limitleri

Kullanıcı ekip tablosuna isim yazar, bir kişinin en fazla nöbet ve en fazla hafta sonu tavanını koyar.

## Sub-features

- `ekip-yapistir` virgüllü isim listesi satırlara bölünür.
- `ekip-tavan` onuncu kişinin iki tavan kutusu dolar.
- `ekip-gorev` görev adları Servis ve Acil olur.

## How to get to it (user POV)

- `⚙️ Ayarlar` açıkken `1. kişi` kutusuna isimleri yapıştır.
- `10. kişi en fazla nöbet` ve `10. kişi en fazla hafta sonu` kutularına sayı yaz.
- Sağdaki `🏷️ Görev İsimleri:` alanına `Servis, Acil` yaz.

## Driving it with verify_kasim.py

Preconditions:

- Sayfa ilk açılışta, ekip boş. Ayarlar paneli bu halde açıktır.
- Doktor çıkış 0.

- **Yapıştır.** `1. kişi` kutusuna `Ayşe Demir, Mehmet Kaya, Zeynep Arslan, Can Öztürk, Elif Şahin, Burak Yıldız, Selin Acar, Emre Koç, Deniz Aksoy, Işık Polat` yaz ve Tab'a bas. `10. kişi` = `Işık Polat`. `11. kişi` boştur.
- **Tavan.** `10. kişi en fazla nöbet` kutusuna `4` yaz ve Tab'a bas. Değer `4` kalır. `10. kişi en fazla hafta sonu` kutusuna `1` yaz ve Tab'a bas. Değer `1` kalır.
- **Görev.** `🏷️ Görev İsimleri:` = `Servis, Acil`, Enter. Sonraki CSV başlıkları `Tarih`, `Servis`, `Acil` olur.

`ui` komutu simülasyondan önce bu üç adımı yapar. Ekran `01-girdiler.png` içinde tavanlar görünür.

## Gotchas

- Tavanı isimler satırlara bölünmeden yazmak, sayıyı birinci kişiye bırakır. Önce `10. kişi` dolsun.
- Son boş satırın sayı kutuları kapalıdır. Işık onuncu satırdadır, on birinci değil.
- Tabloda "en az nöbet" kutusu yoktur. Eski kayıttaki en az değer bu tablodan geçerken sıfırlanabilir; bu feature onu girmez.
