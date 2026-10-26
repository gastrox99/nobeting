# Hızlı giriş

Kullanıcı bir kişiyi, bir türü ve günleri yazıp tek seferde izin, müsait değil, tercih veya kaçınma işaretler.

## Sub-features

- `hizli-izin` Ayşe Demir 16–22 izin.
- `hizli-musait-degil` Mehmet Kaya 7–8 müsait değil.
- `hizli-tercih` Zeynep Arslan 6, 13, 20, 27 tercih.
- `hizli-kacin` Zeynep Arslan 28–29 ve Can Öztürk 2, 9, 16 kaçın.

## How to get to it (user POV)

- `🎨 Tercih Belirle` seç.
- `⚡ Hızlı giriş: kişi + günler` başlığını aç.
- Kişi kutusundan ismi seç, günleri yaz, türü seç, `Uygula`'ya bas.

## Driving it with verify_kasim.py

Preconditions:

- Ekip on kişi olarak yazılmış. `10. kişi` = `Işık Polat`.
- Tercih modu açık değilse önce `🎨 Tercih Belirle`.

- **İzin.** Kişi `Ayşe Demir`, tür `İzin`, günler `16-22`, `Uygula`. Yazı: `Ayşe Demir: 7 gün İzin`.
- **Müsait değil.** Kişi `Mehmet Kaya`, tür `Müsait Değil`, günler `7-8`. Yazı: `Mehmet Kaya: 2 gün Müsait Değil`.
- **Tercih.** Kişi `Zeynep Arslan`, tür `Tercih`, günler `6,13,20,27`. Yazı: `Zeynep Arslan: 4 gün Tercih`.
- **Kaçın.** Aynı kişi, tür `Kaçın`, günler `28-29`. Yazı: `Zeynep Arslan: 2 gün Kaçın`.
- **İkinci kaçın.** Kişi `Can Öztürk`, tür `Kaçın`, günler `2,9,16`. Yazı: `Can Öztürk: 3 gün Kaçın`.

`ui` komutu bu beş girişi simülasyondan önce yapar. Kişi kutusu `.st-key-quick_entry_person`, tür kutusu `.st-key-quick_entry_kind`, gün alanı erişilebilir adı `Günler`.

## Gotchas

- Kişi değişince gün kutusu boşalır. Günleri kişi seçildikten sonra yaz.
- `Uygula` basılmadan Simülasyon'a geçilirse o satır kaydolmaz.
- Tür adları birebir: `İzin`, `Müsait Değil`, `Tercih`, `Kaçın`.
