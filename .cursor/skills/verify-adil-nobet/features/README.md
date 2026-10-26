# Adil Nöbet doğrulama haritası

Kullanıcının gördüğü davranışı doğrulamak için önce bu dizini oku, sonra ilgili feature dosyasını izle.

## Baseline preconditions

- Uygulama `verify_kasim.py launch` ile bu koşuya ait bir portta açılmış olsun. Başkasının açık bıraktığı süreci sürme.
- `NOBET_SEED` dolu olsun (varsayılan `202611`). `DATABASE_URL` olmasın.
- Açılış ayı Kasım 2026 (bugün 26 Ekim 2026 ise uygulama kendiliğinden bu ayı açar).
- `verify_kasim.py doctor --port <port> --pid <pid>` çıkış 0 vermeden tıklama.
- Kanıt `/tmp/verify-adil-nobet-evidence` altında kalsın. Temizlik onu silmesin.

## Driving conventions

- Her tarife, ekip boşken başla.
- Erişilebilir ad kullan. Sıra numarası veya piksel kullanma.
- Komutları `verify_kasim.py` üzerinden çalıştır. `ui` komutu Kasım simülasyon feature'ını uçtan uca sürer.
- Aynı anda iki koşu aynı kanıt klasörüne yazmasın. İkinci koşu için `--evidence` ver.

## Proof and skip reporting

- Tıklamayı ve oluşan listeyi birlikte göster: ekran görüntüsü ve indirilen CSV.
- Kişi başı toplam, hafta sonu, Cuma ile kaçınma, dinlenme, art arda hafta sonu ve farklar `ui.json` içindedir. Farklar iki kez yazılır: sınırlı kişiler hariç ve herkes dahil.
- Feature kimliği dosya adıdır.
- Ulaşılamayan adımı atlanmış sayma. Önkoşulu ve denenen komutu yaz.

## Feature entry contract

Her feature dosyası bir H1 ve tek paragraf, ardından tam olarak şu dört H2 ile biter: `Sub-features`, `How to get to it (user POV)`, `Driving it with verify_kasim.py`, `Gotchas`.

## Features

- [Kasım simülasyonu](./kasim-simulasyon.md) örnek ekibi yükler, Simülasyon'u çalıştırır, CSV ve ölçüleri alır.
- [İyileştir](./iyilestir.md) dolu listede İyileştir'e basar ve bildirimi kaydeder.
- [Hızlı giriş](./hizli-giris.md) kişi ve gün aralığıyla izin, müsait değil, tercih ve kaçınma girer.
- [Ekip limitleri](./ekip-limitleri.md) isim, en fazla nöbet ve en fazla hafta sonu kutularını doldurur.
