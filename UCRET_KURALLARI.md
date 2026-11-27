# UCRET_KURALLARI.md — Nöbet fazla mesaisi ve gece farkı hesaplama kuralları

Sürüm: 1.0 · 26.11.2026 · Kaynak: "Gece Mesaisi" projesi (Gece Farkı Hesaplayıcı v1.2, kullanıcı onaylı kurallar)
Kapsam: Türkiye kamu sağlık personeli (657 DMK ek 33), 2026–2027. Örnek birim: yoğun bakım, pratisyen hekim.

> **Bu dosyayı okuyan yapay zekâ için:** Aşağıdaki kurallar kesindir ve kullanıcı (hekim) tarafından tek tek onaylanmıştır.
> Hesap yaparken burada yazmayan bir kural EKLEME, yazanı "daha mantıklı" diye DEĞİŞTİRME. Bir durum burada yoksa
> hesabı durdur ve kullanıcıya sor. Bölüm 10'daki açık sorular kullanıcı onayı olmadan kapatılmaz.
> Algoritmanın tek doğru uygulaması Bölüm 8'deki referans koddur; Bölüm 7'deki test tablosu birebir tutmalıdır.

---

## 1. Tanımlar

| Terim | Tanım |
|---|---|
| **Gündüz** | 08:00–20:00 arası saatler. |
| **Gece** | 20:00–08:00 arası saatler. |
| **Hafta** | Pazartesi 00:00 – Pazar 24:00. Tüm hesap haftalık yapılır. |
| **Zorunlu saat (R)** | Kişinin o hafta (o ay içindeki günlerde) normal mesaide çalışması gereken saat. Hastane raporunda **TCGS**. |
| **Çalışılan saat (A)** | Kişinin o hafta başlayan vardiyalarının toplam süresi. Hastane raporunda **TCS**. |
| **Haftalık fark (E)** | E = A − R. E > 0 ise **fazla**, E < 0 ise **açık**. |
| **FMS** | Aylık fazla mesai = ΣA − ΣR (ay toplamı). |
| **Resmi tatil (tam gün)** | Takvimdeki tam gün tatiller (Bölüm 3.2). Zorunlu saat 0. |
| **Arife** | Bayram arifesi; tatil 13:00'te başlar. Zorunlu saat 5. 13:00–24:00 bayram saatidir. |
| **Genel tatil** | 1 Ocak, 1 Mayıs, 15 Temmuz. Zorunlu saat 0 **ama bayram değildir**: saatleri normal gündüz/gece kovasına yazılır, ücrette %25 yoktur. İstisna: aynı gün dini bayrama (veya başka bir bayrama) denk gelirse o gün bayramdır. |
| **Bayram saati** | Genel tatil dışındaki tam gün resmi tatillerin 00:00–24:00 saatleri + genel tatile denk gelen dini bayram günlerinin 00:00–24:00 saatleri + arifelerin 13:00–24:00 saatleri. Saat bazında belirlenir. |
| **Kovalar** | Her çalışılan saat 4 kovadan birine girer: `nd` normal gündüz, `bd` bayram gündüz, `nn` normal gece, `bn` bayram gece. |
| **Mazeret kodu** | Zorunlu saatten düşen gün kodu (izin, rapor vb., Bölüm 3.3). |
| **Gece farkı** | 40 saati (haftalık zorunlu saati) aşan sürenin 20:00–08:00'e denk gelen kısmına ödenen %10 ek ücret. |

## 2. Girdi formatı

Bir kişi-ay için gereken girdi:

```
yil: int, ay: int (1–12)
hucreler: { gun_no(int 1..31) : kod(str) }    # boş gün anahtarsız
baslama_tarihi: date | null   # ay ortasında göreve başlayan
ayrilma_tarihi: date | null   # ay ortasında ayrılan
unvan: "uzman" | "tabip" | "dis_eczaci" | "hemsire_ebe" | "lise_saglik" | "diger"
riskli_birim: bool            # yoğun bakım, acil, 112, diyaliz, ameliyathane = true
```

- Kaynak: hastane "Vardiya Saatlik Raporu" (.xls). Başlıkta "2026 YILI EKİM AYI …"; "AD SOYAD" satırından sonra her
  kişi bir satır; gün sütunları 1..31; özet sütunları TCS, TCGS, FMS, GECE FMS…
- Hücredeki kod, vardiyanın **başladığı** güne yazılıdır.
- Hastanenin TCGS/TCS değerleri varsa, programın R ve A toplamı bunlarla **birebir** tutmalıdır (doğrulama, Bölüm 6.10).

## 3. Parametreler

### 3.1 Vardiya kodları (başlangıç saati, süre)

| Kod | Başlangıç | Süre | Not |
|---|---|---|---|
| `24` | 08:00 | 24 saat | 12 gündüz + 12 gece (bayram yoksa) |
| `16` | 16:00 | 16 saat | 16–20 gündüz (4) + 20–08 gece (12) |
| `8` (mesai, M) | 08:00 | 8 saat | tamamı gündüz |
| `12`, `12G` | 08:00 | 12 saat | tamamı gündüz — varsayım, hastanede doğrulanmadı |
| `N12`, `12N` | 20:00 | 12 saat | tamamı gece — varsayım, hastanede doğrulanmadı |

### 3.2 Takvim

| Yıl | Tam gün resmi tatil (AA-GG) | Arife (13:00'ten itibaren) |
|---|---|---|
| 2026 | 01-01, 03-20, 03-21, 03-22, 04-23, 05-01, 05-19, 05-27, 05-28, 05-29, 05-30, 07-15, 08-30, 10-29 | 03-19, 05-26, 10-28 |
| 2027 | 01-01, 03-09, 03-10, 03-11, 04-23, 05-01, 05-16, 05-17, 05-18, 05-19, 07-15, 08-30, 10-29 | 03-08, 05-15, 10-28 |

Genel tatil (bayram sayılmaz): **01-01, 05-01, 07-15**. Bir yılın dini bayramları girilmemişse hesap **durur**.

Dini bayram günleri (genel tatille çakışma kontrolü için): 2026 → 03-20, 03-21, 03-22, 05-27, 05-28, 05-29, 05-30;
2027 → 03-09, 03-10, 03-11, 05-16, 05-17, 05-18, 05-19. Genel tatil bir dini bayram gününe denk gelirse o gün **bayram** sayılır
(kullanıcı kararı 27.11.2026). 2026–2027'de böyle bir çakışma yok.

### 3.3 Gün kodları

| Sınıf | Kodlar | Zorunlu saate etkisi | Çalışılan saat |
|---|---|---|---|
| Mazeret (zorunludan düşer) | `Yİz.` yıllık izin, `Rpr.` rapor, `Hastane Yatışı`, `G.G.Gid.` geçici görev, `Bşl.` göreve başlamadı, `Ayr.` ayrıldı | iş günü 0 | 0 |
| Çalışma günü sayılan | `0`, **idari izin** (personel idari izinli sayılmaz) | iş günü 8 (arife 5) | 0 |
| Vardiya | Bölüm 3.1 | iş günü normal zorunlu (8/5/0) | süre kadar |
| Tatil adı | Tatil/arife gününe yazılmış "CUMHURİYET BAYRAMI Y.G." vb. | yok sayılır (gün normal kuralla) | 0 |
| **Tanınmayan kod** | yukarıdakilerden hiçbiri | **varsayılan YOK** | — |

Tanınmayan kodda hesap yapılmaz: kullanıcı kodu "mazeret", "çalışma günü" veya "vardiya (başlangıç, süre)" olarak
sınıflandırana kadar dur. (İpucu: hangi sınıflandırmada hastane TCGS'si tutuyorsa o önerilir, ama karar kullanıcınındır.)

### 3.4 Ücret parametreleri

| Parametre | Değer |
|---|---|
| Memur aylık katsayısı 2026 Ocak–Haziran | 1,387871 |
| Memur aylık katsayısı 2026 Temmuz–Aralık | 1,575512 |
| Nöbet göstergesi (657 ek 33) | uzman 150, tabip 135, diş tabibi/eczacı 120, hemşire/ebe 90, lise mezunu sağlık 75, diğer 55 |
| Toplu sözleşme ilavesi | +10 puan (tüm ünvanlar) |
| Riskli birim artırımı | %50 |
| Resmi/dini bayram artırımı | %25 (genel tatillerde yok) |
| Riskli + bayram | art arda (çarpımsal): ×1,5 × 1,25 = ×1,875 |
| Gece artırımı (8. Dönem TS) | %10, **artırımlı (riskli/bayram) ücretin üzerine** |
| Damga vergisi | binde 7,59 |

Katsayı her Ocak ve Temmuz değişir. Dönemin katsayısı yoksa hesap durur (tahmin edilmez).

## 4. Algoritma (adım adım)

**Adım 0 — Gün kodunu düzelt.** Başlama tarihinden önceki günlerin kodu `Bşl.`, ayrılma tarihinden sonraki günlerin
kodu `Ayr.` kabul edilir (hücrede ne yazarsa yazsın). Başlama/ayrılma gününün kendisi normal işlenir.

**Adım 1 — Her gün için zorunlu saat** (yalnızca o ayın günleri):
```
eğer gün Cumartesi/Pazar           → 0
eğer gün tam gün resmi tatil       → 0   (genel tatiller dahil)
eğer kod mazeret                    → 0
eğer gün arife                      → 5
değilse                             → 8
```

**Adım 2 — Her vardiyayı saat saat kovalara dağıt.** Vardiya başladığı güne aittir (Pazar 08:00'de başlayan 24'ün
Pazartesi 00–08 saatleri de Pazar'ın haftasına ve Pazar'ın ayına yazılır). Her saat dilimi için o saatin **gerçek
takvim günü** ve saati alınır:
```
gece  = (saat >= 20) veya (saat < 8)
bayram = (gün tam gün resmi tatil VE (genel tatil değil VEYA dini bayram günü)) veya (gün arife VE saat >= 13)
kova  = (bayram ? "b" : "n") + (gece ? "n" : "d")
```
Örnek: bayramın son günü 08:00'de başlayan 24 → 08–20 bayram gündüz (12), 20–24 bayram gece (4), 00–08 normal gece (8).

**Adım 3 — Haftalara topla.** Haftalar Pazartesi–Pazar. **Ay sınırını kesen hafta bölünür:** her ay yalnız kendi
günleriyle hesaplanır; zorunlu saate yalnızca o aya düşen hafta içi mesai günleri girer (kullanıcı onayı 27.11.2026;
ör. Perşembe başlayan ayın ilk haftası = Per+Cum = 16 saat zorunlu). Her hafta için
R, A ve 4 kovanın çalışılan saati toplanır.

**Adım 4 — Haftalık fazlayı kovalara yaz.** E = A − R.
```
eğer E > 0:
    kalan = E
    sırayla [bn, nn, bd, nd] için:           # bayram gece → normal gece → bayram gündüz → normal gündüz
        öde[kova] = min(kalan, haftanın o kovadaki çalışılan saati)
        kalan -= öde[kova]
eğer E <= 0:
    açık += −E                               # o hafta hiçbir şey ödenmez
```
Bu "total üzerinden" yöntemdir: haftanın hangi saatinde 40 saatin dolduğuna bakılmaz, fazla önce geceye yazılır.
**Sıralı/kronolojik doldurma KULLANILMAZ** (hastane müdürüyle teyit, 23.11.2026).

**Adım 5 — Açığı aynı ayın ödenecek saatlerinden düş.**
```
kalan = açık (ayın tüm açık haftalarının toplamı)
sırayla [nn, bn, nd, bd] için:               # normal gece → bayram gece → normal gündüz → bayram gündüz
    x = min(kalan, ay_öde[kova]); ay_öde[kova] -= x; kalan -= x
```
Ay toplamı 0'ın altına inmez; düşülemeyen açık **sonraki aya devretmez**, silinir.

**Adım 6 — Kontrol.** FMS ≥ 0 iken `nd + bd + nn + bn = FMS` olmalı. FMS < 0 ise tüm kovalar 0.

**Adım 7 — Ücret.** Bölüm 5.

## 5. Formüller

```
k        = katsayı[yıl, (ay<=6 ? 1 : 2)]
taban    = (gösterge + 10) × k                         # saatlik nöbet ücreti, artırımsız
normal   = taban × (1 + 0,50 × riskli)                 # riskli birimde ×1,5
bayram   = normal × 1,25                               # çarpımsal
```

**Gece farkı alacağı (bu projenin hesapladığı tek kalem):**
```
brüt  = nn × normal × 0,10  +  bn × bayram × 0,10
damga = yuvarla(brüt × 0,00759 ; 2 ondalık)            # kişi-ay başına
net   = brüt − damga
```
Gündüz kovaları (nd, bd) gece farkı almaz; FMS'nin parçası oldukları için puantajda gösterilir.

Pratisyen, riskli birim, saatlik gece farkı:

| Dönem | normal gece farkı (TL/saat) | bayram gece farkı (TL/saat) |
|---|---|---|
| 2026 Ocak–Haziran | 30,18619425 | 37,7327428125 |
| 2026 Temmuz–Aralık | 34,267386 | 42,8342325 |

**Tam fazla mesai (nöbet) ücreti** — nöbet ücreti, haftalık hesapla bulunan ödenecek gündüz ve gece saatlerine, bayram
ya da normal oluşuna göre ödenir (kullanıcı onayı 27.11.2026): 4 kova (nd, bd, nn, bn) her biri kendi saatlik ücretiyle. Aynı parametrelerin sonucudur
ve nobeting'in yayımlanmış tablosuyla kuruşu kuruşuna tutar (ör. Tabip, riskli, 2026/2: gece 374,08, bayram gece 467,60 TL net):
```
saatlik[nd] = normal            saatlik[nn] = normal × 1,10
saatlik[bd] = bayram            saatlik[bn] = bayram × 1,10
ücret_brüt  = Σ kova_saati × saatlik[kova]  ;  net = brüt × (1 − 0,00759)
```
Not: Hastane nöbet ücretinin geri kalanını zaten ödemiştir; bu projede yalnızca %10 gece farkı alacak olarak hesaplandı.
Tam ücret de aynı kovalarla ödenir: kova saatleri Bölüm 4 (Adım 4–5) ile bulunur; aylık toplu hesap veya nöbet başına sabit
12 gece saati kullanılmaz.

## 6. Sınır durumları

1. **Ay sonunda başlayan nöbet:** 31'inde 08:00 başlayan 24'ün tamamı o aya ve o haftaya yazılır; ertesi günün
   saatleri o günün takvimine göre (tatil/bayram) sınıflanır. Örn. 31 Aralık nöbetinin 1 Ocak 00–08'i normal gecedir
   (1 Ocak genel tatil, bayram değil).
2. **Pazar nöbeti:** Pazar 08:00 başlayan nöbet o haftaya; Pazartesi 00–08 saatleri yeni haftaya geçmez.
3. **Arife:** zorunlu 5 saat (08–13). Arife günü başlayan 24: 08–13 normal gündüz (5), 13–20 bayram gündüz (7),
   20–24 bayram gece (4), ertesi gün (bayram 1. günü) 00–08 bayram gece (8).
4. **Bayramın son günü:** bayram 24:00'te biter; ertesi sabaha taşan saatler normaldir.
5. **Genel tatil (1 Ocak, 1 Mayıs, 15 Temmuz):** zorunlu 0, ama saatleri `nn`/`nd`'ye yazılır, %25 yok.
   Aynı gün dini bayram günüyse bayram sayılır (`bn`/`bd`, %25 var).
6. **Hafta sonuna denk gelen tatil:** zorunlu zaten 0; saatler bayram kurallarına göre sınıflanır.
7. **Tatil gününe yazılmış tatil adı** (ör. "CUMHURİYET BAYRAMI"): kod yok sayılır, hata değildir.
8. **Ay ortası başlama/ayrılma:** kullanıcı tarihi girer; tarih dışı günler `Bşl.`/`Ayr.` → zorunlu 0. Bu kişilerin
   eksik haftaları açık üretmez. Tarih girilmemiş ve FMS < 0 iken ayın başında/sonunda ≥ 5 boş gün varsa program
   tarih **önerir**, kendisi girmez.
9. **Tek nöbetli hafta:** 24 − 40 = −16 açık; aynı ayın diğer haftalarının ödenecek **normal gecesinden** önce düşülür.
10. **Hastane doğrulaması:** başlama/ayrılma tarihi girilmemiş her kişi-ayda ΣR = TCGS ve ΣA = TCS olmalı. Tutmuyorsa
    satır "KONTROL EDİLMEDİ" işaretlenir; olası nedenler: tanınmayan kod, boş iş günü (ay ortası giriş/çıkış), tatil
    takvimi, arife saati, vardiya süresi.
11. **Gece farkının hastane GECE FMS'iyle ilişkisi:** hastane GECE FMS sütunu genelde FMS'ye eşittir (gece/gündüz
    ayırmaz); bu kurallar ise FMS'yi 4 kovaya böler. Toplam FMS aynıdır.

## 7. Çözülmüş örnekler (gerçek veri, v1.2 çıktısıyla doğrulanmış)

Kişi adları anonimdir. Tümü: pratisyen (tabip), yoğun bakım (riskli).

### 7.1 Kişi A — Ekim 2026 (ham hücrelerden tam iz)

Hücreler: `8:24, 12:24, 14:24, 16:Yİz., 17:24, 21:24, 25:24, 28:"CUMHURİYET BAYRAMI Y.G.", 29:24, 31:24`
(1 Ekim Perşembe; 28 Ekim arife Çarşamba; 29 Ekim Perşembe bayram.)

| Hafta | R (zorunlu) | A | E | çalışılan nd/bd/nn/bn | ödenecek nd/bd/nn/bn |
|---|---|---|---|---|---|
| 1–4 | 16 (Per+Cum) | 0 | −16 | 0/0/0/0 | — (açık 16) |
| 5–11 | 40 | 24 | −16 | 12/0/12/0 | — (açık 16) |
| 12–18 | 32 (16'sı Yİz.) | 72 | +40 | 36/0/36/0 | 4/0/36/0 |
| 19–25 | 40 | 48 | +8 | 24/0/24/0 | 0/0/8/0 |
| 26–31 | 29 (8+8+5+0+8) | 48 | +19 | 12/12/20/4 | 0/0/15/4 |

26–31 haftası: 29 Ekim nöbeti → 08–20 bayram gündüz 12, 20–24 bayram gece 4, 30 Ekim 00–08 normal gece 8;
31 Ekim nöbeti → 12 gündüz + 12 gece. E = 19 → önce bn 4, sonra nn 15.

Ay: haftalık ödenecek nn = 36+8+15 = 59, bn = 4, nd = 4. Açık = 32 → nn'den düş: 59 − 32 = **27**.
**Sonuç: nd 4, bd 0, nn 27, bn 4** · ΣR 157 = TCGS 157 ✓ · ΣA 192 = TCS 192 ✓ · FMS 35 = 4+27+4 ✓
Ücret: 27 × 34,267386 + 4 × 42,8342325 = 925,22 + 171,34 = **1.096,56 TL brüt** · damga 8,32 · **net 1.088,24 TL**

### 7.2 Kişi A — Ocak 2026 (haftalık)

| Hafta | R | A | E | ödenecek nd/nn |
|---|---|---|---|---|
| 1–4 (1 Ocak genel tatil) | 8 | 24 | +16 | 4/12 |
| 5–11 | 40 | 48 | +8 | 0/8 |
| 12–18 | 40 | 48 | +8 | 0/8 |
| 19–25 | 40 | 72 | +32 | 0/32 |
| 26–31 | 40 | 24 | −16 | açık 16 |

nn 60 − 16 = **44**, nd 4 · FMS 48 ✓ · brüt 44 × 30,18619425 = **1.328,19 TL**

### 7.3 Kişi A — Mayıs 2026 (bayram + arife, haftalık kova girdisinden)

| Hafta | R | A | E | çalışılan nd/bd/nn/bn | ödenecek nd/bd/nn/bn |
|---|---|---|---|---|---|
| 1–3 (1 Mayıs genel tatil) | 0 | 0 | 0 | 0 | 0 |
| 4–10 | 40 | 72 | +32 | 36/0/36/0 | 0/0/32/0 |
| 11–17 | 40 | 0 | −40 | 0 | açık 40 |
| 18–24 (19 Mayıs) | 32 | 72 | +40 | 36/0/28/8 | 4/0/28/8 |
| 25–31 (26 arife, 27–30 bayram) | 13 | 48 | +35 | 0/24/8/16 | 0/11/8/16 |

Ay: nn 68 − 40 = **28**; sonuç nd 4, bd 11, nn 28, bn 24 · FMS 67 ✓ · brüt 28 × 30,18619425 + 24 × 37,7327428125 = **1.750,80 TL**

### 7.4 Test tablosu — Ekim 2026, 9 kişi (referans kod 9/9 tuttu)

| Kişi | Başlama/ayrılma | R | A | FMS | nd | bd | nn | bn | Brüt | Damga | Net |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | — | 157 | 192 | 35 | 4 | 0 | 27 | 4 | 1096,56 | 8,32 | 1088,24 |
| B | — | 165 | 192 | 27 | 0 | 0 | 15 | 12 | 1028,02 | 7,80 | 1020,22 |
| C | — | 141 | 168 | 27 | 8 | 0 | 15 | 4 | 685,35 | 5,20 | 680,15 |
| D | — | 165 | 192 | 27 | 0 | 0 | 15 | 12 | 1028,02 | 7,80 | 1020,22 |
| E | — | 136 | 192 | 56 | 4 | 0 | 52 | 0 | 1781,90 | 13,52 | 1768,38 |
| F | — | 165 | 168 | 3 | 0 | 0 | 3 | 0 | 102,80 | 0,78 | 102,02 |
| G | — | 165 | 192 | 27 | 0 | 0 | 27 | 0 | 925,22 | 7,02 | 918,20 |
| H | ayrılma 10.10.2026 | 56 | 96 | 40 | 8 | 0 | 32 | 0 | 1096,56 | 8,32 | 1088,24 |
| I | başlama 17.10.2026 | 69 | 96 | 27 | 12 | 0 | 15 | 0 | 514,01 | 3,90 | 510,11 |

Test girdileri:
- H: `2:24, 4:24, 6:24, 10:24, 28:"CUMHURİYET BAYRAMI Y.G.", 29:"CUMHURİYET BAYRAMI"`, ayrılma 2026-10-10
- I: `17:24, 21:24, 24:24, 27:24, 28:"CUMHURİYET BAYRAMI Y.G.", 29:"CUMHURİYET BAYRAMI"`, başlama 2026-10-17
  (tarih girilmezse: R 165, A 96, FMS −69, tüm kovalar 0 — hastane TCGS'si de 165)

## 8. Referans uygulama (Python, Bölüm 7.4 ile doğrulandı)

```python
from datetime import date, timedelta

RESMI_TATIL = {
    2026: ["01-01","03-20","03-21","03-22","04-23","05-01","05-19","05-27","05-28","05-29","05-30","07-15","08-30","10-29"],
    2027: ["01-01","03-09","03-10","03-11","04-23","05-01","05-16","05-17","05-18","05-19","07-15","08-30","10-29"],
}
ARIFE = {2026: ["03-19","05-26","10-28"], 2027: ["03-08","05-15","10-28"]}  # 13:00'ten itibaren tatil
GENEL_TATIL = {"01-01","05-01","07-15"}   # zorunludan düşer ama BAYRAM DEĞİL (dini bayrama denk gelmedikçe)
DINI_BAYRAM = {2026: ["03-20","03-21","03-22","05-27","05-28","05-29","05-30"],
               2027: ["03-09","03-10","03-11","05-16","05-17","05-18","05-19"]}
ARIFE_ZORUNLU = 5
MAZERET = {"Yİz.","Rpr.","Hastane Yatışı","G.G.Gid.","Bşl.","Ayr."}   # zorunludan düşer
CALISMA_GUNU_KODLARI = {"0"}            # zorunlu sayılır, çalışma 0 (idari izin de buraya)
VARDIYA = {"24":(8,24),"16":(16,16),"8":(8,8),"12":(8,12),"12G":(8,12),"N12":(20,12),"12N":(20,12)}
KATSAYI = {(2026,1):1.387871,(2026,2):1.575512}
GOSTERGE = {"uzman":150,"tabip":135,"dis_eczaci":120,"hemsire_ebe":90,"lise_saglik":75,"diger":55}
TS_PUAN, RISKLI, BAYRAM, GECE, DAMGA = 10, 0.50, 0.25, 0.10, 0.00759

def _md(d): return f"{d.month:02d}-{d.day:02d}"
def tam_tatil(d): return _md(d) in RESMI_TATIL.get(d.year, [])
def arife(d): return _md(d) in ARIFE.get(d.year, [])
def bayram_saati(d, saat):
    if tam_tatil(d) and (_md(d) not in GENEL_TATIL or _md(d) in DINI_BAYRAM.get(d.year, [])): return True
    if arife(d) and saat >= 13: return True
    return False

def zorunlu(d, kod):
    if d.weekday() >= 5 or tam_tatil(d): return 0
    if kod in MAZERET: return 0
    return ARIFE_ZORUNLU if arife(d) else 8

def kovalar(d, kod):
    """Nöbet başladığı güne yazılır; saatler gerçek takvim saatine göre sınıflanır."""
    k = {"nd":0,"bd":0,"nn":0,"bn":0}
    if kod not in VARDIYA: return k, 0
    bas, sure = VARDIYA[kod]
    for i in range(sure):
        t = bas + i
        gun, saat = d + timedelta(days=t // 24), t % 24
        gece = saat >= 20 or saat < 8
        b = bayram_saati(gun, saat)
        k[("b" if b else "n") + ("n" if gece else "d")] += 1
    return k, sure

def ay_hesapla(yil, ay, hucreler, bas=None, bit=None):
    """hucreler: {gün_no: kod}. bas/bit: date veya None."""
    son = (date(yil + (ay == 12), ay % 12 + 1, 1) - timedelta(days=1)).day
    haftalar = {}
    for g in range(1, son + 1):
        d = date(yil, ay, g)
        kod = hucreler.get(g)
        if bas and d < bas: kod = "Bşl."
        elif bit and d > bit: kod = "Ayr."
        if kod and kod not in VARDIYA and kod not in MAZERET and kod not in CALISMA_GUNU_KODLARI \
           and not (tam_tatil(d) or arife(d)):
            raise ValueError(f"{g}. gün: tanınmayan kod {kod!r} – sınıflandırılmadan hesap yapılmaz")
        w = d.isocalendar()[:2]                        # Pzt–Paz hafta
        h = haftalar.setdefault(w, {"R":0,"A":0,"nd":0,"bd":0,"nn":0,"bn":0})
        h["R"] += zorunlu(d, kod)
        kv, s = kovalar(d, kod)
        h["A"] += s
        for x in kv: h[x] += kv[x]
    od = {"nd":0,"bd":0,"nn":0,"bn":0}; acik = 0; detay = []
    for w in sorted(haftalar):
        h = haftalar[w]; E = h["A"] - h["R"]; p = {"nd":0,"bd":0,"nn":0,"bn":0}
        if E > 0:
            kalan = E
            for b in ("bn","nn","bd","nd"):           # doldurma sırası
                p[b] = min(kalan, h[b]); kalan -= p[b]
        else:
            acik += -E
        for b in od: od[b] += p[b]
        detay.append({**h, "fark":E, "ode":p})
    kalan = acik
    for b in ("nn","bn","nd","bd"):                   # açığı düşme sırası
        x = min(kalan, od[b]); od[b] -= x; kalan -= x
    R = sum(h["R"] for h in detay); A = sum(h["A"] for h in detay)
    return {"zorunlu":R, "calisilan":A, "FMS":A-R, "acik":acik, "dusulemeyen":kalan,
            "odenecek":od, "haftalar":detay}

def gece_farki_tl(yil, ay, od, unvan="tabip", riskli=True):
    k = KATSAYI[(yil, 1 if ay <= 6 else 2)]
    taban = (GOSTERGE[unvan] + TS_PUAN) * k
    normal = taban * (1 + (RISKLI if riskli else 0))
    bayram = normal * (1 + BAYRAM)                    # art arda (çarpımsal)
    brut = od["nn"] * normal * GECE + od["bn"] * bayram * GECE
    damga = round(brut * DAMGA, 2)
    return {"brut": round(brut, 2), "damga": damga, "net": round(brut - damga, 2)}
```

## 9. nobeting (Adil Nöbet) koduyla eşleme

nobeting'de her nöbet 08:00'de başlayan 24 saatlik nöbettir (Bölüm 3.1'deki diğer vardiya kodları yalnız
hastane listesinden hesap yapan Gece Farkı Hesaplayıcı içindir). İzin (3) mazeret kodudur, Müsait Değil (4)
çalışma günü sayılır. Elle eklenen tatil zorunlu mesaiden düşer ama %25 almaz (`resmi_tatil_gunleri` dışında kalır).

| Bu dosya | nobeting |
|---|---|
| Günlük zorunlu saat (Adım 1) | `nobet_gunler.personal_daily_work_hours` (toplamı `personal_work_hours`) |
| Arife 5 saat / 13:00 | `ARIFE_MESAI_SAATI`, `nobet_ucret.ARIFE_BAYRAM_SAATI` |
| Genel tatil (bayram değil, dini bayrama denk gelirse bayram) | `nobet_ucret.GENEL_TATILLER`, `_bayram_gunu_mu` |
| Adım 2–5 (saat saat kova, haftalık fazla, açık düşme) | `nobet_ucret.haftalik_ucret_saatleri` |
| Kovalar nd / bd / nn / bn | `UCRET_TURLERI`: `gunduz` / `tatil` / `gece` / `tatil_gece` |
| Tam nöbet ücreti (Bölüm 5) | `unvan_saat_ucreti` (yayımlanmış net tablo ya da formül) × kova saatleri |
| Ücretin göründüğü yerler | Analiz > Ücret Özeti ve Excel (`build_schedule_analysis`), takvim kişi kartı (`nobet_takvim`), paylaşım sayfası (`kisi_ucret_dokumu`) |
| Bölüm 7 örnekleri | `test_nobet.py`: `test_kurallar_ekim_2026_gercek_liste` ve komşu testler |

Bu kurallar değişirse önce bu dosya, sonra kod ve testler aynı PR'da güncellenir.

## 10. Açık sorular ve varsayımlar (kullanıcı onayı olmadan değiştirilmez)

1. **Yarım haftalar — ÇÖZÜLDÜ (kullanıcı onayı, 27.11.2026):** Ay sınırını kesen haftada zorunlu saate yalnızca o aya düşen
   hafta içi mesai günleri dahil edilir (iş günü 8, arife 5, tatil 0). ÇSGB yazısı (25.06.2026) bunu cevaplamıyordu; artık kesin kural.
2. **Genel tatillerin zorunlu saatten düşmesi — ÇÖZÜLDÜ (kullanıcı onayı, 27.11.2026):** 1 Ocak, 1 Mayıs, 15 Temmuz
   zorunlu saatten düşer (hastane TCGS'si de böyle).
3. **%25'in genel tatillerde verilmemesi — ÇÖZÜLDÜ (kullanıcı kararı, 27.11.2026):** böyle kalır; %25 verilmez
   (dini bayrama denk gelme istisnası için madde 5).
4. **12 saatlik vardiya kodları — AÇIK:** `12`, `12G` = 08:00/12 saat, `N12`, `12N` = 20:00/12 saat varsayılan. Kullanıcı
   hastanede gece 12 saatlik kodu `N12` olarak hatırlıyor; listede teyit edilene kadar açık.
5. **Genel tatil ile dini bayramın aynı güne denk gelmesi — ÇÖZÜLDÜ (kullanıcı kararı, 27.11.2026):** o gün bayram günü
   olduğu için bayram sayılır (bayram kovası, %25 artırım). Önceki varsayım (genel tatil öncelikli) kaldırıldı.
   2026–2027'de çakışma yok; Bölüm 7'deki örnekler ve sonuçlar değişmez. Gece Farkı Hesaplayıcı v1.3 (27.11.2026) bu kuralı
   uygular: Parametreler > Tatil takvimi'nde her yıl için "dini bayram günleri" listesi var.
6. **Tam nöbet ücretinin ödenmesi — ÇÖZÜLDÜ (kullanıcı onayı, 27.11.2026):** nöbet ücreti, haftalık hesapla bulunan
   gece ve gündüz saatlerine, bayram ya da normal oluşuna göre (4 kova) ödenir. Bölüm 5'teki tam ücret formülü geçerlidir.
7. **Kaynak içi tutarsızlık (çözüldü, bilgi için):** Eski kural notunda "tüm resmi tatiller bayram (1 Ocak … 15 Temmuz dahil)"
   yazıyordu; 23.11.2026 kararıyla bu üç gün bayram kovasından çıkarıldı. Bu dosya son kararı uygular.

## 11. Dayanak

- 657 sayılı DMK ek 33. madde (nöbet ücreti, göstergeler); 8. Dönem Toplu Sözleşme (+10 puan, bayram %25, gece %10).
- ÇSGB Çalışma Genel Müdürlüğü 25.06.2026 tarihli yazı (E-13487995-203.01-624383): önce haftalık 40 saat tamamlanır,
  40 saati aşan 20:00–08:00 saatleri %10 artırımlı.
- Hastane müdürü ile görüşme (23.11.2026): fazla "total üzerinden" dağıtılır, sıralı düşme yok.
- Mutemetlik notu (23.11.2026): 1 Ocak, 1 Mayıs, 15 Temmuz genel tatil; geceleri normal gece sütununa.
- Doğrulama: Gece Farkı Hesaplayıcı v1.2, 10 aylık yoğun bakım verisi, 79 kişi-ayda R/A hastane TCGS/TCS ile birebir.
