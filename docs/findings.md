# Veriden bulgular

Gerçek arşivden ölçülen, tasarımı etkileyen gerçekler. Her madde tarihli.

## Arşiv (2026-10-10)
- `Desktop\proje\Proje`: 237 proje klasörü, 528 DWG. 259 son elektrik çizimi (`<Ad>(<ada-parsel>).dwg`), 252 ham mimari, 17 mühendis taslağı (`Drawing1…`).
- 234 projede ham + son çifti var. 2 projede son çizim yok, 1 projede son çizim adı kalıba uymuyor (`<ad>-<ada>-<parsel>-<ilçe>.dwg`).
- 20 projede 2, 2 projede 3 son çizim var (blok blok ya da revizyon).

## Ham ↔ son ilişkisi (2026-10-10, ilk arşiv projesi)
- Mühendis mimariyi **patlatıp** `m-mimari` ve `0` katmanlarına taşıyor; handle'lar korunmuyor (ortak 167 handle'ın hiçbiri aynı nesne değil).
- Koordinatlar farklı, katlar ayrı ayrı taşınmış (aynı çizimde 3 farklı öteleme).
- `$INSUNITS` güvenilmez: ham cm (5), son "mm" (4), ama ikisi de gerçekte cm (kapı yayı ve yazı yüksekliği kanıtı).
- Son çizimde yalnızca kat planları var. Vaziyet, çatı, kesit ve görünüşler alınmamış.

## Blok adları (2026-10-10, ilk 23 ham + 27 son çizim)
- **Ham çizimlerde blok adları anlamsız:** `GVXVVS`, `WR TW THW`, `ASRETGFGDFGDFG`, `A$C2AC03DC6`… Az sayıda okunur ad var (`BUZDOLABI 23`, `ELEKTRIKLI FIRIN 23`). Mobilya ve vitrifiye **adından tanınamaz**; geometri parmak izi ve bir kerelik etiketleme (LLM/görsel) gerekir. Bu M2'nin (yerleştirme) önkoşulu.
- **Son çizimlerde elektrik sembolleri ofisin sabit kütüphanesinden:** aynı anonim bloklar (`A$C2D997B2E` 27/27, `A$C06821EBC` 26/27, `A$C51797531` 26/27 …) ve birkaç okunur ad (`TOPRAKLAMA`, `CANAKANTEN`, `C40SENSOR`, `SENSORLU`, `U 40W ACIL KITLI`). ~15–20 bloğun tipini bir kez eşlemek, cihaz etiketlerinin çoğunu verir.
- Son çizimlerin katmanları da sabit: `LINYE`, `M-LINYE`, `YAZI`, `MIMARI`/`M-MIMARI`, `SEMBOL`, `ZAYIFAKIM`, `TOPRAKLAMA`, `M-ACIL KACIS`, `ARMATUR`…

## Görünüş özellikleri (2026-10-10, ilk arşiv projesi)
Plan içi ve dışı nesnelerin 6 m komşulukları çok farklı: duvar katmanı yoğunluğu 5,6 / 0,4; mobilya 3,9 / 0,0; kesit katmanı 0,2 / 1,8. "Mahal listesi / alan hesabı" başlığına uzaklık plan içinde 28 m, dışında 145 m.

## Sembol kataloğu, bütün son çizimler (2026-10-10)
- 259 son çizimde 116 farklı blok ve 6190 farklı patlatılmış sembol imzası.
- Bloklar: kullanımların %80'i ilk 7 blokta (toplam 73 883 kullanım); 19 blok 30'dan fazla projede.
- Patlatılmış semboller: toplam 103 222 kullanım; %80'i için 247 imza gerekiyor; 197 imza 30'dan fazla projede.
- En sık patlatılmış imzalar ~245 projenin hepsinde birer kez geçiyor: bunlar ofisin **standart lejant tablosu**. Yanlarında açıklamaları yazıyor ("ALARM BUTONU", "1 FAZLI PAKO ŞALTER", "AMPERMETRE", "KESİŞEN İKİ İLETKEN"…). Lejant, imzaların adını otomatik verebilir; plan içindeki kullanımlar cihaz verisidir (lejant görünüş dışında kaldığı için `devices.locate` onu zaten atar).
- Sonuç: 247 satırı mühendise tek tek onaylatmak gerçekçi değil. Lejant yazısı + LLM ile otomatik öneri, mühendise yalnızca belirsizler.
