# M3 · Kontrol modeli (ADR-002)

> **cad-ai'deki durum (2026-10-10).** Bu plan eski repodan taşındı; `core.models.Issue` gibi atıflar eski repodadır. Yeni projede kontrol modeli henüz yazılmadı.

**Hedef:** Temizlik ve yerleştirme çıktısındaki hataları, mühendis görmeden önce bulmak. Bir proje yalnızca kontrolden kritik bulgu çıkmazsa "hazır" sayılır. Bulunan her hata yeri, nedeni ve düzeltme ipucuyla ajana veya mühendise gider.

İki kontrolcü var: **M3a temizlik kontrolü** ve **M3b yerleştirme kontrolü**. Her biri iki katmanlı: deterministik kurallar + öğrenilmiş eleştirmen.

---

## 0. Bağımsızlık ilkesi

Kontrol modeli, kontrol ettiği modelin kopyası olursa işe yaramaz: aynı hatayı aynı güvenle onaylar. Bu yüzden:
1. **Farklı soru sorar.** Yerleştirme modeli "en iyi konum hangisi" diye sorar; kontrolcü "mühendis bu cihazı burada bırakır mıydı, yoksa taşır/siler miydi" diye sorar.
2. **Negatif örneklerle eğitilir.** Yerleştirme modeli yalnızca doğru örnekleri görür. Kontrolcü bozulmuş ve gerçekten düzeltilmiş örnekleri görür.
3. **Kendi özellikleri vardır.** Oda bütününe ve proje bütününe bakar (tutarlılık, simetri, daireler arası benzerlik); yerleştiricinin yerel aday özelliklerinden farklı.
4. **Ayrı ölçülür.** Ölçüm doğrudan yerleştiricinin çıktısı üzerinde yapılır, mühendisin son çizimi üzerinde değil.

## 1. Çıktı formatı (format değişikliği yok)

Bulgular mevcut `core.models.Issue` kaydıyla yazılır: `code`, `severity` (`error | warning | info`), `message`, `target` (oda/cihaz/duvar adı), `fix_hint`, `data` (skor, kanıt). Kod adları:
- `check.clean.*`: temizlik bulguları (ör. `check.clean.room_not_closed`, `check.clean.leftover`)
- `check.place.*`: yerleştirme bulguları (ör. `check.place.missing:socket`, `check.place.unlikely_position`)
- `rule.tr.*`: yönetmelik kuralları (mevcut kod alanı)

Bulgular `.edl.result.json` `issues` listesine ve Studio soru paneline gider. Ajan `fix_hint`'i okuyup düzeltir; düzeltemezse mühendise soru olur.

## 2. M3a · Temizlik kontrolü

### Deterministik
| Kontrol | Kod | Önem |
| --- | --- | --- |
| Kritik kategoride silinen nesne (korumaya rağmen) | `check.clean.critical_deleted` | error |
| Kapanmayan oda / açık duvar ucu | `check.clean.room_not_closed` | error |
| **Alan karşılaştırması:** ham çizimdeki alan damgaları (`area_stamp`, temizlikte silinir) her odanın mimar alanını verir. Temiz çıktıdan okunan oda alanı ±%3 dışındaysa | `check.clean.area_mismatch` | warning |
| Oda adı metni silinmiş ama oda var | `check.clean.room_name_lost` | warning |
| Kat planı görünüşü sayısı ham çizimdekinden az | `check.clean.view_missing` | error |

Alan damgası kontrolü bedava bir doğruluk kaynağıdır: mimar her odanın alanını zaten yazmış. Temizlikten önce okunur, kayıt edilir, temizlikten sonra karşılaştırılır.

### Öğrenilmiş: artık çöp dedektörü
- `bridge/training/suspects.py` mantığının model hali: temizlikten sonra **kalan** grupları skorlar, "bu çizimci temizinde olmazdı" olasılığı.
- Etiket: insan temizlerinde kalan gruplar (negatif) / M1'in kaldığı ama insanın sildiği gruplar (pozitif). Ayrıca sentetik: temiz bir çizime ham çizimden rastgele silinmiş gruplar geri eklenir.
- M1'den farkı: grup bütününe ve temiz çizimin genel görünümüne bakar (kalan katman sayısı, kalan metin yoğunluğu, kalan küçük kopuk kümeler).

## 3. M3b · Yerleştirme kontrolü

### Deterministik (yönetmelik ve sağduyu)
| Kontrol | Kod | Önem |
| --- | --- | --- |
| Aydınlatmanın anahtarı yok / anahtar hiçbir şeyi yakmıyor | `check.place.unswitched_light` | error |
| Islak hacim bölgesinde uygunsuz cihaz | `rule.tr.wet_zone` | error (değerler TODO-ENGINEER) |
| Cihaz oda dışında, duvar içinde, kapı açılışında, pencere boşluğunda | `check.place.bad_position` | error |
| Üst üste binen / çok yakın cihazlar | `geo.min_spacing` (mevcut) | warning |
| Oda tipine göre zorunlu cihaz yok (mühendisin onayladığı "değişmez" kurallar, `corpus/mine.py` `invariant` sınıfı) | `check.place.missing:<tip>` | warning |
| Devre: cihaz sayısı / yük sınırı (M2.5 sonrası) | `rule.tr.circuit.*` | error |

Mevcut `bridge/checks/geometry.py` bunun bir kısmını yapıyor; genişletilir.

### Öğrenilmiş: eleştirmen
- **Cihaz düzeyi:** her cihaz için `p(mühendis tutar)`, ayrıca `p(taşır)`, `p(siler)`, `p(tipini değiştirir)`.
- **Oda düzeyi:** `p(oda eksik)` her cihaz tipi için ("bu yatak odasında priz eksik").
- **Proje düzeyi tutarlılık:** aynı tipteki daireler ve odalar benzer donatılmış mı (8 daireli bir katta 7'sinde 3 priz, birinde 1 priz → bulgu).
- **Model:** LightGBM sınıflandırıcıları. Girdi: oda bağlamı (`core.learning.RoomContext`), cihazın konumu ve tipi, odanın diğer cihazları, mobilya ipuçları, benzer odaların ortalaması.

### Etiket kaynakları (sırasıyla değerli)
1. **Gerçek düzeltmeler:** Studio'da taslak → mühendisin onayladığı hali. `bridge/compare.py` her farkı `Correction` yapıyor (taşındı / silindi / eklendi / tip değişti). İnceleme sekmesi işaretleri (ADR-213: doğru / yanlış yer / olmamalı / yanlış tip / eksik). Bu en iyi kaynak ama başta az.
2. **Sentetik bozma:** mühendisin son çizimlerinden (doğru kabul) cihazlar rastgele bozulur: başka duvara taşı, sil, kopyala, tipini değiştir, yanlış odaya koy. Bozulan cihaz pozitif, dokunulmayanlar negatif. Sınırsız veri, ama gerçek hatalara benzemeyebilir.
3. **Yerleştiricinin gerçek hataları:** M2 çıktısı ile ayrılmış son çizim karşılaştırıldığında eşleşmeyen cihazlar. Gerçek hata dağılımı budur; sentetik verinin dağılımını buna göre ayarla.

## 4. Ölçüm

| Ölçüt | Hedef |
| --- | --- |
| Sentetik bozmalarda duyarlılık (bulunan / eklenen hata) | ≥ %90 |
| Yerleştiricinin gerçek hatalarında duyarlılık | ≥ %70 |
| Bulgu kesinliği (mühendisin "evet hata" dediği pay) | ≥ %80 |
| Kritik yanlış alarm (doğru projeyi "hazır değil" diye durdurma) | proje başına ≤ 1 |
| Temizlikte alan uyuşmazlığı yakalama | alan damgalı projelerin %100'ünde çalışıyor |

Kesinlik önemli: çok yanlış alarm veren kontrolcüyü mühendis görmezden gelmeye başlar.

## 5. Akıştaki yeri

```
ham ─► M1 temizlik ─► M3a kontrol ──(error)──► ajan düzeltir / mühendise soru
                          │ ok
                          ▼
                    M2 yerleştirme ─► M3b kontrol ──(error)──► ajan düzeltir / mühendise soru
                                          │ ok
                                          ▼
                                     "hazır" (ürün sitesi teslim eder, ADR-217)
```

- Ürün sitesinde müşteriye yalnızca kontrolden geçmiş çıktı gider. Geçmeyen iş "mühendis kontrol ediyor" yoluna düşer (ADR-217 "Next" maddesi).
- Ajan istemine (`prompts/agents/*.md`) kural: teslimden önce kontrolü çalıştır, `error` kalmayana kadar düzelt, düzeltemediğini soru yap.

## 6. İş paketleri

| # | İş | Çıktı | Kabul |
| --- | --- | --- | --- |
| M3.0 | Deterministik temizlik kontrolleri, alan damgası kaydı | `ml/check/clean_rules.py` | 4 eğitim projesinde çalışıyor, yanlış alarm yok |
| M3.1 | Deterministik yerleştirme kontrolleri (`bridge/checks` genişletmesi) | `rule.tr.*`, `check.place.*` kodları | Testler; değerler TODO-ENGINEER işaretli |
| M3.2 | Sentetik bozma üreticisi | `ml/check/perturb.py` | Bozma türleri dengeli, tekrar üretilebilir |
| M3.3 | Eleştirmen modelleri (cihaz, oda, proje) | `ml/check/{train,critic}.py` | §4 sentetik hedefleri |
| M3.4 | Artık çöp dedektörü | `ml/check/leftover.py` | M1 hatalarının ≥ %70'i yakalanıyor |
| M3.5 | Akışa bağlama: Studio, ürün sitesi, ajan istemleri | `issues` → soru paneli | Gerçek 5 projede uçtan uca |
| M3.6 | Gerçek düzeltmelerle yeniden eğitim | Öğrenme döngüsü | Kesinlik ≥ %80 |

M3.0 ve M3.1 makine öğrenmesi gerektirmez; **M1 ile paralel başlayabilir**. M3.2–M3.4, M1 ve M2'nin ilk modelleri çıktıktan sonra.

## 7. Riskler
| Risk | Önlem |
| --- | --- |
| Kontrolcü yerleştiriciyle aynı hatayı yapıyor | §0 bağımsızlık ilkesi; ölçüm doğrudan yerleştirici çıktısında |
| Sentetik hatalar gerçek hatalara benzemiyor | Bozma dağılımı yerleştiricinin gerçek hata dağılımına göre ayarlanır |
| Çok yanlış alarm | Kesinlik hedefi; eşikler mühendis geri bildirimiyle |
| Yönetmelik değerleri bilinmiyor | `TODO-ENGINEER`; değer gelene kadar ilgili kontrol `warning` |
