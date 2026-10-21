# M1 · Temizlik sınıflandırıcısı (ADR-002)

> **cad-ai'deki durum (2026-10-10).** Bu plan eski repoda (cad-llm-bridge, ADR-218) yazıldı ve buraya taşındı. Eski modüllere (`bridge/...`) yapılan atıflar tasarım geçmişidir; yeni projede karşılıkları:
> - Sahne: `cadai/cad/scene.py`. Hizalama ve etiketler: `cadai/cleanup/align.py`. Özellikler: `cadai/cleanup/features.py`. Eğitim ve değerlendirme: `cadai/cleanup/train.py`.
> - **§2.2 ölçüldü:** gerçek çiftlerde handle'lar korunmuyor (mühendis mimariyi patlatıp `m-mimari` katmanına taşıyor). Bu yüzden etiketler yalnızca geometrik hizalamayla çıkarılıyor (ADR-003).
> - Henüz yazılmayanlar: görünüş modeli (§4.1), grup tutarlılığı (§4.3), korumalar ve temiz DXF yazımı (§5), öğrenme döngüsü (§7).

**Hedef:** Mimarın ham DWG'sindeki her nesne için *kalır / silinir* kararını ve kategorisini veriden öğrenen bir model. Mevcut kural motorunun (`bridge/training/engine.py`) yerini almaz, onun **karar katmanı** olur. Korumalar ve mühendisin açık kuralları modelin üstünde kalır.

Sıra: M1 (bu belge) → M2 yerleştirme (`docs/plan-placement.md`) → M3 kontrol (`docs/plan-check.md`).

---

## 0. "%100 çalışır" ne demek

Nesne bazında %100 isabet hiçbir istatistiksel modelde garanti edilemez. Bu yüzden hedefi, ölçülebilir ve teslim edilebilir bir sözleşme olarak tanımlıyoruz:

| Ölçüt | Hedef | Nasıl sağlanır |
| --- | --- | --- |
| **Kritik hata** (duvar, sıva, kolon, pencere, kasa, merdiven, korkuluk silmek) | **0**, ayrılmış test setinde | Motor korumaları (`guard_ok`) + kritik katmanlarda yüksek silme eşiği |
| Otomatik kararların doğruluğu | ≥ %99,9 (çizgi uzunluğuyla ağırlıklı) | Üç bantlı çıktı: kesin tut / kesin sil / **emin değil** |
| "Emin değil" payı | proje başına < %1 nesne ve < 30 grup sorusu | Eşikler test setinde ayarlanır |
| Aşağı akış | Temiz çıktıdan okunan odalar, insan temizliğinden okunanlarla aynı (oda sayısı ve alan ±%2) | M3a kontrolü |

Yani **teslim edilen temizlik %100 doğrudur**: model emin olduğunu uygular, emin olmadığını gruplayıp ajana veya mühendise sorar, korumalar geri kalan riski sıfırlar. Model iyileştikçe "emin değil" payı küçülür.

## 1. Neden bu model önce

- Etiketler **bedava**: her ham + temiz (veya ham + son elektrik) çifti on binlerce nesne etiketi verir. 800 projede onlarca milyon etiket olur.
- Yerleştirme (M2) ve kontrol (M3) temiz altlığı girdi olarak alır. Temizlik hatası her sonraki adımı bozar.
- Bugünkü kural motoru ofis profiline bağımlı (`knowledge/cleanup/offices/*.yaml`). Yeni bir mimar ofisi geldiğinde profil yok. Model ise katman adından bağımsız geometri ve bağlam özelliklerinden genelleyebilir.

## 2. Veri

### 2.1 Kaynak çiftler
| Çift | Etiket kalitesi | Not |
| --- | --- | --- |
| ham → temiz mimari (çizimcinin temizi) | En iyi | `bridge/training/importer.py` handle farkı + patlatılmış blok eşlemesi zaten yapıyor |
| ham → son elektrik çizimi | İyi | Son çizimin elektrik katmanları (`corpus.dataset` zaten "ignore" diye işaretliyor) çıkarılınca kalan mimari = temiz altlık |
| Studio incelemeleri (`Review`) | En değerli, az | Mühendisin düzelttiği her nesne; eğitimde ağırlık ×10 |
| Mevcut motor kararları | Zayıf | Yalnızca kategori için yardımcı etiket; asla tek başına doğru kabul edilmez |

### 2.2 Eşleme sorunu
Handle farkı yalnızca temiz kopya ham dosyadan türetildiyse çalışır. Mimar başka dosyadan kopyaladıysa ya da son çizim yeni bir dosyaya çizildiyse handle'lar kopar ve koordinatlar kayabilir (`corpus/e2e.py` notu: "raw and final have different coordinates").

**Adımlar:**
1. Handle sürekliliği oranını ölç (temizdeki handle'ların ham dosyada aynı tip ve geometriyle bulunma payı). ≥ %80 ise handle eşlemesi.
2. Değilse **hizalama**: iki çizimin duvar çizgileri ve metinleri üzerinden benzerlik dönüşümü (öteleme + ölçek + 90° dönüş) bulunur; ardından nesneler tip + geometri ile 1 mm toleransla eşlenir (importer'daki patlatılmış blok mantığının genellemesi).
3. Eşleme oranı < %95 olan çift eğitime girmez, rapora düşer.

### 2.3 Etiket
- **Birincil (güçlü):** yaprak nesne başına `keep | delete`. Patlatılan blokların çocukları kendi etiketini alır (`resolve.leaf_decisions`).
- **İkincil (zayıf):** `categories.yaml` kategorisi. Motorun ataması + incelemeler. Kategori, eylemi moda göre belirler (`strict` / `machine`) ve mühendise açıklama sağlar.
- **Görünüş dışı** (`out_of_view`) ayrı ele alınır, bkz. §4.1. Eğitim nesneleri yalnızca kat planı görünüşlerinin içinden alınır, yoksa model "her şeyi sil" öğrenir.

### 2.4 Bölme (en önemli tasarım kararı)
`corpus/split.py` proje bazında bölüyor. Bu yetmez: aynı mimar ofisinin projeleri hem eğitimde hem testte olursa model katman adlarını ezberler ve test skoru yalan söyler.
- **Mimar ofisi kümesi:** dosyaların katman adı kümelerinin Jaccard benzerliğiyle kümelenir (ofis kimliği çoğu zaman bilinmiyor).
- Bölme **küme bazında**: test setinde eğitimde hiç görülmemiş mimarlar olur. Ayrıca proje bazındaki bölme de raporlanır (bilinen mimar senaryosu).
- Sabit, hash tabanlı, tekrar üretilebilir; yeni projeler eklenince eski atamalar değişmez.

### 2.5 Depolama
Özel veri, git dışı: `learnstore/ml/cleanup/`
- `pairs.jsonl`: çift listesi, eşleme yöntemi ve oranı
- `labels/<sha>.npz`: nesne kimliği, etiket, ağırlık
- `features/<sha>.npz`: özellik matrisi (sürümlü; özellik kodu değişince yeniden üretilir)
- `models/<sürüm>/`: model, eşikler, özellik listesi, test raporu

## 3. Özellikler

Her yaprak nesne için. Kural motorundaki bütün tanıyıcılar **özellik olur**: bugüne kadar yazılan bilgi çöpe gitmez, modele girdi olur.

| Grup | Özellikler |
| --- | --- |
| Nesne | dxftype, renk, çizgi tipi, kalınlık, kapalı mı, köşe sayısı, uzunluk (m), bbox (m), yay yarıçapı (m), dolu mu, eksene paralel mi |
| Katman adı | Türkçe katlanmış, `- _ . boşluk` ile bölünmüş parçalar; karakter 3-gram hash'i; bilinen kalıplar (`A-WALL`, `DUVAR`, `OLCU`, `DIM`, `MOBILYA`…) |
| Katman istatistiği | katmandaki nesne sayısı, ortalama uzunluk, metin payı, paralel çift payı, katmanın çizim alanındaki yayılımı. Bu, **adı bilinmeyen** bir katmanın rolünü tahmin etmeyi sağlar |
| Blok | blok adı parçaları, derinlik, kardeş sayısı, bloğun toplam boyutu, `symbols_detect` parmak izi eşleşmesi |
| Metin | uzunluk, yükseklik (m), regex bayrakları: ölçü sayısı, kot (`±0.00`), alan (`m²`), oda adı sözlüğü eşleşmesi (`dictionary/rooms.yaml`), pafta/antet kelimeleri |
| Bağlam | 0,5 m / 2 m yarıçapta komşu sayısı; duvar kalınlığında paralel eş çizgi var mı; bina zarfına uzaklık; zarfın içinde mi; grup boyutu (`groups.py`); tam kopya bayrağı |
| Tanıyıcılar | `predicates.py` çıktılarının hepsi ikili/sayısal özellik olarak: `tile_grid`, `grid_axis`, `dimension_cluster`, `door_swing`, `door_leaf`, `wall_fill`, `glass_strip` … |
| Proje | birim, toplam nesne sayısı, görünüş sayısı (normalizasyon için) |

Uygulama: `ml/cleanup/features.py`, girdi `bridge.training.scene.Scene`. Sahne önbelleği zaten var (`load_scene`). Büyük çizimde (840 gibi) hedef < 60 s.

## 4. Model

### 4.1 Aşama 0: görünüş tespiti
Çok paftalı ham çizimlerde ilk karar "hangi bölge kat planı". Yanlış görünüş seçimi tek seferde binlerce nesneyi yanlış siler.
- Bugün: `corpus/sheets.py` pafta başlığından buluyor. Başlık tanınmayınca başarısız (`try` projesi).
- Model: aday bölgeler (nesne yoğunluk kümeleri) × özellikler (duvar çifti yoğunluğu, kapı yayı sayısı, oda adı sayısı, ölçü yoğunluğu, başlık metni) → "kat planı / kesit / görünüş / vaziyet / detay / antet" sınıflandırıcısı.
- Etiket: son çizimde elektrik sembolü bulunan bölgeler = kat planı (bedava etiket), ayrıca mevcut `views`.

### 4.2 Aşama 1: nesne sınıflandırıcı
- **LightGBM** (gradyan artırmalı ağaç). Gerekçe: tablo verisi, milyonlarca satır CPU'da dakikalar içinde eğitilir, özellik önemleri okunur, olasılıklar kalibre edilebilir.
- İki baş: `p(delete)` (ikili, güçlü etiket) ve kategori (çok sınıflı, zayıf etiket).
- Kalibrasyon: ayrılmış doğrulama setinde isotonic.
- Sınıf ağırlığı: kritik kategorilerde yanlış silme 100× cezalı.

### 4.3 Aşama 2: grup tutarlılığı
Nesneler tek tek değil gruplar halinde anlamlı (kapı = kanat + yay + kol). Grup içindeki olasılıklar birleştirilir (`groups.py`): grup çoğunluğu kesinse aykırı tek nesne gruba çekilir; grup bölünmüşse bütün grup "emin değil" olur. Sonra `engine.compress` ile ağaca indirilir.

### 4.4 Sonraki seçenek: graf sinir ağı
LightGBM test setinde tavana takılırsa (özellikle yeni mimarlarda): nesneler düğüm, temas ve yakınlık kenar olan bir GNN. Ancak M1.3 ölçümü bunu gerektirirse. Önce basit ve açıklanabilir olan.

## 5. Motora bağlanma

Yeni öncelik sırası (`engine.py`):
1. `pre` kuralları: görünüş dışı, tam kopya (deterministik ve kesin, olduğu gibi kalır)
2. **Korumalar**: kritik kategoriye düşen hiçbir nesne `guard_ok` olmadan silinmez
3. **Mühendisin açık kuralları**: `rules/overrides`, ofis kuralları (insan kararı her zaman kazanır)
4. **Model**: `p(delete) ≥ t_del` → sil; `≤ t_keep` → tut; arası → **emin değil**
5. Eski katman/blok eşleme ve tanıyıcılar: modelin olmadığı veya özellik çıkarılamadığı durumlar için yedek

- Kayıt: `EntityDecision(source="model", rule_id="model:cleanup@<sürüm>", confidence=p)`. `Source` zaten `"model"` içeriyor; **format değişikliği gerekmez**.
- "Emin değil" grupları `studio/questions.py` → `auto_questions` kuyruğuna girer (mevcut düşük güven yolu).
- Ajan (`prompts/agents/cleanup.md`) yalnızca bu grupları görür; bütün çizimi değil. Maliyet ~$15'tan birkaç dolara iner.

## 6. Ölçüm

`python -m ml.cleanup eval --model <sürüm>`. Mevcut `bridge/training/score.py` bench kapısını kullanır, ayrıca:
- kritik yanlış silme sayısı (hedef 0)
- kaçırılan silme, mürekkep uzunluğuyla ağırlıklı ("ekranda kalan çöp")
- "emin değil" payı ve grup sayısı
- **yeni mimar** ve **bilinen mimar** setleri ayrı
- karşılaştırma: aynı projelerde bugünkü kural motoru (eski temizlikle uyum Drawing4 %93, Zemin Kat %94, Cihan %92, 840 %99)
- aşağı akış: temiz çıktıdan `bridge.extract` ile okunan oda sayısı ve alanları, insan temiziyle

**Kabul kapısı (her yeni model sürümü):** kritik = 0, diğer bütün ölçütler bir önceki sürümden kötü değil. Kapıdan geçmeyen sürüm yayınlanmaz.

## 7. Öğrenme döngüsü
1. Studio'da mühendis "emin değil" gruplarını ve fark ettiği hataları işaretler (`Review`).
2. Haftalık (veya 200 yeni incelemede bir) yeniden eğitim: `python -m ml.cleanup train`.
3. Kabul kapısı → yeni sürüm `learnstore/ml/cleanup/models/current`.
4. Hangi incelemenin hangi sürümde emildiği kaydedilir (`Review.status="absorbed"`, `rule_id` = model sürümü).
5. Aktif öğrenme: soru kuyruğu, modelin en çok kararsız kaldığı ve en çok nesneyi etkileyen gruplara göre sıralanır.

## 8. İş paketleri

| # | İş | Çıktı | Kabul |
| --- | --- | --- | --- |
| M1.0 | 800 projenin envanteri: `corpus.catalog` + `probe`; çift listesi | `pairs.jsonl`, rapor: kaç ham+temiz, kaç ham+son, handle sürekliliği | Rapor mühendisle gözden geçirildi |
| M1.1 | Hizalama ve etiket çıkarma | `labels/*.npz`, eşleme raporu | Eşleme ≥ %95 olan çift sayısı biliniyor; 20 çift elle kontrol |
| M1.2 | Mimar kümeleri ve bölme | `ml/common/split.py`, küme raporu | Test setinde ≥ 15 görülmemiş mimar |
| M1.3 | Özellikler + temel model + değerlendirme | `ml/cleanup/{features,train,eval}.py`, ilk rapor | Kural motorunu yeni mimar setinde geçiyor |
| M1.4 | Görünüş tespiti modeli | `ml/cleanup/views.py` | Başlıksız projelerde (`try`) doğru görünüş; test setinde ≥ %98 |
| M1.5 | Eşikler, grup tutarlılığı, motora bağlama | `engine.py` değişikliği + ADR, testler | Kritik = 0; emin değil < %1 |
| M1.6 | Studio ve ürün sitesi entegrasyonu | Hızlı betik modeli kullanır; ajan yalnızca emin olmadığını görür | Gerçek 5 yeni projede uçtan uca |
| M1.7 | Öğrenme döngüsü | `train` komutu, kabul kapısı, sürüm kaydı | Bir inceleme turunun modele girdiği gösterildi |

Yeni bağımlılıklar (M1.3'te `requirements.txt`'e): `lightgbm`, `scikit-learn` (kalibrasyon, metrikler).

## 9. Riskler
| Risk | Etki | Önlem |
| --- | --- | --- |
| Etiket gürültüsü: çizimciler tutarsız temizlemiş (sert / makine modu, eski temizlikler) | Model tutarsızlığı öğrenir | Çift başına mod tespiti; modu özellik yap; tutarsız çiftleri ölç ve dışla |
| Katman adı ezberi | Yeni mimarda çöküş | Mimar bazında bölme; katman adı özelliği kapalıyken de ölç |
| Hizalanamayan çiftler | Veri az kalır | Handle yolu + geometri yolu; kalan oran raporlanır |
| Büyük çizimlerde bellek | Toplu işte çökme | İş başına süreç, akış halinde özellik yazımı (ADR-205 deneyimi) |
| Model yanlış ama kendinden emin | Sessiz hata | Korumalar + M3a aşağı akış kontrolü + kalibrasyon raporu |
