# Mimari

## Veri akışı (temizlik modeli, M1)
```
Ofis arşivi (salt okunur)
  └─ corpus/inventory.py ─► inventory.json: proje → dosyalar (ham / son / taslak) → eğitim çiftleri
        └─ corpus/process.py convert_all (paralel süreçler)
              cad/oda.py      DWG → DXF            D:\cad-ai-data\dxf\<sha16>.dxf
              cad/scene.py    DXF → sahne (.npz)   D:\cad-ai-data\scenes\<sha16>.npz (+ .json özet/hata)
        └─ corpus/process.py label_all
              cleanup/align.py  ham sahne ↔ son sahne → nesne başına etiket
                                D:\cad-ai-data\labels\<proje>__<sha16>.npz/.json/.png
        └─ cleanup/train.py
              cleanup/features.py  ham sahne → özellik matrisi (önbellek: features\)
              LightGBM → D:\cad-ai-data\models\cleanup-<zaman>\ model.txt, report.json, test PNG'leri
```

## Sahne (`cad/scene.py`)
Bir çizim, **yaprak nesnelere** patlatılmış hali: her blok içeriği özyinelemeli açılır, her yaprak geldiği üst nesneyi (`top`) ve yolunu (`<handle>/<i>/<j>`) bilir. Geometri metre cinsinden düz bir doğru parçası listesidir (eğriler 5 mm hatayla, yazılar dış kutusuyla). Sütun dizileri halinde tek `.npz` dosyası; 200 bin nesne bir saniyede yüklenir. Birim, `$INSUNITS`'e değil kanıta göre (kapı yayları, yazı yükseklikleri) bulunur (`cad/units.py`).

## Etiketler (`cleanup/align.py`, ADR-003)
Mühendis son çizimde mimariyi patlatıp başka katmana ve koordinata taşıyor; handle'lar korunmuyor. Etiket geometriden çıkar:
1. Aynı yön ve uzunluktaki belirgin doğrular, ham → son ötelemesi için oy verir (her 90° dönüş ve birkaç ölçek oranı için).
2. En güçlü ötelemeler doğrulanır. Son çizimin her noktası tek bir görünüşe ait olabilir, çünkü katlar birbirine benzer. En güçlü görünüşün %10'undan zayıf olanlar elenir.
3. Ham nesnenin noktalarının son çizimde bulunma payı: ≥ %80 **kaldı**, ≤ %20 **silindi**, arası **belirsiz**. Kabul edilen görünüşlerin dışındakiler **görünüş dışı**. Başka katmandan bir nesnenin tam üstünde yatanlar (alan çizgisi duvar yüzünde) **belirsiz** sayılır.
Eğitime yalnızca kaldı/silindi girer.

## Model (`cleanup/train.py`, ADR-004)
LightGBM ikili sınıflandırıcı, `p(sil)`. Proje bazında sabit bölme: %15 test, %10 doğrulama. Üç bant: doğrulama projelerinde her otomatik bandın kesinliği ≥ %99,5 olacak şekilde eşik seçilir; arada kalan nesneler **soru** olur. Rapor: otomatik karar payı ve doğruluğu, yanlış silinen uzunluk (özellikle duvar benzeri), test projeleri için hata görüntüleri.

## Web (`web/app.py`)
FastAPI + statik sayfalar (derleme adımı yok):
- **Projeler:** hat durumu, iş düğmeleri, modeller, proje listesi
- **Proje:** dosyalar, sahne bilgisi, etiket görüntüsü ve hizalama günlüğü
- **Temizle:** yeni bir DWG yükle, katmanlarla (olmazsa nesne nesne) temizle, katman tiplerini düzelt, temiz DWG/DXF indir
- **Çizim:** bir temizlik sonucunu seç, cihazları yerleştir (ADR-023); resim, DWG (ELK-* katmanları) ve arşiv projesinde mühendisle karşılaştırma
- **Semboller / Mobilya:** verinin adlandıramadığı şeylerin bir kerelik sözlüğü (LLM önerir, mühendis onaylar)
- **Debug:** işler (canlı ilerleme ve günlük, iptal), uygulama günlüğü, hatalı adımlar (traceback), model raporları, sistem (bellek, disk, ODA, önbellek sayıları)

İşler `jobs.py` ile sırayla çalışır. Ağır adımlar içeride süreç havuzu kullanır; süreç sayısı o anki boş belleğe göre seçilir (işçi başına ~2,5 GB).

## Yeni bir çizimin temizliği: araçlar + Claude (`cleanup/agent.py`, ADR-021, varsayılan)
DWG → DXF → sahne → istatistik görünüş ve temizlik modeli → `regions.py` (pafta bölgeleri ve başlıkları) →
**Claude 1:** hangi bölgeler kat planı (Claude'un arşivden yazdığı görünüş kural kitabıyla) → `tools.py` sabit araçlar
(kapatılan katmanlar, yazı ve ölçü, mobilya imzaları, kapı yayları, tarama, kesikli çizgi, belirsiz katmanda model,
duvardan taşan çentikler; her biri ne sildiğini ayrı tutar) → **Claude 2:** kalanların ve araç renklerinin resmine bakıp
araç kapatır / katman tipini düzeltir / bölge ekler-çıkarır → araçlar bir kez daha → temiz DXF/DWG. Claude çağrıları
`claude -p` ile (abonelik). Çıktıda ayrıca `views.png` (bölge kararları), `tools.png` (araç renkleri), `kept.png`,
`agent.json` (Claude'un cevapları, notları, maliyeti). "Yalnız araçlar" yöntemi Claude'suz aynı akıştır.

## Önceki yöntem: katmanlar ya da nesne nesne (`cleanup/apply.py`, ADR-019, ADR-020)
Mühendisin yöntemi: mimarın çizimine dokunulmaz, katmanlar kapatılır, çizgi kalınlığı kapatılır.

DWG → DXF → sahne → temizlik modeli (hangi nesne planda; belirsiz katmanlarda en iyi tahmin) → yöntem seçimi (`extract.py`):
- **Katman yöntemi** (katmanlar çizimi ayırıyorsa): her katmana bir eleman tipi (`layers.py`: önce mühendisin sözlüğü, sonra isim kuralları). Duvar, kolon, pencere, kapı, merdiven, korkuluk ve şaft kalır; tarama, mobilya, ölçü, aks ve çatı kapanır. Kalan katmanlardan geometriyle ayıklananlar (`elements.py`): patlatılmış tarama çizgileri, kapı yayları ve kanatları, duvardan taşan çentikler, kesikli çizgiler, oda adı dışındaki yazılar.
- **Nesne yöntemi** (her şey tek katmanda ya da katman adları bir şey söylemiyorsa): modelin kararları ve `finish.py` kuralları.

Temiz DXF, mimarın nesnelerinden silinerek yazılır; hiçbir şey yeniden çizilmez. Karışık bloklar patlatılır, duvar ve kolon taramaları dış hatta çevrilir. Çizim beyaz, çizgi kalınlıkları kapalı, birim cm. Çıktılar `D:\cad-ai-data\outputs\<ad>-<zaman>\` altında: `clean.dxf`, `clean.dwg`, `clean.png`, `preview.png`, `layers.json` ve `layers/*.png` (katman tablosu), `questions.json`, `checks.json`, `summary.json`.

Sitede **Temizle** sayfası. Her sonucun katman tablosunda mühendis tipleri düzeltip yeniden temizler; düzeltmeler `DATA/layers/types.json` sözlüğüne yazılır ve sonraki projelerde de kullanılır.

## Kontrol (`check/cleanup_checks.py`, M3a)
Temizlenmiş plandan odalar okunur ve ham çizimdeki mimar alan damgalarıyla karşılaştırılır. Uyarılar: alan farkı > %10, planın yanındaki bir damganın hiçbir odaya düşmemesi (oda açık kalmış). Plandan uzaktaki damgalar (mahal listesi tabloları) yalnızca sayılır.

## Yerleştirme verisi (`placement/`, M2)
- `symbols.py`: son çizimlerdeki elektrik sembolleri. Bloklar adıyla, patlatılmış semboller şekil imzasıyla, yalnız aynı katmandaki parçalar birleşir (ADR-009, ADR-024). Katalog + küçük resim + yanındaki yazılar. Tipler **Semboller** sayfasında bir kez onaylanır.
- `shapes.py`: patlatılmış sembolleri şekilden tanır (ADR-024): onaylı imzaların şekil resmine yakın gruplar ve sapına linye değmiş semboller onaylı tipi alır. Yalnız mühendisin son çizimlerini okurken (`devices.locate(by_shape=True)`); şekiller `symbol_shapes` işiyle toplanır.
- `furniture_shapes.py`: ham planlardaki mobilya (bloklar ve dağınık çizgi grupları, `furniture.loose_instances`) şekil sınıflarına toplanır; mühendis **Mobilya şekilleri** sekmesinde ilk 300 sınıfı adlandırır, ad bütün üyelere ve benzer yeni şekillere geçer (ADR-025).
- `furniture.py`: ham çizimlerdeki mobilya, geometri imzasıyla (adlar anlamsız). **Mobilya** sayfasında onaylanır.
- `devices.py`: son çizimin cihazları, hizalama dönüşümünün tersiyle ham çizim koordinatına taşınır; çevresindeki çizgilerin oturduğu kat seçilir (ADR-007, ADR-022).
- `rooms.py`: temiz plandan odalar, 5 cm raster; oda tipi ham çizimdeki addan (ADR-008).
- `candidates.py`: odada aday noktalar (duvarda 25 cm, tavanda 50 cm) ve özellikleri: kapı, kapının kilit ve menteşe tarafı, köşe, merkez, pencere, mobilya (tipli ve boyut sınıflı); odasına göre yeri ve oda içindeki sırası (ADR-026); kapının arkasındaki oda (ADR-027).
- `dataset.py`: oda başına bir satır (tip, alan, cihazlar, mobilyalar) + aday dosyaları; odalar bizim temizliğimizden (ADR-022).
- `model.py`: oda × tip başına sayı (Poisson LightGBM) ve oda içinde aday sıralayıcı (lambdarank LightGBM, ADR-026); cihazsız katlar eğitim dışı (ADR-023).
- `steer.py`: mühendisin talimatı → Haiku → kural (sayı sınırı, etiketli noktalar önce); kod modelin üstüne uygular (ADR-027).
- `place.py`: Çizim sayfasının işi: temizlik çıktısından kat planları, odalar, cihazlar, lamba–anahtar linyeleri; DWG ve resim (ADR-023).

Yerleştirme kontrolleri (M3b) henüz yok.
