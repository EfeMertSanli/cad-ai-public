# Yerleştirme planı (2026-10-12)

Proje aşamalı ilerliyor. Her aşamanın çıktısı tek başına işe yarar: mühendis o noktaya kadarını hazır alır, gerisini elle bitirir.

| Aşama | Ne | Nasıl | Durum |
|---|---|---|---|
| 1. Temizlik | Mimarın çiziminden altlık | Araçlar + Claude görünüş seçimi (ADR-021) | Bitti, referans: `D:\cad-ai-data\snapshots\temizlik-araclar-claude-2026-10-12` |
| 2. Temel semboller | Odalara cihaz: kaç tane, nereye | Gradient modeli (LightGBM), arşivden | **Şimdi** |
| 3. Bağlantı | Linye, anahtar–lamba bağı, buat ve ek noktaları, güzergah | Kısmen arşivden öğrenilir (bitmiş projelerde linyeler çizili) | 2'den sonra |
| 4. Pano ve hesap | Tali pano, devre gruplama, kolon şeması, yük hesabı | Yönetmelik ve hesap; değerler mühendisten (`TODO-ENGINEER`) | 3'ten sonra |
| 5. Kontrol | Yönetmelik ve tutarlılık denetimi | Mühendisin tarif ettiği kurallar | Her aşamaya eklenebilir |

Çıktı her aşamada doğrudan **DWG**: cihazlar ofisin kendi bloklarıyla ve katmanlarıyla temiz planın üstüne yazılır (hangi tipin hangi blokla çizildiği onaylı sembol sözlüğünde). Ara biçim (eski projedeki EDL) yok. LLM'ler çizim dosyası okumaz/yazmaz; DWG'yi kod okur ve yazar (ODA + ezdxf), LLM gerekirse resim ve tabloya bakıp karar verir.

## Aşama 2 · Temel semboller

**Kapsam:** tavan armatürü, priz, etanj priz, anahtar, TV, data (+ etanj/sensörlü/acil armatür). Pano, linye, buat, topraklama yok.

### Veri (`placement/dataset.py`, ADR-022)
Her arşiv eşleşmesi (ham mimari + mühendisin bitmiş elektrik çizimi) için:
1. **Temiz plan:** ham çizim bizim araç kutumuzla temizlenir; kat planları mühendisin çiziminden bilinir (hizalamanın bulduğu görünüşler), bu yüzden Claude gerekmez. Model gerçek kullanımda da yalnızca bizim temizliğimizi görecek.
2. **Odalar:** temiz plandan okunur (`rooms.py`), tip ham çizimdeki oda adından.
3. **Cihazlar:** mühendisin çiziminden, onaylı sembol tipleriyle (`devices.py`); her cihaz, çevresindeki mühendis çizgilerinin mimarın çizgilerine oturduğu görünüşle ham plana taşınır. Hiçbir plana oturmayanlar (kolon şeması, tablolar) atılır; birbirinin aynısı katların cihazları bir kez sayılır.
4. **Mobilya:** ham çizimden (`furniture.py`).
5. **Adaylar:** odada cihazın konabileceği noktalar (`candidates.py`; duvar boyunca 25 cm, tavanda 50 cm) ve özellikleri: kapıya, köşeye, merkeze, mobilyaya uzaklık, oda tipi ve boyutu.

Çıktılar: `DATA/placement/rooms.jsonl`, `candidates/*.npz`, `quality.json` (proje başına oda, tipli oda, odalara düşen cihaz).

### Model (`placement/model.py`)
- **Sayı:** oda × cihaz tipi başına Poisson LightGBM (oda tipi, alan, en/boy, kapı, mobilya).
- **Yer:** aday nokta başına "mühendis buraya koyar mı" LightGBM; en iyi adaylar en az 50 cm arayla seçilir.
- **Karşılaştırma yöntemi:** "benzer odayı kopyala": aynı tipte, alanı ve oranı en yakın eğitim odasının cihazları göreli konumla taşınır. Gradient modeli bunu geçmeli.

### Ölçüm
Test projelerinde (temizlikle aynı bölme), mühendisin gerçek cihazlarına göre:
- tip başına sayı hatası ve tam doğru sayı payı
- 50 cm içinde kesinlik ve duyarlılık; tahmin edilen sayıyla ve gerçek sayıyla (ikincisi yalnızca yer seçimini ölçer)
- üst üste resimler

Mühendisler arası doğal fark (aynı odaya farklı duvar) bu ölçüyü tavanlar; ileride mühendisin "kabul edilebilir" işaretleri ayrıca izlenecek.

### Durum (2026-10-13)
- **Çizim sayfası** çalışıyor (ADR-023): temizlik sonucu → odalar → cihazlar → DWG (şimdilik basit bloklar, ELK-* katmanları) ve resim; arşiv projesinde mühendisle karşılaştırma.
- **Haiku seçici denendi, bırakıldı:** 5 yerleşim seçeneği arasından seçmek modeli geçmedi (ADR-023 karar 4). Haiku adlandırmada: mobilya ve sembol kataloğuna öneriler (mühendis onayı bekliyor).
- **Mühendis kendisiyle:** aynı projede ikiz odalarda 1 m F1 tavan 0,87, priz 0,83, etanj 0,79, anahtar 0,80. Projeler arası fark büyük.

### Sonra
Ofis bloklarıyla DWG, sembol ve mobilya onayları, ikiz dairelere kopyalama, linyenin geri kalanı (ana hat ve priz zinciri).
