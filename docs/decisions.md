# Kararlar (ADR)

Biçim: bağlam → karar → sonuçlar → ne zaman yeniden bakılır. Kodda numarayla anılır.

## ADR-001 · Yeni proje, veri D: diskinde
- **Bağlam:** eski repo (cad-llm-bridge) 1188 testlik, iki siteli, kural ve ajan ağırlıklı bir sisteme büyüdü. Kullanıcı 2026-10-10'da daha temiz, okunabilir ve debug paneli olan yeni bir projeye geçmeye karar verdi. C: diskinde ~2 GB boş yer var.
- **Karar:** kod `Desktop\cad-ai`'de, yeni bir paket (`cadai`). Eski koddan yalnızca kanıtlanmış fikirler sadeleştirilerek alındı: ODA dönüştürme, sahne patlatma, birim kanıtı. Sanal ortam ve bütün büyük veri `D:\cad-ai-data`'da. Ofis arşivi salt okunur.
- **Sonuçlar:** eski repodaki EDL, yazıcı, oda okuma ve yerleştirici henüz yok; yerleştirme (M2) başlarken hangilerinin taşınacağı ayrıca kararlaştırılır.

## ADR-002 · Üç öğrenilmiş model; kurallar koruma ve aday olarak kalır
Eski repoda ADR-218 olarak alındı, aynen geçerli. Sıra: M1 temizlik → M2 yerleştirme → M3 kontrol. Planlar: `docs/plan-*.md`. Kullanıcı kararları (2026-10-10): eğitim eldeki 200–300 projenin tamamıyla tek seferde yapılır, küçük deneme yok; yerleştirme mobilyaya göre net geometriyle yapılır, ardından kurallar kontrol eder; mühendisin katkısı en aza indirilir.

## ADR-003 · Temizlik etiketleri geometrik hizalamayla
- **Bağlam:** ilk çiftte (bir arşiv projesi) ham ve son çizimde ortak handle 167 tane ve hiçbiri aynı tip+katmanda değil. Mühendis mimariyi patlatıp `m-mimari`/`0` katmanlarına taşımış, koordinatlar ve `$INSUNITS` farklı (ham cm, son "mm" ama gerçekte cm).
- **Karar:** iki sahne metreye çevrilir. Ham→son dönüşüm, belirgin doğruların ötelemesine oy verilerek bulunur (90° dönüşler ve 0.01–100 ölçek oranları). Görünüşler en güçlüden başlayarak kabul edilir: son çizimin her noktası yalnızca bir görünüşe ait olabilir, zayıf oylar (< en güçlünün %10'u) ve zayıf görünüşler (< en güçlünün %10'u kadar eşleşen çizgi) elenir. Nesne etiketi, örnek noktalarının 1,5 cm içinde son çizimde bulunma payıyla verilir (≥ 0.8 kaldı, ≤ 0.2 silindi). Görünüş bölgesi, eşleşmelerin bağlı en büyük kümelerinden hesaplanır.
- **Sonuçlar:** ilk çiftte 3 kat planı doğru bulundu; duvar, kolon, kapı, pencere, merdiven kaldı; aks, ölçü, oda yazısı, vitrifiye silindi; vaziyet, çatı, kesit, görünüş dışarıda kaldı (görsel kontrol). Başka bir nesnenin üstünde yatan nesnelerin hangisinin kaldığı bilinemez, belirsiz sayılır.
- **Ne zaman yeniden bakılır:** mühendisin planı döndürmeden ama ölçekleyerek taşıdığı (ör. 1/50 → 1/100) çiftler ya da hizalanamayan çiftlerin payı > %10 olursa.
- **Güncelleme (bütün arşivde ölçüldü):** ilk geçişte 236 çiftin hepsi "hizalandı" ama bazılarında 10–17 görünüş kabul edilmişti. Bunların çoğu 0,1 ölçekli sahte eşleşmelerdi: 10 kat küçültülen ham çizim, son çizimin yoğun bölgelerine (tarama, yazı) "oturuyor" ve binlerce metre sahte "kaldı" etiketi üretiyor. Ölçülenler: en güçlü dönüşümün ölçeği 233/236 çiftte 1; en güçlü oyun medyanı 266, 5. yüzdeliği ~94; 10 çiftte 30'un altında. Yeni kurallar: en güçlü oy < 40 ise çift eğitime girmez; bütün görünüşler en güçlüyle aynı ölçekte olmalı (mühendis katları taşır ve döndürür ama tek tek ölçeklemez); ek görünüşler en güçlünün oyunun ≥ %25'ini almalı.

## ADR-004 · Temizlik modeli: LightGBM, proje bazında bölme, üç bant
- **Karar:** nesne başına ~90 özellik (`cleanup/features.py`), LightGBM ikili sınıflandırıcı, uzun çizgilere daha çok ağırlık. Proje kimliğinin hash'iyle sabit bölme: %15 test, %10 doğrulama. Eşikler doğrulamada her otomatik bant ≥ %99,5 kesin olacak şekilde seçilir; aradaki nesneler soru olur.
- **Neden:** tablo verisi, milyonlarca satır, CPU'da dakikalar, okunabilir özellik önemleri. Ekran kartı gerekmez.
- **Ne zaman yeniden bakılır:** görülmemiş mimarlarda tavana takılırsa (graf sinir ağı, plan §4.4). Mimar ofisi kümesine göre bölme (plan §2.4) henüz yok: aynı mimarın iki projesi hem eğitimde hem testte olabilir; test skoru bu yüzden biraz iyimser olabilir.

## ADR-005 · (ADR-006 ile değişti) Görünüş seçimi etiketten
İlk tasarımda model yalnızca plan içi nesnelerde eğitilecek, görünüşler hizalamadan bilinecekti. Yeni bir çizimde hizalanacak son çizim olmadığı için ADR-006 ile değiştirildi.

## ADR-006 · Görünüş ve nesne kararı tek modelde, komşuluk özellikleriyle
- **Bağlam:** paftalar birbirine değdiği için çizimi kümelere bölmek başarısız oldu: ilk çiftte bütün paftalar tek küme (370 m) çıktı.
- **Karar:** her nesneye komşuluk özellikleri eklenir (`cleanup/views.py`). 2 m ızgarada duvar yüzü uzunluğu, kapı yayı, oda adı, ölçü yazısı, tarama ve kesit/vaziyet/çatı/antet katmanları 6 m ve 20 m pencerelerle toplanır; ayrıca her görünüş türünün başlık yazısına uzaklık. Tek LightGBM modeli "nesne temizlikten sonra kalır mı" sorusunu bütün nesneler için cevaplar: görünüş dışındakiler silinecek sayılır. Görünüş dışı satırlar eğitimde alt örneklenir, ağırlıkla geri büyütülür.
- **Sonuçlar:** rapor iki sayı verir: uçtan uca (bütün nesneler) ve plan içi. Yeni bir çizim hizalama olmadan temizlenebilir (`cleanup/apply.py`).

## ADR-007 · Yerleştirme verisi: son çizimin cihazları ham çizimin koordinatında
- **Bağlam:** yerleştirme mobilyaya göre öğrenilecek (ADR-002), ama mobilya yalnızca ham çizimde, cihazlar yalnızca son çizimde var.
- **Karar:** temizlik hizalamasının her görünüş için bulduğu ham→son dönüşümünün tersi, son çizimdeki her cihazı ham çizime taşır (`placement/devices.py`). Cihaz, geri taşınmış konumu hangi görünüşün bölgesine düşüyorsa ona atanır; hiçbirine düşmeyenler (lejant, şema) atılır.
- **Cihaz tipi:** ofis sabit bir blok kütüphanesi kullanıyor (`docs/findings.md`), bu yüzden tip blok adına göre bir kez eşlenir (`placement/symbols.py`, Semboller sayfası). LLM önerileri önceden seçili gelir ama mühendis onaylayana kadar eğitimde kullanılmaz.
- **Sonuçlar:** ilk çiftte 436 cihaz ham plana doğru oturdu (görsel kontrol: oda ortasında armatür, duvar dibinde priz).

## ADR-008 · Oda okuma raster üzerinde, mimarın alan damgalarıyla ölçülür
- **Bağlam:** yerleştirmenin oda bağlamına (tip, alan, duvarlar) ihtiyacı var. Eski repodaki çizgi tabanlı oda okuyucu karmaşık ve ham çizimlerde zayıftı (uçtan uca kaybın ana kaynağı).
- **Karar:** temizlikten sonra kalan mimari 5 cm ızgaraya çizilir. Duvarlar 0,55 m büyütülerek ~1,1 m'ye kadar kapı boşlukları kapatılır. Kalan boş alan çekirdekleri odalardır. Her çekirdek, kendi serbest alan bileşeni içinde ve kapatma mesafesinin 1,7 katına kadar geri büyür (`placement/rooms.py`). Oda tipi, ham çizimdeki oda adı yazısından okunur.
- **Ölçü:** ham çizimdeki alan damgaları ("18.03 m²") bedava doğruluk verir. İlk çiftte 50 damgalı oda: medyan fark %-5,8, ±%5 içinde %46, ±%10 içinde %84. Sistematik küçülmenin kaynakları: çizgi kalınlığı ve mühendisin tuttuğu tezgah gibi iç çizgiler. Geri büyütme sınırı 1,45 / 1,7 / 2,2 denendi, en iyisi 1,7.
- **Ne zaman yeniden bakılır:** bütün arşivde ölçülünce; ±%10 payı %80'in altına düşerse ya da tezgah ve dolapların odaları böldüğü görülürse (o zaman yalnızca duvar benzeri çizgiler çizilir).

## ADR-009 · Patlatılmış elektrik sembolleri şekilden tanınır
- **Bağlam:** ilk son çizimde elektrik katmanlarında blok dışında yüzlerce serbest daire, tarama ve kısa çizgi var (`m-LİNYE`'de 235 daire, 253 tarama). Yalnızca bloklara bakmak cihazların önemli bir kısmını kaçırıyor; ilk sayımlarda yatak odası başına 1 priz çıkması bunun işareti.
- **Karar:** elektrik katmanlarındaki (adında LİNYE, SEMBOL, ZAYIF, ARMATÜR… geçen) blok dışı, 1 m'den kısa parçalar, 3 cm içinde birbirine değiyorsa tek sembol sayılır. 5 cm–1 m boyundaki kümeler, döndürme ve aynalamadan bağımsız bir imzayla anahtarlanır: en küçük dikdörtgen kenarları, parça sayısı, toplam uzunluk, daire ve tarama sayısı. Anahtar `~` ile başlar ve bloklarla aynı sözlükte (Semboller sayfası) bir kez etiketlenir.
- **Sonuçlar:** ilk son çizimde 548 patlatılmış sembol (0,8 sn). Projeler arası imza tutarlılığı tam arşivle ölçülecek.

## ADR-010 · Yerleştirme modeli: sayı + aday sıralama, kapılar odaların temasından
- **Karar:** her cihaz sınıfı için iki LightGBM modeli (`placement/model.py`). Sayı modeli (Poisson) oda satırından (tip, alan, oranlar, kapı ve mobilya sayısı) o sınıftan kaç cihaz olacağını tahmin eder. Sıralayıcı, aday noktaları puanlar: duvar cihazları için oda sınırında 25 cm'de bir, tavan cihazları için içeride 50 cm ızgarada. Özellikler yalnızca geometri: kapıya, oda ortasına, köşeye, duvara, en yakın mobilyaya ve her onaylı mobilya tipine uzaklık (`placement/candidates.py`). Yerleştirme: tahmini sayı kadar en iyi aday, aralarında en az 0,5 m. Kurallar yerleştirmez, sonradan kontrol eder (M3b).
- **Kapılar:** bazı mimarlar kapıyı yay değil polyline olarak çiziyor (ilk çiftte yay araması 0 kapı buldu). Kapı, oda rasterinde iki odanın (ya da bir odayla dış boşluğun) duvarsız temas ettiği yer olarak bulunur (`rooms.doorways`). Çizim biçiminden bağımsız. İlk çiftte oda başına 1,6 kapı.
- **Ölçüm:** test projelerinde sınıf başına sayı hatası ve tam sayı payı; yerleştirilen cihazların 0,5 m içinde mühendisinkiyle eşleşme kesinliği ve duyarlılığı. Bu ikisi bir de doğru sayıyla hesaplanır, böylece sıralayıcının payı ayrı görünür.
- **Eğitim verisi:** sembol tipleri mühendis onaylamadan model "ön sonuç" diye işaretlenir. İlk çiftte LLM'in "anahtar" önerdiği sembolün kapıya medyan uzaklığı 1,95 m çıktı; anahtarlar kapı yanında olur, yani öneri muhtemelen yanlış. Onaysız tiplerle eğitilmiş model kullanıma alınmaz.

## ADR-011 · Temizlik iki model + tarz anahtarı; kesinlik hedefi %98
- **Bağlam (ilk tam eğitim, 2026-10-10, 176/28/43 proje):** tek model uçtan uca %94,4, plan içinde %82,7 doğru (0,5 eşiğinde). Ama: (1) 0,5 eşiğinde duvar benzeri çizgilerin medyanda %34'ü silindi; görünüş dışı satırlar (hepsi "sil", ağırlıkla büyütülmüş) modeli silmeye itti. (2) %99,5 kesinlik hedefi tutmadı: eşikler 0,9987 / 0,048'e kaçtı, nesnelerin %99,9'u soru oldu. (3) Test projelerinde mühendisin blokları (mobilya, vitrifiye, araba) tutma payı %10–90 arası değişiyor ve plan içi doğrulukla -0,72 ilişkili: hataların çoğu duvar değil, projeden projeye değişen tarz. (4) Çizimdeki konum (`rel_x`) 6. en etkili özellikti: projeye özgü, genellemeyi bozar.
- **Karar:** iki model. Görünüş modeli bütün nesnelerde "kat planında mı", plan içi model yalnızca plan nesnelerinde "silinir mi". Birleşik: p(sil) = 1 − p(plan) · (1 − p(sil | plan)). Her eğitim projesine `style_keep_blocks` girdisi: plan içi blokların çoğu tutulduysa 1. Yeni çizimde kullanıcı seçer, varsayılan 0 (sert temizlik). `rel_x`, `rel_y` çıkarıldı. Kesinlik hedefi %98: etiketler bir hizalamadan geliyor, insan elinden değil; %99,5 onların gürültüsünün üstünde.
- **Ölçü:** rapora "0,5'te kaybedilen duvar payı" eklendi; korumaların gerekip gerekmediğini bu söyler.

## ADR-012 · "Plan içi" = eşleşen geometrinin 3 m çevresi, kutu değil
- **Bağlam:** ikinci eğitimde (iki model) 0,5 eşiğinde duvar benzeri çizgilerin %59'u silindi. Sebep etiketlerdi: bazı projelerde duvar katmanının %60–82'si "plan içinde silindi" etiketliydi. Bir projede Tekstil'de zayıf bir görünüşün bölge kutusu 420 m genişliğindeydi; içine düşen kesit ve görünüş paftalarının duvarları "plan içinde silindi" sayıldı. İlk düzeltmede (kutu yerine 2 m çevre + boşluk doldurma) kapalı pafta çerçeveleri bütün paftayı doldurdu.
- **Karar:** bir nesne, mühendisin tuttuğu 20 m'den kısa geometrinin 3 m çevresindeyse plan içidir (`align._view_mask`). Boşluk doldurma yok. En büyük parçanın eşleşmelerinin %5'inden azına sahip adacıklar (her paftada tekrarlanan antet ve çerçeve parçaları) atılır. Bundan uzak her şey görünüş dışıdır. Görünüş bölge kutuları yalnızca cihazları ve odaları görünüşlere atamak için kalır.
- **Sonuçlar:** büyük odaların ortası (duvardan 3 m'den uzak) görünüş dışı sayılabilir; orası zaten silinecek mobilya/araba alanı. Kalan "plan içinde silinmiş duvarların" bir kısmı gerçek: mühendis bazı projelerde vaziyet planını ya da bir kesiti son çizime alıp sadeleştiriyor.

## ADR-013 · Korumalar: adı duvar/kolon olan katman plan içinde otomatik silinmez
- **Bağlam (üçüncü eğitim):** plan içi doğruluk %87, ama adı duvar/kolon olan katmanlarda kalması gereken uzunluğun %27,7'si 0,5 eşiğinde silindi. Kayıp birkaç projede yoğun (bir projede: dış duvarı tek parça çizen uzun polyline'lar). Plan içi model uzun çizgileri akslara ve çerçevelere benzetiyor; iç duvarlar doğru kalıyor.
- **Karar:** görünüş modelinin plan içi dediği (p ≥ 0,5) ve katman adında duvar/kolon/perde anahtar kelimesi olan nesnenin silme olasılığı 0,49'a kırpılır: otomatik silinmez, en kötü soru olur (`train.guarded`, `apply.decide`). Eşikler korumalı olasılıklarla seçilir. Rapor plan içi sonuçları korumalı ve korumasız ayrı verir.
- **Sınırı:** katman adı anlamsız olan mimarlarda korumayı ancak görünüş modeli ve özellikler sağlar. Duvar katmanı adı LLM sözlüğüyle genişletilecek (docs/llm.md).

## ADR-014 · Temiz çizime modelin en iyi tahmini uygulanır; belirsizler yalnızca listelenir
- **Bağlam:** dördüncü model bir test projesinde (Cevriye BERKOĞLU, hiç görülmemiş) 0,5 eşiğinde çizimci tarzı temiz bir altlık üretti: duvar, kolon, pencere, merdiven ve kasa kaldı; aks, ölçü, yazı, fayans, mobilya ve vitrifiye gitti. "Cevap anahtarı" (mühendisin son çiziminden çıkan etiket) daha kötüydü: mühendisin değiştirdiği ya da yeniden çizdiği duvarlar "silindi" görünüyor. Yani test ölçüleri (plan içi %86–89) gerçek kaliteyi olduğundan düşük gösteriyor. %98 kesinlik bantları ise bu gürültü yüzünden plan içi nesnelerin %94'ünü soruya itti; sorular çizimde bırakılınca planlar hiç temizlenmemiş oldu.
- **Karar:** çizime kesin kararlar olduğu gibi, belirsiz banttakiler p(sil) ≥ 0,5 ile uygulanır (`apply.best_guess`). Korumalar 0,5'in altında tuttuğu için duvar ve kolon katmanı plan içinde kalır. Belirsiz bant gözden geçirme listesi olarak kalır (`questions.json`, önizlemede turuncu).
- **Sonraki:** belirsizleri nesne nesne değil katman/blok grubu halinde sormak; test ölçüsünü, etiket gürültüsünden etkilenmeyen bir insan değerlendirmesiyle (inceleme ekranı) tamamlamak.

## ADR-015 · Sonnet incelemesi: deneme sonuçları (3 kolay test projesi)
- **Kurulum:** model temizliğinden sonra, plan içindeki kararsız nesne gruplarının (ortalama p 0,2–0,8 ya da model grubu bölmüş) kesit resimleri 15'lik paketlerle başsız Claude Code'a (Sonnet 5.5, abonelik) gönderilir; cevap kalsın/silinsin + ne olduğu (`cleanup/review.py`). Proje başına 45 grup, 3 çağrı, API karşılığı ~$0,33 (Claude Code'un kendi talimatları çağrı başına ~40 bin önbellek tokenı getiriyor; doğrudan API ile daha ucuz olur).
- **İlk deneme başarısız:** birbirine değen her şey tek grup sayılınca duvarlar bütün paftaları birleştirdi (bir grup = bütün kat planı). Sonnet "kat planı, kalsın" deyip içindeki çöpü de geri getirdi. Düzeltme: yalnızca aynı katmandaki parçalar birleşir; 8 m'den büyük gruplar incelenmez; talimatta "kesit/görünüş/çatı/vaziyet paftaları tamamen silinir".
- **İkinci deneme:** Bir projede kararsız gruplarda cevap anahtarıyla uyum model 17/45 → Sonnet 32/45. Modelin sildiği döner merdivenler geri geldi; kesit ve cephe paftalarındaki kalıntılar silindi. Proje C (35 → 34) ve Proje D'te (40 → 38) model zaten iyiydi; Sonnet çok az değiştirdi (173 ve 27 nesne), alan kontrolü değişmedi.
- **Sonuç:** Sonnet kararsız gruplarda tanıma işini iyi yapıyor (ne olduğunu doğru adlandırıyor). Kazanç modelin sistematik hata yaptığı projelerde büyük, iyi projelerde küçük. Ancak "eksik oda" sorununu bu kurulum çözmüyor: eksik dış çeper ve bölme duvarlar 8 m'den büyük yapı grupları, incelemeye girmiyor. Sonraki adım: alan kontrolünün "kapanmayan oda" dediği yerlerde odanın ham ve temiz halini yan yana gösterip silinmiş aday çizgileri numaralı sormak.

## ADR-016 · Modelden sonra kesin bitirme kuralları; çıktı beyaz
- **Bağlam (kullanıcının AutoCAD kontrolü, Cevriye BERKOĞLU):** yeni model "meseleyi çok iyi çözmüş", ama mühendisin hedef altlığından farkları: oda isimleri silinmiş; kolon dolguları ve taramalar kalmış; duvar içinde ve oda dışında kesikli çizgiler kalmış; renkler mimarın katman renkleri. Hedef: odanın içi boş, her odada tek bir ad (Salon, Mutfak, E.Yatak Odası, Kat Holu, Asansör…), yalnızca merdivenin üst kolu kesikli, her şey beyaz.
- **Karar:** öğrenme değil, kesin kurallar (`cleanup/finish.py`), modelin en iyi tahmininden sonra: plan içindeki oda adı yazıları kalır (alan, kot, kaplama, uzun yazılar hariç); plan içindeki taramalar silinir (baca/şaft katmanı hariç); kesikli çizgi tipindeki nesneler silinir (merdiven katmanı hariç; sahneye `dashed` alanı eklendi). Yazıcı (`apply.make_white`) bütün katmanları renk 7'ye, bütün nesneleri ve blok içeriklerini BYLAYER'a çevirir, gerçek renkleri ve MTEXT renk kodlarını siler.
- **Sonraki:** adı silinmiş ya da hiç olmayan odalara ad yerleştirmek (oda okuma + ham çizimdeki ad).

## ADR-017 · Eski sistem bu sitede de çalışır
- **Karar:** "Eski sistemle temizle" işi eski repoyu kendi sanal ortamıyla çalıştırır (`python -m corpus.pipeline clean … --store D:\cad-ai-data\legacy\store`) ve sonucu modelinkiyle aynı çıktı biçimine getirir (`cleanup/legacy.py`). Arşiv projelerinde iki sonucun yanında mühendisin temizliğinin görüntüsü de çizilir. İndirilen dosyalar yöntemi söyleyen adlar alır.
- **İlk karşılaştırma (Cevriye BERKOĞLU):** eski sistem bu mimar için profil olmadığından mobilya, vitrifiye ve kapı yaylarını bıraktı, normal kat planını pafta başlığından yanlış kesti; kullanıcı AutoCAD'de yeni modeli belirgin biçimde daha iyi buldu.

## ADR-018 · Mühendisin hedef çizimiyle birebir karşılaştırma ve düzeltmeler (Cevriye BERKOĞLU)
- **Ölçü:** kullanıcı mühendisin altlığının ilgili kısmını (`Drawing1.dwg`, 3 kat planı, tek katman `m-mimari`, santimetre) verdi. Ham çizim bu hedefe hizalandı; hedef ve bizim çıktı aynı 2 cm ızgaraya çizildi ve 4 cm toleransla karşılaştırıldı (hedefin her çizgisi, ham çizime en iyi oturduğu dönüşümle yerleştirildi). Yazılar ölçüye girmez (oda adları bilinçli bir fark).
- **Bulunan nedenler ve düzeltmeler:**
  1. Bazı kolon ve perde duvarlar yalnızca tarama olarak çizilmişti; "plan içi taramaları sil" kuralı (ADR-016) onları tamamen siliyordu. Artık duvar/kolon/perde/panel katmanındaki tarama sınır çizgisine dönüşür (yeni karar `OUTLINE`, `apply.hatch_to_outline`). Öncelik: önce yapı katmanı, sonra küçük tarama.
  2. Havalandırma şaftı dolguları (≤ 60 cm küçük taramalar) silinmez; mühendis tutuyor. Patlatılmış şaft sembolleri de tanınır (birbirine değen küçük kareler + çaprazlar).
  3. Ölçü / aks / kot katmanları plan içinde silinir (duvar dışındaki çentikler).
  4. Oda adları: bir odada her farklı ad bir kez (açık planda "Giriş" ve "Salon" birlikte). Ayrıca kapalı odaya düşmeyen adlar (balkon, merdiven) kalan çizgilere 1,5 m'den yakınsa kalır.
  5. Kapı/pencere katmanında duvardan dışarı taşan 35 cm'den kısa çentikler silinir.
  6. Çıktı her zaman santimetre: son çizimlerin 258'den 256'sı cm, ham çizimler karışık (157 cm, 41 dm, 41 m, 2 mm). Yazıcı çizimi ölçekler, `$INSUNITS = 5` yazar, `$LTSCALE`'i aynı oranda büyütür.
- **Denenip ölçülerek bırakılanlar:** "içinden çapraz geçen daire dizisi şafttır" (fazla %2,3 → %6,9, eksik değişmedi) ve "odanın içinde kalan 0,5–1,3 m kapı katmanı çizgisi kapı kanadıdır" (eksik +%3; mühendis bazılarını tutuyor).
- **Sonuç:** hedefteki çizgilerin %96,5'i çıktıda var (planlar: %94,6 / %97,1 / %97,6). Plan alanı içinde çıktımızdaki çizgilerin %13'ü hedefte yok; çoğu pencere/kapı kasası uçlarındaki kısa parçalar ve komşu kat kenarları. Kalanlar: holdeki kesikli çatı kapağı (kesikli çizgi kuralı siliyor), birkaç kapı eşiği, mobilya arka çizgileriyle çakışan sıva çizgileri.

## ADR-019 · Katman tipine göre seçim (`cleanup/layers.py`, `cleanup/elements.py`)
- **Gözlem:** mühendis mimarın çizimini çizgi çizgi temizlemiyor; katmanları kapatıp kalanı kendi `m-mimari` katmanına yeniden çiziyor. Mimar duvar ve kolonları taramalı, kalın çiziyor; mühendis yalnızca dış hatları bırakıyor. Arşivin 247 eşleşmesinde ortanca projede mühendisin `m-mimari` çizgilerinin %74'ü mimarın bir çizgisinin 3 cm yakınında; kalan kısım yeniden çizilmiş ya da hizalanamamış (çizgi bazlı etiketler orada yanlış öğretir). Proje E'de (ArchiCAD) `m-mimari`'nin %99,8'i ham çizgilerin üstünde; ama 20 ayrı "Pen" katmanından seçilmiş.
- **Karar:** (1) her mimar katmanına adından bir eleman tipi verilir (duvar, kolon, pencere, kapı, kapı+pencere, merdiven, korkuluk, şaft, oda adı, kapat, belirsiz); ArchiCAD'in `_Pen_No__n` eki atılır. Adı bir şey söylemeyen ("0", "sketch", "Standart") ya da ofise göre anlamı değişen ("Döşeme", "Nokta") katmanlar *belirsiz*dir; onları şimdilik temizlik modeli parça parça seçer, hedef: mühendisin tek ekranda onaylaması. (2) Tutulan katmanların içinde mühendisin bırakmadığı parçalar geometriyle atılır: patlatılmış tarama çizgileri (kısa, eğik, ≥4 paralel komşu), kapı yayları (ARC, kavisli polyline ya da 5 cm'lik parçalara patlatılmış; aykırı noktaları atan çember uydurma, 0,5–1,3 m yarıçap, 60–120°) ve yayın çeyrek kutusundaki kanat/kasa parçaları, küçük işaret daireleri, duvardan taşan çentikler, kesikli çizgiler (merdiven hariç), oda adı dışındaki yazılar, plan dışı her şey. (3) ~~Seçilen geometri sıfırdan yeni bir DXF'e çizilir~~ — ADR-020 ile kaldırıldı: mimarın nesneleri olduğu gibi kalır.
- **Ölçü (Drawing1, Cevriye BERKOĞLU):** hedefin %96,6'sı bulundu, çizgilerimizin %11,6'sı fazla (v3 model+kurallar: %96,5 / %13,2). Başka bir arşiv eşleşmesinde: hedefin %64'ü, katmanlar mühendis tarafından işaretlenmiş gibi verilince %87 (v3: %62,5). Proje F'de mühendisin planı mimarın arşivdeki dosyasıyla örtüşmüyor (başka revizyon); ölçülemez.
- **Denenip bırakılan:** "mühendis-işaretli katman" taklidini arşivden çıkarmak (katmanın ≥%30'u `m-mimari`'de ise açık): karışık katmanlarda (Cevriye `Kapi_Pencere`: %26) yanlış kapatıyor; isimden gelen tip daha doğru.

## ADR-020 · Mimarın çizimine dokunulmaz; önce katmanlar, olmazsa nesne nesne
- **Kaynak:** ofisin elektrik mühendisi (kullanıcı aracılığıyla, 2026-10-10): asıl projeye asla dokunulmaz; katmanlar kapatılır ve AutoCAD'de çizgi kalınlığı (lineweight) gösterimi kapatılır, her zaman böyle çalışılır. Bazı mimarlar her şeyi tek katmana çizer. Arşiv bunu doğruluyor: `m-mimari` çizgileri mimarın çizgilerinin üstünde (Bülent %99,8); "mimarın kalın duvarları" geometri değil, çizgi kalınlığı.
- **Karar:**
  1. Çıktı mimarın kendi nesneleridir (yay yay, blok blok); yeniden çizim yok. Yapılanlar: silme, karışık blokların patlatılması, duvar/kolon/şaft taramasının dış hatta çevrilmesi, beyaz renk, çizgi kalınlıklarının kapatılması (`$LWDISPLAY = 0`, katmanlar varsayılan kalınlık, nesneler katmandan), santimetreye ölçek (`apply.plain_style`, `write_clean`).
  2. Yöntem seçimi (`extract.choose_mode`): plan geometrisinin en az yarısı tipi bilinen katmanlardaysa ve aralarında duvar/kolon varsa **katman yöntemi** (ADR-019); değilse **nesne yöntemi** (temizlik modeli + `finish.py` kuralları, v3). Sayfadan zorlanabilir.
  3. Katman tipi önce mühendisin sözlüğünden (`DATA/layers/types.json`, katmanın temel adıyla; her değişiklik `types_log.jsonl`'a), sonra isim kurallarından gelir. Temizle sayfasında her sonucun katman tablosu (katmanın plandaki resmi, tipi, plandaki ve kalan uzunluğu) var; mühendis tipi düzeltip "yeniden temizle" der, düzeltme sonraki bütün projelerde de kullanılır.
- **Henüz yok:** nesne yönteminde Sonnet kontrolü (`review.py` denendi, işe bağlı değil); katman tiplerinin içerikten (isim değil geometri) tahmini — ArchiCAD "Pen" katmanları için gerekli.

## ADR-021 · Araç kutusu + Claude: sabit programlar bulur, Claude bölge ve grup düzeyinde karar verir
- **Deney (arşiv, yalnızca mühendis çizimlerinden gelen etiketler; eğitim 175 / doğrulama 28 / test 43 proje):** Claude'un arşivden kendi yazdığı katman kural kitabı katman kararında isim kurallarından belirgin iyi değil (test F 0,676 / 0,665; katmanları kesin bilmek de 0,679). Asıl kazanç görünüş seçimi: pafta bölgelere ayrılır, Claude numaralı genel resimden ve başlıklardan kat planlarını seçer. Bölge doğruluğu Claude %82, istatistik görünüş modeli %60. Yalnız görünüş seçimi yetmez (plan içindeki mobilya/tarama/ölçü kalır: fazla %43–49).
- **Karar:** `regions.py` bölgeleri bulur; `tools.py` sabit araçlar (kapatılan katmanlar, yazı ve ölçü, mobilya imzaları, kapı yayları, tarama, kesikli çizgi, belirsiz katmanda model, duvardan taşan çentikler) her biri ayrı ayrı ne sileceğini bulur; `agent.py` iki Claude çağrısı (`claude -p`, abonelik): (1) görünüş seçimi, Claude'un 1197 eğitim bölgesinden yazdığı `DATA/agent/view_rulebook.md` ile; (2) kontrol: kalanların ve araç renklerinin resmi, araç sayıları ve arşivdeki isabetleri, katman tablosu → araç kapatma / katman tipi düzeltme / bölge ekleme-çıkarma. Claude nesne nesne karar vermez. Claude cevap vermezse istatistik görünüş modeli kullanılır.
- **Mobilya:** mühendis kataloğu adlandırana kadar "mobilya" = eğitim projelerinde en az 2 kez görülmüş ve mühendislerin ortalama ≥%80'ini sildiği blok şekilleri (1156 imza); onaylanan tipler (kapı/pencere/merdiven hariç) eklenir.
- **Araçların isabeti (doğrulama+test, sildiğinin mühendis de sildiği payı):** kapatılan katmanlar %89, yazı/ölçü %95, mobilya %84, kapı yayı %78, tarama %93, model %89, çentik %62. Kesikli çizgi ölçülemedi (arşiv sahneleri çizgi tipi alanından önce).
- **Bütün akış (görsel puan, aynı projelerde):** test (8 proje, Claude görünüşü olan) araçlar+Claude bulunan %66,4 / fazla %7,5 / F 0,765; önceki en iyi 0,762; eski model 0,672. Doğrulama (28) 0,689; isim kuralları+Claude görünüş 0,686; eski model 0,613.
- **Bulunup düzeltilen:** tarama çizgisi aracı çok yoğun taramada belleği tüketiyordu (en yakın 64 komşuya sınırlandı); AIA `S-STRS` merdiven katmanı tanınmıyordu, basamaklar ve kesikli üst kol siliniyordu (arşivde mühendisler `S-STRS`'yi %46, `S-STRS-MBND`'yi %89 tutuyor; `STRS` merdiven sözcüğü oldu). İkincisi test projesinde (Ali Kemal ARSLAN) gözle bulundu.
- **Bilinen eksikler:** Claude'un kontrol adımı iki denemede de düzeltme vermedi ve merdiven hatasını görmedi: resimden ince hata yakalaması zayıf. Temiz DWG kullanılmayan blok tanımlarını taşıyor (12 MB).
- **Kullanıcının elle kontrolünden (Ali Kemal ARSLAN, 2026-10-12):** (1) kolonlar ve perdeler silinmişti: Revit/AIA çıktısında bir katın bütün kolonları `A-DETL-GENF` (kapatılan detay katmanı) üzerinde tek dolgu bölgesi. Yeni koruma `elements.structural_fills`: katmana bakmadan, sınır döngüleri dar olan (kısa kenar ≤ 80 cm, uzun kenar 25 cm–8 m, en az 4 kenar, blok dışında; sınır uzunluğunun ≥ %70'i) taramalar kolon/perdedir; hiçbir araç silmez, dış hattı kalır (`OUTLINE`). Oda boyutundaki zemin desenleri ve bloklardaki küçük semboller (kot, kesit oku) dışarıda kalır. (2) Balkonların dış duvarları silinmişti: parapetler AIA `A-FLOR-HRAL` (korkuluk) katmanında; "FLOR" kuralı onu kapatıyordu. `HRAL` artık korkuluk, "FLOR" kuralından önce bakılır.

## ADR-022 · Yerleştirme verisi bizim temizliğimizden; cihazlar yerel çizgi örtüşmesiyle taşınır
- **Karar 1:** Odalar, mühendisin tuttuğu çizgilerden değil, ham çizimin bizim araç kutumuzla temizlenmiş halinden okunur. Yeni bir projede yalnızca bizim temizliğimiz olacak; model eğitimde de aynı türden planı görmeli. Kat planları mühendisin çiziminden (hizalamanın görünüşleri) alınır, veri seti için Claude çağrısı yok.
- **Karar 2:** Cihazlar ham plana, çevresindeki (2 m) mühendis çizgilerinin ham çizgilere ≥ %40 oturduğu dönüşümle taşınır; dönüşüm cihazı kendi bölgesine düşürmeli. Eskiden ilk tutan bölge seçiliyordu: Ali Kemal ARSLAN'da bir görünüşün bölgesi iki katı kapsıyordu ve bir katın cihazları planlar arası boşluğa düşüyordu; kolon şemasındaki TV/data çıkışları da plana sayılıyordu. Aynı tipte ve ham koordinatta 25 cm içinde üst üste düşen cihazlar (birbirinin aynısı katlar tek tipik plana oturur) bir kez sayılır. Ali Kemal ARSLAN: odalara düşen cihaz %57 → %81.
- **Karar 3:** Çıktı doğrudan DWG; EDL ara biçimi kaldırıldı (plan-placement.md).
- **Karar 4:** Gradient modeli "benzer odayı kopyala" yöntemini geçmeli; ikisi aynı test projelerinde ölçülür.

## ADR-023 · Çizim sayfası; cihazsız katlar eğitim dışı; kapı tarafı özelliği; Haiku seçimde değil adlandırmada
- **Karar 1 (Çizim sayfası, `placement/place.py`):** Yerleştirme, Temizle sayfasının bir çıktısını alır. Ham çizim (mobilya ondan okunur) ve temizliğin tuttuğu nesneler gerekir: yeni çıktılar ikisini kaydeder (`scene.npz`, `decision.npz`); eskilerde kaynak çizim yeniden okunur ve ham nesne, çizgilerinin ≥ %80'i temiz DXF'te varsa (1 cm yuvarlak, yönsüz eşleşme) tutulmuş sayılır (Proje F: temizlikte 10.651, eşleşen 10.716). Kat planları temiz çizimin kümeleridir (1 m ızgara, 2 m boşluk köprülenir, kutusu ≥ 30 m²): temizlik zaten yalnız kat planlarını bırakır. Çıktı temiz DXF'in üstüne ELK-* katmanlarında basit bloklar; arşiv projesinde mühendisin cihazlarıyla karşılaştırma resmi.
- **Karar 2 (cihazsız katlar):** Mühendis tipik katı bir kez çizer; aynı plandaki diğer katların odaları "0 cihaz" diye öğretiyordu. Yaşam odası (yatak, salon, mutfak, banyo, hol) başına < 1 cihazı olan görünüşler eğitimden ve ölçümden çıkarılır (odaların %8'i). Aynı test odalarında ölçüm: tavan F1 0,619 → 0,622, priz 0,446 → 0,442, etanj 0,432 → 0,405, anahtar 0,194 → 0,231; sayılar mühendisinkine yaklaştı.
- **Karar 3 (kapı tarafı):** Anahtar kapının kilit tarafına konur. Kapı yayının merkezi menteşe; iki yay ucundan kilit olanı, menteşeden çizgisi kapı boşluğunu kesen uçtur (iki yanında farklı odalar), açık kanadın ucu tek odanın içindedir; karar vermezse kasaya değen uç. Adaylara `d_latch`, `d_hinge` eklendi.
- **Karar 4 (Haiku seçimde değil):** Model her oda için 5 yerleşim önerdi (model, gürültülü model, en benzer 3 arşiv odası), Haiku 5 istem türüyle seçti (39 test odası, ~200 çağrı, ~$1,4). Hiçbiri modelin kendi seçimini geçemedi (1 m F1: model 0,568; en iyi istem "arşiv istatistikleri" 0,553; örnek resimli istem 0,433 = rastgele; üst sınır 0,700). Mühendisin tercihi resimden okunacak kadar belirgin değil. Haiku'nun ucuz görsel tanıması **adlandırmada** kullanılır: mobilya (1.668 şekil, $0,61) ve tipsiz semboller küçük resimden önerilir; öneriler mühendis onaylayana kadar yalnızca deneylerde (`proposals_used`) kullanılır.
- **Karar 5 (ölçüm, 2026-10-13 gece):** Aynı test odalarında 1 m F1 (tavan / priz / etanj / anahtar): önceki model 0,622 / 0,442 / 0,405 / 0,231; kapı tarafıyla 0,624 / 0,472 / 0,426 / 0,267; Haiku mobilya tipleri eklenince artış yok (0,621 / 0,460 / 0,417 / 0,275). Etiketlerin elle örneklemesi ~%65–70 doğru; yataklar çoğunlukla blok değil. Gönderilen model mobilya tiplerini kullanmaz (`placement-20261013-0134`, tip sütunları boş).
- **Karar 6 (sayı yuvarlama):** Sayı = taban(beklenen + eşik); anahtar 0,7, etanj 0,6, diğerleri 0,5. Anahtar F1 0,267 → 0,305; konan sayı mühendisinkine yaklaşır (anahtar 265 → 522, mühendis 455).
- **Karar 7 (oda içi sıralama denendi):** Aday sıralayıcı oda başına sorgu olarak LambdaRank ile eğitildi; kazanç yok (priz 0,472 → 0,450, diğerleri ±0,01). İkili sınıflandırıcı kalır.
- **Karar 8 (ofis sembolleri):** DWG'de lamba, priz ve etanj priz ofisin kendi bloklarıyla çizilir (en çok kullanılan onaylı blok: `A$C69B63251`, `A$C452F5B38`, `A$C66B11EAB`). Geometrileri blok tabanından uzakta çizildiği için (lambanın çarpısı ~3 m ötede) bir kez merkezlenip `DATA/symbols/office_blocks.json`'a alınır (`place.harvest_office_blocks`); dosya yoksa basit semboller. Duvar cihazları odaya döner: 0°'de duvar solda (-x), oda sağda. Anahtarın ofiste bloğu yok (hep patlatılmış), kendi basit sembolümüz.
- **Karar 9 (adsız odalar):** Adı okunamayan odaların (test odalarının %28'i) tipi geometriden tahmin edilir (`model.train_room_types`, `room_type.txt`; alan, oran, kapı, pencere, mobilya boyutları): adı bilinen test odalarında %70 doğru. Adsız 90 test odasında 1 m F1: tavan 0,567 → 0,579, priz 0,329 → 0,361, etanj 0 → 0,255, anahtar 0 → 0,146. Haiku'ya oda resmi gösterip tip sormak (%39 doğru, $0,20) daha az kazandırdı (etanj 0,098, anahtar 0,075); kullanılmaz.
- **Karar 10 (sınıf başına aralık):** Bir odada aynı tip iki cihaz arası en az: lamba 1,5 m, anahtar 1,0 m, etanj 0,3 m (çift çizilir), diğerleri 0,5 m. Lambalar 0,5 m aralıkla yan yana düşüyordu; tavan 1 m F1 0,624 → 0,663.

## ADR-024 · Patlatılmış semboller tek katmanda gruplanır ve şekilden tanınır
- **Teşhis (2026-10-13):** Eğitim verisinde yatak odalarının %72'sinde anahtar yoktu (lambasız yatak odası %11). Mühendisin son çiziminde bile tanınan anahtar lambaların ~1/4'üydü (bir projede: 30 lamba, 8 anahtar). Linye uçları ölçüsü (önceki rapordaki %38) yanıltıcıydı: boş uçların üçte ikisi başka bir linyeye değen ek yerleri; gerçek uçların %94'ü zaten tipli bir cihazda.
- **Neden 1:** Patlatılmış sembolün parçaları katmana bakılmadan birleştiriliyordu; anahtarın sapı zayıf akım / acil çizgilerine değince 2 m'lik bir grup oluyor ve "sembol için büyük" diye atılıyordu. **Karar:** yalnızca aynı katmandaki parçalar birleşir (`symbols.exploded_instances`). Farklı imza sayısı 6.190 → 4.589.
- **Neden 2:** İmza parça sayısını ve uzunluğu tam sayar; aynı anahtar başka dosyada farklı daire bölmesiyle başka anahtar oluyor ve onaysız kalıyordu; sapına bir linye parçası ve sonraki bağlantı noktası değen anahtar da tek seferlik bir imza alıyordu. **Karar (`placement/shapes.py`):** her onaylı imzanın şekil resmi (en küçük dikdörtgenin uzun ekseninde 16×16 ızgara, ayna/yarım tur aynı), oranı, boyu ve daire sayısı; onaysız grup, en yakın onaylı şekle uzaklık ≤ 0,25 ise onun tipini alır. Eşik: 111 onaylı imzada her biri dışarıda bırakılıp yeniden bulundu; 0,25'e kadar 28/28 doğru, hatalar 0,27'den sonra (buat/cihaz değil, aplik/tavan). Bilinmeyen grup (≤ 2,5 m) bir daire, dolgu ya da yaydan başlayıp en yakın değen parça eklenerek büyütülür; her adım karşılaştırılır (`peel`). Yalnız mühendisin son çizimlerini okurken kullanılır (`devices.locate(by_shape=True)`).
- **Sonuç:** veri setinde anahtar 2.430 → 7.362, anahtarsız yatak odası %72 → %14; diğer tipler ~%1–2 arttı. Aynı test odalarında 1 m F1 (yeni, doğru etiketlerle; eski model → yeni model `placement-20261013-1134`): tavan 0,659 → 0,664, priz 0,472 → 0,451, etanj 0,422 → 0,394, anahtar 0,389 → 0,549 (konan 523 → 1.390, mühendis 1.255). Uçtan uca anahtar (1 m isabet / biz / mühendis): Cevriye 21/32/31, Ali Kemal 46/64/75. Priz ve etanjdaki küçük düşüş, eğitime giren görünüşler değiştiği için (cihazlı görünüş sayısı 8.118 → 8.513); izleniyor.
- **Kalan:** bir katın hizalaması tutmayan projelerde (bir projede) o katın bütün cihazları kayıp; bu hizalamanın işi.

## ADR-025 · Mobilya dağınık çizgilerden de okunur; şekil sınıflarını mühendis bir kez adlandırır
- **Teşhis (2026-10-13):** Mobilya yalnızca blok referanslarından okunuyordu ve mühendisin onayladığı tek bir mobilya tipi yoktu (`types.json` yok; Haiku önerileri onay beklediği için kullanılmıyordu). Mimarların çoğu mobilyayı patlatıyor: 25 ham çizimin 15'inde hiç mobilya bloğu yok, yataklar "0" ya da mimari katmanında çoklu çizgi. Cevriye BERKOĞLU'nun odalarında hiç mobilya görünmüyordu.
- **Karar 1 (`furniture.loose_instances`):** kat planı içinde temizliğimizin sildiği, yazı/tarama/ölçü olmayan dağınık çizgiler, aynı katmanda birbirine değenler bir parça olacak şekilde gruplanır (patlatılmış sembol gibi, ADR-024). Parça 0,3–4,5 m, kısa kenarı ≥ 15 cm; kapı yayı (0,5–1,3 m yarıçap, ~90°) olan gruplar alınmaz. Eğitimde de yeni projede de aynı tanım (temizlik maskesi her ikisinde var). İlk 32 çiftte oda başına parça 2,25 → 5,3, mobilyası görünen yatak odası %47 → %95.
- **Karar 1b (sayfa işaretleri, veri filtresi değil):** ilk listede en sık sınıflar oda adı çerçeveleri, ince şeritler ve kapı bloklarıydı. Bunları veriden de atmak (kısa kenar ≥ 25 cm, içinde yazı olan çerçeve yok) etanj F1'i 0,450 → 0,399 düşürdü (banyo adı yazısı vitrifiyenin kutusuna düşüyor). Bu yüzden parçalar yalnızca işaretlenir (ince / içinde yazı / kapı yayı, `furniture._Marks`) ve işaretli liderler Mobilya şekilleri sayfasına konmaz; modele hepsi girer.
- **Karar 2 (`furniture_shapes.py`):** veri seti kurulurken her parçanın şekil resmi (ADR-024'teki 16×16 ızgara) saklanır. Tam imzalar aynı yatağı onlarca anahtara böldüğü için imzalar en sık görülenden başlayarak şekil sınıflarına toplanır (lider kümeleme, uzaklık ≤ 0,25). 22.755 imza → 11.618 sınıf; ilk 300 sınıf parçaların %55'ini, ilk 1.000 sınıf %72'sini kapsar.
- **Karar 3 (Mobilya şekilleri sekmesi):** en çok projede görülen ilk 300 sınıf, liderinin planındaki küçük resmiyle listelenir; mühendisin seçtiği ad sınıfın bütün üyelerine (bütün projelerde) uygulanır (`resolved_types`), yeni çizimde görülmemiş bir şekil en yakın adlandırılmış lidere ≤ 0,25 ise onun adını alır (`Matcher`). Haiku küçük resimlerden öneri yapar, öneri sayfada seçili gelir ama onaylanmadan kullanılmaz (ADR-023 Karar 4). "none", "annotation", kapı, pencere, merdiven diye adlandırılanlar mobilya sayılmaz.
- **Sonuç (adlar olmadan, yalnız parça boyutlarıyla; aynı test odaları, 1 m F1, `placement-20261013-1134` → `-1314`):** tavan 0,662 → 0,665, priz 0,460 → 0,469, etanj 0,389 → 0,450, anahtar 0,538 → 0,568. Adlandırmadan sonra veri seti yeniden kurulur ve model yeniden eğitilir.
- **Bulunup düzeltilen:** bir ham çizimin yazı dizisi sabit genişlikte (344.726 × 7.312 karakter) 9,4 GB istiyor; küçük resim işi o çizimde çöküyordu. Her imza için 3 farklı çizimden örnek tutulur, yüklenemeyen atlanır. Veri seti kurulumu temizlik maskesini ham çizim ve temizlik modeli başına önbelleğe alır (`placement/keep`).

## ADR-026 · Aday, odasına göre okunur; oda içinde sıralanır; açık mutfak ayrı tip
- **Teşhis (2026-10-13):** sayıyı doğru bilsek bile 1 m F1 tavanda 0,74, prizde 0,54, etanjda 0,55, anahtarda 0,69'da kalıyor (test odaları, `placement-20261013-1421`): kayıp çoğunlukla "nereye"de, "kaç tane"de değil. Aday özellikleri mutlak mesafeler; sıralayıcı her adayı bütün arşivdeki adaylara karşı ikili sınıflıyordu, oysa soru "bu odanın noktalarından hangisi".
- **Deney düzeni:** varyantlar aynı veriyle bellekte eğitilir (`tmp/scripts/theory/exp.py`); ayarlar doğrulama projelerinde seçilir, test projelerinde raporlanır. ADR-023'teki yuvarlama eşikleri test projelerinde seçilmişti; bu kez seçim doğrulamada.
- **Karar 1 (göreli özellikler, `candidates._relative`):** odanın kutusundaki yeri (aynaya duyarsız), oda boyuna bölünmüş kapı/merkez/pencere mesafesi ve oda içindeki sırası (kapıya, pencereye, merkeze, kilit tarafına göre; duvar ve tavan adayları ayrı). Tek başına priz +0,025 (doğrulamada da testte de).
- **Karar 2 (oda içi sıralama):** sıralayıcı lambdarank, her oda bir grup; 800 tur, 127 yaprak. 5.000'den çok adaylı "oda" birleşmiş açık alandır, eğitime girmez (lambdarank'in grup sınırı da 10.000).
- **Karar 3 (zengin sayı satırı, `model.room_row`):** sayı modeli oda tipi modelinin geometri özelliklerini, çevreyi, pencere uzunluğunu ve iç derinliği de görür. Yuvarlama eşiği doğrulamada seçildi: etanj 0,6; diğerleri 0,5 (anahtar 0,7 artık en iyi değil).
- **Karar 4 (açık mutfak, `rooms.OPEN_KITCHEN`):** "SALON+MUTFAK" / "Salon & Mutfak" ya da aynı odada hem salon hem mutfak yazısı → `living_kitchen`. Arşivde 11–15 m²'de açık mutfak ortalama 1,7, düz salon 0,7 tavan lambası alıyor; 15–19 m² açık mutfakların %83'ü 2–3 lamba. Kod sona eklenir (`ROOM_CODES`, `TYPE_NAMES` kod sırasında): eski modeller kodlarını korur.
- **Elenen:** tavan lambalarını odayı n eşit parçaya bölüp ortalarına koymak (ağırlıklı k-means): 1 m F1 düştü (0,665 → 0,647). Çok sınıflı sayı modeli (en olası sayı): Poisson'dan iyi değil. Daha da büyük sıralayıcı (1.500 tur, 255 yaprak): artı yok.
- **Sonuç (deneyde, eski veri setiyle, test projeleri):** tavan 0,665 → 0,684, priz 0,469 → 0,501, etanj 0,450 → 0,461, anahtar 0,568 → 0,587. Yeniden kurulan veri setiyle eğitilen modelin sonucu aşağıda.

## ADR-027 · Kapının arkasındaki oda; pano ve acil aydınlatma; mühendis talimatı kurala çevrilir; LLM yerleştirmesi denendi
- **Komşu odalar (`candidates._neighbours`):** her aday için en yakın kapının arkasındaki odanın tipi (kat holü / dış = −1, adsız oda = 0) ve ıslak hacim, mutfak, dış ve herhangi bir oda kapısına mesafe; oda için ıslak hacim kapısı, komşu ve dış kapı sayısı. Banyonun anahtarı holde, banyo kapısının yanına çizilir. Deneyde (eski veri seti, kapılar adaylardan yaklaşık çıkarılarak) doğrulama / test 1 m F1: tavan 0,697/0,677 → 0,705/0,681, priz 0,540/0,514 → 0,551/0,520, anahtar 0,636/0,590 → 0,658/0,595; etanj ikisinde de düştü (0,540/0,475 → 0,535/0,446), o yüzden etanj modelleri bu özellikleri görmez (`model.EXCLUDED`). Üründe komşuluk `rooms.doorways`'ten birebir gelir.
- **Yeni sınıflar:** `classes.py` deneyi, etiketler satırlardaki cihazlardan (0,4 m): daire panosu duvar adaylarında test 0,44 (sayı bilinseydi 0,68), acil aydınlatma tavan adaylarında 0,28 (0,45). İkisi yerleştirmeye taslak olarak girer (`PLACE_CLASSES`, ELK-PANO / ELK-ACIL); sayı eşiği 0,7 doğrulamada seçildi. Buat (0,18) linyeyle birlikte gelir (faz 3); sensör (doğrulamada 38 cihaz) ve kuvvet prizi (doğrulama ve testte yok) için veri yetmiyor.
- **Elenen teoriler:** kattaki toplam sayıyı odalara dağıtmak (Poisson kuyruğu), hiçbir sınıfta oda oda yuvarlamayı geçmedi. "Her daireye bir pano" (mutfak sayısı = pano sayısı) katların yalnız %37'sinde tutuyor.
- **EDL + LLM (`tmp/scripts/theory/edl.py`):** kod odayı metne çevirir: oda çevresinde sırayla adlandırılmış duvar noktaları W1.., tavan noktaları C1.., kapı / kilit tarafı / pencere / köşe etiketleri, boyuttan tahmin edilen mobilya. LLM yalnız bu adlarla cevap verir, yani her düzenleme geçerli bir noktaya düşer. 60 test odasında 1 m F1 (tavan / priz / etanj / anahtar):
  - gradient model: 0,72 / 0,58 / 0,79 / 0,72.
  - EDL'den sıfırdan yerleştirme: Haiku 0,48 / 0,36 / 0,27 / 0,41 ($0,09); Sonnet 0,64 / 0,38 / 0,27 / 0,46 ($1,03).
  - modelin taslağını düzeltme: Haiku 0,65 / 0,48 / 0,36 / 0,57; Sonnet 0,71 / 0,53 / 0,73 / 0,69 ($0,97).
  - taslak + arşivden en benzer 2 oda: Haiku 0,68 / 0,53 / 0,50 / 0,64; Sonnet 0,70 / 0,52 / 0,64 / 0,66 ($1,69).
  - **Sonuç:** hiçbir LLM kolu modeli geçmedi; arşiv örnekleri Haiku'yu toparladı ama yetmedi. LLM'in işi isabet değil, mühendisin isteğini uygulamak.
- **Talimat (`steer.py`):** aynı 60 odada iki talimat denendi.
  - "Salonlara en az 4 priz": Sonnet'in taslağı düzeltmesiyle 14/14 salon uydu ama priz isabeti 0,50'ye düştü ($0,97). Haiku'nun kurala çevirip kodun uygulamasıyla priz 0,57 kaldı ($0,003).
  - "Anahtarlar kilit tarafına": Sonnet ile uyum %76, anahtar isabeti 0,61. Kuralla uyum %69 (kilit etiketli noktası olmayan oda kural dışı), anahtar isabeti 0,66.
  - **Karar:** talimat Haiku ile kurala çevrilir (sayı alt / üst sınırı, etiketli noktaları öne alma; şema dışı kural atılır), kural modelin üstüne uygulanır, kural ve talimat `devices.json`'a yazılır. Haiku ilk denemede "salon"u yalnız `living` saydı; istemde açık mutfaklı salon da yazılı.
- **Bulunup düzeltilen:** paralel iş havuzunda bellek yetmeyen iş kayboluyordu (veri setinde 247 çiftten 4'ü). Artık sona bırakılıp tek başına bir kez daha denenir (`process._run_pool`).

## ADR-028 — Talimat kuralları onaylanır; kalıcı kurallar ofis kural defterinde (2026-10-14)

**Bağlam.** ADR-027'de Haiku talimatı kurala çeviriyor ve kural doğrudan uygulanıyordu. Cevriye denemesinde
"Salonlara en az 4 priz koy, anahtarları kapının kilit tarafına koy" talimatında Haiku "salonlara" kapsamını
ikinci cümleye de taşıdı; anahtar kuralı yalnızca salonlara uygulandı ve bu ancak iş kaydında görülebiliyordu.

**Karar.**
- Çizim sayfasında "Kurala çevir" (`POST /api/steer`) kuralları düzenlenebilir bir tablo olarak gösterir (cihaz,
  oda tipleri, en az, en çok, yakınına). Yerleştir tam olarak tablodaki listeyi gönderir (`rules`); talimat
  değişip çevrilmediyse önce çevirir ve onay ister.
- "Kalıcı" işaretli kurallar `DATA/placement_rules.json` ofis kural defterine yazılır (`PUT /api/rulebook`) ve her
  projede listeye baştan gelir. API `rules` göndermezse iş, defter + talimat çevirisini uygular (eski davranış).
- Kurallar her girişte `steer.check` ile şemaya göre süzülür (bilinmeyen sınıf/oda tipi/etiket, negatif sayı atılır).
- İstem: "her cümle ayrı okunur, oda tipi başka cümleye taşınmaz". Aynı talimatta Haiku artık anahtar kuralını
  bütün odalara veriyor ($0,004).

## ADR-029 — Düzelt sayfası: mühendisin cihaz ve temizlik düzeltmeleri, gerekçeleriyle (2026-10-14)

**Bağlam.** Arşiv yalnızca mühendisin son çizimini veriyor; modelin önerisini neden beğenmediğini ya da neyi kabul
edilebilir bulduğunu söylemiyor. 1 m F1 "kabul edilebilir ama farklı" yerleşimi hata sayıyor. Kural defteri,
denetleyici ve LLM araçları için mühendisin gerekçeleri gerekiyor.

**Karar.**
- `/edit` sayfası, sekmeler: **Cihazlar**, **Temizlik**, **Gerekçeler**. Çizim, sunucunun görünen kutu için çizdiği
  resim (`/api/outputs/{name}/view/{place|clean}.png`, 150 bin nesnelik sahne ~0,2 s), düzenlenen şeyler SVG.
- Cihazlar (`cadai/correct/devices.py`): sürükle-taşı, ekle, sil, tip değiştir, "oda tamam". Duvar cihazları 0,6 m
  içindeki en yakın kalan çizgiye oturur ve tıklanan tarafa bakar. Durum saklanmaz, türetilir: eklendi / silindi /
  değişti / kabul (dokunulmamış ve oda tamam) / bakılmadı. "Bakılmadı" kanıt sayılmaz; bu yüzden "oda tamam" şart.
- Temizlik (`cadai/correct/cleanup.py`): tıklama üst nesneyi (blok bütün), Alt+tıklama tek parçayı, Shift+sürükleme
  kutuyu seçer. Panel sistemin gerekçesini söyler: hangi araç sildi (`decision.npz` `why`), kat planı dışı, ya da
  eski çıktılarda katman tipi. Mühendis "kalmalı / silinmeli" der.
- Her işlem `edits.jsonl`'e eklenir: kim (şifre ekranındaki ad), ne zaman, önce / sonra, gerekçe düğmeleri, açıklama.
  Gerekçe işlemden sonra da verilebilir (`reason` satırı `seq` ile bağlanır). Ctrl+Z son işlemi geri alır.
- "Düzeltilmiş DWG": `placed_corrected.dwg` ve kaynaktan yeniden yazılan `clean_corrected.dwg`, işler olarak.
  ADR-029'dan önceki temizliklerde tarama "yalnız sınır" kararı kayıtlı değil; orada kalan taramalar tam yazılır.
- Temizlik işi artık `decision.npz` içine `applied` ve `why` da yazar.

**Kullanım.** Düzeltmeler eğitime doğrudan girmez. Önce ölçü (kabul oranı, F1'in ne kadar katı olduğu) ve kural /
denetleyici / LLM aracı tasarımı için okunur; eğitime girişi ayrıca karar verilecek.

## ADR-030 — Odalar: duvarlar + kapı kapama çizgileri, ad ve alan yazısından tohumlar (2026-10-14)

**Bağlam.** ADR-008'deki oda okuma, kapıları kapatmak için duvarları her yandan 0,55 m büyütüyordu. Arşivde
mimarın alan yazılarıyla ölçünce (yazının düştüğü odanın alanı yazılanın %10'u içinde mi): kendi temizliğimizle
test çiftlerinde %48. Kayıplar: 1,1 m'den dar odalar (WC, duş, şaft) tamamen yok oluyor; 1,1 m'den geniş kapı
açıklıkları kapanmıyor, odalar birbirine ya da dışarıya akıyor; damga başına "hiçbir odada değil" en büyük kalem.

**Karar** (mühendisin önerisi: "3 kapalı duvar + kapıyla kapanan duvar").
- Duvarlar büyütülmez; yalnız birkaç cm'lik çizim boşlukları kapanır.
- Ham çizimdeki her kapı yayı, menteşeden kapalı kanat ucuna bir çizgiyle kapıyı kapatır. Kapalı uç: ucun
  ötesinde ve menteşenin arkasında 15 cm içinde duvar olan uç (açık kanadın ucu çoğunlukla odada boşta durur).
  Çizgi iki ucundan 10 cm pervaza uzatılır. Kapı çizgisinin pikselleri iki yanındaki odalara dağıtılır (oda
  komşulukları, ADR-027, bozulmaz).
- Oda adları ve alan yazıları tohumdur. Birden çok tohumlu parça, her tohum ayrı çekirdeğe düşene kadar adım adım
  daraltılır, çekirdekler parçanın içinde geri büyür: odalar en dar açıklıkta ayrılır. Tohumsuz parçalar eski
  usulle (0,55 m çekirdekler) bölünür.
- Kenara değen parça dışarıdır; hiç 40 cm'den geniş olmayan parça duvar içidir.
- Tek bir odaya ince bir çizgiyle bitişik, kapısız, adsız, 3 m²'den küçük kapalı şekiller (dolap, tezgah, duş
  teknesi) odanın zeminidir.
- Alan duvar yüzüne kadar ölçülür (çevredeki çizgi piksellerinin yarısı).
- Her oda içindeki alan yazısını taşır (`stamp_m2`). Kontrol, mimarın proje genelindeki alışkanlığını (bizim /
  yazılan medyanı; bazı mimarlar duvar payıyla yazıyor) bölerek %10'dan fazla sapan odaları, iki yazılı (birleşmiş)
  odaları ve alışkanlığın kendisini bildirir. Düzelt sayfası odada okunanı ve yazılanı gösterir.

**Ölçüm** (kendi temizliğimiz, bina içindeki yazılar; ayarlı = mimarın alışkanlığı bölünmüş):

| | eski | yeni |
|---|---|---|
| test çiftleri (200 içinden yazılı olanlar, 1.745 yazı) | %48 (ayarlı %52) | **%73 (ayarlı %76)** |
| eğitim 40 çift | %48 (%50) | %69 (%74) |
| 7 temizlik çıktısı | %46 (%51) | %75 (%76) |

Kapı çizgileri olmadan %31, tohumlar olmadan %57 (eğitim 40, ayarlı). Kalan hatalar: otopark katları (park
yerleri oda gibi kapalı), iç duvarları temizlikte silinmiş planlar, kapı yayı çizilmemiş geniş açıklıklar.

**Sonuç.** Yerleştirme verisi ve modeli yeni odalarla yeniden kuruldu (placement-20261014-1939). Veri:
10.710 → 15.014 oda, odasına düşen cihaz %77,9 → %81,3. Test projelerinde 1 m F1, payda projedeki tüm cihazlar
(tavan, priz, etanj, anahtar): eski okuyucu + 2320 0,527; yeni okuyucu + 2320 0,554; yeni okuyucu + 1939 0,596
(kesinlik 0,649 → 0,698, duyarlılık 0,444 → 0,520). Uçtan uca (6 sınıf, mühendise göre 1 m F1): Cevriye 0,625 → 0,656,
Ali Kemal 0,627 → 0,657, üçüncü proje 0,378 → 0,546. Site en yeni modeli kullanır.

## ADR-031 · Arayüzün İngilizce görünümü: tarayıcıda çeviri katmanı (2026-10-21)

**Bağlam.** Site bir işverene İngilizce sunulacak. Arayüz metinleri Türkçe ve sayfaların koduna, sunucunun
etiketlerine (araç, katman, cihaz adları, gerekçe çipleri, kontrol mesajları, iş başlıkları) dağılmış.

**Karar.** Sayfalar ve sunucu Türkçe yazmaya devam eder. `static/i18n.js` her sayfanın `<head>`'inde yüklenir;
başlıktaki EN/TR düğmesi (ya da `?lang=en` bağlantısı) seçimi tarayıcıda saklar. İngilizce seçiliyken bir
MutationObserver her metin düğümünü, `placeholder`/`title`'ı göründüğü anda çevirir: önce tam metin, sonra sayılı
kalıplar (`%12` → `12%`, `3 oda` → `3 rooms`), sonra uzun metinlerin içindeki bilinen ifadeler (harf sınırlı, en
uzun önce). Serbest metin çevrilmez: mimarın oda adları, Claude'un bölge ve kontrol notları, LLM önerilerinin
gerekçeleri, talimat (`translate="no"`) ve iş günlükleri (`.log`).

**Neden.** Sayfa kodunun her metnini `t()` ile sarmak 1.400 satırı değiştirirdi ve Türkçe arayüzü bozma riski
taşırdı; katman Türkçe görünümü hiç değiştirmez. Yeni bir Türkçe metin eklenirse İngilizcede Türkçe görünür
(sözlüğe eklenene kadar); kontrol: sayfaları `?lang=en` ile headless tarayıcıda açıp kalan Türkçe kelimeleri listelemek.

## ADR-032 · Depoda anonimleştirilmiş iki demo proje (2026-10-21)

**Bağlam.** Çalışma bir işverene gösterilecek; ofis iki projenin paylaşılmasına izin verdi. Kural: isimler
kalabilir, başka kişisel bilgi (adres, sokak, mahalle, ada / parsel / pafta, telefon, e-posta, web, sicil,
kimlik, vergi numarası) çıkmaz. ADR'den önce depoya hiçbir çizim girmiyordu (CLAUDE.md kural 2).

**Karar.** `cadai/demo/build.py` iki test projesi (Cevriye Berkoğlu, Ali Kemal Arslan; eğitimde görülmemiş)
için `demo/` klasörünü kurar: ham, temiz ve yerleşim DWG'si, sayfanın ve her kat planının resmi, sayılar ve
İngilizce README. Her DWG `cadai/demo/anonymise.py`'den geçer: kat planlarının (yerleştirme bölgeleri, 1 m pay)
dışındaki bütün yazılar silinir (etiket ile değeri ayrı yazılar olduğundan yalnız etiketi aramak değeri bırakırdı),
yalnız plan dışında kullanılan blokların (antet) yazıları silinir, plan içinde kişisel görünen yazılar, gömülü
nesneler (OLE, resim, underlay) her yerde silinir. Sonra `leaks()` yazılan DXF'teki bütün metin değerlerini
(genişletilmiş veri dahil) tarar; bir şey bulursa DWG yazılmaz. Asıl ham çizimlerde tarama adres, telefon,
e-posta, web, ada/parsel ve sicil etiketlerini buluyor, anonim dosyalarda hiçbir şey bulmuyor; DWG'ler ODA ile
DXF'e geri çevrilip yeniden tarandı. Resimler yalnız çizgi geometrisinden çizilir (yazı basılmaz). `.gitignore`
yalnız `demo/` altındaki DWG ve PNG'ye izin verir. Sitenin Demo sayfası (`/demo`) aynı klasörü gösterir.

**Bilinen.** ODA File Converter DWG'ye "son kaydeden" olarak oturum açmış Windows kullanıcısını yazar (her DWG
sürümünde; ortam değişkeni ya da $LASTSAVEDBY bunu değiştirmiyor). Bu yazarın kendi kullanıcı adıdır (isim;
depodaki yollarda da geçiyor); bilgisayar adının dosyada olmadığı yapımda denetlenir.

## ADR-033 · Her metin iki dilde, yazıldığı yerde (ADR-031'in yerine, 2026-10-22)

**Bağlam.** ADR-031'in tarayıcıdaki çeviri katmanı Türkçe metinleri sözlük ve kalıplarla tahmin ediyordu; sunucu
mesajları, iş günlükleri, Claude'un notları ve kayıtlı metinler Türkçe kalıyordu. Kullanıcı tam bir İngilizce sürüm
istedi; Türkçe ofis için kalacak, EN/TR düğmesi seçecek; mimarın çizimindeki yazılar olduğu gibi kalacak.

**Karar.** Her metin iki dilde, yazıldığı yerde yazılır.
- Sunucu: `cadai/i18n.py` — `say("Kural yok", "No rules")` bir `Msg`; tarayıcıya {"tr", "en"} olarak gider, sayfa
  seçili dili gösterir. Etiket sözlükleri (araç, katman, cihaz, mobilya tipleri, gerekçe çipleri) `Msg` tutar.
  İş günlüğü satırları ve kontrol uyarıları iki dili birlikte saklar; dil sonradan değişse de doğru gösterilir.
  İstek dili tarayıcının çerezinden (`cadai-lang`) gelir (hata metinleri, indirilen dosya adları); bir iş onu
  başlatanın diliyle çalışır (hata izleri, sunucu günlüğü). Gerekçe çipleri günlüğe Türkçe metinleriyle yazılır
  (eski kayıtlarla aynı anahtar), seçili dilde gösterilir.
- Claude: istemler Türkçe kalır (ayarlanmış ve doğrulanmış); cevap şemasına İngilizce alanlar eklendi (`ne_en`,
  `notes_en`), notlar iki dilde saklanır.
- Sayfalar: `T("Yerleştir", "Place")` kodda, `data-en` / `data-en-html` sabit HTML'de; `L(x)` sunucudan gelen
  iki dilli metni seçer. Tarayıcının kendi dosya seçme düğmesi yerine kendi düğmemiz (Windows dilini izlemesin diye).
- Eski kayıtlar: `python -m cadai.translate_stored` ADR'den önce yalnız Türkçe saklanmış metinleri (Claude notları,
  kontrol uyarıları, öneri gerekçeleri, hizalama günlükleri, iş başlıkları) bir kez çevirir: 2142'si bilinen kalıplardan
  birebir, 80 serbest metin Claude Haiku ile ($0,01). Sonuç `DATA/translations.json`; `i18n.bi()` okur. Eski işlerin
  günlük satırları çevrilmedi (yalnız Debug'da görünür); eski işlerin başlıkları bugünkü iş başlığıyla gösterilir.
- Test: `tests/test_i18n.py` — sayfalarda T() dışında Türkçe kalmadığını, çerezin sunucu metinlerini, iş günlüğünün
  iki dilini, indirme adlarını ve çeviri belleğini denetler. Headless tarayıcıda bütün sayfalar İngilizce açılıp
  taranınca yalnız isimler ve çizim verisi (katman, blok, proje adları) Türkçe kalıyor.

