# LLM'ler bu sistemin neresinde

İlke: **LLM'ler koordinat hesabında kötü, insanların yazdığı dağınık şeyleri yorumlamakta iyi.** Hesabı her zaman kod ve modeller yapar; LLM dağınık isimleri anlamlı hale getirir ve istisnalara bakar.

| Aşama | LLM'in işi | Ne kadar |
| --- | --- | --- |
| Veri hazırlama (bir kez) | Katman adları (`A-DUV-01`), oda adları (`ISL.HAC.`), mobilya blokları (`KARYOLA_160` → yatak) ve elektrik sembollerini sözlüğe çevirmek | En çok burada. Her benzersiz isim bir kez sorulur ve önbelleğe yazılır; 300 projede birkaç bin isim, toplam birkaç on dolar |
| Eğitim | Yok; LightGBM | 0 |
| Temizlik (yeni proje) | Modelin "emin değilim" dediği küçük grupları resme bakarak karara bağlamak; başlığı olmayan paftada kat planını bulmak | Az. Mühendise yalnızca LLM'in de emin olmadığı kalır |
| Yerleştirme | Varsayılan olarak yok. Mühendisin serbest talimatını ("mutfağa 4 tezgah prizi") düzenlemeye çevirmek | Neredeyse 0 |
| Kontrol | Kuralların bulduğu hatayı düzeltmek; düzeltemediğini anlaşılır Türkçe soruya çevirmek | Az |
| Rapor | Proje sonunda kısa Türkçe özet | Çok az |

Hedef: kararların ~%95'i LLM'siz, proje başına maliyet ~$0–1 (eski ajan tabanlı temizlik ~$15).

**Durum (2026-10-10):** henüz hiçbir LLM çağrısı yazılmadı. İlk LLM işi isim sözlüğü olacak: özellik çıkarıcıdaki anahtar kelime listelerinin (`cleanup/features.py` `NAME_KEYWORDS`) yerini, benzersiz katman ve blok adlarının LLM ile bir kez sınıflandırıldığı önbellekli bir sözlük alacak.
