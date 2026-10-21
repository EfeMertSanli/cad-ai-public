# cad-ai

**In English.** cad-ai learns, from an electrical engineering office's archive of ~250 past projects, the path
from an architect's raw DWG to the electrical project: it cleans the drawing down to a base plan (rule tools,
learned models and Claude checking the result from a picture), reads the rooms from walls, door swings and the
architect's labels, and places lights, sockets, switches and panels room by room with gradient-boosted models.
A web app runs it, lets an engineer correct the result with reasons, and runs fully in English or Turkish (EN/TR switch).
On unseen archive projects, 70% of the devices it places are within 1 m of one the engineer drew and it finds
52% of the engineer's devices (F1 0.60; engineers agree with each other at 0.80–0.87). **See [`demo/`](demo/README.md)** for two anonymised projects with DWG files, pictures
and the comparison with the engineer. Design decisions with measurements: `docs/decisions.md` (Turkish).

---

Mimarın ham DWG çiziminden elektrik projesine giden yolu **veriden öğrenen** sistem. Ofisin arşivindeki projelerden (mimarın ham çizimi + mühendisin bitmiş elektrik çizimi) üç model eğitilir:

1. **Temizlik (M1):** ham çizimdeki her nesne kalsın mı, silinsin mi. *Şu an yapılan.*
2. **Yerleştirme (M2):** temiz altlığa, ham çizimdeki mobilyaya göre priz, anahtar ve lamba.
3. **Kontrol (M3):** çıktıdaki hataları mühendisten önce bulmak.

Hesabı kod ve modeller yapar; LLM'ler yalnızca isim sözlüğü, istisnalar ve mühendisle konuşma için kullanılır (`docs/llm.md`).

## Çalıştırma
```
setup.bat      # bir kez: D:\cad-ai-data\venv oluşturur, paketleri kurar
start.bat      # siteyi açar: http://127.0.0.1:8800
```
Sitede ana sayfadaki düğmeler sırasıyla: **Envanteri tara → Çizimleri dönüştür → Etiketleri çıkar → Temizlik modelini eğit.** Her iş arka planda çalışır; ilerleme, günlük ve hatalar **Debug** sayfasında canlı görünür.

## Nerede ne var
| Yer | İçerik |
| --- | --- |
| `C:\Users\aztek\Desktop\cad-ai` | Kod ve belgeler (bu klasör) |
| `C:\Users\aztek\Desktop\proje\Proje` | Ofis arşivi: proje başına bir klasör. **Salt okunur**, hiçbir şey yazılmaz |
| `D:\cad-ai-data` | Büyük veri: sanal ortam, DXF, sahneler, etiketler, özellikler, modeller, günlükler. C: diskinde yer olmadığı için D:'de |

Yollar ortam değişkenleriyle değiştirilebilir: `CADAI_DATA`, `CADAI_PROJECTS`, `CADAI_PORT`, `ODA_PATH`.

## Kod düzeni
```
cadai/
  config.py          yollar
  log.py, jobs.py    günlük ve arka plan işleri (debug paneli bunları gösterir)
  sysinfo.py         bellek bilgisi
  cad/               oda.py (DWG→DXF), units.py (birim tahmini), scene.py (patlatılmış geometri), render.py (PNG)
  corpus/            inventory.py (arşiv taraması, ham/son eşleme), process.py (toplu dönüştürme ve etiketleme)
  cleanup/           align.py (ham↔son hizalama → etiket), features.py, train.py (LightGBM)
  web/               app.py (FastAPI) ve static/ (index, project, debug)
docs/                mimari, kararlar, planlar
tests/               pytest
```

## Belgeler
- `docs/architecture.md`: veri akışı ve modüller
- `docs/decisions.md`: kararlar (ADR)
- `docs/plan-cleanup.md`, `docs/plan-placement.md`, `docs/plan-check.md`: üç modelin planları
- `docs/llm.md`: LLM'lerin hangi aşamada ne kadar kullanıldığı
