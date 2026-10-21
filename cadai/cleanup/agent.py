"""Claude runs the cleanup step by step with the toolbox (ADR-021). Two calls per drawing, through
headless Claude Code (`claude -p`, the subscription; no API key):

1. views : the sheet's regions on one numbered picture plus their titles; Claude picks the floor
           plans with the view rulebook it wrote from the archive (DATA/agent/view_rulebook.md).
2. review: the tools have run inside the chosen plans; Claude sees the cleaned plans, a picture of
           what every tool removed (one colour per tool), the tool counts with their measured
           precision and the layer table, and answers corrections: tools to switch off, layer types
           to change, regions to add or drop. The tools run once more with them. No item-level
           judgement by Claude: the tools find, Claude decides per tool and per layer.
Every call's answer, reason and cost are written to the output folder (agent.json).
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from cadai import config
from cadai.cad.render import render
from cadai.cad.scene import Scene
from cadai.cleanup import layers, regions, tools
from cadai.i18n import say

CLAUDE = r"C:\Users\aztek\.local\bin\claude.exe"
MODEL = "sonnet"
VIEW_RULEBOOK = config.DATA / "agent" / "view_rulebook.md"
# share of what each tool removes that the engineers removed too: archive validation + test projects,
# 71 drawings (ADR-021). "dashed" is not measured: the archive scenes predate the linetype field.
TOOL_PRECISION = {"off_layers": 0.89, "annotations": 0.95, "furniture": 0.84, "door_swings": 0.78,
                  "hatching": 0.93, "model": 0.89, "ticks": 0.62}

TASK = """Bir elektrik mühendisliği ofisi, mimarın DWG çizimini elektrik projesine altlık yapıyor. Mühendis mimarın
çizimine dokunmaz; işine yaramayan şeyleri siler. Altlıkta yalnızca kat planlarının yapısı kalır: duvar, kolon/perde,
pencere, kapı boşluğu ve kasası, merdiven, korkuluk, şaft, döşeme/balkon kenarı ve oda adları. Gidenler: kesit, görünüş,
vaziyet, çatı planı, detay ve tablolar; planların içinde mobilya, vitrifiye, tarama, ölçü, aks, kot, yazılar,
kapı kanadı ve yayı, kesikli çizgiler."""


def call(prompt: str, workdir: Path, tag: str) -> tuple[str, dict]:
    """One headless Claude Code call; the prompt goes through a file (Windows command lines are short)."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / f"{tag}.md").write_text(prompt, encoding="utf-8")
    t0 = time.time()
    proc = subprocess.run([CLAUDE, "-p", f"{tag}.md dosyasını Read aracıyla oku ve oradaki görevi yap. "
                           "Cevabın yalnızca dosyanın istediği JSON olsun.", "--model", MODEL,
                           "--output-format", "json", "--allowedTools", "Read"],
                          cwd=workdir, capture_output=True, text=True, encoding="utf-8", timeout=900,
                          stdin=subprocess.DEVNULL)
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        out = {"result": "", "is_error": True, "error": (proc.stderr or proc.stdout)[-500:]}
    rec = {"step": tag, "cost_usd": out.get("total_cost_usd"), "seconds": round(time.time() - t0, 1),
           "is_error": bool(out.get("is_error")), "answer": (out.get("result") or "")[-3000:]}
    return out.get("result") or "", rec


def _json(text: str, opener: str):
    m = re.search(r"\[.*\]" if opener == "[" else r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _what(ans: list, n: int) -> dict:
    """Claude's short description of region n in both languages (ADR-033)."""
    for d in ans:
        if int(d.get("bölge", d.get("bolge", -1))) == n:
            return {"tr": str(d.get("ne", "")), "en": str(d.get("ne_en") or d.get("ne", ""))}
    return {"tr": "", "en": ""}


def choose_views(scene: Scene, item_region: np.ndarray, regs: list[dict], workdir: Path) -> tuple[set[int] | None, dict]:
    """Region ids Claude calls floor plans to keep, or None when the call failed."""
    regions.overview(scene, item_region, regs, workdir / "views_question.png")
    rulebook = VIEW_RULEBOOK.read_text(encoding="utf-8") if VIEW_RULEBOOK.exists() else ""
    lines = [f"- bölge {n}: {r['box'][2] - r['box'][0]:.0f} x {r['box'][3] - r['box'][1]:.0f} m, {r['m']} m çizgi; "
             f"yazılar: {'; '.join(r['titles']) or '-'}; istatistik modeli p(plan)={r['p_view']:.2f}"
             for n, r in enumerate(regs)]
    prompt = f"""{TASK}

ADIM 1: Paftanın genel görünüşü views_question.png dosyasında (Read ile aç). Kırmızı kutular ve numaralar ayrı
çizim bölgeleri. Hangi bölgelerin altlıkta KALACAĞINA (kat planı) karar ver. Arşivdeki mühendislerin gerçek
kararlarından çıkarılmış kural kitabı:

{rulebook}

Bölgeler:
{chr(10).join(lines)}

Cevap olarak YALNIZCA JSON dizisi ver: [{{"bölge": <n>, "plan": true | false, "ne": "<kısa>", "ne_en": "<aynısı İngilizce>"}}, ...]"""
    text, rec = call(prompt, workdir, "views")
    ans = _json(text, "[")
    if not ans:
        return None, rec
    chosen = set()
    for d in ans:
        n = int(d.get("bölge", d.get("bolge", -1)))
        if 0 <= n < len(regs) and d.get("plan"):
            chosen.add(regs[n]["id"])
    rec["regions"] = [{"n": n, "id": r["id"], "titles": r["titles"][:3], "plan": r["id"] in chosen,
                       "what": _what(ans, n)}
                      for n, r in enumerate(regs)]
    return chosen, rec


def pictures(scene: Scene, result: tools.Result, in_plan: np.ndarray, folder: Path) -> tuple[Path, Path] | None:
    """tools.png: everything in the plans, one colour per tool for what it removed, dark for kept.
    kept.png: only what stays."""
    if not in_plan.any():
        return None
    bb = scene.bboxes()[in_plan]
    bbox = (np.nanpercentile(bb[:, 0], 0.5) - 2, np.nanpercentile(bb[:, 1], 0.5) - 2,
            np.nanpercentile(bb[:, 2], 99.5) + 2, np.nanpercentile(bb[:, 3], 99.5) + 2)
    code = np.where(result.attribution >= 0, result.attribution, np.where(result.decision != tools.DELETE, 99, -9))
    colours = {i: tools.COLOURS[t] for i, t in enumerate(tools.TOOLS)}
    colours[99] = (30, 30, 30)
    a = render(scene, folder / "tools.png", code, bbox=bbox, width=2400, colours=colours)
    b = render(scene, folder / "kept.png", np.where(result.decision != tools.DELETE, 99, -9), bbox=bbox, width=2400,
               colours={99: (30, 30, 30)})
    return a, b


def review(scene: Scene, result: tools.Result, kinds: list[str], in_plan: np.ndarray, regs: list[dict],
           chosen: set[int], workdir: Path) -> tuple[dict | None, dict]:
    """Claude's corrections: {"disable": [tool], "layer_types": {layer: type}, "add_regions": [n], "drop_regions": [n]}."""
    length = scene.lengths()
    legend = "\n".join(f"- {t}: {tools.LABELS[t].tr} (renk RGB{tools.COLOURS[t]}); {result.stats['tools'][t]['items']} nesne, "
                       f"{result.stats['tools'][t]['m']} m"
                       + (f"; arşivde isabeti %{100 * TOOL_PRECISION[t]:.0f}" if t in TOOL_PRECISION else "")
                       for t in tools.TOOLS)
    n_layers = len(scene.meta["layers"])
    plan_m = np.bincount(scene.layer, weights=np.where(in_plan, length, 0), minlength=n_layers)
    kept_m = np.bincount(scene.layer, weights=np.where(result.decision != tools.DELETE, length, 0), minlength=n_layers)
    rows = [f"{scene.meta['layers'][i]} | tip {kinds[i]} | planda {plan_m[i]:.0f} m | kalan {kept_m[i]:.0f} m"
            for i in np.argsort(-plan_m)[:60] if plan_m[i] >= 1]
    region_lines = [f"- bölge {n}: {'KALDI' if r['id'] in chosen else 'silindi'}; yazılar: {'; '.join(r['titles'][:3]) or '-'}"
                    for n, r in enumerate(regs)]
    prompt = f"""{TASK}

ADIM 2 (kontrol): Kat planları seçildi ve sabit araçlar çalıştı. Resimler:
- kept.png: altlıkta KALANLAR (son hal).
- tools.png: planların içinde her aracın sildiği şeyler kendi renginde, koyu = kalanlar.
- views_question.png: paftanın bölgeleri (numaralar aşağıdaki listeyle aynı).
Resimleri Read ile aç ve bir mühendisin gözüyle kontrol et:
a) kept.png'de altlıkta olmaması gereken bir şey kaldı mı (mobilya, tarama, kesit/görünüş parçası, yazı)? Hangi
   katmanda? Varsa o katmanın tipini "drop" yap ya da aracı açık bırak.
b) Bir araç yapıyı mı silmiş (tools.png'de duvar, pencere, merdiven bir aracın renginde)? Varsa o aracı kapat ya da
   ilgili katmanın tipini düzelt (wall, column, window, door, opening, stair, railing, shaft).
c) Yanlış seçilmiş ya da atlanmış bir bölge var mı?
Emin olmadığın şeyi değiştirme; her şey yolundaysa boş düzeltme ver.

Araçlar:
{legend}

Katmanlar (plandaki uzunluğa göre):
{chr(10).join(rows)}

Bölgeler:
{chr(10).join(region_lines)}

Cevap olarak YALNIZCA şu JSON nesnesini ver:
{{"disable": [<araç adı>], "layer_types": {{"<katman adı>": "<tip>"}}, "add_regions": [<n>], "drop_regions": [<n>],
  "notes": "<en fazla 3 cümle: ne gördün, neyi neden değiştirdin>", "notes_en": "<notes'un aynısı, İngilizce>"}}"""
    text, rec = call(prompt, workdir, "review")
    ans = _json(text, "{")
    if not isinstance(ans, dict):
        return None, rec
    valid = set(layers.LABELS) - {"unknown"}
    fixes = {"disable": [t for t in ans.get("disable", []) if t in tools.TOOLS],
             "layer_types": {k: v for k, v in (ans.get("layer_types") or {}).items()
                             if k in scene.meta["layers"] and v in valid},
             "add_regions": [regs[n]["id"] for n in ans.get("add_regions", []) if isinstance(n, int) and 0 <= n < len(regs)],
             "drop_regions": [regs[n]["id"] for n in ans.get("drop_regions", []) if isinstance(n, int) and 0 <= n < len(regs)],
             "notes": {"tr": str(ans.get("notes", ""))[:600], "en": str(ans.get("notes_en") or ans.get("notes", ""))[:600]}}
    rec["fixes"] = fixes
    return fixes, rec


def clean(scene: Scene, p_view: np.ndarray, p_delete: np.ndarray, X: np.ndarray, names: list[str],
          workdir: Path, use_claude: bool = True, log=lambda m: None) -> tuple[tools.Result, np.ndarray, dict]:
    """The whole procedure. Returns the toolbox result, the plan mask and a record of every step."""
    workdir.mkdir(parents=True, exist_ok=True)
    record: dict = {"calls": []}
    kinds, _ = layers.classify(scene)
    item_region, regs = regions.find(scene, p_view, X, names)
    record["regions"] = len(regs)
    chosen = None
    if use_claude and regs:
        chosen, rec = choose_views(scene, item_region, regs, workdir)
        record["calls"].append(rec)
        got = len(chosen) if chosen is not None else None
        log(say(f"Claude görünüş seçimi: {got if got is not None else 'cevap yok'} / {len(regs)} bölge, {rec['seconds']} s",
                f"Claude's view choice: {got if got is not None else 'no answer'} / {len(regs)} regions, {rec['seconds']} s"))
    if chosen is None:                         # no Claude (or no answer): the statistical view model
        in_plan = p_view >= 0.5
        chosen = {r["id"] for r in regs if r["p_view"] >= 0.5}
        record["views"] = say("istatistik model", "statistical model").both()
    else:
        in_plan = np.isin(item_region, list(chosen)) & (p_view >= regions.CANDIDATE_P)
        record["views"] = "claude"
    result = tools.run(scene, in_plan, p_delete, kinds)
    log(say(f"araçlar: {result.stats['tools']}", f"tools: {result.stats['tools']}"))
    if use_claude and record["views"] == "claude":
        pictures(scene, result, in_plan, workdir)
        fixes, rec = review(scene, result, kinds, in_plan, regs, chosen, workdir)
        record["calls"].append(rec)
        if fixes and (fixes["disable"] or fixes["layer_types"] or fixes["add_regions"] or fixes["drop_regions"]):
            chosen = (chosen | set(fixes["add_regions"])) - set(fixes["drop_regions"])
            in_plan = np.isin(item_region, list(chosen)) & (p_view >= regions.CANDIDATE_P)
            result = tools.run(scene, in_plan, p_delete, kinds, disabled=set(fixes["disable"]),
                               layer_types=fixes["layer_types"])
            log(say(f"Claude düzeltmeleri uygulandı: {fixes}", f"Claude's fixes applied: {fixes}"))
        record["fixes"] = fixes
    regions.overview(scene, item_region, regs, workdir / "views.png", chosen=chosen)
    pictures(scene, result, in_plan, workdir)
    record["chosen_regions"] = sorted(int(c) for c in chosen)
    record["region_list"] = [{"n": n, "id": r["id"], "titles": r["titles"][:3], "kept": r["id"] in chosen}
                             for n, r in enumerate(regs)]
    record["stats"] = result.stats
    record["cost_usd"] = round(sum(c.get("cost_usd") or 0 for c in record["calls"]), 3)
    return result, in_plan, record
