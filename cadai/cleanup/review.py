"""LLM review of the cleanup's doubtful object groups (trial, ADR-015).

The model decides every item in seconds; an LLM looks only at the few dozen groups the model was
unsure about, as pictures, the way a drafter would:

1. Object groups: items on the same layer whose segment ends touch (3 cm) form one object (a
   door leaf + its swing, a fixture). Without the same-layer rule walls glued whole sheets into one
   "group" (trial 1: one group = a whole floor plan of 2303 parts). Groups wider than 8 m are
   structure and are not reviewed (the guards handle walls).
2. Doubtful groups inside the floor plans: mean p(delete) between 0.2 and 0.8, or the model's
   best guess splits the group. Longest first, at most `max_groups`.
3. One picture per group: the cleaned plan around it (dark = kept, pale = deleted) with the group
   in red and its number.
4. Batches of pictures go to Claude (headless Claude Code, the subscription; `claude -p`), which
   answers keep / delete per group with a short reason, as JSON. Its answer replaces the model's
   decision for the whole group.
Every call's cost (the API price Claude Code reports) is recorded.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from cadai.cad.render import render
from cadai.cad.scene import Scene
from cadai.i18n import say

CLAUDE = r"C:\Users\aztek\.local\bin\claude.exe"
TOUCH_M = 0.03
BATCH = 15

STANDARD = """Temizlik standardı (çizimci tarzı sert temizlik, elektrik projesine altlık):
YALNIZCA KAT PLANLARI kalır. Kesit, görünüş (cephe), çatı planı, vaziyet planı, detay ve tablo paftalarındaki
HER ŞEY silinir: kırmızı grup böyle bir paftadaysa (yatay katmanlar, cephe pencereleri, çatı eğimi görüyorsan) "delete".
Kat planının içinde:
KALIR: duvar (iki yüzü), sıva, kolon, perde beton, pencere ve doğrama, kapı kasası, merdiven ve basamaklar,
korkuluk, döşeme / teras / balkon kenarı, rampa sınırı, baca ve şaft, binanın dış çeperi.
SİLİNİR: ölçü ve ölçü çizgileri, aks çizgileri ve aks balonları, yazılar ve etiketler, kot işaretleri,
tarama ve fayans deseni, mobilya, vitrifiye (klozet, lavabo, duş, küvet, evye), mutfak tezgahı, beyaz eşya,
araba, kapı kanadı ve açılış yayı, kapı kolu, ağaç ve peyzaj, yön okları, kesit işaretleri."""


def object_groups(scene: Scene, items: np.ndarray) -> np.ndarray:
    """Group id per item (-1 for items not in `items`): connected through touching segment ends."""
    gid = np.full(scene.n, -1, dtype=np.int64)
    sel = np.flatnonzero(items)
    if not len(sel):
        return gid
    pos = np.full(scene.n, -1, dtype=np.int64)
    pos[sel] = np.arange(len(sel))
    mask = items[scene.seg_item]
    segs, owner = scene.segs[mask], pos[scene.seg_item[mask]]
    pts = np.vstack([segs[:, :2], segs[:, 2:]])
    own = np.concatenate([owner, owner])
    pairs = cKDTree(pts).query_pairs(TOUCH_M, output_type="ndarray")
    a, b = own[pairs[:, 0]], own[pairs[:, 1]]
    keep = (a != b) & (scene.layer[sel[a]] == scene.layer[sel[b]])
    graph = coo_matrix((np.ones(keep.sum()), (a[keep], b[keep])), shape=(len(sel), len(sel)))
    _, comp = connected_components(graph, directed=False)
    gid[sel] = comp
    return gid


def doubtful_groups(scene: Scene, p: np.ndarray, applied: np.ndarray, gid: np.ndarray,
                    max_groups: int = 45, min_length: float = 0.3, max_extent: float = 8.0) -> list[dict]:
    length = scene.lengths()
    bb = scene.bboxes()
    order = np.argsort(gid, kind="stable")
    g_sorted = gid[order]
    starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
    out = []
    for s, e in zip(starts, np.r_[starts[1:], len(order)]):
        if g_sorted[s] < 0:
            continue
        idx = order[s:e]
        L = length[idx].sum()
        if L < min_length:
            continue
        mean_p = float(np.average(p[idx], weights=np.maximum(length[idx], 1e-3)))
        split = len(set(applied[idx].tolist())) > 1
        if not (0.2 < mean_p < 0.8 or split):
            continue
        b = bb[idx]
        if max(np.nanmax(b[:, 2]) - np.nanmin(b[:, 0]), np.nanmax(b[:, 3]) - np.nanmin(b[:, 1])) > max_extent:
            continue
        out.append({"items": idx, "length": float(L), "mean_p": mean_p, "split": split,
                    "bbox": [float(np.nanmin(b[:, 0])), float(np.nanmin(b[:, 1])),
                             float(np.nanmax(b[:, 2])), float(np.nanmax(b[:, 3]))]})
    # most ink first, the most doubtful first among equals
    out.sort(key=lambda g: -g["length"] * (1.0 - abs(g["mean_p"] - 0.5)))
    return out[:max_groups]


def group_picture(scene: Scene, applied: np.ndarray, group: dict, number: int, target: Path,
                  size: int = 520) -> Path:
    x0, y0, x1, y1 = group["bbox"]
    span = max(x1 - x0, y1 - y0)
    pad = max(3.5, 0.8 * span)                  # enough context to tell a floor plan from an elevation
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = span / 2 + pad
    colour = np.where(applied == 1, 1, 2).astype(np.int8)
    colour[group["items"]] = 9
    render(scene, target, colour, bbox=(cx - half, cy - half, cx + half, cy + half), width=size,
           colours={2: (205, 205, 215), 1: (40, 40, 40), 9: (220, 0, 0)}, thick={9: 3})
    img = Image.open(target).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 70, 26), fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((6, 6), f"#{number}  {2 * half:.0f} m", fill=(0, 0, 0))
    img.save(target)
    return target


def ask_claude(pictures: list[tuple[int, Path]], workdir: Path, model: str = "sonnet") -> dict:
    """One headless Claude Code call for a batch of group pictures. Returns decisions + cost."""
    listing = "\n".join(f"- grup {n}: {p}" for n, p in pictures)
    prompt = f"""Sen deneyimli bir mimari çizimcisin. Bir mimarın kat planı, elektrik projesine altlık olsun diye temizleniyor.
Aşağıdaki her resim planın küçük bir kesiti. KIRMIZI çizgiler kararı verilecek nesne grubu. KOYU GRİ: şu an temizlenmiş planda kalanlar.
AÇIK GRİ: silinmiş olanlar (bağlam için). Sol üstte grup numarası ve kesitin genişliği (metre) yazar.

{STANDARD}

Her resmi Read aracıyla aç ve kırmızı grubun KALMASI mı SİLİNMESİ mi gerektiğine karar ver.
Resimler:
{listing}

Cevap olarak YALNIZCA şu biçimde bir JSON dizisi ver, başka hiçbir şey yazma:
[{{"group": <numara>, "decision": "keep" | "delete", "what": "<kısa: ne olduğu>", "confidence": <0-1>}}, ...]"""
    t0 = time.time()
    proc = subprocess.run([CLAUDE, "-p", prompt, "--model", model, "--output-format", "json",
                           "--allowedTools", "Read"], cwd=workdir, capture_output=True, text=True,
                          encoding="utf-8", timeout=900, stdin=subprocess.DEVNULL)
    out = json.loads(proc.stdout)
    text = out.get("result") or ""
    match = re.search(r"\[.*\]", text, re.S)
    decisions = json.loads(match.group(0)) if match else []
    return {"decisions": decisions, "cost_usd": out.get("total_cost_usd", 0.0),
            "seconds": round(time.time() - t0, 1), "raw": text[-2000:], "is_error": out.get("is_error")}


def review(scene: Scene, p: np.ndarray, applied: np.ndarray, in_plan: np.ndarray, workdir: Path,
           max_groups: int = 45, log=lambda m: None) -> dict:
    """Run the review; returns the new decisions and a record of what was asked and answered."""
    workdir.mkdir(parents=True, exist_ok=True)
    gid = object_groups(scene, in_plan)
    groups = doubtful_groups(scene, p, applied, gid, max_groups)
    log(say(f"{gid.max() + 1} nesne grubu, {len(groups)} şüpheli grup incelenecek",
            f"{gid.max() + 1} item groups, {len(groups)} doubtful groups to review"))
    pics = [(k + 1, group_picture(scene, applied, g, k + 1, workdir / f"group_{k + 1:03d}.png"))
            for k, g in enumerate(groups)]
    new = applied.copy()
    calls, answers = [], {}
    for start in range(0, len(pics), BATCH):
        batch = pics[start:start + BATCH]
        res = ask_claude(batch, workdir)
        calls.append({k: v for k, v in res.items() if k != "decisions"})
        log(say(f"paket {start // BATCH + 1}: {len(res['decisions'])} cevap, ${res['cost_usd']:.3f}, {res['seconds']} s",
                f"batch {start // BATCH + 1}: {len(res['decisions'])} answers, ${res['cost_usd']:.3f}, {res['seconds']} s"))
        for d in res["decisions"]:
            answers[int(d["group"])] = d
    changed = 0
    record = []
    for k, g in enumerate(groups, 1):
        a = answers.get(k)
        before = "split" if g["split"] else ("delete" if g["mean_p"] >= 0.5 else "keep")
        if a and a.get("decision") in ("keep", "delete"):
            value = 1 if a["decision"] == "keep" else 0
            changed += int((new[g["items"]] != value).sum())
            new[g["items"]] = value
        record.append({"group": k, "items": len(g["items"]), "length_m": round(g["length"], 2),
                       "mean_p": round(g["mean_p"], 3), "model": before, "answer": a})
    return {"applied": new, "groups": record, "calls": calls, "changed_items": changed,
            "cost_usd": round(sum(c["cost_usd"] or 0 for c in calls), 4)}
