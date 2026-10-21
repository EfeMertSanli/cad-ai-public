// Shared helpers for the pages (load after i18n.js).
async function api(path, options) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(L(body.error) || L(body.detail) || res.statusText);
  return body;
}

function runJob(kind, params) {
  return api(`/api/jobs/${kind}`, {method: "POST", headers: {"Content-Type": "application/json"},
                                   body: JSON.stringify(params || {})});
}

function el(tag, attrs, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c != null) e.append(c.nodeType ? c : String(c));
  return e;
}

function pill(text, kind) { return el("span", {class: "pill " + (kind || "")}, text); }

function fmtSeconds(s) {
  if (s == null) return "";
  if (s < 90) return `${Math.round(s)} ${T("sn", "s")}`;
  if (s < 5400) return `${Math.round(s / 60)} ${T("dk", "min")}`;
  return `${(s / 3600).toFixed(1)} ${T("sa", "h")}`;
}

function pct(v, digits) { return v == null ? "-" : T(`%${(v * 100).toFixed(digits || 0)}`, `${(v * 100).toFixed(digits || 0)}%`); }

function header(active) {
  applyLang();
  const links = [["/demo", "Demo"], ["/", T("Projeler", "Projects")], ["/clean", T("Temizle", "Clean")],
    ["/draw", T("Çizim", "Placement")], ["/edit", T("Düzelt", "Correct")], ["/lexicon/symbols", T("Semboller", "Symbols")],
    ["/lexicon/furniture", T("Mobilya", "Furniture")], ["/lexicon/furniture_shapes", T("Mobilya şekilleri", "Furniture shapes")],
    ["/debug", "Debug"]];
  document.body.prepend(el("header", {},
    el("span", {class: "brand"}, "cad-ai"),
    el("nav", {}, links.map(([href, t]) => el("a", {href, class: href === active ? "active" : ""}, t))),
    el("div", {class: "lang", title: T("Dil / Language", "Language / Dil")},
       ...[["en", "English"], ["tr", "Türkçe"]].map(([code, name]) =>       // each name in its own language
         el("button", {class: code === LANG ? "on" : "", onclick: () => code !== LANG && setLang(code)}, name)))));
}

const STATUS_PILL = {queued: [T("sırada", "queued"), ""], running: [T("çalışıyor", "running"), "warn"],
                     done: [T("bitti", "done"), "ok"], failed: [T("hata", "failed"), "bad"], cancelled: [T("iptal", "cancelled"), ""]};
