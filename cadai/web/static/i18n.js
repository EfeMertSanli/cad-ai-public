// Turkish and English (ADR-033). Every string is written in both languages where it is used:
//   T("Yerleştir", "Place")                    in page code
//   L(value)                                   text from the server: {"tr", "en"} or plain text
//   <b data-en="Place">Yerleştir</b>           fixed markup (data-en-html when it holds tags,
//                                              data-en-placeholder / data-en-title for attributes)
// The EN/TR switch (or a link with ?lang=en) stores the choice in the browser and in a cookie, so the server
// writes error texts and download names in the same language.
const LANG = (() => {
  let lang = "tr";
  try {
    const asked = new URLSearchParams(location.search).get("lang");
    lang = asked || localStorage.getItem("cadai-lang") || "tr";
    if (asked) localStorage.setItem("cadai-lang", asked);
  } catch (e) {}
  lang = lang === "en" ? "en" : "tr";
  document.cookie = `cadai-lang=${lang}; path=/; max-age=31536000; SameSite=Lax`;
  document.documentElement.lang = lang;
  return lang;
})();

function T(tr, en) { return LANG === "en" ? en : tr; }

function L(value) {
  if (value && typeof value === "object" && "tr" in value) return LANG === "en" ? (value.en ?? value.tr) : value.tr;
  return value ?? "";
}

function setLang(lang) {
  try { localStorage.setItem("cadai-lang", lang); } catch (e) {}
  document.cookie = `cadai-lang=${lang}; path=/; max-age=31536000; SameSite=Lax`;
  const url = new URL(location.href);
  url.searchParams.delete("lang");
  location.replace(url);
}

let _applied = false;
function applyLang() {             // once per page: header() calls it, before the page fills anything in
  if (LANG !== "en" || _applied) return;
  _applied = true;
  const root = document;
  root.querySelectorAll("[data-en]").forEach(e => { e.textContent = e.dataset.en; });
  root.querySelectorAll("[data-en-html]").forEach(e => { e.innerHTML = e.dataset.enHtml; });
  root.querySelectorAll("[data-en-placeholder]").forEach(e => { e.placeholder = e.dataset.enPlaceholder; });
  root.querySelectorAll("[data-en-title]").forEach(e => { e.title = e.dataset.enTitle; });
}
document.addEventListener("DOMContentLoaded", () => applyLang());

// names shared by several pages
const DEVICE_NAMES = {light_ceiling: T("tavan armatürü", "ceiling light"), socket: T("priz", "socket"),
  socket_wet: T("etanj priz", "wet-area socket"), switch: T("anahtar", "switch"),
  panel: T("daire panosu", "apartment panel"), light_emergency: T("acil aydınlatma", "emergency light")};
const ROOM_NAMES = {bedroom: T("yatak odası", "bedroom"), living: T("salon", "living room"),
  living_kitchen: T("açık mutfaklı salon", "living room with open kitchen"), kitchen: T("mutfak", "kitchen"),
  bathroom: T("banyo", "bathroom"), wc: T("wc", "WC"), hall: T("hol", "hall"), balcony: T("balkon", "balcony"),
  study: T("çalışma", "study"), laundry: T("çamaşır", "laundry"), dressing: T("giyinme", "dressing room"),
  storage: T("kiler/depo", "pantry/storage"), stair: T("merdiven", "stair"), lift: T("asansör", "lift"),
  shop: T("dükkan", "shop"), office: T("ofis", "office"), shelter: T("sığınak", "shelter"),
  plant: T("kazan dairesi", "boiler room"), parking: T("otopark", "car park")};
