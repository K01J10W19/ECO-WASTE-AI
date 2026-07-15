/* CarbIQ SPA — application script.
   Extracted verbatim from the former inline <script> in index.html.
   Loaded as a classic deferred script; depends on the vendored globals
   window.gsap (app/static/vendor/gsap.min.js) and the Tailwind runtime. */
"use strict";
/* =========================================================================
   CarbIQ frontend — Step 7 SPA (design compiled to vanilla JS).

   Real data flow (CLAUDE.md v3.6):
     upload → POST /api/predict            (dual-tower boxes + materials)
            → POST /api/calculate-impact   (Stage-B audit: id echo, weights
                                            or box-area proxy, country)
            → POST /api/recommend          (DMM: 3 ranked end-of-life paths
                                            + child-simple verdict/pros/cons)
   GSAP drives the intro timeline, card staggers, badge pops and the
   carbon counter. No template runtime — plain DOM + fetch.
   ========================================================================= */

const GAMMA = 8000.0;               // display mirror of carbon_service.PIXEL_AREA_GAMMA
const ALERT_KG = 0.12;              // per-item chip/box alert threshold (kg CO2e)
const KM_PER_KG = 1 / 0.12;         // ~0.12 kg CO2e per petrol-car km
const RECYCLABLE = new Set(["plastic", "glass", "metal", "cardboard", "paper"]);

// Display-only reference data per taxonomy class (approximations for the
// inspector cells; the CO2e numbers always come from the backend).
const MATERIAL_META = {
  "plastic":         { tag: "PLA", density: 0.95, decompose: "~450 yrs" },
  "glass":           { tag: "GLS", density: 2.50, decompose: "~1M yrs" },
  "metal":           { tag: "MET", density: 2.70, decompose: "~200 yrs" },
  "cardboard":       { tag: "CRD", density: 0.69, decompose: "~2 months" },
  "paper":           { tag: "PPR", density: 0.80, decompose: "~6 weeks" },
  "biodegradable":   { tag: "BIO", density: 0.60, decompose: "~6 weeks" },
  "general rubbish": { tag: "GEN", density: 0.40, decompose: "varies" },
};
// Reference grid intensities (kgCO2/kWh) for the profile card — display only.
const COUNTRIES = [
  { code: "MY", name: "Malaysia",       grid: 0.585 },
  { code: "SG", name: "Singapore",      grid: 0.408 },
  { code: "JP", name: "Japan",          grid: 0.462 },
  { code: "CN", name: "China",          grid: 0.582 },
  { code: "IN", name: "India",          grid: 0.713 },
  { code: "US", name: "United States",  grid: 0.386 },
  { code: "GB", name: "United Kingdom", grid: 0.207 },
  { code: "DE", name: "Germany",        grid: 0.380 },
  { code: "FR", name: "France",         grid: 0.056 },
  { code: "AU", name: "Australia",      grid: 0.531 },
  { code: "NZ", name: "New Zealand",    grid: 0.112 },
];
// Status badge presets — kept in exact colour lockstep with the rank-based
// active tab classes (1 emerald / 2 amber / 3 rose) and the number chips.
const STATUS_CLS = {
  "Optimal":    "bg-emerald-50 text-emerald-800 border border-emerald-200",
  "Acceptable": "bg-amber-50 text-amber-800 border border-amber-200",
  "Warning":    "bg-rose-50 text-rose-800 border border-rose-200",
};
const TAB_LABEL = {
  recycling: "Recycling", incineration: "Incineration", landfill: "Landfill",
  composting: "Composting", anaerobic_digestion: "Digestion",
  material_recovery: "Recovery",
};

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const state = {
  img: null,                // HTMLImageElement of the analysed photo
  imgUrl: null,             // its object URL — revoked on reset/new scan (leak-free)
  imgW: 0, imgH: 0,         // original pixel dims from /predict
  items: [],                // /predict items (id, class_name, bbox, ...)
  weights: {},              // id -> current weight (kg)
  edited: new Set(),        // ids whose weight the user verified (Stage B)
  audit: {},                // id -> {co2e_kg, factor, source, weight_source}
  auditMeta: { provider: null, total: 0 },
  recs: {},                 // id -> {paths: [3 ranked], provider}
  process: {},              // id -> selected method key
  specTab: {},              // id -> "specs" | "env" (progressive-disclosure sub-tab)
  activeId: null,
  minConf: 0.5,             // confidence pill threshold — client-side VIEW filter
  country: "MY",
  geoMode: "AUTO-GEO",
  reveal: 1,                // gsap-driven box draw-in progress
  busy: false,
};
let carbonShown = { v: 0 };
let canvas, ctx;

/* ------------------------------ helpers ------------------------------ */

function toast(msg, ms = 4200) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  gsap.fromTo(el, { y: 16, opacity: 0 }, { y: 0, opacity: 1, duration: 0.3 });
  clearTimeout(el._t);
  el._t = setTimeout(() => gsap.to(el, {
    opacity: 0, duration: 0.3, onComplete: () => el.classList.add("hidden"),
  }), ms);
}

function setStatus(text, tone = "ok") {
  $("model-status").textContent = text;
  $("model-status").className = tone === "ok"
    ? "text-teal-600 font-semibold" : "text-amber-600 font-semibold";
  $("status-dot").className = "w-1.5 h-1.5 rounded-full pulse-dot "
    + (tone === "ok" ? "bg-teal-500" : "bg-amber-500");
}

async function api(path, payload, signal) {
  const res = await fetch(path, {
    method: "POST",
    headers: payload instanceof FormData ? {} : { "Content-Type": "application/json" },
    body: payload instanceof FormData ? payload : JSON.stringify(payload),
    signal,
  });
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
  return body;
}

function meta(material) {
  return MATERIAL_META[material] || { tag: "OBJ", density: 1.0, decompose: "varies" };
}
function itemWeight(it) {
  return state.weights[it.id] ?? +(it.box_area_px / GAMMA).toFixed(4);
}
function itemCo2(it) {
  const a = state.audit[it.id];
  return a ? a.co2e_kg : it.estimated_carbon_kg;   // Stage-A blind proxy until audited
}
function co2Color(v) {
  return v > ALERT_KG ? (v > ALERT_KG * 2 ? "#f43f5e" : "#f59e0b") : "#14b8a6";
}
// Confidence pill: pure view-layer filter — API payloads always carry ALL
// items; only the canvas, the grid and the telemetry sums react.
function isVisible(it) { return it.confidence >= state.minConf - 1e-9; }
function visibleItems() { return state.items.filter(isVisible); }

function applyConfFilter() {
  document.querySelectorAll("#items-list .item-card").forEach(card => {
    const it = state.items.find(x => x.id === +card.dataset.id);
    card.classList.toggle("hidden", !!it && !isVisible(it));
  });
  const total = state.items.length;
  const vis = visibleItems().length;
  $("object-count").textContent = vis;
  const note = $("filtered-note");
  if (total && vis < total) {
    note.textContent = ` (+${total - vis} filtered)`;
    note.classList.remove("hidden");
  } else {
    note.classList.add("hidden");
  }
}

/* ------------------------------ country / geo ------------------------------ */

// Custom dropdown (replaces the native <select>): renders the option list,
// mirrors the selection into the hidden #country-select input (the backend
// contract), and highlights the active row. Safe to call repeatedly —
// geolocate() re-runs it when a new country joins the list.
function buildCountrySelect() {
  const menu = $("country-menu");
  menu.innerHTML = COUNTRIES.map(c => `
    <button type="button" role="option" data-code="${c.code}"
      aria-selected="${c.code === state.country}"
      class="country-option w-full flex items-center justify-between gap-2 px-3.5 py-2 text-left text-sm transition-colors duration-150 hover:bg-teal-50 ${c.code === state.country ? "bg-teal-50/60 text-teal-700 font-semibold" : "text-zinc-700"}">
      <span class="truncate">${c.code} — ${esc(c.name)}</span>
      ${c.code === state.country ? '<span class="shrink-0 text-teal-500 text-xs">✓</span>' : ""}
    </button>`).join("");
  $("country-select").value = state.country;                   // hidden input sync
  $("country-trigger-label").textContent =
    `${state.country} — ${countryName(state.country)}`;
  menu.querySelectorAll(".country-option").forEach(btn => {
    btn.addEventListener("click", () => {
      closeCountryMenu();
      if (btn.dataset.code === state.country) return;
      state.country = btn.dataset.code;
      state.geoMode = "OVERRIDE";
      buildCountrySelect();          // re-render checkmark + hidden input
      syncGeoUi();
      gsap.fromTo("#geo-badge", { scale: 0.92 }, { scale: 1, duration: 0.4, ease: "back.out(2)" });
      runGridRefresh();   // micro grid-recalc masks + re-audit + re-rank
    });
  });
}

let countryMenuOpen = false;
function openCountryMenu() {
  const menu = $("country-menu");
  menu.classList.remove("hidden");
  requestAnimationFrame(() => menu.classList.remove("opacity-0", "scale-95"));
  $("country-trigger").setAttribute("aria-expanded", "true");
  $("country-chevron").style.transform = "rotate(180deg)";
  countryMenuOpen = true;
}
function closeCountryMenu() {
  if (!countryMenuOpen) return;
  const menu = $("country-menu");
  menu.classList.add("opacity-0", "scale-95");
  setTimeout(() => menu.classList.add("hidden"), 150);   // let the ease finish
  $("country-trigger").setAttribute("aria-expanded", "false");
  $("country-chevron").style.transform = "";
  countryMenuOpen = false;
}

function countryName(code) {
  const hit = COUNTRIES.find(c => c.code === code);
  if (hit) return hit.name;
  try { return new Intl.DisplayNames(["en"], { type: "region" }).of(code) || code; }
  catch { return code; }
}

function syncGeoUi() {
  const c = COUNTRIES.find(x => x.code === state.country);
  $("geo-mode").textContent = state.geoMode;
  $("geo-code").textContent = state.country;
  $("geo-name").textContent = countryName(state.country);
  $("grid-intensity").textContent = c ? c.grid.toFixed(3) : "—";
  $("geo-mode-lower").textContent =
    state.geoMode === "OVERRIDE" ? "manual override" : "auto-detected region";
}

async function geolocate() {
  try {
    const res = await fetch("https://ipapi.co/json/", { signal: AbortSignal.timeout(4000) });
    const data = await res.json();
    const code = String(data.country_code || "").toUpperCase();
    if (/^[A-Z]{2}$/.test(code)) {
      if (!COUNTRIES.some(c => c.code === code)) {
        COUNTRIES.unshift({ code, name: countryName(code), grid: NaN });
      }
      state.country = code;
      buildCountrySelect();   // re-render trigger label + hidden input + ✓
    }
  } catch { /* offline / blocked — keep the MY default silently */ }
  state.geoMode = "AUTO-GEO";
  syncGeoUi();
  gsap.fromTo("#geo-badge", { scale: 0.9, opacity: 0.4 },
    { scale: 1, opacity: 1, duration: 0.5, ease: "back.out(2)" });
}

/* ------------------------------ upload / camera ------------------------------ */

function wireUpload() {
  const overlay = $("drop-overlay");
  overlay.addEventListener("dragover", (e) => e.preventDefault());
  overlay.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) analyze(f);
  });
  $("browse-btn").addEventListener("click", () => $("file-input").click());
  $("file-input").addEventListener("change", (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) analyze(f);
  });
  $("new-scan-btn").addEventListener("click", resetWorkspace);

  $("camera-btn").addEventListener("click", openCamera);
  $("capture-btn").addEventListener("click", captureFrame);
  $("cancel-camera-btn").addEventListener("click", closeCamera);
}

let cameraStream = null;
async function openCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    $("camera-video").srcObject = cameraStream;
    $("camera-overlay").classList.remove("hidden");
    $("camera-overlay").classList.add("flex");
  } catch (err) { toast("Camera unavailable: " + err.message); }
}
function closeCamera() {
  if (cameraStream) { cameraStream.getTracks().forEach(t => t.stop()); cameraStream = null; }
  $("camera-overlay").classList.add("hidden");
  $("camera-overlay").classList.remove("flex");
}
function captureFrame() {
  const v = $("camera-video");
  if (!v.videoWidth) return;
  const cv = document.createElement("canvas");
  cv.width = v.videoWidth; cv.height = v.videoHeight;
  cv.getContext("2d").drawImage(v, 0, 0);
  cv.toBlob((blob) => {
    closeCamera();
    if (blob) analyze(new File([blob], "capture.png", { type: "image/png" }));
  }, "image/png");
}

/* ------------------------------ core flow ------------------------------ */

async function analyze(file) {
  if (state.busy) return;
  state.busy = true;
  // Heavy-AI scan state (Issue 1): left canvas mask + right telemetry
  // skeleton. Set here at the exact start; cleared in finally() below so
  // BOTH success and error paths always restore normal rendering.
  document.body.classList.add("is-scanning-active");
  setStatus("ANALYZING…", "warn");
  try {
    const form = new FormData();
    form.append("image", file);
    const data = await api("/api/predict", form);

    // Fresh scan state.
    state.items = data.items;
    state.imgW = data.image.width;
    state.imgH = data.image.height;
    state.weights = {}; state.edited = new Set();
    state.audit = {}; state.recs = {}; state.process = {}; state.specTab = {};
    state.auditMeta = { provider: "local (γ proxy)", total: null };
    state.activeId = data.items.length
      ? data.items.reduce((a, b) => (b.confidence > a.confidence ? b : a)).id : null;

    // Show the user's own file immediately (same pixels the server analysed).
    if (state.imgUrl) URL.revokeObjectURL(state.imgUrl);
    state.imgUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { state.img = img; draw(); };
    img.src = state.imgUrl;
    $("file-input").value = "";   // allow re-picking the same file later

    const overlay = $("drop-overlay");
    gsap.to(overlay, { opacity: 0, duration: 0.35, onComplete: () => {
      overlay.classList.add("hidden"); overlay.style.opacity = "";
    }});
    $("new-scan-btn").classList.remove("hidden");
    $("footer-hint").textContent = "click a box to inspect";
    $("object-count").textContent = data.items.length;

    renderItems();
    updateTelemetry();
    // GSAP: stagger the fresh cards in + draw-in sweep for the boxes.
    gsap.fromTo(".gsap-stagger-item", { y: 18, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.5, ease: "power2.out", stagger: 0.08 });
    gsap.fromTo(state, { reveal: 0 }, { reveal: 1, duration: 0.7, ease: "power2.out", onUpdate: draw });

    if (!data.items.length) {
      toast("No waste-like objects detected — try a closer, brighter shot.");
      setStatus("READY");
    } else {
      setStatus("AUDITING…", "warn");
      await Promise.all([refreshImpact(), refreshRecommendations()]);
      setStatus("READY");
    }
  } catch (err) {
    toast("Analysis failed: " + err.message);
    setStatus("ERROR", "warn");
  } finally {
    state.busy = false;
    document.body.classList.remove("is-scanning-active");   // never wedge
  }
}

function resetWorkspace() {
  /* NEW SCAN — choreographed slate transition back to the pristine state:
     fade the old scan out (0.2 s), purge ALL scan state + DOM cards, restore
     the blueprint-grid canvas and zeroed telemetry, then pop the upload card
     back to the forefront. View-only + leak-free (object URL revoked,
     pending debounces cancelled, counter tweens killed). */
  if (state.busy) return;                 // never race an in-flight analysis
  clearTimeout(refreshTimer);
  // Defensive: NEW SCAN can fire during a grid recalc (which doesn't set
  // state.busy) — drop any lingering loading masks so the pristine state is
  // truly clean. The in-flight refresh's own finally() is a no-op once the
  // class is gone.
  document.body.classList.remove("is-scanning-active", "is-recalculating-grid");
  gridRefreshInFlight = 0;
  gsap.killTweensOf(carbonShown);
  gsap.killTweensOf(state);
  closeCamera();

  const overlay = $("drop-overlay");
  const cards = document.querySelectorAll("#items-list .item-card");
  const tl = gsap.timeline();

  // 1) Visual purge — old image + boxes + cards fade to 0 in 0.2 s.
  tl.to("#detection-canvas", { opacity: 0, duration: 0.2, ease: "power1.in" }, 0);
  if (cards.length) tl.to(cards, { opacity: 0, y: 8, duration: 0.2, stagger: 0.015 }, 0);

  // 2) Hard state wipe + zeroed telemetry, applied while invisible.
  tl.add(() => {
    if (state.imgUrl) URL.revokeObjectURL(state.imgUrl);
    state.img = null; state.imgUrl = null; state.imgW = 0; state.imgH = 0;
    state.items = []; state.weights = {}; state.edited = new Set();
    state.audit = {}; state.recs = {}; state.process = {}; state.specTab = {};
    state.auditMeta = { provider: null, total: null };
    state.activeId = null; state.reveal = 1;

    carbonShown.v = 0;
    $("carbon-number").textContent = "0.000";
    $("km-equiv").textContent = "0.0";
    $("footer-hint").textContent = "upload an image to begin";
    $("new-scan-btn").classList.add("hidden");
    $("dmm-provider").textContent = "bi-directional focus · sync’d to canvas";
    $("file-input").value = "";

    renderItems();       // empties the accordion (placeholder copy returns)
    updateTelemetry();   // ITEMS 0 · MASS 0.000 · RECYCLABLE 0 · AWAITING SCAN
    draw();              // pristine blueprint-grid viewport, zero boxes
    overlay.classList.remove("hidden");
  });

  // 3) Slate fade-back-in + the upload card pops to the forefront.
  tl.to("#detection-canvas", { opacity: 1, duration: 0.25, ease: "power1.out" });
  tl.fromTo(overlay, { opacity: 0 },
    { opacity: 1, duration: 0.3, clearProps: "opacity" }, "<");
  tl.fromTo("#drop-card", { scale: 0.9, opacity: 0 },
    { scale: 1, opacity: 1, duration: 0.45, ease: "back.out(1.7)",
      clearProps: "transform,opacity" }, "-=0.15");
}

function impactPayloadItems() {
  return state.items.map(it => state.edited.has(it.id)
    ? { id: it.id, material: it.class_name, weight_kg: state.weights[it.id] }
    : { id: it.id, material: it.class_name, box_area_px: it.box_area_px });
}

// SINGLE-FLIGHT dedup: each refresher keeps at most ONE request in the air.
// A new trigger (country flip mid-audit, rapid weight edits past the
// debounce) aborts the superseded call — so a scan can never stack two
// concurrent /api/recommend flights (= two LLM calls) and a slow stale
// response can never overwrite fresher state.
let impactCtl = null, recsCtl = null;

async function refreshImpact() {
  if (!state.items.length) return;
  if (impactCtl) impactCtl.abort();
  const ctl = (impactCtl = new AbortController());
  try {
    const data = await api("/api/calculate-impact",
      { items: impactPayloadItems(), country: state.country }, ctl.signal);
    if (ctl !== impactCtl) return;      // superseded while we were in flight
    data.items.forEach(row => {
      state.audit[row.id] = row;
      if (!state.edited.has(row.id)) state.weights[row.id] = row.weight_kg;
    });
    state.auditMeta = { provider: data.provider, total: data.total_co2e_kg };
  } catch (err) {
    if (err.name === "AbortError") return;   // deliberately superseded — quiet
    toast("Carbon audit unavailable: " + err.message);
  }
  renderItems();
  updateTelemetry();
  draw();
}

async function refreshRecommendations() {
  if (!state.items.length) return;
  if (recsCtl) recsCtl.abort();
  const ctl = (recsCtl = new AbortController());
  try {
    const data = await api("/api/recommend", {
      items: state.items.map(it => state.edited.has(it.id)
        ? { material: it.class_name, weight_kg: state.weights[it.id] }
        : { material: it.class_name, box_area_px: it.box_area_px }),
      country: state.country,
    }, ctl.signal);
    if (ctl !== recsCtl) return;        // superseded while we were in flight
    data.items.forEach((rec, i) => {
      const id = state.items[i].id;
      state.recs[id] = rec;
      // Selection-pointer pivot: if the remembered method vanished OR became
      // nationally inapplicable (e.g. "landfill" highlighted under MY, then
      // the geo dropdown switches to zero-landfill SG), snap the pointer to
      // the newly computed applicable Optimal — never leave it on a dead pill.
      const current = rec.recommendations.find(p => p.method === state.process[id]);
      if (!current || !current.is_applicable) {
        state.process[id] = rec.best_method;   // rank-1 applicable path
      }
    });
    $("dmm-provider").textContent = "DMM · " + data.provider.replaceAll("_", " ");
    // v3.8 grid engine: the server echoes the exact intensity that scaled the
    // disposal factors (live Climatiq probe or local map) — overwrite the
    // static reference value so the profile card shows the datum in force.
    if (data.grid) {
      $("grid-intensity").textContent = data.grid.intensity_kg_per_kwh.toFixed(3);
      $("geo-mode-lower").textContent = data.grid.source === "climatiq"
        ? "live grid intensity" : ($("geo-mode-lower").textContent);
    }
  } catch (err) {
    if (err.name === "AbortError") return;   // deliberately superseded — quiet
    toast("Recommendations unavailable: " + err.message);
  }
  renderItems();
}

/* ------------------------------ telemetry ------------------------------ */

function updateTelemetry() {
  // Sums run over the VISIBLE set only — the confidence pill subtracts
  // filtered-out items from the whole dashboard in real time.
  const items = visibleItems();
  const total = items.reduce((s, it) => s + itemCo2(it), 0);
  const mass = items.reduce((s, it) => s + itemWeight(it), 0);
  const recyclable = items.length
    ? Math.round(100 * items.filter(it => RECYCLABLE.has(it.class_name)).length / items.length)
    : 0;

  gsap.to(carbonShown, {
    v: total, duration: 0.9, ease: "power3.out", overwrite: true,
    onUpdate: () => { $("carbon-number").textContent = carbonShown.v.toFixed(3); },
  });
  $("km-equiv").textContent = (total * KM_PER_KG).toFixed(1);
  $("stat-items").textContent = items.length;
  $("stat-mass").textContent = mass.toFixed(3);
  $("stat-recyclable").textContent = recyclable;
  $("provider-label").textContent = state.auditMeta.provider
    ? `${state.auditMeta.provider} · ${state.country}` : "local (γ proxy)";

  const chip = $("status-chip");
  const base = "rounded-full border px-2.5 py-1 text-[10px] font-mono-d tracking-wider uppercase text-center whitespace-nowrap shrink-0 flex items-center justify-center ";
  if (!state.items.length) {
    chip.textContent = "AWAITING SCAN";
    chip.className = base + "border-teal-200/80 bg-teal-50 text-teal-700";
    return;
  }
  if (!items.length) {
    chip.textContent = "ALL FILTERED";
    chip.className = base + "border-amber-200/80 bg-amber-50 text-amber-600";
    return;
  }
  const elevated = items.some(it => itemCo2(it) > ALERT_KG);
  chip.textContent = elevated ? "ELEVATED IMPACT" : "LOW IMPACT";
  chip.className = base
    + (elevated ? "border-amber-200/80 bg-amber-50 text-amber-600"
                : "border-teal-200/80 bg-teal-50 text-teal-700");
}

/* ------------------------------ item cards ------------------------------ */

// Rank-based semantic ACTIVE tab styling: the selected pill's colour tells
// the story at a glance — Optimal green, Acceptable amber, Warning rose.
// (K applicable paths rank 1..K; anything unmapped falls back to teal.)
const ACTIVE_TAB_CLS = {
  1: "bg-emerald-100 text-emerald-800 border-emerald-300 font-medium",
  2: "bg-amber-100 text-amber-800 border-amber-300 font-medium",
  3: "bg-rose-100 text-rose-800 border-rose-300 font-medium",
};

// Golden key-data chips: wrap carbon figures ("3.5712 kg") in a highlight
// span, TONED to the active path's rank so the whole panel block speaks one
// colour (rank 2 amber, rank 3 rose, else the resting teal). Runs AFTER
// esc(), so LLM/local text can never inject markup — the regex only wraps
// already-escaped digit runs. Applies uniformly to live LLM responses and
// the local knowledge-grid fallback copy.
const NUM_CHIP_CLS = {
  2: "font-bold text-amber-700 bg-amber-50 px-1 py-0.5 rounded border border-amber-100 mx-0.5",
  3: "font-bold text-rose-700 bg-rose-50 px-1 py-0.5 rounded border border-rose-100 mx-0.5",
};
function highlightNums(text, rank) {
  const cls = NUM_CHIP_CLS[rank]
    || "font-bold text-teal-700 bg-teal-50/80 px-1 py-0.5 rounded border border-teal-100 mx-0.5";
  return esc(text).replace(/(\d+(?:\.\d+)?\s?kg)\b/gi,
    `<span class="${cls}">$1</span>`);
}

// ACTION PROTOCOL banner — a static, non-interactive capsule under Pros/Cons
// showing the server's 2-step, country-localized disposal guide as a
// horizontal chevron pipeline. The left accent + step tags are bound to the
// active path's rank (edge colour set inline so it always wins the cascade,
// regardless of Tailwind's class ordering). Steps run through highlightNums →
// esc(), so localized LLM text can never inject markup.
const RANK_ACCENT = {
  1: { edge: "#10b981", bg: "bg-emerald-50/40", tag: "text-emerald-700" },  // emerald-500
  2: { edge: "#f59e0b", bg: "bg-amber-50/40",   tag: "text-amber-700" },    // amber-500
  3: { edge: "#f43f5e", bg: "bg-rose-50/40",    tag: "text-rose-700" },     // rose-500
};
function actionBannerHtml(path, rec) {
  const steps = Array.isArray(path.action_steps) ? path.action_steps : [];
  const media = rec && rec.action_media ? rec.action_media : null;
  if (steps.length < 2 && !media) return "";   // schema guarantees both — defensive
  const a = RANK_ACCENT[path.rank]
    || { edge: "#14b8a6", bg: "bg-teal-50/40", tag: "text-teal-700" };
  const chevron = `<span class="text-slate-400 font-bold px-1 select-none">➔</span>`;
  const stepHtml = steps.slice(0, 2).map((s, i) => `
    <span class="flex items-baseline gap-1.5 min-w-0">
      <span class="shrink-0 font-bold text-[11px] tracking-wide uppercase ${a.tag}">Step ${i + 1}</span>
      <span class="text-slate-700 font-medium text-xs md:text-sm">${highlightNums(s, path.rank)}</span>
    </span>`).join(chevron);

  // Option A media: the SERVER resolved the LLM's search query into a
  // guaranteed-live embed URL (or the verified universal fallback) — the
  // model itself never writes URLs, so a hallucinated link cannot exist.
  // Only the expanded card renders this panel, so the spec ids stay unique.
  let mediaHtml = "";
  if (media) {
    const watchHref = esc(media.video_embed_url.replace("/embed/", "/watch?v="));
    const binHref = "https://www.google.com/search?q="
      + encodeURIComponent(countryName(state.country) + " recycling bin guidelines");
    mediaHtml = `
      <div class="expert-tip-box border-l-2 pl-3 my-2 border-l-slate-300">
        <p class="text-[10px] font-bold tracking-wider uppercase ${a.tag}">Expert tip</p>
        <p class="mt-1 text-[11px] leading-relaxed text-slate-600">${highlightNums(media.expert_tip, path.rank)}</p>
      </div>
      <iframe id="dmm-youtube-iframe" class="w-full aspect-video rounded-lg shadow-sm border border-slate-200 bg-zinc-900/5"
        src="${esc(media.video_embed_url)}" title="${esc(media.video_title)}" loading="lazy"
        referrerpolicy="strict-origin-when-cross-origin"
        allow="accelerometer; clipboard-write; encrypted-media; picture-in-picture" allowfullscreen></iframe>
      <div class="flex items-center justify-between gap-x-3 gap-y-1 flex-wrap">
        <a href="${watchHref}" target="_blank" rel="noopener" class="text-[11px] font-semibold ${a.tag} hover:underline">▶ Watch Full Tutorial</a>
        <span class="font-mono-d text-[9px] uppercase tracking-[0.14em] text-slate-400 select-none">${media.video_provider === "youtube_live" ? "live · youtube v3" : "universal fallback"}</span>
        <a href="${binHref}" target="_blank" rel="noopener" class="text-[11px] font-semibold text-slate-500 hover:text-slate-700 hover:underline">Check Local Bin Guidelines ↗</a>
      </div>`;
  }

  return `
    <div class="bg-slate-50 border border-slate-100 rounded-xl p-4 pt-8 mt-4 relative flex flex-col space-y-4 overflow-hidden" style="border-left: 4px solid ${a.edge}">
      <span class="absolute top-2.5 right-3.5 text-[10px] font-bold tracking-wider uppercase text-slate-400 select-none">⚡ Action Protocol</span>
      <div id="horizontal-step-flow" class="flex flex-row items-center flex-wrap gap-2 md:gap-4">${stepHtml}</div>
      ${mediaHtml}
    </div>`;
}

function tabsHtml(it) {
  const rec = state.recs[it.id];
  if (!rec) {
    return `<div class="mt-4 rounded-full border border-zinc-200/80 bg-zinc-50 px-4 py-2 font-mono-d text-[10px] text-zinc-400 uppercase tracking-[0.18em]">Running decision simulation…</div>`;
  }
  const selected = state.process[it.id];
  const buttons = rec.recommendations.map(p => {
    if (!p.is_applicable) {
      // Nationally banned route: hard-disabled pill — no rank number, no
      // Optimal/Acceptable badge, struck-through label, reason on hover.
      return `<button type="button" disabled data-process="${p.method}" title="${esc(p.restriction_reason || "Not nationally applicable")}" class="process-tab flex-1 min-w-0 rounded-full border border-transparent text-[11px] font-semibold py-1.5 px-1 truncate text-zinc-300 line-through opacity-50 cursor-not-allowed">✕ ${TAB_LABEL[p.method] || p.method}</button>`;
    }
    // Active pill colour maps to the path's computed rank; inactive pills
    // keep the muted gray (border-transparent = zero layout shift on select).
    const cls = p.method === selected
      ? (ACTIVE_TAB_CLS[p.rank] || "bg-teal-100 text-teal-700 border-teal-200 font-medium")
      : "border-transparent font-semibold text-zinc-400 hover:text-zinc-600";
    return `<button data-process="${p.method}" class="process-tab flex-1 min-w-0 rounded-full border text-[11px] py-1.5 px-1 truncate transition-colors duration-200 ${cls}">${p.rank}. ${TAB_LABEL[p.method] || p.method}</button>`;
  }).join("");
  const restricted = rec.recommendations.filter(p => !p.is_applicable);
  const restrictedNote = restricted.length
    ? `<p class="mt-1.5 px-1 font-mono-d text-[9px] tracking-[0.14em] uppercase text-zinc-400">✕ ${restricted.map(p => TAB_LABEL[p.method] || p.method).join(" · ")} — not nationally applicable in ${esc(countryName(state.country))}</p>`
    : "";
  // Panel binds to the selected path; guards guarantee it is an applicable
  // one (the refresh pivot never leaves the pointer on a banned method).
  const path = rec.recommendations.find(p => p.method === selected && p.is_applicable)
    || rec.recommendations.find(p => p.is_applicable)
    || rec.recommendations[0];
  return `
    <div class="mt-4 flex items-center w-full gap-1 rounded-full border border-zinc-200/80 bg-zinc-50 p-1">${buttons}</div>
    ${restrictedNote}
    <div class="mt-3 rounded-xl bg-zinc-50 border border-zinc-200/80 p-4 flex flex-col gap-3 dmm-panel">
      <div class="flex items-center justify-between">
        <span class="rounded-md bg-teal-50 border border-teal-200/80 text-teal-700 font-mono-d text-[9px] tracking-[0.18em] px-2 py-1">DMM</span>
        <span class="rounded-full px-2.5 py-1 text-[10px] font-semibold ${STATUS_CLS[path.status_tag] || STATUS_CLS.Acceptable}">Rank #${path.rank} · ${esc(path.status_tag)} · ${path.carbon_impact_kg.toFixed(3)} kg</span>
      </div>
      <p class="text-xs leading-relaxed text-slate-700">${highlightNums(path.encouraging_verdict, path.rank)}</p>
      <div class="pl-2 flex flex-col gap-1.5">
        <div class="flex items-start gap-2">
          <span class="shrink-0 mt-0.5 w-3.5 h-3.5 rounded-full bg-emerald-50 border border-emerald-200/80 text-emerald-600 text-[9px] font-bold flex items-center justify-center leading-none">+</span>
          <p class="text-[11px] leading-relaxed text-slate-600"><span class="font-semibold text-emerald-700">Pros:</span> ${highlightNums(path.environmental_pros, path.rank)}</p>
        </div>
        <div class="flex items-start gap-2">
          <span class="shrink-0 mt-0.5 w-3.5 h-3.5 rounded-full bg-rose-50 border border-rose-200/80 text-rose-600 text-[9px] font-bold flex items-center justify-center leading-none">–</span>
          <p class="text-[11px] leading-relaxed text-slate-600"><span class="font-semibold text-rose-700">Cons:</span> ${highlightNums(path.environmental_cons, path.rank)}</p>
        </div>
      </div>
      ${actionBannerHtml(path, rec)}
    </div>`;
}

// Carbon Equivalence Contextualizer constants (client-side, 0-lag).
const EQUIV = { phonesPerKg: 121.5, treeMonthsPerKg: 0.5 };
// Rank → progress-bar colour (banned paths carry no rank → neutral slate).
const ENV_BAR_CLS = {
  1: "bg-emerald-500 shadow-sm shadow-emerald-100",
  2: "bg-amber-500 shadow-sm shadow-amber-100",
  3: "bg-rose-500 shadow-sm shadow-rose-100",
};

// The DMM pathway currently in focus — the selected applicable path, else the
// first applicable, else whatever exists (mirrors tabsHtml's binding exactly).
function activeDmmPath(rec, id) {
  const sel = state.process[id];
  return rec.recommendations.find(p => p.method === sel && p.is_applicable)
    || rec.recommendations.find(p => p.is_applicable)
    || rec.recommendations[0];
}

// Environmental Impact sub-panel: a horizontal emissions-breakdown bar chart
// (magnitude-scaled, rank-coloured) + the Carbon Equivalence Contextualizer
// keyed off the ACTIVE pathway. Carbon can be negative (offsets), so bars are
// sized by |value| against the largest |value| — a signed value still prints.
function envPanelHtml(it, rec) {
  if (!rec) {
    return `<p class="text-[11px] text-slate-400 px-1">Environmental breakdown loads with the decision simulation…</p>`;
  }
  const paths = rec.recommendations;
  const maxAbs = Math.max(...paths.map(p => Math.abs(p.carbon_impact_kg)), 1e-4);
  const rows = paths.map(p => {
    const pct = Math.max(3, Math.abs(p.carbon_impact_kg) / maxAbs * 100);
    const barCls = p.is_applicable ? (ENV_BAR_CLS[p.rank] || "bg-teal-500") : "bg-slate-300";
    const label = TAB_LABEL[p.method] || p.method;
    return `
      <div class="flex items-center gap-2.5">
        <span class="w-20 shrink-0 text-[11px] font-medium truncate ${p.is_applicable ? "text-slate-600" : "text-slate-400 line-through"}">${label}</span>
        <div class="flex-1 min-w-0 h-2 rounded-full bg-slate-100 overflow-hidden">
          <div class="env-bar h-2 rounded-full transition-all duration-500 ease-out ${barCls}" data-target="${pct.toFixed(1)}" style="width: ${pct.toFixed(1)}%"></div>
        </div>
        <span class="w-[4.75rem] shrink-0 text-right text-[11px] font-semibold tabular-nums text-slate-700 whitespace-nowrap">${p.carbon_impact_kg.toFixed(3)}<span class="ml-0.5 text-[8px] text-slate-400 font-medium">kg</span></span>
      </div>`;
  }).join("");

  // Equivalence off the ACTIVE pathway. The SIGN carries meaning: a NEGATIVE
  // value is a saving (offset), so the copy flips from "emitted/required" to
  // "avoided/relief given" and tints emerald — the psychology now matches the
  // math (a green rank-1 recycling path reads as a WIN, not as damage).
  const act = activeDmmPath(rec, it.id);
  const activeCarbonValue = act.carbon_impact_kg;
  const isOffset = activeCarbonValue < 0;
  const absC = Math.abs(activeCarbonValue);
  const phones = Math.round(absC * EQUIV.phonesPerKg);
  const treeMonths = (absC * EQUIV.treeMonthsPerKg).toFixed(1);
  const phoneLabel = isOffset ? "Smartphone Charges Avoided" : "Smartphone Charges Emitted";
  const treeLabel = isOffset ? "Months of Tree Relief Given" : "Months of Tree Absorption Required";
  const equivNumCls = isOffset ? "text-emerald-600" : "text-slate-800";
  const equivSubCls = isOffset ? "text-emerald-600/90" : "text-slate-500";
  const equivTileCls = isOffset ? "bg-emerald-50/50 border-emerald-100" : "bg-white border-slate-100";

  return `
    <div class="flex flex-col gap-1.5">
      <p class="text-[10px] font-bold tracking-wider text-slate-400 uppercase mb-1">📊 Emissions by pathway · ${esc(countryName(state.country))} grid</p>
      ${rows}
    </div>
    <div>
      <span class="text-[10px] font-bold tracking-wider text-slate-400 uppercase mb-1 block">🌍 Local Carbon Impact Equivalent</span>
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-2 p-3 bg-slate-50 border border-slate-100/70 rounded-xl">
        <div class="flex items-center gap-2.5 rounded-lg border px-3 py-2 ${equivTileCls}">
          <span class="text-lg leading-none select-none">📱</span>
          <div class="min-w-0">
            <p class="text-sm font-bold tabular-nums leading-tight ${equivNumCls}">${phones.toLocaleString()}</p>
            <p class="text-[10px] leading-tight ${equivSubCls}">${phoneLabel}</p>
          </div>
        </div>
        <div class="flex items-center gap-2.5 rounded-lg border px-3 py-2 ${equivTileCls}">
          <span class="text-lg leading-none select-none">🌳</span>
          <div class="min-w-0">
            <p class="text-sm font-bold tabular-nums leading-tight ${equivNumCls}">${treeMonths}</p>
            <p class="text-[10px] leading-tight ${equivSubCls}">${treeLabel}</p>
          </div>
        </div>
      </div>
    </div>`;
}

function cardHtml(it) {
  const m = meta(it.class_name);
  const audit = state.audit[it.id];
  const weight = itemWeight(it);
  const co2 = itemCo2(it);
  const expanded = state.activeId === it.id;
  const alert = co2 > ALERT_KG;
  const co2Cls = alert ? (co2 > ALERT_KG * 2 ? "text-rose-500" : "text-amber-500") : "text-teal-600";
  const tileCls = alert ? "border-amber-200/80 bg-amber-50 text-amber-600"
                        : "border-teal-200/80 bg-teal-50 text-teal-700";
  const factorLine = audit
    ? `Factor ${audit.carbon_factor_kg_per_kg.toFixed(4)} kg/kg · ${audit.source} · ${audit.weight_source.replaceAll("_", " ")}`
    : `Stage-A proxy · base × area ÷ γ`;
  const volume = weight > 0 ? Math.round(weight / m.density * 1000) : 0;
  const specTab = state.specTab[it.id] || "specs";   // sub-tab client memory

  return `
  <article data-id="${it.id}" class="item-card rounded-2xl border border-zinc-300 bg-zinc-100 shadow-sm shadow-zinc-900/[0.04] cursor-pointer transition-all duration-300 hover:border-zinc-400 gsap-card gsap-stagger-item ${expanded ? "active-item p-6" : "py-2.5 px-6"}">
    <div class="flex items-center gap-4">
      <div class="w-11 h-11 shrink-0 self-start rounded-xl border flex items-center justify-center font-mono-d text-[11px] font-semibold ${tileCls}">${m.tag}</div>
      <div class="flex-1 min-w-0">
        <div class="flex flex-row items-center justify-between w-full gap-3">
          <div class="flex-1 min-w-0">
            <h3 class="text-sm font-bold text-zinc-900 leading-tight truncate">${esc(it.display_name)}</h3>
            <p class="mt-1 font-mono-d text-xs text-zinc-400 tracking-tight whitespace-nowrap truncate">CONF ${it.confidence.toFixed(2)} · ψ ${it.physics.plasticity_index.toFixed(2)} · box ${Math.round(it.box_area_px).toLocaleString()} px²</p>
          </div>
          <div class="flex items-center gap-2.5 shrink-0">
            ${expanded ? `
            <!-- Progressive-disclosure sub-tab switcher — relocated into the
                 header as a compact utility control (0-lag, client memory). -->
            <div class="spec-switch flex flex-row items-center bg-slate-100/80 p-0.5 rounded-lg text-xs max-w-[220px] shrink-0 select-none">
              <button data-spectab="specs" class="spec-tab rounded-md px-2.5 py-1 font-semibold whitespace-nowrap transition-all duration-200 ${specTab === "specs" ? "bg-white text-zinc-900 shadow-sm" : "text-zinc-500 hover:text-zinc-700"}">⚙️ Specs</button>
              <button data-spectab="env" class="spec-tab rounded-md px-2.5 py-1 font-semibold whitespace-nowrap transition-all duration-200 ${specTab === "env" ? "bg-white text-zinc-900 shadow-sm" : "text-zinc-500 hover:text-zinc-700"}">📊 Impact</button>
            </div>` : ""}
            <div class="text-right">
              <p class="text-lg font-extrabold tabular-nums leading-none ${co2Cls}">${co2.toFixed(3)}</p>
              <p class="mt-1 font-mono-d text-[9px] tracking-[0.18em] uppercase text-zinc-400">kg CO₂e</p>
              ${expanded ? `<p class="mt-1 font-mono-d text-[9px] text-zinc-400 tabular-nums whitespace-nowrap">${factorLine}</p>` : ""}
            </div>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" class="shrink-0 text-zinc-400 transition-transform duration-300 ${expanded ? "rotate-180" : ""}"><path d="M6 9l6 6 6-6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>
          </div>
        </div>
        ${expanded ? `
        <div class="mt-3">
          <div id="panel-specs" class="${specTab === "env" ? "hidden opacity-0" : "opacity-100"} transition-all duration-300 flex flex-col space-y-2">
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5 expand-body">
          <label class="flex flex-col md:flex-col lg:flex-row items-start md:items-start lg:items-center justify-between gap-1 md:gap-1 lg:gap-2 rounded-lg border border-zinc-200/80 bg-zinc-50 px-2.5 md:px-2.5 lg:px-3 py-2">
            <span class="font-mono-d text-[8px] md:text-[8px] lg:text-[9px] tracking-tight lg:tracking-[0.18em] uppercase text-zinc-400 shrink-0">Weight${state.edited.has(it.id) ? " · verified" : " · proxy"}</span>
            <span class="flex items-center gap-1.5 w-full">
              <input type="number" step="0.005" min="0" value="${weight}" class="weight-input w-full min-w-0 bg-transparent text-left md:text-left lg:text-right text-[13px] md:text-[13px] lg:text-sm font-semibold text-zinc-900 tabular-nums outline-none">
              <span class="text-[10px] md:text-[10px] lg:text-[11px] text-zinc-400 shrink-0">kg</span>
            </span>
          </label>
          <div class="flex flex-col md:flex-col lg:flex-row items-start md:items-start lg:items-center justify-between gap-1 md:gap-1 lg:gap-2 rounded-lg border border-zinc-200/80 bg-zinc-50 px-2.5 md:px-2.5 lg:px-3 py-2">
            <span class="font-mono-d text-[8px] md:text-[8px] lg:text-[9px] tracking-tight lg:tracking-[0.18em] uppercase text-zinc-400 shrink-0">Density</span>
            <span class="text-[13px] md:text-[13px] lg:text-sm font-semibold text-zinc-500 tabular-nums tracking-tight whitespace-nowrap">${m.density.toFixed(2)} <span class="text-[9px] md:text-[9px] lg:text-[10px] text-zinc-400 font-medium">g/cm³</span></span>
          </div>
          <div class="flex flex-col md:flex-col lg:flex-row items-start md:items-start lg:items-center justify-between gap-1 md:gap-1 lg:gap-2 rounded-lg border border-zinc-200/80 bg-zinc-50 px-2.5 md:px-2.5 lg:px-3 py-2">
            <span class="font-mono-d text-[8px] md:text-[8px] lg:text-[9px] tracking-tight lg:tracking-[0.18em] uppercase text-zinc-400 shrink-0">Est. vol</span>
            <span class="text-[13px] md:text-[13px] lg:text-sm font-semibold text-zinc-500 tabular-nums tracking-tight whitespace-nowrap">${volume.toLocaleString()} <span class="text-[9px] md:text-[9px] lg:text-[10px] text-zinc-400 font-medium">cm³</span></span>
          </div>
          <div class="flex flex-col md:flex-col lg:flex-row items-start md:items-start lg:items-center justify-between gap-1 md:gap-1 lg:gap-2 rounded-lg border border-zinc-200/80 bg-zinc-50 px-2.5 md:px-2.5 lg:px-3 py-2">
            <span class="font-mono-d text-[8px] md:text-[8px] lg:text-[9px] tracking-tight lg:tracking-[0.18em] uppercase text-zinc-400 shrink-0">Decomposes</span>
            <span class="text-[13px] md:text-[13px] lg:text-sm font-semibold text-zinc-500 tracking-tight whitespace-nowrap">${m.decompose}</span>
          </div>
            </div>
          </div>
          <div id="panel-environmental" class="${specTab === "env" ? "opacity-100" : "hidden opacity-0"} transition-all duration-300 flex flex-col space-y-3">
            ${envPanelHtml(it, state.recs[it.id])}
          </div>
        </div>
        ${tabsHtml(it)}` : ""}
      </div>
    </div>
  </article>`;
}

function renderItems() {
  const list = $("items-list");
  if (!state.items.length) {
    list.innerHTML = `<p class="text-xs text-zinc-400 px-1">Run a detection to populate the item inspector — every card links 1:1 to its bounding box.</p>`;
    // Keep the footer pill honest on the empty state too: without this the
    // early return skipped applyConfFilter and NEW SCAN left a stale
    // "3 objects (+2 filtered)" count next to "upload an image to begin".
    applyConfFilter();
    return;
  }
  list.innerHTML = state.items.map(cardHtml).join("");

  list.querySelectorAll(".item-card").forEach(card => {
    const id = +card.dataset.id;
    card.addEventListener("click", () => setActive(state.activeId === id ? null : id));

    card.querySelectorAll(".weight-input").forEach(inp => {
      inp.addEventListener("click", (e) => e.stopPropagation());
      inp.addEventListener("input", (e) => {
        const v = Math.max(0, parseFloat(e.target.value) || 0);
        if (v > 0) {
          state.weights[id] = v;
          state.edited.add(id);
          scheduleRefresh();
        }
      });
    });
    card.querySelectorAll(".process-tab").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        state.process[id] = btn.dataset.process;
        renderItems();
        const panel = list.querySelector(`[data-id="${id}"] .dmm-panel`);
        if (panel) gsap.fromTo(panel, { y: 8, opacity: 0 }, { y: 0, opacity: 1, duration: 0.35 });
      });
    });
    card.querySelectorAll(".spec-tab").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        switchSpecTab(card, id, btn.dataset.spectab);
      });
    });
  });

  applyConfFilter();   // fresh DOM — re-apply the confidence view filter
}

// Grid-recalculation loading state (Issue 2): a country switch or a debounced
// weight edit re-audits + re-ranks. A counter keeps the glass masks up while
// ANY refresh is in flight, so overlapping triggers (rapid country flips)
// can't let an early finally() lift the mask over a still-running refresh.
// Both refreshers swallow their own errors, so Promise.all always settles —
// the finally() guarantees the masks lift even on a network timeout/abort.
let gridRefreshInFlight = 0;
async function runGridRefresh() {
  if (!state.items.length) return;
  gridRefreshInFlight++;
  document.body.classList.add("is-recalculating-grid");
  setStatus("RECALIBRATING…", "warn");
  try {
    await Promise.all([refreshImpact(), refreshRecommendations()]);
  } finally {
    if (--gridRefreshInFlight === 0) {
      document.body.classList.remove("is-recalculating-grid");
      setStatus("READY");
    }
  }
}

let refreshTimer = null;
function scheduleRefresh() {
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(runGridRefresh, 650);
}

// Grow the environmental chart bars from 0 → target (the reveal animation).
function animateEnvBars(card) {
  card.querySelectorAll(".env-bar").forEach(bar => {
    const target = bar.dataset.target;
    bar.style.width = "0%";
    requestAnimationFrame(() => { bar.style.width = target + "%"; });
  });
}

// Progressive-disclosure sub-tab switch — pure DOM, ZERO re-render (0-lag).
// The choice is mirrored into state.specTab so a later renderItems() (weight
// edit / country flip / pathway change) rebuilds the card on the same tab.
// Outgoing panel hides instantly (no stacking / height jump); incoming
// cross-fades in; the chart bars re-grow when Environmental Impact opens.
function switchSpecTab(card, id, tab) {
  if ((state.specTab[id] || "specs") === tab) return;
  state.specTab[id] = tab;
  const specs = card.querySelector("#panel-specs");
  const env = card.querySelector("#panel-environmental");
  if (!specs || !env) return;
  const show = tab === "env" ? env : specs;
  const hide = tab === "env" ? specs : env;

  card.querySelectorAll(".spec-tab").forEach(b => {
    const on = b.dataset.spectab === tab;
    b.classList.toggle("bg-white", on);
    b.classList.toggle("text-zinc-900", on);
    b.classList.toggle("shadow-sm", on);
    b.classList.toggle("text-zinc-500", !on);
  });

  hide.classList.add("hidden", "opacity-0");   // instant hide — no layout stack
  show.classList.remove("hidden");
  requestAnimationFrame(() => {
    show.classList.remove("opacity-0");
    show.classList.add("opacity-100");
    if (tab === "env") animateEnvBars(card);
  });
}

function setActive(id) {
  state.activeId = id;
  renderItems();
  draw();
  if (id != null) {
    const card = document.querySelector(`.item-card[data-id="${id}"]`);
    if (card) {
      $("right-panel").scrollTo({ top: card.offsetTop - 24, behavior: "smooth" });
      const body = card.querySelector(".expand-body");
      if (body) gsap.fromTo(body, { y: 10, opacity: 0 }, { y: 0, opacity: 1, duration: 0.35, ease: "power2.out" });
    }
  }
}

/* ------------------------------ canvas ------------------------------ */

function getImgRect(W, H) {
  if (!state.img) return { x: 0, y: 0, w: W, h: H };
  const maxH = H * 0.92;
  const s = Math.min(W / state.imgW, maxH / state.imgH);
  const iw = state.imgW * s, ih = state.imgH * s;
  return { x: (W - iw) / 2, y: (H - ih) / 2, w: iw, h: ih };
}

function hitTest(e) {
  const r = canvas.getBoundingClientRect();
  const rect = getImgRect(r.width, r.height);
  const px = ((e.clientX - r.left) - rect.x) / rect.w * state.imgW;
  const py = ((e.clientY - r.top) - rect.y) / rect.h * state.imgH;
  // smallest box wins on overlap; confidence-filtered boxes are untouchable
  return state.items
    .filter(it => isVisible(it)
      && px >= it.bbox[0] && px <= it.bbox[2] && py >= it.bbox[1] && py <= it.bbox[3])
    .sort((a, b) => a.box_area_px - b.box_area_px)[0] || null;
}

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function draw() {
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (!W || !H) return;
  canvas.width = W * dpr; canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  ctx.fillStyle = "#f4f4f5";
  ctx.fillRect(0, 0, W, H);
  const rect = getImgRect(W, H);

  if (state.img) {
    ctx.save();
    ctx.shadowColor = "rgba(24,24,27,0.10)";
    ctx.shadowBlur = 24;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.restore();
    ctx.drawImage(state.img, rect.x, rect.y, rect.w, rect.h);
  } else {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
  }

  // fine blueprint grid, confined to the image area
  ctx.save();
  ctx.beginPath();
  ctx.rect(rect.x, rect.y, rect.w, rect.h);
  ctx.clip();
  ctx.strokeStyle = "rgba(24,24,27,0.045)";
  ctx.lineWidth = 1;
  for (let x = rect.x; x < rect.x + rect.w; x += 48) { ctx.beginPath(); ctx.moveTo(x + 0.5, rect.y); ctx.lineTo(x + 0.5, rect.y + rect.h); ctx.stroke(); }
  for (let y = rect.y; y < rect.y + rect.h; y += 48) { ctx.beginPath(); ctx.moveTo(rect.x, y + 0.5); ctx.lineTo(rect.x + rect.w, y + 0.5); ctx.stroke(); }
  ctx.restore();

  if (!state.img) return;
  const reveal = state.reveal;

  state.items.forEach(it => {
    if (!isVisible(it)) return;   // hidden by the confidence pill
    const active = state.activeId === it.id;
    const co2 = itemCo2(it);
    const color = co2Color(co2);
    const x = rect.x + (it.bbox[0] / state.imgW) * rect.w;
    const y = rect.y + (it.bbox[1] / state.imgH) * rect.h;
    const w = ((it.bbox[2] - it.bbox[0]) / state.imgW) * rect.w;
    const h = ((it.bbox[3] - it.bbox[1]) / state.imgH) * rect.h;

    ctx.globalAlpha = reveal;
    if (active) {
      ctx.fillStyle = hexA(color, 0.06);
      ctx.fillRect(x, y, w, h);
    }
    ctx.strokeStyle = active ? color : hexA(color, 0.65);
    ctx.lineWidth = active ? 2.5 : 1.5;
    ctx.strokeRect(x, y, w, h);

    // corner ticks
    const t = 12 * reveal;
    ctx.strokeStyle = color; ctx.lineWidth = active ? 3 : 2;
    [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]].forEach(([cx, cy, dx, dy]) => {
      ctx.beginPath(); ctx.moveTo(cx + dx * t, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + dy * t); ctx.stroke();
    });

    // label chip
    const text = `${it.display_name.toUpperCase()}  ${it.confidence.toFixed(2)}`;
    ctx.font = '600 10px "JetBrains Mono", monospace';
    const tw = ctx.measureText(text).width;
    const ly = y - 22 < 6 ? y + 6 : y - 22;
    ctx.fillStyle = active ? color : "#ffffff";
    ctx.fillRect(x, ly, tw + 14, 17);
    ctx.strokeStyle = hexA(color, 0.5); ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, ly + 0.5, tw + 13, 16);
    ctx.fillStyle = active ? "#ffffff" : color;
    ctx.fillText(text, x + 7, ly + 12);
    ctx.globalAlpha = 1;
  });
}

/* ------------------------------ boot ------------------------------ */

document.addEventListener("DOMContentLoaded", () => {
  canvas = $("detection-canvas");
  ctx = canvas.getContext("2d");
  new ResizeObserver(draw).observe(canvas);
  canvas.addEventListener("click", (e) => {
    const hit = hitTest(e);
    setActive(hit ? hit.id : null);
  });
  canvas.addEventListener("mousemove", (e) => {
    canvas.style.cursor = hitTest(e) ? "pointer" : "default";
  });

  buildCountrySelect();
  // Dropdown shell wiring (bound ONCE — rebuilds only replace the options).
  $("country-trigger").addEventListener("click", (e) => {
    e.stopPropagation();
    countryMenuOpen ? closeCountryMenu() : openCountryMenu();
  });
  document.addEventListener("click", (e) => {
    if (countryMenuOpen && !$("country-dd").contains(e.target)) closeCountryMenu();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeCountryMenu();
  });
  syncGeoUi();
  wireUpload();
  geolocate();
  draw();

  // Confidence pill: 0 ms client-side filtering — no network, pure view.
  const slider = $("conf-slider");
  const syncFill = () =>
    slider.style.setProperty("--fill", (parseFloat(slider.value) * 100) + "%");
  syncFill();
  slider.addEventListener("input", () => {
    state.minConf = parseFloat(slider.value);
    $("conf-value").textContent = state.minConf.toFixed(2);
    syncFill();
    applyConfFilter();    // grid cards collapse/reappear instantly
    updateTelemetry();    // totals self-audit against the visible set
    draw();               // boxes vanish/return on the canvas
  });

  // GSAP intro orchestration.
  const tl = gsap.timeline({ defaults: { ease: "power2.out" } });
  tl.from("header", { y: -16, opacity: 0, duration: 0.5 })
    .from(".gsap-panel-left", { x: -24, opacity: 0, duration: 0.55 }, "-=0.25")
    .from(".gsap-panel-right", { x: 24, opacity: 0, duration: 0.55 }, "<")
    .from(".gsap-telemetry, .gsap-card", { y: 20, opacity: 0, duration: 0.5, stagger: 0.09 }, "-=0.3")
    .from(".gsap-badge", { scale: 0.85, opacity: 0, duration: 0.45, ease: "back.out(2)", stagger: 0.12 }, "-=0.2");
});
