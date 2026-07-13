# CLAUDE.md

> Persistent project context for Claude Code. Read this fully before acting.
> This file is the single source of truth for the project's goals, architecture,
> conventions, and constraints. When something here conflicts with an assumption
> you would otherwise make, **this file wins**.

---

## 1. Project Identity

**Title:** Deep Learning Multi-Target Waste Detection and Carbon Impact Estimation Web Application

**Type:** Final Year Project (FYP). Code must be production-ready, modular, well-documented, and easy for an academic examiner to audit.

**One-line description:** A web app where a user uploads a real-world image containing one or more waste items; a **100% local Dual-Tower Hybrid pipeline** — a **specialized waste object detector** (YOLOv8-N fine-tuned on a blended universal waste corpus of wild litter + household recyclables) boxes ONLY trash-like objects, a context-aware square-padding layer normalizes each crop, a **classical-CV physics extractor** (Method B: Laplacian wrinkles + Canny edges → Plasticity Index ψ) profiles each patch, and a TrashNet-fine-tuned Vision Transformer names the material (7-class taxonomy, with ψ breaking ambiguous plastic-vs-glass calls) — identifies every item; carbon flows through a **dual-stage UX** — a blind, offline estimate **dynamically scaled by each bounding box's geometric pixel area** at upload, then a precision audit of user-verified weights priced by the **real-time Climatiq API** (country-scoped, cached, local-dummy fallback); and a **Decision Making Module (DMM)** forks every item into **3 parallel end-of-life simulations**, ranks them by ascending CO2e, and returns structured, expert-annotated disposal prescriptions per item.

**Author's environment:** Windows 11, PowerShell, local GPU = **NVIDIA GTX 1650 (4 GB VRAM)**. All ML choices must respect this 4 GB limit (inference of both towers fits; only *training* ever exceeded it, and training is retired).

---

## 2. ARCHITECTURE: The Dual-Tower Hybrid (Waste YOLO Detector + Method B + TrashNet ViT) — read this first

The project went through several pivots (custom training → single-stage YOLO-World →
two-stage with abstract anchors → two-stage vanilla YOLOv8 → two-stage RT-DETR-X →
COCO-80 YOLO26-seg → blended-waste segmenter → box-detector regression (v3.2) →
v3.5 dual-stage carbon UX + DMM → **this, v3.6**: the v3.2 towers and the v3.5
numeric engines unchanged, with the DMM's text layer upgraded to an OPTIONAL
child-friendly, country-localized LLM generation pipeline over free
OpenAI-compatible endpoints, deterministic local grid as the ever-present
fallback). The paradigm is
**edge-native and 100% local**: two frozen local models plus a classical-CV layer,
no cloud inference. Localization and classification are **decoupled**, and each tower
plays its architectural strength — **CNN spatial localization for WHERE, ViT global
self-attention for WHAT, classical physics for the ties**:

```
Stage 1 — LOCALIZATION (waste detection)         Stage 2 — CLASSIFICATION (material)
SPECIALIZED waste OBJECT DETECTOR                TrashNet ViT (supervised ViT-B/16)
(YOLOv8-N fine-tuned on a blended                edwinpalegre/ee8225-group4-
universal waste corpus: wild litter +             vit-trashnet-enhanced (98.2% val acc)
household recyclables; GitHub gianluca-          native labels (verified id2label):
sposito/YOLO-Waste-Detection, MIT)         ──►   biodegradable cardboard glass metal
models/yolov8n-waste-det.pt (~6 MB)        224²   paper plastic trash→"general rubbish"
latent space ONLY knows waste — background patch  judges texture/gloss/material
rarely fires; nano backbone = max fps                     │
5 coarse labels (Glass/Metal/Paper/                       ▼
Plastic/Waste) noted as `located_as`,             ψ TIE-BREAK: ambiguous plastic-vs-
NEVER trusted for identity                        glass calls corrected by physics
conf=0.15 (recall-first) + CLASS-AGNOSTIC
NMS (iou=0.45): one box per physical object
         │
         └── Processing layer (anchored directly on box.xyxy):
             (a) CONTEXT-AWARE SQUARE PADDING:
                 box → +15% context margin → square pad (neutral gray, NO stretch) → 224x224
             (b) METHOD B — CLASSICAL CV PHYSICS EXTRACTOR (OpenCV):
                 Laplacian texture variance (micro-wrinkles) + Canny edge density
                 → Plasticity Index ψ ∈ [0,1]  (ψ ≥ 0.5 plastic-like, < 0.5 glass-like)
```

**Engineering justification:**
- **Why a specialist waste detector for Stage 1 (the v3.2 reasoning).** The COCO-80
  generation boxed *everything* COCO-shaped — tables, people, buses — and the ViT
  (which has no "not waste" class) force-classified that background noise into
  materials. A blended-corpus waste model filters environmental noise
  *architecturally*, and its generic "Waste" class catches amorphous litter with no
  COCO look-alike. Choosing plain **object detection** over instance segmentation
  then buys three things on edge hardware: (1) no per-instance mask decoding —
  markedly higher frame-rate throughput on the nano backbone (~6 MB vs ~55 MB);
  (2) no mask-suppression artifacts in high-density waste layouts, where
  overlapping mask NMS can silently merge or drop tightly packed instances —
  rectangles keep every instance distinct; (3) a simpler, fully deterministic
  geometric weight modifier for the carbon formula. Verified live: 5 labels,
  genuine fine-tune, MIT.
- **Why box area as the volume/mass proxy (the academic core feature, adapted).**
  With `result.masks` natively `None`, the geometric box area
  `(x2-x1)*(y2-y1)` becomes the physical volume/mass proxy. A rectangle
  over-covers a tight object contour by ~1.6x (measured mask/box fill factor
  ~0.6 on live waste samples), so the calibration constant γ is scaled
  accordingly (5000 → **8000**) to keep carbon magnitudes comparable across
  locator generations — bigger litter, bigger footprint — until Step 5 replaces
  the proxy with user-entered weights.
- **Why a supervised ViT for Stage 2.** Material recognition on an isolated patch is a
  *global* problem — gloss, texture and colour distributed across the whole crop, not
  localized parts — which suits ViT self-attention. The chosen checkpoint is
  fine-tuned on TrashNet-enhanced (98.17% val accuracy, Apache-2.0): it has *seen
  thousands of real waste items*. Its native 7 labels map 1:1 onto the system
  taxonomy (only `trash` → `general rubbish`).
- **Why Method B exists (transparency confusion).** Glass and thin clear plastic are
  the ViT's hardest boundary — both transparent, similar colour. Classical spatial
  filtering separates them by *physics*, not learned appearance, with zero retraining:
  crushed/disposable plastic exhibits high-frequency micro-wrinkles
  (`cv2.Laplacian` variance high) and dense thin contours (`cv2.Canny` density high),
  while pristine glass is structurally smooth with broad refractive edges. Both cues
  squash to [0,1] and average into ψ. Crucially it is only a **tie-breaker**: clear
  ViT verdicts are never overridden.
- **Why the padding layer exists.** ViTs eat fixed 224x224 grids. Naively resizing a
  long thin crop (a wire, a receipt) squishes its aspect ratio and destroys the very
  texture cues the ViT reads. The processing layer expands the box by 15% (context),
  pads the short side with neutral gray to a perfect square (no distortion), THEN
  resizes — every patch arrives at the ViT undistorted and context-rich.
- **The segmenter's label is a proposal, not a verdict.** Whether Stage 1 calls an
  instance "Plastic" or "Waste" is irrelevant — the label is discarded for identity
  and carried only as the `located_as` diagnostic. Stage 2 (+ ψ) alone decides
  material.
- **Recall-first localization is safe** because Stage 2 re-scores every instance:
  a spurious mask gets a low, flat ViT distribution, and per-item certainty shown to
  the user comes from Stage 2, not Stage 1.
- **Known limitation (document honestly in the report):** the specialist checkpoint
  has no published metrics (community model, ~4.1k training images) — recall on
  unusual items may trail bigger baselines, and a box area *includes background*
  so it is a coarser size proxy than a mask. Mitigations: the low Stage-1
  threshold; γ recalibrated for rectangular over-coverage; and both prior locators
  stay one env-var away (`models/yolov8m-seg-trash.pt` masks,
  `yolo26x-seg.pt` COCO-80) for A/B comparison (report material). ψ's calibration
  constants are heuristics — the payload carries every physics reading +
  `tiebreak_applied` so corrections are fully auditable.

**Method B — Plasticity Index & tie-break rule (formulas):**

```
ψ = 0.5·min(1, LaplacianVar/500) + 0.5·min(1, CannyEdgeDensity/0.10)

Tie-break fires iff  top-2 = {plastic, glass}  AND  |p₁ − p₂| < 0.15:
    winner = plastic if ψ ≥ 0.5 else glass
    if winner ≠ ViT argmax → the two labels swap scores (rank correction;
    probability mass unchanged), payload flags tiebreak_applied = true
```

Constants live in `detection_service.py` (`_LAPLACIAN_REF=500`,
`_EDGE_DENSITY_REF=0.10`, `PLASTICITY_TIEBREAK_MARGIN=0.15`).

**Box-Area Dynamic Carbon Scaling (academic core feature):**

```
Box Area (px²)                = (x₂ − x₁) × (y₂ − y₁)
Final Carbon Impact (kg CO2e) = Base Material Coefficient x (Box Area / γ)

γ (PIXEL_AREA_GAMMA) = 8000.0   — reference pixel density, carbon_service.py
  (recalibrated from the mask-era 5000: rectangles over-cover tight contours
   by ~1.6x, i.e. 5000 / 0.625 = 8000, keeping magnitudes comparable)
```

A box of exactly γ pixels scores 1x its base coefficient; larger boxes scale up
proportionally. Implemented in `carbon_service.estimate_dynamic_impact(label, area)`;
the per-item payload carries BOTH the base coefficient (`carbon_factor_kg_per_kg`)
and the scaled result (`estimated_carbon_kg`).

**Sequential data payload lifecycle:**

```
Upload (multipart image)
  → POST /api/predict (thin controller: validate, save, delegate)
    → detection_service.analyze_waste_pipeline(image_path)
       1. Stage 1: specialist waste detector (models/yolov8n-waste-det.pt,
          auto-fetched if missing), predict (conf 0.15 + agnostic NMS, iou 0.45)
          → per instance: bbox via box.xyxy (clamped), box_confidence,
            located_as (Glass/Metal/Paper/Plastic/Waste — diagnostic),
            box_area_px = (x2-x1)*(y2-y1)
       2. Processing layer (a): PIL, anchored on box.xyxy — +15% context pad
          → square pad (114,114,114) → LANCZOS resize to exactly 224x224
       3. Stage 2: classification_service.classify_crops(patches)
          → per patch: full 7-class ViT softmax (top_k=all), model's "trash"
            label mapped → "general rubbish", sorted desc
       4. Processing layer (b) — Method B: extract_classical_physics_features
          (patch) → {laplacian_variance, edge_density, ψ}; then
          _apply_plasticity_tiebreak(scores, physics) corrects an ambiguous
          plastic-vs-glass argmax (see §2 formulas)
       5. Carbon mapping: carbon_service.get_carbon_factor(material)  → base
          coefficient; carbon_service.estimate_dynamic_impact(material, box_area)
          → base x (box area / γ)
       6. Consolidated JSON:
          items[]: { id, class_name (material), display_name, confidence,
                     box_confidence (Stage 1), located_as (diagnostic),
                     bbox, box_area_px, material_scores[7],
                     physics{laplacian_variance, edge_density, plasticity_index,
                     tiebreak_applied}, carbon_factor_kg_per_kg, estimated_carbon_kg }
          image: { width, height (+ filename, url added by the route) }
  → Frontend (CarbIQ SPA, Step 7): canvas BOUNDING-BOX overlay
    (ctx.strokeRect + corner ticks + label chips, rectangular hit-testing,
    smallest box wins on overlap, CO2e-tiered colouring) linked 1:1 to the
    editable item-card grid via the echoed ids (bi-directional focus);
    cards carry ψ/box diagnostics, weight inputs (debounced re-audit) and
    the ranked DMM tab panel

Follow-up JSON calls (fed by the /predict payload; Step-7 UI wires them up):
  → POST /api/calculate-impact  {items:[{id?, material,
                                 weight_kg? | box_area_px?}], country?}
      STAGE B precision audit — client grid `id`s echoed back VERBATIM
      (split-screen canvas↔grid bi-directional sync); missing weight →
      box_area_px/γ pixel-proxy substitution (weight_source labelled);
      live Climatiq factor per unique (material, country, api_key) via
      cached 1-kg probes, scaled locally; blank key → dummy factors;
      blank/omitted country → Climatiq global dataset
  → POST /api/recommend  {items:[{material, weight_kg? | box_area_px?}],
                          country?}
      DECISION MAKING MODULE — forks each item into 3 taxonomy-branched
      end-of-life paths in parallel, ranks ascending CO2e (rank 1 = Optimal),
      returns status tags + verdicts + pros/cons per path; text fields come
      from the v3.6 LLM pipeline when LLM_API_KEY is set (child-simple,
      localized to `country`) else the local grid — provider labelled
      llm_enriched | local_knowledge_base | local_fallback
```

**Hard rules:**
- **Do NOT delete any legacy training assets** — `ml/` scripts, notebooks, configs,
  runs, and their tests are retained for the FYP report (archive: §12). They are
  bypassed, never imported by `app/`.
- Stage 1 stays vocabulary-free at call time: NO `set_classes()`, no prompt lists —
  that approach is archived (§11) after failing empirically. `located_as` is
  diagnostic-only; never surface it as the material or branch on it.
- The Stage-1 predict call passes `conf` **plus CLASS-AGNOSTIC suppression
  (`agnostic_nms=True, iou=0.45`)**. Default per-class NMS let one physical object
  survive as overlapping boxes under different coarse labels (e.g. "Paper" +
  "Waste"), which Stage 2 then named identically — duplicate frames on one item.
  Agnostic NMS merges candidates across classes: one box per object. The NMS-free
  A/B baselines (YOLO26, RT-DETR) accept and ignore these args. No further knobs.
- Method B is a **tie-breaker, not a classifier**: it may only ever reorder an
  ambiguous plastic-vs-glass top-2; never let ψ overrule a clear ViT verdict or
  touch other materials.
- The pipeline is **100% local at inference time** — model downloads happen once;
  no cloud inference calls. (The Step-5 Climatiq call is a data lookup, not ML.)

### Locked decisions (active pipeline)

| Topic | Decision | Reason |
|---|---|---|
| Python version | **3.11.x** (NOT 3.13) | 3.13 lacks prebuilt wheels for numpy/torch on Windows; 3.11 has wheels for everything. |
| Virtualenv (Windows) | `py -3.11 -m venv .venv` then `.\.venv\Scripts\Activate.ps1` | PowerShell does not support `&&` or `source`. Never give bash-only commands. |
| Ultralytics version | **`ultralytics==8.4.90`** (8.3.x predates YOLO26 — never downgrade) | Loads both the specialist v8-seg weights and the YOLO26 A/B baseline. |
| Stage 1 locator | **Specialist waste OBJECT DETECTOR** `models/yolov8n-waste-det.pt` (GitHub `gianlucasposito/YOLO-Waste-Detection`, MIT; auto-fetched if missing), 5 native labels, **NO `set_classes()`** | Fine-tuned on a blended universal waste corpus: only fires on waste-like objects; nano backbone = edge frame-rate; no mask-decoding overhead or mask-NMS merge artifacts in dense layouts. Labels discarded (kept as `located_as`). |
| Stage 1 A/B alternatives | `MODEL_PATH=models/yolov8m-seg-trash.pt` (v3.1 masks) or `yolo26x-seg.pt` (COCO-80) | One env-var away for report comparisons; the loader dispatches transparently. |
| Processing layer (a) | +15% context pad → square pad (neutral 114-gray) → 224x224 LANCZOS | ViT-native input without aspect-ratio distortion; context preserved. |
| Processing layer (b) | **Method B** — `cv2.Laplacian` variance + `cv2.Canny` density → ψ; refs 500 / 0.10; tie-break margin 0.15 | Classical physics separates transparent glass vs plastic with zero retraining; tie-breaker only, fully audited in the payload. |
| Stage 2 classifier | **`edwinpalegre/ee8225-group4-vit-trashnet-enhanced`** via HF `transformers` image-classification pipeline | Supervised ViT-B/16, TrashNet-enhanced, 98.17% val acc, Apache-2.0; native 7 labels (verified) map 1:1 onto the taxonomy. |
| Label mapping | model `trash` → system `general rubbish`; all other labels identity | `MODEL_LABEL_TO_MATERIAL` in classification_service; tests enforce full coverage. |
| Stage 1 tuning | `conf=0.15` default (per-request override) + **class-agnostic NMS** (`agnostic_nms=True, iou=0.45` — `_NMS_IOU` in `detection_service.py`) | Recall-first (Stage 2 carries per-item certainty); per-class default NMS kept duplicate cross-class boxes on one object — agnostic suppression guarantees one box per physical object (the NMS-free A/B baselines ignore the args). |
| Carbon scaling | `estimated_carbon_kg = base x (box_area_px / γ)`, **γ = 8000** (recalibrated from 5000 for rectangular over-coverage, fill factor ~0.6) | Box area as volume/mass proxy until Step 5's real weights. |
| Carbon provider | **Climatiq** (live, Step 5) with the **dummy per-kg coefficients** in `carbon_service.py` as the ever-present blank-key fallback | Dual-stage UX: blind local proxy at upload, audited live factors on user verification; the app never requires a key. |
| Disposal-path matrix (DMM) | `carbon_service.DISPOSAL_METHOD_FACTORS` — 7 materials × 3 routes of NET kg CO2e/kg code constants; **credits are NEGATIVE** (offsets) | Deterministic, offline, auditable (every factor echoed in the payload); the ranking must never block on the network — live regional factors stay Module 2's audit concern. |
| Recommendation engine | **Decision Making Module (DMM)** — rule-based 3-path parallel carbon simulation + ascending-CO2e ranking (`recommendation_service.py`); numbers are ALWAYS local | Deterministic engine is the gradeable default; must work fully with no LLM key. |
| DMM text layer (v3.6) | **Child-Friendly, Country-Localized LLM Generation Pipeline** — one batched strict-JSON call to any free OpenAI-compatible endpoint (`LLM_API_URL`/`LLM_MODEL`, Groq default) rewrites ONLY verdict/pros/cons (1–2 sentences, ≤25 words, zero jargon, localized to `country`); local `EXPERT_KNOWLEDGE` grid (same hyper-simple register) is the default and the atomic fallback | Free-tier friendly, provider-agnostic, zero new deps (`requests`); any LLM failure degrades to `local_fallback` — recommendations never 502. |
| Backend | **Flask** (app factory + blueprints + services) | Thin controllers, logic in services. |
| Frontend | HTML5 + Tailwind (CDN) + vanilla JS (ES6+, Fetch) + GSAP 3 (CDN); Inter/JetBrains Mono self-hosted under `app/static/assets/` | **CarbIQ SPA (Step 7, live)** — design-compiled dashboard: detection canvas ↔ item grid synced via echoed ids, IP-geolocated country select, dual-stage carbon telemetry, ranked DMM panels; `/carbon-lab` stays as the dependency-free API tester. |
| Database | SQLite (optional, for scan history) | Lightweight, local, file-based. |

---

## 3. Tech Stack

- **Language:** Python 3.11
- **ML / Stage 1:** PyTorch, Ultralytics (`ultralytics==8.4.90`) running the
  **specialist waste object detector** `models/yolov8n-waste-det.pt` (~6 MB;
  GitHub `gianlucasposito/YOLO-Waste-Detection`, auto-fetched via `requests`
  if missing; YOLOv8-N fine-tuned on a blended universal waste corpus, MIT)
- **ML / Stage 2:** Hugging Face `transformers==4.44.2` image-classification pipeline
  with **`edwinpalegre/ee8225-group4-vit-trashnet-enhanced`** (supervised ViT-B/16,
  ~343 MB, HF cache on first load)
- **Image processing:** Pillow (context-aware square padding layer), OpenCV
  (`opencv-python-headless` — Method B: Laplacian variance + Canny edge density
  → Plasticity Index ψ), NumPy
- **Backend:** Flask 3, Flask-SQLAlchemy, python-dotenv, requests, pydantic
- **Frontend:** HTML5, Tailwind CSS (CDN), vanilla JavaScript (Fetch API), GSAP 3 (CDN) — the CarbIQ SPA (Canvas 2D `strokeRect` box rendering, self-hosted Inter/JetBrains Mono); `/carbon-lab` API tester stays dependency-free
- **Database:** SQLite (via SQLAlchemy)
- **External APIs:** Climatiq (carbon, Step 5) — Carbon Interface kept as alternate adapter
- **Optional:** any free OpenAI-compatible LLM endpoint (Groq / OpenRouter / Gemini compat / local Ollama) for the v3.6 DMM text layer — plain `requests`, no SDK
- **Testing:** pytest, pytest-mock (both model towers mocked; no network/GPU in tests)
- **Serving (prod):** gunicorn
- **GPU:** optional CUDA build of torch (`cu121`); `INFERENCE_DEVICE` drives BOTH towers

---

## 4. Repository Layout

```
waste-detection-app/
├── .env.example          # template of required env vars (safe to commit)
├── .env                  # real secrets (gitignored — never commit)
├── config.py             # class-based config: Dev / Prod / Testing
├── run.py                # entry point: python run.py
├── requirements.txt
├── app/
│   ├── __init__.py       # create_app() application factory
│   ├── extensions.py     # db = SQLAlchemy() (unbound)
│   ├── blueprints/
│   │   ├── main/routes.py    # HTML page routes (serves the Test Brain page)
│   │   └── api/routes.py     # JSON API: /predict, /calculate-impact, /recommend, /health
│   ├── services/
│   │   ├── detection_service.py        # ORCHESTRATOR: Stage 1 segment + padding layer + compose
│   │   ├── classification_service.py   # Stage 2: TrashNet ViT material classifier
│   │   ├── carbon_service.py           # dual-stage carbon engine: γ proxy + Climatiq audit + disposal matrix
│   │   └── recommendation_service.py   # Module 3 DMM: 3-path simulation + ranking + knowledge base
│   ├── models/scan.py     # SQLite model for optional scan history
│   ├── schemas/           # pydantic contracts: detection.py, carbon.py, recommendation.py
│   ├── utils/errors.py    # ApiError + register_error_handlers()
│   ├── static/            # uploads/ + assets/ (self-hosted woff2 fonts)
│   └── templates/         # index.html (CarbIQ SPA) · carbon_lab.html (API tester)
├── ml/                   # LEGACY training workspace — KEEP, DO NOT DELETE (FYP report)
├── models/               # legacy exported weights location (best.pt, gitignored)
├── tests/
│   ├── conftest.py       # pytest fixtures (app, client)
│   ├── test_api.py  test_detection.py  test_classification.py  test_carbon.py
│   ├── test_predict_endpoint.py  test_calculate_impact.py  test_recommendation.py
│   └── test_prepare_dataset.py  test_restratify_dataset.py  test_train.py   # legacy, keep green
└── docs/                 # FYP report material
```

**Hard rules:**
- `app/` must never import anything from `ml/`. The legacy training code is inert.
- All weights are gitignored (`*.pt` in the project root, HF cache lives outside the repo).

---

## 5. LOCKED: Stage Vocabularies & Label Mapping

### Stage 1 — the detector's NATIVE waste vocabulary (5 coarse labels)

Stage 1 runs the specialist checkpoint's own classes — verified live from the
weights: `Glass, Metal, Paper, Plastic, Waste`. There is no `LOCALIZER_CLASSES`
list and no `set_classes()` call — **do not reintroduce them** (anchor prompting is
archived in §11 after failing empirically). What matters:

- The model was fine-tuned on waste imagery only (a blended corpus of wild litter
  + household recyclables), so it boxes trash-like objects and largely ignores
  background furniture/floors/plants. Its generic `Waste` class catches amorphous
  litter with no rigid shape.
- The label each instance fired as is carried in the payload as **`located_as`** —
  strictly a diagnostic for auditing ("why does this box exist?"). Identity comes
  from Stage 2 (+ the ψ tie-break); the pipeline never branches on `located_as` —
  even though the detector's labels *look* like materials, they are coarse,
  unvalidated, and NOT the verdict.
- The predict call passes `conf` plus class-agnostic suppression
  (`agnostic_nms=True, iou=0.45`): the 5 coarse classes overlap on real objects
  (one item can fire as both "Paper" and "Waste"), and per-class default NMS
  kept both boxes — agnostic NMS collapses them to one box per physical object.

### Stage 2 — system taxonomy (`classification_service.MATERIAL_CLASSES`)

```python
MATERIAL_CLASSES = [
    "biodegradable", "cardboard", "glass", "metal",
    "paper", "plastic", "general rubbish",
]
```

The **7-class output taxonomy** of the whole system. The ViT's NATIVE labels
(verified live from the checkpoint: `biodegradable, cardboard, glass, metal, paper,
plastic, trash`) resolve onto it through `MODEL_LABEL_TO_MATERIAL`:

| Model label | System material |
|---|---|
| `trash` | `general rubbish` |
| everything else | identity |

Rules:
- The raw material string is the **system-wide join key**: `DISPLAY_NAMES` (UI),
  `carbon_service.DUMMY_CARBON_FACTORS` + `DISPOSAL_METHOD_FACTORS`, and the DMM's
  `DISPOSAL_PATHS` / `EXPERT_KNOWLEDGE` are all keyed on it.
  `tests/test_detection.py::test_taxonomy_lockstep` +
  `tests/test_classification.py::test_every_native_vit_label_lands_in_the_taxonomy` +
  `tests/test_recommendation.py::test_factor_matrix_and_knowledge_base_cover_every_path`
  enforce alignment — when anything changes, update all of them together.
- `"general rubbish"` is the catch-all verdict — it must always exist and always have
  a carbon coefficient.
- The UI shows display names ("general rubbish" → "General Rubbish"), never raw strings.
- The legacy six ALL-CAPS dataset classes in `config.py:CLASS_NAMES` belong to the
  archived training era (§12) — serving code never reads them.

---

## 6. Module Specifications

### Module 1 — AI Multi-Target Waste Detection (Dual-Tower Hybrid, 100% local)
- **Goal:** detect + classify multiple waste items in cluttered, real-world images —
  including crushed/deformed and tightly packed items — with **no training of any
  kind** and no cloud inference.
- **Input:** user-uploaded image (complex background, single or multiple targets).
- **Output:** array of `{ class_name, display_name, confidence, box_confidence,
  located_as, bbox:[x1,y1,x2,y2], box_area_px, material_scores[7],
  physics{...}, carbon_factor_kg_per_kg, estimated_carbon_kg }`.
- **Implementation:**
  - `detection_service.analyze_waste_pipeline(image_path, conf=None)` orchestrates:
    `_locate_objects` (Stage 1: boxes via `box.xyxy`, clamped; geometric
    `box_area_px`) → `_prepare_patches` (processing layer a) →
    `classify_crops` (Stage 2) → `extract_classical_physics_features` +
    `_apply_plasticity_tiebreak` (processing layer b / Method B) → carbon
    annotation (base + dynamic).
  - Boxes clamped to image bounds; sub-2px boxes skipped; `box_area_px` is the
    area of the CLAMPED rectangle.
  - `_load_model` resolution: existing file → load; missing but registered in
    `_WEIGHT_SOURCES` (specialist weights, via direct URL or the HF hub) →
    download + copy into place; bare official name → Ultralytics auto-download.
    "rtdetr" filenames load via the `RTDETR` class (archived pivot compat);
    everything else via `YOLO`.
  - Both towers are cached singletons (`lru_cache`); heavy imports stay lazy so tests
    import services freely.
  - Stage-1 threshold: `CONFIDENCE_THRESHOLD` (default **0.15**, recall-first);
    per-request `conf` form-field override (clamped to [0.01, 1.0]). The predict
    call adds class-agnostic NMS (`agnostic_nms=True, iou=0.45`) — the
    duplicate-box guard. Device from `INFERENCE_DEVICE` for both towers.
  - All failures surface as `ApiError`; empty detections are a valid result.
- **Hardware:** CPU works (well under a second per image for the nano detector);
  GTX 1650 handles both towers' *inference* trivially (v8-N ~0.05 GB + ViT-B/16
  ~0.35 GB — far inside the 4 GB budget).
- **First-run downloads:** `models/yolov8n-waste-det.pt` (~6 MB, GitHub) + the
  TrashNet ViT (~343 MB, HF cache). Internet needed once.

### Module 2 — Carbon Impact Estimation (dual-stage UX, External API + box-area scaling, NO ML) — DONE (Step 5, dual-stage locked in v3.5)
- **STAGE A — blind estimate (photo upload, offline-safe):** the instant a photo
  is analysed, `estimate_dynamic_impact(label, box_area_px)` prices every
  instance locally: base local factor × (clamped box area / γ), γ = 8000 —
  deliberately LOCAL-only and deterministic so `/predict` never blocks on the
  network; it is the no-weight proxy in the detection payload.
- **STAGE B — precision audit (user verification + country alignment):**
  `POST /api/calculate-impact` accepts user-corrected real weights (kg) and an
  optional ISO 3166-1 alpha-2 `country`. When `CLIMATIQ_API_KEY` is set the
  local dummies are bypassed: per-kg factors come live from the Climatiq
  estimate endpoint (`https://api.climatiq.io/data/v1/estimate`, bearer auth,
  10 s timeout) as 1-kg probes cached via **`lru_cache(maxsize=64)` on the
  unique (material, country, api_key) tuple** (inputs normalised pre-cache:
  material stripped/lower-cased, country stripped/upper-cased); weight
  scaling then happens locally, keeping upstream request density minimal
  (one call per unique factor, not per item). Materials map to **ORDERED
  candidate activity ids** via `CLIMATIQ_MATERIAL_MAP` — a BEIS id
  (GB-scoped) plus an EPA id (US-scoped) per material, all 14 verified live
  2026-07-14; `MATERIAL_TO_CLIMATIQ_ACTIVITY` survives as the derived
  primary-id view. The estimate selector has NO category/sector fields (those
  are catalogue search filters) — the plural translation ("metal" →
  "metals"/"mixed_metals") lives inside the activity id. A region miss
  (strictly Climatiq `error_code no_emission_factors_found`) advances
  through the candidate ids and only after ALL miss retries unscoped
  (global); any other upstream error fails loudly with the API's own
  message, never silently (**operator note:** confirm/adjust ids in the
  Climatiq Data Explorer for your data plan).
- **v3.5 UX mechanics (split-screen grid + geolocation):**
  - *Item `id` echo:* each request item may carry the client's integer `id`
    (the /predict item id keying the canvas box ↔ editable grid row); the
    response echoes it back VERBATIM per item (never renumbered server-side)
    so the frontend performs instant bi-directional focus tracking on weight
    edits. Provided ids must be unique (schema-enforced 400 otherwise).
  - *Pixel-proxy weight substitution:* `weight_kg` is now OPTIONAL — when
    absent, `box_area_px / γ` (clamped geometric box area) becomes the
    effective weight via the shared `carbon_service.resolve_effective_weight`
    (the same helper the DMM uses); at least one size signal is required and
    every response item labels `weight_source`
    (`user_weight` | `box_area_proxy`) plus the effective `weight_kg` priced.
  - *Graceful country defaulting:* `country` typically arrives as the
    frontend's IP-geolocated default and region-scopes the live factors;
    omitted, blank or whitespace values coerce to None → the region selector
    is omitted entirely and Climatiq resolves against its global dataset.
- **Fallback path (always available):** blank key → `DUMMY_CARBON_FACTORS`
  (biodegradable 0.57, cardboard 0.94, glass 0.85, metal 4.50, paper 1.09,
  plastic 3.10, general rubbish 1.20). The app boots and all tests pass with
  no key; every response labels its `source`/`provider`
  (`climatiq` | `local_dummy` | `mixed`).
- **Endpoint contract:** body
  `{items:[{id?, material, weight_kg?, box_area_px?}], country?}` validated by
  `schemas/carbon.CalculateImpactRequest` (weights in (0, 1000] kg, area ≤
  1000·γ, ≤100 items, unique ids, ≥1 size signal per item); returns per-item
  `{id, material, weight_kg (effective), weight_source,
  carbon_factor_kg_per_kg, co2e_kg, source}` and the aggregate `total_co2e_kg`.
- **Module 3 factor side:** `DISPOSAL_METHOD_FACTORS` (7 materials × 3 routes,
  NET kg CO2e/kg, credits negative) + `estimate_disposal_impact(material,
  method, weight_kg)` live here too — local-only, app-context-free, thread-safe
  (the DMM fans them out in parallel).
- Error handling: invalid/missing weight and unknown material → 400; Climatiq
  auth/timeout/network/shape problems → 502 with a user-facing message —
  all via `ApiError`, nothing 500s silently (no naked stack traces).

### Module 3 — Recommendation System: the DECISION MAKING MODULE (DMM) — DONE (Step 6; v3.6 Child-Friendly, Country-Localized LLM Generation Pipeline)
- **Goal:** convert the carbon engine's quantitative data into qualitative,
  ORDERED prescriptions — not one default disposal answer but a ranked
  comparison of realistic end-of-life choices, with expert commentary.
- **Multi-path parallel carbon simulation:** each incoming item forks into
  **3 end-of-life simulations evaluated in parallel**
  (`ThreadPoolExecutor(max_workers=3)` fan-out onto the carbon engine),
  branched by taxonomy:
  - dry recyclables (`plastic, glass, metal, cardboard, paper`) →
    `recycling | incineration | landfill`
  - organics (`biodegradable`) → `composting | anaerobic_digestion | landfill`
  - residual (`general rubbish`) → `material_recovery | incineration | landfill`
- **Sorting engine & ranking core:** the 3 CO2e outputs are sorted **ascending**
  (lowest footprint / deepest negative offset wins; ties fall back to method
  name so ranking is fully deterministic): Rank 1 = Optimal green path,
  Rank 2 = Acceptable, Rank 3 = Warning (worst-case baseline).
- **Local knowledge grid (v3.6 hyper-simple register):** `EXPERT_KNOWLEDGE`
  (7×3 matrix) attaches plain-language `environmental_pros` /
  `environmental_cons` to every path — 1–2 punchy sentences, ≤25 words,
  child-readable, grounded facts (tests enforce the word budget); the
  `encouraging_verdict` is composed at runtime from the SORTED outcome
  (rank + kg CO2e saved/added), so recalibrated factors can reshuffle ranks
  without the copy drifting. Per-path payload:
  `{ method, method_display, rank, status_tag, carbon_factor_kg_per_kg,
  carbon_impact_kg, encouraging_verdict, environmental_pros,
  environmental_cons }`.
- **v3.6 LLM text layer (optional, free-provider):** when `LLM_API_KEY` is
  set, ONE batched call per request to the OpenAI-compatible chat-completions
  endpoint at `LLM_API_URL` (free tiers: Groq — default, `LLM_MODEL`
  `llama-3.3-70b-versatile` — OpenRouter, Gemini compat, or fully local
  Ollama) rewrites the three literary fields per path. The
  `LLM_SYSTEM_PROMPT` enforces the "Hyper-Simple & Country-Aware" standard:
  ≤25 words per field, jargon banlist (no "carbon-negative"/"offset"/"CO2e"),
  everyday impact imagery, verdicts that weave in the exact numbers, and
  pros/cons grounded in the request `country`'s reality (beaches/landfills/
  power grid) — or universal "global average" text when no country is given.
  The LLM sees the numbers READ-ONLY (strict-JSON in/out, lenient fence
  parsing); replacements are staged and applied atomically only after full
  7×3-coverage validation. Fast transient failures (429 bursts, 5xx "model
  overloaded", network hiccups, bad output) are RETRIED up to 3 attempts
  with short backoff (read timeouts are not — that budget is spent); only
  after that does the layer log a warning and serve the local grid —
  `provider` labels the outcome:
  `llm_enriched` | `local_knowledge_base` (no key) | `local_fallback`.
- **Weight resolution (dual-stage aware):** a user-verified `weight_kg` (Stage B)
  always wins; otherwise `box_area_px / γ` (the Stage-A blind proxy — the same
  calibration `/predict` uses), via the SHARED
  `carbon_service.resolve_effective_weight` helper that
  `/api/calculate-impact` also runs. At least one is required; the payload
  labels `weight_source` (`user_weight` | `box_area_proxy`).
- **Endpoint:** `POST /api/recommend` — body
  `{items:[{material, weight_kg?, box_area_px?}], country?}` validated by
  `schemas/recommendation.RecommendRequest` (≤100 items, weight (0, 1000] kg,
  area ≤ 1000·γ, blank country → None/global); returns per-item ranked
  `recommendations[3]` + `best_method` + `max_saving_kg`, an aggregate
  `summary` (optimal-vs-worst totals — may be negative thanks to offsets),
  the echoed `country` and the `provider` tag.
- **Numeric core stays 100% local & deterministic:** the simulation, factors
  and ranking never touch the network, need no key and no app context
  (thread-pool safe) — the LLM layer is the module's ONLY network touchpoint,
  runs once per request in the request thread, may only rewrite text, and can
  never block, reorder or 502 the ranking. Live regional FACTORS remain
  Module 2's audit concern. Honest GHG-only lens: inert landfilled plastic
  out-scores incineration on pure CO2e (rank 2), and the cons text carries
  the 400-year microplastic caveat the number cannot see (report talking
  point).

### Module 4 — Web Application: the CarbIQ SPA — DONE (Step 7)
- **`templates/index.html` — the CarbIQ dashboard** (a design-tool layout
  compiled to dependency-light vanilla JS: Tailwind CDN + GSAP 3 CDN + Fetch,
  fonts self-hosted from `app/static/assets/`; no template runtime):
  - **Upload:** drag-drop, file picker, or live **camera capture**
    (getUserMedia → canvas → File) → `POST /api/predict`; the user's own
    pixels render immediately while the towers run.
  - **Detection canvas:** letterboxed contain-fit with blueprint grid;
    bounding boxes with corner ticks + JetBrains-Mono label chips, coloured
    by per-item CO2e tier (teal / amber ≥0.12 kg / rose ≥0.24 kg); GSAP
    draw-in sweep; hover cursor + click hit-testing (smallest box wins).
  - **Split-screen grid sync:** every canvas box links 1:1 to an editable
    item card via the echoed `id` — clicking either side focuses both
    (`.active-item` ring ↔ heavier stroke + tint), with smooth scroll-to.
  - **Dual-stage carbon UX:** cards prefill the Stage-A γ proxy, then
    `POST /api/calculate-impact` (country + ids) swaps in audited factors —
    weight edits are debounced (650 ms) into re-audits; the card label
    flips `proxy → verified`, and the factor line shows
    `factor · source · weight_source`.
  - **DMM panel per card:** `POST /api/recommend` fills the 3 ranked
    process tabs (rank-numbered labels, Optimal/Acceptable/Warning chips,
    per-path CO2e) + the child-simple verdict and pros/cons; the section
    header shows the text provider (`llm_enriched` / fallback).
  - **Telemetry:** GSAP-tweened total CO2e counter, petrol-km equivalence,
    provider label, item/mass/recyclable-share stats, ELEVATED/LOW impact
    chip; **geo badge + country select** pre-populated by an IP-geolocation
    lookup (ipapi.co, 4 s timeout, silent MY fallback) — manual changes flip
    AUTO-GEO → OVERRIDE and re-run audit + recommendations.
  - **Resilience:** every fetch failure lands in a GSAP toast + status pill;
    the numbers degrade to the Stage-A proxy, never a blank screen.
- **`templates/carbon_lab.html` (`/carbon-lab`):** retained dependency-free
  API tester for Modules 2+3 (request/response JSON panels).

---

## 7. Coding Conventions & Best Practices

- **Thin controllers:** route handlers in `blueprints/` only validate input, call a service, and return JSON. No business logic in routes.
- **Services own logic:** detection / classification / carbon / recommendation logic lives in `app/services/`. Services are import-safe and unit-testable in isolation.
- **Config:** class-based in `config.py`; the factory selects via `FLASK_ENV`. Never hardcode secrets — read from `.env` through `config.py`.
- **App factory:** `create_app(env_name)` builds the app. No global `app` object.
- **Errors:** services raise `app.utils.errors.ApiError(message, status_code)`. A registered handler converts it to JSON.
- **Validation:** pydantic schemas in `app/schemas/` are the documented JSON contract (incl. `box_area_px`, `material_scores`, `physics`, `estimated_carbon_kg`); validate payloads before touching services.
- **Secrets:** `.env` is gitignored. `.env.example` documents required keys with blank values.
- **External calls:** wrap all `requests` calls with timeouts and explicit error handling; never let an upstream failure 500 silently.
- **Logging:** use `app.logger` / module loggers, not `print`.
- **Tests:** mock BOTH model towers and external APIs — tests must not hit the network, download weights, or require a GPU. Heavy imports (`ultralytics`, `transformers`, PIL inside services) stay lazy for this reason.
- **Style:** clear names, docstrings on every module/function explaining intent, small functions.
- When adding a dependency, pin its version in `requirements.txt`.

---

## 8. Environment Variables (`.env`)

```
FLASK_ENV=development
SECRET_KEY=<long random string; generate with: python -c "import secrets; print(secrets.token_hex(32))">

CARBON_PROVIDER=climatiq          # or carbon_interface
CLIMATIQ_API_KEY=<from climatiq.io dashboard — blank = local dummy factors>
CARBON_INTERFACE_API_KEY=         # only if using that provider

LLM_API_KEY=                      # OPTIONAL v3.6 text layer — blank = local grid copy
LLM_MODEL=llama-3.3-70b-versatile # any model at the endpoint below
LLM_API_URL=https://api.groq.com/openai/v1/chat/completions
                                  # any OpenAI-compatible chat-completions URL:
                                  # Groq (free default) | OpenRouter | Gemini compat
                                  # | http://localhost:11434/v1/chat/completions (Ollama)

MODEL_PATH=models/yolov8n-waste-det.pt   # Stage 1 specialist detector (auto-fetched if missing)
                                          # A/B: models/yolov8m-seg-trash.pt | yolo26x-seg.pt
VIT_MODEL_NAME=edwinpalegre/ee8225-group4-vit-trashnet-enhanced   # Stage 2 (HF model id)
CONFIDENCE_THRESHOLD=0.15         # Stage 1, recall-first (code also fixes agnostic NMS, iou=0.45)
INFERENCE_DEVICE=0                # BOTH towers: "cpu" or CUDA index ("0")
DATABASE_URL=sqlite:///waste_app.db
```

Notes:
- All API keys can be blank for local testing — the app boots and tests pass without them.
- First `/api/predict` request downloads both model weights (see Module 1) — internet once.
- γ (the pixel-area calibration constant) is code, not env: `carbon_service.PIXEL_AREA_GAMMA`.

---

## 9. Common Commands

> Windows / PowerShell. Do NOT use `&&` or `source`.

```powershell
# Create & activate the 3.11 virtualenv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked: Set-ExecutionPolicy -Scope Process -Bypass   (then retry)

# Install dependencies (ultralytics >= 8.4 for YOLO26; transformers for the ViT)
python -m pip install --upgrade pip
pip install -r requirements.txt
# OPTIONAL GPU inference (GTX 1650, CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Run the app
copy .env.example .env        # then edit .env
python run.py                 # http://127.0.0.1:5000  → Dual-Tower Test Brain page

# Tests
pytest -q

# Verify GPU is visible (only if using INFERENCE_DEVICE=0)
python -c "import torch; print(torch.cuda.is_available())"
```

**Sanity check you're inside the venv:** the prompt should start with `(.venv)` and pip should NOT say "Defaulting to user installation."

Legacy training commands (dataset download, `train.py`, `evaluate.py`, `export.py`)
still work but are **archived** — see §12. Do not run them as part of the active pipeline.

---

## 10. Build Roadmap

Each completed step gets its own commit + push — see §13 for the commit convention.

| Step | Description | Status |
|---|---|---|
| 1 | Project foundation — factory, config, blueprints, DB, errors, tests | **DONE** |
| 2 | Dataset prep — inspection, validation, `data.yaml`, balance report | **DONE (legacy)** |
| 3 | YOLO11n training — transfer learning, augmentation, GTX-1650 settings | **DONE (legacy)** |
| 4 | Evaluation + export + `detection_service` + `POST /api/predict` | **DONE (legacy)** |
| 4.5 | Pivot v1: single-stage YOLO-World zero-shot brain | **DONE (superseded)** |
| 4.6 | Pivot v2: TWO-STAGE pipeline — YOLO-World locator (abstract anchors) + CLIP 7-class material classifier + interactive Test Brain inspector | **DONE (locator superseded by 4.7)** |
| 4.7 | CRITICAL FIX: vanilla YOLOv8-L (stock COCO-80) replaces YOLO-World as Stage 1 after abstract anchors produced 0 detections; COCO labels kept as `located_as` diagnostics | **DONE (locator superseded by 4.8)** |
| 4.8 | UPGRADE: RT-DETR-X detection transformer replaces YOLOv8 as Stage 1 — global self-attention, NMS-free (IoU knob removed) | **DONE (superseded by 4.9)** |
| 4.9 | DUAL-TOWER HYBRID: YOLO26-X-SEG instance segmentation + context-aware square padding + supervised TrashNet ViT replaces CLIP + pixel-area dynamic carbon scaling (γ=5000) + polygon-mask frontend | **DONE (locator superseded by 4.10)** |
| 4.10 | SPECIALIST LOCATOR + METHOD B: blended TACO+TrashNet waste segmenter (yolov8m-seg-trash.pt) replaces COCO-80 Stage 1 + classical-CV Plasticity Index ψ tie-breaker | **DONE (locator superseded by 4.11)** |
| 4.11 | DETECTION REGRESSION: specialist waste OBJECT DETECTOR (yolov8n-waste-det.pt, blended corpus) replaces segmentation as Stage 1 — box-area carbon proxy with γ recalibrated 5000→8000, classic strokeRect frontend, cleaner background rejection, nano-speed edge throughput; Method B + ViT unchanged | **DONE** |
| 5 | Carbon module — live Climatiq factors (cached 1-kg probes per (material, country, api_key), region-scoped) with local-dummy fallback + `POST /api/calculate-impact` (pydantic-validated weights/country, per-item + total CO2e, provider labelling) — the v3.5 **dual-stage carbon UX** (Stage A blind γ proxy / Stage B precision audit) | **DONE** |
| 6 | DECISION MAKING MODULE (v3.5): 3-path parallel end-of-life carbon simulation (taxonomy-branched: dry recyclables / organics / residual), ascending-CO2e sorting & ranking core (Optimal / Acceptable / Warning), 7×3 disposal-factor matrix with negative offset credits, structured expert knowledge base (pros/cons) + rank-aware verdicts, `POST /api/recommend` (audited weight or box-area proxy) | **DONE** |
| 6.5 | v3.6 DMM TEXT LAYER: Child-Friendly, Country-Localized LLM Generation Pipeline — hyper-simple system prompt (≤25 words/field, jargon banlist, numbers woven in), one batched strict-JSON call to a free OpenAI-compatible endpoint (Groq default; OpenRouter/Gemini/Ollama), `country` context injection with "global average" default, atomic staged application, seamless `local_fallback` on any failure, knowledge grid rewritten to the same simple register | **DONE** |
| 7 | **CarbIQ SPA frontend — design-compiled Tailwind/GSAP dashboard: canvas↔grid id-synced split-screen, camera capture, IP-geolocated country defaulting, editable weights driving debounced Stage-B audits + DMM refresh, ranked process tabs, animated telemetry, self-hosted fonts** | **DONE (this step)** |
| 8 | Full test suite, gunicorn deployment guide, FYP documentation | **Next** |

**Deliverables checklist — ALL CORE ITEMS COMPLETE:** multi-object upload ✓ ·
auto bounding boxes ✓ · per-item confidence (both towers) ✓ · per-item
material evidence ✓ · size-aware carbon estimates (box-area scaling) ✓ ·
real-time carbon API with country scoping + fallback ✓ · total + per-item
CO2e ✓ · structured disposal instructions ✓ (DMM: ranked 3-path
prescriptions with expert commentary) · per-item weight inputs ✓ (debounced
Stage-B re-audit) · polished responsive UI ✓ (CarbIQ SPA). Remaining: Step 8
hardening + FYP documentation.

---

## 11. ARCHIVED: Superseded locator & classifier designs (report narrative)

Kept for the report's design-evolution chapter — the generations before v3:

**Pivot v1 — single-stage YOLO-World (object-noun prompts).** One frozen YOLO-World
model did both jobs with six object-level prompts (`"plastic bottle"`, `"aluminum
can"`, `"glass jar"`, `"cardboard box"`, `"paper packaging"`, `"organic waste food"`)
at `conf=0.25`. **Why replaced:** object-noun prompts couple material identity to
*shape*, so crushed/deformed items were missed or mislabeled, and one confidence score
conflated "is something there?" with "what is it?".

**Pivot v2 — two-stage with YOLO-World abstract anchors.** Stage 1 used YOLO-World
with 6 trash-synonyms, then 5 abstract semantic anchors (`"container"`, `"packaging"`,
`"food"`, `"waste"`, `"object"`). **Why replaced (empirical failure):** ZERO
detections on pristine everyday items even at `conf=0.05` — open-vocab alignment
grounds *concrete noun phrases*, not abstract super-categories. The two-stage
decomposition itself was correct and survives; only the locator changed.

**Pivot v2.1 — two-stage with vanilla YOLOv8-L.** `yolov8l.pt` (stock COCO-80,
`agnostic_nms=True`, `iou=0.45`) restored reliable recall and introduced the
`located_as` diagnostic. **Why replaced:** CNN + NMS could merge adjacent items in
dense scenes (the IoU suppression step).

**Pivot v2.2 — two-stage with RT-DETR-X.** The detection transformer solved the
dense-scene merging via global self-attention and one-to-one (NMS-free) queries.
**Why replaced (upgrade, not failure):** box-only output — no pixel masks, so no
size-aware carbon scaling; and YOLO26's end-to-end head now delivers the same
NMS-free guarantee from a CNN that *also* segments. v3 keeps every v2.2 property
(supervised COCO, NMS-free, `located_as`) and adds masks.

**Pivot v3 — dual-tower with COCO-80 YOLO26-X-SEG.** The 2026 CNN segmenter
(`yolo26x-seg.pt`, end-to-end NMS-free) introduced the pixel masks and the
padding layer that survive today. **Why the locator was replaced (upgrade):**
its COCO latent space boxed *background* objects — a live test on a bus scene
segmented the bus and pedestrians, which the ViT (having no "not waste" class)
force-classified as "paper" at 97%+ — and amorphous litter with no COCO
look-alike was missed entirely. The specialist blended-corpus segmenter (v3.1)
fixes both architecturally; the COCO baseline stays one env-var away
(`MODEL_PATH=yolo26x-seg.pt`) for A/B comparisons. Trade-off documented: the
community checkpoint has no published metrics and reintroduces default NMS
(v8-family), accepted in exchange for waste-only recall.

**Pivot v3.1 — dual-tower with the blended-waste SEGMENTER.** The specialist
`yolov8m-seg-trash.pt` (TACO + TrashNet fine-tune) introduced waste-only
localization and carried the mask-based pixel-area carbon scaling (γ=5000).
**Why the locator was replaced (upgrade):** detection-only v3.2 trades the tight
mask contours for (1) ~9x smaller weights and no mask-decoding overhead — real
frame-rate gains on edge hardware; (2) no mask-suppression merge artifacts in
dense, tightly packed layouts; (3) empirically cleaner background rejection on
out-of-domain scenes (2 vs 5 false fires at conf=0.15 on the bus-scene probe).
The cost — box areas over-cover tight contours — is absorbed by recalibrating
γ to 8000 (measured fill factor ~0.6). The segmenter stays one env-var away
(`MODEL_PATH=models/yolov8m-seg-trash.pt`) for A/B mask-vs-box comparisons.

**Stage-2 v1 — zero-shot CLIP (`openai/clip-vit-base-patch32`).** Scored each crop
against the taxonomy via the prompt `"a photo of {} waste"`. **Why replaced:** the
supervised TrashNet ViT has actually *trained on waste imagery* (98.17% val acc on
trashnet-enhanced vs. CLIP's unsupervised text-image alignment), and TrashNet-style
single-object patches are exactly what the processing layer produces. Trade-off
documented: the taxonomy is now fixed by the checkpoint (adding a class means
retraining), whereas CLIP's was editable text.

---

## 12. ARCHIVED: Legacy Custom-Training Era (kept for the FYP report)

> Everything in this section is **inactive**. The files stay in the repo for the
> academic write-up (baseline comparison + methodology chapters). Never wire them
> back into `app/`.

- **Model:** YOLO11n (`yolo11n.pt` — note: no "v" in the name) transfer-trained on
  Kaggle `viswaprakash1990/garbage-detection` (10,464 images, YOLO format, CC BY 4.0,
  via Roboflow `material-identification/garbage-classification-3` v2).
- **Classes (locked to that dataset, order immutable):**
  `0:BIODEGRADABLE 1:CARDBOARD 2:GLASS 3:METAL 4:PAPER 5:PLASTIC` — still present in
  `config.py:CLASS_NAMES` solely for the archived scripts/tests. (The active 7-class
  taxonomy deliberately echoes these six, lowercase, plus `general rubbish`.)
- **Class imbalance:** ~10:1 (BIODEGRADABLE 65% of boxes vs PAPER 6%).
- **Split fix:** the shipped Roboflow split had severe distribution shift (PAPER:
  33 valid boxes vs 1,376 test; GLASS: 0 test images), making validation metrics
  meaningless. `ml/scripts/restratify_dataset.py` produced a class-balanced 80/10/10
  re-split in `ml/data/garbage_stratified/`.
- **Training settings that mattered:** `imgsz=640, batch=8, amp=False` — the
  **GTX 16-series produces NaN losses under FP16**; FP32 batch=8 peaked ~2.8 GB.
  ~8 min/epoch, 6–8 h per 50-epoch run (a key motivation for abandoning training).
- **Final results (production model `waste_yolo11n_v2`, held-out test):**
  mAP50 **0.671**, mAP50-95 0.467, precision 0.774, recall 0.567, F1 0.655.
  Per-class mAP50: GLASS 0.795, METAL 0.698, PLASTIC 0.695, PAPER 0.672,
  BIODEGRADABLE 0.628, CARDBOARD 0.541. Artifacts in `ml/runs/waste_yolo11n_v2/eval/`.
- **Report angle:** quantitative baseline motivating the zero-shot/foundation pivots
  (recall 0.567 → missed ~43% of objects; fixed taxonomy; hours per iteration).
- Dataset folder name contains a space (`"GARBAGE CLASSIFICATION"`) — quote the path.
- `data.yaml` gotcha: Ultralytics resolves a relative `path:` against its global
  `datasets_dir`, not the yaml location; `train.py` rewrote it to absolute at runtime.

---

## 13. Gotchas & Guardrails for Claude Code

- **Never** suggest Python 3.13 or bash-only commands for this Windows/PowerShell user.
- **Never** delete or "clean up" the `ml/` workspace, legacy scripts, notebooks, or
  their tests — they are FYP report material (§2, §12).
- The locator weight is **`models/yolov8n-waste-det.pt`** (specialist detector;
  auto-fetched from GitHub via `_WEIGHT_SOURCES` when missing). A/B alternatives:
  `models/yolov8m-seg-trash.pt` (v3.1 masks) and `yolo26x-seg.pt` — NO dash between
  "26" and "x"; requires `ultralytics>=8.4` (pinned 8.4.90) — never downgrade.
- `_load_model` still dispatches "rtdetr" names to the `RTDETR` class (archived
  pivot compatibility); everything else loads via `YOLO`. Legacy trained weights
  were `yolo11n.pt` (no "v"). Don't mix naming schemes.
- First model load downloads `models/yolov8n-waste-det.pt` (~6 MB, GitHub) +
  ~343 MB TrashNet ViT (HF cache). Keep `*.pt` gitignored; never commit weights.
- The carbon size proxy is the **clamped box area** — a rectangle includes
  background, which is exactly why γ is 8000 here vs the mask-era 5000. If a
  future locator brings masks back, revisit γ alongside it.
- **Never** reintroduce `set_classes()` / anchor prompts into Stage 1 — running the
  checkpoint's native vocabulary is a deliberate, empirically-motivated decision
  (§2, §11). `located_as` is diagnostic-only — even though the specialist's labels
  LOOK like materials (Glass/Metal/Paper/Plastic/Waste), never display them as the
  verdict or branch on them.
- Stage-1 suppression is **class-agnostic by design**: the predict call fixes
  `agnostic_nms=True, iou=0.45` (`_NMS_IOU` in `detection_service.py`) — per-class
  default NMS let one object survive as overlapping "Paper" + "Waste" boxes that
  the ViT then named identically (duplicate frames). Never remove these args or
  revert to per-class suppression; the NMS-free A/B baselines (YOLO26, RT-DETR)
  simply ignore them. Add no other NMS knobs (`max_det`, etc.).
- Method B is a TIE-BREAKER only: it may reorder an ambiguous plastic-vs-glass top-2
  (gap < `PLASTICITY_TIEBREAK_MARGIN` = 0.15) and nothing else. Its constants
  (`_LAPLACIAN_REF`, `_EDGE_DENSITY_REF`, margin) are code constants in
  `detection_service.py`; every correction is flagged via `physics.tiebreak_applied`.
- Stage-1 `conf=0.15` is deliberately low (recall-first); don't "fix" it upward —
  Stage 2's scores carry the per-item certainty shown to users.
- Stage 2 is the **supervised TrashNet ViT** (image-classification pipeline), NOT
  CLIP (archived §11): no candidate labels, no hypothesis template. Always request
  the full distribution (`top_k` ≥ num_labels — the pipeline's default of 5 silently
  truncates 7 classes).
- The ViT's native `trash` label MUST map to `general rubbish` via
  `MODEL_LABEL_TO_MATERIAL`; the 7 material strings, `DISPLAY_NAMES`,
  `DUMMY_CARBON_FACTORS`, `DISPOSAL_METHOD_FACTORS`, `DISPOSAL_PATHS` and
  `EXPERT_KNOWLEDGE` stay in **lockstep** (§5) — tests enforce full 7×3 coverage.
- γ (`PIXEL_AREA_GAMMA` = 8000) is a code constant in `carbon_service.py`, not env.
- Climatiq estimate selectors take an **activity id only** — never add
  category/sector fields to the payload (they're search-endpoint filters, the
  API ignores them). Regional coverage is dataset-scoped (BEIS→GB, EPA→US), so
  `CLIMATIQ_MATERIAL_MAP` holds ordered candidate ids per material; the
  global fallback may fire ONLY on `error_code no_emission_factors_found` —
  auth/quota/malformed-selector errors must stay loud 502s.
- The audit endpoint's item `id` is the CLIENT's grid row key: echo it back
  verbatim (null when absent), never renumber, filter or reorder items
  server-side — item order in == item order out. Weight substitution goes
  through the shared `carbon_service.resolve_effective_weight` for BOTH
  `/api/calculate-impact` and the DMM — never fork a second copy of that rule.
- The **DMM's NUMERIC core is local + deterministic by design**: never add
  network calls, API keys, or Flask app-context dependence to the simulation,
  ranking or disposal-factor lookups — live regional factors belong to
  `POST /api/calculate-impact` (Module 2). The v3.6 LLM layer is the module's
  ONLY network touchpoint: it may ONLY rewrite the three text fields (never
  ranks, methods, numbers or item order), it runs once per request in the
  request thread (never inside the path thread-pool), and EVERY failure mode
  must degrade to the local grid with `provider: "local_fallback"` —
  recommendations must never 502 because of the LLM.
- The LLM endpoint is **OpenAI-compatible chat-completions via `requests`**
  (`LLM_API_URL`/`LLM_MODEL`/`LLM_API_KEY`) — do not add provider SDKs, and
  keep `TestingConfig.LLM_API_KEY = ""` so tests stay hermetic (LLM tests
  monkeypatch `requests.post`, mirroring the Climatiq pattern).
- Keep the v3.6 register in lockstep: `EXPERT_KNOWLEDGE` copy and the runtime
  verdicts obey the same "≤25 words, child-simple, no jargon" standard as
  `LLM_SYSTEM_PROMPT` (a test enforces the word budget) — if the standard
  changes, change the prompt, the grid and the test together.
- `DISPOSAL_METHOD_FACTORS` are NET per-kg constants and **credits are
  NEGATIVE** — never clamp them to ≥ 0 (the ranking depends on offsets), and
  never constrain `carbon_impact_kg` / summary totals to non-negative in
  schemas or the frontend.
- The DMM ranks by **ascending CO2e only** (method-name tie-break) — do not
  re-order paths by any other heuristic; `status_tag` maps 1:1 from rank
  (1 Optimal / 2 Acceptable / 3 Warning). Verdict copy is composed at runtime
  from the sorted outcome — never hard-wire rank assumptions into
  `EXPERT_KNOWLEDGE` text.
- Do not squish crops for the ViT — the processing layer's square padding exists to
  preserve aspect ratio (§2); never replace it with a naive resize.
- **Never** put business logic in route handlers — services only.
- **Never** commit `.env`, model weights, or the dataset (`ml/data/`).
- **Never** make tests depend on the network, weight downloads, or a GPU — mock both
  towers and external APIs.
- The app must run end-to-end **without** an LLM key (the DMM's numbers are
  rule-based and the local text grid always exists) and **without** a Climatiq
  key (local dummy factors take over automatically).
- Keep the carbon numbers AND the DMM ranking free of any trained model —
  coefficients, arithmetic and a structured knowledge base only; the LLM may
  phrase the story, never compute it.

---

## 14. Git, Security & Commit Workflow

**Remote:** `https://github.com/K01J10W19/ECO-WASTE-AI.git` (default branch: `main`).

### 14.1 Security — never leak secrets or data
- **`.gitignore` is the safety net, not the only check.** Before every commit, run
  `git status` and visually confirm no secret/data file is staged.
- The following must **never** enter version control (all covered by `.gitignore`):
  - `.env` and any `*.env` / `.env.*` file (only `.env.example` with **blank** values is committed).
  - Private keys & certs: `*.pem`, `*.key`, `*.crt`, `*.p12`, `*.pfx`, `id_rsa*`, `id_ed25519*`.
  - Credential files: `secrets/`, `secrets.*`, `credentials.*`, `service-account*.json`, `client_secret*.json`.
  - ML weights & datasets: `*.pt`, `*.pth`, `*.onnx`, `ml/data/`, `ml/runs/` (too large / licensed).
  - Runtime data: `*.db`, `*.sqlite*`, `app/static/uploads/*`, `instance/`, `*.log`.
- If a secret is ever committed by mistake: **rotate the key immediately** (treat it as
  compromised) and purge it from history (`git filter-repo` / BFG) before pushing.
- `.env.example` documents required keys with **empty** values — keep it in sync with real
  env needs, but never copy real values into it.

### 14.2 Commit convention — one commit per completed roadmap step
Each completed step in the **Build Roadmap (§10)** gets its own commit. The message
describes the action performed (what was built and why it satisfies that step):

```
Step <N>: <short action title>

- <bullet describing what was implemented>
- <bullet describing key files / decisions>
- Roadmap §10 Step <N> complete
```

### 14.3 Push workflow (Windows / PowerShell)
```powershell
# per completed roadmap step
git add -A
git status                       # verify NO secrets/data staged
git commit -m "Step N: <action description>"
git push -u origin main          # subsequent pushes: git push
```
- Authentication uses **Git Credential Manager** (already configured system-wide).
- Commit & push **only when a roadmap step is genuinely complete** (code + tests pass),
  not for half-finished work. Keep history clean and auditable for the examiner.
