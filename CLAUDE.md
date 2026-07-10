# CLAUDE.md

> Persistent project context for Claude Code. Read this fully before acting.
> This file is the single source of truth for the project's goals, architecture,
> conventions, and constraints. When something here conflicts with an assumption
> you would otherwise make, **this file wins**.

---

## 1. Project Identity

**Title:** Deep Learning Multi-Target Waste Detection and Carbon Impact Estimation Web Application

**Type:** Final Year Project (FYP). Code must be production-ready, modular, well-documented, and easy for an academic examiner to audit.

**One-line description:** A web app where a user uploads a real-world image containing one or more waste items; a **100% local Dual-Tower Hybrid pipeline** — a **specialized waste object detector** (YOLOv8-N fine-tuned on a blended universal waste corpus of wild litter + household recyclables) boxes ONLY trash-like objects, a context-aware square-padding layer normalizes each crop, a **classical-CV physics extractor** (Method B: Laplacian wrinkles + Canny edges → Plasticity Index ψ) profiles each patch, and a TrashNet-fine-tuned Vision Transformer names the material (7-class taxonomy, with ψ breaking ambiguous plastic-vs-glass calls) — identifies every item; each item's carbon footprint is **dynamically scaled by its bounding box's geometric pixel area** (placeholder coefficients now, real-time Climatiq API with user-entered weights in Step 5); and it returns tailored recycling/disposal recommendations per item.

**Author's environment:** Windows 11, PowerShell, local GPU = **NVIDIA GTX 1650 (4 GB VRAM)**. All ML choices must respect this 4 GB limit (inference of both towers fits; only *training* ever exceeded it, and training is retired).

---

## 2. ARCHITECTURE: The Dual-Tower Hybrid (Waste YOLO Detector + Method B + TrashNet ViT) — read this first

The project went through several pivots (custom training → single-stage YOLO-World →
two-stage with abstract anchors → two-stage vanilla YOLOv8 → two-stage RT-DETR-X →
COCO-80 YOLO26-seg → blended-waste segmenter → **this, v3.2**). The paradigm is
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
conf=0.15 (recall-first, conf-only call)
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
          auto-fetched if missing), predict (conf only, 0.15)
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
  → Frontend: canvas BOUNDING-BOX overlay (ctx.strokeRect borders + light
    interior tint, rectangular hit-testing, smallest box wins on overlap);
    hover/click a box (or list row) → inspector with the ViT score bars +
    the ψ physics readout + the carbon formula; raw JSON panel
```

**Hard rules:**
- **Do NOT delete any legacy training assets** — `ml/` scripts, notebooks, configs,
  runs, and their tests are retained for the FYP report (archive: §12). They are
  bypassed, never imported by `app/`.
- Stage 1 stays vocabulary-free at call time: NO `set_classes()`, no prompt lists —
  that approach is archived (§11) after failing empirically. `located_as` is
  diagnostic-only; never surface it as the material or branch on it.
- The Stage-1 predict call passes **`conf` only** — never add tuning knobs (`iou`,
  `agnostic_nms`); library defaults handle suppression for the v8-seg weights, and
  the YOLO26 A/B baseline is NMS-free anyway.
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
| Stage 1 tuning | `conf=0.15` default (per-request override) — **the only knob; NMS-free** | Recall-first; Stage 2 carries per-item certainty. |
| Carbon scaling | `estimated_carbon_kg = base x (box_area_px / γ)`, **γ = 8000** (recalibrated from 5000 for rectangular over-coverage, fill factor ~0.6) | Box area as volume/mass proxy until Step 5's real weights. |
| Carbon provider | **Climatiq** (Step 5). Until then: **dummy per-kg coefficients** in `carbon_service.py` keyed on the 7 materials | Keeps the JSON contract flowing end-to-end today; Step 5 swaps internals only. |
| Recommendation engine | **Rule-based core + OPTIONAL LLM enrichment** | Deterministic engine is the gradeable default; LLM activates only when an API key is present. Must work fully with no LLM key. |
| Backend | **Flask** (app factory + blueprints + services) | Thin controllers, logic in services. |
| Frontend | HTML5 + Tailwind CSS + native JS (ES6+, Fetch API) + GSAP | SPA. Current interim page is a vanilla "Test Brain" tester with a classic bounding-box overlay + inspector; polished UI is Step 7. |
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
- **Frontend:** HTML5, Tailwind CSS, vanilla JavaScript (Fetch API), GSAP (Step 7); interim test page is dependency-free vanilla HTML/CSS/JS (Canvas 2D `strokeRect` bounding-box rendering)
- **Database:** SQLite (via SQLAlchemy)
- **External APIs:** Climatiq (carbon, Step 5) — Carbon Interface kept as alternate adapter
- **Optional:** Anthropic/LLM API for recommendation enrichment
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
│   │   ├── carbon_service.py           # base coefficients + pixel-area dynamic scaling (γ)
│   │   └── recommendation_service.py   # rule-based + optional LLM (Step 6)
│   ├── models/scan.py     # SQLite model for optional scan history
│   ├── schemas/           # pydantic validation models (JSON contract incl. box_area/physics)
│   ├── utils/errors.py    # ApiError + register_error_handlers()
│   ├── static/            # css/ js/ uploads/
│   └── templates/index.html   # "Test Brain" tester page (full SPA in Step 7)
├── ml/                   # LEGACY training workspace — KEEP, DO NOT DELETE (FYP report)
├── models/               # legacy exported weights location (best.pt, gitignored)
├── tests/
│   ├── conftest.py       # pytest fixtures (app, client)
│   ├── test_api.py  test_detection.py  test_classification.py  test_carbon.py
│   ├── test_predict_endpoint.py
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
- The predict call passes `conf` only; suppression is the library's default
  behaviour for these weights.

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
  `carbon_service.DUMMY_CARBON_FACTORS`, and (Step 6) recommendation rules are keyed
  on it. `tests/test_detection.py::test_taxonomy_lockstep` +
  `tests/test_classification.py::test_every_native_vit_label_lands_in_the_taxonomy`
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
    call passes conf only. Device from `INFERENCE_DEVICE` for both towers.
  - All failures surface as `ApiError`; empty detections are a valid result.
- **Hardware:** CPU works (well under a second per image for the nano detector);
  GTX 1650 handles both towers' *inference* trivially (v8-N ~0.05 GB + ViT-B/16
  ~0.35 GB — far inside the 4 GB budget).
- **First-run downloads:** `models/yolov8n-waste-det.pt` (~6 MB, GitHub) + the
  TrashNet ViT (~343 MB, HF cache). Internet needed once.

### Module 2 — Carbon Impact Estimation (External API + pixel-area scaling, NO ML)
- **Current:** `carbon_service.DUMMY_CARBON_FACTORS` — placeholder per-kg CO2e base
  coefficients for **all 7 materials** (biodegradable 0.57, cardboard 0.94, glass 0.85,
  metal 4.50, paper 1.09, plastic 3.10, general rubbish 1.20), plus the **box-area
  dynamic scaling**: `estimate_dynamic_impact(label, box_area_px)` = base ×
  (area / γ), γ = `PIXEL_AREA_GAMMA` = 8000 (recalibrated for rectangular
  over-coverage). Coefficients are order-of-magnitude realistic, clearly marked dummy.
- **Step 5:** swap lookup internals for live Climatiq calls (material + user-entered
  weight (kg) + ISO country code), async per item, aggregate total CO2e. Public
  signatures (`get_carbon_factor`, `estimate_impact`, `estimate_dynamic_impact`)
  must not change; the pixel-area proxy remains as the no-weight fallback.
- Must handle: missing/invalid weight, unknown material mapping, API timeout/error,
  missing API key — all as clean, user-facing errors via `ApiError`.

### Module 3 — Recommendation System (rule-based core + optional LLM)
- Input: the 7-class material verdicts + carbon values.
- **Default path:** deterministic rule-based engine — modular mapping from material →
  structured disposal guidance. **Optional path:** LLM enrichment when `LLM_API_KEY`
  is set; MUST degrade gracefully to rule-based output otherwise.

### Module 4 — Web Application
- **Current interim page (`templates/index.html`):** the "Dual-Tower Test Brain" —
  drag-drop/file-picker upload, instant local preview, **Analyze** button, Stage-1
  threshold slider, **classic bounding-box canvas overlay** (`ctx.strokeRect` with
  crisp semi-transparent per-material borders + a very light interior tint,
  rectangular hover/click hit-testing, smallest box wins on overlap, label chips
  over the top-left corner), and an **interactive inspector**: when analysis
  completes the most confident item is auto-pinned showing its 7-class ViT score
  bars, the box pixel area, the Method B physics readout (ψ, wrinkle variance,
  edge density, and whether the tie-break corrected the ranking), and the full
  carbon formula readout (base × area ÷ γ); hovering or clicking any box (or list
  row) walks the other items. Raw JSON panel below. Vanilla HTML/CSS/JS, zero
  dependencies.
- **Step 7:** the polished SPA — Tailwind layout, GSAP transitions, per-item weight
  inputs + country selector, carbon dashboard, recommendation list.

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
CLIMATIQ_API_KEY=<from climatiq.io dashboard — needed in Step 5>
CARBON_INTERFACE_API_KEY=         # only if using that provider

LLM_API_KEY=                      # OPTIONAL — leave blank for rule-based only
LLM_MODEL=claude-sonnet-4-6

MODEL_PATH=models/yolov8n-waste-det.pt   # Stage 1 specialist detector (auto-fetched if missing)
                                          # A/B: models/yolov8m-seg-trash.pt | yolo26x-seg.pt
VIT_MODEL_NAME=edwinpalegre/ee8225-group4-vit-trashnet-enhanced   # Stage 2 (HF model id)
CONFIDENCE_THRESHOLD=0.15         # Stage 1, recall-first — the ONLY Stage-1 knob (NMS-free)
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
| 4.11 | **DETECTION REGRESSION: specialist waste OBJECT DETECTOR (yolov8n-waste-det.pt, blended corpus) replaces segmentation as Stage 1 — box-area carbon proxy with γ recalibrated 5000→8000, classic strokeRect frontend, cleaner background rejection, nano-speed edge throughput; Method B + ViT unchanged** | **DONE (this step)** |
| 5 | Carbon module — real Climatiq calls in `carbon_service` + `POST /api/calculate-impact` | **Next** |
| 6 | Recommendation module — rule-based + optional LLM enrichment | Pending |
| 7 | Frontend — full SPA: Tailwind/GSAP, weight forms, results dashboard | Pending |
| 8 | Full test suite, gunicorn deployment guide, FYP documentation | Pending |

**Deliverables checklist:** multi-object upload ✓ · auto instance masks + boxes ✓ ·
per-item confidence (both towers) ✓ · per-item material evidence (ViT bars) ✓ ·
size-aware carbon estimates (pixel-area scaling) ✓ · per-item weight inputs (Step 7) ·
async real-time carbon API (Step 5) · total + per-item CO2e (Step 5/7) · structured
disposal instructions (Step 6) · polished responsive UI (Step 7).

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
- **Never** add NMS parameters (`iou`, `agnostic_nms`) to the Stage-1 predict call —
  it passes `conf` only; library defaults do the rest for whichever weights are set.
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
  `MODEL_LABEL_TO_MATERIAL`; the 7 material strings, `DISPLAY_NAMES`, and
  `DUMMY_CARBON_FACTORS` stay in **lockstep** (§5) — tests enforce both.
- γ (`PIXEL_AREA_GAMMA` = 8000) is a code constant in `carbon_service.py`, not env.
- Do not squish crops for the ViT — the processing layer's square padding exists to
  preserve aspect ratio (§2); never replace it with a naive resize.
- **Never** put business logic in route handlers — services only.
- **Never** commit `.env`, model weights, or the dataset (`ml/data/`).
- **Never** make tests depend on the network, weight downloads, or a GPU — mock both
  towers and external APIs.
- The app must run end-to-end **without** an LLM key (rule-based fallback) and, today,
  **without** a Climatiq key (dummy coefficients until Step 5).
- Keep carbon estimation free of any trained model — coefficients + arithmetic only.

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
