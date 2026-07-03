# CLAUDE.md

> Persistent project context for Claude Code. Read this fully before acting.
> This file is the single source of truth for the project's goals, architecture,
> conventions, and constraints. When something here conflicts with an assumption
> you would otherwise make, **this file wins**.

---

## 1. Project Identity

**Title:** Deep Learning Multi-Target Waste Detection and Carbon Impact Estimation Web Application

**Type:** Final Year Project (FYP). Code must be production-ready, modular, well-documented, and easy for an academic examiner to audit.

**One-line description:** A web app where a user uploads a real-world image containing one or more waste items; a standalone YOLO model localises and classifies each item; the system estimates each item's carbon footprint via a real-time Carbon Emission API using user-entered weights; and it returns tailored recycling/disposal recommendations per item.

**Author's environment:** Windows 11, PowerShell, local GPU = **NVIDIA GTX 1650 (4 GB VRAM)**. All ML choices must respect this 4 GB limit.

---

## 2. Current Status & Locked Decisions

When you (Claude Code) join, assume the following are already settled. Do **not** re-litigate them.

| Topic | Decision | Reason |
|---|---|---|
| Python version | **3.11.x** (NOT 3.13) | 3.13 has no prebuilt wheels for numpy 1.26 / torch; forces source builds that fail on Windows. 3.11 has wheels for everything. |
| Virtualenv (Windows) | `py -3.11 -m venv .venv` then `.\.venv\Scripts\Activate.ps1` | PowerShell does not support `&&` or `source`. Never give bash-only commands. |
| YOLO weights | **`yolo11n.pt`** (no "v") | Ultralytics dropped the "v" for the YOLO11 generation. `yolov11n.pt` does not exist. |
| Detection model | Standalone YOLO11 (Ultralytics) | Single-stage detector; localisation + classification in one pass. |
| Carbon provider | **Climatiq** (default) | Has proper *material* emission factors (plastic, cardboard, glass...). Carbon Interface is vehicle/flight focused and a poor fit. Code keeps an adapter so the provider is swappable. |
| Recommendation engine | **Rule-based core + OPTIONAL LLM enrichment** | The original brief contradicted itself ("LLM-driven" vs "rule-based, no AI"). Resolution: deterministic rule-based engine is the gradeable, auditable default; LLM layer activates only when an API key is present. The app must work fully with no LLM key. |
| Backend | **Flask** (app factory + blueprints + services) | Matches the architecture diagram. Thin controllers, logic in services. |
| Frontend | HTML5 + Tailwind CSS + native JS (ES6+, Fetch API) + GSAP | Single-page app, drag-drop upload, bounding-box canvas, dynamic forms. |
| Database | SQLite (optional, for scan history) | Lightweight, local, file-based. |
| Dataset | **`viswaprakash1990/garbage-detection`** (Kaggle) | Already in YOLO format. 10,464 images, pre-split, zero orphans, zero malformed labels (confirmed by `inspect_dataset.py`). See §7. |

---

## 3. Tech Stack

- **Language:** Python 3.11
- **ML / Detection:** PyTorch, Ultralytics YOLO (`ultralytics==8.3.0`)
- **Image processing:** OpenCV (`opencv-python-headless`), Pillow, NumPy
- **Backend:** Flask 3, Flask-SQLAlchemy, python-dotenv, requests, pydantic
- **Frontend:** HTML5, Tailwind CSS, vanilla JavaScript (Fetch API), GSAP
- **Database:** SQLite (via SQLAlchemy)
- **External APIs:** Climatiq (carbon) — Carbon Interface kept as alternate adapter
- **Optional:** Anthropic/LLM API for recommendation enrichment
- **Testing:** pytest, pytest-mock
- **Serving (prod):** gunicorn
- **GPU:** CUDA build of torch (`cu121`) for the GTX 1650

---

## 4. System Architecture

```
Frontend (HTML/Tailwind/JS)
   | POST /api/predict  (image upload)
   v
Flask Backend Controller (thin routers in blueprints/)
   |---> detection_service  -> YOLO11 (Ultralytics/PyTorch)
   |                            returns array of {class, confidence, bbox}
   |---> carbon_service      -> Climatiq API (material, weight, country)
   |---> recommendation_service -> rule-based (+ optional LLM)
   v
Consolidated JSON  ->  Interactive Web Dashboard
```

**Request flow (end to end):**
1. User drops an image on the homepage drop-zone.
2. `POST /api/predict` (multipart) -> `detection_service` runs YOLO -> returns detected items with bounding boxes, class labels, confidence scores.
3. Frontend draws boxes on a canvas and renders a dynamic weight-input form per detected item.
4. User enters estimated weight per item + selects geographic location (ISO country code).
5. `POST /api/calculate-impact` -> `carbon_service` calls Climatiq async per item -> aggregates total CO2e.
6. `POST /api/recommend` (or combined in the same response) -> `recommendation_service` returns per-item disposal guidance.
7. Dashboard shows: per-item emission breakdown, total CO2e, and the recommendation list.

**Design principle:** Carbon estimation is intentionally **decoupled from ML** — it uses a trusted external API for precision and easy audit. The only trained model in the system is YOLO.

---

## 5. Repository Layout

```
waste-detection-app/
├── .env.example          # template of required env vars (safe to commit)
├── .env                  # real secrets (gitignored — never commit)
├── .gitignore
├── README.md
├── requirements.txt
├── config.py             # class-based config: Dev / Prod / Testing
├── run.py                # entry point: python run.py
├── app/
│   ├── __init__.py       # create_app() application factory
│   ├── extensions.py     # db = SQLAlchemy() (unbound)
│   ├── blueprints/
│   │   ├── main/routes.py    # HTML page routes (the SPA shell)
│   │   └── api/routes.py     # JSON API: /predict, /calculate-impact, /recommend, /health
│   ├── services/
│   │   ├── detection_service.py       # YOLO inference (Step 4)
│   │   ├── carbon_service.py          # Climatiq integration + adapter (Step 5)
│   │   └── recommendation_service.py  # rule-based + optional LLM (Step 6)
│   ├── models/
│   │   └── scan.py        # SQLite model for optional scan history
│   ├── schemas/          # pydantic validation models
│   ├── utils/
│   │   └── errors.py      # ApiError + register_error_handlers()
│   ├── static/
│   │   ├── css/  js/  uploads/
│   └── templates/
│       └── index.html
├── ml/                   # TRAINING workspace (separate lifecycle from the app)
│   ├── data/             # dataset (gitignored)
│   ├── configs/data.yaml # Ultralytics dataset config (finalised Step 2)
│   ├── scripts/
│   │   ├── inspect_dataset.py   # dataset audit / class balance  (DONE)
│   │   ├── prepare_dataset.py   # validation + data.yaml         (DONE, Step 2)
│   │   ├── restratify_dataset.py# class-balanced 80/10/10 re-split (Step 4 split fix)
│   │   ├── train.py             # transfer learning              (Step 3)
│   │   ├── evaluate.py          # metrics + confusion matrix     (Step 4)
│   │   └── export.py            # export .pt / ONNX              (Step 4)
│   ├── notebooks/
│   └── runs/             # training outputs (gitignored)
├── models/
│   └── best.pt           # exported weights the APP loads (gitignored until trained)
├── tests/
│   ├── conftest.py       # pytest fixtures (app, client)
│   ├── test_api.py
│   ├── test_prepare_dataset.py
│   ├── test_detection.py
│   └── test_carbon.py
└── docs/                 # FYP report material
```

**Hard rule:** keep `ml/` (training) and `app/` (serving) separate. The web app only ever *loads* an exported model from `models/`; it must never import training code.

---

## 6. LOCKED: The Six Waste Classes

Confirmed by `inspect_dataset.py` against the real dataset. **This order is immutable.**
`config.py`, `ml/configs/data.yaml`, and every label file use this exact order and these exact
ALL-CAPS names.

```
0: BIODEGRADABLE
1: CARDBOARD
2: GLASS
3: METAL
4: PAPER
5: PLASTIC
```

The model uses the ALL-CAPS names **internally**; the web UI always shows the **friendly
display name** via this mapping in `config.py`:

```python
APP_CLASS_DISPLAY_NAMES = {
    "BIODEGRADABLE": "Biodegradable",
    "CARDBOARD":     "Cardboard",
    "GLASS":         "Glass",
    "METAL":         "Metal",
    "PAPER":         "Paper",
    "PLASTIC":       "Plastic",
}
```

---

## 7. LOCKED: Dataset Facts (from inspection)

Established by `ml/scripts/inspect_dataset.py` on the real download. Treat as ground truth.

| Property | Value |
|---|---|
| Dataset root (original) | `ml/data/GARBAGE CLASSIFICATION/` (note the space — always quote the path) |
| Dataset root (TRAINING) | `ml/data/garbage_stratified/` — class-balanced re-split that `data.yaml` now points at (see split-fix note below) |
| Splits | Original ships `train/ valid/ test/`, but its **class distribution is broken** (see below); training/eval uses the re-stratified copy |
| Total images | 10,464 |
| Total labels | 10,464 (perfect 1:1 match) |
| Orphan images | 0 |
| Orphan labels | 0 |
| Malformed lines | 0 |
| Label format | YOLO (confirmed) |
| Source | Roboflow `material-identification/garbage-classification-3` v2 (CC BY 4.0) |

**Class balance (box instances):**

| ID | Class | Boxes | Share |
|---|---|---|---|
| 0 | BIODEGRADABLE | 45,407 | 65% |
| 1 | CARDBOARD | 4,698 | 7% |
| 2 | GLASS | 7,809 | 11% |
| 3 | METAL | 5,841 | 8% |
| 4 | PAPER | 4,390 | 6% |
| 5 | PLASTIC | 5,945 | 9% |

Imbalance ratio ~10:1 (BIODEGRADABLE vs PAPER). **Step 3 MUST compensate:** mosaic +
mixup augmentation (`copy_paste` is a no-op for bbox-only data), and monitor **per-class AP**.
BIODEGRADABLE will converge fastest; watch PAPER and CARDBOARD for underfitting.

**CRITICAL — the shipped split is broken (found in Step 4 eval).** The original Roboflow
`train/valid/test` split has severe class distribution shift: PAPER = train 2,981 / valid **33**
/ test 1,376; PLASTIC = valid **1.1%** / test **44%**; GLASS has **0** test images. This makes
minority-class *validation* metrics meaningless — PAPER "mAP" was measured on 33 boxes (→ 0.036),
yet the same model scores **0.58** on test's 1,376 PAPER boxes. The headline 51% val mAP was an
artifact; the model's true mAP50 is ~0.65. **Resolution:** `ml/scripts/restratify_dataset.py`
merges all three splits and writes a fresh, class-balanced 80/10/10 split to
`ml/data/garbage_stratified/` (each class ≈ its global share in every split), then repoints
`ml/configs/data.yaml`. The original folder is left untouched. **Always train/evaluate on the
stratified split**; re-run the script if the dataset is re-downloaded. The box-count table above
is unchanged (those are dataset-wide totals).

---

## 8. Module Specifications

### Module 1 — AI Multi-Target Waste Detection (Standalone YOLO)
- **Goal:** Train a single-stage YOLO11 model to localise + classify multiple waste items in cluttered, real-world images.
- **Input:** user-uploaded image (complex background, single or multiple targets).
- **Output:** array of `{ class, confidence, bbox:[x1,y1,x2,y2] }`.
- **Dataset:** Kaggle `viswaprakash1990/garbage-detection` (see §7 for locked facts).
- **Tasks:** dataset validation + YAML mapping; real-time augmentation (spatial transforms, colour jitter, mosaic); transfer learning from `yolo11n.pt` tuned for 4 GB VRAM; validation + checkpointing; export to `.pt` / ONNX.
- **Evaluation metrics:** mAP@0.5, mAP@0.5:0.95, precision, recall, F1-score, confusion matrix.
- **GTX 1650 guidance:** start with `imgsz=640`, `batch=8` (drop to 4 if OOM), `model=yolo11n.pt`. **Keep AMP OFF (`amp=False`)** — the GTX 16-series produces NaN losses under FP16; FP32 batch=8 fits in ~2.8 GB (verified). Reduce batch before reducing image size if memory errors occur.
- **Step 3 training (DONE):** launch with `python ml/scripts/train.py` (CLI overrides: `--epochs --batch --imgsz --device --amp/--no-amp --resume`; `--resume` continues from `ml/runs/waste_yolo11n/weights/last.pt`). Final CFG: `model=yolo11n.pt`, `data=ml/configs/data.yaml`, `epochs=50`, `imgsz=640`, `batch=8`, `device=0` (auto-falls-back to `cpu` if no CUDA), `workers=2`, `amp=False` (FP32 — GTX 16-series NaN under FP16; batch=8 peaks ~2.8 GB), `patience=15`, `save_period=5`; augmentation `mosaic=1.0`, `mixup=0.1`, `close_mosaic=10`, `degrees=10`, `flipud=0.3`, `fliplr=0.5`, `translate=0.1`, `scale=0.5`, `hsv_h/s/v=0.015/0.7/0.4`. **No `copy_paste`** — it is a no-op on bbox-only data. Expected time on the GTX 1650: **~6–8 h for 50 epochs** (~8 min/epoch in FP32; early stopping at `patience=15` often ends it sooner). Outputs land in `ml/runs/waste_yolo11n/weights/best.pt` (plus `last.pt`).
- **Step 4 evaluation + serving (DONE):** `evaluate.py` (metrics on the TEST split), `export.py` (copies chosen weights → `models/best.pt`, verifies locked class order), `detection_service.run_detection` (cached singleton model, never per-request), and thin `POST /api/predict`. **Production model = `waste_yolo11n_v2`** (trained on the stratified split). **Held-out TEST metrics:** overall **mAP50 0.671**, mAP50-95 0.467, precision 0.774, recall 0.567, F1 0.655. Per-class mAP50: GLASS 0.795, METAL 0.698, PLASTIC 0.695, PAPER 0.672, BIODEGRADABLE 0.628, CARDBOARD 0.541. Artifacts in `ml/runs/waste_yolo11n_v2/eval/` (metrics.json/csv + confusion_matrix.png). Recall is the weak axis (precise but misses ~43% of objects); the biggest future accuracy lever is a larger model (`yolo11s`, try `--batch 4` on 4 GB) — do it as a separate optional run, not by resuming.

### Module 2 — Carbon Impact Estimation (External API, NO ML)
- After detection, ask the user (per item) for estimated weight (kg) and a geographic location (ISO country code, e.g. `MY`).
- Call Climatiq asynchronously per item using **material type + weight + country**.
- Map each YOLO class to an appropriate Climatiq material emission factor (maintained in a lookup in `carbon_service`).
- Aggregate per-item CO2e into a total.
- Return per-item breakdown + total (kg CO2e).
- Must handle: missing/invalid weight, unknown material mapping, API timeout/error, missing API key — all as clean, user-facing errors via `ApiError`.

### Module 3 — Recommendation System (rule-based core + optional LLM)
- Input: list of detected waste types + their carbon values.
- **Default path:** deterministic rule-based engine — a maintainable, modular mapping from class -> structured disposal guidance (e.g. Plastic -> "Blue recycling bin; rinse before disposal; avoid single-use plastics where alternatives exist.").
- **Optional path:** if `LLM_API_KEY` is set, enrich/contextualise recommendations via an LLM acting as a sustainability consultant. The system MUST degrade gracefully to rule-based output if the LLM call fails or no key is present.
- Output: per-item actionable recommendation(s).

### Module 4 — Web Application
- Single-page app, clean responsive UI, MVC-where-appropriate.
- Flow: Home dashboard -> drag-drop upload -> async AJAX to Flask -> YOLO pipeline -> JSON of detected items -> dynamic render of items with bounding boxes -> user enters weight + location -> async carbon fetch -> aggregated results dashboard (carbon breakdown + recommendation list).
- Frontend: Tailwind for layout, GSAP for transitions, Fetch API for async calls.

---

## 9. Coding Conventions & Best Practices

- **Thin controllers:** route handlers in `blueprints/` only validate input, call a service, and return JSON. No business logic in routes.
- **Services own logic:** all detection / carbon / recommendation logic lives in `app/services/`. Services are import-safe and unit-testable in isolation.
- **Config:** class-based in `config.py` (`DevelopmentConfig`, `ProductionConfig`, `TestingConfig`). The factory selects via `FLASK_ENV`. Never hardcode secrets — read from `.env` through `config.py`.
- **App factory:** `create_app(env_name)` builds the app. No global `app` object. This keeps tests isolated.
- **Errors:** services raise `app.utils.errors.ApiError(message, status_code)`. A registered handler converts it to JSON. Avoid scattering try/except across controllers.
- **Validation:** validate request payloads with pydantic schemas in `app/schemas/` before touching a service.
- **Secrets:** `.env` is gitignored. `.env.example` documents required keys with blank values.
- **External calls:** wrap all `requests` calls with timeouts and explicit error handling; never let an upstream failure 500 silently.
- **Logging:** use `app.logger` / module loggers, not `print`.
- **Tests:** every service gets unit tests; mock external APIs (Climatiq, LLM) with `pytest-mock` — tests must not hit the network or require a GPU.
- **Style:** clear names, docstrings on every module/function explaining intent, small functions.

---

## 10. Environment Variables (`.env`)

```
FLASK_ENV=development
SECRET_KEY=<long random string; generate with: python -c "import secrets; print(secrets.token_hex(32))">

CARBON_PROVIDER=climatiq          # or carbon_interface
CLIMATIQ_API_KEY=<from climatiq.io dashboard — needed in Step 5>
CARBON_INTERFACE_API_KEY=         # only if using that provider

LLM_API_KEY=                      # OPTIONAL — leave blank for rule-based only
LLM_MODEL=claude-sonnet-4-6

MODEL_PATH=models/best.pt
CONFIDENCE_THRESHOLD=0.35
INFERENCE_DEVICE=cpu              # "cpu" (safe default for serving) or "0" for GPU 0
DATABASE_URL=sqlite:///waste_app.db
```

Notes:
- For Step 1 setup, **all keys can be blank** — the app boots and tests pass without them.
- Climatiq key is only required once Module 2 is wired up.
- LLM key is only required if the optional enrichment layer is enabled.

---

## 11. Common Commands

> Windows / PowerShell. Do NOT use `&&` or `source`.

```powershell
# Create & activate the 3.11 virtualenv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked: Set-ExecutionPolicy -Scope Process -Bypass   (then retry)

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
# GPU build of torch for the GTX 1650 (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Run the app
copy .env.example .env        # then edit .env
python run.py                 # http://127.0.0.1:5000

# Tests
pytest -q

# Verify GPU is visible
python -c "import torch; print(torch.cuda.is_available())"   # expect True

# Dataset download (Kaggle CLI)
python -m kaggle datasets download -d viswaprakash1990/garbage-detection -p ml/data --unzip

# Dataset scripts (Step 2)
python ml/scripts/inspect_dataset.py
python ml/scripts/prepare_dataset.py --dry-run
python ml/scripts/prepare_dataset.py

# Fix the broken class split (Step 4) -> ml/data/garbage_stratified/ + repoint data.yaml
python ml/scripts/restratify_dataset.py --dry-run
python ml/scripts/restratify_dataset.py

# ML pipeline (Steps 3–4)
python ml/scripts/train.py
python ml/scripts/evaluate.py
python ml/scripts/export.py
```

**Sanity check you're inside the venv:** the prompt should start with `(.venv)` and pip should NOT say "Defaulting to user installation."

---

## 12. Build Roadmap

Each completed step gets its own commit + push — see §15 for the commit convention.

| Step | Description | Status |
|---|---|---|
| 1 | Project foundation — factory, config, blueprints, DB, errors, tests | **DONE** |
| 2 | Dataset prep — inspection, validation, `data.yaml`, balance report | **DONE** |
| 3 | YOLO training — transfer learning, augmentation, GTX-1650 settings | **DONE** |
| 4 | Evaluation + export + `detection_service` + `POST /api/predict` | **DONE** |
| 5 | Carbon module — `carbon_service` + `POST /api/calculate-impact` | **Next** |
| 6 | Recommendation module — rule-based + optional LLM enrichment | Pending |
| 7 | Frontend — drag-drop, bounding-box canvas, weight forms, results dashboard | Pending |
| 8 | Full test suite, gunicorn deployment guide, FYP documentation | Pending |

---

## 13. Deliverables Checklist

- [ ] Upload multi-object real-world images via a responsive drop-zone.
- [ ] Auto-draw bounding boxes over detected waste in the UI.
- [ ] Show per-item classification confidence scores.
- [ ] Dynamic per-item weight inputs on the canvas.
- [ ] Async real-time external Carbon Emission API calls.
- [ ] Display total and per-item carbon footprints (kg CO2e).
- [ ] Show actionable, structured recycling/disposal instructions.
- [ ] Modern, responsive UI following good UX practices.

---

## 14. Gotchas & Guardrails for Claude Code

- **Never** suggest Python 3.13 or bash-only commands for this Windows/PowerShell user.
- **Never** write `yolov11n.pt`; it's `yolo11n.pt`.
- **Never** put business logic in route handlers — services only.
- **Never** commit `.env`, model weights (`*.pt`), or the dataset (`ml/data/`).
- **Never** make tests depend on the network or a GPU — mock external APIs.
- **Never** reorder the class list — `0:BIODEGRADABLE` … `5:PLASTIC` is locked to the dataset (§6).
- The app must run end-to-end **without** an LLM key (rule-based fallback).
- Respect the 4 GB VRAM limit: prefer the `n` (nano) model, modest batch sizes; reduce batch before image size on OOM.
- **GTX 16-series (1650/1660) + AMP = NaN losses.** Train in FP32 (`amp=False`, the default in `train.py`). Symptom: all losses (`box`/`cls`/`dfl`) go `nan` from epoch 1 while warmup LR is still tiny, collapsing mAP to noise. FP32 does NOT blow the 4 GB budget here (batch=8 peaks ~2.8 GB). Use `--amp` only on a newer GPU with working FP16.
- Keep carbon estimation free of any trained model — it's API-only by design.
- Dataset folder name contains a space (`"GARBAGE CLASSIFICATION"`) — always quote the path.
- Ultralytics resolves a **relative** `path:` in `data.yaml` against its global `datasets_dir` setting, NOT the yaml's location. `train.py` compensates by rewriting the dataset root to an absolute path at runtime (`resolve_data_config`, emits a temp `*.resolved.yaml`). Keep `ml/configs/data.yaml` portable (relative `path: ../data/GARBAGE CLASSIFICATION`) — do **not** hardcode an absolute path into the committed yaml.
- When adding a dependency, pin its version in `requirements.txt`.

---

## 15. Git, Security & Commit Workflow

**Remote:** `https://github.com/K01J10W19/ECO-WASTE-AI.git` (default branch: `main`).

### 15.1 Security — never leak secrets or data
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

### 15.2 Commit convention — one commit per completed roadmap step
Each completed step in the **Build Roadmap (§12)** gets its own commit. The message
describes the action performed (what was built and why it satisfies that step):

```
Step <N>: <short action title>

- <bullet describing what was implemented>
- <bullet describing key files / decisions>
- Roadmap §12 Step <N> complete; deliverables: <which §13 items this advances>
```

Example:
```
Step 1: Project foundation — app factory, config, error handling, tests

- create_app() factory + class-based config (Dev/Prod/Testing)
- Blueprints (main SPA shell + JSON api), /api/health returns 200
- ApiError + registered JSON error handler; pytest harness green
- Roadmap §12 Step 1 complete
```

### 15.3 Push workflow (Windows / PowerShell)
```powershell
# one-time
git init
git branch -M main
git remote add origin https://github.com/K01J10W19/ECO-WASTE-AI.git

# per completed roadmap step
git add -A
git status                       # verify NO secrets/data staged
git commit -m "Step N: <action description>"
git push -u origin main          # subsequent pushes: git push
```
- Authentication uses **Git Credential Manager** (already configured system-wide).
- Commit & push **only when a roadmap step is genuinely complete** (code + tests pass),
  not for half-finished work. Keep history clean and auditable for the examiner.
