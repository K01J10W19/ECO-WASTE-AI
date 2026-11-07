# CarbIQ — Deep-Learning Multi-Target Waste Detection & Carbon Impact Estimation

A **100 % local, stateless** web application. Upload a photo containing one or more
waste items and CarbIQ runs a **Dual-Tower Hybrid** pipeline to:

1. **Detect & localise** every waste item (Stage 1 — a specialist YOLOv8-N waste
   object detector),
2. **Classify the material** of each item (Stage 2 — a TrashNet-fine-tuned Vision
   Transformer, with a classical-CV "Method B" physics tie-breaker for the hard
   plastic-vs-glass calls),
3. **Estimate the carbon footprint** per item (size-aware local proxy at upload, then
   a region-scoped Climatiq audit on user-verified weights), and
4. **Recommend disposal** — the Decision-Making Module ranks three end-of-life routes
   by ascending CO₂e and returns child-simple, country-localised guidance.

The app is **completely in-memory: no database, no accounts, no persistence.** It also
runs **end-to-end without any API keys** — every external integration (Climatiq, LLM
text layer, YouTube tutorials) degrades gracefully to a local, offline fallback.

---

## 1. System / Environment Requirements

| Requirement | Detail |
|---|---|
| **OS** | Windows 10/11, macOS, or Linux |
| **Python** | **3.11.x** (use 3.11 — 3.13 lacks prebuilt numpy/torch wheels on Windows) |
| **Disk** | ~1.5 GB free (bundled detector ~64 MB + the ViT classifier ~343 MB downloaded on first run) |
| **Internet** | Needed **once**, on the first `/api/predict`, to download the ViT classifier into the Hugging Face cache. Afterwards the app is fully offline-capable. |
| **GPU** | Optional. The default is CPU (works fine). An NVIDIA CUDA GPU is auto-used if you set `INFERENCE_DEVICE=0` and install the CUDA build of torch (see §4). |

---

## 2. Install Dependencies

**Windows (PowerShell)** — do NOT use `&&` or `source`:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked: Set-ExecutionPolicy -Scope Process -Bypass  (then retry)
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux (bash)**:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`.env` holds secrets, so it is **not committed** to the repository. Create your own
from the template (every key blank → local fallbacks, so the app runs immediately):

```powershell
copy .env.example .env        # macOS / Linux: cp .env.example .env
```

To enable the live Climatiq / LLM / YouTube features, paste your own keys into `.env`
(see `.env.example` for every option). When **deploying to a hosting platform**, do not
upload `.env` — set the same variables in the platform's environment-variable settings
instead. Every setting has a safe built-in default, so the app also starts with no
`.env` at all. **No keys are required to run or grade the system.**

---

## 3. Run the Application

```bash
python run.py
```
Then open **http://127.0.0.1:5000** in a browser (Google Chrome recommended).

Drag-and-drop an image (or use the live camera) to run a detection. The first request
downloads the ViT classifier (~343 MB, one time) — subsequent runs are instant.

**Run the test suite** (fully mocked — no network, weights, or GPU needed):
```bash
pytest -q
```

---

## 4. (Optional) GPU Acceleration

The default torch wheel is CPU-only and works out of the box. For an NVIDIA GPU
(e.g. GTX 1650, CUDA 12.1), after installing requirements:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Then set `INFERENCE_DEVICE=0` in `.env`. Verify with:
```bash
python -c "import torch; print(torch.cuda.is_available())"   # -> True
```

---

## 5. Model Weights

- **`models/yolov8n-waste-det.pt`** — the Stage-1 specialist waste detector (**bundled**,
  ~6 MB). Required; already in place.
- **`models/yolov8m-seg-trash.pt`** — an optional A/B segmentation locator (bundled).
- The **Stage-2 ViT classifier** (`edwinpalegre/ee8225-group4-vit-trashnet-enhanced`,
  ~343 MB) is **not bundled** — it downloads automatically from Hugging Face on first
  run and is cached outside the project.

Large A/B-baseline weights used only for report comparisons (e.g. `yolo26x-seg.pt`,
`rtdetr-x.pt`) are intentionally **not included** to keep this archive small; they
auto-download from Ultralytics if you ever point `MODEL_PATH` at them.

---

## 6. Project Layout

```
.
├── run.py                 # entry point → python run.py
├── config.py              # class-based config (Dev / Prod / Testing)
├── requirements.txt
├── Dockerfile             # production image (CPU torch, models baked in, gunicorn)
├── deploy_azure.py        # one-command deploy to Azure Container Apps
├── .github/workflows/     # builds the Docker image in the cloud → ghcr.io
├── .env / .env.example    # configuration (keys blank = local fallbacks)
├── app/                   # Flask app: factory, blueprints, services, SPA
│   ├── blueprints/        #   api routes (/predict, /calculate-impact, /recommend, /health)
│   ├── services/          #   detection, classification, carbon, recommendation, youtube
│   ├── static/            #   SPA (css, js, vendored Tailwind/GSAP, fonts)
│   └── templates/         #   index.html (SPA) + carbon_lab.html (API tester)
├── models/                # bundled model weights (Stage-1 detector)
├── ml/                    # legacy training scripts/notebooks (archived, not run by app)
└── tests/                 # pytest suite (mocked)
```

---

## 7. Deploy — Azure Container Apps (Azure for Students)

The two models peak at about 0.9 GB of RAM, so 512 MB free web tiers (e.g. Render's)
cannot run the app. It runs on **Azure Container Apps** at 1 vCPU / 2 GiB, scaled to
zero when idle — covered by the Azure for Students credit (no credit card).

**How it fits together:** every push to `main` that touches the app makes GitHub
Actions build the `Dockerfile` (CPU torch, both models baked in) and publish it to
`ghcr.io/k01j10w19/carbiq` — no local Docker needed. `deploy_azure.py` then tells
Azure to run that image.

**One-time setup**
1. Activate [Azure for Students](https://azure.microsoft.com/free/students) with your
   university email.
2. Install the [Azure CLI](https://aka.ms/installazurecliwindowsx64), open a **new**
   terminal, and run `az login`.
3. After the first GitHub Actions build finishes (repo → **Actions** tab), check the
   package is public: GitHub profile → **Packages** → `carbiq` → **Package settings**
   → *Change visibility* → **Public** (Azure pulls it without a password; the image
   contains no secrets).

**Deploy / redeploy** (inside the venv):
```powershell
python deploy_azure.py              # app only, offline fallbacks
python deploy_azure.py --secrets    # also copy the API keys from your .env
```
The script prints the app URL (`https://carbiq.<random>.southeastasia.azurecontainerapps.io`).
To ship a change: push to `main`, wait for the Actions build, run the script again.

**Behaviour:** after a period with no visitors the app scales to zero (costs nothing);
the next visit takes about a minute while the container starts and the models load.
Uploaded images and the SQLite scan history live on the container's temporary disk
and reset on restart — the app is stateless by design. Live logs:
`az containerapp logs show -n carbiq -g carbiq-rg --follow`.

The same `Dockerfile` runs on any Docker host (it honours an injected `$PORT`).

---

## 8. Troubleshooting

- **`python` opens the Microsoft Store (Windows):** use `py -3.11` explicitly.
- **First detection is slow / needs internet:** that is the one-time ViT download.
  Once cached, it is instant and offline.
- **`torch` / CUDA error on a machine without an NVIDIA GPU:** ensure `.env` has
  `INFERENCE_DEVICE=cpu` (the shipped default).
- **The app still works with no keys** — blank Climatiq/LLM/YouTube keys are expected
  and switch the app to its deterministic local fallbacks.
