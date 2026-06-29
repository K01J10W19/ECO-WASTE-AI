# Deep Learning Multi-Target Waste Detection & Carbon Impact Estimation

A web application that detects multiple waste items in a single real-world
image (YOLO11), estimates their carbon footprint via a real-time Carbon
Emission API, and returns tailored disposal recommendations.

## Modules
1. **Detection** — Standalone YOLO11 object detection (Plastic, Paper, Glass,
   Metal, Cardboard, Biodegradable).
2. **Carbon Impact** — Real-time external API (Climatiq / Carbon Interface).
3. **Recommendations** — Rule-based engine (+ optional LLM enrichment).
4. **Web App** — Flask (blueprints + services), Tailwind + JS frontend.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then fill in API keys
python run.py                 # http://127.0.0.1:5000
```

## GPU setup (GTX 1650, CUDA)
The default torch wheel is CPU-only. After `pip install -r requirements.txt`:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Verify: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.

## Project layout
- `app/` — Flask application (factory, blueprints, services, models, templates)
- `ml/`  — dataset prep, training, evaluation, export scripts
- `models/` — exported weights consumed by the app at inference time
- `tests/` — pytest suite
- `docs/` — FYP documentation

## Testing
```bash
pytest -q
```
