"""
Central configuration.

We use class-based config so different environments (dev / prod / test)
share defaults but can override only what they need. The app factory
(app/__init__.py) picks a class based on FLASK_ENV.
"""
import os
from dotenv import load_dotenv

# Load variables from .env into os.environ exactly once, at import time.
load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class BaseConfig:
    # --- Flask core ---
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-insecure-key")

    # --- Database ---
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'waste_app.db')}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- File uploads ---
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "app", "static", "uploads")
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB max upload
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

    # --- Model / inference (Dual-Tower Hybrid: waste YOLO detector + TrashNet ViT) ---
    # Stage 1 locator: a SPECIALIZED waste OBJECT-DETECTION model (YOLOv8-N
    # fine-tuned on a blended universal waste corpus of wild litter +
    # household recyclables; GitHub gianlucasposito/YOLO-Waste-Detection, MIT).
    # Its latent space only knows waste, so background objects rarely fire,
    # and the nano backbone keeps edge latency minimal. The file auto-downloads
    # on first load if missing. Its 5 coarse labels are diagnostics only
    # (located_as) — the ViT decides material. The geometric BOX AREA drives
    # the dynamic carbon scaling (gamma recalibrated in carbon_service).
    # A/B alternatives that still resolve here: models/yolov8m-seg-trash.pt
    # (v3.1 segmenter) or a bare official name like "yolo26x-seg.pt".
    MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "yolov8n-waste-det.pt"))
    # Stage 2 material classifier: supervised ViT fine-tuned on TrashNet-enhanced
    # (Hugging Face model id; native labels map onto the 7-class taxonomy in
    # classification_service.py).
    VIT_MODEL_NAME = os.getenv("VIT_MODEL_NAME",
                               "edwinpalegre/ee8225-group4-vit-trashnet-enhanced")
    # Stage 1 is recall-first: a LOW threshold captures crushed/deformed items;
    # per-item certainty comes from Stage 2's material scores instead.
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.15))
    # Device used for BOTH stages at serving time. "cpu" is the safe default for
    # a web server (no CUDA needed, no contention); set to "0" to use GPU 0.
    INFERENCE_DEVICE = os.getenv("INFERENCE_DEVICE", "cpu")

    # --- LEGACY (archived custom-training pipeline; see CLAUDE.md) ---
    # Kept ONLY for the retired ml/ training scripts and their tests, which are
    # preserved for the FYP report. The serving pipeline no longer reads these.
    CLASS_NAMES = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
    APP_CLASS_DISPLAY_NAMES = {
        "BIODEGRADABLE": "Biodegradable",
        "CARDBOARD": "Cardboard",
        "GLASS": "Glass",
        "METAL": "Metal",
        "PAPER": "Paper",
        "PLASTIC": "Plastic",
    }

    # --- Carbon API ---
    CARBON_PROVIDER = os.getenv("CARBON_PROVIDER", "climatiq")
    CLIMATIQ_API_KEY = os.getenv("CLIMATIQ_API_KEY", "")
    CARBON_INTERFACE_API_KEY = os.getenv("CARBON_INTERFACE_API_KEY", "")

    # --- LLM (optional v3.6 child-friendly, country-localized text layer) ---
    # Any OpenAI-compatible chat-completions endpoint works; free tiers:
    #   Groq        https://api.groq.com/openai/v1/chat/completions   (default)
    #   OpenRouter  https://openrouter.ai/api/v1/chat/completions
    #   Gemini      https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
    #   Ollama      http://localhost:11434/v1/chat/completions        (fully local)
    # Blank LLM_API_KEY = deterministic local knowledge-grid text (always works).
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    LLM_API_URL = os.getenv("LLM_API_URL",
                            "https://api.groq.com/openai/v1/chat/completions")

    # --- YouTube Data API v3 (optional Action-Protocol tutorial videos) ---
    # Resolves the LLM's localized video_search_query to ONE live tutorial
    # embed. Blank key = the verified universal fallback video (no network).
    # Free quota: 10,000 units/day; each search costs 100 — the service
    # caches per query so repeated scans are free.
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")


class DevelopmentConfig(BaseConfig):
    DEBUG = True


class ProductionConfig(BaseConfig):
    DEBUG = False


class TestingConfig(BaseConfig):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False
    # Tests must be hermetic: never inherit real API keys from .env.
    # Tests that exercise the live paths set these explicitly (HTTP mocked).
    CLIMATIQ_API_KEY = ""
    LLM_API_KEY = ""
    YOUTUBE_API_KEY = ""


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}
