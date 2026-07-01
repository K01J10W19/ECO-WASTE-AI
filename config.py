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

    # --- Model / inference ---
    MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models", "best.pt"))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.35))
    # The fixed class list the model is trained on (order matters for YOLO).
    # ALL-CAPS to match the dataset's data.yaml and ml/configs/data.yaml exactly
    # (LOCKED — see CLAUDE.md §6 & §7). The model uses these names internally.
    CLASS_NAMES = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
    # The web UI never shows the raw ALL-CAPS name — it maps to a friendly label.
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

    # --- LLM (optional enrichment layer) ---
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")


class DevelopmentConfig(BaseConfig):
    DEBUG = True


class ProductionConfig(BaseConfig):
    DEBUG = False


class TestingConfig(BaseConfig):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}
