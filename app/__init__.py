"""
Application factory.

create_app() builds and returns a configured Flask app. Using a factory
(instead of a global app object) is the standard pattern for testable,
multi-environment Flask projects.
"""
import os
import logging
from flask import Flask

from config import config_by_name
from app.extensions import db


def create_app(env_name: str = "development") -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_by_name.get(env_name, config_by_name["development"]))

    # Make sure the upload folder exists at boot.
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # --- Logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- Extensions ---
    db.init_app(app)

    # --- Blueprints ---
    from app.blueprints.main.routes import main_bp
    from app.blueprints.api.routes import api_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    # --- Error handlers ---
    from app.utils.errors import register_error_handlers
    register_error_handlers(app)

    # --- DB tables ---
    with app.app_context():
        from app.models import scan  # noqa: F401  (import so model is registered)
        db.create_all()

    app.logger.info("App created in '%s' mode", env_name)
    return app
