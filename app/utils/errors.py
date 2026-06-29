"""
Centralised error handling.

We define a small ApiError exception that services raise on expected
failures (bad input, upstream API down, etc.). Controllers don't need
try/except everywhere — the handler turns ApiError into clean JSON.
"""
from flask import jsonify


class ApiError(Exception):
    """Raise for expected, client-facing failures."""
    def __init__(self, message: str, status_code: int = 400, details=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def register_error_handlers(app):
    @app.errorhandler(ApiError)
    def handle_api_error(err: ApiError):
        return jsonify(error=err.message, details=err.details), err.status_code

    @app.errorhandler(413)
    def handle_too_large(err):
        return jsonify(error="File too large. Max upload size is 10 MB."), 413

    @app.errorhandler(404)
    def handle_not_found(err):
        return jsonify(error="Resource not found."), 404

    @app.errorhandler(500)
    def handle_server_error(err):
        app.logger.exception("Unhandled server error")
        return jsonify(error="Internal server error."), 500
