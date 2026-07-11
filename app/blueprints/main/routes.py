"""HTML page routes (server-rendered shell; logic lives in /api)."""
from flask import Blueprint, render_template

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    """Single-page app shell."""
    return render_template("index.html")


@main_bp.route("/carbon-lab")
def carbon_lab():
    """Manual tester for Module 2 (/api/calculate-impact) + Module 3 (/api/recommend)."""
    return render_template("carbon_lab.html")
