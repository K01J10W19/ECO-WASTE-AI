"""
Flask extension instances are created here (unbound) and initialised inside
the app factory. This avoids circular imports and lets tests build isolated
app instances.
"""
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
