"""SQLite model for optional scan-history tracking."""
from datetime import datetime
from app.extensions import db


class Scan(db.Model):
    __tablename__ = "scans"

    id = db.Column(db.Integer, primary_key=True)
    image_filename = db.Column(db.String(255), nullable=False)
    detected_items = db.Column(db.JSON, nullable=False)      # raw detection array
    total_co2e = db.Column(db.Float, nullable=True)          # kg CO2e
    location = db.Column(db.String(8), nullable=True)        # ISO country code
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "image_filename": self.image_filename,
            "detected_items": self.detected_items,
            "total_co2e": self.total_co2e,
            "location": self.location,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
