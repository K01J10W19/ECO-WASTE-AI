"""
export.py  -  Step 4: package the trained model for the web app.

Copies the chosen checkpoint to ``models/best.pt`` (the path the Flask app loads
via ``MODEL_PATH``) and verifies it loads and exposes the LOCKED class order.
Optionally exports an ONNX copy as a bonus artifact.

Run from the project root (Windows / PowerShell)::

    python ml/scripts/export.py            # copy best.pt -> models/best.pt (+ verify)
    python ml/scripts/export.py --onnx     # also export models/best.onnx

The app always loads the ``.pt`` through Ultralytics; ONNX is an optional extra.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

# This file lives at ml/scripts/; parents[2] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = PROJECT_ROOT / "ml" / "runs" / "waste_yolo11n_v2" / "weights" / "best.pt"
MODELS_DIR = PROJECT_ROOT / "models"
APP_MODEL_PATH = MODELS_DIR / "best.pt"

# LOCKED class order (CLAUDE.md §6) — the exported model MUST match this exactly.
LOCKED_CLASS_NAMES = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]

logger = logging.getLogger("export")


def copy_weights(weights: Path) -> Path:
    """Copy ``weights`` to models/best.pt (creating models/). Returns the target."""
    if not weights.exists():
        raise SystemExit(
            f"ERROR: weights not found at {weights}\n"
            f"Train first (python ml/scripts/train.py) or pass --weights."
        )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, APP_MODEL_PATH)
    logger.info("Copied %s -> %s", weights, APP_MODEL_PATH)
    return APP_MODEL_PATH


def verify_model(model_path: Path) -> None:
    """
    Load the copied model and assert its class names match the LOCKED order.
    Fails loudly (SystemExit) on any mismatch so a bad model can't reach the app.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = [model.names[i] for i in range(len(model.names))]
    if names != LOCKED_CLASS_NAMES:
        raise SystemExit(
            "FATAL: exported model class order does not match the locked order.\n"
            f"  expected: {LOCKED_CLASS_NAMES}\n"
            f"  got     : {names}\n"
            "Do NOT deploy this model — retrain with the correct data.yaml."
        )
    logger.info("Verified class order: %s", names)


def export_onnx(weights: Path) -> Path:
    """Export an ONNX copy into models/. Returns the destination path."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    onnx_src = Path(model.export(format="onnx", imgsz=640, opset=12))
    onnx_dst = MODELS_DIR / "best.onnx"
    shutil.move(str(onnx_src), str(onnx_dst))
    logger.info("Exported ONNX -> %s", onnx_dst)
    return onnx_dst


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Package the trained model for the app (Step 4).")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help=f"Checkpoint to package (default: {DEFAULT_WEIGHTS}).")
    parser.add_argument("--onnx", action="store_true",
                        help="Also export models/best.onnx (bonus artifact).")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    target = copy_weights(args.weights)
    verify_model(target)

    print("\n=== EXPORT COMPLETE ===")
    print(f"  app model : {target.resolve()}   (MODEL_PATH)")

    if args.onnx:
        onnx_path = export_onnx(args.weights)
        print(f"  onnx      : {onnx_path.resolve()}   (bonus; app uses the .pt)")

    print("\nThe app will load this model at MODEL_PATH. Next: wire up the frontend / API.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
