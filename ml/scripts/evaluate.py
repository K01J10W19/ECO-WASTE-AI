"""
evaluate.py  -  Step 4: evaluate the trained YOLO11 model on the TEST split.

Loads the trained weights and runs Ultralytics validation on the **held-out
test** split (not validation) so the reported numbers are an honest estimate of
generalisation. Prints an overall + per-class metrics table (mAP@0.5,
mAP@0.5:0.95, precision, recall, F1) and saves machine- and human-readable
copies (``metrics.json`` + ``metrics.csv``) for the FYP report.

Run from the project root (Windows / PowerShell)::

    python ml/scripts/evaluate.py                         # test split, default weights
    python ml/scripts/evaluate.py --weights path/to.pt    # a different checkpoint
    python ml/scripts/evaluate.py --split val             # measure the val split instead

Ultralytics also auto-saves confusion_matrix.png / PR_curve.png / F1_curve.png
into the evaluation folder; the path is printed at the end.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# This file lives at ml/scripts/; parents[2] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# ml/scripts is on sys.path when run as a script, so we can reuse train.py's
# dataset-path resolver (Ultralytics mis-resolves a relative data.yaml path).
sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA_YAML = str(PROJECT_ROOT / "ml" / "configs" / "data.yaml")
# The production model is the balanced-split retrain (see CLAUDE.md §7).
DEFAULT_WEIGHTS = PROJECT_ROOT / "ml" / "runs" / "waste_yolo11n_v2" / "weights" / "best.pt"

# LOCKED class order (CLAUDE.md §6) — used only for readable labelling.
CLASS_NAMES = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]

logger = logging.getLogger("evaluate")


def _f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall; 0.0 when both are 0."""
    denom = precision + recall
    return (2 * precision * recall / denom) if denom else 0.0


def collect_metrics(result, names: dict) -> dict:
    """
    Turn an Ultralytics validation ``result`` into a plain dict of overall and
    per-class metrics (mAP50, mAP50-95, precision, recall, F1).
    """
    box = result.box
    overall = {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "f1": _f1(float(box.mp), float(box.mr)),
    }
    per_class = []
    for k, class_id in enumerate(box.ap_class_index):
        p, r = float(box.p[k]), float(box.r[k])
        per_class.append({
            "class": names.get(int(class_id), str(class_id)),
            "mAP50": float(box.ap50[k]),
            "mAP50_95": float(box.ap[k]),
            "precision": p,
            "recall": r,
            "f1": _f1(p, r),
        })
    return {"overall": overall, "per_class": per_class}


def print_table(metrics: dict) -> None:
    """Pretty-print the overall + per-class metrics table to stdout."""
    o = metrics["overall"]
    print("\n=== TEST-SET METRICS ===")
    header = f"{'class':<15}{'mAP50':>9}{'mAP50-95':>10}{'precision':>11}{'recall':>9}{'F1':>8}"
    print(header)
    print("-" * len(header))
    print(f"{'ALL':<15}{o['mAP50']:>9.3f}{o['mAP50_95']:>10.3f}"
          f"{o['precision']:>11.3f}{o['recall']:>9.3f}{o['f1']:>8.3f}")
    print("-" * len(header))
    for c in metrics["per_class"]:
        print(f"{c['class']:<15}{c['mAP50']:>9.3f}{c['mAP50_95']:>10.3f}"
              f"{c['precision']:>11.3f}{c['recall']:>9.3f}{c['f1']:>8.3f}")


def flag_weak_classes(metrics: dict, rel_gap: float = 0.15) -> None:
    """Flag any class whose mAP50 is >``rel_gap`` below the best class's mAP50."""
    per_class = metrics["per_class"]
    if not per_class:
        return
    best = max(per_class, key=lambda c: c["mAP50"])
    threshold = best["mAP50"] * (1 - rel_gap)
    weak = [c for c in per_class if c["mAP50"] < threshold]
    print("\n=== INTERPRETATION ===")
    print(f"Best class: {best['class']} (mAP50={best['mAP50']:.3f}).")
    if weak:
        names = ", ".join(f"{c['class']} ({c['mAP50']:.3f})" for c in weak)
        print(f"Classes >{int(rel_gap * 100)}% below the best (discuss in the report): {names}")
    else:
        print(f"No class is more than {int(rel_gap * 100)}% below the best — well balanced.")


def save_metrics(metrics: dict, out_dir: Path, meta: dict) -> tuple[Path, Path]:
    """Write metrics.json and metrics.csv into ``out_dir``; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {**meta, **metrics}

    json_path = out_dir / "metrics.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class", "mAP50", "mAP50_95", "precision", "recall", "f1"])
        o = metrics["overall"]
        writer.writerow(["ALL", o["mAP50"], o["mAP50_95"], o["precision"], o["recall"], o["f1"]])
        for c in metrics["per_class"]:
            writer.writerow([c["class"], c["mAP50"], c["mAP50_95"],
                             c["precision"], c["recall"], c["f1"]])
    return json_path, csv_path


def evaluate(weights: Path, split: str, imgsz: int) -> int:
    """Run validation on ``split`` and persist metrics. Returns an exit code."""
    if not weights.exists():
        print(f"ERROR: weights not found at {weights}\n"
              f"Train first (python ml/scripts/train.py) or pass --weights.")
        return 1

    from ultralytics import YOLO
    import train  # reuse the dataset-path resolver

    data = train.resolve_data_config(DATA_YAML)
    device = train.resolve_device(0)  # GPU 0 if available, else cpu

    run_dir = weights.parent.parent          # ml/runs/<name>/
    logger.info("Evaluating %s on the '%s' split (imgsz=%d, device=%s)",
                weights, split, imgsz, device)

    model = YOLO(str(weights))
    result = model.val(
        data=data, split=split, imgsz=imgsz, device=device,
        plots=True, project=str(run_dir), name="eval", exist_ok=True, verbose=False,
    )

    metrics = collect_metrics(result, model.names)
    print_table(metrics)
    flag_weak_classes(metrics)

    save_dir = Path(getattr(result, "save_dir", run_dir / "eval"))
    meta = {"weights": str(weights), "split": split, "imgsz": imgsz}
    json_path, csv_path = save_metrics(metrics, save_dir, meta)

    print("\n=== ARTIFACTS ===")
    print(f"  metrics.json : {json_path.resolve()}")
    print(f"  metrics.csv  : {csv_path.resolve()}")
    print(f"  plots (confusion_matrix.png / PR_curve.png / F1_curve.png): {save_dir.resolve()}")
    print("\nNext, package the model for the app:")
    print("    python ml/scripts/export.py")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate the trained YOLO11 model (Step 4).")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help=f"Path to weights (default: {DEFAULT_WEIGHTS}).")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"],
                        help="Dataset split to evaluate (default: test).")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    return evaluate(args.weights, args.split, args.imgsz)


if __name__ == "__main__":
    sys.exit(main())
