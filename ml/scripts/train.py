"""
train.py  -  Step 3: YOLO11 transfer-learning trainer for waste detection.

Fine-tunes COCO-pretrained ``yolo11n.pt`` on the garbage-detection dataset
(finalised in Step 2, see ``ml/configs/data.yaml``). The configuration is tuned
for a 4 GB GPU (NVIDIA GTX 1650): nano model, ``imgsz=640``, ``batch=8``, mixed
precision. If no CUDA device is present the script falls back to CPU (very slow)
so it never hard-crashes on a machine without a GPU.

Augmentation note
-----------------
``copy_paste`` is INTENTIONALLY OMITTED. In Ultralytics it only has an effect on
SEGMENTATION datasets (it needs polygon masks). Our dataset is bounding-box only,
so ``copy_paste`` would be a silent no-op. To fight the ~10:1 class imbalance we
rely on ``mosaic`` (4-image stitching -> more minority-class exposure per batch)
and a mild ``mixup`` instead, both of which work for detection.

Run from the project root (Windows / PowerShell)::

    python ml/scripts/train.py                     # full 50-epoch run
    python ml/scripts/train.py --batch 4           # if the GPU runs out of memory
    python ml/scripts/train.py --batch 4 --imgsz 512
    python ml/scripts/train.py --resume            # continue from the last checkpoint

Outputs land in ``ml/runs/waste_yolo11n/`` (weights under ``weights/best.pt`` and
``weights/last.pt``).
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

# --------------------------------------------------------------------------- #
# Heavy dependencies (torch, ultralytics) are imported lazily inside the
# functions that need them. That keeps this module import-safe for unit tests
# (which mock ``ultralytics``) and on machines without a GPU build of torch.
# --------------------------------------------------------------------------- #

# This file lives at ml/scripts/; parents[2] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger("train")


# --------------------------------------------------------------------------- #
# Training configuration.
# Every key here is a valid Ultralytics ``model.train()`` argument. Overrides
# for the most common knobs are exposed on the command line (see parse_args).
# --------------------------------------------------------------------------- #
CFG = {
    "model":      "yolo11n.pt",         # COCO-pretrained nano weights (transfer learning)
    "data":       "ml/configs/data.yaml",
    "epochs":     50,
    "imgsz":      640,
    "batch":      8,                    # GTX 1650 4GB; reduce to 4 if OOM
    "device":     0,                    # GPU 0; auto-falls-back to "cpu" if no CUDA
    "workers":    2,                    # keep low on Windows (DataLoader stability)
    "project":    "ml/runs",
    "name":       "waste_yolo11n",
    "exist_ok":   True,
    "patience":   15,                   # early stopping
    "save_period": 5,                   # checkpoint every 5 epochs
    "amp":        False,                # FP32. GTX 16-series (1650/1660) produce NaN
                                        # losses under FP16/AMP; FP32 is numerically
                                        # stable (see CLAUDE.md §14). Re-enable per-run
                                        # with --amp if you ever move to a newer GPU.
    "verbose":    True,

    # --- Augmentation ---
    # NOTE: copy_paste is INTENTIONALLY OMITTED (see module docstring): it only
    # works for segmentation data (needs masks); our labels are bbox-only.
    "mosaic":     1.0,                  # stitch 4 images -> more minority-class exposure/batch
    "mixup":      0.1,                  # mild image blending (works for detection)
    "close_mosaic": 10,                 # disable mosaic for the last 10 epochs (cleaner convergence)
    "degrees":    10.0,                 # rotation (waste appears at any angle)
    "flipud":     0.3,                  # vertical flip (waste has no canonical "up")
    "fliplr":     0.5,                  # horizontal flip (default)
    "translate":  0.1,
    "scale":      0.5,
    "hsv_h":      0.015,
    "hsv_s":      0.7,
    "hsv_v":      0.4,
}

def run_weights_dir(cfg: dict) -> Path:
    """Absolute weights directory for a run: ``<project>/<name>/weights``."""
    return PROJECT_ROOT / cfg["project"] / cfg["name"] / "weights"


# Printed verbatim when the GPU runs out of memory.
OOM_MESSAGE = (
    "CUDA out of memory. Re-run with:  python ml/scripts/train.py --batch 4 "
    "(or add --imgsz 512 if it still fails)."
)


# --------------------------------------------------------------------------- #
# Device / config assembly
# --------------------------------------------------------------------------- #
def resolve_device(requested):
    """
    Decide which device to train on.

    ``requested`` is the CLI/CFG device (e.g. ``0`` for the first GPU or
    ``"cpu"``). Returns ``"cpu"`` when CUDA is unavailable (or torch cannot be
    imported), logging a clear warning that CPU training will be very slow.
    """
    if str(requested).lower() == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch is not importable; falling back to CPU training.")
        return "cpu"
    if torch.cuda.is_available():
        return requested
    logger.warning(
        "CUDA is NOT available - falling back to CPU. Training will be VERY slow. "
        "Install the CUDA build of torch (see README GPU setup) to use the GTX 1650."
    )
    return "cpu"


def build_cfg(args: argparse.Namespace) -> dict:
    """
    Produce the final training config by layering CLI overrides onto ``CFG``.

    Only the flags actually supplied on the command line override the defaults.
    Three ways to pick starting weights (mutually exclusive, in priority order):
      * ``--resume``  -> continue an INTERRUPTED run from its ``last.pt`` (same
        optimizer/EMA/epoch counter). Ultralytics refuses if the run already hit
        its epoch target; see docs/README for extending a finished run.
      * ``--weights`` -> start a FRESH run initialised from given weights (a
        fine-tune: new optimizer + LR warmup/schedule, new epoch counter). Pair
        with ``--name`` so it lands in its own run folder.
      * neither       -> transfer-learn from the COCO ``yolo11n.pt`` (default).
    The module-level ``CFG`` is never mutated.
    """
    cfg = dict(CFG)

    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch is not None:
        cfg["batch"] = args.batch
    if args.imgsz is not None:
        cfg["imgsz"] = args.imgsz
    if args.amp is not None:
        cfg["amp"] = args.amp
    if args.name:
        cfg["name"] = args.name

    requested_device = args.device if args.device is not None else cfg["device"]
    cfg["device"] = resolve_device(requested_device)

    if args.resume:
        last = run_weights_dir(cfg) / "last.pt"
        if not last.exists():
            logger.warning(
                "--resume was passed but no checkpoint exists at %s. "
                "Ultralytics will error if it cannot find one.", last
            )
        cfg["model"] = str(last)
        cfg["resume"] = True
    elif args.weights:
        weights = Path(args.weights)
        if not weights.exists():
            logger.warning("--weights path does not exist: %s", weights)
        cfg["model"] = str(weights)

    return cfg


def resolve_data_config(data_arg: str) -> str:
    """
    Return a dataset config path with an ABSOLUTE ``path:`` so Ultralytics
    locates the dataset correctly.

    Ultralytics resolves a *relative* ``path:`` in a data.yaml against its global
    ``datasets_dir`` setting, NOT against the yaml's own location. Our committed
    ``ml/configs/data.yaml`` uses ``path: ../data/GARBAGE CLASSIFICATION`` (relative
    to the yaml), so Ultralytics would look under its datasets dir and fail. We
    rewrite the dataset root to an absolute path (resolved relative to the yaml
    file, as intended) and emit a temp ``*.resolved.yaml`` for Ultralytics to
    consume. This keeps the committed yaml portable and touches no global settings.
    """
    import yaml

    src = Path(data_arg)
    if not src.is_absolute():
        src = (PROJECT_ROOT / src).resolve()
    if not src.is_file():
        logger.warning("Data config not found: %s (passing through unchanged).", data_arg)
        return data_arg

    cfg = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    ds_root = Path(str(cfg.get("path", ".")))
    if not ds_root.is_absolute():
        ds_root = (src.parent / ds_root).resolve()
    cfg["path"] = str(ds_root)

    out = Path(tempfile.gettempdir()) / "waste_yolo11n.data.resolved.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    logger.info("Dataset root resolved to: %s", ds_root)
    if not ds_root.exists():
        logger.warning("Dataset root does not exist: %s", ds_root)
    return str(out)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def format_cfg_summary(cfg: dict) -> str:
    """Return a readable, aligned table of the final config for confirmation."""
    key_width = max(len(k) for k in cfg)
    header = "=== FINAL TRAINING CONFIG ==="
    rows = [f"  {key:<{key_width}} : {value}" for key, value in cfg.items()]
    return "\n".join([header, *rows])


def _is_oom_error(exc: BaseException) -> bool:
    """True if ``exc`` is a CUDA out-of-memory error (typed or message-based)."""
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # noqa: BLE001 - torch may be absent; fall back to message check
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _print_success(cfg: dict) -> None:
    """Print absolute checkpoint paths and the next pipeline command."""
    weights_dir = run_weights_dir(cfg)
    print("\n=== TRAINING COMPLETE ===")
    print(f"  best weights : {(weights_dir / 'best.pt').resolve()}")
    print(f"  last weights : {(weights_dir / 'last.pt').resolve()}")
    print("\nNext, evaluate the model:")
    print("    python ml/scripts/evaluate.py")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def run_training(cfg: dict) -> int:
    """
    Load the model and run ``model.train(**cfg)``.

    Returns a process exit code: 0 on success, 1 if the GPU ran out of memory
    (after printing an actionable hint). Any other error is re-raised so it is
    not silently swallowed.
    """
    # Point Ultralytics at an absolute dataset root (see resolve_data_config).
    cfg["data"] = resolve_data_config(cfg["data"])

    from ultralytics import YOLO  # lazy import so tests can mock it

    logger.info("Loading weights: %s", cfg["model"])
    model = YOLO(cfg["model"])

    try:
        model.train(**cfg)
    except Exception as exc:  # noqa: BLE001 - we classify then re-raise non-OOM errors
        if _is_oom_error(exc):
            logger.error("Training aborted: CUDA out of memory.")
            print(OOM_MESSAGE)
            return 1
        raise

    _print_success(cfg)
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line overrides. Unset flags keep their CFG defaults."""
    parser = argparse.ArgumentParser(
        description="Train YOLO11n on the garbage-detection dataset (Step 3)."
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch", type=int, default=None,
                        help="Override batch size (drop to 4 on GTX 1650 OOM).")
    parser.add_argument("--imgsz", type=int, default=None, help="Override training image size.")
    parser.add_argument("--device", default=None,
                        help='Override device (e.g. 0 for GPU 0, or "cpu").')
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable/disable AMP mixed precision (--amp / --no-amp). "
                             "Default OFF: GTX 16-series NaN under FP16.")
    parser.add_argument("--weights", default=None,
                        help="Start a FRESH run fine-tuned from these weights "
                             "(e.g. ml/runs/waste_yolo11n/weights/last.pt). Pair with --name.")
    parser.add_argument("--name", default=None,
                        help="Override the run folder name (default: waste_yolo11n).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an INTERRUPTED run from its last.pt (same optimizer/EMA).")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """CLI entry point: build config, show it, then train."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    cfg = build_cfg(args)
    print(format_cfg_summary(cfg))
    return run_training(cfg)


if __name__ == "__main__":
    sys.exit(main())
