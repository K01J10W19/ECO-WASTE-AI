"""
prepare_dataset.py  -  Step 2: dataset preparation & validation pipeline.

Turns the raw Kaggle "garbage-detection" dataset (already in YOLO format) under
``ml/data/`` into a clean, validated dataset with a finalised
``ml/configs/data.yaml`` ready for YOLO11 training.

The dataset ships pre-split (``train/ valid/ test/``, each with ``images/`` +
``labels/``). When an existing split is detected we validate it in place; only if
NO split exists do we create a reproducible 80/10/10 split from a flat folder.

Run from the project root (Windows / PowerShell)::

    python ml/scripts/prepare_dataset.py --dry-run   # validate + report only
    python ml/scripts/prepare_dataset.py             # validate, split if needed, write data.yaml

Design notes:
  - Pure standard library + Pillow + PyYAML. No GPU, no network, no heavy deps.
  - Idempotent: re-running never duplicates files.
  - Never crashes on bad samples; it reports and skips them.
  - Exits non-zero only on *blocking* problems (class ids out of range, or no
    valid image/label pairs found at all).
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image

# --------------------------------------------------------------------------- #
# Canonical classes.
# LOCKED order — MUST match config.py:CLASS_NAMES, ml/configs/data.yaml, and the
# dataset's own label indices exactly (see CLAUDE.md §6 & §7). The model uses
# these ALL-CAPS names internally; the web UI maps them to friendly names below.
# --------------------------------------------------------------------------- #
CLASS_NAMES = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
NUM_CLASSES = len(CLASS_NAMES)

DISPLAY_NAME_MAP = {
    "BIODEGRADABLE": "Biodegradable",
    "CARDBOARD": "Cardboard",
    "GLASS": "Glass",
    "METAL": "Metal",
    "PAPER": "Paper",
    "PLASTIC": "Plastic",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# On-disk split folder name -> canonical split key used by Ultralytics.
SPLIT_ALIASES = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}
_SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}

# This file lives at ml/scripts/; parents[2] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "ml" / "data"
DATA_YAML_PATH = PROJECT_ROOT / "ml" / "configs" / "data.yaml"

logger = logging.getLogger("prepare_dataset")


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class SplitPaths:
    """A single split's image and label directories."""

    name: str  # canonical split key: "train" / "val" / "test" (or "all" when flat)
    images_dir: Path
    labels_dir: Path


@dataclass
class ValidationReport:
    """Aggregated validation results for one or more splits. Never raises."""

    valid_pairs: list = field(default_factory=list)           # list[tuple[Path, Path]]
    images_without_labels: list = field(default_factory=list)  # list[Path]
    labels_without_images: list = field(default_factory=list)  # list[Path]
    corrupt_images: list = field(default_factory=list)         # list[Path]
    malformed_labels: list = field(default_factory=list)       # list[tuple[Path, int, str]]
    class_counts: Counter = field(default_factory=Counter)     # class_id -> #boxes
    all_class_ids: set = field(default_factory=set)            # every parseable class id

    def merge(self, other: "ValidationReport") -> None:
        """Fold another report's findings into this one."""
        self.valid_pairs += other.valid_pairs
        self.images_without_labels += other.images_without_labels
        self.labels_without_images += other.labels_without_images
        self.corrupt_images += other.corrupt_images
        self.malformed_labels += other.malformed_labels
        self.class_counts += other.class_counts
        self.all_class_ids |= other.all_class_ids

    @property
    def has_blocking_problems(self) -> bool:
        """A run is blocked only if nothing usable was found."""
        return not self.valid_pairs


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_label_line(line: str, num_classes: int):
    """
    Validate a single YOLO label line ("<class> <cx> <cy> <w> <h>").

    Returns ``(class_id, error)`` where ``class_id`` is the parsed int when the
    first token is an integer (even if the rest is invalid), else ``None``;
    ``error`` is a human-readable reason string, or ``None`` when the line is a
    valid YOLO annotation.
    """
    parts = line.split()
    if len(parts) != 5:
        return None, f"expected 5 tokens, got {len(parts)}"
    try:
        class_id = int(parts[0])
    except ValueError:
        return None, f"class id '{parts[0]}' is not an integer"
    try:
        coords = [float(x) for x in parts[1:]]
    except ValueError:
        return class_id, "bounding-box coords are not all floats"
    if not all(0.0 <= c <= 1.0 for c in coords):
        return class_id, "bounding-box coords outside [0.0, 1.0]"
    if not 0 <= class_id < num_classes:
        return class_id, f"class id {class_id} out of range [0, {num_classes - 1}]"
    return class_id, None


def _list_files(directory: Path, exts: set[str]) -> list[Path]:
    """Return sorted files in ``directory`` whose suffix is in ``exts``."""
    if not directory or not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts
    )


def validate_split(split: SplitPaths, num_classes: int) -> ValidationReport:
    """
    Validate one split: image<->label pairing, image readability, and YOLO label
    correctness. A pair is "valid" only when the image opens AND every label line
    parses. Bad samples are recorded and skipped, never raised.
    """
    report = ValidationReport()
    images = _list_files(split.images_dir, IMAGE_EXTS)
    labels = _list_files(split.labels_dir, {".txt"})
    img_by_stem = {p.stem: p for p in images}
    lab_by_stem = {p.stem: p for p in labels}

    for stem, img in img_by_stem.items():
        if stem not in lab_by_stem:
            report.images_without_labels.append(img)
    for stem, lab in lab_by_stem.items():
        if stem not in img_by_stem:
            report.labels_without_images.append(lab)

    for stem, img in img_by_stem.items():
        lab = lab_by_stem.get(stem)
        if lab is None:
            continue  # already recorded as an orphan image

        try:
            with Image.open(img) as im:
                im.verify()
        except Exception:  # noqa: BLE001 - any decode error means the image is unusable
            report.corrupt_images.append(img)
            continue

        line_ok = True
        for lineno, raw in enumerate(
            lab.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            if not raw.strip():
                continue
            class_id, error = validate_label_line(raw, num_classes)
            if class_id is not None:
                report.all_class_ids.add(class_id)
            if error:
                report.malformed_labels.append((lab, lineno, error))
                line_ok = False
            else:
                report.class_counts[class_id] += 1

        if line_ok:
            report.valid_pairs.append((img, lab))

    return report


def reconcile_classes(report: ValidationReport, num_classes: int) -> None:
    """
    Assert every class id seen in the labels is within the declared range.
    Fails loudly (SystemExit) so a mismatch can never slip into training.
    """
    out_of_range = sorted(c for c in report.all_class_ids if not 0 <= c < num_classes)
    if out_of_range:
        raise SystemExit(
            f"FATAL: labels reference class ids {out_of_range} outside the declared "
            f"range [0, {num_classes - 1}]. Fix the labels or CLASS_NAMES before training."
        )


# --------------------------------------------------------------------------- #
# Layout detection
# --------------------------------------------------------------------------- #
def _images_labels_dirs(base: Path):
    """Return ``(images_dir, labels_dir)`` under ``base`` if both exist, else (None, None)."""
    images_dir = base / "images"
    labels_dir = base / "labels"
    if images_dir.is_dir() and labels_dir.is_dir():
        return images_dir, labels_dir
    return None, None


def _find_splits(root: Path) -> list[SplitPaths]:
    """Find train/val/test split folders (each with images/ + labels/) under ``root``."""
    if not root.exists():
        return []
    found: list[SplitPaths] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        canonical = SPLIT_ALIASES.get(child.name.lower())
        if not canonical:
            continue
        images_dir, labels_dir = _images_labels_dirs(child)
        if images_dir and labels_dir:
            found.append(SplitPaths(canonical, images_dir, labels_dir))
    found.sort(key=lambda s: _SPLIT_ORDER.get(s.name, 9))
    return found


def _find_flat(root: Path) -> Optional[SplitPaths]:
    """Detect a flat ``images/`` + ``labels/`` layout directly under ``root``."""
    images_dir, labels_dir = _images_labels_dirs(root)
    if images_dir and labels_dir:
        return SplitPaths("all", images_dir, labels_dir)
    return None


def detect_layout(data_root: Path):
    """
    Locate the dataset under ``data_root`` and detect its layout.

    Returns ``(kind, dataset_root, splits)``:
      - kind == "split": ``splits`` is a list of existing train/val/test SplitPaths.
      - kind == "flat":  ``splits`` is a single-item list (images/ + labels/).

    Searches ``data_root`` itself and its immediate subdirectories (so a wrapper
    folder such as "GARBAGE CLASSIFICATION" is handled). Raises SystemExit with a
    clear message if nothing usable is found.
    """
    if not data_root.exists():
        raise SystemExit(
            f"Dataset root not found: {data_root}\n"
            f"Download it first, e.g.:\n"
            f"  python -m kaggle datasets download -d viswaprakash1990/garbage-detection "
            f"-p ml/data --unzip"
        )

    candidates = [data_root] + [d for d in sorted(data_root.iterdir()) if d.is_dir()]

    for root in candidates:
        splits = _find_splits(root)
        if splits:
            return "split", root, splits

    for root in candidates:
        flat = _find_flat(root)
        if flat:
            return "flat", root, [flat]

    raise SystemExit(
        f"Could not find a usable dataset under {data_root}. Expected either "
        f"train/val/test folders (each with images/ + labels/) or a flat "
        f"images/ + labels/ pair. Run inspect_dataset.py to see what is there."
    )


# --------------------------------------------------------------------------- #
# Splitting (only used when the dataset is flat)
# --------------------------------------------------------------------------- #
def split_dataset(pairs, seed: int = 42, ratios=(0.8, 0.1, 0.1)) -> dict:
    """
    Partition ``pairs`` (list of ``(image, label)``) into disjoint train/val/test.
    Deterministic for a given ``seed`` so splits are reproducible.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }


def copy_split(assignment: dict, out_root: Path) -> None:
    """
    Copy (never move) each split's files into
    ``out_root/{images,labels}/<split>/``. Idempotent: existing files are skipped.
    """
    for split_name, pairs in assignment.items():
        img_out = out_root / "images" / split_name
        lab_out = out_root / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lab_out.mkdir(parents=True, exist_ok=True)
        for img, lab in pairs:
            dst_img = img_out / img.name
            dst_lab = lab_out / lab.name
            if not dst_img.exists():
                shutil.copy2(img, dst_img)
            if not dst_lab.exists():
                shutil.copy2(lab, dst_lab)


# --------------------------------------------------------------------------- #
# data.yaml
# --------------------------------------------------------------------------- #
def _relpath_posix(target: Path, start: Path) -> str:
    """Relative path from ``start`` to ``target`` using forward slashes."""
    return Path(os.path.relpath(target, start)).as_posix()


def write_data_yaml(
    path: Path,
    dataset_root_rel: str,
    split_map: dict,
    names: list,
    class_counts: Optional[Counter] = None,
) -> None:
    """
    Write the Ultralytics ``data.yaml``.

    ``dataset_root_rel`` is the dataset root relative to the yaml's own folder;
    ``split_map`` maps split keys ("train"/"val"/"test") to the images subpath
    relative to that root. ``class_counts`` (optional) is appended as a
    human-readable balance comment.
    """
    lines = [
        "# ml/configs/data.yaml",
        "# Auto-generated by ml/scripts/prepare_dataset.py (Step 2).",
        "# Class order is LOCKED - DO NOT reorder (YOLO label indices must match).",
        "# Dataset uses ALL-CAPS names; the web app maps them to friendly display",
        "# names via APP_CLASS_DISPLAY_NAMES in config.py.",
        "",
        f"path: {dataset_root_rel}",
    ]
    for key in ("train", "val", "test"):
        if key in split_map:
            lines.append(f"{key}: {split_map[key]}")
    lines += ["", f"nc: {len(names)}", "names:"]
    lines += [f"  {i}: {name}" for i, name in enumerate(names)]

    if class_counts:
        total = sum(class_counts.values()) or 1
        lines += ["", "# Class balance (box instances):"]
        for i, name in enumerate(names):
            count = class_counts.get(i, 0)
            lines.append(f"#   {name:<13}: {count:>6}  ({100 * count / total:4.1f}%)")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Human-facing report
# --------------------------------------------------------------------------- #
def print_balance_table(per_split_counts: dict, names: list) -> None:
    """Print a per-class, per-split box-instance table and flag rare classes."""
    splits = list(per_split_counts.keys())
    header = f"{'CLASS':<15}" + "".join(f"{s:>10}" for s in splits) + f"{'TOTAL':>10}"
    print("\n=== CLASS BALANCE (box instances) ===")
    print(header)
    print("-" * len(header))

    totals_per_class = []
    for i, name in enumerate(names):
        row_total = sum(per_split_counts[s].get(i, 0) for s in splits)
        totals_per_class.append(row_total)
        row = f"{name:<15}" + "".join(f"{per_split_counts[s].get(i, 0):>10}" for s in splits)
        print(row + f"{row_total:>10}")

    grand = sum(totals_per_class) or 1
    largest = max(totals_per_class) if totals_per_class else 0
    print("-" * len(header))
    print(f"{'TOTAL':<15}" + "".join(f"{sum(per_split_counts[s].values()):>10}" for s in splits)
          + f"{grand:>10}")

    for i, name in enumerate(names):
        if largest and totals_per_class[i] < 0.05 * largest:
            print(f"  ! WARNING: class '{name}' is <5% of the largest class "
                  f"({totals_per_class[i]} vs {largest}). Emphasise it in Step 3 augmentation.")


def print_validation_summary(report: ValidationReport) -> None:
    """Print counts and a few examples of any problems found."""
    print("\n=== VALIDATION SUMMARY ===")
    print(f"  Valid image/label pairs : {len(report.valid_pairs)}")
    print(f"  Images without labels   : {len(report.images_without_labels)}")
    print(f"  Labels without images   : {len(report.labels_without_images)}")
    print(f"  Corrupt images          : {len(report.corrupt_images)}")
    print(f"  Malformed label lines   : {len(report.malformed_labels)}")
    print(f"  Class ids seen          : {sorted(report.all_class_ids)}")

    for label, examples in (
        ("images without labels", report.images_without_labels),
        ("labels without images", report.labels_without_images),
        ("corrupt images", report.corrupt_images),
    ):
        if examples:
            preview = ", ".join(p.name for p in examples[:5])
            print(f"    e.g. {label}: {preview}")
    if report.malformed_labels:
        f, ln, reason = report.malformed_labels[0]
        print(f"    e.g. malformed label: {f.name}:{ln} -> {reason}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(data_root: Path, dry_run: bool, seed: int) -> int:
    """Execute the full pipeline. Returns a process exit code (0 == success)."""
    kind, dataset_root, splits = detect_layout(data_root)
    print(f"Detected '{kind}' layout at: {dataset_root}")
    for sp in splits:
        print(f"  - {sp.name}: {sp.images_dir}  |  {sp.labels_dir}")

    # Validate every split; keep per-split reports for the balance table.
    split_reports: dict = {}
    merged = ValidationReport()
    for sp in splits:
        rep = validate_split(sp, NUM_CLASSES)
        split_reports[sp.name] = rep
        merged.merge(rep)

    print_validation_summary(merged)
    reconcile_classes(merged, NUM_CLASSES)

    if merged.has_blocking_problems:
        print("\nFATAL: no valid image/label pairs found. Nothing to prepare.")
        return 1

    # Decide the data.yaml pointers (and split flat data if needed).
    if kind == "split":
        yaml_root = dataset_root
        split_map = {
            sp.name: sp.images_dir.relative_to(dataset_root).as_posix() for sp in splits
        }
        per_split_counts = {sp.name: split_reports[sp.name].class_counts for sp in splits}
    else:  # flat
        assignment = split_dataset(merged.valid_pairs, seed=seed)
        per_split_counts = {
            name: _count_boxes(pairs) for name, pairs in assignment.items()
        }
        if dry_run:
            print("\n[dry-run] Would create an 80/10/10 split "
                  f"(train={len(assignment['train'])}, val={len(assignment['val'])}, "
                  f"test={len(assignment['test'])}) under ml/data/garbage-detection/.")
            yaml_root = data_root / "garbage-detection"
        else:
            yaml_root = data_root / "garbage-detection"
            copy_split(assignment, yaml_root)
            print(f"\nCopied split into: {yaml_root}")
        split_map = {"train": "images/train", "val": "images/val", "test": "images/test"}

    print_balance_table(per_split_counts, CLASS_NAMES)

    if dry_run:
        print("\n[dry-run] Validation only - data.yaml was NOT written. "
              "Re-run without --dry-run to finalise.")
        return 0

    dataset_root_rel = _relpath_posix(yaml_root, DATA_YAML_PATH.parent)
    write_data_yaml(DATA_YAML_PATH, dataset_root_rel, split_map, CLASS_NAMES, merged.class_counts)
    print(f"\nWrote {DATA_YAML_PATH} (path: {dataset_root_rel})")

    print("\nStep 2 complete. Next, start training:")
    print("    python ml/scripts/train.py")
    return 0


def _count_boxes(pairs) -> Counter:
    """Count valid box instances per class for a list of (image, label) pairs."""
    counts: Counter = Counter()
    for _img, lab in pairs:
        for raw in lab.read_text(encoding="utf-8", errors="replace").splitlines():
            if not raw.strip():
                continue
            class_id, error = validate_label_line(raw, NUM_CLASSES)
            if class_id is not None and not error:
                counts[class_id] += 1
    return counts


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate and finalise the garbage-detection dataset for YOLO11 training."
    )
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help="Path to the raw dataset root (default: ml/data).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and report only; do not copy files or write data.yaml.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the 80/10/10 split when the data is flat (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    return run(args.data_root, dry_run=args.dry_run, seed=args.seed)


if __name__ == "__main__":
    sys.exit(main())
