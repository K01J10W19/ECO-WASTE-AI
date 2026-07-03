"""
restratify_dataset.py  -  fix the pathological train/valid/test class split.

The Kaggle/Roboflow "garbage-detection" download ships a pre-made split whose
class distribution is severely shifted between splits (discovered during Step 4
evaluation). For example, on the original split:

    PAPER  : train 2981 | valid   33 | test 1376   (0.17% of valid!)
    PLASTIC: train 4146 | valid  214 | test 1585   (1.13% of valid)
    GLASS  : train 5429 | valid 2380 | test    0   (absent from test)

That makes validation metrics meaningless for the minority classes (PAPER mAP is
"measured" on 33 boxes) and produces a misleading overall mAP. This script MERGES
all three original splits and creates a fresh, class-balanced 80/10/10 split so
every class is represented proportionally in train, valid and test.

Method
------
Multi-label stratification (each image holds boxes of several classes). We key
each image by the RAREST class present in it (by global box frequency), group
images by that key, and split each group 80/10/10. Keying on the rarest class
guarantees minority classes (PAPER, CARDBOARD) are spread evenly across splits.

Non-destructive: the original ``GARBAGE CLASSIFICATION/`` folder is never touched;
output goes to a new ``garbage_stratified/`` root and ``ml/configs/data.yaml`` is
repointed at it.

Run from the project root (Windows / PowerShell)::

    python ml/scripts/restratify_dataset.py --dry-run   # report new balance only
    python ml/scripts/restratify_dataset.py             # copy + write data.yaml
"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

# LOCKED class order (see CLAUDE.md §6). Index == YOLO class id.
CLASS_NAMES = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLIT_DIRS = ("train", "valid", "test")
RATIOS = (0.8, 0.1, 0.1)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = PROJECT_ROOT / "ml" / "data" / "GARBAGE CLASSIFICATION"
DEFAULT_OUT = PROJECT_ROOT / "ml" / "data" / "garbage_stratified"
DATA_YAML_PATH = PROJECT_ROOT / "ml" / "configs" / "data.yaml"

logger = logging.getLogger("restratify")


# --------------------------------------------------------------------------- #
# Reading the source dataset
# --------------------------------------------------------------------------- #
def collect_pairs(src_root: Path) -> list:
    """Return ``[(image_path, label_path), ...]`` across all source splits.

    Only images that have a matching ``labels/<stem>.txt`` are included.
    """
    pairs = []
    for split in SPLIT_DIRS:
        img_dir = src_root / split / "images"
        lab_dir = src_root / split / "labels"
        if not img_dir.is_dir():
            continue
        labels = {p.stem: p for p in lab_dir.glob("*.txt")} if lab_dir.is_dir() else {}
        for img in sorted(img_dir.iterdir()):
            if img.is_file() and img.suffix.lower() in IMAGE_EXTS and img.stem in labels:
                pairs.append((img, labels[img.stem]))
    return pairs


def classes_in_label(label_path: Path) -> set:
    """Return the set of class ids that appear in a YOLO label file."""
    ids = set()
    for raw in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            ids.add(int(raw.split()[0]))
        except ValueError:
            continue
    return ids


def count_boxes(pairs) -> Counter:
    """Count box instances per class over a list of ``(image, label)`` pairs."""
    counts: Counter = Counter()
    for _img, lab in pairs:
        for raw in lab.read_text(encoding="utf-8", errors="replace").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                counts[int(raw.split()[0])] += 1
            except ValueError:
                continue
    return counts


# --------------------------------------------------------------------------- #
# Stratified split
# --------------------------------------------------------------------------- #
def rarity_rank(box_counts: Counter) -> dict:
    """Map class id -> rarity rank (0 == rarest globally)."""
    ordered = sorted(range(NUM_CLASSES), key=lambda c: box_counts.get(c, 0))
    return {cls: rank for rank, cls in enumerate(ordered)}


def stratify_key(label_path: Path, ranks: dict) -> int:
    """Stratum for an image = the rarest class present in it (-1 if empty)."""
    ids = classes_in_label(label_path)
    if not ids:
        return -1
    return min(ids, key=lambda c: ranks.get(c, NUM_CLASSES))


def stratified_split(pairs, ranks: dict, ratios=RATIOS, seed: int = 42) -> dict:
    """Split ``pairs`` into train/valid/test, stratified by each image's rarest class.

    Every stratum is shuffled deterministically (``seed``) and split by ``ratios``.
    Small strata are nudged so valid and test each get at least one item when the
    stratum has >= 3 images, so no class is starved from a split.
    """
    groups = defaultdict(list)
    for img, lab in pairs:
        groups[stratify_key(lab, ranks)].append((img, lab))

    rng = random.Random(seed)
    out = {"train": [], "valid": [], "test": []}
    for _stratum, items in sorted(groups.items(), key=lambda kv: kv[0]):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        if n >= 3:  # guarantee a val + test example for small strata
            n_val = max(1, n_val)
            n_train = min(n_train, n - n_val - 1)  # leave >= 1 for test
        out["train"] += items[:n_train]
        out["valid"] += items[n_train : n_train + n_val]
        out["test"] += items[n_train + n_val :]
    return out


# --------------------------------------------------------------------------- #
# Writing the new dataset
# --------------------------------------------------------------------------- #
def copy_assignment(assignment: dict, out_root: Path) -> None:
    """Copy each split's ``(image, label)`` pairs into ``out_root/<split>/{images,labels}``.

    Image and label always share a stem; on a name collision within a split a
    numeric suffix is added to both so they stay paired.
    """
    for split, pairs in assignment.items():
        img_out = out_root / split / "images"
        lab_out = out_root / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lab_out.mkdir(parents=True, exist_ok=True)
        used: set = set()
        for img, lab in pairs:
            stem, i = img.stem, 1
            while stem in used:
                stem = f"{img.stem}_{i}"
                i += 1
            used.add(stem)
            shutil.copy2(img, img_out / f"{stem}{img.suffix.lower()}")
            shutil.copy2(lab, lab_out / f"{stem}.txt")


def write_data_yaml(path: Path, dataset_root_rel: str) -> None:
    """Write a directory-based Ultralytics data.yaml pointing at the new root."""
    lines = [
        "# ml/configs/data.yaml",
        "# Regenerated by ml/scripts/restratify_dataset.py (Step 4 split fix).",
        "# The original Roboflow split had severe class distribution shift across",
        "# train/valid/test; this points at a fresh, class-balanced 80/10/10 split.",
        "# Class order is LOCKED - do NOT reorder (see CLAUDE.md §6).",
        "",
        f"path: {dataset_root_rel}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {NUM_CLASSES}",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_balance(assignment: dict) -> None:
    """Print a per-split, per-class box-instance table with each split's share."""
    per_split = {split: count_boxes(pairs) for split, pairs in assignment.items()}
    header = f"{'CLASS':<14}" + "".join(f"{s:>10}" for s in ("train", "valid", "test")) + f"{'TOTAL':>10}"
    print("\n=== NEW CLASS BALANCE (box instances) ===")
    print(header)
    print("-" * len(header))
    for i, name in enumerate(CLASS_NAMES):
        tr, va, te = per_split["train"][i], per_split["valid"][i], per_split["test"][i]
        print(f"{name:<14}{tr:>10}{va:>10}{te:>10}{tr + va + te:>10}")
    print("-" * len(header))
    for split in ("train", "valid", "test"):
        tot = sum(per_split[split].values()) or 1
        shares = ", ".join(f"{CLASS_NAMES[i]} {100 * per_split[split][i] / tot:.1f}%"
                           for i in range(NUM_CLASSES))
        print(f"{split:<6} n={sum(per_split[split].values()):>6} | {shares}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(src_root: Path, out_root: Path, dry_run: bool, seed: int, write_yaml: bool) -> int:
    """Execute the re-split. Returns a process exit code (0 == success)."""
    if out_root.resolve() == src_root.resolve():
        raise SystemExit("Refusing to write the stratified split back onto the source root.")
    if not src_root.exists():
        raise SystemExit(f"Source dataset not found: {src_root}")

    pairs = collect_pairs(src_root)
    if not pairs:
        raise SystemExit(f"No image/label pairs found under {src_root}.")
    logger.info("Collected %d image/label pairs from %s", len(pairs), src_root)

    ranks = rarity_rank(count_boxes(pairs))
    logger.info("Class rarity (rarest first): %s",
                [CLASS_NAMES[c] for c in sorted(ranks, key=ranks.get)])

    assignment = stratified_split(pairs, ranks, seed=seed)
    print(f"\nSplit sizes (images): train={len(assignment['train'])}, "
          f"valid={len(assignment['valid'])}, test={len(assignment['test'])}")
    print_balance(assignment)

    if dry_run:
        print("\n[dry-run] No files copied and data.yaml not written. "
              "Re-run without --dry-run to apply.")
        return 0

    if out_root.exists():
        logger.info("Removing existing output root %s for a clean rebuild.", out_root)
        shutil.rmtree(out_root)
    copy_assignment(assignment, out_root)
    print(f"\nCopied stratified dataset to: {out_root}")

    if write_yaml:
        rel = f"../data/{out_root.name}"
        write_data_yaml(DATA_YAML_PATH, rel)
        print(f"Wrote {DATA_YAML_PATH} (path: {rel})")

    print("\nDone. Retrain on the balanced split:")
    print("    python ml/scripts/train.py")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create a class-balanced 80/10/10 re-split.")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC,
                        help="Source dataset root (default: the original GARBAGE CLASSIFICATION).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output root for the stratified split.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (reproducible).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the new balance without copying files or writing data.yaml.")
    parser.add_argument("--no-yaml", action="store_true",
                        help="Do not overwrite ml/configs/data.yaml.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    return run(args.src, args.out, dry_run=args.dry_run, seed=args.seed, write_yaml=not args.no_yaml)


if __name__ == "__main__":
    sys.exit(main())
