"""
inspect_dataset.py  —  Step 2, gating action.

Walks ml/data/ and reports the REAL structure of the downloaded Kaggle
"garbage-detection" dataset so we can write the prep pipeline against facts,
not assumptions.

It reports:
  - the directory tree (2 levels)
  - any data.yaml / classes.txt it finds, and their contents
  - image and label counts, and the label format (YOLO vs not)
  - the actual class indices used across label files
  - orphans (image without label, label without image)

Run from the project root:
    python ml/scripts/inspect_dataset.py
"""
import os
from collections import Counter
from pathlib import Path

DATA_ROOT = Path("ml/data")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def show_tree(root: Path, max_depth: int = 2):
    print(f"\n=== DIRECTORY TREE (under {root}) ===")
    if not root.exists():
        print(f"  !! {root} does not exist. Download the dataset into it first.")
        return
    root = root.resolve()
    for path in sorted(root.rglob("*")):
        depth = len(path.relative_to(root).parts)
        if depth > max_depth:
            continue
        indent = "  " * (depth - 1)
        marker = "/" if path.is_dir() else ""
        print(f"{indent}{path.name}{marker}")


def show_config_files(root: Path):
    print("\n=== DATASET CONFIG FILES ===")
    found = False
    for name in ("data.yaml", "data.yml", "classes.txt", "classes.names", "obj.names"):
        for f in root.rglob(name):
            found = True
            print(f"\n--- {f} ---")
            try:
                print(f.read_text(encoding="utf-8").strip())
            except Exception as e:
                print(f"  (could not read: {e})")
    if not found:
        print("  No data.yaml / classes.txt found. We'll define classes manually.")


def scan_images_and_labels(root: Path):
    print("\n=== IMAGE / LABEL SCAN ===")
    images, labels = [], []
    for p in root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
        elif p.suffix.lower() == ".txt" and p.name.lower() not in {"classes.txt"}:
            labels.append(p)

    print(f"  Images found: {len(images)}")
    print(f"  Label .txt files found: {len(labels)}")

    # Inspect a few label files to confirm YOLO format & collect class ids.
    class_ids = Counter()
    bad_lines = 0
    sampled = 0
    for lab in labels:
        try:
            for line in lab.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    bad_lines += 1
                    continue
                cls = parts[0]
                coords = parts[1:]
                try:
                    class_ids[int(cls)] += 1
                    if not all(0.0 <= float(c) <= 1.0 for c in coords):
                        bad_lines += 1
                except ValueError:
                    bad_lines += 1
        except Exception:
            bad_lines += 1
        sampled += 1

    print(f"\n  Class indices present (id -> #boxes): {dict(sorted(class_ids.items()))}")
    print(f"  Number of distinct classes: {len(class_ids)}")
    print(f"  Suspicious/non-YOLO lines: {bad_lines}")

    # Orphan check (match by stem).
    img_stems = {p.stem for p in images}
    lab_stems = {p.stem for p in labels}
    imgs_no_label = img_stems - lab_stems
    labs_no_image = lab_stems - img_stems
    print(f"\n  Images with NO matching label: {len(imgs_no_label)}")
    print(f"  Labels with NO matching image: {len(labs_no_image)}")
    if imgs_no_label:
        print(f"    e.g. {list(sorted(imgs_no_label))[:5]}")


if __name__ == "__main__":
    print("Inspecting dataset under:", DATA_ROOT.resolve())
    show_tree(DATA_ROOT)
    show_config_files(DATA_ROOT)
    scan_images_and_labels(DATA_ROOT)
    print("\nDone. Paste this whole output back into the chat so we can finalise Step 2.")
