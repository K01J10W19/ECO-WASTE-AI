"""
Unit tests for ml/scripts/prepare_dataset.py.

A tiny synthetic dataset is built with PIL under tmp_path (no real dataset,
no network, no GPU). We assert that validation flags a malformed label line and
an orphan image, that the split is disjoint and reproducible, and that a
data.yaml with the right class count is written.
"""
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml
from PIL import Image

# prepare_dataset lives in ml/scripts/, which is not a Python package. Add it to
# sys.path so it can be imported as a plain module.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "ml" / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import prepare_dataset as pds  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_image(path: Path, size=(16, 16), color=(120, 120, 120)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _write_label(path: Path, lines) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def synthetic_split(tmp_path: Path) -> pds.SplitPaths:
    """
    Build a flat images/ + labels/ layout containing:
      - a, c : valid image + valid label
      - b    : valid image + label with one malformed line
      - orphan.jpg : image with NO label
      - d.txt      : label with NO image
    """
    images = tmp_path / "images"
    labels = tmp_path / "labels"

    for stem in ("a", "b", "c", "orphan"):
        _make_image(images / f"{stem}.jpg")

    _write_label(labels / "a.txt", ["0 0.5 0.5 0.2 0.2"])
    _write_label(
        labels / "b.txt",
        ["1 0.5 0.5 0.1 0.1", "2 0.5 0.5 1.5 0.1"],  # second coord >1.0 -> malformed
    )
    _write_label(labels / "c.txt", ["3 0.4 0.4 0.3 0.3"])
    _write_label(labels / "d.txt", ["0 0.5 0.5 0.2 0.2"])  # orphan label (no d.jpg)

    return pds.SplitPaths("all", images, labels)


# --------------------------------------------------------------------------- #
# validate_label_line
# --------------------------------------------------------------------------- #
def test_validate_label_line_accepts_good_line():
    class_id, error = pds.validate_label_line("0 0.5 0.5 0.2 0.2", pds.NUM_CLASSES)
    assert class_id == 0
    assert error is None


@pytest.mark.parametrize(
    "line",
    [
        "0 0.5 0.5 0.2",            # too few tokens
        "x 0.5 0.5 0.2 0.2",        # class id not an int
        "0 0.5 0.5 1.5 0.2",        # coord out of [0,1]
        "9 0.5 0.5 0.2 0.2",        # class id out of range
    ],
)
def test_validate_label_line_rejects_bad_lines(line):
    _class_id, error = pds.validate_label_line(line, pds.NUM_CLASSES)
    assert error is not None


# --------------------------------------------------------------------------- #
# validate_split
# --------------------------------------------------------------------------- #
def test_validation_flags_malformed_and_orphans(synthetic_split):
    report = pds.validate_split(synthetic_split, pds.NUM_CLASSES)

    # Orphan image (no label) is flagged.
    assert any(p.name == "orphan.jpg" for p in report.images_without_labels)
    # Orphan label (no image) is flagged.
    assert any(p.name == "d.txt" for p in report.labels_without_images)
    # The malformed line in b.txt is flagged.
    assert any(f.name == "b.txt" for (f, _ln, _reason) in report.malformed_labels)

    # Only clean pairs count as valid: a and c, but NOT b (malformed) or orphan.
    valid_stems = {img.stem for img, _lab in report.valid_pairs}
    assert valid_stems == {"a", "c"}


# --------------------------------------------------------------------------- #
# reconcile_classes
# --------------------------------------------------------------------------- #
def test_reconcile_classes_raises_on_out_of_range():
    report = pds.ValidationReport()
    report.all_class_ids = {0, 6}  # 6 is out of range for 6 classes (0..5)
    with pytest.raises(SystemExit):
        pds.reconcile_classes(report, pds.NUM_CLASSES)


def test_reconcile_classes_passes_when_in_range():
    report = pds.ValidationReport()
    report.all_class_ids = {0, 5}
    pds.reconcile_classes(report, pds.NUM_CLASSES)  # must not raise


# --------------------------------------------------------------------------- #
# split_dataset
# --------------------------------------------------------------------------- #
def test_split_is_disjoint_and_covers_everything():
    pairs = [(Path(f"img{i}.jpg"), Path(f"img{i}.txt")) for i in range(10)]
    assignment = pds.split_dataset(pairs, seed=42)

    def stems(part):
        return {img.stem for img, _lab in assignment[part]}

    train, val, test = stems("train"), stems("val"), stems("test")
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == {f"img{i}" for i in range(10)}


def test_split_is_reproducible():
    pairs = [(Path(f"img{i}.jpg"), Path(f"img{i}.txt")) for i in range(10)]
    assert pds.split_dataset(pairs, seed=42) == pds.split_dataset(pairs, seed=42)


# --------------------------------------------------------------------------- #
# write_data_yaml
# --------------------------------------------------------------------------- #
def test_write_data_yaml_has_correct_class_count(tmp_path: Path):
    out = tmp_path / "data.yaml"
    pds.write_data_yaml(
        out,
        dataset_root_rel="../data/GARBAGE CLASSIFICATION",
        split_map={"train": "train/images", "val": "valid/images", "test": "test/images"},
        names=pds.CLASS_NAMES,
        class_counts=Counter({0: 10, 1: 5}),
    )
    data = yaml.safe_load(out.read_text(encoding="utf-8"))

    assert data["nc"] == 6
    assert len(data["names"]) == 6
    assert data["names"][0] == "BIODEGRADABLE"
    assert data["names"][5] == "PLASTIC"
    assert data["path"] == "../data/GARBAGE CLASSIFICATION"
    assert data["val"] == "valid/images"
