"""
Unit tests for ml/scripts/restratify_dataset.py.

Synthetic label sets (no real images/GPU/network) verify the core guarantee:
after a stratified re-split, EVERY class appears in EVERY split — which the
original Roboflow split failed (PAPER was 0.17% of valid, GLASS absent from test).
"""
import sys
from collections import Counter
from pathlib import Path

# restratify_dataset lives in ml/scripts/, which is not a package.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "ml" / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import restratify_dataset as rds  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_pairs(tmp_path: Path, specs):
    """Build ``(image_path, label_path)`` pairs from ``specs`` = [(count, [class_ids])].

    Label files are written to disk (the split logic reads them); image paths are
    placeholders (never opened by the split logic).
    """
    pairs, k = [], 0
    for count, class_ids in specs:
        for _ in range(count):
            k += 1
            lab = tmp_path / f"lab_{k}.txt"
            lab.write_text("\n".join(f"{c} 0.5 0.5 0.2 0.2" for c in class_ids) + "\n",
                           encoding="utf-8")
            pairs.append((tmp_path / f"img_{k}.jpg", lab))
    return pairs


def _classes_per_split(assignment):
    """Map split -> set of class ids present."""
    return {s: set(rds.count_boxes(pairs)) for s, pairs in assignment.items()}


# --------------------------------------------------------------------------- #
# rarity + stratum keying
# --------------------------------------------------------------------------- #
def test_rarity_rank_orders_by_frequency():
    counts = Counter({0: 100, 1: 5, 2: 50, 3: 8, 4: 3, 5: 20})
    ranks = rds.rarity_rank(counts)
    # rarest (class 4) gets rank 0; most common (class 0) gets the highest rank.
    assert ranks[4] == 0
    assert ranks[0] == rds.NUM_CLASSES - 1


def test_stratify_key_picks_rarest_present(tmp_path):
    ranks = {0: 5, 1: 4, 2: 3, 3: 2, 4: 0, 5: 1}  # class 4 rarest
    lab = tmp_path / "l.txt"
    lab.write_text("0 0.5 0.5 0.1 0.1\n4 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    assert rds.stratify_key(lab, ranks) == 4  # 4 is rarer than 0


# --------------------------------------------------------------------------- #
# the core guarantee
# --------------------------------------------------------------------------- #
def test_every_class_appears_in_every_split(tmp_path):
    # class 4 is the rare minority (like PAPER), always co-occurring with common class 0.
    specs = [(60, [0]), (18, [0, 4]), (12, [0, 3]), (20, [2])]
    pairs = _make_pairs(tmp_path, specs)
    ranks = rds.rarity_rank(rds.count_boxes(pairs))
    assignment = rds.stratified_split(pairs, ranks, seed=42)

    all_classes = set(rds.count_boxes(pairs))
    per_split = _classes_per_split(assignment)
    for split in ("train", "valid", "test"):
        assert per_split[split] == all_classes, f"{split} missing {all_classes - per_split[split]}"


def test_split_is_disjoint_and_lossless(tmp_path):
    pairs = _make_pairs(tmp_path, [(60, [0]), (18, [0, 4]), (12, [0, 3]), (20, [2])])
    ranks = rds.rarity_rank(rds.count_boxes(pairs))
    a = rds.stratified_split(pairs, ranks, seed=42)

    ids = {s: {p[1].name for p in a[s]} for s in a}
    assert ids["train"].isdisjoint(ids["valid"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["valid"].isdisjoint(ids["test"])
    assert ids["train"] | ids["valid"] | ids["test"] == {p[1].name for p in pairs}


def test_ratios_are_approximately_80_10_10(tmp_path):
    pairs = _make_pairs(tmp_path, [(800, [0]), (100, [0, 4]), (100, [2])])  # 1000 images
    ranks = rds.rarity_rank(rds.count_boxes(pairs))
    a = rds.stratified_split(pairs, ranks, seed=42)
    n = len(pairs)
    assert abs(len(a["train"]) / n - 0.8) < 0.03
    assert abs(len(a["valid"]) / n - 0.1) < 0.03
    assert abs(len(a["test"]) / n - 0.1) < 0.03


def test_split_is_reproducible(tmp_path):
    pairs = _make_pairs(tmp_path, [(60, [0]), (18, [0, 4]), (12, [0, 3])])
    ranks = rds.rarity_rank(rds.count_boxes(pairs))
    a1 = rds.stratified_split(pairs, ranks, seed=7)
    a2 = rds.stratified_split(pairs, ranks, seed=7)
    names = lambda a, s: sorted(p[1].name for p in a[s])
    assert all(names(a1, s) == names(a2, s) for s in ("train", "valid", "test"))


# --------------------------------------------------------------------------- #
# collect_pairs merges the three source splits
# --------------------------------------------------------------------------- #
def test_collect_pairs_merges_all_source_splits(tmp_path):
    root = tmp_path / "ds"
    for split, stem in (("train", "a"), ("valid", "b"), ("test", "c")):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        (root / split / "images" / f"{stem}.jpg").write_bytes(b"fake")
        (root / split / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n",
                                                             encoding="utf-8")
    pairs = rds.collect_pairs(root)
    assert sorted(p[0].stem for p in pairs) == ["a", "b", "c"]
