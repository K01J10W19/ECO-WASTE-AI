"""
Unit tests for ml/scripts/train.py.

No real training, no GPU, no model download, no network. ``ultralytics`` is
replaced with a stub module so importing/using train.py never touches PyTorch.

Coverage:
  - every CFG key is a valid Ultralytics train() argument (allowlist),
  - copy_paste is absent (segmentation-only no-op for our bbox data),
  - ml/configs/data.yaml matches the locked 6-class order,
  - config.py CLASS_NAMES agrees with data.yaml,
  - argparse overrides (--batch 4 --epochs 10) reach model.train().
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

# train.py lives in ml/scripts/, which is not a Python package. Add it to
# sys.path so it can be imported as a plain module (mirrors test_prepare_dataset).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _PROJECT_ROOT / "ml" / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import train  # noqa: E402

_DATA_YAML = _PROJECT_ROOT / "ml" / "configs" / "data.yaml"
_LOCKED_NAMES = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]

# Hardcoded allowlist of valid Ultralytics 8.3 ``model.train()`` arguments.
# Kept explicit on purpose: it guards against typos or invalid keys sneaking
# into CFG (which Ultralytics would otherwise reject only at runtime).
ALLOWED_TRAIN_ARGS = {
    # core
    "model", "data", "epochs", "time", "patience", "batch", "imgsz", "save",
    "save_period", "cache", "device", "workers", "project", "name", "exist_ok",
    "pretrained", "optimizer", "verbose", "seed", "deterministic", "single_cls",
    "rect", "cos_lr", "close_mosaic", "resume", "amp", "fraction", "profile",
    "freeze", "multi_scale", "overlap_mask", "mask_ratio", "dropout", "val",
    "plots",
    # optimiser / loss
    "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "warmup_momentum",
    "warmup_bias_lr", "box", "cls", "dfl", "pose", "kobj", "label_smoothing", "nbs",
    # augmentation
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "flipud", "fliplr", "bgr", "mosaic", "mixup", "copy_paste",
    "copy_paste_mode", "auto_augment", "erasing", "crop_fraction",
}


# --------------------------------------------------------------------------- #
# CFG validity
# --------------------------------------------------------------------------- #
def test_every_cfg_key_is_a_valid_train_arg():
    unknown = set(train.CFG) - ALLOWED_TRAIN_ARGS
    assert not unknown, f"CFG has non-Ultralytics train args: {sorted(unknown)}"


def test_copy_paste_is_not_in_cfg():
    # copy_paste needs polygon masks; our dataset is bbox-only, so it is a no-op.
    assert "copy_paste" not in train.CFG


def test_cfg_locks_key_choices():
    assert train.CFG["model"] == "yolo11n.pt"          # nano weights, no "v"
    assert train.CFG["data"] == "ml/configs/data.yaml"
    assert train.CFG["batch"] == 8                      # GTX 1650 starting point
    assert train.CFG["imgsz"] == 640
    assert train.CFG["amp"] is False                    # FP32: GTX 16-series NaN under FP16
    assert train.CFG["mosaic"] == 1.0 and train.CFG["mixup"] == 0.1


# --------------------------------------------------------------------------- #
# data.yaml <-> config.py consistency (locked class order)
# --------------------------------------------------------------------------- #
def test_data_yaml_has_locked_six_classes():
    assert _DATA_YAML.exists(), f"missing {_DATA_YAML}"
    data = yaml.safe_load(_DATA_YAML.read_text(encoding="utf-8"))
    assert data["nc"] == 6
    ordered = [data["names"][i] for i in range(6)]
    assert ordered == _LOCKED_NAMES


def test_config_class_names_match_data_yaml():
    import config
    data = yaml.safe_load(_DATA_YAML.read_text(encoding="utf-8"))
    yaml_names = [data["names"][i] for i in range(data["nc"])]
    assert len(config.BaseConfig.CLASS_NAMES) == 6
    assert config.BaseConfig.CLASS_NAMES == yaml_names
    assert config.BaseConfig.CLASS_NAMES == _LOCKED_NAMES


# --------------------------------------------------------------------------- #
# argparse overrides reach model.train()
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_ultralytics(monkeypatch):
    """Replace ``ultralytics`` with a stub exposing a MagicMock ``YOLO`` class."""
    stub = types.ModuleType("ultralytics")
    stub.YOLO = MagicMock(name="YOLO")
    monkeypatch.setitem(sys.modules, "ultralytics", stub)
    return stub


def test_argparse_overrides_land_in_train_call(fake_ultralytics):
    exit_code = train.main(["--batch", "4", "--epochs", "10", "--device", "cpu"])
    assert exit_code == 0

    yolo_cls = fake_ultralytics.YOLO
    yolo_cls.assert_called_once_with(train.CFG["model"])  # model = YOLO("yolo11n.pt")

    train_mock = yolo_cls.return_value.train
    train_mock.assert_called_once()
    passed = train_mock.call_args.kwargs
    assert passed["batch"] == 4
    assert passed["epochs"] == 10
    assert passed["device"] == "cpu"


def test_defaults_are_used_when_no_overrides(fake_ultralytics):
    exit_code = train.main(["--device", "cpu"])  # cpu avoids needing a real GPU
    assert exit_code == 0

    passed = fake_ultralytics.YOLO.return_value.train.call_args.kwargs
    assert passed["batch"] == 8
    assert passed["epochs"] == 50
    assert passed["amp"] is False  # AMP off by default on this hardware


def test_amp_toggle_overrides_default(fake_ultralytics):
    # --amp re-enables mixed precision for a single run.
    train.main(["--device", "cpu", "--amp"])
    passed = fake_ultralytics.YOLO.return_value.train.call_args.kwargs
    assert passed["amp"] is True


def test_weights_and_name_start_a_fresh_finetune(fake_ultralytics):
    # --weights initialises a NEW run from given weights (fine-tune), not a resume.
    train.main(["--device", "cpu", "--weights", "ml/runs/waste_yolo11n/weights/last.pt",
                "--name", "waste_yolo11n_ft", "--epochs", "50"])
    passed = fake_ultralytics.YOLO.return_value.train.call_args.kwargs
    # Model is loaded from the given weights (path separators are OS-normalised).
    model_arg = fake_ultralytics.YOLO.call_args.args[0].replace("\\", "/")
    assert model_arg == "ml/runs/waste_yolo11n/weights/last.pt"
    assert passed["name"] == "waste_yolo11n_ft"
    assert passed["epochs"] == 50
    assert "resume" not in passed  # fine-tune is a fresh run, not a resume


def test_resume_respects_name_override(fake_ultralytics):
    # --resume + --name resumes the last.pt of the *named* run.
    train.main(["--device", "cpu", "--resume", "--name", "waste_yolo11n"])
    passed = fake_ultralytics.YOLO.return_value.train.call_args.kwargs
    assert passed["resume"] is True
    assert passed["model"].endswith("last.pt")
    assert "waste_yolo11n" in passed["model"].replace("\\", "/")


def test_cfg_is_not_mutated_by_build(fake_ultralytics):
    train.main(["--batch", "4", "--epochs", "10", "--device", "cpu"])
    # The module-level defaults must survive a run with overrides.
    assert train.CFG["batch"] == 8
    assert train.CFG["epochs"] == 50
    assert train.CFG["device"] == 0


# --------------------------------------------------------------------------- #
# resolve_data_config: relative dataset path -> absolute (Ultralytics gotcha)
# --------------------------------------------------------------------------- #
def test_resolve_data_config_makes_dataset_path_absolute(tmp_path):
    # Mirror the committed layout: yaml in configs/, dataset one level up in data/.
    ds_root = tmp_path / "data" / "GARBAGE CLASSIFICATION"
    (ds_root / "valid" / "images").mkdir(parents=True)
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    src = cfg_dir / "data.yaml"
    src.write_text(
        "path: ../data/GARBAGE CLASSIFICATION\n"
        "train: train/images\nval: valid/images\ntest: test/images\n"
        "nc: 6\nnames:\n  0: BIODEGRADABLE\n",
        encoding="utf-8",
    )

    out = train.resolve_data_config(str(src))
    resolved = yaml.safe_load(Path(out).read_text(encoding="utf-8"))

    # path is now absolute and points at the real dataset root next to the yaml.
    assert Path(resolved["path"]).is_absolute()
    assert Path(resolved["path"]) == ds_root.resolve()
    # split subpaths and class map are preserved unchanged.
    assert resolved["val"] == "valid/images"
    assert resolved["nc"] == 6
