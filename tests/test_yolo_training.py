from __future__ import annotations

import pytest

from backend.services.yolo_training import build_yolo_train_args


def test_build_yolo_train_args_validates_dataset_and_normalizes_values(tmp_path) -> None:
    dataset_yaml = tmp_path / "data.yaml"
    dataset_yaml.write_text("path: .\ntrain: images/train\nval: images/val\nnames: [player]\n", encoding="utf-8")

    prepared = build_yolo_train_args(
        {
            "dataset_yaml": str(dataset_yaml),
            "base_model": "yolo26s.pt",
            "epochs": 0,
            "imgsz": 99,
            "batch": 4,
            "device": "0",
            "run_name": "Soccer Detector!",
        }
    )

    assert prepared["base_model"] == "yolo26s.pt"
    assert prepared["args"]["data"] == str(dataset_yaml.resolve())
    assert prepared["args"]["epochs"] == 1
    assert prepared["args"]["imgsz"] == 320
    assert prepared["args"]["name"] == "Soccer-Detector"
    assert prepared["args"]["device"] == "0"


def test_build_yolo_train_args_rejects_missing_dataset() -> None:
    with pytest.raises(ValueError, match="dataset YAML"):
        build_yolo_train_args({"dataset_yaml": "C:/missing/data.yaml"})
