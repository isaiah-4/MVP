from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import yaml
from PIL import Image


@dataclass(frozen=True)
class TrainingPreset:
    base_model: str
    image_size: int
    batch_size: int
    epochs: int
    patience: int
    min_dataset_long_side: int
    notes: tuple[str, ...]


RECOMMENDED_PROFILES = {
    "ball": {
        "edge": TrainingPreset(
            base_model="yolo26n.pt",
            image_size=832,
            batch_size=12,
            epochs=220,
            patience=40,
            min_dataset_long_side=832,
            notes=(
                "Fastest local option for the basketball detector.",
                "Use when training budget is limited or inference latency matters more than recall.",
            ),
        ),
        "balanced": TrainingPreset(
            base_model="yolo26s.pt",
            image_size=960,
            batch_size=8,
            epochs=250,
            patience=50,
            min_dataset_long_side=960,
            notes=(
                "Recommended default for this repo.",
                "YOLO26 is the latest Ultralytics detector family and improves small-object recognition.",
                "Higher image size matters more for the ball than a much larger backbone does.",
            ),
        ),
        "quality": TrainingPreset(
            base_model="yolo26m.pt",
            image_size=1280,
            batch_size=6,
            epochs=300,
            patience=60,
            min_dataset_long_side=1280,
            notes=(
                "Highest-quality local training preset in this repo.",
                "Best when the dataset version is exported at high resolution and you can afford slower inference.",
            ),
        ),
    },
    "player": {
        "edge": TrainingPreset(
            base_model="yolo26n.pt",
            image_size=640,
            batch_size=16,
            epochs=120,
            patience=35,
            min_dataset_long_side=640,
            notes=(
                "Fastest local option for player detection.",
                "Use when your target is lightweight local inference on a laptop.",
            ),
        ),
        "balanced": TrainingPreset(
            base_model="yolo26s.pt",
            image_size=640,
            batch_size=12,
            epochs=150,
            patience=45,
            min_dataset_long_side=640,
            notes=(
                "Recommended default for this repo.",
                "YOLO26 gives a better accuracy/speed tradeoff than the older YOLOv5 baseline the notebooks started with.",
            ),
        ),
        "quality": TrainingPreset(
            base_model="yolo26m.pt",
            image_size=960,
            batch_size=8,
            epochs=180,
            patience=55,
            min_dataset_long_side=960,
            notes=(
                "Use for best local accuracy if the final detector can be a bit heavier.",
                "The larger backbone is more useful for crowded player scenes than for easy full-body views.",
            ),
        ),
    },
}


EXPECTED_CLASS_TOKENS = {
    "ball": ("ball",),
    "player": ("player", "person", "human"),
}

OPTIONAL_CLASS_TOKENS = {
    "ball": ("hoop", "rim", "basket"),
    "player": tuple(),
}


def set_seed(seed: int) -> int:
    random.seed(seed)
    return seed


def build_training_config(
    *,
    task_key: str,
    env_prefix: str,
    output_model_path: str,
    default_workspace: str,
    default_project: str,
    default_version: int,
) -> dict[str, Any]:
    task_key = task_key.lower().strip()
    env_prefix = env_prefix.upper().strip()

    if task_key not in RECOMMENDED_PROFILES:
        raise KeyError(f"Unsupported task_key: {task_key}")

    requested_profile = (
        os.environ.get(f"COURTVISION_{env_prefix}_PROFILE")
        or os.environ.get("COURTVISION_TRAIN_PROFILE")
        or "balanced"
    ).lower()
    if requested_profile not in RECOMMENDED_PROFILES[task_key]:
        valid_profiles = ", ".join(sorted(RECOMMENDED_PROFILES[task_key]))
        raise ValueError(
            f"Unsupported training profile '{requested_profile}' for task '{task_key}'. "
            f"Expected one of: {valid_profiles}"
        )

    preset = RECOMMENDED_PROFILES[task_key][requested_profile]
    workers_default = min(8, os.cpu_count() or 2)
    seed = int(os.environ.get("COURTVISION_TRAIN_SEED", "7"))
    set_seed(seed)

    config = {
        "task_key": task_key,
        "env_prefix": env_prefix,
        "seed": seed,
        "profile": requested_profile,
        "task_name": f"basketball-{task_key}-detector",
        "workspace": os.environ.get(f"ROBOFLOW_{env_prefix}_WORKSPACE", default_workspace),
        "project": os.environ.get(f"ROBOFLOW_{env_prefix}_PROJECT", default_project),
        "version": int(os.environ.get(f"ROBOFLOW_{env_prefix}_VERSION", str(default_version))),
        "dataset_format": os.environ.get(f"ROBOFLOW_{env_prefix}_FORMAT", "yolov8"),
        "base_model": os.environ.get(f"COURTVISION_{env_prefix}_BASE_MODEL", preset.base_model),
        "epochs": int(os.environ.get(f"COURTVISION_{env_prefix}_EPOCHS", str(preset.epochs))),
        "image_size": int(os.environ.get(f"COURTVISION_{env_prefix}_IMGSZ", str(preset.image_size))),
        "batch_size": int(os.environ.get(f"COURTVISION_{env_prefix}_BATCH", str(preset.batch_size))),
        "patience": int(os.environ.get(f"COURTVISION_{env_prefix}_PATIENCE", str(preset.patience))),
        "workers": int(os.environ.get(f"COURTVISION_{env_prefix}_WORKERS", str(workers_default))),
        "device": os.environ.get("COURTVISION_TRAIN_DEVICE") or None,
        "runs_dir": Path(os.environ.get("COURTVISION_TRAIN_RUNS_DIR", "runs/detect")),
        "output_model_path": Path(
            os.environ.get(f"COURTVISION_{env_prefix}_OUTPUT_MODEL", output_model_path)
        ),
        "summary_path": Path(
            os.environ.get(
                f"COURTVISION_{env_prefix}_SUMMARY_PATH",
                f"{Path(output_model_path).with_suffix('')}.train.json",
            )
        ),
        "expected_class_tokens": EXPECTED_CLASS_TOKENS[task_key],
        "optional_class_tokens": OPTIONAL_CLASS_TOKENS[task_key],
        "min_dataset_long_side": int(
            os.environ.get(
                f"COURTVISION_{env_prefix}_MIN_DATASET_LONG_SIDE",
                str(preset.min_dataset_long_side),
            )
        ),
        "notes": list(preset.notes),
    }

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("Set ROBOFLOW_API_KEY before running this notebook.")
    config["roboflow_api_key"] = api_key
    return config


def download_dataset(config: dict[str, Any]) -> Path:
    from roboflow import Roboflow

    rf = Roboflow(api_key=config["roboflow_api_key"])
    project = rf.workspace(config["workspace"]).project(config["project"])
    version = project.version(config["version"])
    dataset = version.download(config["dataset_format"])
    return Path(dataset.location).resolve()


def _resolve_split_path(dataset_root: Path, split_value: str | None, split_name: str) -> Path:
    candidates: list[Path] = []
    if split_value:
        raw_path = Path(split_value)
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append((dataset_root / raw_path).resolve())

    candidates.extend(
        [
            (dataset_root / split_name).resolve(),
            (dataset_root / split_name / "images").resolve(),
            (dataset_root / dataset_root.name / split_name).resolve(),
            (dataset_root / dataset_root.name / split_name / "images").resolve(),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve split '{split_name}' from value {split_value!r} inside {dataset_root}."
    )


def normalize_data_yaml(dataset_root: Path) -> Path:
    data_yaml_path = dataset_root / "data.yaml"
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Expected data.yaml at {data_yaml_path}")

    with data_yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Unexpected data.yaml format at {data_yaml_path}")

    data["path"] = str(dataset_root)
    for split_name in ("train", "val", "test"):
        if split_name not in data:
            continue
        resolved_path = _resolve_split_path(dataset_root, data.get(split_name), split_name)
        data[split_name] = str(resolved_path)

    with data_yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)

    return data_yaml_path


def inspect_dataset_images(
    data_yaml_path: Path,
    *,
    sample_limit: int = 64,
) -> dict[str, Any]:
    with data_yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    train_path = Path(data["train"])
    image_paths = sorted(
        path
        for path in train_path.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    sampled_paths = image_paths[:sample_limit]

    if not sampled_paths:
        return {
            "sampled_images": 0,
            "width_min": None,
            "width_median": None,
            "width_max": None,
            "height_min": None,
            "height_median": None,
            "height_max": None,
            "long_side_median": None,
        }

    widths: list[int] = []
    heights: list[int] = []
    for image_path in sampled_paths:
        with Image.open(image_path) as image:
            width, height = image.size
        widths.append(int(width))
        heights.append(int(height))

    long_sides = [max(width, height) for width, height in zip(widths, heights, strict=True)]
    return {
        "sampled_images": len(sampled_paths),
        "width_min": min(widths),
        "width_median": int(median(widths)),
        "width_max": max(widths),
        "height_min": min(heights),
        "height_median": int(median(heights)),
        "height_max": max(heights),
        "long_side_median": int(median(long_sides)),
    }


def build_dataset_summary(
    config: dict[str, Any],
    data_yaml_path: Path,
    image_stats: dict[str, Any],
) -> dict[str, Any]:
    with data_yaml_path.open("r", encoding="utf-8") as handle:
        dataset_config = yaml.safe_load(handle)

    summary = {
        "dataset_root": str(data_yaml_path.parent),
        "data_yaml_path": str(data_yaml_path),
        "classes": dataset_config.get("names"),
        "train": dataset_config.get("train"),
        "val": dataset_config.get("val"),
        "test": dataset_config.get("test"),
        "image_stats": image_stats,
        "warnings": [],
        "notes": list(config["notes"]),
    }

    long_side_median = image_stats.get("long_side_median")
    if long_side_median is not None and long_side_median < config["min_dataset_long_side"]:
        summary["warnings"].append(
            "The exported dataset images appear smaller than the recommended long side for this "
            f"profile ({long_side_median}px vs recommended {config['min_dataset_long_side']}px). "
            "Create a higher-resolution Roboflow dataset version before retraining if ball recall is weak."
        )

    _validate_class_names(config, summary)
    return summary


def _normalize_class_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, dict):
        ordered_items = [raw_names[key] for key in sorted(raw_names)]
        return [str(name).lower() for name in ordered_items]
    if isinstance(raw_names, list):
        return [str(name).lower() for name in raw_names]
    return []


def _validate_class_names(config: dict[str, Any], dataset_summary: dict[str, Any]) -> None:
    class_names = _normalize_class_names(dataset_summary.get("classes"))
    expected_tokens = tuple(config["expected_class_tokens"])
    optional_tokens = tuple(config["optional_class_tokens"])

    if not class_names:
        dataset_summary["warnings"].append("Could not read class names from data.yaml.")
        return

    if not any(any(token in class_name for token in expected_tokens) for class_name in class_names):
        raise ValueError(
            f"Dataset classes {class_names} do not match the expected {config['task_key']} tokens "
            f"{expected_tokens}. Update the Roboflow project/version for this notebook."
        )

    if optional_tokens and not any(
        any(token in class_name for token in optional_tokens) for class_name in class_names
    ):
        dataset_summary["warnings"].append(
            "No hoop/rim class was detected in this dataset version. Tracking will still train, "
            "but shot detection quality will be limited without a hoop class."
        )


def build_training_kwargs(config: dict[str, Any], data_yaml_path: Path) -> dict[str, Any]:
    return {
        "data": str(data_yaml_path),
        "epochs": config["epochs"],
        "imgsz": config["image_size"],
        "batch": config["batch_size"],
        "patience": config["patience"],
        "workers": config["workers"],
        "seed": config["seed"],
        "project": str(config["runs_dir"]),
        "name": config["task_name"],
        "exist_ok": True,
        "plots": True,
        "device": config["device"],
    }


def copy_best_weights(save_dir: Path, output_model_path: Path) -> Path:
    best_weights_path = save_dir / "weights" / "best.pt"
    if not best_weights_path.exists():
        raise FileNotFoundError(f"Training did not produce {best_weights_path}")
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights_path, output_model_path)
    return output_model_path.resolve()


def write_training_summary(
    *,
    config: dict[str, Any],
    dataset_summary: dict[str, Any],
    best_weights_path: Path,
    published_model_path: Path,
    map50: float,
    map50_95: float,
) -> Path:
    payload = {
        "config": {
            key: _json_safe_value(value)
            for key, value in config.items()
            if key != "roboflow_api_key"
        },
        "dataset_summary": dataset_summary,
        "best_weights_path": str(best_weights_path),
        "published_model_path": str(published_model_path),
        "map50": float(map50),
        "map50_95": float(map50_95),
    }
    summary_path = Path(config["summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return summary_path.resolve()


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value
