"""Validate a YOLO detection dataset before training.

Run from the repository root:
    python -m src.data.validate_yolo_dataset --data data/yolo/data.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _load_config(data_yaml: Path) -> dict[str, Any]:
    with data_yaml.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {data_yaml}")
    return config


def validate_dataset(data_yaml: Path) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for a YOLO detection dataset."""
    errors: list[str] = []
    warnings: list[str] = []
    data_yaml = data_yaml.resolve()

    if not data_yaml.is_file():
        return [f"Dataset YAML file does not exist: {data_yaml}"], warnings

    try:
        config = _load_config(data_yaml)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        return [f"Could not read dataset YAML: {exc}"], warnings

    names = config.get("names")
    if isinstance(names, list):
        class_names = names
    elif isinstance(names, dict):
        try:
            class_names = [names[key] for key in sorted(names, key=lambda item: int(item))]
        except (ValueError, KeyError):
            return ["'names' must use consecutive integer class IDs starting at 0."], warnings
    else:
        return ["Dataset YAML must define 'names' as a list or integer-keyed mapping."], warnings

    if not class_names:
        errors.append("At least one class must be defined in 'names'.")

    # Ultralytics resolves relative dataset paths from the process working directory.
    dataset_root_value = config.get("path", data_yaml.parent)
    dataset_root = Path(dataset_root_value)
    if not dataset_root.is_absolute():
        dataset_root = (Path.cwd() / dataset_root).resolve()

    for split in ("train", "val", "test"):
        split_value = config.get(split)
        if not split_value:
            if split == "test":
                warnings.append("No 'test' split is configured.")
            else:
                errors.append(f"Dataset YAML is missing the required '{split}' split.")
            continue

        image_dir = Path(split_value)
        if not image_dir.is_absolute():
            image_dir = dataset_root / image_dir
        image_dir = image_dir.resolve()

        if not image_dir.is_dir():
            if split == "test":
                warnings.append(f"Test image directory not found: {image_dir}")
            else:
                errors.append(f"{split} image directory not found: {image_dir}")
            continue

        images = sorted(
            path for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            if split == "test":
                warnings.append(f"No images found in test split: {image_dir}")
            else:
                errors.append(f"No images found in {split} split: {image_dir}")
            continue

        label_dir = image_dir.parent.parent / "labels" / image_dir.name
        missing_labels = 0
        for image_path in images:
            label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
            if not label_path.exists():
                # Missing label files can represent background images, but should be reviewed.
                missing_labels += 1
                continue

            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                errors.append(f"Cannot read {label_path}: {exc}")
                continue

            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                fields = line.split()
                location = f"{label_path}:{line_number}"
                if len(fields) != 5:
                    errors.append(f"{location}: expected 5 values (class x_center y_center width height).")
                    continue
                try:
                    class_id = int(fields[0])
                    x_center, y_center, width, height = map(float, fields[1:])
                except ValueError:
                    errors.append(f"{location}: labels must contain numeric values.")
                    continue

                if class_id < 0 or class_id >= len(class_names):
                    errors.append(f"{location}: class ID {class_id} is outside [0, {len(class_names) - 1}].")
                if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
                    errors.append(f"{location}: normalized coordinates must be between 0 and 1.")
                if width <= 0 or height <= 0:
                    errors.append(f"{location}: bounding-box width and height must be greater than 0.")

        if missing_labels:
            warnings.append(
                f"{split}: {missing_labels} image(s) have no matching label file; "
                "check that these are intentional background images."
            )

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Check YOLO dataset paths and label files.")
    parser.add_argument("--data", type=Path, default=Path("data/yolo/data.yaml"),
                        help="Path to the YOLO dataset YAML (default: data/yolo/data.yaml).")
    args = parser.parse_args()

    errors, warnings = validate_dataset(args.data)
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")

    if errors:
        print(f"Dataset validation failed with {len(errors)} error(s).")
        return 1

    print("Dataset validation passed (review any warnings above).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
