"""Unit tests for the YOLO dataset validation utility."""
import tempfile
import unittest
from pathlib import Path

import yaml

from src.data.validate_yolo_dataset import validate_dataset


class ValidateYoloDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.yaml_path = self.root / "data.yaml"
        self.names = {0: "Glioma", 1: "Meningioma"}
        for split in ("train", "val", "test"):
            (self.root / "images" / split).mkdir(parents=True)
            (self.root / "labels" / split).mkdir(parents=True)
            (self.root / "images" / split / "sample.jpg").write_bytes(b"not a real image")
            (self.root / "labels" / split / "sample.txt").write_text(
                "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
            )
        self.write_config()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_config(self):
        config = {
            "path": str(self.root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": self.names,
        }
        self.yaml_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def test_valid_dataset_has_no_errors_or_warnings(self):
        errors, warnings = validate_dataset(self.yaml_path)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_invalid_class_id_is_reported(self):
        label = self.root / "labels" / "train" / "sample.txt"
        label.write_text("4 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        errors, _ = validate_dataset(self.yaml_path)
        self.assertTrue(any("class ID 4" in error for error in errors))

    def test_out_of_range_coordinates_are_reported(self):
        label = self.root / "labels" / "val" / "sample.txt"
        label.write_text("0 1.2 0.5 0.2 0.2\n", encoding="utf-8")
        errors, _ = validate_dataset(self.yaml_path)
        self.assertTrue(any("between 0 and 1" in error for error in errors))

    def test_missing_label_file_is_a_warning(self):
        (self.root / "labels" / "test" / "sample.txt").unlink()
        errors, warnings = validate_dataset(self.yaml_path)
        self.assertEqual(errors, [])
        self.assertTrue(any("background images" in warning for warning in warnings))

    def test_missing_validation_directory_is_an_error(self):
        import shutil
        shutil.rmtree(self.root / "images" / "val")
        errors, _ = validate_dataset(self.yaml_path)
        self.assertTrue(any("val image directory not found" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
