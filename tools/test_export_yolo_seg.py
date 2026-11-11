"""Synthetic export checks. Run with python3 -m unittest discover -s tools."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


EXPORTER_PATH = Path(__file__).with_name("export-yolo-seg.py")
SPEC = importlib.util.spec_from_file_location("export_yolo_seg", EXPORTER_PATH)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class FakeModel:
    task = "segment"
    names = {0: "person", 1: "dog"}
    labels = None

    def set_classes(self, labels):
        self.labels = labels

    def export(self, **kwargs):
        if self.labels is None:
            raise AssertionError("YOLOE classes must be set before export")
        onnx_path = Path(kwargs["project"]).parent / "yoloe-test.onnx"
        onnx_path.write_bytes(b"synthetic ONNX")
        return str(onnx_path)


class ExportTests(unittest.TestCase):
    def run_yoloe_export(self, root, extra_args):
        model = FakeModel()
        ultralytics = ModuleType("ultralytics")
        ultralytics.YOLOE = lambda _checkpoint: model

        checkpoint = root / "yoloe-test.pt"
        checkpoint.write_bytes(b"trusted synthetic checkpoint")
        output = root / "converted"
        with patch.dict(sys.modules, {"ultralytics": ultralytics}), patch.object(
            sys, "argv", ["export-yolo-seg.py", "--input-pt", str(checkpoint), "--output", str(output), *extra_args]
        ), contextlib.redirect_stdout(io.StringIO()):
            exporter.main()
        return model, output

    def test_yoloe_uses_default_prompts_and_creates_named_folder_and_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            model, output = self.run_yoloe_export(Path(temp), ["--zip"])
            default_labels = exporter.read_labels(Path(__file__).with_name("yoloe-default-labels.txt"))
            model_dir = output / "yoloe-test"
            self.assertGreaterEqual(len(default_labels), 100)
            self.assertTrue({"person", "dog", "white dog", "red car"}.issubset(default_labels))
            self.assertEqual(model.labels, default_labels)
            self.assertFalse((output / "yoloe-test.onnx").exists())
            self.assertEqual({path.name for path in model_dir.iterdir()}, {"model.onnx", "lightcraft-yolo.json"})
            manifest = json.loads((model_dir / "lightcraft-yolo.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["labels"], default_labels)
            with zipfile.ZipFile(output / "yoloe-test.zip") as archive:
                self.assertEqual(set(archive.namelist()), {"yoloe-test/model.onnx", "yoloe-test/lightcraft-yolo.json"})

    def test_yoloe_accepts_custom_prompt_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            labels_path = root / "labels.txt"
            labels_path.write_text("white dog\nred car\n", encoding="utf-8")
            model, output = self.run_yoloe_export(root, ["--labels", str(labels_path)])
            manifest = json.loads((output / "yoloe-test" / "lightcraft-yolo.json").read_text(encoding="utf-8"))
            self.assertEqual(model.labels, ["white dog", "red car"])
            self.assertEqual(manifest["labels"], ["white dog", "red car"])


if __name__ == "__main__":
    unittest.main()
