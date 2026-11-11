#!/usr/bin/env python3
"""Export a trusted Ultralytics segmentation checkpoint for LightCraft's Rust runtime."""

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


def names_from_model(model):
    names = model.names
    if isinstance(names, dict):
        return [str(names[key]).strip() for key in sorted(names, key=lambda value: int(value))]
    if isinstance(names, (list, tuple)):
        return [str(name).strip() for name in names]
    return []


def infer_kind(checkpoint):
    filename = checkpoint.name.lower()
    matches = [kind for kind in ("yolo26", "yoloe") if kind in filename]
    if len(matches) != 1:
        raise ValueError(
            f"cannot infer model type from checkpoint filename {checkpoint.name!r}; "
            "include 'yolo26' or 'yoloe', or pass --kind explicitly"
        )
    return matches[0]


def read_labels(path):
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise ValueError(f"{path} contains no labels")
    return labels


def validate_labels(labels):
    if not 1 <= len(labels) <= 1024:
        raise ValueError("the exported model must have between 1 and 1024 labels")
    normalized = [label.strip().lower() for label in labels]
    if any(not label or len(label.encode("utf-8")) > 256 or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in label) for label in normalized):
        raise ValueError("labels must be unique, non-empty text of at most 256 UTF-8 bytes")
    if len(set(normalized)) != len(normalized):
        raise ValueError("labels must be unique, non-empty text of at most 256 UTF-8 bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=("auto", "yolo26", "yoloe"),
        default="auto",
        help="model type (default: infer from checkpoint filename)",
    )
    parser.add_argument("--input-pt", type=Path, required=True, help="trusted .pt checkpoint")
    parser.add_argument(
        "--labels",
        type=Path,
        help="YOLOE text prompts, one per line (default: tools/yoloe-default-labels.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="parent directory for the model folder (default: directory containing --input-pt)",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        dest="create_zip",
        help="also create a ZIP containing the model folder in the output directory",
    )
    args = parser.parse_args()

    if not args.input_pt.is_file():
        parser.error(f"checkpoint not found: {args.input_pt}")
    if args.output is None:
        args.output = args.input_pt.parent
    model_name = args.input_pt.stem
    if model_name in ("", ".", ".."):
        parser.error(f"cannot derive a model folder name from checkpoint: {args.input_pt.name!r}")
    model_dir = args.output / model_name
    if args.kind == "auto":
        try:
            args.kind = infer_kind(args.input_pt)
        except ValueError as error:
            parser.error(str(error))
    if args.kind == "yolo26" and args.labels:
        parser.error("--labels is only used with YOLOE")
    if args.output.exists() and not args.output.is_dir():
        parser.error(f"output is not a directory: {args.output}")
    if model_dir.exists():
        parser.error(f"model output already exists: {model_dir}")
    zip_path = args.output / f"{model_name}.zip"
    if args.create_zip and zip_path.exists():
        parser.error(f"ZIP output already exists: {zip_path}")
    if args.kind == "yoloe":
        labels_path = args.labels or Path(__file__).resolve().with_name("yoloe-default-labels.txt")
        try:
            labels = read_labels(labels_path)
            validate_labels(labels)
        except (OSError, ValueError) as error:
            parser.error(str(error))

    print("Warning: loading a .pt checkpoint can execute pickle code; only convert a checkpoint you trust.")
    if args.kind == "yoloe":
        try:
            from ultralytics import YOLOE
        except ImportError as error:
            parser.error(f"install an Ultralytics version with YOLOE support in this conversion environment: {error}")
        model = YOLOE(str(args.input_pt))
    else:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            parser.error(f"install Ultralytics in this conversion environment first: {error}")
        model = YOLO(str(args.input_pt))
    if model.task != "segment":
        parser.error(f"checkpoint must be an instance-segmentation model (found task {model.task!r})")
    if args.kind == "yoloe":
        model.set_classes(labels)
    else:
        labels = names_from_model(model)
    validate_labels(labels)

    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lightcraft-yolo-export-", dir=args.output) as staging:
        exported = model.export(
            format="onnx",
            imgsz=640,
            batch=1,
            dynamic=False,
            nms=False,
            simplify=False,
            opset=17,
            device="cpu",
            project=staging,
            name="export",
            exist_ok=True,
        )
        exported = Path(exported)
        if not exported.is_file():
            raise FileNotFoundError(f"Ultralytics did not produce an ONNX file: {exported}")

        staged_model = Path(staging) / "model.onnx"
        try:
            shutil.copyfile(exported, staged_model)
        finally:
            if exported.resolve() != staged_model.resolve():
                exported.unlink(missing_ok=True)
        manifest = {"format": "lightcraft-yolo-seg", "version": 1, "image_size": 640, "labels": labels}
        staged_manifest = Path(staging) / "lightcraft-yolo.json"
        staged_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staged_model_dir = Path(staging) / "bundle"
        staged_model_dir.mkdir()
        os.replace(staged_model, staged_model_dir / staged_model.name)
        os.replace(staged_manifest, staged_model_dir / staged_manifest.name)
        staged_zip = Path(staging) / zip_path.name
        if args.create_zip:
            with zipfile.ZipFile(staged_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(staged_model_dir.iterdir()):
                    archive.write(path, arcname=f"{model_name}/{path.name}")
        os.replace(staged_model_dir, model_dir)
        if args.create_zip:
            os.replace(staged_zip, zip_path)

    print(f"LightCraft model written to {model_dir}")
    if args.create_zip:
        print(f"ZIP archive written to {zip_path}")


if __name__ == "__main__":
    main()
