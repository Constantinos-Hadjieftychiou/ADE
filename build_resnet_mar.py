#!/usr/bin/env python3
"""
build_resnet_mar.py  (multi-model version)

Builds TorchServe .mar files for several ImageNet classifiers and puts them in ./model_store.

Currently builds:
  - resnet-18
  - resnet-50
  - mobilenet-v2

Each model gets:
  - TorchScript file (e.g. resnet18_scripted.pt)
  - .mar file in ./model_store using the 'image_classifier' handler
  - Shared index_to_name.json with ImageNet class labels
"""

import json
import subprocess
from pathlib import Path
from urllib.request import urlopen

import torch
from torchvision.models import (
    resnet18,
    resnet50,
    mobilenet_v2,
    ResNet18_Weights,
    ResNet50_Weights,
    MobileNet_V2_Weights,
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_STORE = PROJECT_DIR / "model_store"
MODEL_STORE.mkdir(exist_ok=True)

IDX2NAME_PATH = PROJECT_DIR / "index_to_name.json"

# Define the models you want to build here
MODEL_SPECS = {
    "resnet-18": {
        "ctor": resnet18,
        "weights": ResNet18_Weights.DEFAULT,
        "scripted_filename": "resnet18_scripted.pt",
    },
    "resnet-50": {
        "ctor": resnet50,
        "weights": ResNet50_Weights.DEFAULT,
        "scripted_filename": "resnet50_scripted.pt",
    },
    "mobilenet-v2": {
        "ctor": mobilenet_v2,
        "weights": MobileNet_V2_Weights.DEFAULT,
        "scripted_filename": "mobilenet_v2_scripted.pt",
    },
}


def build_torchscript(model_name: str, spec: dict) -> Path:
    scripted_path = PROJECT_DIR / spec["scripted_filename"]
    if scripted_path.exists():
        print(f"[build_mar] TorchScript already exists for {model_name} at {scripted_path}")
        return scripted_path

    print(f"[build_mar] Loading {model_name} with ImageNet weights…")
    model = spec["ctor"](weights=spec["weights"])
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(scripted_path)
    print(f"[build_mar] Saved TorchScript model for {model_name} to {scripted_path}")
    return scripted_path


def build_index_to_name():
    if IDX2NAME_PATH.exists():
        print(f"[build_mar] index_to_name.json already exists at {IDX2NAME_PATH}")
        return
    print("[build_mar] Downloading ImageNet class names…")
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    with urlopen(url) as resp:
        lines = resp.read().decode("utf-8").strip().splitlines()
    mapping = {str(i): name for i, name in enumerate(lines)}
    with IDX2NAME_PATH.open("w") as f:
        json.dump(mapping, f)
    print(f"[build_mar] Wrote index_to_name.json to {IDX2NAME_PATH}")


def build_mar(model_name: str, scripted_path: Path):
    mar_path = MODEL_STORE / f"{model_name}.mar"
    if mar_path.exists():
        print(f"[build_mar] {mar_path} already exists, skipping archiver.")
        return

    print(f"[build_mar] Creating {mar_path} with torch-model-archiver…")
    cmd = [
        "torch-model-archiver",
        "--model-name",
        model_name,
        "--version",
        "1.0",
        "--serialized-file",
        str(scripted_path),
        "--handler",
        "image_classifier",
        "--extra-files",
        str(IDX2NAME_PATH),
        "--export-path",
        str(MODEL_STORE),
        "--force",
    ]
    print("[build_mar] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[build_mar] ✅ Created {mar_path}")


def main():
    build_index_to_name()

    for model_name, spec in MODEL_SPECS.items():
        print(f"\n=== Building artifacts for {model_name} ===")
        scripted_path = build_torchscript(model_name, spec)
        build_mar(model_name, scripted_path)


if __name__ == "__main__":
    main()
