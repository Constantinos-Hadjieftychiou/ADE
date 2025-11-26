#!/usr/bin/env python3
"""
build_resnet_mar.py  (multi-model version)

Build TorchServe .mar files for several ImageNet classifiers and put them in ./model_store.

Currently builds:
  - resnet-18
  - resnet-50
  - mobilenet-v2

For each model we create:
  - a TorchScript file (e.g. resnet18_scripted.pt)
  - a .mar file in ./model_store using the built-in 'image_classifier' handler
  - a shared index_to_name.json with ImageNet class labels, so handler can map
    numeric class indices to human-readable names.
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

# Base directory of this script (project root for these assets).
PROJECT_DIR = Path(__file__).resolve().parent

# TorchServe model store directory where .mar files will be exported.
MODEL_STORE = PROJECT_DIR / "model_store"
MODEL_STORE.mkdir(exist_ok=True)

# Shared mapping from class index -> human-readable label for ImageNet.
IDX2NAME_PATH = PROJECT_DIR / "index_to_name.json"

# Configuration for each model we want to build.
# Each entry contains:
#   - ctor:      function that constructs the torchvision model
#   - weights:   which pretrained weights to load
#   - scripted_filename: filename where we save the TorchScript version
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
    """
    Build (or reuse) a TorchScript version of a given torchvision model.

    Steps:
      1. Check if a scripted .pt file already exists; if yes, reuse it.
      2. Otherwise, load the model with pretrained ImageNet weights.
      3. Set the model to eval() mode.
      4. Use torch.jit.script to generate a TorchScript module.
      5. Save it to disk and return the path.

    Args:
        model_name: Name of the model (e.g. "resnet-18").
        spec:       Dictionary with keys "ctor", "weights", "scripted_filename".

    Returns:
        Path to the saved TorchScript (.pt) file.
    """
    scripted_path = PROJECT_DIR / spec["scripted_filename"]

    # If we've already built this once, don't redo the work.
    if scripted_path.exists():
        print(f"[build_mar] TorchScript already exists for {model_name} at {scripted_path}")
        return scripted_path

    print(f"[build_mar] Loading {model_name} with ImageNet weights…")
    # Construct the model with pretrained weights and switch to inference mode.
    model = spec["ctor"](weights=spec["weights"])
    model.eval()

    # Script the model. This converts the nn.Module into a TorchScript module
    # that TorchServe can load without relying on Python source.
    scripted = torch.jit.script(model)

    # Persist the TorchScript module to disk.
    scripted.save(scripted_path)
    print(f"[build_mar] Saved TorchScript model for {model_name} to {scripted_path}")
    return scripted_path


def build_index_to_name() -> None:
    """
    Ensure that index_to_name.json exists.

    If the file is missing:
      - Download ImageNet class names from the official PyTorch hub URL.
      - Build a mapping { "0": "tench", "1": "goldfish", ... }.
      - Save it to index_to_name.json in PROJECT_DIR.

    TorchServe's 'image_classifier' handler uses this file to convert the
    numeric output class index into a human-readable label.
    """
    if IDX2NAME_PATH.exists():
        print(f"[build_mar] index_to_name.json already exists at {IDX2NAME_PATH}")
        return

    print("[build_mar] Downloading ImageNet class names…")

    # URL used by torchvision examples to expose ImageNet class labels.
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

    # Retrieve the text file and split into one label per line.
    with urlopen(url) as resp:
        lines = resp.read().decode("utf-8").strip().splitlines()

    # Build a dictionary mapping "index" -> "class name"
    mapping = {str(i): name for i, name in enumerate(lines)}

    # Write JSON to disk, pretty compact is fine here (no indent).
    with IDX2NAME_PATH.open("w") as f:
        json.dump(mapping, f)

    print(f"[build_mar] Wrote index_to_name.json to {IDX2NAME_PATH}")


def build_mar(model_name: str, scripted_path: Path) -> None:
    """
    Use torch-model-archiver to create a TorchServe .mar file for one model.

    The resulting .mar will:
      - Use the built-in 'image_classifier' handler.
      - Contain the TorchScript file (scripted_path).
      - Include the extra index_to_name.json for label decoding.
      - Be placed in the MODEL_STORE directory.

    Args:
        model_name:    Name that TorchServe will see (e.g. "resnet-18").
        scripted_path: Path to the TorchScript .pt file for this model.
    """
    mar_path = MODEL_STORE / f"{model_name}.mar"

    # If we've already created the .mar, skip re-archiving.
    if mar_path.exists():
        print(f"[build_mar] {mar_path} already exists, skipping archiver.")
        return

    print(f"[build_mar] Creating {mar_path} with torch-model-archiver…")

    # Command-line call to torch-model-archiver. This is the standard tool
    # provided by TorchServe to bundle models into .mar archives.
    cmd = [
        "torch-model-archiver",
        "--model-name",
        model_name,
        "--version",
        "1.0",
        "--serialized-file",
        str(scripted_path),
        "--handler",
        "image_classifier",          # built-in TorchServe image classification handler
        "--extra-files",
        str(IDX2NAME_PATH),          # used by handler to map indices -> labels
        "--export-path",
        str(MODEL_STORE),            # directory where .mar will be written
        "--force",                   # overwrite if something with this name exists
    ]

    print("[build_mar] Running:", " ".join(cmd))
    # Run the archiver and fail fast if it returns non-zero (check=True).
    subprocess.run(cmd, check=True)
    print(f"[build_mar] ✅ Created {mar_path}")


def main() -> None:
    """
    Main entry point:

      1. Ensure index_to_name.json exists.
      2. For each model in MODEL_SPECS:
           - Build (or reuse) TorchScript file.
           - Build (or reuse) .mar file in MODEL_STORE.

    This script is intended to be run once per environment, or whenever you
    change model definitions / weights and need fresh .mar artifacts.
    """
    # Step 1: Make sure ImageNet label mapping is available.
    build_index_to_name()

    # Step 2: Loop through desired models and build their artifacts.
    for model_name, spec in MODEL_SPECS.items():
        print(f"\n=== Building artifacts for {model_name} ===")
        scripted_path = build_torchscript(model_name, spec)
        build_mar(model_name, scripted_path)


if __name__ == "__main__":
    main()
