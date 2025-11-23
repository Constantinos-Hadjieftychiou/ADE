#!/usr/bin/env python3
"""
build_resnet_mar.py

Builds a TorchServe .mar for ResNet-18 and puts it in ./model_store.
If resnet-18.mar already exists, it does nothing.
"""

import json
import subprocess
from pathlib import Path
from urllib.request import urlopen

import torch
from torchvision.models import resnet18, ResNet18_Weights

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_STORE = PROJECT_DIR / "model_store"
MODEL_STORE.mkdir(exist_ok=True)

MAR_PATH = MODEL_STORE / "resnet-18.mar"
SCRIPTED_PATH = PROJECT_DIR / "resnet18_scripted.pt"
IDX2NAME_PATH = PROJECT_DIR / "index_to_name.json"


def build_torchscript():
    if SCRIPTED_PATH.exists():
        print(f"[build_resnet_mar] TorchScript already exists at {SCRIPTED_PATH}")
        return
    print("[build_resnet_mar] Loading ResNet-18 with ImageNet weights…")
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(SCRIPTED_PATH)
    print(f"[build_resnet_mar] Saved TorchScript model to {SCRIPTED_PATH}")


def build_index_to_name():
    if IDX2NAME_PATH.exists():
        print(f"[build_resnet_mar] index_to_name.json already exists at {IDX2NAME_PATH}")
        return
    print("[build_resnet_mar] Downloading ImageNet class names…")
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    with urlopen(url) as resp:
        lines = resp.read().decode("utf-8").strip().splitlines()
    mapping = {str(i): name for i, name in enumerate(lines)}
    with IDX2NAME_PATH.open("w") as f:
        json.dump(mapping, f)
    print(f"[build_resnet_mar] Wrote index_to_name.json to {IDX2NAME_PATH}")


def build_mar():
    if MAR_PATH.exists():
        print(f"[build_resnet_mar] {MAR_PATH} already exists, skipping archiver.")
        return
    print(f"[build_resnet_mar] Creating {MAR_PATH} with torch-model-archiver…")
    cmd = [
        "torch-model-archiver",
        "--model-name",
        "resnet-18",
        "--version",
        "1.0",
        "--serialized-file",
        str(SCRIPTED_PATH),
        "--handler",
        "image_classifier",
        "--extra-files",
        str(IDX2NAME_PATH),
        "--export-path",
        str(MODEL_STORE),
        "--force",
    ]
    print("[build_resnet_mar] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[build_resnet_mar] ✅ Created {MAR_PATH}")


def main():
    build_torchscript()
    build_index_to_name()
    build_mar()


if __name__ == "__main__":
    main()
