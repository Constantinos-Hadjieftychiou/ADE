#!/usr/bin/env python3
"""
Build TorchServe model archives (.mar files) for ResNet, MobileNet, and other models.

A .mar (model archive) is a ZIP file containing:
- The model weights (TorchScript or .pt file)
- A handler (Python code to preprocess input and postprocess output)
- Optional metadata and config files

This script creates .mar files for TorchServe to load and serve.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def create_handler_file(handler_path: str, model_name: str) -> None:
    """Create a TorchServe handler Python file."""
    handler_code = r'''
import io
import json
import logging
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


class ImageClassificationHandler:
    """TorchServe handler for image classification models."""

    def __init__(self):
        self.model = None
        self.device = None
        self.transforms = None

    def initialize(self, context):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_dir = context.system_properties.get("model_dir")
        model_path = os.path.join(model_dir, "model.pt")

        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        logger.info("Model initialized successfully")

    def handle(self, data, context):
        responses = []
        for request_body in data:
            try:
                image = Image.open(io.BytesIO(request_body)).convert("RGB")
                input_tensor = self.transforms(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.model(input_tensor)

                probabilities = F.softmax(output, dim=1)
                top5_prob, top5_idx = torch.topk(probabilities, 5, dim=1)

                result = {
                    "top_predictions": [
                        {"class_idx": int(idx.item()), "probability": float(prob.item())}
                        for idx, prob in zip(top5_idx[0], top5_prob[0])
                    ],
                }
                responses.append(json.dumps(result))
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                responses.append(json.dumps({"error": str(e)}))
        return responses


_service = ImageClassificationHandler()


def handle(data, context):
    return _service.handle(data, context)


def initialize(context):
    _service.initialize(context)
'''
    with open(handler_path, "w", encoding="utf-8") as f:
        f.write(handler_code)
    logger.info(f"Created handler: {handler_path}")


def create_index_to_name_file(output_path: str) -> None:
    """Create index_to_name.json mapping class indices to labels."""
    index_to_name = {
        "0": "airplane",
        "1": "automobile",
        "2": "bird",
        "3": "cat",
        "4": "deer",
        "5": "dog",
        "6": "frog",
        "7": "horse",
        "8": "ship",
        "9": "truck",
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index_to_name, f, indent=2)
    logger.info(f"Created index_to_name mapping: {output_path}")


def create_model_config_file(output_path: str) -> None:
    """Create model-config.yaml for TorchServe model-specific configuration."""
    config = """\
# TorchServe model configuration
handler: image_classifier
batch_size: 1
min_workers: 1
max_workers: 1
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(config)
    logger.info(f"Created model config: {output_path}")


def build_mar(
    model_name: str,
    scripted_model_path: str,
    handler_path: str,
    output_dir: str,
    extra_files: Optional[List[str]] = None,
    version: str = "1.0",
) -> str:
    """Build a TorchServe model archive (.mar file) using torch-model-archiver."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "torch-model-archiver",
        "--model-name", model_name,
        "--version", str(version),
        "--serialized-file", os.path.abspath(scripted_model_path),
        "--handler", os.path.abspath(handler_path),
        "--export-path", output_dir,
        "--force",
    ]

    if extra_files:
        existing = [os.path.abspath(p) for p in extra_files if os.path.exists(p)]
        if existing:
            # torch-model-archiver expects a single comma-separated --extra-files value
            cmd.extend(["--extra-files", ",".join(existing)])

    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info(f"Archiver stdout: {result.stdout.strip()}")
        if result.stderr:
            logger.warning(f"Archiver stderr: {result.stderr.strip()}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build .mar: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise

    mar_path = os.path.join(output_dir, f"{model_name}.mar")
    if os.path.exists(mar_path):
        logger.info(f"Successfully created: {mar_path}")
        return mar_path
    raise RuntimeError(f"Expected .mar file not found at {mar_path}")


def create_scripted_model(model_name: str, output_path: str) -> None:
    """Create a TorchScript model (.pt) from a torchvision pretrained model."""
    import torch
    import torchvision.models as models

    logger.info(f"Creating TorchScript model for {model_name}")

    # Note: torchvision API has changed across versions; keep your pinned versions in sbatch.
    if model_name == "resnet-18":
        model = models.resnet18(pretrained=True)
    elif model_name == "resnet-50":
        model = models.resnet50(pretrained=True)
    elif model_name == "mobilenet-v2":
        model = models.mobilenet_v2(pretrained=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    scripted_model = torch.jit.trace(model, dummy_input)
    scripted_model.save(output_path)
    logger.info(f"Saved TorchScript model: {output_path}")


def main() -> None:
    project_dir = Path(__file__).parent.absolute()
    model_store = project_dir / "model_store"
    model_store.mkdir(exist_ok=True)

    models_to_build = ["resnet-18", "resnet-50", "mobilenet-v2"]

    for model_name in models_to_build:
        logger.info(f"Building .mar for {model_name}")

        # If you already downloaded/provided the MAR, skip rebuilding.
        expected_mar = model_store / f"{model_name}.mar"
        if expected_mar.exists():
            logger.info(f".mar already exists at {expected_mar}; skipping build")
            continue

        safe_name = model_name.replace("-", "_")

        scripted_model_path = project_dir / f"{safe_name}_scripted.pt"
        handler_path = project_dir / f"handler_{safe_name}.py"
        index_to_name_path = project_dir / "index_to_name.json"
        model_config_path = project_dir / "model-config.yaml"

        if not scripted_model_path.exists():
            logger.info(f"TorchScript does not exist at {scripted_model_path}; creating...")
            create_scripted_model(model_name, str(scripted_model_path))
        else:
            logger.info(f"TorchScript already exists at {scripted_model_path}")

        create_handler_file(str(handler_path), model_name)
        create_index_to_name_file(str(index_to_name_path))
        create_model_config_file(str(model_config_path))

        try:
            mar_path = build_mar(
                model_name=model_name,
                scripted_model_path=str(scripted_model_path),
                handler_path=str(handler_path),
                output_dir=str(model_store),
                extra_files=[str(index_to_name_path)],
                version="1.0",
            )
            logger.info(f"✓ Built {model_name}: {mar_path}")
        except Exception as e:
            logger.error(f"✗ Failed to build {model_name}: {e}")
            sys.exit(1)

    logger.info("All .mar files ready!")


if __name__ == "__main__":
    main()