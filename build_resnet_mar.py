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
import shutil
import subprocess
import sys
from pathlib import Path

# Configure logging with timestamp and level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def create_handler_file(handler_path: str, model_name: str) -> None:
    """
    Create a TorchServe handler Python file.
    
    The handler defines:
    - initialize(context): called once on model load (setup)
    - handle(data, context): called per request (forward pass)
    
    This generic handler:
    - Loads the model from context
    - Preprocesses input (normalize, resize for images)
    - Runs forward pass
    - Postprocesses output (softmax, top-k classes)
    
    Args:
        handler_path: where to write the handler.py file
        model_name: name of the model (for logging purposes)
    """
    # Python code string that will be written to handler_path
    # This code will be embedded in the .mar archive and executed by TorchServe
    handler_code = '''
import logging
# For debug/info output from the handler
import torch
# PyTorch library for model inference
import torch.nn.functional as F
# PyTorch functional API (softmax, etc.)
from torchvision import transforms
# Image preprocessing utilities from torchvision
from PIL import Image
# Image loading library
import io
# In-memory byte stream handling
import json
# JSON serialization for responses

logger = logging.getLogger(__name__)


class ImageClassificationHandler:
    """TorchServe handler for image classification models."""
    
    def __init__(self):
        """Initialize handler state (model, device, transforms will be set in initialize())."""
        # Placeholder for model and other state
        self.model = None
        # Model instance (loaded in initialize())
        self.device = None
        # GPU or CPU device for inference
        self.transforms = None
        # Image preprocessing pipeline
    
    def initialize(self, context):
        """
        Called once when TorchServe loads the model on startup.
        
        context provides:
        - context.system_properties: device info, model directory
        - context.model_yaml_config: model-specific configuration
        
        Args:
            context: TorchServe context object with system properties
        """
        # Determine if GPU is available, else use CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Get model directory from TorchServe context
        model_dir = context.system_properties.get("model_dir")
        # Construct path to TorchScript model file (always named model.pt in .mar)
        model_path = os.path.join(model_dir, "model.pt")
        
        # Load TorchScript model from disk (pre-compiled PyTorch model)
        self.model = torch.jit.load(model_path, map_location=self.device)
        # Set to eval mode: disables dropout, freezes batch norm layers
        self.model.eval()
        
        # Define image preprocessing pipeline (resize, normalize, convert to tensor)
        self.transforms = transforms.Compose([
            # Resize all images to 224x224 (standard ImageNet size)
            transforms.Resize((224, 224)),
            # Convert PIL Image to PyTorch tensor (values 0-1)
            transforms.ToTensor(),
            # Normalize using ImageNet statistics (pre-computed means/stds)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean for R, G, B
                std=[0.229, 0.224, 0.225],   # ImageNet std dev for R, G, B
            ),
        ])
        
        logger.info("Model initialized successfully")
    
    def handle(self, data, context):
        """
        Called for each inference request by TorchServe.
        
        Receives HTTP request body, preprocesses, runs model, returns predictions.
        
        Args:
            data: list of HTTP request bodies (bytes, one per request)
            context: request context from TorchServe
            
        Returns:
            list of response strings (JSON), one per request
        """
        responses = []
        
        # Process each request in the batch
        for request_body in data:
            try:
                # Decode request body as JPEG image bytes
                image = Image.open(io.BytesIO(request_body)).convert("RGB")
                # Convert to RGB (handles RGBA, grayscale, etc.)
                
                # Apply preprocessing transforms (resize, normalize)
                input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
                # unsqueeze(0): add batch dimension (1, 3, 224, 224)
                # .to(device): move to GPU or CPU
                
                # Run model inference (no gradient computation needed)
                with torch.no_grad():
                    # Inference mode: speeds up computation by skipping backward pass
                    output = self.model(input_tensor)
                
                # Apply softmax to convert logits to probabilities
                probabilities = F.softmax(output, dim=1)
                
                # Get top-5 predictions (class indices and probabilities)
                top5_prob, top5_idx = torch.topk(probabilities, 5, dim=1)
                
                # Format response as JSON with top-5 predictions
                result = {
                    "top_predictions": [
                        {
                            "class_idx": int(idx.item()),
                            # Convert tensor to Python int
                            "probability": float(prob.item()),
                            # Convert tensor to Python float
                        }
                        for idx, prob in zip(top5_idx[0], top5_prob[0])
                    ],
                }
                responses.append(json.dumps(result))
            
            except Exception as e:
                # Log error and return error response (don't crash)
                logger.error(f"Error processing request: {e}")
                responses.append(json.dumps({"error": str(e)}))
        
        return responses


# Create singleton handler instance (TorchServe discovery pattern)
_service = ImageClassificationHandler()


def handle(data, context):
    """Entry point for TorchServe batch inference requests."""
    # Delegate to singleton instance handle method
    return _service.handle(data, context)


def initialize(context):
    """Entry point for TorchServe model initialization on startup."""
    # Delegate to singleton instance initialize method
    _service.initialize(context)
'''
    
    # Write handler code to file
    with open(handler_path, "w") as f:
        f.write(handler_code)
    logger.info(f"Created handler: {handler_path}")


def create_index_to_name_file(output_path: str) -> None:
    """
    Create index_to_name.json mapping class indices to human-readable labels.
    
    This file is optional but useful for model serving (can be used to
    label predictions with class names like "cat", "dog", etc.).
    
    Args:
        output_path: where to write index_to_name.json
    """
    # CIFAR-10 dataset has 10 classes, indexed 0-9
    index_to_name = {
        "0": "airplane",      # Index 0 -> airplane
        "1": "automobile",    # Index 1 -> automobile
        "2": "bird",          # Index 2 -> bird
        "3": "cat",           # Index 3 -> cat
        "4": "deer",          # Index 4 -> deer
        "5": "dog",           # Index 5 -> dog
        "6": "frog",          # Index 6 -> frog
        "7": "horse",         # Index 7 -> horse
        "8": "ship",          # Index 8 -> ship
        "9": "truck",         # Index 9 -> truck
    }
    
    # Write mapping to JSON file
    with open(output_path, "w") as f:
        json.dump(index_to_name, f, indent=2)
    logger.info(f"Created index_to_name mapping: {output_path}")


def create_model_config_file(output_path: str) -> None:
    """
    Create model-config.yaml for TorchServe model-specific configuration.
    
    This file can specify:
    - handler: which handler class/function to use
    - batch_size: default batch size for model
    - worker settings: min/max workers
    - other model-specific tuning parameters
    
    Args:
        output_path: where to write model-config.yaml
    """
    # YAML configuration for the model
    config = """
# TorchServe model configuration
handler: image_classifier
# Handler function to use (defined in handler.py)
batch_size: 1
# Process one request at a time (no batching)
min_workers: 1
# Minimum number of worker threads for this model
max_workers: 1
# Maximum number of worker threads for this model
"""
    # Write configuration to file
    with open(output_path, "w") as f:
        f.write(config)
    logger.info(f"Created model config: {output_path}")


def build_mar(
    model_name: str,
    scripted_model_path: str,
    handler_path: str,
    output_dir: str,
    extra_files: list = None,
) -> str:
    """
    Build a TorchServe model archive (.mar file) using torch-model-archiver.
    
    A .mar file is a ZIP archive containing:
    - model.pt: the TorchScript model
    - handler.py: the request handler
    - index_to_name.json: class label mapping
    - Any other extra files
    
    Args:
        model_name: name for the model (e.g., "resnet-18")
        scripted_model_path: path to .pt file (TorchScript)
        handler_path: path to handler.py
        output_dir: where to save .mar file
        extra_files: list of extra files to include (e.g., metadata)
    
    Returns:
        path to generated .mar file
    """
    # Ensure output directory exists
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Build torch-model-archiver command
    cmd = [
        "torch-model-archiver",
        # Main archiver tool command
        "--model-name", model_name,
        # Name of the model (will be served as /predictions/<model-name>)
        "--serialized-file", os.path.abspath(scripted_model_path),
        # Path to TorchScript model (.pt file)
        "--handler", os.path.abspath(handler_path),
        # Path to handler Python file
        "--export-path", output_dir,
        # Where to save the .mar file
        "--force",
        # Overwrite existing .mar if present
    ]
    
    # Add optional extra files (index_to_name.json, requirements.txt, etc.)
    if extra_files:
        # Iterate over list of extra files to include
        for extra_file in extra_files:
            # Only add if file actually exists
            if os.path.exists(extra_file):
                cmd.extend(["--extra-files", os.path.abspath(extra_file)])
    
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        # Execute archiver command as subprocess
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Archiver stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"Archiver stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        # Command failed; log details and raise
        logger.error(f"Failed to build .mar: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise
    
    # Return path to generated .mar file
    mar_path = os.path.join(output_dir, f"{model_name}.mar")
    if os.path.exists(mar_path):
        logger.info(f"Successfully created: {mar_path}")
        return mar_path
    else:
        raise RuntimeError(f"Expected .mar file not found at {mar_path}")


def create_scripted_model(model_name: str, output_path: str) -> None:
    """
    Create a TorchScript model (.pt file) from a torchvision pre-trained model.
    
    TorchScript is a subset of Python that can be compiled and optimized.
    Using TorchScript models in TorchServe is more efficient than regular .pt files.
    
    Supported models:
    - resnet-18, resnet-50: ResNet convolutional architectures
    - mobilenet-v2: MobileNet v2 (lightweight, good for mobile/edge)
    
    Args:
        model_name: model name (e.g., "resnet-18")
        output_path: where to save the .pt file
    """
    import torch
    # PyTorch library
    import torchvision.models as models
    # Pre-trained models from torchvision
    
    logger.info(f"Creating TorchScript model for {model_name}")
    
    # Load pretrained model from torchvision (downloads if not cached)
    if model_name == "resnet-18":
        # ResNet-18: 18 layers, good accuracy/speed tradeoff
        model = models.resnet18(pretrained=True)
    elif model_name == "resnet-50":
        # ResNet-50: 50 layers, higher accuracy but slower
        model = models.resnet50(pretrained=True)
    elif model_name == "mobilenet-v2":
        # MobileNet v2: lightweight, optimized for mobile devices
        model = models.mobilenet_v2(pretrained=True)
    else:
        # Unknown model name
        raise ValueError(f"Unknown model: {model_name}")
    
    # Set to eval mode (disables dropout, freezes batch norm)
    model.eval()
    
    # Create dummy input for tracing (dummy image: 1x3x224x224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Use torch.jit.trace to convert model to TorchScript
    # (alternative: torch.jit.script for dynamic shapes, but requires annotations)
    scripted_model = torch.jit.trace(model, dummy_input)
    
    # Save to disk (.pt file)
    scripted_model.save(output_path)
    logger.info(f"Saved TorchScript model: {output_path}")


def main() -> None:
    """Main entry point: build .mar files for all models."""
    # Get project root directory (where this script is located)
    project_dir = Path(__file__).parent.absolute()
    
    # Directory to store .mar files
    model_store = project_dir / "model_store"
    # Create if doesn't exist
    model_store.mkdir(exist_ok=True)
    
    # List of models to build .mar archives for
    models_to_build = [
        "resnet-18",
        "resnet-50",
        "mobilenet-v2",
    ]
    
    # Build each model
    for model_name in models_to_build:
        logger.info(f"Building .mar for {model_name}")
        
        # Sanitize model name for filenames (replace hyphens with underscores)
        safe_name = model_name.replace("-", "_")
        
        # Define paths for intermediate and final files
        scripted_model_path = project_dir / f"{safe_name}_scripted.pt"
        # Where TorchScript .pt file will be stored
        handler_path = project_dir / f"handler_{safe_name}.py"
        # Where handler.py will be written
        index_to_name_path = project_dir / "index_to_name.json"
        # Class label mapping (shared across models)
        model_config_path = project_dir / "model-config.yaml"
        # Model configuration (shared across models)
        
        # Step 1: Create TorchScript model if it doesn't exist
        if not scripted_model_path.exists():
            # Model doesn't exist yet; create it
            logger.info(f"TorchScript does not exist at {scripted_model_path}; creating...")
            create_scripted_model(model_name, str(scripted_model_path))
        else:
            # Model already exists; skip
            logger.info(f"TorchScript already exists at {scripted_model_path}")
        
        # Step 2: Create handler file
        create_handler_file(str(handler_path), model_name)
        
        # Step 3: Create metadata files (index_to_name.json, model-config.yaml)
        create_index_to_name_file(str(index_to_name_path))
        create_model_config_file(str(model_config_path))
        
        # Step 4: Build .mar archive
        try:
            # Call archiver to create .mar file
            mar_path = build_mar(
                model_name=model_name,
                scripted_model_path=str(scripted_model_path),
                handler_path=str(handler_path),
                output_dir=str(model_store),
                extra_files=[str(index_to_name_path)],
            )
            logger.info(f"✓ Built {model_name}: {mar_path}")
        except Exception as e:
            # Archiving failed
            logger.error(f"✗ Failed to build {model_name}: {e}")
            sys.exit(1)
    
    logger.info("All .mar files built successfully!")


if __name__ == "__main__":
    main()
