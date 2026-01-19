
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
