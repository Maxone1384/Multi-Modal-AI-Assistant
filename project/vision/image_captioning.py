# vision/image_captioning.py

"""
Lightweight Image Captioning module using BLIP (base).
- Usage:
    from vision.image_captioning import generate_caption
    cap = generate_caption("data/test.jpg")
"""

from typing import Union
from PIL import Image
import torch

# Lazy imports / lazy model load to avoid heavy startup on import
_PROCESSOR = None
_MODEL = None

def _ensure_model_loaded(model_name: str = "Salesforce/blip-image-captioning-base", device: str = "cpu"):
    global _PROCESSOR, _MODEL
    if _PROCESSOR is None or _MODEL is None:
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
        except Exception as e:
            raise ImportError("transformers not installed. run: pip install transformers") from e

        # load processor & model (base is reasonably light)
        print(f"Loading image-captioning model {model_name} on {device} ...")
        _PROCESSOR = BlipProcessor.from_pretrained(model_name)
        # use float32 for CPU stability
        _MODEL = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
        _MODEL.to(device)
        # put model in eval mode
        _MODEL.eval()
        print("Model loaded.")

def _open_image(image_input: Union[str, Image.Image]) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, str):
        return Image.open(image_input).convert("RGB")
    raise ValueError("image_input must be a file path or PIL.Image.Image instance")

def generate_caption(image_input: Union[str, Image.Image], max_length: int = 64) -> str:
    """
    Generate a caption for the given image.

    Args:
        image_input: path to image file OR a PIL.Image.Image object
        max_length: maximum token length for generated caption

    Returns:
        caption (str)
    """
    # ensure model loaded (on CPU)
    _ensure_model_loaded(device="cpu")

    try:
        image = _open_image(image_input)
    except Exception as e:
        return f"ERROR: cannot open image - {e}"

    # prepare inputs
    try:
        inputs = _PROCESSOR(images=image, return_tensors="pt")
        # forward pass (no grad)
        with torch.no_grad():
            generated_ids = _MODEL.generate(**inputs, max_length=max_length, num_beams=3, do_sample=False)
        caption = _PROCESSOR.decode(generated_ids[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        # Catch memory/OOM or other inference errors and return readable message
        return f"ERROR during caption generation: {e}"


# Simple CLI test
if __name__ == "__main__":
    import os
    test_path = "data/test.jpg"
    if not os.path.exists(test_path):
        print("Put a test image at 'data/test.jpg' to run the demo.")
    else:
        print("Generating caption for", test_path)
        print(generate_caption(test_path))
