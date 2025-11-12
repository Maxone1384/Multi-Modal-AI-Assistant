# vision/vqa.py

"""
Lightweight Visual Question Answering (VQA) module.
Uses the 'dandelin/vilt-b32-finetuned-vqa' model (small & CPU-friendly).
"""

from typing import Union
from PIL import Image
import torch

# Lazy model variables
_TOKENIZER = None
_MODEL = None
_PROCESSOR = None


def _ensure_vqa_model_loaded(model_name="dandelin/vilt-b32-finetuned-vqa", device="cpu"):
    """
    Loads model and processor only once.
    """
    global _MODEL, _PROCESSOR, _TOKENIZER
    if _MODEL is None or _PROCESSOR is None:
        try:
            from transformers import ViltProcessor, ViltForQuestionAnswering
        except Exception as e:
            raise ImportError("transformers not installed. Run: pip install transformers") from e

        print(f"🔹 Loading VQA model '{model_name}' on {device} ...")
        _PROCESSOR = ViltProcessor.from_pretrained(model_name)
        _MODEL = ViltForQuestionAnswering.from_pretrained(model_name)
        _MODEL.to(device)
        _MODEL.eval()
        print("✅ VQA model loaded successfully.")


def _open_image(image_input: Union[str, Image.Image]) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    elif isinstance(image_input, str):
        return Image.open(image_input).convert("RGB")
    raise ValueError("image_input must be a file path or a PIL.Image.Image object.")


def answer_question(image_input: Union[str, Image.Image], question: str) -> str:
    """
    Answers a question about the given image.
    """
    _ensure_vqa_model_loaded(device="cpu")

    try:
        image = _open_image(image_input)
    except Exception as e:
        return f"ERROR: Could not open image - {e}"

    try:
        inputs = _PROCESSOR(image, question, return_tensors="pt")
        with torch.no_grad():
            outputs = _MODEL(**inputs)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            answer = _MODEL.config.id2label[idx]
        return answer
    except Exception as e:
        return f"ERROR during VQA: {e}"


# Simple test example
if __name__ == "__main__":
    import os
    test_path = "data/test.jpg"
    if not os.path.exists(test_path):
        print("Please place a test image at 'data/test.jpg' to run this demo.")
    else:
        q = "What is in the image?"
        print(f"Q: {q}")
        print("A:", answer_question(test_path, q))
