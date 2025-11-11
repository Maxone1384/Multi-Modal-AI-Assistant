from transformers import BlipForQuestionAnswering, BlipProcessor
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

def answer_question(image_path, question):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer
