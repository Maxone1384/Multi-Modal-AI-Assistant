# nlp/text_generator.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# مدل سبک برای پاسخ‌دهی
MODEL_NAME = "google/flan-t5-base"

# بارگذاری توکنایزر و مدل
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def generate_text(prompt):
    """
    Generates a response from the input text (prompt).
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
