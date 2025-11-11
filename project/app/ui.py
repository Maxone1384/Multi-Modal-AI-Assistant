import gradio as gr
from vision.image_captioning import describe_image
from vision.vqa import answer_question_about_image
from audio.speech_to_text import audio_to_text
from audio.text_to_speech import text_to_audio
from nlp.text_generator import generate_text
from generative.text_to_image import text_to_image

# 🎛 توابع رابط (به ترتیب از ورودی کاربر)
def handle_input(input_type, text=None, image=None, audio=None):
    if input_type == "🖼 توضیح تصویر":
        return describe_image(image)
    elif input_type == "❓ پرسش درباره تصویر":
        return answer_question_about_image(image, text)
    elif input_type == "🎙 تبدیل صدا به متن":
        return audio_to_text(audio)
    elif input_type == "🔊 تبدیل متن به صدا":
        return text_to_audio(text)
    elif input_type == "🎨 تولید تصویر":
        return text_to_image(text)
    elif input_type == "💬 تولید پاسخ متنی":
        return generate_text(text)
    else:
        return "❌ نوع ورودی نامشخص است."

# 🌐 ساخت رابط Gradio
iface = gr.Interface(
    fn=handle_input,
    inputs=[
        gr.Radio(
            ["🖼 توضیح تصویر", "❓ پرسش درباره تصویر", "🎙 تبدیل صدا به متن", 
             "🔊 تبدیل متن به صدا", "🎨 تولید تصویر", "💬 تولید پاسخ متنی"],
            label="انتخاب قابلیت"
        ),
        gr.Textbox(label="ورودی متنی (اختیاری)"),
        gr.Image(label="آپلود تصویر", type="filepath"),
        gr.Audio(label="آپلود صوت (اختیاری)", type="filepath")
    ],
    outputs="text",
    title="🤖 دستیار هوشمند مولتی‌مدل",
    description="متن، تصویر یا صدا بده — و خروجی مناسب دریافت کن!"
)

if __name__ == "__main__":
    iface.launch()
