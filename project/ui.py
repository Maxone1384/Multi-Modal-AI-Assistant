import gradio as gr
from vision.image_captioning import generate_caption
from vision.vqa import answer_question
from audio.speech_to_text import speech_to_text
from audio.text_to_speech import text_to_speech
from generative.text_to_image import text_to_image

def multimodal_interface(image, audio, text, question):
    outputs = {}
    
    if image is not None:
        outputs["Caption"] = generate_caption(image.name)
        if question:
            outputs["VQA Answer"] = answer_question(image.name, question)
            
    if audio is not None:
        outputs["Transcribed Text"] = speech_to_text(audio.name)
        
    if text:
        outputs["Generated Image"] = text_to_image(text)
        outputs["Audio Response"] = text_to_speech(text)
        
    return outputs

iface = gr.Interface(
    fn=multimodal_interface,
    inputs=[
        gr.Image(type="file", label="Upload Image"),
        gr.Audio(type="file", label="Upload Audio"),
        gr.Textbox(label="Input Text"),
        gr.Textbox(label="Question about Image")
    ],
    outputs=[
        gr.JSON(label="Outputs")
    ],
    title="Multi-Modal AI Assistant",
    description="آپلود تصویر، صدا، متن و دریافت خروجی هوشمند"
)

iface.launch()
