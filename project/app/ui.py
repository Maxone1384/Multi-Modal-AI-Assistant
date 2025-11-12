import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- Load lightweight model (DistilGPT-2) ---
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# --- Chat function ---
def chat_response(message, history):
    history = history or []
    prompt = ""

    # Combine chat history for context
    for user_msg, bot_msg in history:
        prompt += f"User: {user_msg}\nAI: {bot_msg}\n"
    prompt += f"User: {message}\nAI:"

    # Encode input and generate response
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=250, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the model’s last answer
    if "AI:" in response:
        response = response.split("AI:")[-1].strip()

    history.append((message, response))
    return history, ""

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h2 style='text-align:center; color:#333;'>💬 Smart AI Chat (DistilGPT-2)</h2>")

    chatbot = gr.Chatbot(label="Chat with AI", height=500)
    msg = gr.Textbox(label="Type your message:", placeholder="Ask me anything...", lines=1)
    clear = gr.Button("Clear Chat")

    msg.submit(chat_response, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
