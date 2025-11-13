from huggingface_hub import InferenceClient
import gradio as gr

client = InferenceClient("mistralai/Mistral-7B-Instruct", token="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")

def chat_response(message, history):
    history = history or []
    prompt = "".join([f"User: {u}\nAI: {b}\n" for u, b in history])
    prompt += f"User: {message}\nAI:"

    reply = client.text_generation(prompt, max_new_tokens=200)
    history.append((message, reply))
    return history, ""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h2 style='text-align:center;'>💬 Chat with Mistral-7B</h2>")
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="Type your message...", label="Message")
    clear = gr.Button("Clear Chat")

    msg.submit(chat_response, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
