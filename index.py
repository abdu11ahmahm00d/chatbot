import vllm
import gradio as gr
from vllm.chat import Llama2ChatModel
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat-hf")

model = Llama2ChatModel(model, tokenizer)

vllm_instance = vllm.VLLM(model, decoding_algorithm="top_p_sampling", top_p=0.8, max_length=50)

chat_history = []


def chatbot(message):
    try:
        response, chat_history = vllm_instance.generate(message, chat_history)
        return response
    except Exception as e:
        print(e)
        return "Sorry, something went wrong. Please try again later."


ui = gr.ChatInterface(
    fn=chatbot,
    title="Llama 2 Chatbot",
    description="A simple chat interface powered by llama 2 and vllm.",
    examples=["Hi", "What is your name?", "Tell me a joke."]
)

ui.launch()
