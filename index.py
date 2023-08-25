import vllm
import gradio as gr
from vllm.models import Llama2ChatModel

# Load the model
model = Llama2ChatModel.from_pretrained("llama-2-chat")

# Create a vllm instance with a different name and a lower max length
vllm_instance = vllm.VLLM(model, decoding_algorithm="top_p_sampling", top_p=0.8, max_length=50)

# Initialize an empty chat history
chat_history = []

def chatbot(message):
  # Use a try-except block to catch and handle any errors
  try:
    # Pass the message and the chat history to vllm
    response, chat_history = vllm_instance.generate(message, chat_history)
    # Return the response
    return response
  except Exception as e:
    # Print or log the error message
    print(e)
    # Return a generic error message to the user
    return "Sorry, something went wrong. Please try again later."

# Create a gradio interface
ui = gr.ChatInterface(
  fn=chatbot,
  title="Llama 2 Chatbot",
  description="A simple chat interface powered by llama 2 and vllm.",
  examples=["Hi", "What is your name?", "Tell me a joke."]
)

# Launch the interface
ui.launch()