import spaces
import gradio as gr
import datetime
from rag_chatbot import load_bot, answer_question

bot = load_bot()

def log_interaction(question, answer):
    timestamp = datetime.datetime.utcnow().isoformat()
    print(f"[{timestamp}] Q: {question}\nA: {answer}")

@spaces.GPU
def chat(user_input):
    answer = answer_question(bot, user_input)
    log_interaction(user_input, answer)
    return answer

iface = gr.Interface(fn=chat,
                     inputs="text",
                     outputs="text",
                     title="University Chatbot")

iface.launch()
