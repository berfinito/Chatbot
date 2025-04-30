from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()


with open(r"C:\Users\hadfi\LLM_Project\data\room_and_equipment_bookings.txt", "r", encoding="utf-8") as f:
    context_text = f.read()


max_context_length = 500  
context_text = context_text[:max_context_length]


print("Type your question (type 'exit' to quit):")
while True:
    question = input("\nYou: ").strip()
    if question.lower() == "exit":
        break

    prompt = f"{context_text}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            top_p=0.9
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Bot:", answer)
