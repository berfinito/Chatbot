from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


model = AutoModelForSeq2SeqLM.from_pretrained("./fine_tuned_flan_t5")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_flan_t5")

while True:
    question = input("Ask a question (or 'exit'): ").strip()
    if question.lower() == "exit":
        break

    input_text = f"Question: {question}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Answer:", answer)
