from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


supervised_model = AutoModelForSeq2SeqLM.from_pretrained("./fine_tuned_flan_t5")
supervised_tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_flan_t5")


unsupervised_model_name = "google/flan-t5-base"
unsupervised_tokenizer = AutoTokenizer.from_pretrained(unsupervised_model_name)
unsupervised_model = AutoModelForSeq2SeqLM.from_pretrained(unsupervised_model_name)


with open(r"C:\Users\hadfi\LLM_Project\data\room_and_equipment_bookings.txt", "r", encoding="utf-8") as f:
    context_text = f.read()


max_context_length = 500
context_text = context_text[:max_context_length]


supervised_model.eval()
unsupervised_model.eval()


def is_generic_or_empty(answer):
    generic_keywords = ["a computer store", "a password", "a group", "ensure", "available", "yes", "no"]
    return (
        len(answer.strip()) < 10
        or any(answer.lower().strip().startswith(k) for k in generic_keywords)
    )

print("Ask a question (type 'exit' to quit):")
while True:
    question = input("\nYou: ").strip()
    if question.lower() == "exit":
        break

    
    supervised_prompt = f"{context_text}\n\nQuestion: {question}\nAnswer:"
    inputs = supervised_tokenizer(supervised_prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        supervised_output = supervised_model.generate(**inputs, max_new_tokens=80)
    supervised_answer = supervised_tokenizer.decode(supervised_output[0], skip_special_tokens=True)

    
    if is_generic_or_empty(supervised_answer):
        print("Supervised Answer (too generic):", supervised_answer)

        
        unsupervised_prompt = f"{context_text}\n\nQuestion: {question}\nAnswer:"
        inputs = unsupervised_tokenizer(unsupervised_prompt, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            unsupervised_output = unsupervised_model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.7,
                top_p=0.9
            )
        unsupervised_answer = unsupervised_tokenizer.decode(unsupervised_output[0], skip_special_tokens=True)
        print("Bot (Fallback):", unsupervised_answer)

    else:
        print("Bot (Supervised):", supervised_answer)
