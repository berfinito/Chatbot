from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import torch


supervised_model = AutoModelForSeq2SeqLM.from_pretrained("./fine_tuned_flan_t5")
supervised_tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_flan_t5")
unsupervised_model_name = "google/flan-t5-base"
unsupervised_tokenizer = AutoTokenizer.from_pretrained(unsupervised_model_name)
unsupervised_model = AutoModelForSeq2SeqLM.from_pretrained(unsupervised_model_name)


retriever = SentenceTransformer("all-MiniLM-L6-v2")


with open(r"C:\Users\hadfi\LLM_Project\data\room_and_equipment_bookings.txt", "r", encoding="utf-8") as f:
    full_context = f.read()

passages = [p.strip() for p in full_context.split("\n") if len(p.strip()) > 30]
passage_embeddings = retriever.encode(passages, convert_to_tensor=True)


def is_generic_or_empty(answer: str, question: str = "") -> bool:
    answer = answer.strip().lower()
    question = question.strip().lower()
    if len(answer) < 15 or question in answer:
        return True
    weak_phrases = ["Rephrase your question please", "I don't quite understand", "PLease contact support"]
    return any(p in answer for p in weak_phrases)


def retrieve_relevant_chunks(question, top_k=3):
    question_embedding = retriever.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, passage_embeddings, top_k=top_k)[0]
    retrieved = [passages[hit['corpus_id']] for hit in hits]
    return "\n".join(retrieved)


supervised_model.eval()
unsupervised_model.eval()

print("Ask a question (type 'exit' to quit):")
while True:
    question = input("\nYou: ").strip()
    if question.lower() == "exit":
        break

    
    retrieved_context = retrieve_relevant_chunks(question)

   
    supervised_prompt = f"{retrieved_context}\n\nQuestion: {question}\nAnswer:"
    inputs = supervised_tokenizer(supervised_prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        supervised_output = supervised_model.generate(**inputs, max_new_tokens=80)
    supervised_answer = supervised_tokenizer.decode(supervised_output[0], skip_special_tokens=True)

   
    if is_generic_or_empty(supervised_answer, question):
        print("Supervised Answer (too generic):", supervised_answer)

        
        unsupervised_prompt = f"{retrieved_context}\n\nQuestion: {question}\nAnswer:"
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
