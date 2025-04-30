import os
import torch
import gc
import re
import time
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import subprocess

# Paths
BASE_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
SFT_ADAPTER_PATH = "sft_dapt_8b_qlora_output"
VECTORSTORE_PATH = "faiss_full_data_index_bge_v2"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# RAG constants
NUM_CHUNKS_TO_RETRIEVE = 3
RETRIEVAL_SIMILARITY_THRESHOLD = 0.3
CONTEXT_SIMILARITY_THRESHOLD = 0.5
MAX_NEW_TOKENS = 100
MODEL_MAX_CONTEXT_WINDOW = 4096
RESERVED_FOR_ANSWER = MAX_NEW_TOKENS + 50
MAX_INPUT_TOKENS = MODEL_MAX_CONTEXT_WINDOW - RESERVED_FOR_ANSWER

# Globals
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Build Prompt
RAG_PROMPT_TEMPLATE_STR = """You are a factual assistant for the University of Bradford. Your **ONLY** source of information is the text provided below in the 'Context'.
You **MUST NOT** use any prior knowledge or information outside of the provided 'Context'.

Carefully extract information directly from the 'Context' to answer the 'Question'. Prefer quoting or summarizing facts exactly as phrased in the 'Context'. 

If the user's question is informal or casual, you must still answer in a formal university FAQ style based only on the information found in the 'Context'.
You must restate any relevant rules or policies even if the question uses casual language.

Give impersonal, third-person university FAQ style answers.

Avoid interpretation, assumption, or extrapolation. Your response must contain only the direct answer found in the 'Context' if possible.

You must never assume, infer, or guess any information not explicitly stated in the 'Context', even if it might seem obvious or common sense.
If a detail (such as room equipment, facilities, capacities, corridors, or other properties) is not specifically written in the 'Context', you must NOT mention it.

If the question asks for a specific fact (such as location, contact, address, or one attribute), provide a concise answer in one or two sentences.

If the question asks for a broad description (such as listing features, services, equipment, or general information), provide a full structured answer covering all related information from the 'Context', but without adding anything extra.

If the 'Context' does **not** contain the necessary information, your response MUST be EXACTLY the phrase: "I cannot answer this question based on the provided context." and nothing else.

Context:
{context}

Task: Based ONLY on the 'Context' above, answer the following question concisely.

Question:
{question}

Answer:
"""

rag_prompt_obj = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE_STR)

# Functions
def split_into_sentences(text: str) -> list:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def extract_key_phrases(text: str) -> set:
    doc = nlp(text)
    phrases = set()

    # Add noun chunks (longer meaningful phrases)
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip().lower()
        if len(phrase) > 2:
            phrases.add(phrase)

    # Add named entities like organization names, building names
    for ent in doc.ents:
        if ent.label_ in {"ORG", "GPE", "FAC", "LOC"}:
            phrase = ent.text.strip().lower()
            if len(phrase) > 2:
                phrases.add(phrase)

    return phrases

def is_answer_faithful(answer: str, context: str) -> bool:
    context_lower = context.lower()

    # Phrase-based matching
    phrases = extract_key_phrases(answer)
    phrase_match_count = sum(1 for phrase in phrases if phrase in context_lower)
    phrase_match_ratio = phrase_match_count / len(phrases) if phrases else 0

    # Word-based matching
    important_words = [w.lower() for w in re.findall(r'\b\w+\b', answer) if len(w) > 2]
    word_match_count = sum(1 for word in important_words if word in context_lower)
    word_match_ratio = word_match_count / len(important_words) if important_words else 0

        
    return phrase_match_ratio >= CONTEXT_SIMILARITY_THRESHOLD or word_match_ratio >= CONTEXT_SIMILARITY_THRESHOLD

def filter_faithful_sentences(answer: str, context: str) -> str:
    sentences = split_into_sentences(answer)
    faithful_sentences = []

    for sentence in sentences:
        if is_answer_faithful(sentence, context):
            faithful_sentences.append(sentence)

    if faithful_sentences:
        return " ".join(faithful_sentences)
    else:
        return "I cannot answer this question based on the provided context."

def clean_context(documents: list) -> str:
    return "\n\n".join(doc.page_content for doc in documents)

def clean_incomplete_sentence(text: str) -> str:
    text = text.strip()
    last_dot = text.rfind(".")
    if last_dot == -1:
        return text
    if last_dot == len(text) - 1 or (text[last_dot + 1].isspace() if last_dot + 1 < len(text) else False):
        return text[:last_dot + 1]
    return text

# Load bot
def load_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, SFT_ADAPTER_PATH)
    model = model.to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        repetition_penalty=1.15,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    return {"vectorstore": vectorstore, "tokenizer": tokenizer, "pipeline": pipe}

# Answer
def answer_question(bot, question: str) -> str:
    vectorstore = bot["vectorstore"]
    tokenizer = bot["tokenizer"]
    pipe = bot["pipeline"]

    retrieved_docs = vectorstore.similarity_search_with_score(question, k=NUM_CHUNKS_TO_RETRIEVE)
    if all(score < RETRIEVAL_SIMILARITY_THRESHOLD for _, score in retrieved_docs):
        return "I cannot answer this question based on the provided context."

    context_str = clean_context([doc for doc, _ in retrieved_docs])
    full_prompt = rag_prompt_obj.format(context=context_str, question=question)
    if len(tokenizer.encode(full_prompt, return_tensors=None)) > MAX_INPUT_TOKENS:
        return "Error: The generated prompt with context is too long for the model's input limit."

    response = pipe(full_prompt)
    full_text = response[0]["generated_text"]
    answer_marker = "\nAnswer:\n"
    pos = full_text.find(answer_marker)

    if pos != -1:
        answer = full_text[pos + len(answer_marker):].strip()
        answer = re.sub(r'</?think>.*?</think>', '', answer, flags=re.DOTALL | re.IGNORECASE).strip()
        first_split = answer.split("\n\n")[0] if "\n\n" in answer else answer.split("\n")[0]
        cleaned = clean_incomplete_sentence(first_split.strip())
    else:
        cleaned = "I cannot answer this question based on the provided context."

    return filter_faithful_sentences(cleaned, context_str)
