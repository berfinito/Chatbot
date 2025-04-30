import os
import torch
import gc 
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig
from peft import PeftModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain_huggingface import HuggingFacePipeline 
from langchain_core.documents import Document 
from typing import List, Dict, Any
import re 
import spacy
# === Configuration ===

# --- Model Paths ---
BASE_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
SFT_ADAPTER_PATH = "sft_dapt_8b_qlora_output"

# --- RAG Components ---

VECTORSTORE_PATH = "faiss_full_data_index_bge_v2"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
NUM_CHUNKS_TO_RETRIEVE = 3
RETRIEVAL_SIMILARITY_THRESHOLD = 0.3

CONTEXT_SIMILARITY_THRESHOLD=0.5

# --- Hardware & Inference Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_IN_4BIT = True
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# --- Generation Parameters ---
MAX_NEW_TOKENS = 100 # Max tokens for the *generated answer*
TEMPERATURE = 0 # Keep low for RAG
TOP_P = 0.95
REPETITION_PENALTY = 1.15 # Value > 1.0 discourages repetition

# --- *** ADDED: Context Window Handling *** ---
MODEL_MAX_CONTEXT_WINDOW = 4096 
RESERVED_FOR_ANSWER = MAX_NEW_TOKENS + 50
MAX_INPUT_TOKENS = MODEL_MAX_CONTEXT_WINDOW - RESERVED_FOR_ANSWER

# === Helper Functions ===
def split_into_sentences(text: str) -> list:
    """Use spaCy to split text into real sentences."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def extract_key_phrases(text: str) -> set:
    """Extracts meaningful noun chunks and named entities."""
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

def is_answer_faithful(answer: str, context: str, threshold: float = 0.5, debug: bool = False) -> bool:
    context_lower = context.lower()

    # Phrase-based matching
    phrases = extract_key_phrases(answer)
    phrase_match_count = sum(1 for phrase in phrases if phrase in context_lower)
    phrase_match_ratio = phrase_match_count / len(phrases) if phrases else 0

    # Word-based matching
    important_words = [w.lower() for w in re.findall(r'\b\w+\b', answer) if len(w) > 2]
    word_match_count = sum(1 for word in important_words if word in context_lower)
    word_match_ratio = word_match_count / len(important_words) if important_words else 0

    if debug:
        print(f"[DEBUG] Phrase match ratio: {phrase_match_ratio:.2f}, Word match ratio: {word_match_ratio:.2f}")
        
    return phrase_match_ratio >= threshold or word_match_ratio >= threshold

def filter_faithful_sentences(answer: str, context: str) -> str:
    """Keep only faithful sentences from the answer."""
    sentences = split_into_sentences(answer)
    faithful_sentences = []

    for sentence in sentences:
        if is_answer_faithful(sentence, context, CONTEXT_SIMILARITY_THRESHOLD):
            faithful_sentences.append(sentence)

    if faithful_sentences:
        return " ".join(faithful_sentences)
    else:
        return "I cannot answer this question based on the provided context."

def retrieve_with_threshold(vectorstore, query, k=3, threshold=0.3):
    """Retrieve top k docs, but refuse if too dissimilar."""
    # Get docs with scores
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
    
    if all(score < threshold for _, score in docs_and_scores):
        return None  # All documents are too dissimilar
    else:
        docs = [doc for doc, _ in docs_and_scores]
        return docs
    
def clean_context(documents: List[Document]) -> str:
    """Concatenates page content of retrieved documents."""
    return "\n\n".join(doc.page_content for doc in documents)

def clean_incomplete_sentence(text: str) -> str:
    """
    Trims potential incomplete sentence fragments after the last likely
    sentence-ending period. Avoids truncating if the last period seems
    part of a URL, email, or similar structure.
    """
    text = text.strip() # Remove leading/trailing whitespace first
    last_dot_index = text.rfind(".")

    if last_dot_index == -1:
        # No period found, return original text
        return text
    elif last_dot_index == len(text) - 1:

        potential_url_or_email = False

        match = re.search(r'\b[a-zA-Z0-9-]+(\.[a-zA-Z]{2,})+\.?$', text)
        if match:
            potential_url_or_email = True

        if potential_url_or_email:
            # If it looks like a URL/email ending with a dot, keep it.
            return text
        else:
            # If it ends in a dot but doesn't look like URL/Email, keep the dot.
            return text # Keep the final dot as it seems intentional punctuation.

    elif last_dot_index < len(text) - 1:
        # Period is not the last character, check the next one
        char_after_dot = text[last_dot_index + 1]
        if char_after_dot.isspace():
            # Character after dot is whitespace, likely end of sentence. Truncate.
            return text[:last_dot_index + 1]
        else:
            # Character after dot is not whitespace, likely part of URL/email/etc. Keep original.
            return text
    else:
        # This case should logically not be reached if last_dot_index is valid
        return text

def clear_memory():
    """Clears GPU cache and runs Python garbage collector."""
    print("🧹 Clearing CUDA cache and collecting garbage...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)
    print("✅ Memory cleared.")

# === RAG Setup ===

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

def build_rag_pipeline(llm, retriever):
    """Builds the LangChain RAG pipeline. (Original structure kept)"""
    rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE_STR)

    rag_chain = (
        {"context": retriever | RunnableLambda(clean_context), "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    print("✅ RAG chain created.")
    return rag_chain

def format_prompt_with_context(template: PromptTemplate, context: str, question: str) -> str:
    """Formats the prompt using the template, context, and question."""
    return template.format(context=context, question=question)

# === Main Chatbot Script ===
if __name__ == "__main__":
    print("🚀 Initializing RAG Chatbot...")
    print(f"Using device: {DEVICE}")

    nlp = spacy.load("en_core_web_sm")
    # --- 1. Load Retriever ---
    retriever = None
    embeddings = None
    vectorstore = None
    if not os.path.exists(VECTORSTORE_PATH):
        print(f"❌ Error: Vector store not found at '{VECTORSTORE_PATH}'")
        print("👉 Please ensure `prepare_knowledge_full.py` ran successfully and check the VECTORSTORE_PATH.")
        exit(1)

    print(f"🧠 Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"💾 Loading FAISS index from: {VECTORSTORE_PATH}")
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True # Be cautious with this setting
        )
        print(f"✅ Retriever initialized to fetch {NUM_CHUNKS_TO_RETRIEVE} chunks.")
    except Exception as e:
        print(f"❌ Error loading retriever: {e}")
        exit(1)

    # --- 2. Configure Quantization ---
    bnb_config = None
    load_dtype = torch.float16
    if LOAD_IN_4BIT:
        print(f"⚙️ Configuring 4-bit quantization (compute dtype: {compute_dtype})...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        load_dtype = compute_dtype # Use compute_dtype when loading in 4-bit

    # --- 3. Load SFT Model & Tokenizer ---
    model = None
    tokenizer = None
    llm_pipeline = None
    rag_chain = None

    try:
        print(f"🔄 Loading Base Model '{BASE_MODEL_NAME}' and Tokenizer...")
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("⚠️ Set tokenizer pad_token to eos_token")


        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto", # Automatically distributes model layers across available GPUs/CPU
            torch_dtype=load_dtype,
            trust_remote_code=True,
        )
        print(f"✅ Base model '{BASE_MODEL_NAME}' loaded.")

        if not os.path.exists(SFT_ADAPTER_PATH):
             print(f"❌ Error: SFT Adapter path not found: {SFT_ADAPTER_PATH}")
             exit(1)

        print(f"🔄 Applying SFT adapters from {SFT_ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(model, SFT_ADAPTER_PATH)
        print(f"✅ SFT adapters applied from {SFT_ADAPTER_PATH}.")

        model.generation_config.temperature = None
        model.generation_config.top_p = None

        # --- 4. Create Text Generation Pipeline ---
        print("⚙️ Creating text generation pipeline...")
        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        print("✅ Pipeline created.")

        print("🔗 Preparing RAG components...")
        llm_for_chain = HuggingFacePipeline(pipeline=llm_pipeline) # Wrap the pipeline for potential LangChain use
        rag_prompt_obj = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE_STR)
        print("✅ RAG Components Ready!")


    except Exception as e:
        print(f"❌ Error during model loading or pipeline setup: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # --- 6. Start Chat Loop ---
    print("\n💬 Enter your question (or type 'quit'/'exit' to stop):")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                print("👋 Exiting chatbot.")
                break
            if not user_input.strip():
                continue

            print("🤖 Thinking...")
            start_time = time.time()

            # --- *** RAG Invocation with Token Check *** ---
            # 1. Retrieve relevant documents
            retrieved_docs = retrieve_with_threshold(vectorstore, user_input, k=NUM_CHUNKS_TO_RETRIEVE, threshold=RETRIEVAL_SIMILARITY_THRESHOLD)
            context_str = ""
            if retrieved_docs is None:
                answer = "I cannot answer this question based on the provided context."
            else:
                context_str = clean_context(retrieved_docs)

            # 2. Format the prompt with retrieved context
            full_prompt_text = format_prompt_with_context(rag_prompt_obj, context_str, user_input)

            # 3. Check token count BEFORE sending to LLM
            input_token_ids = tokenizer.encode(full_prompt_text, return_tensors=None, add_special_tokens=True) # Use None for list output
            input_token_count = len(input_token_ids)

            end_time = time.time()

            if input_token_count > MAX_INPUT_TOKENS:
                print(f"⚠️ Warning: Input ({input_token_count} tokens) exceeds maximum allowed ({MAX_INPUT_TOKENS}).")
                # Strategy: Simple refusal. Could implement context truncation here instead.
                answer = "Error: The generated prompt with context is too long for the model's input limit. Please try a shorter question."
                end_time = time.time()
            elif context_str:

                response = llm_pipeline(full_prompt_text)
                full_response_text = response[0]['generated_text'] # Get the full text
                end_time = time.time() # Calculate time after generation

                # --- START: New Answer Extraction Logic ---
                answer = "" # Initialize answer
                answer_marker = "\nAnswer:\n"
                marker_pos = full_response_text.find(answer_marker)

                if marker_pos != -1:
                    # Extract text *after* the marker
                    potential_answer_block = full_response_text[marker_pos + len(answer_marker):].strip()

                    cleaned_block = re.sub(r'</?think>.*?</think>', '', potential_answer_block, flags=re.DOTALL | re.IGNORECASE).strip()


                    first_double_newline = cleaned_block.find("\n\n")
                    if first_double_newline != -1:
                        answer = cleaned_block[:first_double_newline].strip()
                    else:
                        first_newline = cleaned_block.find("\n")
                        if first_newline != -1:
                            answer = cleaned_block[:first_newline].strip()
                        else:
                            answer = cleaned_block # Assume the whole cleaned block is the answer if single line

                else:
                    # Fallback if "Answer:" marker isn't found
                    print("⚠️ WARNING: 'Answer:' marker not found in response. Using basic fallback extraction.")
                    prompt_end_marker_index = full_response_text.rfind(user_input)
                    if prompt_end_marker_index != -1:
                        answer = full_response_text[prompt_end_marker_index + len(user_input):].strip()
                    else:
                        answer = full_response_text # Keep everything as last resort

                # --- END: New Answer Extraction Logic ---

                # --- Post-processing checks ---

                negative_phrase = "I cannot answer this question based on the provided context."
                if negative_phrase.lower() in answer.lower():
                    answer = negative_phrase
                else:

                    answer = clean_incomplete_sentence(answer.strip()) # Use the 'v2' version

                answer = filter_faithful_sentences(answer, context_str)
            # 5. Print result
            print(f"Bot: {answer}") # Print the final processed answer
            print(f"(Response time: {end_time - start_time:.2f} seconds)")


        except KeyboardInterrupt:
            print("\n👋 Exiting chatbot due to interrupt.")
            break
        except Exception as e:
            print(f"❌ An error occurred during processing: {e}")
            import traceback
            traceback.print_exc()

            break # Current behavior: exit on error

    # --- Cleanup ---
    print("🧹 Final cleanup...")
    del model
    del tokenizer
    del llm_pipeline
    del llm_for_chain # Added this
    del rag_prompt_obj # Added this
    del retriever
    del vectorstore
    del embeddings
    clear_memory()
    print("🏁 Chatbot script finished.")