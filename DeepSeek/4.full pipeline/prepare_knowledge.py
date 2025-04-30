import os
import re
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import time

# === Configuration ===

FULL_DATASET_FILE = "../full_university_data.txt"
VECTORSTORE_PATH = "faiss_full_data_index_bge_v2" 
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DOCUMENT_SEPARATOR = "--- SOURCE SEPARATOR ---"

# --- Chunking Strategy ---
CHUNK_SIZE = 1000 
CHUNK_OVERLAP = 150

# --- Helper Function ---
def parse_and_split_document(doc_text, chunk_size, chunk_overlap, source_file_name):
    """
    Parses metadata from the start of the document text (handling two patterns)
    and splits the remaining content into chunks.
    Returns a list of Langchain Document objects with metadata.
    """
    lines = doc_text.strip().split('\n')
    metadata = {"source_file": source_file_name}
    content_start_index = 0
    metadata_lines_count = 0 

    if not lines:
        return []


    if lines[0].lower().startswith("source_url:"):
        metadata["source_url"] = lines[0].split(":", 1)[1].strip()
        metadata_lines_count = 1

        if len(lines) > 1:
            if lines[1].lower().startswith("source_doc:"):
                metadata["source_doc"] = lines[1].split(":", 1)[1].strip()
                metadata_lines_count = 2
                if len(lines) > 2 and lines[2].strip() == "":
                    metadata_lines_count = 3
                    if len(lines) > 3 and lines[3].strip(): 
                        metadata["general_title"] = lines[3].strip()
                        metadata_lines_count = 4
                        if len(lines) > 4 and lines[4].strip():
                             metadata["document_title"] = lines[4].strip()
                             metadata_lines_count = 5
                        else: metadata["document_title"] = "N/A"
                    else:
                         metadata["general_title"] = "N/A"
                         metadata["document_title"] = "N/A"
                else:
                    print(f"⚠️ Warning: Expected blank line after 'source_doc:' in document starting with '{lines[0][:50]}...'")
                    title_start_index = 2 
                    if len(lines) > title_start_index and lines[title_start_index].strip(): # General Title
                        metadata["general_title"] = lines[title_start_index].strip()
                        metadata_lines_count = title_start_index + 1
                        if len(lines) > title_start_index + 1 and lines[title_start_index+1].strip(): # Content Title
                             metadata["document_title"] = lines[title_start_index+1].strip()
                             metadata_lines_count = title_start_index + 2
                        else: metadata["document_title"] = "N/A"
                    else:
                         metadata["general_title"] = "N/A"
                         metadata["document_title"] = "N/A"

            elif lines[1].strip() == "":
                # Pattern 1: Source URL, Blank Line, Titles...
                metadata["source_doc"] = "N/A" # No source doc
                metadata_lines_count = 2 # Source URL + Blank Line
                # Look for titles starting from line index 2
                if len(lines) > 2 and lines[2].strip(): # General Title
                    metadata["general_title"] = lines[2].strip()
                    metadata_lines_count = 3
                    if len(lines) > 3 and lines[3].strip(): # Content Title
                         metadata["document_title"] = lines[3].strip()
                         metadata_lines_count = 4
                    else: metadata["document_title"] = "N/A"
                else:
                     metadata["general_title"] = "N/A"
                     metadata["document_title"] = "N/A"
            else:
                # Unexpected structure after Source URL
                print(f"⚠️ Warning: Unexpected line after 'source_url:' in document starting with '{lines[0][:50]}...'")
                metadata["source_doc"] = "N/A"
                metadata["general_title"] = "N/A"
                metadata["document_title"] = "N/A"
                metadata_lines_count = 1 # Only Source URL was reliably parsed
        else:
            # Only Source URL line exists
             metadata["source_doc"] = "N/A"
             metadata["general_title"] = "N/A"
             metadata["document_title"] = "N/A"

    else:
        # Document doesn't start with Source URL - treat whole thing as content?
        print(f"⚠️ Warning: Document does not start with 'source_url:'. Treating as content. First line: '{lines[0][:50]}...'")
        metadata["source_url"] = "N/A"
        metadata["source_doc"] = "N/A"
        metadata["general_title"] = "N/A"
        metadata["document_title"] = "N/A"
        metadata_lines_count = 0 # No metadata lines identified

    # Content starts after the identified metadata lines
    content_start_index = metadata_lines_count
    main_content = "\n".join(lines[content_start_index:]).strip()

    if not main_content:
        print(f"⚠️ Warning: Document starting with '{lines[0][:50]}...' has no content after metadata.")
        return []

    # Split the main content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    content_chunks = text_splitter.split_text(main_content)

    # Create Langchain Document objects for each chunk
    doc_objects = []
    for i, chunk in enumerate(content_chunks):
        chunk_metadata = metadata.copy()
        chunk_metadata["chunk_index"] = i
        doc_objects.append(Document(page_content=chunk, metadata=chunk_metadata))

    return doc_objects

# === Main Script ===
if __name__ == "__main__":
    print("🚀 Starting RAG Knowledge Base Preparation for Full Dataset...")
    start_time = time.time()

    # --- 1. Load Full Text ---
    if not os.path.exists(FULL_DATASET_FILE):
        print(f"❌ Error: Full dataset file not found at '{FULL_DATASET_FILE}'")
        print("👉 Please update the FULL_DATASET_FILE variable in this script.")
        exit(1)

    print(f"📄 Loading full dataset: {FULL_DATASET_FILE}")
    try:
        with open(FULL_DATASET_FILE, 'r', encoding='utf-8') as f:
            full_text = f.read()
        print(f"✅ Full dataset loaded. Total characters: {len(full_text)}")
    except Exception as e:
        print(f"❌ Error reading text file: {e}")
        exit(1)

    # --- 2. Split into Documents ---
    print(f"✂️ Splitting text into documents using separator: '{DOCUMENT_SEPARATOR}'")
    # Add newline before separator for more robust splitting if separator is at the very beginning
    individual_doc_texts = re.split(f'\n?{re.escape(DOCUMENT_SEPARATOR)}\n?', full_text)
    # Remove leading/trailing whitespace and filter out empty strings
    individual_doc_texts = [doc.strip() for doc in individual_doc_texts if doc and doc.strip()]
    print(f"✅ Split into {len(individual_doc_texts)} source documents.")

    if not individual_doc_texts:
        print("❌ Error: No documents found after splitting. Check the separator and file content.")
        exit(1)

    # --- 3. Parse, Chunk, and Create Document Objects ---
    print("🧩 Parsing metadata and chunking documents...")
    all_chunk_docs = []
    source_filename = os.path.basename(FULL_DATASET_FILE)
    for i, doc_text in enumerate(individual_doc_texts):
        if not doc_text: continue # Skip empty documents
        if (i + 1) % 100 == 0: # Print progress
             print(f"  Processing document {i+1}/{len(individual_doc_texts)}...")
        parsed_chunks = parse_and_split_document(doc_text, CHUNK_SIZE, CHUNK_OVERLAP, source_filename)
        if parsed_chunks: # Only extend if chunks were actually created
            all_chunk_docs.extend(parsed_chunks)

    if not all_chunk_docs:
        print("❌ Error: No chunks were created after processing all documents. Check parsing logic and content.")
        exit(1)

    print(f"✅ Created {len(all_chunk_docs)} Langchain Document objects with metadata.")
    print(f"📄 Example Metadata from first chunk: {all_chunk_docs[0].metadata}")

    # --- 4. Initialize Embeddings ---
    print(f"🧠 Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True} # Normalize for BGE models

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print("✅ Embeddings initialized.")
    except Exception as e:
         print(f"❌ Error initializing embedding model: {e}")
         print("👉 Check if the model name is correct and SentenceTransformers is installed.")
         exit(1)

    # --- 5. Create and Save Vector Store ---
    print(f"💾 Creating FAISS vector store from {len(all_chunk_docs)} chunks...")
    try:
        vectorstore = FAISS.from_documents(all_chunk_docs, embeddings)

        print(f"💾 Saving FAISS index to: {VECTORSTORE_PATH}")
        abs_vectorstore_path = os.path.abspath(VECTORSTORE_PATH)
        os.makedirs(os.path.dirname(abs_vectorstore_path), exist_ok=True) # Create directory if needed

        vectorstore.save_local(folder_path=abs_vectorstore_path)
        print("✅ Vector store created and saved successfully!")

    except Exception as e:
        print(f"❌ Error creating or saving vector store: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    end_time = time.time()
    print(f"⏱️ Total time taken: {end_time - start_time:.2f} seconds.")
    print("🏁 Knowledge base preparation finished.")