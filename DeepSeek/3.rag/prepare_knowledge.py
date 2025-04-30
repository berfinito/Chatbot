import os
import re
import torch 
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# === Configuration ===
SOURCE_DOCUMENT = "../health_and_safety.txt" 
VECTORSTORE_PATH = "faiss_health_safety_index_bge" 
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# --- Chunking Strategy ---
class HeaderTextSplitter(TextSplitter):
    """Splits text based on '### Title: Subtitle' headers."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._separator = r"\n### .*?: .*?\n"

    def split_text(self, text: str) -> list[str]:
        text_to_split = "\n" + text
        chunks = []
        last_end = 0
        # Find all header occurrences
        for match in re.finditer(self._separator, text_to_split):
            # Add the text between the last header end and the current header start
            chunk_text = text_to_split[last_end:match.start()].strip()
            if chunk_text: # Add non-empty chunks
                chunks.append(chunk_text)
            # Update the end position for the next iteration
            last_end = match.start() # Keep the header with the *next* chunk

        # Add the remaining text after the last header
        last_chunk_text = text_to_split[last_end:].strip()
        if last_chunk_text:
            chunks.append(last_chunk_text)

        final_chunks = []
        sub_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        for chunk in chunks:
            if len(chunk) > 1000: 
                split_sub_chunks = sub_splitter.split_text(chunk)
                final_chunks.extend(split_sub_chunks)
            elif chunk: 
                final_chunks.append(chunk)

        print(f"Split text into {len(final_chunks)} final chunks.")
        if not final_chunks and text: 
             print("No separators found, attempting recursive split on whole text.")
             final_chunks = sub_splitter.split_text(text)
             print(f"Split text into {len(final_chunks)} chunks using recursive splitter.")

        return final_chunks

# Use the custom splitter
text_splitter = HeaderTextSplitter()

# === Main Script ===
if __name__ == "__main__":
    print(f"🚀 Starting RAG Knowledge Base Preparation...")

    # --- 1. Load Document ---
    if not os.path.exists(SOURCE_DOCUMENT):
        print(f"❌ Error: Source document not found at '{SOURCE_DOCUMENT}'")
        exit(1)

    print(f"📄 Loading document: {SOURCE_DOCUMENT}")
    loader = TextLoader(SOURCE_DOCUMENT, encoding='utf-8')
    documents = loader.load()
    print(f"✅ Document loaded. Number of Langchain Document objects: {len(documents)}")
    full_text = "\n\n".join([doc.page_content for doc in documents])


    # --- 2. Chunk Document ---
    print("Chunking document...")
    chunks = text_splitter.split_text(full_text)
    print(f"✅ Document split into {len(chunks)} text chunks.")

    if not chunks:
        print("❌ Error: No chunks were created. Check text splitter logic and document content.")
        exit(1)

    chunk_docs = [Document(page_content=chunk, metadata={"source": SOURCE_DOCUMENT}) for chunk in chunks]
    print(f"✅ Created {len(chunk_docs)} Langchain Document objects.")


    # --- 3. Initialize Embeddings ---
    print(f"🧠 Initializing embedding model: {EMBEDDING_MODEL_NAME}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True} #
    )
    print("✅ Embeddings initialized.")

    # --- 4. Create and Save Vector Store ---
    print(f"💾 Creating FAISS vector store from chunks...")
    try:
        vectorstore = FAISS.from_documents(chunk_docs, embeddings)

        print(f"💾 Saving FAISS index to: {VECTORSTORE_PATH}")

        abs_vectorstore_path = os.path.abspath(VECTORSTORE_PATH)
        os.makedirs(os.path.dirname(abs_vectorstore_path), exist_ok=True)
        vectorstore.save_local(abs_vectorstore_path) 
        print("✅ Vector store created and saved successfully!")
    except Exception as e:
        print(f"❌ Error creating or saving vector store: {e}")
        import traceback
        traceback.print_exc()

    print("🏁 Knowledge base preparation finished.")
