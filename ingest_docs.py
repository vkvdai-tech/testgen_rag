import os
import glob

# Updated modern imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Define Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "polity_docs")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

def run_ingestion():
    print("🔍 Scanning for PDFs in:", DOCS_DIR)
    pdf_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))

    if not pdf_files:
        print("❌ No PDF files found in rag_engine/polity_docs/. Please add at least one PDF and run again.")
        return

    print(f"📄 Found {len(pdf_files)} PDF(s). Processing...")

    all_docs = []
    for pdf in pdf_files:
        print(f"   -> Loading: {os.path.basename(pdf)}")
        try:
            loader = PyPDFLoader(pdf)
            documents = loader.load()
            all_docs.extend(documents)
        except Exception as e:
            print(f"   ⚠️ Could not load {os.path.basename(pdf)}: {e}")

    print(f"📖 Loaded {len(all_docs)} pages total.")

    # Chunking Strategy optimized for Polity/Legal texts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " Article ", " Clause ", " ", ""]
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"🧩 Split into {len(chunks)} text chunks.")

    # Initialize Embeddings (Local & Free)
    print("⚙️ Initializing embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create & Save ChromaDB Vector Store
    print("💾 Creating ChromaDB vector database...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name="polity_collection"
    )

    print("✅ Ingestion complete! Persistent database saved at:", DB_DIR)

if __name__ == "__main__":
    run_ingestion()