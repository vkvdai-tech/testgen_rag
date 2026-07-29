import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

def test_retrieval(query: str, k: int = 3):
    print(f"\n🔎 Querying database for: '{query}'\n" + "="*50)
    
    # Load embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load ChromaDB vector store
    vector_db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="polity_collection"
    )
    
    # Perform Similarity Search
    results = vector_db.similarity_search_with_score(query, k=k)
    
    for i, (doc, score) in enumerate(results, 1):
        source = os.path.basename(doc.metadata.get("source", "Unknown Source"))
        page = doc.metadata.get("page", "N/A")
        
        print(f"\n--- [Result #{i}] | Source: {source} (Page {page}) | Distance Score: {score:.4f} ---")
        print(doc.page_content.strip())
        print("-" * 50)

if __name__ == "__main__":
    # Test query
    sample_topic = "Preamble basic structure philosophy constitution"
    test_retrieval(sample_topic, k=3)