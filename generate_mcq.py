import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Initialize OpenAI Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def retrieve_context(query: str, k: int = 4) -> str:
    """Retrieves top-k context chunks from ChromaDB."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="polity_collection"
    )
    
    results = vector_db.similarity_search_with_score(query, k=k)
    context_blocks = []
    
    for i, (doc, score) in enumerate(results, 1):
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "N/A")
        context_blocks.append(f"[Source: {source}, Page: {page}]\n{doc.page_content.strip()}")
        
    return "\n\n---\n\n".join(context_blocks)

def generate_upsc_mcq(topic: str):
    print(f"\n🔍 Retrieving context for topic: '{topic}'...")
    context = retrieve_context(topic, k=4)
    
    prompt = f"""
You are an expert UPSC Civil Services examination paper setter specializing in Indian Polity.
Based strictly on the provided reference context below, create 1 high-quality UPSC-style Multiple Choice Question (MCQ).

--- REFERENCE CONTEXT ---
{context}
------------------------

Guidelines for Question Creation:
1. Use standard UPSC formatting (e.g., 2 or 3 statements, followed by "Which of the statements given above is/are correct?").
2. Ensure options follow the pattern: (a) 1 only, (b) 2 only, (c) Both 1 and 2, (d) Neither 1 nor 2 (or similar multi-statement format).
3. Include a detailed, clear explanation citing why each statement is correct or incorrect based on the reference context.

Generate the output in the following format:

**Question:**
[Question text and numbered statements]

**Options:**
(a) ...
(b) ...
(c) ...
(d) ...

**Correct Answer:** [Option]

**Explanation:**
[Detailed explanation covering each statement]
"""

    print("🤖 Generating MCQ via OpenAI API...\n")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Fast and cost-effective; use gpt-4o for complex questions
        messages=[
            {"role": "system", "content": "You are a precise UPSC examination paper setter."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    print("================ UPSC MCQ GENERATED ================\n")
    print(response.choices[0].message.content)
    print("\n====================================================")

if __name__ == "__main__":
    # Test topic
    test_topic = "Basic Structure doctrine and Kesavananda Bharati case"
    generate_upsc_mcq(test_topic)